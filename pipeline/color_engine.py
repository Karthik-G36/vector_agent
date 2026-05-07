"""
Perceptual color quantization using CIELAB color space.

Median-cut in LAB space gives perceptually even palette reduction —
colours that look similar to humans stay together, unlike RGB K-Means which
groups mathematically close but perceptually different hues.

Gradient detection analyses per-region pixel variation: if LAB values vary
directionally across the region (linear or radial), the region is flagged and
the pipeline generates an SVG <linearGradient> or <radialGradient> instead of
a flat fill, preserving 3D depth, metallic sheen, and bevel highlights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ColorRegion:
    label: int
    mask: np.ndarray                  # H×W uint8 (0 or 255)
    bgr_color: Tuple[int, int, int]   # representative BGR
    hex_color: str                    # "#rrggbb"
    pixel_count: int

    gradient_type: str                # "solid" | "linear" | "radial"
    gradient_stops: List[str]         # ["#hex1", "#hex2", ...]  (low→high offset)
    gradient_angle: float             # degrees, meaningful only for "linear"
    gradient_cx: float                # 0..1 fraction of image width  (radial centre)
    gradient_cy: float                # 0..1 fraction of image height (radial centre)
    gradient_r: float                 # 0..1 fraction of max(w,h)    (radial radius)


# ──────────────────────────────────────────────────────────────────────────────
# Colour-space helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def _bgr_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


# ──────────────────────────────────────────────────────────────────────────────
# Median-cut quantisation in LAB space
# ──────────────────────────────────────────────────────────────────────────────

class _Bucket:
    __slots__ = ("idx", "lab", "bgr")

    def __init__(self, idx: np.ndarray, lab: np.ndarray, bgr: np.ndarray):
        self.idx = idx
        self.lab = lab
        self.bgr = bgr

    def split(self) -> Tuple["_Bucket", "_Bucket"]:
        axis = int(np.argmax(self.lab.max(axis=0) - self.lab.min(axis=0)))
        order = np.argsort(self.lab[:, axis])
        mid = len(order) // 2
        lo, hi = order[:mid], order[mid:]
        return (
            _Bucket(self.idx[lo], self.lab[lo], self.bgr[lo]),
            _Bucket(self.idx[hi], self.lab[hi], self.bgr[hi]),
        )

    def representative_bgr(self) -> np.ndarray:
        # Median is robust to outlier fringe pixels (e.g. anti-aliasing mixing
        # white text edge pixels into the red background bucket skews the mean
        # toward pink; the median stays near the dominant colour).
        return np.median(self.bgr, axis=0).round().astype(np.uint8)


def _median_cut(img_bgr: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        palette   – [N×3 uint8] BGR palette
        label_map – [H×W int32] index into palette for each pixel
    """
    h, w = img_bgr.shape[:2]
    img_lab = _bgr_to_lab(img_bgr)

    flat_bgr = img_bgr.reshape(-1, 3).astype(np.float32)
    flat_lab = img_lab.reshape(-1, 3)
    idx = np.arange(h * w, dtype=np.int32)

    buckets: List[_Bucket] = [_Bucket(idx, flat_lab, flat_bgr)]
    while len(buckets) < n_colors:
        buckets.sort(key=lambda b: len(b.idx), reverse=True)
        biggest = buckets.pop(0)
        if len(biggest.idx) < 2:
            buckets.append(biggest)
            break
        b1, b2 = biggest.split()
        buckets.extend([b1, b2])

    palette = np.zeros((len(buckets), 3), dtype=np.uint8)
    labels = np.zeros(h * w, dtype=np.int32)
    for i, b in enumerate(buckets):
        palette[i] = b.representative_bgr()
        labels[b.idx] = i

    return palette, labels.reshape(h, w)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient classification
# ──────────────────────────────────────────────────────────────────────────────

def _lstsq_r2(A: np.ndarray, Y: np.ndarray) -> float:
    """Multi-output R² for linear model A @ coef ≈ Y."""
    coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    ss_res = float(((Y - A @ coef) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-9)


def _extract_stops(px_bgr: np.ndarray, proj: np.ndarray, n: int = 3) -> List[str]:
    lo, hi = proj.min(), proj.max()
    edges = np.linspace(lo, hi, n + 1)
    stops: List[str] = []
    for i in range(n):
        m = (proj >= edges[i]) & (proj < edges[i + 1])
        if not m.any():
            m = np.ones(len(proj), dtype=bool)
        bgr = px_bgr[m].mean(axis=0).round().astype(np.uint8)
        stops.append(_bgr_hex(tuple(bgr)))
    return stops


def _classify_gradient(
    mask: np.ndarray,
    img_bgr: np.ndarray,
    img_lab: np.ndarray,
) -> dict:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    n = len(ys)

    fallback_bgr = img_bgr[ys, xs].mean(axis=0).round().astype(np.uint8) if n else np.array([128, 128, 128], dtype=np.uint8)
    solid = {"type": "solid", "stops": [_bgr_hex(tuple(fallback_bgr))],
             "angle": 0.0, "cx": 0.5, "cy": 0.5, "r": 0.5}

    if n < 50:
        return solid

    # Erode the mask before sampling so we analyse interior pixels only.
    # Anti-aliasing fringe at region boundaries mixes adjacent colours
    # (e.g. red + white → pink), which corrupts gradient stop colours and
    # inflates variance, causing false gradient detection.
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(mask, erode_k)
    iys, ixs = np.where(inner > 0)
    if len(iys) >= 40:
        ys, xs, n = iys, ixs, len(iys)

    px_lab = img_lab[ys, xs].astype(np.float64)
    px_bgr = img_bgr[ys, xs].astype(np.float64)

    total_var = float(np.var(px_lab, axis=0).sum())
    if total_var < 22.0:
        return solid

    xn = xs / float(w)
    yn = ys / float(h)

    # ── linear test ──────────────────────────────────────────────────────────
    A_lin = np.column_stack([xn, yn, np.ones(n)])
    r2_lin = _lstsq_r2(A_lin, px_lab)

    # ── radial test ───────────────────────────────────────────────────────────
    cxc, cyc = xn.mean(), yn.mean()
    dist = np.sqrt((xn - cxc) ** 2 + (yn - cyc) ** 2)
    A_rad = np.column_stack([dist, np.ones(n)])
    r2_rad = _lstsq_r2(A_rad, px_lab)

    if r2_rad > 0.42 and r2_rad >= r2_lin - 0.08:
        stops = _extract_stops(px_bgr, dist, n=3)
        return {
            "type": "radial",
            "stops": stops,
            "angle": 0.0,
            "cx": float(cxc),
            "cy": float(cyc),
            "r": float(min(dist.max() + 0.08, 0.85)),
        }

    if r2_lin > 0.28:
        cov_x = float(np.cov(xn, px_lab[:, 0])[0, 1])
        cov_y = float(np.cov(yn, px_lab[:, 0])[0, 1])
        angle = float(math.degrees(math.atan2(cov_y, cov_x)) % 360)
        rad = math.radians(angle)
        proj = xn * math.cos(rad) + yn * math.sin(rad)
        stops = _extract_stops(px_bgr, proj, n=3)
        return {"type": "linear", "stops": stops, "angle": angle,
                "cx": 0.5, "cy": 0.5, "r": 0.5}

    return solid


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def segment_image(img_bgr: np.ndarray, n_colors: int) -> List[ColorRegion]:
    """
    Segment *img_bgr* into colour regions via CIELAB median-cut.

    Each region carries gradient metadata so the SVG builder can emit
    <linearGradient> / <radialGradient> elements instead of flat fills.

    Returns regions sorted by pixel_count descending (background first).
    """
    h, w = img_bgr.shape[:2]
    min_px = max(20, int(h * w * 0.0002))

    img_lab = _bgr_to_lab(img_bgr)
    palette, label_map = _median_cut(img_bgr, n_colors)

    regions: List[ColorRegion] = []
    for i, bgr in enumerate(palette):
        mask = ((label_map == i) * 255).astype(np.uint8)
        px = int((mask > 0).sum())
        if px < min_px:
            continue

        gd = _classify_gradient(mask, img_bgr, img_lab)
        b, g, r_ch = int(bgr[0]), int(bgr[1]), int(bgr[2])

        regions.append(ColorRegion(
            label=i,
            mask=mask,
            bgr_color=(b, g, r_ch),
            hex_color=_bgr_hex((b, g, r_ch)),
            pixel_count=px,
            gradient_type=gd["type"],
            gradient_stops=gd["stops"],
            gradient_angle=gd["angle"],
            gradient_cx=gd["cx"],
            gradient_cy=gd["cy"],
            gradient_r=gd["r"],
        ))

    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return regions
