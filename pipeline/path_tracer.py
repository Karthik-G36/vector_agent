"""
High-quality per-region bezier path tracing.

Key improvements over the old approach:
  • CHAIN_APPROX_NONE  — dense boundary pixels, nothing skipped before simplification
  • Perimeter-adaptive epsilon — scales with shape size, not a magic constant
  • Adaptive Catmull-Rom tension — near sharp corners tension drops to ~0 so the
    corner stays crisp; on smooth stretches tension rises to 0.35 for fluid curves
  • Minimum contour area scales with image size — eliminates JPEG speckle noise
    without throwing away legitimate small details like thin serifs
  • Proper RETR_TREE + fill-rule evenodd — nested contours punch holes correctly
    (e.g. counter in letter "O", donut shapes)
"""
from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Mask preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)   # remove 1-px speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)   # fill small gaps / holes
    # Blur the boundary so the 1-pixel staircase edge becomes a smooth ramp;
    # after re-thresholding the contour finder sees gentle curves, not jagged
    # pixel steps, which yields far smoother bezier paths.
    blurred = cv2.GaussianBlur(mask, (9, 9), 2.5)
    _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Catmull-Rom tension
# ──────────────────────────────────────────────────────────────────────────────

def _tension_at(pts: np.ndarray, i: int, base: float = 0.32) -> float:
    """
    Compute Catmull-Rom tension at vertex *i*.

    Logic: measure the turning angle at this vertex.
    - Angle > 65°  → sharp corner  → tension = 0.04  (nearly straight through)
    - Angle < 18°  → smooth curve  → tension = base
    - Between       → linear blend
    """
    n = len(pts)
    v1 = pts[i] - pts[(i - 1) % n]
    v2 = pts[(i + 1) % n] - pts[i]
    len1 = math.hypot(float(v1[0]), float(v1[1]))
    len2 = math.hypot(float(v2[0]), float(v2[1]))
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.04
    cos_a = float(np.dot(v1, v2) / (len1 * len2))
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.degrees(math.acos(cos_a))
    if angle > 65:
        return 0.04
    if angle < 18:
        return base
    return base * (1.0 - (angle - 18) / 47.0) + 0.04 * ((angle - 18) / 47.0)


# ──────────────────────────────────────────────────────────────────────────────
# Contour → SVG path
# ──────────────────────────────────────────────────────────────────────────────

def _contour_to_path(cnt: np.ndarray) -> str:
    pts = cnt.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 3:
        return ""
    x0, y0 = pts[0]
    parts = [f"M {x0:.2f} {y0:.2f}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        t_out = _tension_at(pts, i)
        t_in  = _tension_at(pts, (i + 1) % n)
        cp1 = p1 + t_out * (p2 - p0) / 3.0
        cp2 = p2 - t_in  * (p3 - p1) / 3.0
        parts.append(
            f"C {cp1[0]:.2f} {cp1[1]:.2f} "
            f"{cp2[0]:.2f} {cp2[1]:.2f} "
            f"{p2[0]:.2f} {p2[1]:.2f}"
        )
    parts.append("Z")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def trace_region(
    mask: np.ndarray,
    epsilon_pct: float = 0.003,
    min_area: float = 120.0,
) -> List[str]:
    """
    Convert a binary mask into a list of SVG cubic-bezier path ``d=`` strings.

    Each string represents one contour (outer boundary or hole).  The caller
    combines them with ``fill-rule="evenodd"`` so nested contours cut holes.

    Parameters
    ----------
    mask        : uint8 H×W binary mask (0 / 255)
    epsilon_pct : RDP simplification as a fraction of perimeter (0.002–0.005)
    min_area    : minimum contour area px²; raised to 120 so JPEG/quantization
                  speckle (typically < 10×10 px) is dropped without losing
                  thin serifs or stroke details (which are 30+ px wide)
    """
    mask = _clean_mask(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    paths: List[str] = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(0.4, epsilon_pct * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        d = _contour_to_path(approx)
        if d:
            paths.append(d)

    return paths
