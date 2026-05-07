"""
Visual effect detection: shadows, glows, and soft-edge regions.

The detector analyses the *original* (pre-quantisation) pixels near each
region boundary to judge whether the edge is hard or soft:

  hard edge  → colour changes abruptly at the boundary  → solid region
  soft edge  → std-dev of grayscale across boundary band is high → shadow or glow

Detected effects are returned as EffectDef objects whose ``svg_filter_xml``
goes straight into the SVG <defs> block.

Shadow heuristic : dark region (LAB L < 55) with soft boundary
Glow   heuristic : bright region (LAB L > 200) with soft boundary
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class EffectDef:
    filter_id: str
    svg_filter_xml: str   # complete <filter …>…</filter> element
    apply_to_hex: str     # hex colour of the region this filter belongs to


# ──────────────────────────────────────────────────────────────────────────────
# Edge softness — measured against original image, not quantised mask
# ──────────────────────────────────────────────────────────────────────────────

def _boundary_softness(mask: np.ndarray, img_gray: np.ndarray, band: int = 7) -> float:
    """
    Sample std-dev of grayscale pixel values in the *original* image along the
    region boundary (a ±band-pixel wide dilation ring around the mask edge).

    Returns a score in [0, 1]:
      ≈ 0  hard edge (clean logo contour)
      ≈ 1  very soft edge (blurred shadow / glow halo)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    dilated = cv2.dilate(mask, kernel)
    eroded  = cv2.erode(mask, kernel)
    ring = (dilated - eroded) > 0

    # sample std-dev of grayscale inside the ring from the original image
    samples = img_gray[ring].astype(float)
    if len(samples) < 10:
        return 0.0

    std = float(np.std(samples))
    # empirical: std > 35 → very soft; std < 6 → hard; scale to [0,1]
    return float(min(std / 40.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_effects(
    regions: List[Tuple[np.ndarray, Tuple[int, int, int], str]],
    img_bgr: np.ndarray,
    img_lab: np.ndarray,
) -> List[EffectDef]:
    """
    Scan each ``(mask, bgr_color, hex_color)`` region for shadow/glow signatures.

    Parameters
    ----------
    regions  : list of (mask, bgr_color, hex_color) — one entry per ColorRegion
    img_bgr  : original BGR image (used for edge-softness analysis)
    img_lab  : LAB image matching img_bgr

    Returns a list of EffectDef, possibly empty.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    effects: List[EffectDef] = []
    idx = 0

    for mask, bgr_color, hex_color in regions:
        ys, xs = np.where(mask > 0)
        if len(ys) < 40:
            continue

        mean_L = float(img_lab[ys, xs, 0].mean())   # OpenCV LAB L: 0-255
        softness = _boundary_softness(mask, img_gray)

        # ── drop-shadow candidate ─────────────────────────────────────────────
        # OpenCV LAB L channel: 0=black, 255=white. Shadows are typically
        # mid-dark (L < 135 ≈ L* < 53). Softness > 0.14 means the boundary
        # transitions gradually in the original image.
        if mean_L < 135 and softness > 0.14:
            blur_std = max(2, round(softness * 14))
            fid = f"drop-shadow-{idx}"
            effects.append(EffectDef(
                filter_id=fid,
                svg_filter_xml=(
                    f'<filter id="{fid}" '
                    f'x="-25%" y="-25%" width="150%" height="150%">'
                    f'<feGaussianBlur in="SourceAlpha" '
                    f'stdDeviation="{blur_std}" result="blur"/>'
                    f'<feOffset dx="0" dy="0" result="offset"/>'
                    f'<feComposite in="SourceGraphic" in2="offset" operator="over"/>'
                    f'</filter>'
                ),
                apply_to_hex=hex_color,
            ))
            idx += 1

        # ── glow / highlight candidate ────────────────────────────────────────
        # L > 210 ≈ L* > 82, combined with soft edges = glow/bloom region
        elif mean_L > 210 and softness > 0.18:
            blur_std = max(2, round(softness * 12))
            fid = f"glow-{idx}"
            effects.append(EffectDef(
                filter_id=fid,
                svg_filter_xml=(
                    f'<filter id="{fid}" '
                    f'x="-30%" y="-30%" width="160%" height="160%">'
                    f'<feGaussianBlur stdDeviation="{blur_std}" result="blur"/>'
                    f'<feMerge>'
                    f'<feMergeNode in="blur"/>'
                    f'<feMergeNode in="SourceGraphic"/>'
                    f'</feMerge>'
                    f'</filter>'
                ),
                apply_to_hex=hex_color,
            ))
            idx += 1

    return effects
