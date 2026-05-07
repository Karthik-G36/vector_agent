"""
Assembles a complete, production-quality SVG from colour regions and effects.

Structure of the generated SVG
───────────────────────────────
<svg xmlns="..." xmlns:xlink="..." viewBox="0 0 W H">
  <defs>
    <!-- linearGradient / radialGradient for every non-solid region -->
    <!-- feGaussianBlur / feMerge filter defs for shadows and glows  -->
  </defs>
  <!-- background rect  (covers the full canvas with bg_hex) -->
  <!-- foreground paths sorted dark->light (shadows under highlights) -->
</svg>

Rendering order
───────────────
Background-coloured regions are skipped entirely — the <rect> already fills
the canvas with bg_hex.  Remaining (foreground) regions are sorted by
luminance ascending so dark regions (3D shadows, outlines) are drawn first
and light regions (letter faces, highlights) land on top.  This mirrors the
correct z-order for 3D text logos without needing explicit layer metadata.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .color_engine import ColorRegion
from .effect_detector import EffectDef
from .path_tracer import trace_region


def _hex_distance(h1: str, h2: str) -> float:
    """Euclidean RGB distance between two '#rrggbb' hex strings."""
    def parse(h: str):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = parse(h1)
    r2, g2, b2 = parse(h2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def _luminance(hex_color: str) -> float:
    """Perceptual luminance (0–255) of a '#rrggbb' colour."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


# ──────────────────────────────────────────────────────────────────────────────
# Gradient coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def _linear_coords(angle_deg: float):
    """
    Convert a gradient angle (degrees CCW from positive-x) to SVG x1/y1/x2/y2
    percentages so the gradient sweeps across a unit square.
    """
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    x1 = round(50.0 - dx * 50.0, 2)
    y1 = round(50.0 - dy * 50.0, 2)
    x2 = round(50.0 + dx * 50.0, 2)
    y2 = round(50.0 + dy * 50.0, 2)
    return x1, y1, x2, y2


def _stops_markup(stops: List[str]) -> str:
    n = len(stops)
    if n == 0:
        return ""
    parts: List[str] = []
    for i, color in enumerate(stops):
        pct = round(100 * i / max(n - 1, 1))
        parts.append(f'<stop offset="{pct}%" stop-color="{color}"/>')
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Public builder
# ──────────────────────────────────────────────────────────────────────────────

def build_svg(
    regions: List[ColorRegion],
    img_w: int,
    img_h: int,
    effects: Optional[List[EffectDef]] = None,
    bg_hex: str = "#ffffff",
) -> str:
    """
    Build a complete SVG string from quantised colour regions.

    Gradient regions get <linearGradient> / <radialGradient> fills.
    Effect regions (shadows, glows) get SVG <filter> defs applied.

    Parameters
    ----------
    regions : colour regions sorted largest→smallest (background first)
    img_w / img_h : pixel dimensions of the *working* image
    effects : optional list of EffectDef objects from effect_detector
    bg_hex  : fallback background colour
    """
    effects = effects or []

    defs: List[str] = []
    body: List[str] = []

    # map hex → filter_id for quick lookup
    hex_to_filter: Dict[str, str] = {e.apply_to_hex: e.filter_id for e in effects}

    # Colour-distance threshold for identifying background regions.
    # Any region whose representative colour is within 32 RGB units of bg_hex
    # is treated as a background variant (JPEG stripe, slight tonal shift, etc.)
    # and excluded from path rendering — the <rect> already covers it.
    # Using colour distance alone (no area requirement) catches individual
    # stripe regions that are each < 30 % of the canvas but still background.
    BG_COLOR_DIST = 32.0

    def _is_bg(region: ColorRegion) -> bool:
        return _hex_distance(region.hex_color, bg_hex) < BG_COLOR_DIST

    # ── gradient defs (foreground regions only) ───────────────────────────────
    for i, region in enumerate(regions):
        if _is_bg(region):
            continue
        if region.gradient_type == "linear" and len(region.gradient_stops) >= 2:
            x1, y1, x2, y2 = _linear_coords(region.gradient_angle)
            gid = f"lg-{i}"
            defs.append(
                f'<linearGradient id="{gid}" '
                f'x1="{x1}%" y1="{y1}%" x2="{x2}%" y2="{y2}%">'
                f'{_stops_markup(region.gradient_stops)}'
                f'</linearGradient>'
            )

        elif region.gradient_type == "radial" and len(region.gradient_stops) >= 2:
            cx_pct = round(region.gradient_cx * 100, 1)
            cy_pct = round(region.gradient_cy * 100, 1)
            r_pct  = round(region.gradient_r  * 100, 1)
            gid = f"rg-{i}"
            defs.append(
                f'<radialGradient id="{gid}" '
                f'cx="{cx_pct}%" cy="{cy_pct}%" r="{r_pct}%" '
                f'fx="{cx_pct}%" fy="{cy_pct}%">'
                f'{_stops_markup(region.gradient_stops)}'
                f'</radialGradient>'
            )

    # ── filter defs ───────────────────────────────────────────────────────────
    for eff in effects:
        defs.append(eff.svg_filter_xml)

    # ── region paths ──────────────────────────────────────────────────────────
    # Build a list of (original_index, region) pairs for foreground regions,
    # then sort by luminance ascending so dark elements (3D shadows, outlines)
    # are drawn before bright elements (letter faces, highlights).
    # SVG renders in document order (back-to-front), so this ensures shadows
    # are always underneath the white text, not drawn on top of it.
    fg_regions = [
        (i, r) for i, r in enumerate(regions) if not _is_bg(r)
    ]
    fg_regions.sort(key=lambda ir: _luminance(ir[1].hex_color))

    for i, region in fg_regions:
        paths = trace_region(region.mask)
        if not paths:
            continue

        # resolve fill reference
        if region.gradient_type == "linear" and len(region.gradient_stops) >= 2:
            fill = f"url(#lg-{i})"
        elif region.gradient_type == "radial" and len(region.gradient_stops) >= 2:
            fill = f"url(#rg-{i})"
        else:
            fill = region.hex_color

        # resolve filter reference
        filter_attr = ""
        if region.hex_color in hex_to_filter:
            filter_attr = f' filter="url(#{hex_to_filter[region.hex_color]})"'

        combined_d = " ".join(paths)
        body.append(
            f'  <path fill="{fill}" fill-rule="evenodd"{filter_attr} d="{combined_d}"/>'
        )

    # ── assemble ──────────────────────────────────────────────────────────────
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {img_w} {img_h}">',
    ]

    if defs:
        lines.append("  <defs>")
        for d in defs:
            lines.append(f"    {d}")
        lines.append("  </defs>")

    lines.append(f'  <rect width="{img_w}" height="{img_h}" fill="{bg_hex}"/>')
    lines.extend(body)
    lines.append("</svg>")

    return "\n".join(lines)
