"""Compose per-region traced SVGs into a single layered SVG.

Stage 7 — Vector Layer Assembly:
  - Assigns each path its dominant color (from Stage 6 k-means).
  - Stacks layers largest-first (background first, foreground last).
  - Merges same-color regions into one named <g> layer.
  - Applies Ramer-Douglas-Peucker (RDP) simplification to reduce
    redundant anchor points (epsilon=0.5).
  - Warns if total path count exceeds 1000 (5000 = definitely too many).
  - Drops noise paths (d= too short to be a meaningful shape).
"""
import re
from pathlib import Path


# Regions whose dominant color is within this L-inf distance are merged into
# one <g> layer.
_COLOR_MERGE_TOLERANCE = 20

# Inline path polygons with fewer than this many characters in their d="" are
# almost certainly noise specks — skip them when composing.
_MIN_PATH_D_LEN = 20

# RDP epsilon: higher = more aggressive simplification, fewer anchor points.
# 0.5px is a good default — preserves shape, cuts redundant nodes.
_RDP_EPSILON = 0.5

# Warn thresholds for path count.
_PATH_COUNT_WARN  = 1000
_PATH_COUNT_ERROR = 5000


# ── RDP simplification ────────────────────────────────────────────────────────

def _rdp_simplify(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker line simplification (pure Python, no rdp package needed)."""
    if len(points) < 3:
        return points

    # Find the point with the maximum distance from the line start→end
    start, end = points[0], points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_len = (dx * dx + dy * dy) ** 0.5

    max_dist = 0.0
    max_idx  = 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if line_len == 0:
            dist = ((px - start[0]) ** 2 + (py - start[1]) ** 2) ** 0.5
        else:
            dist = abs(dy * px - dx * py + end[0] * start[1] - end[1] * start[0]) / line_len
        if dist > max_dist:
            max_dist = dist
            max_idx  = i

    if max_dist > epsilon:
        left  = _rdp_simplify(points[:max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:],     epsilon)
        return left[:-1] + right
    return [start, end]


def _parse_path_points(d: str) -> list[tuple[float, float]]:
    """Extract numeric coordinate pairs from a simple M...L...Z path string."""
    tokens = re.findall(r"[-+]?\d*\.?\d+", d)
    coords = [float(t) for t in tokens]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]


def _simplify_path_element(path_el: str, epsilon: float) -> str:
    """
    Apply RDP to a <path d="..."/> element.
    Only simplifies M/L/Z polygon paths (Potrace Bezier paths are left as-is
    since they contain curve commands c/s which RDP cannot handle).
    """
    d_match = re.search(r'\bd="([^"]*)"', path_el)
    if not d_match:
        return path_el

    d = d_match.group(1).strip()

    # Skip Bezier paths (contain curve commands)
    if re.search(r'[cCsStTqQaA]', d):
        return path_el

    points = _parse_path_points(d)
    if len(points) < 3:
        return path_el

    simplified = _rdp_simplify(points, epsilon)
    if len(simplified) < 3:
        return path_el

    new_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in simplified) + " Z"
    return path_el[:d_match.start(1)] + new_d + path_el[d_match.end(1):]


# ── SVG parsing helpers ───────────────────────────────────────────────────────

def _viewbox_and_dims(svg: str) -> tuple[str, str, str]:
    vb = re.search(r'viewBox="([^"]*)"', svg)
    w  = re.search(r'<svg\b[^>]+\bwidth="([^"]*)"', svg)
    h  = re.search(r'<svg\b[^>]+\bheight="([^"]*)"', svg)
    return (
        vb.group(1) if vb else "0 0 100 100",
        w.group(1)  if w  else "100",
        h.group(1)  if h  else "100",
    )


def _extract_body(svg: str) -> str:
    match = re.search(r"<svg\b[^>]*>(.*)</svg>", svg, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_paths(body: str) -> list[str]:
    paths  = re.findall(r'<path\b[^>]*/>', body, re.DOTALL)
    paths += re.findall(r'<path\b[^>]*>.*?</path>', body, re.DOTALL)
    return paths


def _is_noise(path_element: str) -> bool:
    m = re.search(r'\bd="([^"]*)"', path_element)
    return not m or len(m.group(1).strip()) < _MIN_PATH_D_LEN


def _color_distance(c1: tuple, c2: tuple) -> int:
    return max(abs(a - b) for a, b in zip(c1, c2))


def _group_by_color(segments: list[dict]) -> list[dict]:
    """Merge segments with similar dominant colors into one layer."""
    groups: list[dict] = []
    for seg in segments:
        body  = _extract_body(seg["traced_svg"])
        color = seg["color"]
        matched = False
        for g in groups:
            if _color_distance(color, g["color"]) <= _COLOR_MERGE_TOLERANCE:
                g["bodies"].append(body)
                matched = True
                break
        if not matched:
            groups.append({"color": color, "bodies": [body]})
    return groups


# ── Main entry point ─────────────────────────────────────────────────────────

def compose_svg(segments: list[dict], output_path: str) -> None:
    """
    Build a single layered SVG from per-region traces.

    segments: list of {"traced_svg": str, "color": (r, g, b)}
              ordered largest-first so the biggest region renders at the bottom.

    Per Stage 7:
      - Each layer is a named <g id="layer_N"> with its dominant fill color.
      - RDP simplification is applied to M/L/Z polygon paths.
      - Noise paths (tiny speck contours) are dropped.
      - Path count is reported with a warning if above thresholds.
    """
    if not segments:
        raise ValueError("compose_svg: no segments provided")

    viewbox, width, height = _viewbox_and_dims(segments[0]["traced_svg"])
    groups = _group_by_color(segments)
    vb_nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", viewbox)]
    if len(vb_nums) == 4:
        bg_x, bg_y, bg_w, bg_h = vb_nums
    else:
        bg_x, bg_y, bg_w, bg_h = 0, 0, width, height

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' width="{width}" height="{height}" viewBox="{viewbox}">',
    ]

    total_paths = 0
    for i, g in enumerate(groups):
        r, g_val, b = g["color"]
        fill = f"#{r:02x}{g_val:02x}{b:02x}"
        lines.append(f'  <g id="layer_{i}" fill="{fill}" stroke="none">')
        if i == 0:
            lines.append(f'    <rect x="{bg_x}" y="{bg_y}" width="{bg_w}" height="{bg_h}"/>')
            total_paths += 1
            lines.append("  </g>")
            continue
        for body in g["bodies"]:
            paths = _extract_paths(body)
            for p in paths:
                if _is_noise(p):
                    continue
                simplified = _simplify_path_element(p, _RDP_EPSILON)
                lines.append(f"    {simplified}")
                total_paths += 1
        lines.append("  </g>")

    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    # Stage 7 path count guard
    if total_paths > _PATH_COUNT_ERROR:
        print(f"      [warn] {total_paths} paths — far too many (>5000). Increase Potrace --opttolerance or RDP epsilon.")
    elif total_paths > _PATH_COUNT_WARN:
        print(f"      [warn] {total_paths} paths — above 1000. Consider raising RDP epsilon if shapes look over-detailed.")
    else:
        print(f"      Composed {len(groups)} color layers, {total_paths} paths -> {output_path}")
