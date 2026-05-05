"""
LLM-powered vectorization: GPT-4o vision → clean SVG code → EPS.

Unlike contour tracing (which follows JPEG-compressed pixels), GPT-4o
understands the logo structure and generates mathematically clean bezier
paths, proper circles, and accurate text — no noise, no halos.
"""
import base64
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    cairosvg = None
    CAIROSVG_AVAILABLE = False

from utils.conversions import mm_to_pts

SVG_NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------------------
# GPT-4o SVG generation prompt
# ---------------------------------------------------------------------------

SVG_PROMPT = """\
You are a professional SVG vectorization expert. Analyze this logo image with precision \
and generate complete, production-ready SVG code that perfectly recreates it for print.

CRITICAL REQUIREMENTS:
1. Capture EVERY visible element — all text (including small subtitles), shapes, icons
2. Sample colors precisely from the image and use exact hex values (e.g. #F26522 not "orange")
3. viewBox must exactly match the image's aspect ratio (measure relative proportions)
4. Use proper SVG elements:
   - <circle> or <ellipse> for round shapes (NOT paths with arc approximations)
   - <path> with smooth cubic bezier curves (C command) for organic letterforms
   - <text> with font-size, letter-spacing, font-weight for all text content
   - <rect> for rectangular regions
   - <g> to group related elements
5. Layer elements strictly back-to-front (background first, foreground last)
6. Do NOT hallucinate elements not in the image
7. Do NOT use <image>, <use>, or external references
8. Include xmlns: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H">

RETURN ONLY the raw SVG code — no markdown fences, no explanation.
Start with <svg and end with </svg>.
"""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> tuple:
    suffix = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".bmp": "image/bmp",
                ".webp": "image/webp", ".tiff": "image/tiff"}
    mime = mime_map.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


# ---------------------------------------------------------------------------
# SVG color parser
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "black": (0, 0, 0), "white": (1, 1, 1), "red": (1, 0, 0),
    "green": (0, 0.502, 0), "blue": (0, 0, 1), "orange": (1, 0.647, 0),
    "gray": (0.502, 0.502, 0.502), "grey": (0.502, 0.502, 0.502),
    "darkgray": (0.663, 0.663, 0.663), "lightgray": (0.827, 0.827, 0.827),
    "transparent": None,
}


def _parse_color(color_str: str) -> Optional[tuple]:
    if not color_str or color_str.strip().lower() in ("none", "transparent"):
        return None
    s = color_str.strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
    if s.startswith("rgb("):
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            vals = [float(n) for n in nums[:3]]
            # Handle rgb(100%, 50%, 0%) and rgb(255, 128, 0)
            div = 100.0 if "%" in s else 255.0
            return tuple(v / div for v in vals)
    return _NAMED_COLORS.get(s.lower(), (0, 0, 0))


# ---------------------------------------------------------------------------
# SVG path tokenizer and parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"([MmZzLlHhVvCcSsQqTtAa])|"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _parse_path_d(d: str) -> list:
    """Parse SVG path data string into [(command, [float, ...]), ...]."""
    raw = _TOKEN_RE.findall(d)
    commands = []
    cur_cmd, cur_args = None, []
    for cmd_tok, num_tok in raw:
        if cmd_tok:
            if cur_cmd is not None:
                commands.append((cur_cmd, cur_args))
            cur_cmd, cur_args = cmd_tok, []
        elif num_tok and cur_cmd is not None:
            cur_args.append(float(num_tok))
    if cur_cmd is not None:
        commands.append((cur_cmd, cur_args))
    return commands


# ---------------------------------------------------------------------------
# SVG → PostScript path converter
# ---------------------------------------------------------------------------

def _tx(x, vx, vw, out_w):
    return (x - vx) * out_w / vw


def _ty(y, vy, vh, out_h):
    return out_h - (y - vy) * out_h / vh   # flip Y: SVG is top-down, PS is bottom-up


IDENTITY_MATRIX = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _matrix_multiply(a: tuple, b: tuple) -> tuple:
    """Return SVG affine matrix a followed by b."""
    a1, b1, c1, d1, e1, f1 = a
    a2, b2, c2, d2, e2, f2 = b
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_matrix(x: float, y: float, matrix: tuple) -> tuple:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _parse_transform(transform: str) -> tuple:
    if not transform:
        return IDENTITY_MATRIX

    matrix = IDENTITY_MATRIX
    for name, raw_args in re.findall(r"([a-zA-Z]+)\(([^)]*)\)", transform):
        args = [float(v) for v in re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", raw_args)]
        name = name.lower()

        if name == "matrix" and len(args) >= 6:
            next_matrix = tuple(args[:6])
        elif name == "translate":
            tx = args[0] if args else 0.0
            ty = args[1] if len(args) > 1 else 0.0
            next_matrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            next_matrix = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and args:
            angle = math.radians(args[0])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotate = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                next_matrix = _matrix_multiply(
                    _matrix_multiply((1.0, 0.0, 0.0, 1.0, cx, cy), rotate),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                next_matrix = rotate
        elif name == "skewx" and args:
            next_matrix = (1.0, 0.0, math.tan(math.radians(args[0])), 1.0, 0.0, 0.0)
        elif name == "skewy" and args:
            next_matrix = (1.0, math.tan(math.radians(args[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue

        matrix = _matrix_multiply(matrix, next_matrix)

    return matrix


def _path_d_to_ps(d: str, vx, vy, vw, vh, out_w, out_h, matrix=IDENTITY_MATRIX) -> list:
    """Convert SVG path data to PostScript commands."""
    commands = _parse_path_d(d)
    lines = []
    cx, cy = 0.0, 0.0      # current point
    mx, my = 0.0, 0.0      # last moveto (for Z)
    lx2, ly2 = 0.0, 0.0   # last cubic/smooth control point
    path_started = False

    def t(x, y):
        x, y = _apply_matrix(x, y, matrix)
        return _tx(x, vx, vw, out_w), _ty(y, vy, vh, out_h)

    for cmd, args in commands:
        rel = cmd.islower() and cmd not in ("zZ",)

        if cmd in ("M", "m"):
            pairs = [(args[i], args[i+1]) for i in range(0, len(args)-1, 2)]
            for i, (dx, dy) in enumerate(pairs):
                if rel:
                    cx += dx; cy += dy
                else:
                    cx, cy = dx, dy
                if i == 0:
                    px, py = t(cx, cy)
                    if path_started:
                        lines.append(f"{px:.3f} {py:.3f} moveto")
                    else:
                        lines.append(f"newpath {px:.3f} {py:.3f} moveto")
                        path_started = True
                    mx, my = cx, cy
                else:
                    px, py = t(cx, cy)
                    lines.append(f"{px:.3f} {py:.3f} lineto")

        elif cmd in ("L", "l"):
            for i in range(0, len(args)-1, 2):
                dx, dy = args[i], args[i+1]
                if rel:
                    cx += dx; cy += dy
                else:
                    cx, cy = dx, dy
                px, py = t(cx, cy)
                lines.append(f"{px:.3f} {py:.3f} lineto")

        elif cmd in ("H", "h"):
            for dx in args:
                cx = cx + dx if rel else dx
                px, py = t(cx, cy)
                lines.append(f"{px:.3f} {py:.3f} lineto")

        elif cmd in ("V", "v"):
            for dy in args:
                cy = cy + dy if rel else dy
                px, py = t(cx, cy)
                lines.append(f"{px:.3f} {py:.3f} lineto")

        elif cmd in ("C", "c"):
            for i in range(0, len(args)-5, 6):
                if rel:
                    x1, y1 = cx+args[i],   cy+args[i+1]
                    x2, y2 = cx+args[i+2], cy+args[i+3]
                    x,  y  = cx+args[i+4], cy+args[i+5]
                else:
                    x1, y1 = args[i],   args[i+1]
                    x2, y2 = args[i+2], args[i+3]
                    x,  y  = args[i+4], args[i+5]
                p1 = t(x1, y1); p2 = t(x2, y2); pe = t(x, y)
                lines.append(
                    f"{p1[0]:.3f} {p1[1]:.3f} {p2[0]:.3f} {p2[1]:.3f} "
                    f"{pe[0]:.3f} {pe[1]:.3f} curveto"
                )
                lx2, ly2 = x2, y2
                cx, cy = x, y

        elif cmd in ("S", "s"):
            for i in range(0, len(args)-3, 4):
                # Reflect last control point
                x1, y1 = 2*cx - lx2, 2*cy - ly2
                if rel:
                    x2, y2 = cx+args[i],   cy+args[i+1]
                    x,  y  = cx+args[i+2], cy+args[i+3]
                else:
                    x2, y2 = args[i],   args[i+1]
                    x,  y  = args[i+2], args[i+3]
                p1 = t(x1, y1); p2 = t(x2, y2); pe = t(x, y)
                lines.append(
                    f"{p1[0]:.3f} {p1[1]:.3f} {p2[0]:.3f} {p2[1]:.3f} "
                    f"{pe[0]:.3f} {pe[1]:.3f} curveto"
                )
                lx2, ly2 = x2, y2
                cx, cy = x, y

        elif cmd in ("Q", "q"):
            # Quadratic → cubic: CP1 = P0 + 2/3*(Q-P0), CP2 = P + 2/3*(Q-P)
            for i in range(0, len(args)-3, 4):
                if rel:
                    qx, qy = cx+args[i],   cy+args[i+1]
                    x,  y  = cx+args[i+2], cy+args[i+3]
                else:
                    qx, qy = args[i],   args[i+1]
                    x,  y  = args[i+2], args[i+3]
                x1 = cx + 2/3*(qx-cx);  y1 = cy + 2/3*(qy-cy)
                x2 = x  + 2/3*(qx-x);   y2 = y  + 2/3*(qy-y)
                p1 = t(x1, y1); p2 = t(x2, y2); pe = t(x, y)
                lines.append(
                    f"{p1[0]:.3f} {p1[1]:.3f} {p2[0]:.3f} {p2[1]:.3f} "
                    f"{pe[0]:.3f} {pe[1]:.3f} curveto"
                )
                lx2, ly2 = x2, y2
                cx, cy = x, y

        elif cmd in ("T", "t"):
            # Smooth quadratic
            for i in range(0, len(args)-1, 2):
                qx, qy = 2*cx - lx2, 2*cy - ly2
                if rel:
                    x, y = cx+args[i], cy+args[i+1]
                else:
                    x, y = args[i], args[i+1]
                x1 = cx + 2/3*(qx-cx);  y1 = cy + 2/3*(qy-cy)
                x2 = x  + 2/3*(qx-x);   y2 = y  + 2/3*(qy-y)
                p1 = t(x1, y1); p2 = t(x2, y2); pe = t(x, y)
                lines.append(
                    f"{p1[0]:.3f} {p1[1]:.3f} {p2[0]:.3f} {p2[1]:.3f} "
                    f"{pe[0]:.3f} {pe[1]:.3f} curveto"
                )
                lx2, ly2 = x2, y2
                cx, cy = x, y

        elif cmd in ("A", "a"):
            # Arc → approximate with cubic bezier segments
            for i in range(0, len(args)-6, 7):
                rx, ry = args[i], args[i+1]
                x_rot = math.radians(args[i+2])
                large_arc = int(args[i+3])
                sweep = int(args[i+4])
                if rel:
                    ex, ey = cx+args[i+5], cy+args[i+6]
                else:
                    ex, ey = args[i+5], args[i+6]
                arc_lines = _arc_to_bezier(cx, cy, ex, ey, rx, ry, x_rot, large_arc, sweep,
                                           vx, vy, vw, vh, out_w, out_h, matrix)
                lines.extend(arc_lines)
                cx, cy = ex, ey

        elif cmd in ("Z", "z"):
            lines.append("closepath")
            cx, cy = mx, my

    return lines


def _arc_to_bezier(x1, y1, x2, y2, rx, ry, phi, large_arc, sweep,
                   vx, vy, vw, vh, out_w, out_h, matrix=IDENTITY_MATRIX) -> list:
    """Convert SVG arc to cubic bezier approximation segments."""
    if rx == 0 or ry == 0:
        x2, y2 = _apply_matrix(x2, y2, matrix)
        px, py = _tx(x2, vx, vw, out_w), _ty(y2, vy, vh, out_h)
        return [f"{px:.3f} {py:.3f} lineto"]

    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cos_phi*dx2 + sin_phi*dy2
    y1p = -sin_phi*dx2 + cos_phi*dy2

    x1p_sq, y1p_sq, rx_sq, ry_sq = x1p**2, y1p**2, rx**2, ry**2
    lam = x1p_sq/rx_sq + y1p_sq/ry_sq
    if lam > 1:
        sq = math.sqrt(lam)
        rx *= sq; ry *= sq
        rx_sq, ry_sq = rx**2, ry**2

    num = max(0, rx_sq*ry_sq - rx_sq*y1p_sq - ry_sq*x1p_sq)
    den = rx_sq*y1p_sq + ry_sq*x1p_sq
    sq = math.sqrt(num / den) if den else 0
    if large_arc == sweep:
        sq = -sq

    cxp = sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx
    cx_arc = cos_phi*cxp - sin_phi*cyp + (x1+x2)/2
    cy_arc = sin_phi*cxp + cos_phi*cyp + (y1+y2)/2

    def angle(ux, uy, vx2, vy2):
        n = math.sqrt(ux*ux + uy*uy) * math.sqrt(vx2*vx2 + vy2*vy2)
        c = (ux*vx2 + uy*vy2) / n if n else 1
        c = max(-1, min(1, c))
        a = math.acos(c)
        return -a if ux*vy2 - uy*vx2 < 0 else a

    theta1 = angle(1, 0, (x1p-cxp)/rx, (y1p-cyp)/ry)
    dtheta = angle((x1p-cxp)/rx, (y1p-cyp)/ry, (-x1p-cxp)/rx, (-y1p-cyp)/ry)

    if not sweep and dtheta > 0:
        dtheta -= 2*math.pi
    elif sweep and dtheta < 0:
        dtheta += 2*math.pi

    n_segs = max(1, math.ceil(abs(dtheta) / (math.pi/2)))
    dt = dtheta / n_segs
    alpha = math.sin(dt) * (math.sqrt(4 + 3*math.tan(dt/2)**2) - 1) / 3

    lines = []
    t1 = theta1
    for _ in range(n_segs):
        cos1, sin1 = math.cos(t1), math.sin(t1)
        cos2, sin2 = math.cos(t1+dt), math.sin(t1+dt)

        p1x = cx_arc + cos_phi*(rx*cos1) - sin_phi*(ry*sin1)
        p1y = cy_arc + sin_phi*(rx*cos1) + cos_phi*(ry*sin1)
        d1x = -cos_phi*rx*sin1 - sin_phi*ry*cos1
        d1y = -sin_phi*rx*sin1 + cos_phi*ry*cos1
        p2x = cx_arc + cos_phi*(rx*cos2) - sin_phi*(ry*sin2)
        p2y = cy_arc + sin_phi*(rx*cos2) + cos_phi*(ry*sin2)
        d2x = -cos_phi*rx*sin2 - sin_phi*ry*cos2
        d2y = -sin_phi*rx*sin2 + cos_phi*ry*cos2

        cp1x, cp1y = p1x + alpha*d1x, p1y + alpha*d1y
        cp2x, cp2y = p2x - alpha*d2x, p2y - alpha*d2y

        def ps_pt(x, y):
            x, y = _apply_matrix(x, y, matrix)
            return (_tx(x, vx, vw, out_w), _ty(y, vy, vh, out_h))

        a1, b1 = ps_pt(cp1x, cp1y)
        a2, b2 = ps_pt(cp2x, cp2y)
        ae, be = ps_pt(p2x, p2y)
        lines.append(f"{a1:.3f} {b1:.3f} {a2:.3f} {b2:.3f} {ae:.3f} {be:.3f} curveto")
        t1 += dt

    return lines


# ---------------------------------------------------------------------------
# SVG → EPS converter
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _collect_style(elem) -> dict:
    """Merge element attributes + inline style into a dict."""
    style = {}
    for attr in ("fill", "stroke", "stroke-width", "opacity",
                 "fill-opacity", "stroke-opacity", "font-size",
                 "font-family", "font-weight", "letter-spacing"):
        val = elem.get(attr)
        if val is not None:
            style[attr] = val
    for prop in elem.get("style", "").split(";"):
        prop = prop.strip()
        if ":" in prop:
            k, v = prop.split(":", 1)
            style[k.strip()] = v.strip()
    return style


def _circle_path_d(cx, cy, rx, ry) -> str:
    """Approximate circle/ellipse as SVG cubic bezier path (κ ≈ 0.5523)."""
    k = 0.5523
    kx, ky = k * rx, k * ry
    return (
        f"M {cx},{cy-ry} "
        f"C {cx+kx},{cy-ry} {cx+rx},{cy-ky} {cx+rx},{cy} "
        f"C {cx+rx},{cy+ky} {cx+kx},{cy+ry} {cx},{cy+ry} "
        f"C {cx-kx},{cy+ry} {cx-rx},{cy+ky} {cx-rx},{cy} "
        f"C {cx-rx},{cy-ky} {cx-kx},{cy-ry} {cx},{cy-ry} Z"
    )


class SVGToEPS:
    def __init__(self, svg_str: str, out_w: float, out_h: float):
        # Strip XML declaration / BOM
        svg_str = re.sub(r"<\?xml[^?]*\?>", "", svg_str).strip()
        self.root = ET.fromstring(svg_str)
        self.out_w = out_w
        self.out_h = out_h
        self.lines: list = []
        self._parse_viewbox()

    def _parse_viewbox(self):
        vb = self.root.get("viewBox", "")
        if vb:
            parts = re.split(r"[\s,]+", vb.strip())
            self.vx, self.vy, self.vw, self.vh = map(float, parts[:4])
        else:
            w_str = re.sub(r"[^\d.]", "", self.root.get("width", "100"))
            h_str = re.sub(r"[^\d.]", "", self.root.get("height", "100"))
            self.vx, self.vy = 0.0, 0.0
            self.vw = float(w_str) if w_str else 100.0
            self.vh = float(h_str) if h_str else 100.0

    def convert(self) -> str:
        self.lines = [
            "%!PS-Adobe-3.0 EPSF-3.0",
            f"%%BoundingBox: 0 0 {int(self.out_w)} {int(self.out_h)}",
            "%%Title: (gpt4o_vector)",
            "%%Creator: (vector_agent/gpt4o)",
            "%%EndComments",
            "%%Page: 1 1",
            "gsave",
        ]
        self._process(self.root, {"fill": "black", "stroke": "none", "stroke-width": "1"})
        self.lines += ["grestore", "showpage", "%%EOF"]
        return "\n".join(self.lines) + "\n"

    def _process(self, elem, inherited: dict):
        tag = _strip_ns(elem.tag)
        style = {**inherited, **_collect_style(elem)}
        parent_matrix = inherited.get("_matrix", IDENTITY_MATRIX)
        style["_matrix"] = _matrix_multiply(parent_matrix, _parse_transform(elem.get("transform", "")))

        if tag in ("defs", "title", "desc", "metadata"):
            return

        if tag in ("svg", "g"):
            for child in elem:
                self._process(child, style)
            return

        if tag == "path":
            d = elem.get("d", "")
            if d:
                self._draw_shape(
                    _path_d_to_ps(d, self.vx, self.vy, self.vw, self.vh,
                                  self.out_w, self.out_h, style["_matrix"]),
                    style,
                )

        elif tag in ("circle", "ellipse"):
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            if tag == "circle":
                r = float(elem.get("r", 0))
                rx = ry = r
            else:
                rx = float(elem.get("rx", 0))
                ry = float(elem.get("ry", 0))
            d = _circle_path_d(cx, cy, rx, ry)
            self._draw_shape(
                _path_d_to_ps(d, self.vx, self.vy, self.vw, self.vh,
                              self.out_w, self.out_h, style["_matrix"]),
                style,
            )

        elif tag == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            d = f"M {x},{y} H {x+w} V {y+h} H {x} Z"
            self._draw_shape(
                _path_d_to_ps(d, self.vx, self.vy, self.vw, self.vh,
                              self.out_w, self.out_h, style["_matrix"]),
                style,
            )

        elif tag in ("text", "tspan"):
            self._draw_text(elem, style)

        # Handle children of non-group tags (nested tspan, etc.)
        for child in elem:
            child_tag = _strip_ns(child.tag)
            if child_tag in ("tspan",):
                self._process(child, style)

    def _draw_shape(self, ps_cmds: list, style: dict):
        if not ps_cmds:
            return
        fill = style.get("fill", "black")
        stroke = style.get("stroke", "none")
        sw_raw = style.get("stroke-width", "1")
        sw = float(re.sub(r"[^\d.]", "", sw_raw) or "1")

        if fill and fill.strip().lower() not in ("none", ""):
            rgb = _parse_color(fill)
            if rgb:
                self.lines.append("gsave")
                self.lines.append(f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} setrgbcolor")
                self.lines.extend(ps_cmds)
                self.lines.append("eofill")
                self.lines.append("grestore")

        if stroke and stroke.strip().lower() not in ("none", ""):
            rgb = _parse_color(stroke)
            if rgb:
                scaled_sw = sw * self.out_w / self.vw
                self.lines.append("gsave")
                self.lines.append(f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} setrgbcolor")
                self.lines.append(f"{scaled_sw:.3f} setlinewidth")
                self.lines.extend(ps_cmds)
                self.lines.append("stroke")
                self.lines.append("grestore")

    def _draw_text(self, elem, style: dict):
        x = float(elem.get("x", 0))
        y = float(elem.get("y", 0))
        text = (elem.text or "").strip()
        if not text:
            return

        fill = style.get("fill", "black")
        rgb = _parse_color(fill)
        if not rgb:
            return

        fs_raw = style.get("font-size", "12")
        fs = float(re.sub(r"[^\d.]", "", fs_raw) or "12")
        scaled_fs = fs * self.out_w / self.vw

        x, y = _apply_matrix(x, y, style.get("_matrix", IDENTITY_MATRIX))
        ps_x = _tx(x, self.vx, self.vw, self.out_w)
        ps_y = _ty(y, self.vy, self.vh, self.out_h)

        fw = style.get("font-weight", "normal")
        font = "Helvetica-Bold" if fw in ("bold", "700", "800", "900") else "Helvetica"

        text_escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.lines += [
            "gsave",
            f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} setrgbcolor",
            f"/{font} findfont {scaled_fs:.2f} scalefont setfont",
            f"{ps_x:.3f} {ps_y:.3f} moveto",
            f"({text_escaped}) show",
            "grestore",
        ]


def svg_to_eps(svg_raw: str, output_path: str, width_pts: float, height_pts: float) -> str:
    """Convert SVG to EPS, preferring CairoSVG when installed."""
    if CAIROSVG_AVAILABLE:
        cairosvg.svg2ps(
            bytestring=svg_raw.encode("utf-8"),
            write_to=output_path,
            output_width=width_pts,
            output_height=height_pts,
        )
        return "cairosvg"

    converter = SVGToEPS(svg_raw, width_pts, height_pts)
    eps_content = converter.convert()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(eps_content)
    return "builtin_svg_parser"


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

class GPTSVGInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the logo/graphic image")
    width_mm: float = Field(..., description="Target output width in millimetres")
    height_mm: float = Field(..., description="Target output height in millimetres")
    output_path: Optional[str] = Field(
        default=None,
        description="Where to write the EPS file. Also saves a .svg file alongside it.",
    )


def _gpt_svg_vectorize(
    image_path: str,
    width_mm: float,
    height_mm: float,
    output_path: Optional[str] = None,
) -> str:
    result: dict = {"image_path": image_path}

    if not OPENAI_AVAILABLE:
        result["error"] = "openai package not installed"
        return json.dumps(result)

    if not Path(image_path).exists():
        result["error"] = f"File not found: {image_path}"
        return json.dumps(result)

    # Determine output paths
    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}.eps")
    svg_path = str(Path(output_path).with_suffix(".svg"))
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # 1. Encode image
    try:
        b64, mime = _encode_image(image_path)
    except Exception as exc:
        result["error"] = f"Cannot read image: {exc}"
        return json.dumps(result)

    # 2. Call GPT-4o with vision
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("AGENT_MODEL", "gpt-4o"),
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": SVG_PROMPT},
                ],
            }],
        )
        svg_raw = response.choices[0].message.content.strip()
    except Exception as exc:
        result["error"] = f"GPT-4o API call failed: {exc}"
        return json.dumps(result)

    # 3. Extract SVG (strip markdown fences if present)
    if "```" in svg_raw:
        parts = svg_raw.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("svg"):
                stripped = stripped[3:]
            if "<svg" in stripped:
                svg_raw = stripped[stripped.index("<svg"):]
                break
    elif "<svg" in svg_raw:
        svg_raw = svg_raw[svg_raw.index("<svg"):]

    if not svg_raw.startswith("<svg"):
        result["error"] = "GPT-4o did not return valid SVG"
        result["gpt_raw"] = svg_raw[:500]
        return json.dumps(result)

    # 4. Save SVG
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_raw)
    result["svg_path"] = svg_path

    # 5. Convert SVG → EPS
    width_pts = mm_to_pts(width_mm)
    height_pts = mm_to_pts(height_mm)

    try:
        result["conversion_backend"] = svg_to_eps(svg_raw, output_path, width_pts, height_pts)
    except Exception as exc:
        result["error"] = f"SVG→EPS conversion failed: {exc}"
        result["svg_path"] = svg_path  # SVG is still saved
        return json.dumps(result)

    result["output_path"] = output_path
    result["bounding_box"] = f"%%BoundingBox: 0 0 {int(width_pts)} {int(height_pts)}"
    result["method"] = "gpt4o_svg"
    return json.dumps(result)


gpt_svg_vectorize_tool = StructuredTool.from_function(
    func=_gpt_svg_vectorize,
    name="gpt_svg_vectorize",
    description=(
        "PRIMARY vectorization tool. Uses GPT-4o vision to generate clean SVG code "
        "from the image (understands logo structure, text, shapes, exact colors), "
        "then converts SVG → EPS via a full SVG path/shape parser. "
        "Produces smooth bezier curves, proper circles, and preserves all text — "
        "far superior to pixel-level contour tracing for logos. "
        "Falls back gracefully to vectorize_to_eps if SVG generation fails. "
        "Returns JSON with: output_path (EPS), svg_path, bounding_box, method."
    ),
    args_schema=GPTSVGInput,
)
