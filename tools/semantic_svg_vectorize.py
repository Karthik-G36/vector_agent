import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    VISION_LIBS_AVAILABLE = True
except ImportError:
    VISION_LIBS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .gpt_svg_vectorize import svg_to_eps
from utils.conversions import mm_to_pts


SEMANTIC_ANALYSIS_PROMPT = """\
Analyze this image for semantic SVG reconstruction.

Return ONLY valid JSON with this exact shape:
{
  "candidate": true,
  "design_type": "3d_text_effect|flat_logo|unknown",
  "lettering_style": "block|sans|serif|script|custom|unknown",
  "main_text": "exact main text or empty",
  "top_text": "exact small top text or empty",
  "bottom_lines": ["exact bottom line 1", "exact bottom line 2"],
  "background": "light_gradient|dark|transparent|flat",
  "primary_shape": "circle|none",
  "colors": {
    "background": "#RRGGBB",
    "primary": "#RRGGBB",
    "front_text": "#RRGGBB",
    "extrude": "#RRGGBB",
    "shadow": "#RRGGBB",
    "small_text": "#RRGGBB"
  },
  "notes": "short factual notes"
}

Use candidate=true only when the image is a designed graphic that can be rebuilt from
simple block/sans/serif text, simple geometric shapes, shadows, gradients, and layered text.
Use candidate=false for photos, illustrations, unknown objects, cursive/script lettering,
hand-lettered/custom lettering, or text whose exact outline is the main design. Those must
be traced from the source pixels, not regenerated from an approximate font.

Do not invent text. If unreadable, use an empty string.
"""


def _encode_image(image_path: str) -> tuple[str, str]:
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
    }
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii"), mime_map.get(suffix, "image/png")


def _safe_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()


def _safe_color(value: object, fallback: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value.strip()):
        return value.strip()
    return fallback


def _bgr_to_hex(color: tuple[int, int, int]) -> str:
    b, g, r = (int(v) for v in color)
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def _contour_to_svg_path(cnt, epsilon_ratio: float = 0.0018, coord_scale: float = 1.0) -> str:
    epsilon = max(0.22, epsilon_ratio * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2).astype(float) / coord_scale
    if len(pts) < 3:
        return ""

    parts = [f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"]
    for x, y in pts[1:]:
        parts.append(f"L {x:.3f} {y:.3f}")
    parts.append("Z")
    return " ".join(parts)


def _mask_to_svg_path(
    mask,
    min_area: float,
    epsilon_ratio: float = 0.0012,
    coord_scale: float = 1.0,
) -> str:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    path_parts = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        path = _contour_to_svg_path(cnt, epsilon_ratio=epsilon_ratio, coord_scale=coord_scale)
        if path:
            path_parts.append(path)
    return " ".join(path_parts)


def _shift_mask(mask, dx: int, dy: int):
    h, w = mask.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _source_geometry_3d_trace_svg(image_path: str, fidelity_scale: int = 4) -> str:
    if not VISION_LIBS_AVAILABLE:
        raise RuntimeError("opencv-python, numpy, or Pillow not installed")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    scale = max(1, min(int(fidelity_scale), 6))
    work_bgr = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    wh, ww = work_bgr.shape[:2]
    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)

    border_pixels = np.concatenate([
        img_bgr[0, :, :],
        img_bgr[-1, :, :],
        img_bgr[:, 0, :],
        img_bgr[:, -1, :],
    ])
    bg_color = tuple(int(v) for v in np.median(border_pixels, axis=0))
    bg_hex = _bgr_to_hex(bg_color)

    bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.int16)
    lab_dist = np.linalg.norm(lab.astype(np.int16) - bg_lab, axis=2)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    # 3D text effects often use a red background where the cast shadow is only a
    # darker red. Do not quantize it away: find layers by visual role instead.
    neutral = saturation < 58
    face_mask = ((neutral & (value > 184)) | (gray > 210)).astype(np.uint8) * 255
    highlight_mask = (neutral & (value > 238)).astype(np.uint8) * 255
    bevel_light_mask = (neutral & (value >= 188) & (value <= 238)).astype(np.uint8) * 255
    bevel_mid_mask = (neutral & (value >= 142) & (value < 188)).astype(np.uint8) * 255
    bevel_dark_mask = (neutral & (value >= 82) & (value < 142)).astype(np.uint8) * 255

    red_hue = (hue < 8) | (hue > 170)
    bg_v = int(cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2HSV)[0, 0, 2])
    source_shadow_mask = (
        red_hue
        & (saturation > 85)
        & (value < max(40, bg_v - 12))
        & (lab_dist > 6)
    ).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, scale), max(3, scale)))
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(5, scale * 2 + 1), max(5, scale * 2 + 1)),
    )
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel_small)

    bevel_light_mask = cv2.morphologyEx(bevel_light_mask, cv2.MORPH_CLOSE, kernel_small)
    bevel_mid_mask = cv2.morphologyEx(bevel_mid_mask, cv2.MORPH_CLOSE, kernel_small)
    bevel_dark_mask = cv2.morphologyEx(bevel_dark_mask, cv2.MORPH_CLOSE, kernel_small)
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_CLOSE, kernel_small)
    source_shadow_mask = cv2.morphologyEx(source_shadow_mask, cv2.MORPH_CLOSE, kernel_large)

    if np.count_nonzero(face_mask) < wh * ww * 0.01:
        # Last-resort foreground extraction for non-white 3D graphics.
        face_mask = (lab_dist > 22).astype(np.uint8) * 255
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel_large)

    # Synthetic depth layers are derived from source geometry, so custom strokes are
    # preserved while the 3D effect does not depend on fragile color clustering.
    shadow_far = cv2.dilate(_shift_mask(face_mask, 34 * scale, 28 * scale), kernel_large, iterations=2)
    shadow_near = cv2.dilate(_shift_mask(face_mask, 16 * scale, 14 * scale), kernel_small, iterations=1)
    source_shadow_mask = cv2.bitwise_or(source_shadow_mask, shadow_far)
    lower_bevel = cv2.bitwise_and(_shift_mask(face_mask, -10 * scale, 13 * scale), cv2.bitwise_not(face_mask))
    expanded_face = cv2.dilate(face_mask, kernel_large, iterations=2)
    bevel_light = cv2.bitwise_and(bevel_light_mask, expanded_face)
    bevel_mid = cv2.bitwise_and(bevel_mid_mask, expanded_face)
    bevel_dark = cv2.bitwise_and(bevel_dark_mask, expanded_face)
    highlight = cv2.bitwise_and(highlight_mask, face_mask)

    min_area = max(6 * scale * scale, wh * ww * 0.000006)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        f'  <rect width="{w}" height="{h}" fill="{bg_hex}" />',
    ]

    layers = [
        ("#780000", source_shadow_mask, 0.90, 0.00055),
        ("#990000", shadow_near, 0.58, 0.00055),
        ("#aeb0b0", lower_bevel, 1.0, 0.00050),
        ("#989b9b", bevel_dark, 1.0, 0.00045),
        ("#c7c9c9", bevel_mid, 1.0, 0.00042),
        ("#e0e1e1", bevel_light, 1.0, 0.00040),
        ("#ffffff", face_mask, 1.0, 0.00034),
        ("#f8f8f8", highlight, 1.0, 0.00032),
    ]

    for color, mask, opacity, epsilon in layers:
        path_data = _mask_to_svg_path(
            mask,
            min_area=min_area,
            epsilon_ratio=epsilon,
            coord_scale=scale,
        )
        if not path_data:
            continue
        opacity_attr = "" if opacity >= 1 else f' opacity="{opacity:.2f}"'
        svg_lines.append(
            f'  <path fill="{color}"{opacity_attr} fill-rule="evenodd" d="{path_data}" />'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def _fallback_scene(image_path: str) -> dict:
    return {
        "candidate": False,
        "design_type": "unknown",
        "lettering_style": "unknown",
        "main_text": "",
        "top_text": "",
        "bottom_lines": [],
        "background": "flat",
        "primary_shape": "none",
        "colors": {
            "background": "#d9d9d9",
            "primary": "#ef0000",
            "front_text": "#ffffff",
            "extrude": "#c90000",
            "shadow": "#555555",
            "small_text": "#ef0000",
        },
        "notes": "Semantic analysis unavailable; refusing to infer lettering from filename.",
    }


def _analyze_scene(image_path: str) -> dict:
    if not OPENAI_AVAILABLE:
        return _fallback_scene(image_path)

    try:
        b64, mime = _encode_image(image_path)
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("AGENT_MODEL", "gpt-4o"),
            response_format={"type": "json_object"},
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": SEMANTIC_ANALYSIS_PROMPT},
                ],
            }],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return _fallback_scene(image_path)


def _build_3d_text_svg(scene: dict) -> str:
    colors = scene.get("colors") if isinstance(scene.get("colors"), dict) else {}
    bg = _safe_color(colors.get("background"), "#d9d9d9")
    primary = _safe_color(colors.get("primary"), "#ef0000")
    front = _safe_color(colors.get("front_text"), "#ffffff")
    extrude = _safe_color(colors.get("extrude"), "#c90000")
    shadow = _safe_color(colors.get("shadow"), "#555555")
    small = _safe_color(colors.get("small_text"), primary)

    main_text = _safe_text(scene.get("main_text")) or "TEXT"
    top_text = _safe_text(scene.get("top_text"))
    bottom_lines = scene.get("bottom_lines") if isinstance(scene.get("bottom_lines"), list) else []
    bottom_lines = [_safe_text(line) for line in bottom_lines if _safe_text(line)]

    width, height = 1662, 1000
    cx, cy = width / 2, height / 2
    font_stack = "Impact, Anton, 'Arial Black', sans-serif"
    main_font_size = 300 if len(main_text) <= 6 else max(180, int(1500 / len(main_text)))
    text_length = min(1450, max(760, len(main_text) * main_font_size * 0.72))

    top_markup = ""
    if top_text:
        top_markup = f"""
  <g transform="translate({cx:.1f} 54)">
    <rect x="-125" y="-20" width="250" height="34" rx="17" fill="none" stroke="{small}" stroke-width="3"/>
    <line x1="-28" y1="-20" x2="-28" y2="14" stroke="{small}" stroke-width="3"/>
    <text y="4" text-anchor="middle" font-family="'Arial Narrow', Arial, sans-serif"
          font-size="22" font-weight="700" letter-spacing="3" fill="{small}">{top_text}</text>
  </g>"""

    bottom_markup = ""
    if bottom_lines:
        y = 904
        lines = []
        for i, line in enumerate(bottom_lines[:4]):
            lines.append(
                f'    <text x="{cx:.1f}" y="{y + i * 26}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="22" font-weight="700" '
                f'letter-spacing="2" fill="{small}">{line}</text>'
            )
        bottom_markup = "\n  <g>\n" + "\n".join(lines) + "\n  </g>"

    extrusion_layers = []
    for offset in range(44, 0, -4):
        opacity = 0.95 - offset * 0.006
        extrusion_layers.append(
            f'    <text x="{cx + offset:.1f}" y="{560 + offset * 0.78:.1f}" '
            f'text-anchor="middle" font-family="{font_stack}" font-size="{main_font_size}" '
            f'font-weight="900" letter-spacing="-8" textLength="{text_length}" '
            f'lengthAdjust="spacingAndGlyphs" fill="{extrude}" opacity="{opacity:.3f}">{main_text}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <defs>
    <radialGradient id="bg" cx="50%" cy="42%" r="76%">
      <stop offset="0%" stop-color="#f8f8f8"/>
      <stop offset="62%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="#b6b6b6"/>
    </radialGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="24" dy="32" stdDeviation="18" flood-color="{shadow}" flood-opacity="0.42"/>
    </filter>
    <filter id="frontShadow" x="-15%" y="-15%" width="130%" height="140%">
      <feDropShadow dx="8" dy="12" stdDeviation="6" flood-color="#7a0000" flood-opacity="0.42"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)"/>{top_markup}
  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="362" fill="{primary}"/>
  <g filter="url(#softShadow)" transform="rotate(-3 {cx:.1f} 510)">
{chr(10).join(extrusion_layers)}
  </g>
  <text x="{cx:.1f}" y="560" text-anchor="middle" font-family="{font_stack}"
        font-size="{main_font_size}" font-weight="900" letter-spacing="-8"
        textLength="{text_length}" lengthAdjust="spacingAndGlyphs"
        fill="{front}" filter="url(#frontShadow)" transform="rotate(-3 {cx:.1f} 510)">{main_text}</text>{bottom_markup}
</svg>
"""


class SemanticSVGInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the graphic image")
    width_mm: float = Field(..., description="Target output width in millimetres")
    height_mm: float = Field(..., description="Target output height in millimetres")
    output_path: Optional[str] = Field(
        default=None,
        description="Where to write the EPS file. Also saves a semantic .svg alongside it.",
    )


def _semantic_svg_vectorize(
    image_path: str,
    width_mm: float,
    height_mm: float,
    output_path: Optional[str] = None,
) -> str:
    result = {"image_path": image_path}

    if not Path(image_path).exists():
        result["error"] = f"File not found: {image_path}"
        return json.dumps(result)

    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}.eps")

    svg_path = str(Path(output_path).with_suffix(".svg"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    scene = _analyze_scene(image_path)
    result["scene"] = scene

    lettering_style = str(scene.get("lettering_style", "unknown")).lower()
    source_trace_styles = {"script", "custom"}
    if not scene.get("candidate") or scene.get("design_type") != "3d_text_effect":
        result["error"] = "Image is not a semantic 3D text-effect candidate"
        result["fallback_recommended"] = "trace_source_geometry"
        return json.dumps(result)

    if lettering_style in source_trace_styles:
        try:
            svg = _source_geometry_3d_trace_svg(image_path)
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg)
            backend = svg_to_eps(svg, output_path, mm_to_pts(width_mm), mm_to_pts(height_mm))
        except Exception as exc:
            result["error"] = f"Source-geometry 3D trace failed: {exc}"
            result["fallback_recommended"] = "trace_source_geometry"
            return json.dumps(result)

        result["output_path"] = output_path
        result["svg_path"] = svg_path
        result["method"] = "source_geometry_3d_trace"
        result["conversion_backend"] = backend
        result["warning"] = (
            "Custom/script lettering was traced from source geometry to avoid invented strokes. "
            "The SVG uses derived face, bevel, highlight, and cast-shadow layers to preserve a 3D look, "
            "but it does not recreate editable text."
        )
        return json.dumps(result)

    if lettering_style == "unknown":
        result["error"] = (
            "Semantic reconstruction skipped because lettering style is unknown; "
            "regenerating it would risk invented strokes."
        )
        result["fallback_recommended"] = "trace_source_geometry"
        return json.dumps(result)

    svg = _build_3d_text_svg(scene)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    try:
        backend = svg_to_eps(svg, output_path, mm_to_pts(width_mm), mm_to_pts(height_mm))
    except Exception as exc:
        result["error"] = f"Semantic SVG saved, EPS conversion failed: {exc}"
        result["svg_path"] = svg_path
        return json.dumps(result)

    result["output_path"] = output_path
    result["svg_path"] = svg_path
    result["method"] = "semantic_svg_3d_text"
    result["conversion_backend"] = backend
    result["warning"] = (
        "Semantic reconstruction uses live text/font rendering. For exact production output, "
        "convert text to outlines with a stable installed font."
    )
    return json.dumps(result)


semantic_svg_vectorize_tool = StructuredTool.from_function(
    func=_semantic_svg_vectorize,
    name="semantic_svg_vectorize",
    description=(
        "Use before pixel tracing for designed 3D text effects or mockups. "
        "Reconstructs a clean semantic SVG using text, circles, gradients, filters, and layered text "
        "instead of flattening shadows/backgrounds into K-means blobs. Returns JSON with output_path, "
        "svg_path, method, conversion_backend, and scene."
    ),
    args_schema=SemanticSVGInput,
)
