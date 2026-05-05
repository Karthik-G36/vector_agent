import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

from .gpt_svg_vectorize import svg_to_eps
from .semantic_svg_vectorize import _bgr_to_hex, _mask_to_svg_path
from utils.conversions import mm_to_pts


def _quantized_regions(img_bgr, n_colors: int):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(img_rgb)
    quant = pil.quantize(colors=n_colors, method=PILImage.Quantize.FASTOCTREE, dither=0)
    labels = np.array(quant)
    palette = quant.getpalette()

    regions = []
    for label in np.unique(labels):
        mask = (labels == label).astype(np.uint8) * 255
        px = int(np.count_nonzero(mask))
        r, g, b = palette[int(label) * 3:int(label) * 3 + 3]
        regions.append((px, mask, (b, g, r), int(label)))
    regions.sort(key=lambda item: item[0], reverse=True)
    return regions


def _high_fidelity_svg(
    image_path: str,
    *,
    fidelity_scale: int = 4,
    n_colors: int = 80,
    min_area_ratio: float = 0.0000025,
    epsilon_ratio: float = 0.00028,
) -> str:
    if not LIBS_AVAILABLE:
        raise RuntimeError("opencv-python, numpy, or Pillow not installed")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    scale = max(1, min(int(fidelity_scale), 6))
    work_bgr = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    wh, ww = work_bgr.shape[:2]

    colors = max(8, min(int(n_colors), 128))
    regions = _quantized_regions(work_bgr, colors)
    if not regions:
        raise RuntimeError("No traceable regions found")

    bg_color = regions[0][2]
    min_area = max(3 * scale * scale, wh * ww * float(min_area_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(2, scale), max(2, scale)))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        f'  <rect width="{w}" height="{h}" fill="{_bgr_to_hex(bg_color)}" />',
    ]

    # Paint largest-to-smallest so broad tonal fields land first and fine details remain visible.
    for _, mask, color, _ in regions[1:]:
        # Closing preserves subpixel-looking soft edges better than opening, which erases bevels.
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        path_data = _mask_to_svg_path(
            cleaned,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
            coord_scale=scale,
        )
        if not path_data:
            continue
        lines.append(
            f'  <path fill="{_bgr_to_hex(color)}" fill-rule="evenodd" d="{path_data}" />'
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


class HighFidelitySVGInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the graphic image")
    width_mm: float = Field(..., description="Target output width in millimetres")
    height_mm: float = Field(..., description="Target output height in millimetres")
    output_path: Optional[str] = Field(
        default=None,
        description="Where to write the EPS file. Also saves a high-fidelity .svg alongside it.",
    )
    fidelity_scale: int = Field(
        default=4,
        description="Internal upscale factor before tracing. Higher means larger/slower SVG.",
    )
    n_colors: int = Field(
        default=80,
        description="Number of tonal color layers to preserve. Higher means larger/slower SVG.",
    )


def _high_fidelity_svg_vectorize(
    image_path: str,
    width_mm: float,
    height_mm: float,
    output_path: Optional[str] = None,
    fidelity_scale: int = 4,
    n_colors: int = 80,
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

    try:
        svg = _high_fidelity_svg(
            image_path,
            fidelity_scale=fidelity_scale,
            n_colors=n_colors,
        )
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        backend = svg_to_eps(svg, output_path, mm_to_pts(width_mm), mm_to_pts(height_mm))
    except Exception as exc:
        result["error"] = f"High-fidelity SVG conversion failed: {exc}"
        return json.dumps(result)

    result["output_path"] = output_path
    result["svg_path"] = svg_path
    result["method"] = "high_fidelity_tonal_trace"
    result["conversion_backend"] = backend
    result["fidelity_scale"] = fidelity_scale
    result["n_colors"] = n_colors
    result["warning"] = (
        "High-fidelity trace preserves many tonal layers and tiny details, so SVG/EPS files "
        "will be much larger. It is visually faithful but not semantic/editable text."
    )
    return json.dumps(result)


high_fidelity_svg_vectorize_tool = StructuredTool.from_function(
    func=_high_fidelity_svg_vectorize,
    name="high_fidelity_svg_vectorize",
    description=(
        "Use when the user wants maximum visual fidelity for any 2D or 3D raster graphic, "
        "even if the SVG becomes large. Upscales internally, preserves many tonal color layers, "
        "keeps tiny details, and uses very low path simplification. Returns JSON with output_path, "
        "svg_path, method, conversion_backend, fidelity_scale, n_colors, and warning."
    ),
    args_schema=HighFidelitySVGInput,
)
