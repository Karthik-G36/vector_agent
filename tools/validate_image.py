import json
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from PIL import Image
except ImportError:
    Image = None

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
MIN_DIM = 100
RECOMMENDED_DPI = 300


class ValidateImageInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the image file to validate")


def _validate_image(image_path: str) -> str:
    result: dict = {"valid": False, "image_path": image_path, "issues": [], "warnings": []}

    path = Path(image_path)
    if not path.exists():
        result["issues"].append(f"File not found: {image_path}")
        return json.dumps(result)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        result["issues"].append(
            f"Unsupported format '{path.suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
        return json.dumps(result)

    result["format"] = path.suffix.lower().lstrip(".")
    result["file_size_kb"] = round(path.stat().st_size / 1024, 1)

    if Image is None:
        result["issues"].append("Pillow not installed — cannot inspect image metadata")
        return json.dumps(result)

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            result["width_px"] = img.width
            result["height_px"] = img.height
            result["mode"] = img.mode
            result["has_transparency"] = img.mode in ("RGBA", "LA", "P")

            raw_dpi = img.info.get("dpi", (96.0, 96.0))
            if isinstance(raw_dpi, (int, float)):
                raw_dpi = (float(raw_dpi), float(raw_dpi))
            dpi_x = float(raw_dpi[0]) if raw_dpi[0] > 0 else 96.0
            dpi_y = float(raw_dpi[1]) if raw_dpi[1] > 0 else 96.0
            result["dpi_x"] = round(dpi_x, 1)
            result["dpi_y"] = round(dpi_y, 1)
            result["width_mm"] = round(img.width / dpi_x * 25.4, 2)
            result["height_mm"] = round(img.height / dpi_y * 25.4, 2)

    except Exception as exc:
        result["issues"].append(f"Cannot open image: {exc}")
        return json.dumps(result)

    if result["width_px"] < MIN_DIM or result["height_px"] < MIN_DIM:
        result["issues"].append(
            f"Image too small ({result['width_px']}×{result['height_px']}px, min {MIN_DIM}×{MIN_DIM})"
        )

    if result["dpi_x"] < 72:
        result["issues"].append(f"DPI too low: {result['dpi_x']} (minimum 72)")
    elif result["dpi_x"] < RECOMMENDED_DPI:
        result["warnings"].append(
            f"DPI {result['dpi_x']} is below recommended {RECOMMENDED_DPI} — output quality may be reduced"
        )

    dpi_score = min(result.get("dpi_x", 96) / RECOMMENDED_DPI, 1.0)
    size_score = min(min(result.get("width_px", 0), result.get("height_px", 0)) / 1000, 1.0)
    result["quality_score"] = round(dpi_score * 0.6 + size_score * 0.4, 3)

    result["valid"] = len(result["issues"]) == 0
    return json.dumps(result)


validate_image_tool = StructuredTool.from_function(
    func=_validate_image,
    name="validate_image",
    description=(
        "Validates an image file before vectorization. Checks format, dimensions, DPI, and color mode. "
        "Returns JSON with: valid (bool), width_px, height_px, width_mm, height_mm, dpi_x, dpi_y, mode, "
        "has_transparency, quality_score (0-1), issues (blockers), warnings (non-fatal). "
        "ALWAYS call this tool first."
    ),
    args_schema=ValidateImageInput,
)
