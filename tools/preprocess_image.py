import json
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from skimage.transform import rotate
    from skimage.feature import canny
    from scipy.ndimage import interpolation as ndimage_interp
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

VALID_OPS = {"deskew", "denoise", "clahe", "sharpen"}


class PreprocessInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the input image")
    ops: List[str] = Field(
        default=["deskew", "denoise", "clahe", "sharpen"],
        description=(
            "Ordered list of preprocessing operations to apply. "
            "Valid values: deskew, denoise, clahe, sharpen. "
            "For clean digital images skip deskew. For high-quality scans skip all."
        ),
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save the result. If omitted, saves next to input as <name>_preprocessed.<ext>.",
    )


def _deskew(img_bgr):
    if not SKIMAGE_AVAILABLE:
        return img_bgr, 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    edges = canny(gray.astype(float) / 255.0)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 10:
        return img_bgr, 0.0
    angle = _compute_skew_angle(coords)
    if abs(angle) < 0.5:
        return img_bgr, angle
    rotated = rotate(img_bgr, angle, resize=False, preserve_range=True, mode="edge").astype(np.uint8)
    return rotated, angle


def _compute_skew_angle(coords):
    """Minimal PCA-based skew estimation."""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle = float(np.degrees(np.arctan2(principal[0], principal[1])))
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle


def _denoise(img_bgr):
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoising(img_bgr, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _clahe(img_bgr):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img_bgr.shape) == 3:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a, b = cv2.split(lab)
        l_ch = clahe_obj.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a, b]), cv2.COLOR_LAB2BGR)
    return clahe_obj.apply(img_bgr)


def _sharpen(img_bgr):
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)


def _preprocess_image(image_path: str, ops: List[str], output_path: Optional[str] = None) -> str:
    result: dict = {"image_path": image_path, "ops_applied": [], "ops_skipped": [], "notes": []}

    if not CV2_AVAILABLE:
        result["error"] = "opencv-python not installed"
        return json.dumps(result)

    unknown = [op for op in ops if op not in VALID_OPS]
    if unknown:
        result["error"] = f"Unknown ops: {unknown}. Valid: {sorted(VALID_OPS)}"
        return json.dumps(result)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        result["error"] = f"Cannot read image: {image_path}"
        return json.dumps(result)

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None

    for op in ops:
        if op == "deskew":
            if not SKIMAGE_AVAILABLE:
                result["ops_skipped"].append("deskew (scikit-image not installed)")
                continue
            bgr, angle = _deskew(bgr)
            if alpha is not None:
                alpha, _ = _deskew(alpha)
            result["ops_applied"].append(f"deskew (angle={angle:.2f}°)")

        elif op == "denoise":
            bgr = _denoise(bgr)
            result["ops_applied"].append("denoise")

        elif op == "clahe":
            bgr = _clahe(bgr)
            result["ops_applied"].append("clahe")

        elif op == "sharpen":
            bgr = _sharpen(bgr)
            result["ops_applied"].append("sharpen")

    out = cv2.merge([bgr, alpha]) if has_alpha else bgr

    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_preprocessed{p.suffix}")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    cv2.imwrite(output_path, out)
    result["output_path"] = output_path
    return json.dumps(result)


preprocess_image_tool = StructuredTool.from_function(
    func=_preprocess_image,
    name="preprocess_image",
    description=(
        "Applies preprocessing operations to an image to improve vectorization quality. "
        "ops is an ordered list: deskew (fix rotation), denoise (reduce noise), "
        "clahe (local contrast), sharpen (unsharp mask). "
        "Skip deskew for clean digital exports. Skip all for high-quality scans. "
        "Returns JSON with output_path, ops_applied, ops_skipped."
    ),
    args_schema=PreprocessInput,
)
