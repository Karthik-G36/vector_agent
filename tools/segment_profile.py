import json
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    import cv2
    import numpy as np
    from PIL import Image
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

FLAT_MAX_COLORS = 8
FLAT_MAX_EDGE_DENSITY = 0.12


class SegmentProfileInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the image to segment and profile")


def _segment_and_profile(image_path: str) -> str:
    result: dict = {"image_path": image_path}

    if not LIBS_AVAILABLE:
        result["error"] = "opencv-python or Pillow not installed"
        return json.dumps(result)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        result["error"] = f"Cannot read image: {image_path}"
        return json.dumps(result)

    h, w = img_bgr.shape[:2]
    result["width_px"] = w
    result["height_px"] = h

    # --- Color clustering via K-means ---
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    n_clusters = min(16, max(2, len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))) ) // 100 + 2))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)

    dominant_colors = []
    for idx in sorted_idx:
        b, g, r = centers[unique_labels[idx]].astype(int)
        coverage = round(float(counts[idx]) / len(labels), 4)
        dominant_colors.append({"rgb": [int(r), int(g), int(b)], "coverage": coverage})

    result["n_colors"] = int(n_clusters)
    result["dominant_colors"] = dominant_colors[:8]

    # --- Edge density (Canny) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / (w * h)
    result["edge_density"] = round(edge_density, 5)

    # --- Transparency check ---
    try:
        with Image.open(image_path) as pil_img:
            result["has_transparency"] = pil_img.mode in ("RGBA", "LA", "P")
    except Exception:
        result["has_transparency"] = False

    # --- Classification ---
    is_flat = n_clusters <= FLAT_MAX_COLORS and edge_density < FLAT_MAX_EDGE_DENSITY
    result["image_type"] = "flat" if is_flat else "photo"
    result["classification_reason"] = (
        f"n_colors={n_clusters} ({'≤' if n_clusters <= FLAT_MAX_COLORS else '>'}{FLAT_MAX_COLORS}), "
        f"edge_density={edge_density:.4f} ({'≤' if edge_density <= FLAT_MAX_EDGE_DENSITY else '>'}{FLAT_MAX_EDGE_DENSITY})"
    )

    # --- Complexity score (0=simple/flat, 1=complex/photo) ---
    color_complexity = min(n_clusters / 16, 1.0)
    edge_complexity = min(edge_density / 0.3, 1.0)
    result["complexity_score"] = round(color_complexity * 0.5 + edge_complexity * 0.5, 3)

    return json.dumps(result)


segment_and_profile_tool = StructuredTool.from_function(
    func=_segment_and_profile,
    name="segment_and_profile",
    description=(
        "Segments an image using K-means color clustering and profiles its complexity. "
        "Returns JSON with: image_type ('flat' or 'photo'), n_colors, dominant_colors "
        "(list of {rgb, coverage}), edge_density, complexity_score (0-1), has_transparency. "
        "The image_type field MUST be passed to vectorize_to_eps to select the correct route: "
        "'flat' → potrace (clean vector paths), 'photo' → pycairo contour tracing."
    ),
    args_schema=SegmentProfileInput,
)
