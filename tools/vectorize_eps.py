import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from utils.conversions import mm_to_pts

POTRACE_BIN = os.getenv("POTRACE_BIN", "potrace")


class VectorizeEPSInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the (preprocessed) image")
    image_type: str = Field(
        ...,
        description="'flat' or 'photo' — from segment_and_profile output. "
                    "flat → potrace or quantize+bezier, photo → kmeans+bezier.",
    )
    width_mm: float = Field(..., description="Target output width in millimetres")
    height_mm: float = Field(..., description="Target output height in millimetres")
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save the EPS file. Defaults to <name>.eps next to input.",
    )


# ---------------------------------------------------------------------------
# Step 1 — Upscale + remove JPEG artifacts before any tracing
# ---------------------------------------------------------------------------

def _prepare_image(img_bgr, target_min_dim: int = 800):
    """
    Upscale small images with Lanczos, then apply bilateral filter to
    dissolve JPEG compression halos while keeping crisp edges.
    Without this step, artifact halos around letters get traced as
    extra noise outlines.
    """
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < target_min_dim:
        scale = target_min_dim / min_dim
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    # d=9: neighbourhood; sigmaColor=50: merges pixels within ±50 intensity;
    # sigmaSpace=50: spatial reach — removes JPEG halos, preserves hard edges.
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=50, sigmaSpace=50)


# ---------------------------------------------------------------------------
# Step 2 — Color region extraction (two strategies)
# ---------------------------------------------------------------------------

def _min_pixel_threshold(h: int, w: int) -> int:
    """0.005 % of image area — keeps small text, drops single-pixel noise."""
    return max(10, int(h * w * 0.00005))


def _get_regions_quantize(img_bgr, n_colors: int) -> list:
    """
    PIL FASTOCTREE quantization for flat logos.
    Preserves exact original palette colors; regions sorted largest-first
    so the background is always painted before foreground shapes.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(img_rgb)
    quant = pil.quantize(colors=n_colors, method=PILImage.Quantize.FASTOCTREE, dither=0)
    idx_arr = np.array(quant)
    palette = quant.getpalette()  # flat [R0,G0,B0, R1,G1,B1 ...]

    h, w = img_bgr.shape[:2]
    min_px = _min_pixel_threshold(h, w)

    regions = []
    for i in range(n_colors):
        mask = (idx_arr == i).astype(np.uint8) * 255
        px = int(np.count_nonzero(mask))
        if px < min_px:
            continue
        r, g, b = palette[i * 3: i * 3 + 3]
        regions.append((mask, (b, g, r), px))  # store BGR

    regions.sort(key=lambda x: x[2], reverse=True)   # largest area first
    return [(m, c) for m, c, _ in regions]


def _get_regions_kmeans(img_bgr, n_clusters: int) -> list:
    """K-means for photos; regions sorted largest-first."""
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    k = max(2, min(n_clusters, 16))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(img_bgr.shape[:2])
    centers = centers.astype(np.uint8)

    h, w = img_bgr.shape[:2]
    min_px = _min_pixel_threshold(h, w)

    regions = []
    for i in range(k):
        mask = (labels == i).astype(np.uint8) * 255
        px = int(np.count_nonzero(mask))
        if px < min_px:
            continue
        regions.append((mask, tuple(int(c) for c in centers[i]), px))

    regions.sort(key=lambda x: x[2], reverse=True)
    return [(m, c) for m, c, _ in regions]


# ---------------------------------------------------------------------------
# Step 3 — Mask cleaning (remove JPEG speckles, fill micro-holes)
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological open (removes isolated noise pixels) then close
    (fills tiny gaps in letter strokes).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# Step 4 — Catmull-Rom → PostScript cubic bezier (smooth curves)
# ---------------------------------------------------------------------------

def _contour_to_bezier_ps(
    cnt: np.ndarray,
    sx: float,
    sy: float,
    img_h: int,
    tension: float = 0.4,
    max_pts: int = 400,
) -> Optional[str]:
    """
    Convert an OpenCV contour to PostScript curveto commands using
    Catmull-Rom spline interpolation.

    Catmull-Rom formula:
        CP1 = P1 + tension * (P2 - P0) / 3
        CP2 = P2 - tension * (P3 - P1) / 3
    This gives smooth G1-continuous curves through every control point.
    tension=0 → straight lines, tension=1 → very round.
    """
    pts = cnt.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 3:
        return None

    # Subsample if very dense (CHAIN_APPROX_NONE on large images)
    if n > max_pts:
        step = max(1, n // max_pts)
        pts = pts[::step]
        n = len(pts)

    x0, y0 = pts[0]
    lines = [f"  {x0 * sx:.3f} {(img_h - y0) * sy:.3f} moveto"]

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        cp1 = p1 + tension * (p2 - p0) / 3.0
        cp2 = p2 - tension * (p3 - p1) / 3.0

        lines.append(
            f"  {cp1[0]*sx:.3f} {(img_h - cp1[1])*sy:.3f} "
            f"{cp2[0]*sx:.3f} {(img_h - cp2[1])*sy:.3f} "
            f"{p2[0]*sx:.3f} {(img_h - p2[1])*sy:.3f} curveto"
        )

    lines.append("  closepath")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5 — EPS writer (bezier paths + eofill for holes)
# ---------------------------------------------------------------------------

def _write_eps_bezier(
    contour_groups: list,   # [(contours, bgr_color), ...] already sorted bg-first
    img_h: int,
    img_w: int,
    width_pts: float,
    height_pts: float,
    output_path: str,
    tension: float = 0.4,
) -> None:
    sx = width_pts / img_w
    sy = height_pts / img_h

    lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {int(width_pts)} {int(height_pts)}",
        "%%Title: (vector_output)",
        "%%Creator: (vector_agent / bezier)",
        "%%EndComments",
        "%%BeginProlog",
        "%%EndProlog",
        "%%Page: 1 1",
        "gsave",
    ]

    for contours, bgr_color in contour_groups:
        b, g, r = bgr_color
        lines.append(f"{r / 255:.4f} {g / 255:.4f} {b / 255:.4f} setrgbcolor")

        path_started = False
        for cnt in contours:
            seg = _contour_to_bezier_ps(cnt, sx, sy, img_h, tension=tension)
            if seg is None:
                continue
            if not path_started:
                lines.append("newpath")
                path_started = True
            lines.append(seg)

        if path_started:
            # eofill uses even-odd rule → child contours automatically punch holes
            lines.append("eofill")

    lines += ["grestore", "showpage", "%%EOF"]

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Flat route: potrace binary (best quality when available)
# ---------------------------------------------------------------------------

def _potrace_available() -> bool:
    return shutil.which(POTRACE_BIN) is not None


def _vectorize_flat_potrace(
    image_path: str, width_pts: float, height_pts: float, output_path: str
) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}
    _, bmp = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp_bmp = tmp.name
    cv2.imwrite(tmp_bmp, bmp)
    try:
        proc = subprocess.run(
            [POTRACE_BIN, "--eps",
             "--width", f"{width_pts:.2f}pt",
             "--height", f"{height_pts:.2f}pt",
             "-o", output_path, tmp_bmp],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return {"error": f"potrace failed: {proc.stderr.strip()}"}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    finally:
        try:
            os.unlink(tmp_bmp)
        except OSError:
            pass
    return {"method": "potrace", "output_path": output_path}


# ---------------------------------------------------------------------------
# Contour-bezier route (flat fallback + photo)
# ---------------------------------------------------------------------------

def _vectorize_contours(
    image_path: str,
    image_type: str,
    width_pts: float,
    height_pts: float,
    output_path: str,
) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}

    # Upscale + remove JPEG artifacts before colour extraction
    img = _prepare_image(img, target_min_dim=800)
    h, w = img.shape[:2]

    if image_type == "flat":
        regions = (
            _get_regions_quantize(img, n_colors=8)
            if PIL_AVAILABLE
            else _get_regions_kmeans(img, n_colors=8)
        )
        tension = 0.3   # tighter curves to preserve sharp logo edges
    else:
        regions = _get_regions_kmeans(img, n_clusters=12)
        tension = 0.5

    contour_groups = []
    for mask, bgr_color in regions:
        mask = _clean_mask(mask)
        # CHAIN_APPROX_NONE: keep every boundary pixel — the bezier fitter
        # needs dense points to produce accurate smooth curves.
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid = [c for c in cnts if len(c) >= 3]
        if valid:
            contour_groups.append((valid, bgr_color))

    if not contour_groups:
        return {"error": "No contours found — image may be empty or uniform"}

    _write_eps_bezier(
        contour_groups, h, w, width_pts, height_pts, output_path, tension=tension
    )
    strategy = "quantize" if image_type == "flat" and PIL_AVAILABLE else "kmeans"
    return {"method": f"bezier_contour({strategy})", "output_path": output_path}


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def _vectorize_to_eps(
    image_path: str,
    image_type: str,
    width_mm: float,
    height_mm: float,
    output_path: Optional[str] = None,
) -> str:
    result: dict = {"image_path": image_path, "image_type": image_type}

    if not CV2_AVAILABLE:
        result["error"] = "opencv-python not installed"
        return json.dumps(result)

    if image_type not in ("flat", "photo"):
        result["error"] = f"image_type must be 'flat' or 'photo', got '{image_type}'"
        return json.dumps(result)

    width_pts = mm_to_pts(width_mm)
    height_pts = mm_to_pts(height_mm)
    result["bounding_box"] = f"%%BoundingBox: 0 0 {int(width_pts)} {int(height_pts)}"
    result["width_pts"] = round(width_pts, 2)
    result["height_pts"] = round(height_pts, 2)

    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}.eps")

    os.makedirs(Path(output_path).parent, exist_ok=True)

    if image_type == "flat" and _potrace_available():
        sub = _vectorize_flat_potrace(image_path, width_pts, height_pts, output_path)
    else:
        if image_type == "flat" and not _potrace_available():
            result["notes"] = "potrace not found — using bezier+quantize fallback"
        sub = _vectorize_contours(image_path, image_type, width_pts, height_pts, output_path)

    result.update(sub)
    return json.dumps(result)


vectorize_to_eps_tool = StructuredTool.from_function(
    func=_vectorize_to_eps,
    name="vectorize_to_eps",
    description=(
        "Converts a raster image to an EPS vector file with smooth bezier curves. "
        "image_type MUST come from segment_and_profile: "
        "'flat' uses potrace if available, else PIL quantize + Catmull-Rom bezier tracing; "
        "'photo' uses K-means + Catmull-Rom bezier tracing. "
        "JPEG artifacts are removed by bilateral filtering before tracing. "
        "width_mm/height_mm set the EPS %%BoundingBox (pts = mm × 2.8346). "
        "Returns JSON with: output_path, bounding_box, method, width_pts, height_pts."
    ),
    args_schema=VectorizeEPSInput,
)
