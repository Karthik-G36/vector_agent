"""
Main vectorisation pipeline.

Two routes — selected automatically:

  Route A  vtracer  (default, strongly preferred)
  ────────────────────────────────────────────────
  Stage 1  Preprocess      Lanczos upscale to >=1400 px + bilateral denoising
  Stage 1b Flatten BG      Flood-fill background + Gaussian blur kills JPEG banding
  Stage 2  LLM analyse     GPT-4o returns image_type / complexity flag
  Stage 3  vtracer         Rust vectoriser -> layered SVG (replaces stages 3-5 of
                           the old quantize->trace->assemble chain)
  Stage 4  EPS export      Inkscape CLI -> CairoSVG -> built-in parser chain

  Route B  Classic  (fallback if vtracer not installed)
  ───────────────────────────────────────────────────────
  Stages 1-6 of the original CIELAB median-cut pipeline.

Why vtracer is better
─────────────────────
The old approach (colour quantization -> OpenCV contour tracing) creates
inaccurate blobs.  vtracer uses a production-grade Rust algorithm:
  - Bezier spline fitting with corner detection
  - Perceptual colour quantisation
  - Stacked layer ordering (shadows under text automatically)

Result dict always contains:
  svg_path    path to intermediate SVG
  eps_path    path to final EPS
  backend     EPS conversion backend
  analysis    raw LLM dict (or {})
  route       "vtracer" | "classic"
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.conversions import mm_to_pts

from .eps_exporter import export_to_eps
from .llm_analyzer import analyze_image
from . import vtracer_engine

# Minimum working dimension
_MIN_DIM = 1400


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Lanczos4 upscale to >=1400 px + bilateral denoising."""
    h, w = img_bgr.shape[:2]
    if min(h, w) < _MIN_DIM:
        scale = _MIN_DIM / min(h, w)
        img_bgr = cv2.resize(
            img_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=55, sigmaSpace=55)


def _flatten_background(img_bgr: np.ndarray, tolerance: int = 22) -> np.ndarray:
    """
    Flood-fill from all four corners, then Gaussian-blur those pixels.

    JPEG compression creates 8x8-block boundaries in a nominally-uniform
    background.  After Lanczos upscaling they become ~30-px bands.  The
    bilateral filter preserves them as 'edges'.  Pre-blurring the background
    makes it uniform so the vectoriser doesn't waste layers on background noise.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img_work = img_bgr.copy()

    for cy, cx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(
            img_work, mask, (cx, cy), (0, 0, 0),
            loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
        )

    bg_px = mask[1:-1, 1:-1] > 0
    if bg_px.sum() < 500:
        return img_bgr

    blurred = cv2.GaussianBlur(img_bgr, (21, 21), 7.0)
    result = img_bgr.copy()
    result[bg_px] = blurred[bg_px]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────────────────────────────────────

def vectorize(
    image_path: str,
    output_eps: str,
    width_mm: float = 100.0,
    height_mm: float = 100.0,
    save_svg: bool = True,
) -> dict:
    """
    Run the vectorisation pipeline and return a result dict.

    Parameters
    ----------
    image_path : absolute path to input raster image
    output_eps : where to write the final EPS file
    width_mm   : target EPS canvas width in millimetres
    height_mm  : target EPS canvas height in millimetres
    save_svg   : whether to persist the intermediate SVG alongside the EPS
    """
    result: dict = {"image_path": image_path}

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    print("  [1/4] Preprocessing (Lanczos upscale + denoise + BG flatten)...")
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = _preprocess(img_bgr)
    img = _flatten_background(img)
    h, w = img.shape[:2]
    print(f"         working resolution: {w}x{h} px")

    # ── Stage 2: LLM analysis (for complexity flag) ───────────────────────────
    print("  [2/4] LLM analysis (GPT-4o)...")
    analysis = analyze_image(image_path)
    result["analysis"] = analysis

    image_type = str(analysis.get("image_type", "flat")).lower()
    has_gradients = bool(analysis.get("has_gradients", False))
    is_complex = image_type in {"gradient", "3d", "complex"} or has_gradients
    print(f"         image_type={image_type}  complex={is_complex}")

    # ── Stage 3: Vectorise ────────────────────────────────────────────────────
    if vtracer_engine.is_available():
        route = "vtracer"
        params = vtracer_engine.params_from_analysis(analysis)
        image_class = analysis.get("image_class", "illustration")
        print(f"  [3/4] vtracer vectorisation (class={image_class}  "
              f"colormode={params.get('colormode','color')}  "
              f"color_precision={params.get('color_precision','?')}  "
              f"layer_diff={params.get('layer_difference','?')})...")
        svg_str = vtracer_engine.trace(img, **params)
        result["n_regions"] = svg_str.count("<path")
        result["n_gradients"] = 0
        result["n_effects"] = 0
        print(f"         {result['n_regions']} paths generated")
    else:
        # Fallback to classic CIELAB pipeline (much lower quality)
        route = "classic"
        print("  [3/4] Classic pipeline (vtracer not available — install with: pip install vtracer)...")
        svg_str = _classic_pipeline(img, is_complex, analysis, w, h)
        result["n_regions"] = svg_str.count("<path")
        result["n_gradients"] = 0
        result["n_effects"] = 0

    result["route"] = route

    # ── Save SVG ──────────────────────────────────────────────────────────────
    svg_path = str(Path(output_eps).with_suffix(".svg"))
    if save_svg:
        Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write(svg_str)
        svg_kb = Path(svg_path).stat().st_size // 1024
        print(f"         SVG saved: {Path(svg_path).name} ({svg_kb} KB)")
    result["svg_path"] = svg_path

    # ── Stage 4: EPS export ───────────────────────────────────────────────────
    print("  [4/4] Exporting EPS...")
    width_pts  = mm_to_pts(width_mm)
    height_pts = mm_to_pts(height_mm)
    backend = export_to_eps(svg_str, output_eps, width_pts, height_pts)

    eps_kb = Path(output_eps).stat().st_size / 1024
    print(f"         EPS saved: {Path(output_eps).name} ({eps_kb:.1f} KB)  backend={backend}")

    result["eps_path"] = output_eps
    result["backend"] = backend
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Classic fallback (used only when vtracer is not installed)
# ──────────────────────────────────────────────────────────────────────────────

def _classic_pipeline(
    img: np.ndarray,
    is_complex: bool,
    analysis: dict,
    w: int,
    h: int,
) -> str:
    """Original CIELAB median-cut pipeline — fallback when vtracer is absent."""
    from .color_engine import segment_image
    from .effect_detector import detect_effects
    from .svg_builder import build_svg

    def _bg_hex(img_bgr: np.ndarray) -> str:
        bh, bw = img_bgr.shape[:2]
        corners = np.array([
            img_bgr[0, 0], img_bgr[0, bw - 1],
            img_bgr[bh - 1, 0], img_bgr[bh - 1, bw - 1],
        ], dtype=float)
        b, g, r = corners.mean(axis=0).round().astype(int)
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    raw_k = int(analysis.get("total_k", 0))
    k_floor = 8 if is_complex else 5
    if raw_k < k_floor:
        raw_k = 10 if is_complex else 7
    k = max(k_floor, min(raw_k, 20))

    regions = segment_image(img, k)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    regions_info = [(r.mask, r.bgr_color, r.hex_color) for r in regions]
    effects = detect_effects(regions_info, img, img_lab)
    bg_color = _bg_hex(img)
    return build_svg(regions, w, h, effects=effects, bg_hex=bg_color)
