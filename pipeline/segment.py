"""Color-based segmentation: quantize image → per-color binary masks.

For logos and flat-color artwork this approach is far more reliable than
SAM2 auto mode, which segments by spatial regions and tears apart letter
forms. Color quantization groups pixels by actual color, so each mask is
a clean, connected region of one palette entry.

Stage 6: After masking, dominant color is extracted from the ORIGINAL
(pre-quantize) image pixels that fall inside each mask using k-means (k=1).
This preserves the true color rather than the flattened palette entry.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Number of palette colors to quantize to for segmentation boundaries.
_QUANTIZE_COLORS = 8

# Drop color layers whose combined pixel count is below this fraction of the
# total image area — eliminates JPEG compression fringe and anti-alias dust.
_MIN_AREA_RATIO = 0.002   # 0.2 %

# Maximum number of color layers to trace. Sorted largest-first so the
# background is always layer_0 and small accent colors are at the top.
_MAX_LAYERS = 16

# Masks with mostly edge pixels are usually anti-alias/JPEG fringe layers, not
# semantic artwork layers. We either merge/drop them before Potrace sees them.
_MIN_COMPONENT_AREA_RATIO = 0.00005
_MORPH_KERNEL_SIZE = 5

# High-contrast two-tone artwork is better handled as foreground/background
# than as palette entries. This catches logos such as dark text on pale paper.
_LOGO_DARK_RATIO_MAX = 0.25
_LOGO_LIGHT_RATIO_MIN = 0.55
_LOGO_MID_RATIO_MAX = 0.20


def _dominant_color_kmeans(original_rgb: np.ndarray, mask_bool: np.ndarray) -> tuple[int, int, int]:
    """
    Extract dominant color from original image pixels inside the mask.
    Uses k-means k=1 — returns the centroid as (R, G, B).
    Falls back to mean if sklearn is unavailable.
    """
    pixels = original_rgb[mask_bool]   # shape: (N, 3)
    if len(pixels) == 0:
        return (0, 0, 0)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=1, n_init=10, random_state=0)
        km.fit(pixels)
        r, g, b = km.cluster_centers_[0].astype(int)
    except ImportError:
        # fallback: arithmetic mean
        r, g, b = pixels.mean(axis=0).astype(int)

    return (int(r), int(g), int(b))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    )


def _looks_like_dark_on_light_logo(original_rgb: np.ndarray) -> bool:
    lum = _luminance(original_rgb)
    dark = np.mean(lum < 96)
    light = np.mean(lum > 210)
    mid = 1.0 - dark - light
    return (
        0.005 <= dark <= _LOGO_DARK_RATIO_MAX
        and light >= _LOGO_LIGHT_RATIO_MIN
        and mid <= _LOGO_MID_RATIO_MAX
    )


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    best_var = -1.0
    threshold = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            threshold = t
    return threshold


def _binary_close(mask_bool: np.ndarray) -> np.ndarray:
    """Close tiny anti-alias holes without requiring OpenCV/scipy."""
    img = Image.fromarray(np.where(mask_bool, 255, 0).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(_MORPH_KERNEL_SIZE))
    img = img.filter(ImageFilter.MinFilter(_MORPH_KERNEL_SIZE))
    return np.array(img, dtype=np.uint8) > 127


def _remove_small_components(mask_bool: np.ndarray, min_px: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return mask_bool

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros(mask_bool.shape, dtype=bool)
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_px:
            cleaned |= labels == label
    return cleaned


def _clean_mask(mask_bool: np.ndarray, total_px: int) -> np.ndarray:
    min_component_px = max(8, int(total_px * _MIN_COMPONENT_AREA_RATIO))
    closed = _binary_close(mask_bool)
    return _remove_small_components(closed, min_component_px)


def _make_logo_segments(original_rgb: np.ndarray, out_dir: Path) -> list[dict]:
    lum = _luminance(original_rgb)
    threshold = _otsu_threshold(lum.astype(np.uint8))
    dark_mask = lum <= threshold
    total_px = dark_mask.size
    dark_mask = _clean_mask(dark_mask, total_px)
    bg_mask = ~dark_mask

    segments = []
    for idx, mask_bool in enumerate((bg_mask, dark_mask)):
        area = int(np.sum(mask_bool))
        if area == 0:
            continue
        color = _dominant_color_kmeans(original_rgb, mask_bool)
        mask_arr = np.where(mask_bool, 0, 255).astype(np.uint8)
        mask_path = out_dir / f"mask_{idx:03d}.png"
        Image.fromarray(mask_arr).save(str(mask_path))
        segments.append({"mask_path": str(mask_path), "color": color, "area": area})
        print(f"      mask_{idx:03d}: area={area:,}, dominant_color=rgb{color}")

    segments.sort(key=lambda x: x["area"], reverse=True)
    for r in segments:
        del r["area"]
    return segments


def segment_image(input_path: str, output_dir: str) -> list[dict]:
    """
    Segment input image by color quantization, then extract per-region
    dominant colors from the original (unquantized) pixels via k-means.

    Steps:
      1. Quantize to _QUANTIZE_COLORS palette colors (no dithering) to get
         clean segment boundaries.
      2. For each palette entry covering >= _MIN_AREA_RATIO of the image,
         write a binary mask: black = this color, white = everything else.
      3. Run k-means (k=1) on the ORIGINAL image pixels inside each mask
         to get the true dominant color (Stage 6).
      4. Return list of {"mask_path": str, "color": (r,g,b)} sorted
         largest-first (largest area = bottom SVG layer).

    The masks use the Potrace convention: black = region to trace.
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    total_px = w * h
    min_px = int(total_px * _MIN_AREA_RATIO)

    original_rgb = np.array(img, dtype=np.uint8)  # HxW x 3 — original pixels
    out_dir = Path(output_dir)

    if _looks_like_dark_on_light_logo(original_rgb):
        print("      logo-like dark-on-light artwork detected; using Otsu foreground mask")
        results = _make_logo_segments(original_rgb, out_dir)
        print(f"      {len(results)} logo layers from {w}x{h} image")
        return results

    # Quantize for boundary detection only
    quantized = img.quantize(colors=_QUANTIZE_COLORS, dither=Image.Dither.NONE)
    indices = np.array(quantized, dtype=np.uint8)  # HxW index map

    results = []

    for idx in range(_QUANTIZE_COLORS):
        mask_bool = (indices == idx)               # True where this color
        mask_bool = _clean_mask(mask_bool, total_px)
        area = int(np.sum(mask_bool))
        if area < min_px:
            continue

        # Stage 6: dominant color from original pixels inside this mask
        color = _dominant_color_kmeans(original_rgb, mask_bool)

        # Black = segment (Potrace traces dark), white = background
        mask_arr = np.where(mask_bool, 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr)
        mask_path = out_dir / f"mask_{idx:03d}.png"
        mask_img.save(str(mask_path))

        results.append({"mask_path": str(mask_path), "color": color, "area": area})
        print(f"      mask_{idx:03d}: area={area:,}, dominant_color=rgb{color}")

    # Sort largest-first so background renders at bottom of SVG
    results.sort(key=lambda x: x["area"], reverse=True)
    results = results[:_MAX_LAYERS]

    # Remove "area" key — callers only expect mask_path + color
    for r in results:
        del r["area"]

    print(f"      {len(results)} color layers from {w}x{h} image")
    return results


def merge_masks_by_color(segments: list[dict], _output_dir) -> list[dict]:
    """
    Pass-through: color-quantized masks are already one-per-color.
    Kept for API compatibility with main.py / api.py callers.
    """
    return segments
