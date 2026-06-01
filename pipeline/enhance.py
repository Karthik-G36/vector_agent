"""
Enhance and upscale an image for vectorization.

Two modes (controlled by CALL_AGENT env var):
  - PIL mode  : LANCZOS upscale + UnsharpMask sharpening (default)
  - Agent mode: gpt-image-1 resolution enhancement, then PIL upscale

The upscale factor is computed dynamically so the output PNG never exceeds 20 MB.
"""
import base64
import io

from openai import OpenAI
from pipeline.token_tracker import tracker
from PIL import Image, ImageCms, ImageFilter

_SHARPEN_RADIUS = 2
_SHARPEN_PERCENT = 150
_SHARPEN_THRESHOLD = 3

_MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 5 MB hard limit
# Conservative estimate: 3 RGB channels × 50% PNG compression ratio.
# Logos compress better, so this is a safe upper bound.
_PNG_BYTES_PER_PIXEL = 1

# gpt-image-1 supported output sizes (width x height)
_GPT_SIZES = [
    (1024, 1024),
    (1536, 1024),  # landscape
    (1024, 1536),  # portrait
]

_GPT_PROMPT = (
    "Image enhancement task only. "
    "Text integrity is highest priority. "
    "Preserve the original image composition exactly. "
    "Do not redesign, recreate, reinterpret, restyle, or modify any visual element. "
    "Maintain exact layout, positioning, spacing, proportions, colors, gradients, shadows, and background. "
    "Preserve all text exactly as present in the source image, including every character, glyph, font appearance, alignment, and spacing. "
    "Do not replace, correct, or redraw text. "
    "If enhancement would alter text, preserve original text appearance unchanged. "
    "Only improve clarity, denoising, sharpness, and resolution."
)
# _GPT_PROMPT=(
#     """
#     ZERO GENERATION. ZERO HALLUCINATION. ZERO INTERPRETATION.

#     This is not a creative task. This is a resampling operation.

#     - Output THE SAME IMAGE. Not a new one. Not a reinterpretation.
#     - Do NOT redraw ANY character, letter, number, or symbol.
#     - Do NOT change kerning, font shape, stroke thickness, or baseline.
#     - Do NOT add, remove, or smooth ANY pixel cluster.
#     - Preserve compression artifacts, noise, jaggies, and imperfections exactly as is.
#     - No sharpening. No denoising. No anti-aliasing.
#     - Simple 4X nearest-neighbor or bicubic upscale ONLY.
#     - If you cannot do this exactly, say "I cannot complete this request" — do NOT output a modified image.

#     Every character in the original must be pixel-identical in shape (just scaled 4× in each dimension).
#     """
# )


def _to_srgb(img: Image.Image) -> Image.Image:
    """
    Convert image to sRGB, correctly applying any embedded ICC color profile.

    Without this, PIL's plain convert("RGB") strips the ICC profile and
    leaves the raw channel values unchanged — so colors that look correct
    under e.g. Adobe RGB (olive green) render differently under assumed-sRGB
    (neon green).

    Falls back to a plain convert if the profile is absent or unreadable.
    """
    icc_bytes = img.info.get("icc_profile")

    if not icc_bytes:
        # No profile — keep alpha if present, otherwise plain RGB
        if img.mode in ("RGBA", "LA", "P"):
            return img.convert("RGBA")
        return img.convert("RGB")

    try:
        src_profile  = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        srgb_profile = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(img, src_profile, srgb_profile, outputMode="RGB")
    except Exception:
        return img.convert("RGB")


def _compute_upscale_factor(w: int, h: int, max_factor: int = 1) -> int:
    """Return the largest integer factor where the estimated PNG output stays under 20 MB."""
    for factor in range(max_factor, 0, -1):
        if w * factor * h * factor * _PNG_BYTES_PER_PIXEL <= _MAX_OUTPUT_BYTES:
            return factor
    return 1


def _pick_gpt_size(w: int, h: int) -> str:
    ratio = w / h
    best = min(_GPT_SIZES, key=lambda s: abs((s[0] / s[1]) - ratio))
    return f"{best[0]}x{best[1]}"


def _to_rgb_png_bytes(input_path: str) -> bytes:
    """Return the image as sRGB PNG bytes suitable for the OpenAI API."""
    with Image.open(input_path) as img:
        rgb = _to_srgb(img)
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def enhance_with_agent(input_path: str, output_path: str) -> None:
    """
    Enhance resolution via gpt-image-1, then apply PIL upscale + sharpen.
    Preserves aspect ratio by selecting the closest supported gpt-image-1 size.
    The upscale factor is chosen dynamically to keep output under 20 MB.
    """
    with Image.open(input_path) as img:
        orig_w, orig_h = img.size

    size = _pick_gpt_size(orig_w, orig_h)
    print(f"      [gpt-image-1] {orig_w}x{orig_h} -> requesting {size} ...")

    client = OpenAI()
    png_bytes = _to_rgb_png_bytes(input_path)

    response = client.images.edit(
        model="gpt-image-1",
        image=("image.png", png_bytes, "image/png"),
        prompt=_GPT_PROMPT,
        size=size,
    )

    usage = getattr(response, "usage", None)
    if usage:
        tracker.record(
            label="gpt-image-1 (enhance)",
            model="gpt-image-1",
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
    raw = base64.b64decode(response.data[0].b64_json)
    with Image.open(io.BytesIO(raw)) as gpt_img:
        gpt_w, gpt_h = gpt_img.size
        factor = _compute_upscale_factor(gpt_w, gpt_h)
        target_w = gpt_w * factor
        target_h = gpt_h * factor
        upscaled = gpt_img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        sharpened = upscaled.filter(
            ImageFilter.UnsharpMask(
                radius=_SHARPEN_RADIUS,
                percent=_SHARPEN_PERCENT,
                threshold=_SHARPEN_THRESHOLD,
            )
        )
        sharpened.save(output_path, format="PNG", optimize=False)

    print(f"      [gpt-image-1] done: {orig_w}x{orig_h} -> {target_w}x{target_h} ({factor}x LANCZOS + sharpen)")


def enhance_image(input_path: str, output_path: str) -> None:
    with Image.open(input_path) as img:
        orig_w, orig_h = img.size
        img = _to_srgb(img)

        factor = _compute_upscale_factor(orig_w, orig_h)
        target_w = orig_w * factor
        target_h = orig_h * factor

        upscaled = img.resize((target_w, target_h), Image.LANCZOS)
        sharpened = upscaled.filter(
            ImageFilter.UnsharpMask(
                radius=_SHARPEN_RADIUS,
                percent=_SHARPEN_PERCENT,
                threshold=_SHARPEN_THRESHOLD,
            )
        )
        sharpened.save(output_path, format="PNG", optimize=False)

    print(f"      {orig_w}x{orig_h} -> {target_w}x{target_h} ({factor}x LANCZOS + sharpen)")
