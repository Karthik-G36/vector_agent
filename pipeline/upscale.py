"""
Real-ESRGAN upscaling step (optional).

Overwrites the input file in-place with a 4x upscaled, artifact-free version.
Only imported when USE_REALESRGAN=true in .env — torch never loads otherwise.

Requires: torch, torchvision, opencv-python, realesrgan, basicsr, facexlib, gfpgan
"""

# Torchvision compatibility patch — must run before any basicsr/realesrgan import.
# Newer torchvision removed functional_tensor but basicsr still imports from it.
import sys
import types

try:
    from torchvision.transforms.functional import rgb_to_grayscale as _rgb2gray
    _compat = types.ModuleType("torchvision.transforms.functional_tensor")
    _compat.rgb_to_grayscale = _rgb2gray
    sys.modules["torchvision.transforms.functional_tensor"] = _compat
except Exception:
    pass

import io

import cv2
import numpy as np
from PIL import Image, ImageCms
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/"
    "download/v0.1.0/RealESRGAN_x4plus.pth"
)

_upsampler: RealESRGANer | None = None


def _get_upsampler(tile: int) -> RealESRGANer:
    global _upsampler
    if _upsampler is None:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        _upsampler = RealESRGANer(
            scale=4,
            model_path=_MODEL_URL,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
    return _upsampler


def _to_srgb(img: Image.Image) -> Image.Image:
    """Convert to sRGB honouring any embedded ICC profile, then return an RGB image."""
    icc_bytes = img.info.get("icc_profile")
    if not icc_bytes:
        return img.convert("RGB")
    try:
        src_profile  = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        srgb_profile = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(img, src_profile, srgb_profile, outputMode="RGB")
    except Exception:
        return img.convert("RGB")


def upscale_image(image_path: str, tile: int = 512) -> None:
    """
    Upscale image at image_path 4x with Real-ESRGAN and overwrite it in-place.
    tile=512 is safe for CPU and low-VRAM GPU; use tile=0 for full-image on high-VRAM GPU.
    ICC profile is applied before upscaling so colors are correctly mapped to sRGB.
    """
    upsampler = _get_upsampler(tile)

    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        has_icc = False #bool(img.info.get("icc_profile"))
        rgb = _to_srgb(img)
        img_np = np.array(rgb)

    if has_icc:
        print(f"      [ICC] profile detected — converted to sRGB before upscaling")

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    output_bgr, _ = upsampler.enhance(img_bgr, outscale=4)

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    out_h, out_w = output_rgb.shape[:2]

    Image.fromarray(output_rgb).save(image_path, format="PNG")
    print(f"      Real-ESRGAN: {orig_w}x{orig_h} -> {out_w}x{out_h}")
