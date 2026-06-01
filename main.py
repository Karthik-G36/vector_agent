#!/usr/bin/env python3
"""
Vector Agent — Image to Vector Pipeline
Usage: python main.py <image_path> [options]

Accepts: PNG, JPG, JPEG, WEBP, BMP, TIFF, TIF, GIF
Pipeline:
  1. Enhance with LANCZOS upscale + UnsharpMask  (or gpt-image-1 when --call-agent)
  2. Upscale with Real-ESRGAN (--use-realesrgan / USE_REALESRGAN=true)
  3. Remove background (--remove-bg / REMOVE_BG=true)
  4. Segment with SAM 2 (--use-sam2 / USE_SAM2=true) → per-region binary masks
     OR trace whole image (default)
  5. Convert SVG to EPS via Inkscape

CLI flags override the matching .env variable for that run.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pipeline.run import PipelineConfig, run_pipeline

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() == "true"


def _resolve(arg_value, env_key: str, default: bool = False) -> bool:
    """CLI arg > env var > default."""
    if arg_value is not None:
        return arg_value
    return _env_bool(env_key, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance image, then convert to SVG and EPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--no-enhance", action="store_true", help="Skip enhancement step entirely")
    parser.add_argument(
        "--call-agent", action=argparse.BooleanOptionalAction, default=None,
        help="Use gpt-image-1 for enhancement (overrides CALL_AGENT env)",
    )
    parser.add_argument(
        "--use-vtracer", action=argparse.BooleanOptionalAction, default=None,
        help="Use vtracer for PNG→SVG instead of Inkscape (overrides USE_VTRACER env)",
    )
    parser.add_argument(
        "--use-vtracer-agent", action=argparse.BooleanOptionalAction, default=None,
        help="After vtracer trace, run GPT-4o vision agent to optimize params (overrides USE_VTRACER_AGENT env)",
    )
    parser.add_argument(
        "--use-realesrgan", action=argparse.BooleanOptionalAction, default=None,
        help="Run Real-ESRGAN AI upscale after PIL upscale (overrides USE_REALESRGAN env)",
    )
    parser.add_argument(
        "--realesrgan-tile", type=int, default=None,
        help="Real-ESRGAN tile size (overrides REALESRGAN_TILE env, default: 512)",
    )
    parser.add_argument(
        "--remove-bg", action=argparse.BooleanOptionalAction, default=None,
        help="Remove background before tracing (overrides REMOVE_BG env)",
    )
    parser.add_argument(
        "--use-sam2", action=argparse.BooleanOptionalAction, default=None,
        help="Segment with SAM 2 then compose layered SVG (overrides USE_SAM2 env)",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    if img_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(
            f"Error: Unsupported format '{img_path.suffix}'.\n"
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    esrgan_tile = args.realesrgan_tile if args.realesrgan_tile is not None else int(os.getenv("REALESRGAN_TILE", "512"))

    cfg = PipelineConfig(
        call_agent        = _resolve(args.call_agent,        "CALL_AGENT",        False),
        use_vtracer       = _resolve(args.use_vtracer,       "USE_VTRACER",       False),
        use_vtracer_agent = _resolve(args.use_vtracer_agent, "USE_VTRACER_AGENT", False),
        use_esrgan        = _resolve(args.use_realesrgan,    "USE_REALESRGAN",    False),
        esrgan_tile       = esrgan_tile,
        remove_bg         = _resolve(args.remove_bg,         "REMOVE_BG",         False),
        use_sam2          = _resolve(args.use_sam2,          "USE_SAM2",          False),
        no_enhance        = args.no_enhance,
    )

    run_pipeline(img_path, out_dir, cfg)

    from pipeline.token_tracker import tracker
    tracker.summary()
    print(f"\nDone! Files written to {out_dir}/")


if __name__ == "__main__":
    main()
