#!/usr/bin/env python3
"""
Vector Agent — Image to Vector Pipeline
Usage: python main.py <image_path> [options]

Accepts: PNG, JPG, JPEG, WEBP, BMP, TIFF, TIF, GIF
Pipeline:
  1. Enhance with LANCZOS upscale + UnsharpMask  (or gpt-image-1 when --call-agent)
  2. Upscale with Real-ESRGAN (--use-realesrgan / USE_REALESRGAN=true)
  3. Remove background (--remove-bg / REMOVE_BG=true)
  4. Trace PNG to SVG via vtracer (--use-vtracer / USE_VTRACER=true) or Inkscape
  5. Convert SVG to EPS via Inkscape

CLI flags override the matching .env variable for that run.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pipeline.enhance import enhance_image
from pipeline.vectorize import svg_to_eps

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
    out_dir.mkdir(exist_ok=True)
    stem = img_path.stem

    call_agent  = _resolve(args.call_agent,      "CALL_AGENT",     False)
    use_vtracer = _resolve(args.use_vtracer,     "USE_VTRACER",    False)
    use_esrgan  = _resolve(args.use_realesrgan,  "USE_REALESRGAN", False)
    remove_bg   = _resolve(args.remove_bg,       "REMOVE_BG",      False)
    esrgan_tile = args.realesrgan_tile if args.realesrgan_tile is not None else int(os.getenv("REALESRGAN_TILE", "512"))

    # Pick PNG→SVG tracer
    if use_vtracer:
        from pipeline.vtracer_convert import png_to_svg
        tracer_label = "vtracer"
    else:
        from pipeline.vectorize import png_to_svg
        tracer_label = "Inkscape"

    total_steps = (3
                   + (1 if use_esrgan and not args.no_enhance else 0)
                   + (1 if remove_bg else 0))
    step = 0

    print(f"\n=== Vector Agent ===")
    print(f"Input : {img_path}")
    print(f"Output: {out_dir}/")
    print(f"Tracer: {tracer_label}")

    def next_step(label: str) -> None:
        nonlocal step
        step += 1
        print(f"\n[{step}/{total_steps}] {label}")

    # ── Step 1: Enhance ───────────────────────────────────────────────────────
    if args.no_enhance:
        source_png = img_path
        next_step("Enhancement skipped (--no-enhance)")
    else:
        enhanced_png = out_dir / f"{stem}_enhanced.png"
        if call_agent:
            next_step("Enhancing with gpt-image-1 + LANCZOS 4x + sharpen...")
            from pipeline.enhance import enhance_with_agent
            enhance_with_agent(str(img_path), str(enhanced_png))
        else:
            next_step("Enhancing (LANCZOS 4x + sharpen)...")
            enhance_image(str(img_path), str(enhanced_png))
        print(f"      Saved: {enhanced_png.name}")
        source_png = enhanced_png

    # ── Step 2: Real-ESRGAN upscale (optional) ────────────────────────────────
    if use_esrgan and not args.no_enhance:
        next_step("Upscaling with Real-ESRGAN (overwriting enhanced file)...")
        from pipeline.upscale import upscale_image
        upscale_image(str(source_png), tile=esrgan_tile)

    # ── Step 3: Remove background (optional) ─────────────────────────────────
    if remove_bg:
        nobg_png = out_dir / f"{stem}_nobg.png"
        next_step("Removing background (rembg)...")
        from pipeline.removebg import remove_background
        remove_background(str(source_png), str(nobg_png))
        print(f"      Saved: {nobg_png.name}")
        source_png = nobg_png

    # ── Step 4: PNG → SVG ─────────────────────────────────────────────────────
    svg_path = out_dir / f"{stem}.svg"
    next_step(f"Tracing PNG to SVG via {tracer_label}...")
    try:
        png_to_svg(str(source_png), str(svg_path))
    except Exception as e:
        if use_vtracer:
            print(f"      [vtracer] failed: {e}")
            print(f"      [vtracer] falling back to Inkscape...")
            from pipeline.vectorize import png_to_svg as inkscape_png_to_svg
            inkscape_png_to_svg(str(source_png), str(svg_path))
        else:
            raise
    print(f"      Saved: {svg_path.name}")

    # ── Step 5: SVG → EPS via Inkscape ────────────────────────────────────────
    eps_path = out_dir / f"{stem}.eps"
    next_step("Converting SVG to EPS via Inkscape...")
    svg_to_eps(str(svg_path), str(eps_path))
    print(f"      Saved: {eps_path.name}")

    print(f"\nDone! Files written to {out_dir}/")


if __name__ == "__main__":
    main()
