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
    out_dir.mkdir(exist_ok=True)
    stem = img_path.stem

    call_agent        = _resolve(args.call_agent,         "CALL_AGENT",         False)
    use_vtracer       = _resolve(args.use_vtracer,        "USE_VTRACER",        False)
    use_vtracer_agent = _resolve(args.use_vtracer_agent,  "USE_VTRACER_AGENT",  False)
    use_esrgan        = _resolve(args.use_realesrgan,     "USE_REALESRGAN",     False)
    remove_bg         = _resolve(args.remove_bg,          "REMOVE_BG",          False)
    use_sam2          = _resolve(args.use_sam2,           "USE_SAM2",           False)
    esrgan_tile = args.realesrgan_tile if args.realesrgan_tile is not None else int(os.getenv("REALESRGAN_TILE", "512"))

    # Pick PNG→SVG tracer (not used when SAM2 is on — masks are traced individually)
    if use_sam2:
        tracer_label = "SAM2 + Inkscape (layered)"
    elif use_vtracer:
        from pipeline.vtracer_convert import png_to_svg
        tracer_label = "vtracer"
    else:
        from pipeline.vectorize import png_to_svg
        tracer_label = "Inkscape"

    # When ESRGAN is on: 2 sub-steps (ESRGAN + enhance/sharpen) replace the single enhance step
    _esrgan_extra = 1 if use_esrgan and not args.no_enhance else 0
    total_steps = (3
                   + _esrgan_extra
                   + (1 if remove_bg else 0)
                   + (1 if use_sam2 else 0)
                   + (1 if use_vtracer and use_vtracer_agent else 0))
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

        if use_esrgan:
            # Real-ESRGAN runs first — upscale the original, then enhance
            next_step("Upscaling with Real-ESRGAN (priority step)...")
            from pipeline.upscale import upscale_image
            # ESRGAN overwrites in-place, so copy original to enhanced_png first
            import shutil
            shutil.copy2(str(img_path), str(enhanced_png))
            upscale_image(str(enhanced_png), tile=esrgan_tile)
            source_png = enhanced_png

            if call_agent:
                next_step("Enhancing Real-ESRGAN output with gpt-image-1...")
                from pipeline.enhance import enhance_with_agent
                enhance_with_agent(str(enhanced_png), str(enhanced_png))
            else:
                next_step("Sharpening Real-ESRGAN output (LANCZOS + sharpen)...")
                enhance_image(str(enhanced_png), str(enhanced_png))
        elif call_agent:
            next_step("Enhancing with gpt-image-1 + LANCZOS 4x + sharpen...")
            from pipeline.enhance import enhance_with_agent
            enhance_with_agent(str(img_path), str(enhanced_png))
            source_png = enhanced_png
        else:
            next_step("Enhancing (LANCZOS 4x + sharpen)...")
            enhance_image(str(img_path), str(enhanced_png))
            source_png = enhanced_png

        print(f"      Saved: {enhanced_png.name}")

    # ── Step 2: Remove background (optional) ─────────────────────────────────
    if remove_bg:
        nobg_png = out_dir / f"{stem}_nobg.png"
        next_step("Removing background (rembg)...")
        from pipeline.removebg import remove_background
        remove_background(str(source_png), str(nobg_png))
        print(f"      Saved: {nobg_png.name}")
        source_png = nobg_png

    # ── Step 4: Segment → layered SVG  OR  single-image trace ───────────────
    svg_path = out_dir / f"{stem}.svg"
    if use_sam2:
        from pipeline.segment import segment_image, merge_masks_by_color
        from pipeline.vectorize import trace_mask_to_svg
        from pipeline.compose import compose_svg

        masks_dir = out_dir / f"{stem}_masks"
        masks_dir.mkdir(exist_ok=True)

        next_step("Segmenting with SAM 2...")
        segments = segment_image(str(source_png), str(masks_dir))

        next_step("Tracing masks and composing layered SVG...")
        # Merge same-color masks before tracing so letter forms aren't torn apart.
        # SAM2 over-segments logos into many same-color fragments; merging produces
        # one clean combined mask per color group → one trace per color layer.
        merged = merge_masks_by_color(segments, masks_dir)

        svg_segments = []
        for seg in merged:
            traced_svg_path = masks_dir / (Path(seg["mask_path"]).stem + "_traced.svg")
            try:
                trace_mask_to_svg(seg["mask_path"], str(traced_svg_path))
                svg_segments.append({
                    "traced_svg": traced_svg_path.read_text(encoding="utf-8"),
                    "color": seg["color"],
                })
            except Exception as e:
                print(f"      [warn] skipping {Path(seg['mask_path']).name}: {e}")

        if not svg_segments:
            raise RuntimeError("SAM 2 segmentation produced no traceable masks.")

        compose_svg(svg_segments, str(svg_path))
    else:
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

        if use_vtracer and use_vtracer_agent:
            next_step("Optimizing vtracer params with GPT-4o vision agent...")
            from pipeline.vtracer_agent import optimize
            from pipeline.vtracer_convert import DEFAULT_PARAMS
            optimize(str(source_png), str(svg_path), DEFAULT_PARAMS)

    print(f"      Saved: {svg_path.name}")

    # ── Step 5: SVG → EPS via Inkscape ────────────────────────────────────────
    eps_path = out_dir / f"{stem}.eps"
    next_step("Converting SVG to EPS via Inkscape...")
    svg_to_eps(str(svg_path), str(eps_path))
    print(f"      Saved: {eps_path.name}")

    print(f"\nDone! Files written to {out_dir}/")


if __name__ == "__main__":
    main()
