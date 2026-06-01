"""
Core pipeline orchestration — callable from CLI (main.py) and API (api/worker.py).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.enhance import enhance_image
from pipeline.vectorize import svg_to_eps


@dataclass
class PipelineConfig:
    call_agent: bool = False
    use_vtracer: bool = False
    use_vtracer_agent: bool = False
    use_esrgan: bool = False
    esrgan_tile: int = 512
    remove_bg: bool = False
    use_sam2: bool = False
    no_enhance: bool = False


def run_pipeline(img_path: Path, out_dir: Path, cfg: PipelineConfig) -> dict:
    """
    Run the full vectorization pipeline.
    Returns {"svg": str, "eps": str}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    if cfg.use_sam2:
        tracer_label = "SAM2 + Inkscape (layered)"
    elif cfg.use_vtracer:
        tracer_label = "vtracer"
    else:
        tracer_label = "Inkscape"

    _esrgan_extra = 1 if cfg.use_esrgan and not cfg.no_enhance else 0
    total_steps = (3
                   + _esrgan_extra
                   + (1 if cfg.remove_bg else 0)
                   + (1 if cfg.use_sam2 else 0)
                   + (1 if cfg.use_vtracer and cfg.use_vtracer_agent else 0))
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
    if cfg.no_enhance:
        source_png = img_path
        next_step("Enhancement skipped (--no-enhance)")
    else:
        enhanced_png = out_dir / f"{stem}_enhanced.png"

        if cfg.use_esrgan:
            next_step("Upscaling with Real-ESRGAN (priority step)...")
            from pipeline.upscale import upscale_image
            shutil.copy2(str(img_path), str(enhanced_png))
            upscale_image(str(enhanced_png), tile=cfg.esrgan_tile)
            source_png = enhanced_png

            if cfg.call_agent:
                next_step("Enhancing Real-ESRGAN output with gpt-image-1...")
                from pipeline.enhance import enhance_with_agent
                enhance_with_agent(str(enhanced_png), str(enhanced_png))
            else:
                next_step("Sharpening Real-ESRGAN output (LANCZOS + sharpen)...")
                enhance_image(str(enhanced_png), str(enhanced_png))
        elif cfg.call_agent:
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
    if cfg.remove_bg:
        nobg_png = out_dir / f"{stem}_nobg.png"
        next_step("Removing background (rembg)...")
        from pipeline.removebg import remove_background
        remove_background(str(source_png), str(nobg_png))
        print(f"      Saved: {nobg_png.name}")
        source_png = nobg_png

    # ── Step 3/4: Segment → layered SVG  OR  single-image trace ──────────────
    svg_path = out_dir / f"{stem}.svg"
    if cfg.use_sam2:
        from pipeline.segment import segment_image, merge_masks_by_color
        from pipeline.vectorize import trace_mask_to_svg
        from pipeline.compose import compose_svg

        masks_dir = out_dir / f"{stem}_masks"
        masks_dir.mkdir(exist_ok=True)

        next_step("Segmenting with SAM 2...")
        segments = segment_image(str(source_png), str(masks_dir))

        next_step("Tracing masks and composing layered SVG...")
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
            if cfg.use_vtracer:
                from pipeline.vtracer_convert import png_to_svg
            else:
                from pipeline.vectorize import png_to_svg
            png_to_svg(str(source_png), str(svg_path))
        except Exception as e:
            if cfg.use_vtracer:
                print(f"      [vtracer] failed: {e}")
                print(f"      [vtracer] falling back to Inkscape...")
                from pipeline.vectorize import png_to_svg as inkscape_png_to_svg
                inkscape_png_to_svg(str(source_png), str(svg_path))
            else:
                raise

        if cfg.use_vtracer and cfg.use_vtracer_agent:
            next_step("Optimizing vtracer params with GPT-4o vision agent...")
            from pipeline.vtracer_agent import optimize
            from pipeline.vtracer_convert import DEFAULT_PARAMS
            optimize(str(source_png), str(svg_path), DEFAULT_PARAMS)

    print(f"      Saved: {svg_path.name}")

    # ── Step 5: SVG → EPS via Inkscape ───────────────────────────────────────
    eps_path = out_dir / f"{stem}.eps"
    next_step("Converting SVG to EPS via Inkscape...")
    svg_to_eps(str(svg_path), str(eps_path))
    print(f"      Saved: {eps_path.name}")

    return {"svg": str(svg_path), "eps": str(eps_path)}
