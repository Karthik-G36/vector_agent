"""
Convert PNG → SVG using vtracer (Rust-based color vectorizer).

vtracer is run in an isolated subprocess so that native crashes (segfaults in
the Rust extension) kill only the child process — the main pipeline survives
and can fall back to Inkscape.

Install: pip install vtracer
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


# ── vtracer parameters ────────────────────────────────────────────────────────
_COLOR_PRECISION  = 6
_FILTER_SPECKLE   = 3
_LAYER_DIFFERENCE = 16
_MODE             = "spline"
_HIERARCHICAL     = "stacked"
_CORNER_THRESHOLD = 60
_LENGTH_THRESHOLD = 2.0
_MAX_TRACE_DIM    = 3000

DEFAULT_PARAMS = dict(
    colormode="color",
    hierarchical=_HIERARCHICAL,
    mode=_MODE,
    filter_speckle=_FILTER_SPECKLE,
    color_precision=_COLOR_PRECISION,
    layer_difference=_LAYER_DIFFERENCE,
    corner_threshold=_CORNER_THRESHOLD,
    length_threshold=_LENGTH_THRESHOLD,
)


def _prepare_image(input_png: str) -> tuple[str, str | None]:
    """
    Ensure the image is RGB and within _MAX_TRACE_DIM before tracing.
    Returns (path_to_use, temp_path_or_None).
    """
    with Image.open(input_png) as img:
        orig_mode = img.mode
        w, h = img.size

    scale         = min(1.0, _MAX_TRACE_DIM / max(w, h))
    needs_resize  = scale < 1.0
    needs_convert = orig_mode != "RGB"

    if not (needs_resize or needs_convert):
        return input_png, None

    with Image.open(input_png) as img:
        rgb = img.convert("RGB")
        if needs_resize:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = rgb.resize((new_w, new_h), Image.LANCZOS)
            print(f"      [vtracer] resized {w}x{h} -> {new_w}x{new_h} for tracing")
        else:
            print(f"      [vtracer] converted {orig_mode} -> RGB for tracing")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=tempfile.gettempdir()
    )
    rgb.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name, tmp.name


def _run_vtracer_subprocess(src: str, out: str, params: dict | None = None) -> None:
    """
    Run vtracer inside a child process.
    A native crash in the Rust extension kills only the child — the caller
    receives a RuntimeError instead of the whole process dying silently.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    script = (
        "import vtracer, json\n"
        f"p = json.loads({repr(json.dumps(p))})\n"
        f"vtracer.convert_image_to_svg_py({repr(src)}, {repr(out)}, **p)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "(no output — likely native crash / segfault)"
        raise RuntimeError(
            f"vtracer subprocess failed (exit {result.returncode}):\n{detail}"
        )


def png_to_svg(input_png: str, output_svg: str) -> None:
    """Vectorize a PNG to SVG using vtracer (isolated subprocess)."""
    tmp_path: str | None = None
    try:
        src, tmp_path = _prepare_image(input_png)
        src_abs = str(Path(src).resolve())
        out_abs = str(Path(output_svg).resolve())
        print(f"      [vtracer] tracing {Path(src_abs).name} ...")
        _run_vtracer_subprocess(src_abs, out_abs)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not Path(output_svg).exists():
        raise RuntimeError(f"vtracer finished but SVG not found at {output_svg}")


def trace_with_params(input_png: str, output_svg: str, params: dict) -> None:
    """Re-trace with explicit params dict (used by the vtracer agent optimizer)."""
    tmp_path: str | None = None
    try:
        src, tmp_path = _prepare_image(input_png)
        src_abs = str(Path(src).resolve())
        out_abs = str(Path(output_svg).resolve())
        _run_vtracer_subprocess(src_abs, out_abs, params)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not Path(output_svg).exists():
        raise RuntimeError(f"vtracer finished but SVG not found at {output_svg}")
