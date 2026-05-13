"""
Convert PNG → SVG and SVG → EPS using Inkscape.

Pre-processing before trace:
  - Slight Gaussian blur (0.8 px) to merge gradient banding and reduce noise.
  - This prevents the trace from picking up micro color variations as separate
    path layers (which cause concentric rings, rough edges, and speckle artifacts).
"""
import base64
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter


# ── Inkscape discovery ─────────────────────────────────────────────────────────

_INKSCAPE_CANDIDATES = [
    # Windows - latest versions
    r"C:\Program Files\Inkscape\bin\inkscape.exe",
    r"C:\Program Files\Inkscape\inkscape.exe",
    r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
    r"C:\Program Files (x86)\Inkscape\inkscape.exe",

    # Windows - user/local installs
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inkscape\bin\inkscape.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inkscape\inkscape.exe"),
    os.path.expandvars(r"%APPDATA%\Inkscape\bin\inkscape.exe"),

    # Chocolatey
    r"C:\ProgramData\chocolatey\bin\inkscape.exe",

    # Scoop
    os.path.expandvars(r"%USERPROFILE%\scoop\apps\inkscape\current\bin\inkscape.exe"),

    # Winget/MS Store related
    os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Inkscape.Inkscape_*"
    ),

    # Linux
    "/usr/bin/inkscape",
    "/usr/local/bin/inkscape",
    "/snap/bin/inkscape",
    "/var/lib/flatpak/exports/bin/org.inkscape.Inkscape",

    # macOS
    "/Applications/Inkscape.app/Contents/MacOS/inkscape",
    "/Applications/Inkscape.app/Contents/Resources/bin/inkscape",
    "/opt/homebrew/bin/inkscape",
    "/usr/local/bin/inkscape",

    # Fallback (PATH lookup)
    "inkscape",
]

def _inkscape() -> str:
    for c in _INKSCAPE_CANDIDATES:
        p = Path(c)
        if p.is_file():
            return str(p)
        if shutil.which(c):
            return c
    raise RuntimeError(
        "Inkscape not found.\n"
        "Install from https://inkscape.org/release/ then restart your terminal."
    )


def _run_inkscape(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_inkscape(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _fwd(p: Path) -> str:
    """Return a forward-slash path string safe for Inkscape action strings."""
    return p.as_posix()


# ── PNG pre-processing ────────────────────────────────────────────────────────

# Blur radius before tracing: merges gradient banding, suppresses compression
# noise, and prevents the tracer from creating separate paths for each minor
# color variation (which causes concentric rings and speckle artifacts).
_BLUR_RADIUS = 0.8

# Set True to reduce the image to _QUANTIZE_COLORS flat colors before tracing.
# Eliminates anti-aliased edge bleed and gradient rings at the cost of
# simplifying the color palette in the SVG output.
IS_CALL_QUANTIZE = False
_QUANTIZE_COLORS = 8


def _image_quantization(img: Image.Image, color_count: int = 8) -> Image.Image:
    # quantize() returns palette mode (P); convert back to RGB for Inkscape
    return img.quantize(colors=color_count, dither=Image.Dither.NONE).convert("RGB")


def _preprocess_for_trace(input_png: str) -> str:
    """
    Apply Gaussian blur (and optional color quantization) before Inkscape tracing.
    Returns path to a temp file that must be deleted by the caller.
    """
    with Image.open(input_png) as img:
        rgb = img.convert("RGB")
        smoothed = rgb.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS))
        result = _image_quantization(smoothed, _QUANTIZE_COLORS) if IS_CALL_QUANTIZE else smoothed

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=tempfile.gettempdir())
    result.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


# ── PNG → SVG ──────────────────────────────────────────────────────────────────

# Trace parameters: scans, smooth, stack, remove_background, speckles,
#                   smooth_corners, optimize
#
# scans=16         — 16 color passes (good detail without over-tracing)
# smooth=true      — smooth path curves
# stack=true       — stack overlapping paths (layered color effect)
# remove_background=false — keep the base layer (prevents transparent text)
# speckles=20      — ignore regions smaller than 20 px (removes noise artifacts)
# smooth_corners=1.0 — higher = smoother curve corners
# optimize=0.2     — path node reduction
_TRACE_ARGS = "16,true,true,false,20,1.0,0.2"


def png_to_svg(input_png: str, output_svg: str) -> None:
    """
    Vectorize a PNG to SVG using Inkscape's bitmap tracer.

    Pre-processes with a slight Gaussian blur to suppress gradient banding and
    compression noise before tracing. Strips <image> elements post-export so
    only vector paths remain.
    """
    inp_preprocessed = _preprocess_for_trace(input_png)
    inp = Path(inp_preprocessed).resolve()
    out = Path(output_svg).resolve()

    try:
        actions = ";".join([
            f"file-open:{_fwd(inp)}",
            "select-all",
            f"object-trace:{_TRACE_ARGS}",
            f"export-filename:{_fwd(out)}",
            "export-type:svg",
            "export-do",
        ])

        result = _run_inkscape(["--actions", actions])

        if out.exists() and _has_paths(out):
            _strip_raster_images(out)
            return

        # Fallback: embed original PNG as base64 in SVG
        print("      [SVG] Inkscape trace produced no paths — embedding PNG as SVG image")
        if result.stderr.strip():
            print(f"      [SVG] Inkscape stderr: {result.stderr.strip()[:200]}")
        _embed_png_in_svg(input_png, output_svg)

    finally:
        try:
            os.unlink(inp_preprocessed)
        except OSError:
            pass


def _has_paths(svg_path: Path) -> bool:
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"<path\b", text))


def _strip_raster_images(svg_path: Path) -> None:
    """Remove embedded <image> elements from SVG, keeping only vector paths."""
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r"<image\b[^>]*/\s*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<image\b[^>]*>.*?</image>", "", text, flags=re.DOTALL)
    svg_path.write_text(text, encoding="utf-8")


def _embed_png_in_svg(input_png: str, output_svg: str) -> None:
    """Embed PNG as base64-encoded image element inside an SVG wrapper."""
    with Image.open(input_png) as img:
        w, h = img.size

    with open(input_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'  <image x="0" y="0" width="{w}" height="{h}" '
        f'xlink:href="data:image/png;base64,{b64}"/>\n'
        "</svg>\n"
    )
    Path(output_svg).write_text(svg, encoding="utf-8")


# ── SVG → EPS ─────────────────────────────────────────────────────────────────

def svg_to_eps(input_svg: str, output_eps: str) -> None:
    """Convert an SVG file to EPS using Inkscape."""
    inp = Path(input_svg).resolve()
    out = Path(output_eps).resolve()

    result = _run_inkscape([
        str(inp),
        "--export-type=eps",
        f"--export-filename={out}",
    ])

    if result.returncode != 0:
        raise RuntimeError(
            f"Inkscape SVG to EPS failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    if not out.exists():
        raise RuntimeError(f"Inkscape finished but EPS file not found at {out}")
