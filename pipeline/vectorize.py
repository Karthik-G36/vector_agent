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

# Blur radius before tracing: merges JPEG block noise and gradient banding.
# 1.2px is enough to smooth compression artifacts without losing edge detail.
_BLUR_RADIUS = 1.2

# Quantize to flat colors before tracing so Inkscape sees clean palette
# boundaries instead of JPEG gradients — dramatically reduces path count
# and eliminates duplicate near-color layers.
IS_CALL_QUANTIZE = True
_QUANTIZE_COLORS = 16


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
# scans=16         — 16 passes catches all major color layers without over-splitting
# smooth=true      — smooth path curves
# stack=true       — stack overlapping paths (layered color effect)
# remove_background=false — keep the base layer (prevents transparent text)
# speckles=10      — ignore regions < 10px; low enough for detail, high enough to skip noise
# smooth_corners=1.0 — higher = smoother curve corners
# optimize=0.2     — path node reduction
_TRACE_ARGS = "16,true,true,false,10,1,0.2"

# L-inf distance between two fill colors to treat them as the same layer.
# Merges near-duplicate layers created by JPEG compression noise.
_LAYER_MERGE_TOLERANCE = 8

# Minimum character count of a path d= attribute to keep it.
# Shorter paths are single-point noise specks.
_MIN_PATH_D_LEN = 30


def _svg_color_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_distance(c1: tuple, c2: tuple) -> int:
    return max(abs(a - b) for a, b in zip(c1, c2))


def _cleanup_traced_svg(svg_path: Path) -> None:
    """
    Post-process Inkscape-traced SVG:
      1. Drop noise paths (d= too short to be a real shape).
      2. Merge near-duplicate color layers (within _LAYER_MERGE_TOLERANCE).
    """
    text = svg_path.read_text(encoding="utf-8")

    # Parse layers: list of (fill_hex, [path_elements])
    layer_pat = re.compile(
        r'<g\b[^>]+\bid="layer_\d+"[^>]+\bfill="(#[0-9a-fA-F]{6})"[^>]*>(.*?)</g>',
        re.DOTALL,
    )
    path_pat = re.compile(r'<path\b[^>]*/>', re.DOTALL)
    d_pat = re.compile(r'\bd="([^"]*)"')

    layers: list[tuple[tuple, list[str]]] = []
    for m in layer_pat.finditer(text):
        fill_hex = m.group(1)
        body = m.group(2)
        color = _svg_color_to_rgb(fill_hex)
        paths = path_pat.findall(body)
        # Drop noise specks
        kept = [p for p in paths if (dm := d_pat.search(p)) and len(dm.group(1).strip()) >= _MIN_PATH_D_LEN]
        if kept:
            layers.append((color, kept))

    if not layers:
        return  # nothing to rewrite

    # Merge near-duplicate color layers (keep first occurrence's color)
    merged: list[tuple[tuple, list[str]]] = []
    for color, paths in layers:
        for mc, mp in merged:
            if _rgb_distance(color, mc) <= _LAYER_MERGE_TOLERANCE:
                mp.extend(paths)
                break
        else:
            merged.append((color, list(paths)))

    # Rebuild SVG header (preserve original width/height/viewBox)
    vb  = re.search(r'viewBox="([^"]*)"', text)
    w   = re.search(r'<svg\b[^>]+\bwidth="([^"]*)"', text)
    h   = re.search(r'<svg\b[^>]+\bheight="([^"]*)"', text)
    viewbox = vb.group(1) if vb else "0 0 100 100"
    width   = w.group(1)  if w  else "100"
    height  = h.group(1)  if h  else "100"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{viewbox}">',
    ]
    total_paths = 0
    for i, (color, paths) in enumerate(merged):
        fill = "#{:02x}{:02x}{:02x}".format(*color)
        lines.append(f'  <g id="layer_{i}" fill="{fill}" stroke="none">')
        for p in paths:
            lines.append(f"    {p}")
            total_paths += 1
        lines.append("  </g>")
    lines.append("</svg>")

    svg_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"      [cleanup] {len(merged)} layers, {total_paths} paths (merged {len(layers)-len(merged)} duplicate layers)")


def png_to_svg(input_png: str, output_svg: str) -> None:
    """
    Vectorize a PNG to SVG using Inkscape's bitmap tracer.

    Pre-processes with a slight Gaussian blur to suppress gradient banding and
    compression noise before tracing. Post-processes to remove noise paths and
    merge near-duplicate color layers.
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
            _cleanup_traced_svg(out)
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


# ── Binary mask → SVG ────────────────────────────────────────────────────────
# Masks on disk: black(0)=segment, white(255)=background (inverted from SAM 2).
# Potrace traces dark regions — correct for our format.
# OpenCV fallback: findContours on inverted (white=segment) → polygon paths.


def _potrace_exe() -> str | None:
    return shutil.which("potrace")


def _clean_potrace_svg(svg_path: Path) -> None:
    """Strip fill attributes so compose.py can apply region colors via cascade."""
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r'\s+fill="[^"]*"', "", text)
    text = re.sub(r'\s+stroke="[^"]*"', "", text)
    svg_path.write_text(text, encoding="utf-8")


def _trace_with_potrace(mask_png: str, output_svg: str) -> None:
    # Potrace on Windows only accepts PNM/BMP — convert PNG to BMP first.
    bmp_path = Path(mask_png).with_suffix(".bmp")
    try:
        with Image.open(mask_png) as im:
            im.convert("L").save(str(bmp_path), format="BMP")

        result = subprocess.run(
            [
                _potrace_exe(),
                str(bmp_path),
                "--svg",
                "--turdsize", "10",
                "--alphamax", "1.0",
                "--opttolerance", "0.2",
                "-o", output_svg,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        try:
            bmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    out = Path(output_svg)
    if result.returncode != 0 or not out.exists() or not _has_paths(out):
        raise RuntimeError(f"Potrace failed: {result.stderr.strip()[:300]}")
    _clean_potrace_svg(out)


def _trace_with_opencv(mask_png: str, output_svg: str) -> None:
    """
    Contour-based SVG trace — fallback when Potrace is unavailable.
    Reads the inverted mask (black=segment), re-inverts for findContours,
    then converts contours to SVG polygon paths.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "Neither 'potrace' (CLI) nor 'opencv-python' (pip) is available.\n"
            "Install one: choco install potrace  OR  pip install opencv-python"
        )

    mask = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"OpenCV could not read mask: {mask_png}")
    h, w = mask.shape

    # Re-invert: our masks are black=segment; findContours expects white=foreground
    fg = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        raise RuntimeError(f"No contours found in {mask_png}")

    path_elements = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        eps = max(0.5, 0.002 * cv2.arcLength(c, True))
        simplified = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        d = "M " + " L ".join(f"{pt[0]},{pt[1]}" for pt in simplified) + " Z"
        path_elements.append(f'<path d="{d}"/>')

    if not path_elements:
        raise RuntimeError(f"No traceable contours in {mask_png}")

    svg = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        *path_elements,
        "</svg>",
    ])
    Path(output_svg).write_text(svg, encoding="utf-8")


def trace_mask_to_svg(mask_png: str, output_svg: str) -> None:
    """
    Trace a binary mask (black=segment, white=background) to SVG.
    Uses Potrace if on PATH (fast, smooth Bezier curves).
    Falls back to OpenCV contour tracing (polygon paths, no external binary needed).
    """
    if _potrace_exe():
        _trace_with_potrace(mask_png, output_svg)
        print(f"      [potrace] traced {Path(mask_png).name}")
    else:
        _trace_with_opencv(mask_png, output_svg)
        print(f"      [opencv]  traced {Path(mask_png).name}")


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
