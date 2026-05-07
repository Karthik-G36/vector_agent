"""
High-fidelity SVG → EPS export with a three-level fallback chain.

Priority
────────
1. Inkscape CLI  (inkscape --export-ps-level=3)
   Best quality. Handles SVG filters, clip-paths, gradients, patterns
   natively; produces PostScript Level 3 which supports smooth shading
   (shfill operator) for true gradient rendering in EPS.

2. CairoSVG  (cairosvg.svg2ps)
   Good quality. Respects linearGradient / radialGradient and most
   filter primitives.  Installed as a Python package.

3. Built-in SVGToEPS parser  (tools/gpt_svg_vectorize.SVGToEPS)
   Fallback of last resort.  Pure-Python; handles basic paths, rects,
   circles and text but cannot render SVG filters.

The function writes the EPS file to ``eps_path`` and returns the name of
the backend that succeeded.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# One-time availability probes (run at import, not on every call)
# ──────────────────────────────────────────────────────────────────────────────

def _inkscape_available() -> bool:
    return shutil.which("inkscape") is not None


def _cairo_available() -> bool:
    """
    Check whether the Cairo system library is present.

    CairoSVG is installed as a Python package but depends on the native
    libcairo shared library (libcairo-2.dll on Windows, libcairo.so on Linux).
    If the DLL is missing, cairocffi raises OSError at import time.
    We probe once here so the exporter never attempts CairoSVG when it will
    always fail, eliminating the noisy failure message from the pipeline log.
    """
    try:
        import cairocffi  # noqa: F401 — probe only
        return True
    except (ImportError, OSError):
        return False


_CAIRO_OK = _cairo_available()


# ──────────────────────────────────────────────────────────────────────────────
# Backend implementations
# ──────────────────────────────────────────────────────────────────────────────

def _via_inkscape(svg_path: str, eps_path: str) -> bool:
    try:
        proc = subprocess.run(
            [
                "inkscape", svg_path,
                f"--export-filename={eps_path}",
                "--export-ps-level=3",
                "--export-area-drawing",
            ],
            capture_output=True, text=True, timeout=90,
        )
        return proc.returncode == 0 and Path(eps_path).exists() and Path(eps_path).stat().st_size > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _via_cairosvg_file(svg_path: str, eps_path: str, w_pts: float, h_pts: float) -> bool:
    try:
        import cairosvg
        cairosvg.svg2ps(
            url=svg_path,
            write_to=eps_path,
            output_width=w_pts,
            output_height=h_pts,
        )
        return Path(eps_path).exists() and Path(eps_path).stat().st_size > 0
    except Exception:
        return False


def _via_cairosvg_bytes(svg_str: str, eps_path: str, w_pts: float, h_pts: float) -> bool:
    try:
        import cairosvg
        cairosvg.svg2ps(
            bytestring=svg_str.encode("utf-8"),
            write_to=eps_path,
            output_width=w_pts,
            output_height=h_pts,
        )
        return Path(eps_path).exists() and Path(eps_path).stat().st_size > 0
    except Exception:
        return False


def _via_builtin(svg_str: str, eps_path: str, w_pts: float, h_pts: float) -> bool:
    try:
        from tools.gpt_svg_vectorize import SVGToEPS
        eps_content = SVGToEPS(svg_str, w_pts, h_pts).convert()
        with open(eps_path, "w", encoding="utf-8") as fh:
            fh.write(eps_content)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def export_to_eps(
    svg_str: str,
    eps_path: str,
    width_pts: float,
    height_pts: float,
) -> str:
    """
    Convert *svg_str* to an EPS file at *eps_path*.

    Returns the name of the backend that succeeded:
    ``"inkscape"`` | ``"cairosvg"`` | ``"builtin"``

    Raises RuntimeError if every backend fails.
    """
    Path(eps_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_svg: str = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".svg", mode="w", encoding="utf-8", delete=False
        ) as fh:
            fh.write(svg_str)
            tmp_svg = fh.name

        if _inkscape_available():
            if _via_inkscape(tmp_svg, eps_path):
                return "inkscape"
            print("    [EPS] Inkscape failed — trying next backend...")

        if _CAIRO_OK:
            if _via_cairosvg_file(tmp_svg, eps_path, width_pts, height_pts):
                return "cairosvg"
            if _via_cairosvg_bytes(svg_str, eps_path, width_pts, height_pts):
                return "cairosvg"
            print("    [EPS] CairoSVG failed — using built-in SVG parser...")

        if _via_builtin(svg_str, eps_path, width_pts, height_pts):
            return "builtin"

        raise RuntimeError("All EPS export backends failed.")
    finally:
        if tmp_svg:
            try:
                os.unlink(tmp_svg)
            except OSError:
                pass
