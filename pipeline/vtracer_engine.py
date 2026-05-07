"""
vtracer-based vectorization — replaces stages 3-5 of the classic pipeline.

vtracer is a production-grade Rust library that does bitmap -> layered SVG
far better than OpenCV contour tracing:
  - Proper Bezier spline fitting with tension/corner detection
  - Perceptual colour quantisation with configurable precision
  - Hierarchical stacked layers (correct z-order out of the box)
  - Handles anti-aliasing boundaries cleanly

Settings are selected by the LLM based on image_class:
  logo         - binary mode, fewer layers, simple geometry
  illustration - color mode, more layers, gradient bands preserved
  photo        - color mode, aggressive speckle filter, fewer path points
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import importlib.util
VTRACER_OK = importlib.util.find_spec("vtracer") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Per-class presets  (used as defaults; LLM values override these)
# ──────────────────────────────────────────────────────────────────────────────

# colormode="binary" crashes the Windows vtracer wheel — always use "color".
# For flat logos we compensate with higher filter_speckle and lower
# color_precision, which produces comparably clean output without the crash.
_PRESETS: dict[str, dict[str, Any]] = {
    "logo": dict(
        colormode="color",
        filter_speckle=8,
        color_precision=6,
        path_precision=5,
        layer_difference=16,
    ),
    "illustration": dict(
        colormode="color",
        filter_speckle=4,
        color_precision=8,
        path_precision=6,
        layer_difference=8,
    ),
    "photo": dict(
        colormode="color",
        filter_speckle=16,
        color_precision=6,
        path_precision=3,
        layer_difference=16,
    ),
}

_FALLBACK_PRESET = _PRESETS["illustration"]

# vtracer panics on the Windows wheel when the image is too large.
# 900 px is the safe upper bound found by testing.
_VTRACER_MAX_DIM = 900


def is_available() -> bool:
    return VTRACER_OK


def params_from_analysis(analysis: dict) -> dict:
    """
    Build vtracer kwargs from the LLM analysis dict.

    Priority:
      1. analysis["vtracer"]  — LLM-chosen values (most specific)
      2. class preset         — table-based defaults for image_class
      3. _FALLBACK_PRESET     — "illustration" if class is unknown
    """
    image_class = str(analysis.get("image_class", "illustration")).lower()
    base = dict(_PRESETS.get(image_class, _FALLBACK_PRESET))

    # Merge in any LLM-specific overrides
    llm_vt = analysis.get("vtracer", {})
    if isinstance(llm_vt, dict):
        for key in ("colormode", "filter_speckle", "color_precision",
                    "path_precision", "layer_difference"):
            if key in llm_vt:
                base[key] = llm_vt[key]

    # colormode="binary" crashes the Windows vtracer wheel regardless of image.
    # Force color mode and let filter_speckle/color_precision control simplicity.
    base["colormode"] = "color"

    return base


def _resize_for_vtracer(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= _VTRACER_MAX_DIM:
        return img_bgr
    scale = _VTRACER_MAX_DIM / max(h, w)
    return cv2.resize(
        img_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helper — isolates Rust panics from the main process
# ──────────────────────────────────────────────────────────────────────────────

# Inline script executed in a child process.  sys.argv layout:
#   [0] = "-c"  [1] = in_path  [2] = out_path  [3] = JSON kwargs (optional)
_CHILD_SCRIPT = (
    "import sys, json, vtracer;"
    "kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {};"
    "vtracer.convert_image_to_svg_py(sys.argv[1], sys.argv[2], **kw)"
)


def _run_in_subprocess(in_path: str, out_path: str, **kwargs: Any) -> bool:
    """
    Run vtracer in an isolated child process.

    Returns True on success, False if the child crashes or times out.

    Rationale: the Windows vtracer wheel can call os::abort() (Rust panic)
    which kills the entire Python process — no exception is raised, no
    try/except helps.  Running in a subprocess means only the child dies;
    the main pipeline continues and can retry with different settings.
    """
    import json
    import subprocess
    import sys

    cmd = [sys.executable, "-c", _CHILD_SCRIPT, in_path, out_path]
    if kwargs:
        cmd.append(json.dumps(kwargs))

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Public trace function
# ──────────────────────────────────────────────────────────────────────────────

def trace(img_bgr: np.ndarray, **kwargs: Any) -> str:
    """
    Vectorize a BGR numpy array and return an SVG string.

    Attempt order:
      1. Subprocess with LLM-selected kwargs
      2. Subprocess with no kwargs (vtracer defaults) — safe on all platforms
      3. RuntimeError if both fail

    The subprocess isolation ensures a Rust panic in vtracer never kills
    the main pipeline process.
    """
    if not VTRACER_OK:
        raise RuntimeError(
            "vtracer is not installed or failed to import. "
            "Run:  pip install vtracer"
        )

    img_small = _resize_for_vtracer(img_bgr)
    orig_h, orig_w = img_bgr.shape[:2]
    small_h, small_w = img_small.shape[:2]

    tmp_in  = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    tmp_in.close()
    tmp_out.close()

    try:
        cv2.imwrite(tmp_in.name, img_small)

        # Attempt 1: LLM-tuned params
        if kwargs and _run_in_subprocess(tmp_in.name, tmp_out.name, **kwargs):
            print(f"         vtracer params used: {kwargs}")
        else:
            if kwargs:
                print("         vtracer kwargs caused a crash — retrying with defaults")
            # Attempt 2: vtracer default settings (always stable)
            if not _run_in_subprocess(tmp_in.name, tmp_out.name):
                raise RuntimeError("vtracer failed with both custom and default settings")
            print("         vtracer params used: defaults")

        if not Path(tmp_out.name).exists() or Path(tmp_out.name).stat().st_size == 0:
            raise RuntimeError("vtracer produced no output")

        with open(tmp_out.name, "r", encoding="utf-8") as fh:
            svg = fh.read()
    finally:
        Path(tmp_in.name).unlink(missing_ok=True)
        Path(tmp_out.name).unlink(missing_ok=True)

    # Restore original viewBox so EPS exports at the right dimensions
    svg = svg.replace(
        f'viewBox="0 0 {small_w} {small_h}"',
        f'viewBox="0 0 {orig_w} {orig_h}"',
    )
    return svg
