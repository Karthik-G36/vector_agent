#!/usr/bin/env python3
"""
Professional vectorization — using `vtracer`.
(Permanently fixes the "oil painting" and hallucination effects by using
industry-standard Rust tracing algorithms to generate perfectly flat SVG paths).

Flow:
    image  →  vtracer (Clean layered SVG)
           →  SVGToEPS converter  →  output.eps

Usage:
    python vectorize_pure.py logo_1.png
"""

import argparse
import os
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import vtracer
except ImportError:
    print("ERROR: vtracer is not installed. Please run: pip install vtracer")
    sys.exit(1)

from tools.gpt_svg_vectorize import svg_to_eps
from utils.conversions import mm_to_pts

def _save(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(
        description="Vectorize an image to EPS using professional vtracer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Input image (PNG, JPG, TIFF, BMP, WebP)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output EPS path. Default: ./output/<name>.eps")
    parser.add_argument("--width-mm", type=float, default=100.0,
                        help="Output width in millimetres")
    parser.add_argument("--height-mm", type=float, default=100.0,
                        help="Output height in millimetres")
    parser.add_argument("--save-svg", action="store_true",
                        help="Keep intermediate .svg file next to the EPS")
    
    # We keep these arguments for backwards compatibility but ignore them
    parser.add_argument("--passes", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--refine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default="gpt-4o", help=argparse.SUPPRESS)
    parser.add_argument("--photo", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    # Resolve paths
    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).exists():
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    if args.output:
        eps_path = str(Path(args.output).resolve())
    else:
        out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        eps_path = str((out_dir / f"{Path(image_path).stem}.eps").resolve())

    svg_path = str(Path(eps_path).with_suffix(".svg"))

    print(f"\nVectorizing: {Path(image_path).name}")
    print(f"Output EPS : {eps_path}")
    print(f"Dimensions : {args.width_mm}mm × {args.height_mm}mm")
    print("Mode       : Professional Tracing (vtracer)\n")

    # --- vtracer Tracing ---
    print("  [Step 1] Running vtracer professional vectorization...")
    try:
        # Avoid passing kwargs that might cause Rust Panics on this specific wheel build
        vtracer.convert_image_to_svg_py(image_path, svg_path)
    except Exception as e:
        print(f"\nERROR: vtracer failed: {e}")
        sys.exit(1)

    print(f"          -> Clean flat SVG generated.")
    print(f"\n  SVG saved  : {Path(svg_path).name} ({Path(svg_path).stat().st_size // 1024} KB)")

    # --- Read SVG ---
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # --- Convert SVG → EPS ---
    print("  [Step 2] Converting mathematical SVG paths -> EPS...")
    width_pts = mm_to_pts(args.width_mm)
    height_pts = mm_to_pts(args.height_mm)

    try:
        backend = svg_to_eps(svg_content, eps_path, width_pts, height_pts)
        print(f"          -> Conversion backend: {backend}")
    except Exception as exc:
        print(f"\nERROR during SVG->EPS conversion: {exc}")
        print("SVG file is still available at:", svg_path)
        sys.exit(1)

    eps_kb = Path(eps_path).stat().st_size / 1024
    print(f"  EPS saved  : {Path(eps_path).name} ({eps_kb:.1f} KB)")

    if not args.save_svg:
        pass

    print(f"\nDone. EPS: {eps_path}\n")

if __name__ == "__main__":
    main()
