#!/usr/bin/env python3
"""
Advanced vectorisation pipeline.

Usage:
    python vectorize_advanced.py logo.png
    python vectorize_advanced.py logo.png -o logo.eps --width-mm 120 --height-mm 80
"""

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 stdout so Unicode characters in print() never crash on Windows
# terminals that default to CP1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.orchestrator import vectorize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced gradient-preserving logo vectorisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Input raster image (PNG / JPG / BMP)")
    parser.add_argument("--output", "-o", default=None, help="Output EPS path")
    parser.add_argument("--width-mm",  type=float, default=100.0, help="Canvas width in mm")
    parser.add_argument("--height-mm", type=float, default=100.0, help="Canvas height in mm")
    parser.add_argument("--no-svg", action="store_true", help="Do not save intermediate SVG")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).exists():
        print(f"ERROR: file not found: {image_path}")
        sys.exit(1)

    if args.output:
        eps_path = str(Path(args.output).resolve())
    else:
        out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        eps_path = str((out_dir / f"{Path(image_path).stem}_advanced.eps").resolve())

    print("\n=== Advanced Vectorisation Pipeline v2 ===")
    print(f"  Input  : {Path(image_path).name}")
    print(f"  Output : {eps_path}")
    print(f"  Canvas : {args.width_mm} x {args.height_mm} mm\n")

    try:
        result = vectorize(
            image_path=image_path,
            output_eps=eps_path,
            width_mm=args.width_mm,
            height_mm=args.height_mm,
            save_svg=not args.no_svg,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print("\n------------------------------------------")
    print(f"  SVG      : {result.get('svg_path', 'not saved')}")
    print(f"  EPS      : {result['eps_path']}")
    print(f"  Backend  : {result.get('backend', '?')}")
    print(f"  Regions  : {result.get('n_regions', '?')}  "
          f"({result.get('n_gradients', 0)} gradient, "
          f"{result.get('n_effects', 0)} filter effect)")
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()
