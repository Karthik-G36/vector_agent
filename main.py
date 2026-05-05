#!/usr/bin/env python3
"""
vector_agent — raster-to-EPS vectorization pipeline powered by a LangChain ReAct agent.

Usage:
  python main.py logo.png --output logo.eps --width-mm 100 --height-mm 120
  python main.py logo.png --output logo.eps --width-mm 100 --height-mm 120 --pms "PMS 485" "PMS 286"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _check_env():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a raster image to a print-ready EPS vector file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Path to the input image (PNG, JPG, TIFF, BMP)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output EPS file path. Defaults to <input_name>.eps in ./output/")
    parser.add_argument("--width-mm", type=float, default=100.0,
                        help="Target output width in millimetres")
    parser.add_argument("--height-mm", type=float, default=100.0,
                        help="Target output height in millimetres")
    parser.add_argument("--pms", nargs="*", metavar="PMS_CODE",
                        help="Pantone PMS color codes to apply, e.g. --pms 'PMS 485' 'PMS 286'")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress agent verbose output")
    args = parser.parse_args()

    _check_env()

    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).exists():
        print(f"ERROR: Input file not found: {image_path}")
        sys.exit(1)

    if args.output:
        output_path = str(Path(args.output).resolve())
    else:
        out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        output_path = str((out_dir / f"{stem}.eps").resolve())

    print(f"Input : {image_path}")
    print(f"Output: {output_path}")
    print(f"Size  : {args.width_mm}mm × {args.height_mm}mm")
    if args.pms:
        print(f"PMS   : {', '.join(args.pms)}")
    print()

    from agent import run_vectorization_agent
    result = run_vectorization_agent(
        image_path=image_path,
        output_path=output_path,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        pms_codes=args.pms,
        verbose=not args.quiet,
    )

    print("\n--- Agent output ---")
    print(result.get("output", result))

    if Path(output_path).exists():
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"\nEPS file written: {output_path} ({size_kb:.1f} KB)")
    else:
        print("\nWARNING: Expected EPS output not found at", output_path)


if __name__ == "__main__":
    main()
