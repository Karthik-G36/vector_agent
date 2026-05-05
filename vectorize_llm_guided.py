#!/usr/bin/env python3
"""
AI-Guided Color Clustering (The Final Fix).
Uses the LLM as an intelligent Art Director to determine the optimal number
of solid colors in the logo. This mathematically restricts OpenCV K-Means,
forcing anti-aliased edge pixels to snap to solid colors, completely
eliminating the "oil painting" effect.

Usage:
    python vectorize_llm_guided.py logo.png
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package is not installed. Please run: pip install openai")
    sys.exit(1)

from tools.gpt_svg_vectorize import svg_to_eps
from tools.vectorize_eps import _prepare_image, _get_regions_kmeans
from utils.conversions import mm_to_pts

def get_optimal_cluster_count(image_path: str) -> int:
    """Uses GPT-4o to dynamically determine the exact number of solid colors."""
    print("  [Step 1] Asking LLM to analyze design colors...")
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    client = OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this logo. How many distinct, solid base colors are actually "
                                "in this design (including the background)?\n\n"
                                "CRITICAL INSTRUCTIONS:\n"
                                "1. Ignore ALL blurry, dark, or anti-aliased edge pixels.\n"
                                "2. Ignore any compression artifacts or shadows.\n"
                                "3. Just count the core, flat colors (e.g., Red shape, Grey shape, Black background = 3).\n"
                                "4. Return ONLY a JSON object with the integer count, like this: {\"k\": 3}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=100,
        )
        
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
        k = int(data.get("k", 3))
        
        # Safety bound (logos rarely need > 5 solid colors)
        k = max(2, min(k, 8))
        print(f"          -> LLM determined optimal cluster count: k = {k}")
        return k
        
    except Exception as e:
        print(f"          -> LLM analysis failed ({e}). Falling back to k = 3.")
        return 3

def _approx_contour_to_svg(cnt: np.ndarray, tension: float = 0.15) -> str:
    """Convert an approximated OpenCV polygon to an SVG path."""
    pts = cnt.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 3:
        return ""

    x0, y0 = pts[0]
    path_parts = [f"M {x0:.3f} {y0:.3f}"]

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        cp1 = p1 + tension * (p2 - p0) / 3.0
        cp2 = p2 - tension * (p3 - p1) / 3.0

        path_parts.append(
            f"C {cp1[0]:.3f} {cp1[1]:.3f}, "
            f"{cp2[0]:.3f} {cp2[1]:.3f}, "
            f"{p2[0]:.3f} {p2[1]:.3f}"
        )

    path_parts.append("Z")
    return " ".join(path_parts)


def vectorize_to_svg(image_path: str, k: int) -> str:
    print("  [Step 2] Mathematically mapping exact colors...")
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = _prepare_image(orig_img, target_min_dim=1000)
    h, w = img.shape[:2]

    # Slight blur to merge minor compression noise
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # K-Means strictly bounded to the LLM's cluster count
    # This mathematically prevents intermediate anti-aliasing layers!
    regions = _get_regions_kmeans(img_blur, n_clusters=k)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
    ]
    
    bg_color = img[0, 0]
    bg_hex = f"#{bg_color[2]:02x}{bg_color[1]:02x}{bg_color[0]:02x}"
    svg_lines.append(f'  <rect width="{w}" height="{h}" fill="{bg_hex}" />')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for mask, bgr_color in regions:
        # Smooth boundaries slightly
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > 50]
        
        if valid:
            hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
            if hex_color == bg_hex:
                continue
                
            path_d_list = []
            for cnt in valid:
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                path_d = _approx_contour_to_svg(approx, tension=0.15)
                if path_d:
                    path_d_list.append(path_d)
                    
            if path_d_list:
                full_d = " ".join(path_d_list)
                svg_lines.append(f'  <path fill="{hex_color}" fill-rule="evenodd" d="{full_d}" />')

    svg_lines.append('</svg>')
    print("          -> Crisp, restricted SVG generated.")
    return "\n".join(svg_lines)


def _save(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(
        description="Vectorize an image using LLM-Guided Color Clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Input image")
    parser.add_argument("--output", "-o", default=None, help="Output EPS path")
    parser.add_argument("--width-mm", type=float, default=100.0)
    parser.add_argument("--height-mm", type=float, default=100.0)
    parser.add_argument("--save-svg", action="store_true")
    
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())
    if not Path(image_path).exists():
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    if args.output:
        eps_path = str(Path(args.output).resolve())
    else:
        out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        eps_path = str((out_dir / f"{Path(image_path).stem}_llm.eps").resolve())

    svg_path = str(Path(eps_path).with_suffix(".svg"))

    print(f"\nVectorizing: {Path(image_path).name}")
    print(f"Output EPS : {eps_path}")
    print("Mode       : LLM-Guided Color Clustering\n")

    # Step 1
    k = get_optimal_cluster_count(image_path)

    # Step 2
    svg = vectorize_to_svg(image_path, k)

    # Save SVG
    _save(svg_path, svg)
    print(f"\n  SVG saved  : {Path(svg_path).name} ({Path(svg_path).stat().st_size // 1024} KB)")

    # Step 3
    print("  [Step 3] Converting mathematical SVG paths -> EPS...")
    width_pts = mm_to_pts(args.width_mm)
    height_pts = mm_to_pts(args.height_mm)

    try:
        backend = svg_to_eps(svg, eps_path, width_pts, height_pts)
        print(f"          -> Conversion backend: {backend}")
    except Exception as exc:
        print(f"\nERROR during SVG->EPS conversion: {exc}")
        sys.exit(1)

    eps_kb = Path(eps_path).stat().st_size / 1024
    print(f"  EPS saved  : {Path(eps_path).name} ({eps_kb:.1f} KB)")
    print(f"\nDone. EPS: {eps_path}\n")

if __name__ == "__main__":
    main()
