"""
Advanced GPT-4o structured image analysis.

Asks the model a single question that returns a rich JSON object describing
every visual layer in the logo — gradients, shadows, 3D surfaces, text,
background.  The pipeline uses this to:

  • Set the right quantisation depth (total_k)
  • Route complex 3D logos through gradient-aware processing
  • Generate better SVG directly when the classical path is insufficient

The prompt is carefully engineered to avoid hallucination:
  — low temperature (0), json_object format
  — explicit instruction to count gradient *zones* not just flat colours
  — explicit instruction to ignore JPEG compression artefacts
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path


_PROMPT = """\
You are an expert visual analyst for a professional logo vectorisation system.
Analyse this image and return a single JSON object — no prose, no markdown.

Required schema:
{
  "image_type": "flat | gradient | 3d | complex",
  "image_class": "logo | illustration | photo",
  "has_gradients":  <true|false>,
  "has_shadows":    <true|false>,
  "has_glow":       <true|false>,
  "has_bevel":      <true|false>,
  "has_text":       <true|false>,
  "text_content":   ["string", ...],
  "background_color": "#rrggbb",
  "color_palette":  ["#hex", ...],
  "n_solid_colors": <int>,
  "n_gradient_regions": <int>,
  "layers": [
    {
      "name": "descriptive label",
      "type": "solid | linear_gradient | radial_gradient | glow | drop_shadow | inner_shadow | bevel | emboss | text",
      "colors": ["#hex1", "#hex2"],
      "gradient_direction": "top-to-bottom | bottom-to-top | left-to-right | right-to-left | diagonal-tl-br | diagonal-tr-bl | radial-center-out | radial-edge-in",
      "z_order": "background | lower | middle | upper | foreground",
      "opacity": <0.0–1.0>,
      "description": "one-sentence description"
    }
  ],
  "total_k": <int>,
  "vtracer": {
    "colormode":       "color",
    "filter_speckle":  <int 4-16>,
    "color_precision": <int 4-8>,
    "path_precision":  <int 3-8>,
    "layer_difference":<int 8-32>
  }
}

IMAGE CLASS RULES:
  logo        — simple icon / wordmark with flat or binary-clean colours; 2-5 colours total;
                no photographic detail. Use colormode=binary for pure flat logos.
  illustration — artwork with gradients, 3D effects, bevels, shadows, or many colours (6+).
                Always use colormode=color. This class covers most brand logos with 3D effects.
  photo       — photographic or highly detailed raster image.

VTRACER SETTINGS GUIDE (tune based on image_class; colormode is ALWAYS "color"):
  logo:         colormode=color, filter_speckle=8,  color_precision=6, path_precision=5, layer_difference=16
  illustration: colormode=color, filter_speckle=4,  color_precision=8, path_precision=6, layer_difference=8
  photo:        colormode=color, filter_speckle=16, color_precision=6, path_precision=3, layer_difference=16

  Adjust further:
    • More gradient bands / bevel shading → lower layer_difference (6-10)
    • Noisy or JPEG-compressed input     → higher filter_speckle (8-12)
    • Needs smooth curves                → higher path_precision (6-8)
    • Simple geometry / few colours      → lower color_precision (4-6)

COUNTING RULES (critical for total_k):
  • Count every distinct visual zone — each gradient face of a 3D object is a zone.
  • Count background as 1 zone.
  • total_k = n_solid_colors + (n_gradient_regions x 2).
  • Minimum 3, maximum 18.
  • Ignore JPEG compression noise, anti-aliasing fringe, and minor shadow halos.

GRADIENT IDENTIFICATION:
  • A metallic sheen, 3D bevel, orb highlight, or depth shadow is a gradient zone.
  • Radial gradient: bright spot at a point that fades outward.
  • Linear gradient: colour drifts in one direction across a shape.
"""


def analyze_image(image_path: str) -> dict:
    """
    Call GPT-4o with the logo image and return the analysis dict.

    Falls back to an empty dict on any error so the pipeline can continue
    with safe defaults.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {}

    try:
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "bmp": "image/bmp", "webp": "image/webp"}.get(suffix, "image/png")

        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("AGENT_MODEL", "gpt-4o"),
            max_tokens=1400,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        )

        data = json.loads(resp.choices[0].message.content)

        # Clamp total_k to a safe range
        n_solid = int(data.get("n_solid_colors", 3))
        n_grad  = int(data.get("n_gradient_regions", 0))
        raw_k   = int(data.get("total_k", n_solid + n_grad * 2))
        data["total_k"] = max(3, min(raw_k, 18))

        vt = data.get("vtracer", {})
        print(
            f"    [LLM] type={data.get('image_type','?')}  "
            f"class={data.get('image_class','?')}  "
            f"k={data['total_k']}  "
            f"gradients={data.get('has_gradients')}  "
            f"shadows={data.get('has_shadows')}  "
            f"vtracer=colormode:{vt.get('colormode','?')} "
            f"speckle:{vt.get('filter_speckle','?')} "
            f"precision:{vt.get('color_precision','?')}"
        )
        return data

    except Exception as exc:
        print(f"    [LLM] analysis failed ({exc}); using conservative defaults.")
        return {}
