"""
Agent-guided vtracer parameter optimization.

After the initial vtracer trace, renders the SVG → PNG via Inkscape and
sends both the enhanced source PNG and the render to GPT-4o vision.
The model scores quality (1-10) and returns better vtracer params.
This loop repeats until the score reaches the quality threshold or
max iterations are exhausted.
"""
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI
from pipeline.token_tracker import tracker

_MAX_ITERATIONS = 3
_QUALITY_THRESHOLD = 7  # stop early once score reaches this


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _render_svg_to_png(svg_path: str) -> str | None:
    """Render SVG → temp PNG via Inkscape. Returns path or None on failure."""
    from pipeline.vectorize import _inkscape
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        r = subprocess.run(
            [_inkscape(), svg_path,
             "--export-type=png",
             f"--export-filename={tmp.name}"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and Path(tmp.name).stat().st_size > 0:
            return tmp.name
    except Exception:
        pass
    Path(tmp.name).unlink(missing_ok=True)
    return None


def _ask_vision(enhanced_png: str, svg_render_png: str, params: dict) -> dict:
    """Send both images to GPT-4o vision for quality scoring + param suggestions."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("AGENT_MODEL", "gpt-4o")

    user_text = f"""You are a vectorization quality expert.

Compare these two images:
• Image 1 — enhanced source PNG (ground truth)
• Image 2 — vtracer SVG output rendered as PNG

Current vtracer parameters:
{json.dumps(params, indent=2)}

Identify quality problems (e.g. missing colors, noisy specks, too many/few color layers,
jagged edges, lost fine detail, over-simplified shapes).

Rate the SVG fidelity 1–10 (10 = perfect match to source).

Then provide optimal parameters to fix the issues. Valid ranges:
- color_precision  : 1–8   (higher = more colors captured; lower = simpler palette)
- filter_speckle   : 1–64  (higher = removes more noise dots; lower = keeps fine detail)
- layer_difference : 1–64  (higher = fewer color layers / simpler; lower = more layers)
- corner_threshold : 1–180 (higher = smoother corners; lower = sharper corners)
- length_threshold : 1.0–10.0 (higher = drops short paths / cleaner; lower = keeps detail)

Respond with ONLY a raw JSON object (no markdown fences):
{{
  "quality_score": <1-10>,
  "issues": "<one-sentence summary of main problems>",
  "color_precision": <int>,
  "filter_speckle": <int>,
  "layer_difference": <int>,
  "corner_threshold": <int>,
  "length_threshold": <float>
}}"""

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(enhanced_png)}",
                    "detail": "high",
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_b64(svg_render_png)}",
                    "detail": "high",
                }},
            ],
        }],
    )
    usage = response.usage
    if usage:
        tracker.record(
            label="vtracer-agent (vision)",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
    return json.loads(response.choices[0].message.content)


_TUNABLE = ("color_precision", "filter_speckle", "layer_difference",
            "corner_threshold", "length_threshold")


def optimize(enhanced_png: str, svg_path: str, initial_params: dict) -> dict:
    """
    Iteratively improve vtracer output quality via GPT-4o vision feedback.

    Each iteration:
      1. Renders the current SVG → PNG.
      2. Asks GPT-4o vision to score quality and suggest new params.
      3. If score < threshold and iterations remain, re-traces with new params.

    Returns the params dict that produced the best-scored result.
    """
    from pipeline.vtracer_convert import trace_with_params

    current_params = dict(initial_params)
    best_score = 0
    best_params = dict(initial_params)

    for i in range(_MAX_ITERATIONS):
        print(f"      [vtracer-agent] iter {i + 1}/{_MAX_ITERATIONS} — rendering SVG for analysis...")

        render_png = _render_svg_to_png(svg_path)
        if render_png is None:
            print("      [vtracer-agent] SVG render failed — stopping optimization")
            break

        try:
            result = _ask_vision(enhanced_png, render_png, current_params)
        except Exception as e:
            print(f"      [vtracer-agent] vision call failed: {e}")
            break
        finally:
            Path(render_png).unlink(missing_ok=True)

        score = result.get("quality_score", 0)
        issues = result.get("issues", "")
        print(f"      [vtracer-agent] score {score}/10 — {issues}")

        if score > best_score:
            best_score = score
            best_params = {k: result[k] for k in _TUNABLE if k in result}

        if score >= _QUALITY_THRESHOLD:
            print(f"      [vtracer-agent] quality target reached ({score}/10)")
            break

        if i < _MAX_ITERATIONS - 1:
            new_params = {k: result[k] for k in _TUNABLE if k in result}
            current_params.update(new_params)
            print(f"      [vtracer-agent] re-tracing with: {new_params}")
            try:
                trace_with_params(enhanced_png, svg_path, current_params)
            except Exception as e:
                print(f"      [vtracer-agent] re-trace failed: {e}")
                break

    print(f"      [vtracer-agent] optimization done — best score {best_score}/10")
    return best_params
