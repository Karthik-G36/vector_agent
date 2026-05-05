import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import get_all_tools

load_dotenv()

SYSTEM_PROMPT = """\
You are an expert image vectorization agent. Convert raster images (logos, graphics) into
print-ready EPS vector files using the tools available. Prefer semantic/LLM-powered tools
because they produce cleaner output than pixel-level contour tracing.

## Execution order - follow exactly:

### Step 1 - validate_image
Always call this first. Stop if "valid" is false (has "issues"). Warnings are non-fatal; continue.
Note the quality_score for step 3.

### Step 2 - preprocess_image
Decide ops based on the image source:
- Clean digital PNG/JPEG (quality_score >= 0.8) -> ops=[] (skip all ops, pass-through)
- Scanned or rotated document -> ops=["deskew", "denoise", "clahe", "sharpen"]
- Blurry/low-contrast -> ops=["denoise", "clahe", "sharpen"]
Use the output_path returned here for all subsequent tools.

### Step 3 - enhance_with_gpt  (ONLY if quality_score < 0.7)
Skip for clean sources. When called: apply_recommendations=true. Use returned output_path going forward.

### Step 4a - high_fidelity_svg_vectorize  <- PRIMARY when visual fidelity matters
Call high_fidelity_svg_vectorize first when the user asks for high fidelity, maximum detail,
less pixel/detail loss, or faithful visual reproduction for any 2D or 3D image. This route
preserves many tonal layers and tiny details with a larger SVG. Use default fidelity_scale=4
and n_colors=80 unless the user explicitly asks for smaller/larger output.
- If this succeeds (no "error" key): proceed to step 6.
- If this fails: continue to Step 4b.

### Step 4b - semantic_svg_vectorize  <- PRIMARY for editable/semantic 3D text effects/mockups
Call semantic_svg_vectorize first. It analyzes whether the image is a Photoshop text effect,
3D text mockup, poster text, badge text, or a design made from text + simple shapes + shadows.
This reconstructs clean SVG using text, circles, gradients, filters, and layered text instead
of tracing shadows/background gradients into ugly blobs. For script/custom 3D lettering, it
uses source_geometry_3d_trace to preserve the original contours, bevels, and shadows without
inventing new strokes.
- If this succeeds (no "error" key): proceed to step 6.
- If it says the image is not a semantic 3D text-effect candidate: continue to Step 4c.
- If it returns fallback_recommended="trace_source_geometry", skip Step 4c and go directly to Step 5.

### Step 4c - gpt_svg_vectorize  <- PRIMARY for compact semantic logos and flat graphics
Call this for normal logos and flat graphics. GPT-4o understands logo structure and generates
mathematically clean SVG (proper bezier curves, circles, accurate text) - far superior to
pixel tracing. Pass width_mm and height_mm from the user's request and the output_path.
Do NOT call this for script/custom lettering that semantic_svg_vectorize marked as
fallback_recommended="trace_source_geometry"; that would redraw/invent strokes.
- If this succeeds (no "error" key): proceed to step 6.
- If this fails: fall through to step 5.

### Step 5 - fallback only: segment_and_profile + vectorize_to_eps
Use ONLY if high_fidelity_svg_vectorize/semantic_svg_vectorize/gpt_svg_vectorize returned an "error" or the semantic route
reported that the image is not a candidate.
a) Call segment_and_profile on the current image -> read image_type ("flat" or "photo").
b) Call vectorize_to_eps with that image_type, width_mm, height_mm, and output_path.

### Step 6 - apply_pms_color  (ONLY if user provided PMS codes)
Skip entirely if no PMS codes were given. Pass the EPS file from step 4 or 5.

### Step 7 - Return result
Report: final EPS path, SVG path (if generated), method used, conversion backend, and any warnings.

## Rules:
- Never guess file paths - always use paths returned by tool calls.
- If any tool returns an "error" key, log it and continue with the fallback or report it.
- mm -> pts formula: pts = mm x 2.8346 (the %%BoundingBox uses pts, not mm).
"""


def run_vectorization_agent(
    image_path: str,
    output_path: str,
    width_mm: float,
    height_mm: float,
    pms_codes: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    model_name = os.getenv("AGENT_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=0)
    tools = get_all_tools()

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    pms_str = (
        f"PMS colors to apply: {', '.join(pms_codes)}"
        if pms_codes
        else "No PMS colors - keep original colors from the image."
    )
    user_input = (
        f"Vectorize the image at: {image_path}\n"
        f"Save EPS output to: {output_path}\n"
        f"Target dimensions: {width_mm}mm x {height_mm}mm\n"
        f"{pms_str}"
    )

    messages = []

    if verbose:
        for chunk in agent.stream(
            {"messages": [("user", user_input)]},
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            msg_type = type(last).__name__

            if msg_type == "AIMessage":
                if getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        args_preview = ", ".join(
                            f"{k}={repr(v)[:60]}" for k, v in tc["args"].items()
                        )
                        print(f"\n-> [tool_call] {tc['name']}({args_preview})")
                elif last.content:
                    print(f"\n[agent] {last.content}")
            elif msg_type == "ToolMessage":
                preview = (last.content[:300] + "...") if len(last.content) > 300 else last.content
                print(f"  <- [{last.name}] {preview}")

            messages = chunk["messages"]
    else:
        result = agent.invoke({"messages": [("user", user_input)]})
        messages = result["messages"]

    final_answer = messages[-1].content if messages else ""
    return {"output": final_answer, "messages": messages}
