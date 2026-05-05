import base64
import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

GPT_VISION_MODEL = os.getenv("AGENT_MODEL", "gpt-4o")

ASSESSMENT_PROMPT = """\
Analyze this image for logo vectorization quality. Respond ONLY with valid JSON matching this schema:
{
  "quality_score": <float 0-1>,
  "issues": [<string>, ...],
  "recommended_ops": [<one or more of: "deskew", "denoise", "clahe", "sharpen">],
  "background": <"white" | "transparent" | "dark" | "complex">,
  "edges": <"crisp" | "soft" | "noisy">,
  "notes": <string>
}
Be concise. quality_score 1.0 = perfect for vectorization, 0.0 = unusable."""


class EnhanceGPTInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the image to assess and enhance")
    context: Optional[str] = Field(
        default="",
        description="Optional context e.g. 'company logo, dark background expected'",
    )
    apply_recommendations: bool = Field(
        default=True,
        description="If True, apply the GPT-recommended OpenCV ops automatically after assessment",
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save the enhanced image. Defaults to <name>_enhanced.<ext>.",
    )


def _encode_image(image_path: str) -> tuple[str, str]:
    suffix = Path(image_path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff", ".webp": "image/webp",
    }.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _apply_ops(img_bgr, ops: list) -> tuple:
    applied = []
    for op in ops:
        if op == "deskew":
            try:
                from skimage.transform import rotate
                from skimage.feature import canny
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                edges = canny(gray.astype(float) / 255.0)
                coords = np.column_stack(np.where(edges > 0))
                if len(coords) >= 10:
                    centered = coords - coords.mean(axis=0)
                    cov = np.cov(centered.T)
                    ev, evec = np.linalg.eigh(cov)
                    principal = evec[:, np.argmax(ev)]
                    angle = float(np.degrees(np.arctan2(principal[0], principal[1])))
                    angle = angle - 90 if angle > 45 else (angle + 90 if angle < -45 else angle)
                    if abs(angle) >= 0.5:
                        img_bgr = rotate(img_bgr, angle, resize=False,
                                         preserve_range=True, mode="edge").astype(np.uint8)
                        applied.append(f"deskew({angle:.1f}°)")
            except ImportError:
                pass

        elif op == "denoise":
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
            else:
                img_bgr = cv2.fastNlMeansDenoising(img_bgr, None, 10, 7, 21)
            applied.append("denoise")

        elif op == "clahe":
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(img_bgr.shape) == 3:
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                l_ch, a, b = cv2.split(lab)
                img_bgr = cv2.cvtColor(cv2.merge([clahe_obj.apply(l_ch), a, b]), cv2.COLOR_LAB2BGR)
            else:
                img_bgr = clahe_obj.apply(img_bgr)
            applied.append("clahe")

        elif op == "sharpen":
            gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2.0)
            img_bgr = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)
            applied.append("sharpen")

    return img_bgr, applied


def _enhance_with_gpt(
    image_path: str,
    context: str = "",
    apply_recommendations: bool = True,
    output_path: Optional[str] = None,
) -> str:
    result: dict = {"image_path": image_path}

    if not OPENAI_AVAILABLE:
        result["error"] = "openai package not installed"
        return json.dumps(result)

    if not CV2_AVAILABLE:
        result["error"] = "opencv-python not installed"
        return json.dumps(result)

    try:
        b64, mime = _encode_image(image_path)
    except Exception as exc:
        result["error"] = f"Cannot read image: {exc}"
        return json.dumps(result)

    prompt = ASSESSMENT_PROMPT
    if context:
        prompt += f"\n\nAdditional context: {context}"

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=GPT_VISION_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        assessment = json.loads(raw)
    except Exception as exc:
        result["error"] = f"GPT vision call failed: {exc}"
        return json.dumps(result)

    result["assessment"] = assessment
    result["quality_score"] = assessment.get("quality_score", 0.5)
    result["recommended_ops"] = assessment.get("recommended_ops", [])

    if apply_recommendations and assessment.get("recommended_ops"):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img, applied = _apply_ops(img, assessment["recommended_ops"])
            if not output_path:
                p = Path(image_path)
                output_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")
            os.makedirs(Path(output_path).parent, exist_ok=True)
            cv2.imwrite(output_path, img)
            result["output_path"] = output_path
            result["ops_applied"] = applied

    return json.dumps(result)


enhance_with_gpt_tool = StructuredTool.from_function(
    func=_enhance_with_gpt,
    name="enhance_with_gpt",
    description=(
        "Uses GPT-4o vision to assess image quality and recommend preprocessing steps, "
        "then optionally applies them. Returns JSON with: quality_score (0-1), "
        "assessment (GPT's full analysis), recommended_ops, output_path, ops_applied. "
        "Call this when quality_score from validate_image is below 0.7, or when the image "
        "looks like a low-quality scan. Skip for clean digital exports."
    ),
    args_schema=EnhanceGPTInput,
)
