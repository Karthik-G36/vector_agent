import json
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# Representative PMS → (R, G, B) and CMYK approximations
# Format: "PMS CODE": ((R, G, B), (C, M, Y, K))  values 0-255 for RGB, 0-100 for CMYK
PMS_DB: dict = {
    "PMS 100":   ((244, 234,  68), ( 0,  0, 80,  0)),
    "PMS 109":   ((255, 209,   0), ( 0, 12, 99,  0)),
    "PMS 116":   ((255, 194,   0), ( 0, 19, 99,  0)),
    "PMS 130":   ((247, 168,   0), ( 0, 35, 99,  0)),
    "PMS 151":   ((255, 121,   0), ( 0, 55, 99,  0)),
    "PMS 021":   ((254,  80,   0), ( 0, 74, 99,  0)),
    "PMS 185":   ((228,   0,  43), ( 0, 99, 70,  0)),
    "PMS 485":   ((218,  37,  29), ( 0, 94, 86,  0)),
    "PMS 032":   ((239,  51,  64), ( 0, 84, 64,  0)),
    "PMS 200":   ((196,  18,  48), ( 0, 99, 68,  5)),
    "PMS 206":   ((162,   0,  45), ( 0, 99, 60, 20)),
    "PMS 220":   ((145,   0,  66), ( 0, 99, 47, 30)),
    "PMS 266":   ((122,  81, 169), (48, 65,  0,  0)),
    "PMS 265":   ((122,  85, 183), (45, 62,  0,  0)),
    "PMS 072":   ( (17,  29, 146), (97, 86,  0, 15)),
    "PMS 286":   ((  0,  83, 165), (99, 57,  0,  0)),
    "PMS 300":   ((  0, 101, 189), (99, 43,  0,  0)),
    "PMS 313":   ((  0, 155, 199), (80, 10,  0,  0)),
    "PMS 326":   ((  0, 179, 152), (75,  0, 38,  0)),
    "PMS 347":   ((  0, 154,  68), (88,  0, 75,  0)),
    "PMS 354":   ((  0, 176,  80), (79,  0, 76,  0)),
    "PMS 362":   ( (84, 163,  71), (66,  0, 84, 10)),
    "PMS 375":   ((148, 215,  81), (33,  0, 79,  0)),
    "PMS 390":   ((198, 214,   0), (12,  0, 99,  0)),
    "PMS 421":   ((173, 173, 173), ( 0,  0,  0, 35)),
    "PMS 430":   ((128, 143, 152), (18,  7,  0, 42)),
    "PMS 877":   ((150, 153, 155), ( 0,  0,  0, 43)),  # silver
    "PMS 871":   ((175, 141,  51), ( 0, 15, 85, 25)),  # gold
    "PMS WARM RED": ((244,  67,  54), ( 0, 82, 71,  0)),
    "PMS PROCESS BLACK": ((0, 0, 0), (0, 0, 0, 100)),
    "PMS REFLEX BLUE":   ((0,  20, 137), (99, 90,  0, 10)),
    "PMS RHODAMINE RED": ((209,  0, 116), ( 0, 99, 15,  0)),
    "PMS GREEN":         ((  0, 168,  89), (87,  0, 73,  0)),
    "PMS PURPLE":        ((107,  32, 146), (55, 87,  0,  5)),
}


def _nearest_pms(r: int, g: int, b: int) -> tuple:
    """Return (pms_code, rgb) of the closest PMS color by Euclidean distance."""
    best_name, best_dist = None, float("inf")
    for name, (rgb, _) in PMS_DB.items():
        dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, PMS_DB[best_name]


# Match setrgbcolor: "R G B setrgbcolor"
_RGB_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+setrgbcolor"
)


class ApplyPMSInput(BaseModel):
    eps_path: str = Field(..., description="Absolute path to the EPS file to recolor")
    pms_codes: List[str] = Field(
        ...,
        description=(
            "List of PMS color codes to map to, e.g. ['PMS 485', 'PMS 286']. "
            "Each RGB color in the EPS is replaced by the nearest PMS code's CMYK equivalent. "
            "If empty list, all colors are auto-mapped to their nearest PMS."
        ),
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save the recolored EPS. Defaults to <name>_pms.eps.",
    )


def _apply_pms_color(eps_path: str, pms_codes: List[str], output_path: Optional[str] = None) -> str:
    result: dict = {"eps_path": eps_path}

    try:
        with open(eps_path, "r", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        result["error"] = f"Cannot read EPS: {exc}"
        return json.dumps(result)

    # Build target palette: if pms_codes given, restrict mapping to those
    if pms_codes:
        unknown = [c for c in pms_codes if c.upper() not in {k.upper() for k in PMS_DB}]
        if unknown:
            result["warning"] = f"Unknown PMS codes (will be ignored): {unknown}"
        target_palette = {k: v for k, v in PMS_DB.items()
                          if k.upper() in {c.upper() for c in pms_codes}}
        if not target_palette:
            result["error"] = "None of the provided PMS codes are in the database"
            return json.dumps(result)
    else:
        target_palette = PMS_DB

    mappings: dict = {}
    replacements: dict = {}

    def _replace_color(m: re.Match) -> str:
        r_f, g_f, b_f = float(m.group(1)), float(m.group(2)), float(m.group(3))
        # Determine if values are 0-1 or 0-255
        if r_f <= 1.0 and g_f <= 1.0 and b_f <= 1.0:
            r, g, b = int(r_f * 255), int(g_f * 255), int(b_f * 255)
        else:
            r, g, b = int(r_f), int(g_f), int(b_f)

        cache_key = (r, g, b)
        if cache_key in replacements:
            return replacements[cache_key]

        # Find nearest in target_palette
        best_name, best_dist = None, float("inf")
        for name, (rgb, _) in target_palette.items():
            dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
            if dist < best_dist:
                best_dist = dist
                best_name = name

        _, (pms_rgb, pms_cmyk) = best_name, target_palette[best_name]
        c, mg, y, k = [v / 100.0 for v in pms_cmyk]
        new_r, new_g, new_b = [v / 255.0 for v in pms_rgb]

        new_line = (
            f"% PMS {best_name} C={pms_cmyk[0]} M={pms_cmyk[1]} Y={pms_cmyk[2]} K={pms_cmyk[3]}\n"
            f"{new_r:.4f} {new_g:.4f} {new_b:.4f} setrgbcolor"
        )
        replacements[cache_key] = new_line
        mappings[f"rgb({r},{g},{b})"] = {"pms": best_name, "cmyk": pms_cmyk}
        return new_line

    new_content = _RGB_RE.sub(_replace_color, content)

    if not output_path:
        p = Path(eps_path)
        output_path = str(p.parent / f"{p.stem}_pms.eps")

    try:
        with open(output_path, "w") as f:
            f.write(new_content)
    except Exception as exc:
        result["error"] = f"Cannot write EPS: {exc}"
        return json.dumps(result)

    result["output_path"] = output_path
    result["color_mappings"] = mappings
    result["colors_replaced"] = len(mappings)
    return json.dumps(result)


apply_pms_color_tool = StructuredTool.from_function(
    func=_apply_pms_color,
    name="apply_pms_color",
    description=(
        "Replaces RGB colors in an EPS file with their nearest Pantone (PMS) equivalents. "
        "Pass pms_codes=['PMS 485', 'PMS 286'] to constrain mapping to those colors, "
        "or pass an empty list to auto-map every color to its nearest PMS. "
        "Only call this tool if the user provided PMS codes — skip it otherwise. "
        "Returns JSON with: output_path, color_mappings (original rgb → pms + cmyk), colors_replaced."
    ),
    args_schema=ApplyPMSInput,
)
