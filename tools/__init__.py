from .validate_image import validate_image_tool
from .preprocess_image import preprocess_image_tool
from .segment_profile import segment_and_profile_tool
from .enhance_gpt import enhance_with_gpt_tool
from .high_fidelity_svg_vectorize import high_fidelity_svg_vectorize_tool
from .semantic_svg_vectorize import semantic_svg_vectorize_tool
from .gpt_svg_vectorize import gpt_svg_vectorize_tool
from .vectorize_eps import vectorize_to_eps_tool        # fallback only
from .apply_pms_color import apply_pms_color_tool

ALL_TOOLS = [
    validate_image_tool,
    preprocess_image_tool,
    segment_and_profile_tool,
    enhance_with_gpt_tool,
    high_fidelity_svg_vectorize_tool, # max fidelity route for any 2D/3D raster graphic
    semantic_svg_vectorize_tool, # semantic route for 3D text effects/mockups
    gpt_svg_vectorize_tool,      # primary - LLM-powered, clean bezier output
    vectorize_to_eps_tool,       # fallback - contour tracing for photos or GPT failure
    apply_pms_color_tool,
]


def get_all_tools():
    return ALL_TOOLS
