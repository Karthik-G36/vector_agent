# Vector Agent

Converts raster logo images (PNG, JPG, BMP) into print-ready EPS vector files.

Two modes: a **standalone advanced pipeline** (recommended) and a **ReAct agent** that orchestrates multiple specialist tools automatically.

---

## Quick Start

### 1. Prerequisites

**Python 3.10+** and **Git** are required.

```bash
# Clone / enter the project
cd vector_agent

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r vector_agent/requirements.txt
```

**OpenAI API key** — required for all LLM features:

```bash
# Inside vector_agent/
cp .env.example .env
# Edit .env and set:  OPENAI_API_KEY=sk-...
```

**Inkscape** *(optional but strongly recommended for best EPS quality)*:

| Platform | Install |
|----------|---------|
| Windows  | https://inkscape.org/release/ — add to PATH during install |
| macOS    | `brew install inkscape` |
| Ubuntu   | `sudo apt install inkscape` |

Without Inkscape the pipeline falls back to CairoSVG then a built-in parser — both produce correct output but Inkscape uses PostScript Level 3 with true gradient rendering.

---

## Mode 1 — Advanced Pipeline (recommended)

`vectorize_advanced.py` — CIELAB colour quantisation + gradient preservation + SVG filters + 3D depth cues.

```bash
# Basic usage — output goes to ./output/<name>_advanced.eps
python vectorize_advanced.py logo.png

# Custom output path and canvas size
python vectorize_advanced.py logo.png --output brand.eps --width-mm 120 --height-mm 80

# Skip saving the intermediate SVG
python vectorize_advanced.py logo.png --no-svg
```

### What the pipeline does

```
Input image
   │
   ▼  Stage 1 — Preprocess
   │  Lanczos upscale to ≥1400 px + bilateral denoising (kills JPEG halos)
   │
   ▼  Stage 2 — LLM Analysis  (GPT-4o)
   │  Returns structured JSON: image type, layer breakdown, gradient zones,
   │  shadow/glow/bevel flags, total colour-region count (k)
   │
   ▼  Stage 3 — Perceptual Segmentation
   │  CIELAB median-cut quantisation (k regions)
   │  Per-region gradient detection: solid / linear / radial
   │
   ▼  Stage 4 — Effect Detection
   │  Scans boundary softness against original pixels
   │  Dark + soft edge → feGaussianBlur shadow filter
   │  Bright + soft edge → feMerge glow filter
   │
   ▼  Stage 5 — SVG Assembly
   │  <defs> with <linearGradient>, <radialGradient>, <filter> elements
   │  Regions rendered back-to-front (background → foreground)
   │
   ▼  Stage 6 — EPS Export
      Inkscape CLI (PS Level 3) → CairoSVG → built-in parser
```

### Output files

| File | Description |
|------|-------------|
| `output/<name>_advanced.svg` | Intermediate SVG with gradients and filter defs |
| `output/<name>_advanced.eps` | Final print-ready EPS |

---

## Mode 2 — ReAct Agent

`main.py` runs a LangChain ReAct agent that picks the best tool for each image automatically.

```bash
python main.py logo.png
python main.py logo.png --output logo.eps --width-mm 100 --height-mm 120
python main.py logo.png --output logo.eps --width-mm 100 --height-mm 120 --pms "PMS 485" "PMS 286"
python main.py logo.png --quiet   # suppress step-by-step output
```

### Agent tool chain

```
validate_image          → quality score + issue list
preprocess_image        → denoise / deskew / sharpen
enhance_with_gpt        → GPT-powered upscale (if quality < 0.7)
high_fidelity_svg_vectorize   → max-detail tonal trace (4× upscale, 80 colours)
semantic_svg_vectorize        → 3D text effects: text + shapes + gradients
gpt_svg_vectorize             → GPT-4o generates clean SVG code directly
vectorize_to_eps              → fallback contour tracing → EPS
apply_pms_color               → Pantone colour mapping (optional)
```

---

## Mode 3 — Legacy Scripts

```bash
# Classic LLM-guided colour clustering (original approach, kept for reference)
python vectorize_llm_guided.py logo.png

# Pure vtracer vectorisation (no AI, fastest)
python vectorize_pure.py logo.png
```

---

## Environment Variables

Copy `.env.example` to `.env` and edit:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key |
| `AGENT_MODEL` | No | `gpt-4o` | GPT model name |
| `OUTPUT_DIR` | No | `./output` | Output directory |
| `POTRACE_BIN` | No | `potrace` | Path to potrace binary |

---

## Project Structure

```
vector_agent/
├── vectorize_advanced.py      ← NEW: standalone advanced pipeline entry point
├── main.py                    ← ReAct agent entry point
├── agent.py                   ← LangChain ReAct agent + system prompt
├── vectorize_llm_guided.py    ← legacy: LLM k-means pipeline
├── vectorize_pure.py          ← legacy: vtracer pipeline
│
├── pipeline/                  ← NEW: advanced pipeline modules
│   ├── color_engine.py        ← CIELAB median-cut + gradient detection
│   ├── path_tracer.py         ← adaptive-tension bezier path fitting
│   ├── effect_detector.py     ← shadow/glow detection → SVG filter defs
│   ├── svg_builder.py         ← SVG assembly with <defs>, gradients, filters
│   ├── llm_analyzer.py        ← structured GPT-4o image analysis
│   ├── eps_exporter.py        ← Inkscape → CairoSVG → built-in chain
│   └── orchestrator.py        ← 6-stage pipeline controller
│
├── tools/                     ← agent tool functions
│   ├── validate_image.py
│   ├── preprocess_image.py
│   ├── segment_profile.py
│   ├── enhance_gpt.py
│   ├── high_fidelity_svg_vectorize.py
│   ├── semantic_svg_vectorize.py
│   ├── gpt_svg_vectorize.py   ← updated: gradient/3D-aware SVG prompt
│   ├── vectorize_eps.py
│   └── apply_pms_color.py
│
├── utils/
│   └── conversions.py         ← mm ↔ pts ↔ px unit conversions
│
├── output/                    ← generated files (gitignored)
├── .env.example
└── requirements.txt
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
→ Make sure the virtual environment is activated and `pip install -r requirements.txt` completed without errors.

**`OPENAI_API_KEY is not set`**
→ Copy `.env.example` → `.env` and fill in your key.

**EPS looks like flat colours even though the logo has gradients**
→ Install Inkscape (see Prerequisites). Without it, SVG filter effects and gradients are approximated by the built-in fallback parser.

**Very large SVG / slow export**
→ This is normal for 3D logos — more gradient regions = more paths. Use `--no-svg` if you only need the EPS.

**`potrace` not found warning**
→ Only affects the agent's fallback path for flat B&W logos. Install potrace or ignore — the pipeline uses bezier contour tracing as fallback.
