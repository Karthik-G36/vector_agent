# Vector Agent

Converts any raster image into a print-ready **SVG + EPS** vector file.

Pipeline: PIL upscale → (optional AI enhancement) → PNG → SVG → EPS

---

## Pipeline

```
Input image  (PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF)
   │
   ▼  Step 1 — Enhance
   │  Default : PIL LANCZOS upscale + UnsharpMask sharpening
   │  Optional: gpt-image-1 resolution enhancement → then PIL upscale
   │            (set CALL_AGENT=true)
   │
   ▼  Step 2 — Upscale (optional)
   │  Real-ESRGAN AI upscale — better edge sharpness on complex images
   │  (set USE_REALESRGAN=true)
   │
   ▼  Step 3 — PNG → SVG
   │  Default : Inkscape bitmap tracer (Gaussian blur pre-processing)
   │  Optional: vtracer — better for gradient logos, no concentric rings
   │            (set USE_VTRACER=true, falls back to Inkscape on failure)
   │
   ▼  Step 4 — SVG → EPS
      Inkscape PostScript export — print-ready EPS
```

The upscale factor is computed automatically so the output PNG never exceeds **20 MB**.

---

## Quick Start (CLI)

### 1. Prerequisites

**Python 3.10+** required.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Inkscape >= 1.2** *(required for tracing and EPS export)*:

| Platform | Install |
|----------|---------|
| Windows  | https://inkscape.org/release/ — check "Add to PATH" during install |
| macOS    | `brew install inkscape` |
| Ubuntu   | `sudo apt install inkscape` |

**Environment:**

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY if using CALL_AGENT=true
```

### 2. Run

```bash
# Full pipeline
python main.py logo.png

# Skip enhancement (trace original directly)
python main.py logo.png --no-enhance

# Custom output directory
python main.py logo.png --output-dir ./results
```

Output files are saved to `./output/` by default:

| File | Description |
|------|-------------|
| `<name>_enhanced.png` | Enhanced / upscaled PNG |
| `<name>.svg` | Vectorized SVG |
| `<name>.eps` | Print-ready EPS |

---

## REST API

### Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

### POST `/generate`

Run the full pipeline. Accepts JSON.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Image URL (`http/https`) **or** base64 string (plain or `data:image/png;base64,...`) |
| `filename` | string | No | Output file stem. Auto-detected from URL; defaults to `"image"` for base64 |
| `call_agent` | bool | No | Use gpt-image-1 enhancement. Overrides `CALL_AGENT` env |
| `no_enhance` | bool | No | Skip enhancement step entirely |
| `use_vtracer` | bool | No | Use vtracer for PNG→SVG. Overrides `USE_VTRACER` env |
| `use_realesrgan` | bool | No | Run Real-ESRGAN upscale. Overrides `USE_REALESRGAN` env |

**Example — image URL:**
```json
{
  "image": "https://example.com/logo.png",
  "filename": "logo",
  "use_vtracer": true
}
```

**Example — base64:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "filename": "logo",
  "call_agent": false
}
```

**Success response:**
```json
{
  "job_id": "f3a9c12e-4b1d-...",
  "status": "done",
  "files": {
    "enhanced_png": "/files/f3a9c12e-4b1d-.../logo_enhanced.png",
    "svg":          "/files/f3a9c12e-4b1d-.../logo.svg",
    "eps":          "/files/f3a9c12e-4b1d-.../logo.eps"
  },
  "meta": {
    "stem":           "logo",
    "tracer":         "Inkscape",
    "call_agent":     false,
    "use_realesrgan": false,
    "elapsed_sec":    22.4
  },
  "expires_at": "2026-05-11T11:00:00+00:00"
}
```

**Error response:**
```json
{
  "detail": "Unsupported image format '.pdf'. Supported: .bmp, .gif, .jpg, ..."
}
```

---

### GET `/files/{job_id}/{filename}`

Download a generated file.

```bash
curl -O http://localhost:8000/files/f3a9c12e-.../logo.svg
curl -O http://localhost:8000/files/f3a9c12e-.../logo.eps
```

Returns `404` if the job does not exist or files have expired. Generated files are automatically deleted after `FILE_TTL_MINUTES` (default: 60 minutes).

---

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `./output` | Directory for generated files |
| `CALL_AGENT` | `false` | Use gpt-image-1 enhancement (requires `OPENAI_API_KEY`) |
| `OPENAI_API_KEY` | — | OpenAI API key — required only when `CALL_AGENT=true` |
| `USE_VTRACER` | `false` | Use vtracer for PNG→SVG instead of Inkscape |
| `USE_REALESRGAN` | `false` | Run Real-ESRGAN AI upscale after PIL upscale |
| `REALESRGAN_TILE` | `512` | Real-ESRGAN tile size (`0` = full image, needs high VRAM) |
| `FILE_TTL_MINUTES` | `60` | Minutes before generated API files are auto-deleted |

Payload flags (`call_agent`, `use_vtracer`, `use_realesrgan`) override their corresponding env variables per request.

---

## Optional Dependencies

### vtracer (recommended for gradient logos)

```bash
pip install vtracer
```

Better than Inkscape for logos with gradients — avoids concentric rings and color bleed. Falls back to Inkscape automatically if it crashes.

### Real-ESRGAN (AI upscaling)

Install PyTorch first (CPU-only, ~200 MB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install realesrgan basicsr facexlib gfpgan opencv-python numpy
```

The model weights (~67 MB) download automatically on first run.

---

## Project Structure

```
vector_agent/
├── main.py                  ← CLI entry point
├── api.py                   ← REST API (FastAPI)
├── pipeline/
│   ├── enhance.py           ← PIL upscale + optional gpt-image-1 enhancement
│   ├── vectorize.py         ← Inkscape PNG→SVG and SVG→EPS
│   ├── vtracer_convert.py   ← vtracer PNG→SVG (isolated subprocess)
│   └── upscale.py           ← Real-ESRGAN AI upscaling
├── requirements.txt
├── .env.example
└── output/                  ← generated files (gitignored)
```

---

## Troubleshooting

**`Inkscape not found`**
→ Install Inkscape >= 1.2 from https://inkscape.org/release/ and ensure it is on your PATH.

**`OPENAI_API_KEY is not set`**
→ Only needed when `CALL_AGENT=true`. Copy `.env.example` to `.env` and set the key.

**SVG has concentric rings on gradient logos**
→ Enable vtracer: set `USE_VTRACER=true` in `.env` or pass `"use_vtracer": true` in the API payload.

**SVG is a plain image box (not traced)**
→ Inkscape trace produced no paths. Check that Inkscape's `bin/` folder is on PATH and the input image is not corrupted.

**vtracer crashes silently**
→ The pipeline automatically falls back to Inkscape. This is often a Python version incompatibility with the vtracer native extension.

**API request times out**
→ The pipeline is synchronous and can take 20–90 seconds depending on options. Set your HTTP client timeout to at least 120 seconds. If running behind Nginx, set `proxy_read_timeout 120s`.
