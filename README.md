# Vector Agent

Converts any raster image into a print-ready **SVG + EPS** vector file.

Two modes of operation: **CLI** for local use and a **REST API** with async webhook delivery for server use.

---

## Pipeline

```
Input image  (PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF)
   │
   ▼  Step 1 — Enhance
   │  Default : PIL LANCZOS upscale + UnsharpMask sharpening
   │  Optional: Real-ESRGAN AI upscale → then PIL sharpen  (USE_REALESRGAN=true)
   │  Optional: gpt-image-1 resolution enhancement          (CALL_AGENT=true)
   │
   ▼  Step 2 — Remove Background  (optional, REMOVE_BG=true)
   │  rembg U2Net model — strips background before tracing
   │
   ▼  Step 3 — PNG → SVG
   │  Default : Inkscape bitmap tracer (Gaussian blur pre-processing)
   │  Optional: vtracer — better for gradient logos, no concentric rings
   │            (USE_VTRACER=true, falls back to Inkscape on failure)
   │  Optional: GPT-4o vision agent — iteratively tunes vtracer parameters
   │            (USE_VTRACER_AGENT=true, up to 3 iterations)
   │  Optional: SAM 2 segmentation → per-region masks → layered SVG
   │            (USE_SAM2=true)
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
# Edit .env — set OPENAI_API_KEY if using CALL_AGENT=true or USE_VTRACER_AGENT=true
```

### 2. Run

```bash
# Default pipeline (LANCZOS enhance → Inkscape trace → EPS)
python main.py logo.png

# Use vtracer instead of Inkscape
python main.py logo.png --use-vtracer

# Use vtracer + GPT-4o agent to auto-tune tracing params
python main.py logo.png --use-vtracer --use-vtracer-agent

# AI upscale with Real-ESRGAN before tracing
python main.py logo.png --use-realesrgan

# Remove background before tracing
python main.py logo.png --remove-bg

# Skip enhancement entirely
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

**Token usage** is printed at the end of every run that calls an OpenAI API:

```
=== OpenAI Token Usage ===
Call                    Model     Prompt  Completion     Total
──────────────────────────────────────────────────────────────
vtracer-agent (vision)  gpt-4o     3,276          76     3,352
──────────────────────────────────────────────────────────────
Session total                                             3,352
```

---

## REST API

The API is **asynchronous** — jobs run in the background and notify your server via a **webhook** when done. Each job can take 5–10 minutes; the HTTP response returns immediately with a `job_id`.

### Start the server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

---

### POST `/vectorize`

Submit a vectorization job. Returns a `job_id` in under 1 second.

**Request body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | string | Yes | Base64 data URI (`data:image/png;base64,...`) **or** a publicly reachable image URL |
| `callback_url` | string | Yes | URL the server will POST the result to when the job finishes |
| `use_vtracer` | bool | No | Use vtracer for PNG→SVG (default: `false`) |
| `use_vtracer_agent` | bool | No | Run GPT-4o vision agent to auto-tune vtracer params (default: `false`) |
| `use_esrgan` | bool | No | Run Real-ESRGAN AI upscale (default: `false`) |
| `esrgan_tile` | int | No | Real-ESRGAN tile size (default: `512`) |
| `remove_bg` | bool | No | Remove background before tracing (default: `false`) |
| `use_sam2` | bool | No | Segment with SAM 2 → layered SVG (default: `false`) |
| `call_agent` | bool | No | Use gpt-image-1 for enhancement (default: `false`) |
| `no_enhance` | bool | No | Skip enhancement step entirely (default: `false`) |

**Example — image URL:**
```json
{
  "image": "https://example.com/logo.png",
  "callback_url": "https://your-app.com/webhook/vector-done",
  "use_vtracer": true
}
```

**Example — base64:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "callback_url": "https://your-app.com/webhook/vector-done",
  "use_vtracer": true,
  "use_vtracer_agent": true
}
```

**Response (HTTP 202):**
```json
{
  "job_id": "f3a9c12e-4b1d-4c2a-9f1e-123456789abc",
  "status": "queued"
}
```

---

### Webhook payload

When the job finishes the server POSTs to your `callback_url`. The webhook is retried up to **3 times** with 2 s / 5 s / 10 s backoff if your server returns a non-2xx response.

**On success:**
```json
{
  "job_id": "f3a9c12e-4b1d-4c2a-9f1e-123456789abc",
  "status": "done",
  "svg_url": "/result/f3a9c12e-.../image.svg",
  "eps_url": "/result/f3a9c12e-.../image.eps",
  "token_usage": {
    "total_tokens": 3352,
    "breakdown": [
      {
        "label": "vtracer-agent (vision)",
        "model": "gpt-4o",
        "prompt_tokens": 3276,
        "completion_tokens": 76,
        "total_tokens": 3352
      }
    ]
  }
}
```

**On failure:**
```json
{
  "job_id": "f3a9c12e-4b1d-4c2a-9f1e-123456789abc",
  "status": "failed",
  "error": "vtracer subprocess failed (exit 1): ..."
}
```

Use the `svg_url` / `eps_url` values to download files via `GET /result/{job_id}/{filename}`.

---

### GET `/status/{job_id}`

Fallback polling endpoint — use this if your server cannot receive webhooks.

**Response:**
```json
{
  "job_id": "f3a9c12e-...",
  "status": "processing",
  "error": null
}
```

| `status` value | Meaning |
|----------------|---------|
| `queued` | Job accepted, waiting to start |
| `processing` | Pipeline is running |
| `done` | Completed — files are ready |
| `failed` | Pipeline error — see `error` field |

---

### GET `/result/{job_id}/{filename}`

Download an output file using the `svg_url` or `eps_url` from the webhook payload.

```bash
curl -O http://localhost:8000/result/f3a9c12e-.../image.svg
curl -O http://localhost:8000/result/f3a9c12e-.../image.eps
```

Returns `404` if the job does not exist, `409` if the job is not yet done.

---

## Environment Variables

Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required when `CALL_AGENT=true` or `USE_VTRACER_AGENT=true` |
| `CALL_AGENT` | `false` | Use gpt-image-1 for image enhancement |
| `AGENT_MODEL` | `gpt-4o` | Model used by the vtracer vision agent |
| `USE_VTRACER` | `false` | Use vtracer for PNG→SVG instead of Inkscape |
| `USE_VTRACER_AGENT` | `false` | Run GPT-4o agent to auto-tune vtracer params |
| `USE_REALESRGAN` | `false` | Run Real-ESRGAN AI upscale after PIL upscale |
| `REALESRGAN_TILE` | `512` | Real-ESRGAN tile size (`0` = full image, needs high VRAM) |
| `REMOVE_BG` | `false` | Remove background before tracing |
| `USE_SAM2` | `false` | Segment image with SAM 2 → layered SVG |

CLI flags and API request fields override the corresponding env variable for that run.

---

## Optional Dependencies

### vtracer (recommended for gradient logos)

```bash
pip install vtracer
```

Better than Inkscape for logos with gradients — avoids concentric rings and color bleed. Falls back to Inkscape automatically on crash.

### Real-ESRGAN (AI upscaling)

Install PyTorch first (CPU-only, ~200 MB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install realesrgan basicsr facexlib gfpgan opencv-python numpy
```

Model weights (~67 MB) download automatically on first run.

### SAM 2 (segmentation)

```bash
pip install sam2 huggingface_hub
```

Model checkpoint (~300 MB) downloaded automatically on first run via Hugging Face.

### rembg (background removal)

```bash
pip install rembg onnxruntime
```

U2Net model (~170 MB) downloaded automatically on first run.

---

## Project Structure

```
vector_agent/
├── main.py                    ← CLI entry point
├── api/
│   ├── app.py                 ← FastAPI application (endpoints)
│   ├── job_store.py           ← In-memory job state tracking
│   ├── webhook.py             ← Webhook dispatcher with retry/backoff
│   └── worker.py              ← Background job runner (ThreadPoolExecutor)
├── pipeline/
│   ├── run.py                 ← Core pipeline orchestration (CLI + API shared)
│   ├── enhance.py             ← PIL upscale + optional gpt-image-1 enhancement
│   ├── vectorize.py           ← Inkscape PNG→SVG and SVG→EPS
│   ├── vtracer_convert.py     ← vtracer PNG→SVG (isolated subprocess)
│   ├── vtracer_agent.py       ← GPT-4o vision agent for vtracer param tuning
│   ├── upscale.py             ← Real-ESRGAN AI upscaling
│   ├── removebg.py            ← Background removal (rembg)
│   ├── segment.py             ← SAM 2 segmentation
│   ├── compose.py             ← Layered SVG composition
│   └── token_tracker.py       ← OpenAI token usage tracking
├── requirements.txt
├── .env.example
└── output/                    ← CLI output files (gitignored)
```

---

## Troubleshooting

**`Inkscape not found`**
→ Install Inkscape >= 1.2 from https://inkscape.org/release/ and ensure it is on your PATH.

**`OPENAI_API_KEY` missing in API mode**
→ The API reads from `.env` via `python-dotenv`. Ensure `.env` exists in the working directory where you run `uvicorn` and contains `OPENAI_API_KEY=sk-...`.

**`vision call failed: 'OPENAI_API_KEY'`**
→ Same as above — the `.env` file is not loaded. Required only when `use_vtracer_agent=true` or `call_agent=true`.

**SVG has concentric rings on gradient logos**
→ Enable vtracer: `USE_VTRACER=true` in `.env` or `"use_vtracer": true` in the API request.

**SVG is a plain image box (not traced)**
→ Inkscape trace produced no paths. Verify Inkscape's `bin/` is on PATH and the input image is not corrupted.

**vtracer crashes silently**
→ The pipeline automatically falls back to Inkscape. Usually a Python version incompatibility with the vtracer native extension.

**Webhook not received**
→ Use `GET /status/{job_id}` to check if the job completed. Check server logs for `[webhook]` lines — the server retries 3 times before giving up. Ensure your `callback_url` is publicly reachable from the server.
