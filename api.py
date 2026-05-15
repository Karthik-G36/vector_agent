#!/usr/bin/env python3
"""
Vector Agent — REST API
  POST /generate                  → run full pipeline, return download URLs
  GET  /files/{job_id}/{filename} → download a generated file

Payload (JSON):
  image         — required — image URL (http/https) OR base64 string
                             (plain base64 or data URL: data:image/png;base64,...)
  filename      — optional — output stem name (auto-detected from URL; defaults to "image")
  call_agent    — optional bool — overrides CALL_AGENT env
  no_enhance    — optional bool — overrides NO_ENHANCE env
  use_vtracer   — optional bool — overrides USE_VTRACER env
  use_realesrgan— optional bool — overrides USE_REALESRGAN env

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000
"""
import base64
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Vector Agent API", version="1.0.0")

OUTPUT_ROOT      = Path(os.getenv("OUTPUT_DIR", "output"))
FILE_TTL_MINUTES = int(os.getenv("FILE_TTL_MINUTES", "60"))

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

MEDIA_TYPES = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".eps": "application/postscript",
}

_CONTENT_TYPE_TO_SUFFIX: dict[str, str] = {
    "image/png":  ".png",
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/tiff": ".tiff",
    "image/gif":  ".gif",
}


# ── Request model ─────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    image:          str            # URL or base64 (plain or data: prefix)
    filename:       Optional[str]  = None   # output stem; auto-detected if omitted
    call_agent:     Optional[bool] = None
    no_enhance:     Optional[bool] = None
    use_vtracer:    Optional[bool] = None
    use_realesrgan: Optional[bool] = None
    remove_bg:      Optional[bool] = None
    use_sam2:       Optional[bool] = None


# ── Image input helpers ───────────────────────────────────────────────────────

def _is_url(image: str) -> bool:
    return image.startswith("http://") or image.startswith("https://")


async def _fetch_from_url(url: str) -> tuple[bytes, str, str]:
    """
    Download image from URL.
    Returns (bytes, suffix, stem).
    """
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = client.build_request("GET", url)
        r = await client.send(response)
        r.raise_for_status()
        image_bytes = r.content

    # Derive suffix from URL path first, then fall back to Content-Type header
    clean_url = url.split("?")[0]
    suffix = Path(clean_url).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        content_type = r.headers.get("content-type", "")
        suffix = next(
            (s for ct, s in _CONTENT_TYPE_TO_SUFFIX.items() if ct in content_type),
            ".png",
        )

    stem = Path(clean_url).stem or "image"
    return image_bytes, suffix, stem


def _decode_base64_image(image: str) -> tuple[bytes, str]:
    """
    Decode base64 image string — accepts:
      - Plain base64 string
      - Data URL:  data:image/png;base64,<data>
    Returns (bytes, suffix).
    """
    suffix = ".png"  # default when type cannot be determined

    if image.startswith("data:"):
        match = re.match(r"data:image/(\w+);base64,(.+)", image, re.DOTALL)
        if not match:
            raise ValueError("Unrecognised data URL format. Expected: data:image/<type>;base64,<data>")
        img_type = match.group(1).lower()
        suffix   = f".{img_type}" if img_type != "jpeg" else ".jpg"
        image    = match.group(2)

    try:
        image_bytes = base64.b64decode(image)
    except Exception:
        raise ValueError("Invalid base64 string — could not decode image data.")

    return image_bytes, suffix


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve(payload_value: Optional[bool], env_key: str, default: bool = False) -> bool:
    """Payload value takes priority over env; env over hardcoded default."""
    if payload_value is not None:
        return payload_value
    return os.getenv(env_key, str(default)).lower() == "true"


def _is_expired(job_dir: Path) -> bool:
    age_seconds = time.time() - job_dir.stat().st_ctime
    return age_seconds > FILE_TTL_MINUTES * 60


def _safe_stem(raw: str) -> str:
    """Strip characters unsafe for filenames."""
    return re.sub(r"[^\w\-]", "_", raw)[:64] or "image"


# ── Background cleanup ────────────────────────────────────────────────────────

def _cleanup_loop() -> None:
    """Delete job folders older than FILE_TTL_MINUTES. Runs every 30 minutes."""
    while True:
        time.sleep(30 * 60)
        if not OUTPUT_ROOT.exists():
            continue
        for job_dir in OUTPUT_ROOT.iterdir():
            if job_dir.is_dir() and _is_expired(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ── API 1: POST /generate ─────────────────────────────────────────────────────

@app.post("/generate")
async def generate(request: GenerateRequest):
    # ── Resolve image bytes + suffix + stem ──────────────────────────────────
    try:
        if _is_url(request.image):
            image_bytes, suffix, url_stem = await _fetch_from_url(request.image)
            stem = _safe_stem(request.filename or url_stem)
        else:
            image_bytes, suffix = _decode_base64_image(request.image)
            stem = _safe_stem(request.filename or "image")
    except (ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    # ── Resolve config flags (payload > env > default) ────────────────────────
    cfg_call_agent     = _resolve(request.call_agent,     "CALL_AGENT",     False)
    cfg_no_enhance     = _resolve(request.no_enhance,     "NO_ENHANCE",     False)
    cfg_use_vtracer    = _resolve(request.use_vtracer,    "USE_VTRACER",    False)
    cfg_use_realesrgan = _resolve(request.use_realesrgan, "USE_REALESRGAN", False)
    cfg_remove_bg      = _resolve(request.remove_bg,      "REMOVE_BG",      False)
    cfg_use_sam2       = _resolve(request.use_sam2,       "USE_SAM2",       False)
    esrgan_tile        = int(os.getenv("REALESRGAN_TILE", "512"))

    # ── Set up job directory ──────────────────────────────────────────────────
    job_id  = str(uuid.uuid4())
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    start_time  = time.time()
    tracer_used = "vtracer" if cfg_use_vtracer else "Inkscape"

    try:
        # Save image to disk
        original_path = job_dir / f"original{suffix}"
        original_path.write_bytes(image_bytes)

        source_png = original_path

        # ── Step 1: Enhance ───────────────────────────────────────────────────
        if not cfg_no_enhance:
            import shutil as _shutil
            enhanced_png = job_dir / f"{stem}_enhanced.png"

            if cfg_use_realesrgan:
                # Real-ESRGAN has priority — upscale original first
                from pipeline.upscale import upscale_image
                _shutil.copy2(str(original_path), str(enhanced_png))
                upscale_image(str(enhanced_png), tile=esrgan_tile)
                source_png = enhanced_png

                if cfg_call_agent:
                    # Then pass ESRGAN output through gpt-image-1
                    from pipeline.enhance import enhance_with_agent
                    enhance_with_agent(str(enhanced_png), str(enhanced_png))
                else:
                    from pipeline.enhance import enhance_image
                    enhance_image(str(enhanced_png), str(enhanced_png))
            elif cfg_call_agent:
                from pipeline.enhance import enhance_with_agent
                enhance_with_agent(str(original_path), str(enhanced_png))
                source_png = enhanced_png
            else:
                from pipeline.enhance import enhance_image
                enhance_image(str(original_path), str(enhanced_png))
                source_png = enhanced_png

        # ── Step 3: Remove background (optional) ─────────────────────────────
        if cfg_remove_bg:
            nobg_png = job_dir / f"{stem}_nobg.png"
            from pipeline.removebg import remove_background
            remove_background(str(source_png), str(nobg_png))
            source_png = nobg_png

        # ── Step 4: Segment → layered SVG  OR  single-image trace ───────────
        svg_path = job_dir / f"{stem}.svg"
        if cfg_use_sam2:
            from pipeline.segment import segment_image, merge_masks_by_color
            from pipeline.vectorize import trace_mask_to_svg
            from pipeline.compose import compose_svg

            masks_dir = job_dir / f"{stem}_masks"
            masks_dir.mkdir(exist_ok=True)

            segments = segment_image(str(source_png), str(masks_dir))
            merged = merge_masks_by_color(segments, masks_dir)

            svg_segments = []
            for seg in merged:
                traced_svg_path = masks_dir / (Path(seg["mask_path"]).stem + "_traced.svg")
                try:
                    trace_mask_to_svg(seg["mask_path"], str(traced_svg_path))
                    svg_segments.append({
                        "traced_svg": traced_svg_path.read_text(encoding="utf-8"),
                        "color": seg["color"],
                    })
                except Exception:
                    pass

            if not svg_segments:
                raise RuntimeError("SAM 2 segmentation produced no traceable masks.")

            compose_svg(svg_segments, str(svg_path))
            tracer_used = "SAM2 + Inkscape (layered)"
        elif cfg_use_vtracer:
            from pipeline.vtracer_convert import png_to_svg
            try:
                png_to_svg(str(source_png), str(svg_path))
            except Exception:
                tracer_used = "Inkscape (vtracer fallback)"
                from pipeline.vectorize import png_to_svg as inkscape_svg
                inkscape_svg(str(source_png), str(svg_path))
        else:
            from pipeline.vectorize import png_to_svg
            png_to_svg(str(source_png), str(svg_path))

        # ── Step 5: SVG → EPS ─────────────────────────────────────────────────
        eps_path = job_dir / f"{stem}.eps"
        from pipeline.vectorize import svg_to_eps
        svg_to_eps(str(svg_path), str(eps_path))

        # ── Build response ────────────────────────────────────────────────────
        elapsed    = round(time.time() - start_time, 1)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=FILE_TTL_MINUTES)

        files: dict[str, str] = {}
        if not cfg_no_enhance:
            files["enhanced_png"] = f"/files/{job_id}/{stem}_enhanced.png"
        if cfg_remove_bg:
            files["nobg_png"] = f"/files/{job_id}/{stem}_nobg.png"
        files["svg"] = f"/files/{job_id}/{stem}.svg"
        files["eps"] = f"/files/{job_id}/{stem}.eps"

        return {
            "job_id": job_id,
            "status": "done",
            "files": files,
            "meta": {
                "stem":           stem,
                "tracer":         tracer_used,
                "call_agent":     cfg_call_agent,
                "use_realesrgan": cfg_use_realesrgan,
                "elapsed_sec":    elapsed,
            },
            "expires_at": expires_at.isoformat(),
        }

    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── API 2: GET /files/{job_id}/{filename} ─────────────────────────────────────

@app.get("/files/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    # Block path traversal
    if any(c in filename for c in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    job_dir = OUTPUT_ROOT / job_id
    if not job_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Files not found or have expired. Re-submit the image to regenerate.",
        )

    if _is_expired(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=404,
            detail="Files have expired. Re-submit the image to regenerate.",
        )

    file_path = job_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in this job.")

    media_type = MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )
