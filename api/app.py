"""
Vector Agent — FastAPI application.

Endpoints:
  POST /vectorize              Accept image (base64 or URL) + callback_url, return job_id immediately.
  GET  /status/{job_id}        Fallback polling — returns current job status.
  GET  /result/{job_id}/{file} Download SVG or EPS output file.

The primary notification mechanism is the webhook: once the job finishes
(or fails), the server POSTs the result payload to the client's callback_url.
GET /status is kept as a fallback for clients that cannot receive webhooks.
"""
from __future__ import annotations

import base64
from pathlib import Path

import httpx
from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.job_store import JobStatus, create_job, get_job
from api.worker import submit_job
from pipeline.run import PipelineConfig

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

_MIME_TO_EXT: dict[str, str] = {
    "image/png":  ".png",
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/tiff": ".tiff",
    "image/gif":  ".gif",
}
SUPPORTED_EXTENSIONS = set(_MIME_TO_EXT.values()) | {".tif"}

app = FastAPI(
    title="Vector Agent API",
    description="Raster image → SVG + EPS via an async webhook pipeline.",
    version="1.0.0",
)


class VectorizeRequest(BaseModel):
    image: str = Field(
        ...,
        description=(
            "Either a base64 data URI ('data:image/png;base64,...') "
            "or a publicly reachable image URL."
        ),
    )
    callback_url: str = Field(..., description="URL the server will POST the result to")
    use_vtracer: bool = False
    use_vtracer_agent: bool = False
    use_esrgan: bool = False
    esrgan_tile: int = 512
    remove_bg: bool = False
    use_sam2: bool = False
    call_agent: bool = False
    no_enhance: bool = False


async def _resolve_image(image: str) -> tuple[bytes, str]:
    """
    Returns (raw_bytes, file_extension).

    Accepts:
      - Base64 data URI : "data:image/png;base64,iVBORw0..."
      - Image URL       : "https://example.com/logo.png"
    """
    if image.startswith("data:"):
        # data:<mime>;base64,<data>
        try:
            header, data = image.split(",", 1)
            mime = header.split(":")[1].split(";")[0].lower()
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="Malformed base64 data URI.")

        ext = _MIME_TO_EXT.get(mime)
        if not ext:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported image type '{mime}' in data URI. "
                       f"Supported: {', '.join(sorted(_MIME_TO_EXT))}",
            )
        try:
            img_bytes = base64.b64decode(data)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid base64 data in image field.")

        return img_bytes, ext

    # Treat as URL
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(image)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Image URL returned HTTP {exc.response.status_code}: {image}",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to fetch image URL: {exc}")

    # Determine extension from Content-Type header first, then URL path
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    ext = _MIME_TO_EXT.get(content_type)
    if not ext:
        url_ext = Path(image.split("?")[0]).suffix.lower()
        if url_ext in SUPPORTED_EXTENSIONS:
            ext = url_ext

    if not ext:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot determine image type from URL. "
                   f"Content-Type was '{content_type}'. Use a direct image URL or a base64 data URI.",
        )

    return response.content, ext


@app.post("/vectorize", status_code=202)
async def vectorize(background_tasks: BackgroundTasks, req: VectorizeRequest):
    """
    Accept an image (base64 data URI or URL) and a callback_url.
    Returns a job_id immediately (< 1 s).

    When the pipeline finishes, the server POSTs to callback_url:

        {
          "job_id": "...",
          "status": "done" | "failed",
          "svg_url": "/result/{job_id}/image.svg",
          "eps_url": "/result/{job_id}/image.eps",
          "token_usage": { "total_tokens": 3352, "breakdown": [...] },
          "error": "..."
        }
    """
    img_bytes, ext = await _resolve_image(req.image)

    job = create_job(req.callback_url)

    job_dir = JOBS_DIR / job.job_id
    job_dir.mkdir(parents=True)

    img_path = job_dir / f"image{ext}"
    img_path.write_bytes(img_bytes)

    out_dir = job_dir / "output"
    out_dir.mkdir()

    cfg = PipelineConfig(
        call_agent        = req.call_agent,
        use_vtracer       = req.use_vtracer,
        use_vtracer_agent = req.use_vtracer_agent,
        use_esrgan        = req.use_esrgan,
        esrgan_tile       = req.esrgan_tile,
        remove_bg         = req.remove_bg,
        use_sam2          = req.use_sam2,
        no_enhance        = req.no_enhance,
    )

    background_tasks.add_task(submit_job, job, img_path, out_dir, cfg)

    return {"job_id": job.job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    """
    Fallback polling endpoint. Returns the current job status.
    Use the webhook (callback_url) as the primary notification mechanism.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "error": job.error,
    }


@app.get("/result/{job_id}/{filename}")
async def result(job_id: str, filename: str):
    """Download an output file (SVG or EPS) by the URLs provided in the webhook payload."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job is '{job.status.value}', not done yet")

    safe_name = Path(filename).name
    file_path = JOBS_DIR / job_id / "output" / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "image/svg+xml" if safe_name.endswith(".svg") else "application/postscript"
    return FileResponse(str(file_path), media_type=media_type, filename=safe_name)
