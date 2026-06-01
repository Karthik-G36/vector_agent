"""
Background job runner.

Offloads the blocking pipeline to a thread pool so the FastAPI event loop
stays free. After the pipeline finishes (success or failure), fires the
webhook to the client's callback_url.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.job_store import Job, JobStatus, update_job
from api.webhook import fire
from pipeline.run import PipelineConfig, run_pipeline
from pipeline.token_tracker import tracker

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _run_job_sync(job: Job, img_path: Path, out_dir: Path, cfg: PipelineConfig) -> None:
    """Runs in a thread — all blocking pipeline code lives here."""
    job.status = JobStatus.PROCESSING
    update_job(job)

    token_snapshot = tracker.count()
    try:
        result = run_pipeline(img_path, out_dir, cfg)
        job.status = JobStatus.DONE
        job.svg_path = result["svg"]
        job.eps_path = result["eps"]
        job.token_usage = tracker.as_dict(token_snapshot)
        tracker.summary()
    except Exception as exc:
        logger.exception(f"[worker] job {job.job_id} failed")
        job.status = JobStatus.FAILED
        job.error = str(exc)
    finally:
        update_job(job)


async def submit_job(
    job: Job,
    img_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> None:
    """Async entry point — called as a FastAPI BackgroundTask."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _run_job_sync, job, img_path, out_dir, cfg)

    payload: dict = {"job_id": job.job_id, "status": job.status.value}

    if job.status == JobStatus.DONE:
        svg_name = Path(job.svg_path).name
        eps_name = Path(job.eps_path).name
        payload["svg_url"] = f"/result/{job.job_id}/{svg_name}"
        payload["eps_url"] = f"/result/{job.job_id}/{eps_name}"
        if job.token_usage:
            payload["token_usage"] = job.token_usage
    else:
        payload["error"] = job.error

    await fire(job.callback_url, payload)
