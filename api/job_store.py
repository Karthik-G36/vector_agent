"""
In-memory job store.
Each job tracks status, output paths, token usage, and the webhook callback URL.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    callback_url: str
    svg_path: Optional[str] = None
    eps_path: Optional[str] = None
    error: Optional[str] = None
    token_usage: Optional[dict] = field(default=None)


_jobs: dict[str, Job] = {}


def create_job(callback_url: str) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, status=JobStatus.QUEUED, callback_url=callback_url)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def update_job(job: Job) -> None:
    _jobs[job.job_id] = job
