"""
Webhook dispatcher with exponential-backoff retry.

Fires a POST to the client's callback_url with the job result payload.
Retries up to _MAX_RETRIES times on non-2xx responses or network errors.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_SECONDS = [2, 5, 10]  # delay before each retry attempt


async def fire(url: str, payload: dict) -> bool:
    """
    POST payload to url. Returns True if the server acknowledged (2xx).
    Retries up to _MAX_RETRIES times with backoff on failure.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
                logger.info(f"[webhook] retry {attempt}/{_MAX_RETRIES} in {delay}s → {url}")
                await asyncio.sleep(delay)
            try:
                r = await client.post(url, json=payload)
                if r.status_code < 300:
                    logger.info(f"[webhook] delivered to {url} (HTTP {r.status_code})")
                    return True
                logger.warning(f"[webhook] attempt {attempt + 1} got HTTP {r.status_code}")
            except Exception as exc:
                logger.warning(f"[webhook] attempt {attempt + 1} error: {exc}")

    logger.error(f"[webhook] all {_MAX_RETRIES + 1} attempts failed for {url}")
    return False
