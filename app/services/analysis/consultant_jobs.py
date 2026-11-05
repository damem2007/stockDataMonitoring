"""Background job runner for the AI investment consultant (Market Explorer).

The consultant flow is long-running (5-year evidence for up to ten
instruments, then provider synthesis). Running it as a job lets the frontend
show real stage progress, per-instrument evidence as it completes, the
provider attempt/outcome, and the final plan, instead of a simulated loader.
Jobs are in-process and short-lived; the durable copy of a result is the saved
analysis the user chooses to keep.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.services.analysis.market_explorer import PROCESS_STEPS, market_explorer_recommendation_payload

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 60
STAGE_IDS = [step["id"] for step in PROCESS_STEPS]
STAGE_PROGRESS: dict[str, tuple[float, float]] = {
    "profile": (0.0, 4.0),
    "universe": (4.0, 10.0),
    "evidence": (10.0, 72.0),
    "prediction": (72.0, 88.0),
    "alternatives": (88.0, 92.0),
    "allocation": (92.0, 97.0),
    "monitor": (97.0, 100.0),
}


@dataclass
class ConsultantJob:
    id: str
    user_id: str | None
    payload: dict[str, Any]
    status: str = "queued"
    stage: str = "profile"
    progress: float = 0.0
    message: str = "Queued for analysis."
    events: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    provider: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_JOBS: dict[str, ConsultantJob] = {}
_LOCK = threading.Lock()


def start_consultant_job(payload: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    """Create a job and run the consultant flow on a worker thread."""

    _prune_locked()
    job = ConsultantJob(id=uuid.uuid4().hex, user_id=str(user_id) if user_id else None, payload=dict(payload or {}))
    with _LOCK:
        _JOBS[job.id] = job
    thread = threading.Thread(target=_run_job, args=(job.id,), name=f"consultant-job-{job.id[:8]}", daemon=True)
    thread.start()
    return job_snapshot(job.id, user_id)


def job_snapshot(job_id: str, user_id: str | None = None, include_result: bool = True) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job.user_id and user_id and job.user_id != str(user_id):
            return None
        return _snapshot(job, include_result=include_result)


def _run_job(job_id: str) -> None:
    job = _JOBS.get(job_id)
    if job is None:
        return
    _update(job, status="running", stage="profile", message="Starting the investor discovery analysis...")
    started = time.monotonic()
    try:
        result = market_explorer_recommendation_payload(job.payload, on_progress=lambda event: _handle_event(job, event))
    except Exception as exc:  # pragma: no cover - surfaced to the client as a failed job
        logger.exception("consultant_job outcome=failed job_id=%s", job.id)
        _update(job, status="failed", error=f"{type(exc).__name__}: {exc}", message="The analysis could not be completed.", finished=True)
        return
    with _LOCK:
        job.result = result
        job.provider = {
            **job.provider,
            "provider": result.get("provider"),
            "model": result.get("model"),
            "mode": result.get("mode"),
            "fallbackReason": (result.get("aiDisclosure") or {}).get("fallbackReason") if isinstance(result.get("aiDisclosure"), dict) else None,
        }
    _update(job, status="done", stage="monitor", progress=100.0, message=f"Analysis complete in {(time.monotonic() - started):.0f}s.", finished=True)
    logger.info("consultant_job outcome=done job_id=%s elapsed_s=%.1f provider=%s", job.id, time.monotonic() - started, result.get("provider"))


def _handle_event(job: ConsultantJob, event: dict[str, Any]) -> None:
    stage = str(event.get("stage") or job.stage)
    if stage not in STAGE_PROGRESS:
        stage = job.stage
    low, high = STAGE_PROGRESS[stage]
    progress = low
    total = event.get("total")
    index = event.get("index")
    if stage == "evidence" and isinstance(total, int) and total > 0 and isinstance(index, int):
        progress = low + (high - low) * min(max(index, 0), total) / total
    elif stage == "prediction":
        outcome = event.get("outcome")
        progress = low + (high - low) * (0.15 if outcome == "attempt" else 0.9 if outcome in {"ready", "fallback"} else 0.05)
    elif stage != "evidence":
        progress = high
    with _LOCK:
        job.stage = stage
        job.progress = max(job.progress, round(progress, 1))
        job.message = str(event.get("message") or job.message)
        job.updated_at = time.time()
        job.events.append({"at": datetime.now(timezone.utc).isoformat(), "stage": stage, "message": job.message})
        if len(job.events) > 40:
            del job.events[: len(job.events) - 40]
        candidate = event.get("candidate")
        if isinstance(candidate, dict) and candidate.get("symbol"):
            job.candidates = [row for row in job.candidates if row.get("symbol") != candidate.get("symbol")] + [candidate]
        if isinstance(event.get("candidates"), list):
            job.universe = [str(item) for item in event["candidates"]]
        if event.get("provider"):
            job.provider = {**job.provider, "provider": event.get("provider"), "model": event.get("model"), "outcome": event.get("outcome")}


def _update(job: ConsultantJob, *, status: str | None = None, stage: str | None = None, progress: float | None = None, message: str | None = None, error: str | None = None, finished: bool = False) -> None:
    with _LOCK:
        if status:
            job.status = status
        if stage:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if message:
            job.message = message
            job.events.append({"at": datetime.now(timezone.utc).isoformat(), "stage": job.stage, "message": message})
        if error:
            job.error = error
        job.updated_at = time.time()
        if finished:
            job.finished_at = time.time()


def _snapshot(job: ConsultantJob, include_result: bool) -> dict[str, Any]:
    stage_index = STAGE_IDS.index(job.stage) if job.stage in STAGE_IDS else 0
    stages = []
    for index, step in enumerate(PROCESS_STEPS):
        state = "done" if index < stage_index or job.status == "done" else "active" if index == stage_index and job.status in {"running", "queued"} else "failed" if index == stage_index and job.status == "failed" else "pending"
        stages.append({"id": step["id"], "title": step["title"], "detail": step["detail"], "state": state})
    return {
        "jobId": job.id,
        "status": job.status,
        "stage": job.stage,
        "stageIndex": stage_index,
        "stages": stages,
        "progress": job.progress,
        "message": job.message,
        "events": job.events[-12:],
        "candidates": sorted(job.candidates, key=lambda row: float(row.get("score") or 0), reverse=True),
        "universe": job.universe,
        "provider": job.provider,
        "error": job.error,
        "createdAt": datetime.fromtimestamp(job.created_at, tz=timezone.utc).isoformat(),
        "updatedAt": datetime.fromtimestamp(job.updated_at, tz=timezone.utc).isoformat(),
        "elapsedMs": round(((job.finished_at or time.time()) - job.created_at) * 1000),
        "result": job.result if include_result and job.status == "done" else None,
    }


def _prune_locked() -> None:
    now = time.time()
    with _LOCK:
        stale = [job_id for job_id, job in _JOBS.items() if now - job.created_at > JOB_TTL_SECONDS and job.status in {"done", "failed"}]
        for job_id in stale:
            _JOBS.pop(job_id, None)
        if len(_JOBS) > MAX_JOBS:
            for job_id in sorted(_JOBS, key=lambda key: _JOBS[key].created_at)[: len(_JOBS) - MAX_JOBS]:
                if _JOBS[job_id].status in {"done", "failed"}:
                    _JOBS.pop(job_id, None)
