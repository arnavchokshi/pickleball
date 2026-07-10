"""Worker routes (INFRA-2): pull-based job queue served to the GPU worker
daemon (`server/worker/daemon.py`).

Bearer-token gated on `PICKLEBALL_WORKER_BEARER_TOKEN` -- a SEPARATE
machine-principal credential from the user-facing JWT auth in
`server/security.py` (`require_user`). The worker never authenticates as a
user and never sees a JWT.

Mounted whenever accounts are enabled (independent of
`PICKLEBALL_QUEUE_ENABLED`, which only controls whether `POST /api/jobs`
writes a queued doc instead of executing inline) so the queue surface exists
ahead of the cutover flip.

`_reclaim_stale_jobs` (risk #5 in the approved design: stale jobs must not
go un-swept when zero workers are polling) is shared with
`server/routes/jobs_v2.py`, which imports it directly from this module and
calls it from `get_job_v2` so job status stays honest even without a worker
loop running.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel
from pymongo import ReturnDocument

from ..bundle_policy import gate_reported_status

STALE_HEARTBEAT_S = 300
MAX_ATTEMPTS_BEFORE_FAIL = 2
CLAIM_POLL_INTERVAL_S = 2
DEFAULT_WAIT_S = 25

WithDynamicEta = Callable[[dict[str, Any]], dict[str, Any]]

_PRIVATE_JOB_FIELDS = ("_id", "user_id")


class HeartbeatBody(BaseModel):
    stage: str
    percent: int
    message: str = ""


class CompleteBody(BaseModel):
    # ``succeeded`` remains accepted during rolling deploys, but the policy
    # gate below can only downgrade that legacy execution result to partial.
    status: Literal["complete", "partial", "failed", "succeeded"]
    error: str | None = None
    pipeline_stage_summary: list[dict[str, Any]] | None = None
    s3_artifacts_prefix: str | None = None
    s3_bundle_prefix: str | None = None
    missing_capabilities: list[Any] | None = None
    trust_bands: dict[str, Any] | None = None
    bundle_policy: dict[str, Any] | None = None


def _now() -> float:
    return time.time()


def _public_job(doc: dict[str, Any]) -> dict[str, Any]:
    payload = dict(doc)
    for field in _PRIVATE_JOB_FIELDS:
        payload.pop(field, None)
    return payload


def _reclaim_stale_jobs(db: Any, *, now: float | None = None) -> int:
    """Requeue (attempts < 2) or fail (attempts >= 2, progress preserved)
    any `claimed`/`running` job whose heartbeat has gone stale.

    Runs inline from both `GET /api/worker/next-job` (before claiming) and
    the user-facing `GET /api/jobs/{id}` -- no cron, no separate sweeper
    process -- so a crashed/preempted worker is detected the moment ANYONE
    asks about job state. Returns the count of docs mutated.
    """
    now = _now() if now is None else now
    cutoff = now - STALE_HEARTBEAT_S
    stuck = list(
        db.jobs.find({"status": {"$in": ["claimed", "running"]}, "heartbeat_at": {"$lt": cutoff}})
    )
    for doc in stuck:
        attempts = int(doc.get("attempts", 0))
        if attempts < MAX_ATTEMPTS_BEFORE_FAIL:
            db.jobs.update_one(
                {"_id": doc["_id"]},
                {"$set": {"status": "queued", "worker_id": None, "updated_at": now}},
            )
        else:
            # `progress` is intentionally left untouched here: the caller
            # asked us to preserve the last-seen progress on a hard failure.
            db.jobs.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "status": "failed",
                        "error": "worker lost (stale heartbeat)",
                        "updated_at": now,
                    }
                },
            )
    return len(stuck)


def _claim_next_queued_job(db: Any, *, worker_id: str, now: float | None = None) -> dict[str, Any] | None:
    now = _now() if now is None else now
    return db.jobs.find_one_and_update(
        {"status": "queued"},
        {
            "$set": {
                "status": "claimed",
                "worker_id": worker_id,
                "heartbeat_at": now,
                "claimed_at": now,
            },
            "$inc": {"attempts": 1},
        },
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


def build_worker_router(*, db: Any, worker_token: str, with_dynamic_eta: WithDynamicEta) -> APIRouter:
    router = APIRouter()
    expected_authorization = f"Bearer {worker_token}" if worker_token else None

    def require_worker(authorization: str | None = Header(default=None)) -> str:
        if expected_authorization is None or authorization != expected_authorization:
            raise HTTPException(status_code=403, detail="invalid worker bearer token")
        return "worker"

    @router.get("/api/worker/next-job", response_model=None)
    def next_job(
        wait_s: int = DEFAULT_WAIT_S,
        worker_id: str = Header(default="unknown-worker", alias="X-Worker-Id"),
        _: str = Depends(require_worker),
    ) -> Any:
        _reclaim_stale_jobs(db)
        deadline = _now() + max(0, wait_s)
        while True:
            claimed = _claim_next_queued_job(db, worker_id=worker_id)
            if claimed is not None:
                return {
                    "job_id": claimed["_id"],
                    "clip_id": claimed["clip_id"],
                    "s3_raw_key": claimed["s3"]["video_key"],
                    "s3_sidecar_key": claimed["s3"].get("sidecar_key"),
                    "video_filename": claimed["video_name"],
                    "max_frames": claimed.get("max_frames"),
                    "attempts": claimed["attempts"],
                }
            if _now() >= deadline:
                return Response(status_code=204)
            time.sleep(CLAIM_POLL_INTERVAL_S)

    @router.post("/api/worker/jobs/{job_id}/heartbeat", status_code=204, response_model=None)
    def heartbeat(job_id: str, body: HeartbeatBody, _: str = Depends(require_worker)) -> Response:
        doc = db.jobs.find_one({"_id": job_id})
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        now = _now()
        progress = dict(doc.get("progress") or {})
        progress.update(
            {
                "percent": max(0, min(100, int(body.percent))),
                "stage": body.stage,
                "message": body.message,
                "updated_at": now,
            }
        )
        db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "running", "heartbeat_at": now, "progress": progress, "updated_at": now}},
        )
        return Response(status_code=204)

    @router.post("/api/worker/jobs/{job_id}/complete")
    def complete(job_id: str, body: CompleteBody, _: str = Depends(require_worker)) -> dict[str, Any]:
        doc = db.jobs.find_one({"_id": job_id})
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        now = _now()
        # NS-01.5: the worker's bundle status is the job status. Never infer
        # completion from process success or translate partial into ready.
        job_status = gate_reported_status(
            status=body.status,
            missing_capabilities=body.missing_capabilities,
            trust_bands=body.trust_bands,
            bundle_policy=body.bundle_policy,
        )
        progress = dict(doc.get("progress") or {})
        if job_status == "complete":
            progress.update(
                {
                    "percent": 100,
                    "stage": "Replay ready",
                    "message": "Replay artifacts are ready.",
                    "eta_seconds": 0,
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            result: dict[str, Any] | None = {
                "manifest_url": f"/api/jobs/{job_id}/manifest",
                "s3_artifacts_prefix": body.s3_artifacts_prefix,
                "s3_bundle_prefix": body.s3_bundle_prefix,
                "pipeline_stage_summary": body.pipeline_stage_summary or [],
                "missing_capabilities": body.missing_capabilities or [],
                "trust_bands": body.trust_bands or {},
                "bundle_policy": body.bundle_policy,
            }
        elif job_status == "partial":
            progress.update(
                {
                    "percent": 100,
                    "stage": "Partial result",
                    "message": "Replay is inspectable with explicitly missing capabilities.",
                    "eta_seconds": 0,
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            result = {
                "manifest_url": f"/api/jobs/{job_id}/manifest",
                "s3_artifacts_prefix": body.s3_artifacts_prefix,
                "s3_bundle_prefix": body.s3_bundle_prefix,
                "pipeline_stage_summary": body.pipeline_stage_summary or [],
                "missing_capabilities": body.missing_capabilities or [],
                "trust_bands": body.trust_bands or {},
                "bundle_policy": body.bundle_policy,
            }
        else:
            progress.update(
                {
                    "stage": "Failed",
                    "message": body.error or "worker reported failure",
                    "eta_seconds": None,
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            result = None
        db.jobs.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": job_status,
                    "error": body.error,
                    "result": result,
                    "missing_capabilities": body.missing_capabilities or [],
                    "trust_bands": body.trust_bands or {},
                    "progress": progress,
                    "updated_at": now,
                }
            },
        )
        updated = db.jobs.find_one({"_id": job_id})
        return with_dynamic_eta(_public_job(updated))

    return router
