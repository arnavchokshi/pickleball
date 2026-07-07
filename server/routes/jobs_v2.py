"""Account-era job routes (INFRA-1): Mongo-backed job records over the
existing GPU runner.

The pull-worker queue is INFRA-2; in THIS lane jobs still execute via the
injected `GpuRunner` in FastAPI `BackgroundTasks`, but the job document lives
in Mongo (mirroring the legacy JSON job shape, ETA logic included) so the
record survives restarts and is owner-scoped. Inputs are staged by downloading
the clip's raw bytes from S3 into the same `upload_root` layout the legacy
path uses, so `_execute_job` (injected from `server.render_app`) runs
unchanged.

Queue-era fields (`attempts`, `worker_id`, `heartbeat_at`) are written now so
INFRA-2's claim/heartbeat logic lands on documents that already carry them.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from ..gpu_runner import GpuRunner, GpuRunProgress, GpuRunRequest
from ..security import AuthConfig, bearer_auth_dependency

ProgressPayload = Callable[..., dict[str, Any]]
ExecuteJob = Callable[[Any, GpuRunner, GpuRunRequest], None]
WithDynamicEta = Callable[[dict[str, Any]], dict[str, Any]]

_PRIVATE_JOB_FIELDS = ("_id", "user_id")


class CreateJobBody(BaseModel):
    clip_id: str
    max_frames: int | None = None


def _public_job(doc: dict[str, Any]) -> dict[str, Any]:
    payload = dict(doc)
    for field in _PRIVATE_JOB_FIELDS:
        payload.pop(field, None)
    return payload


class MongoJobStore:
    """Duck-type of `server.render_app.JobStore` over a Mongo collection, so
    the injected `_execute_job` drives Mongo documents unchanged."""

    def __init__(self, collection: Any, with_dynamic_eta: WithDynamicEta) -> None:
        self._collection = collection
        self._with_dynamic_eta = with_dynamic_eta

    def get(self, job_id: str) -> dict[str, Any]:
        doc = self._collection.find_one({"_id": job_id})
        if doc is None:
            raise KeyError(job_id)
        return self._with_dynamic_eta(_public_job(doc))

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        changes["updated_at"] = time.time()
        self._collection.update_one({"_id": job_id}, {"$set": changes})
        return self.get(job_id)


def build_jobs_v2_router(
    *,
    db: Any,
    s3_client: Any,
    bucket: str,
    auth_config: AuthConfig,
    runner: GpuRunner,
    upload_root: Path,
    run_jobs_inline: bool,
    execute_job: ExecuteJob,
    progress_payload: ProgressPayload,
    with_dynamic_eta: WithDynamicEta,
) -> APIRouter:
    router = APIRouter()
    require_user = bearer_auth_dependency(auth_config)
    store = MongoJobStore(db.jobs, with_dynamic_eta)

    def _download_to(key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        with path.open("wb") as handle:
            for chunk in iter(lambda: response["Body"].read(1024 * 1024), b""):
                handle.write(chunk)

    def _stage_inputs_and_execute(
        *,
        job_id: str,
        clip_slug: str,
        video_filename: str,
        video_key: str,
        sidecar_key: str | None,
        max_frames: int | None,
    ) -> None:
        job_dir = upload_root / job_id
        input_dir = job_dir / "input"
        artifacts_dir = job_dir / "artifacts"
        input_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        video_path = input_dir / video_filename
        sidecar_path: Path | None = None
        try:
            _download_to(video_key, video_path)
            if sidecar_key:
                candidate = input_dir / "capture_sidecar.json"
                try:
                    _download_to(sidecar_key, candidate)
                    sidecar_path = candidate
                except Exception:  # noqa: BLE001 - sidecar is optional; absence is normal
                    sidecar_path = None
        except Exception as exc:  # noqa: BLE001 - download failures are job state, not crashes
            store.update(
                job_id,
                status="failed",
                error=f"input download failed: {type(exc).__name__}: {exc}",
                result=None,
                progress=progress_payload(
                    GpuRunProgress(percent=8, stage="Failed", message=str(exc), eta_seconds=None),
                    status="failed",
                    completed_at=time.time(),
                ),
            )
            return
        store.update(
            job_id,
            progress=progress_payload(
                GpuRunProgress(percent=8, stage="Inputs saved", message="Raw clip pulled from S3."),
                status="queued",
            ),
        )
        request = GpuRunRequest(
            job_id=job_id,
            clip=clip_slug,
            input_dir=input_dir,
            video_path=video_path,
            artifacts_dir=artifacts_dir,
            capture_sidecar_path=sidecar_path,
            max_frames=max_frames,
        )
        execute_job(store, runner, request)

    @router.post("/api/jobs", status_code=202)
    def create_job_v2(
        background_tasks: BackgroundTasks,
        body: CreateJobBody,
        user_id: str = Depends(require_user),
    ) -> dict[str, Any]:
        clip = db.clips.find_one({"_id": body.clip_id, "user_id": user_id})
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        now = time.time()
        job_doc: dict[str, Any] = {
            "_id": job_id,
            "id": job_id,
            "user_id": user_id,
            "clip_id": clip["_id"],
            "clip": clip.get("slug") or "clip",
            "video_name": clip["filename"],
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "progress": progress_payload(
                GpuRunProgress(percent=0, stage="Queued", message="Waiting for the GPU worker."),
                status="queued",
                updated_at=now,
            ),
            "result": None,
            "error": None,
            "attempts": 0,
            "worker_id": None,
            "heartbeat_at": None,
            "max_frames": body.max_frames,
            "s3": {"video_key": clip["video_key"], "sidecar_key": clip.get("sidecar_key")},
            "links": {
                "status": f"/api/jobs/{job_id}",
                "manifest": f"/api/jobs/{job_id}/manifest",
            },
        }
        db.jobs.insert_one(job_doc)
        db.clips.update_one(
            {"_id": clip["_id"]},
            {"$set": {"job_id": job_id, "updated_at": datetime.now(timezone.utc)}},
        )
        kwargs = {
            "job_id": job_id,
            "clip_slug": job_doc["clip"],
            "video_filename": clip["filename"],
            "video_key": clip["video_key"],
            "sidecar_key": clip.get("sidecar_key"),
            "max_frames": body.max_frames,
        }
        if run_jobs_inline:
            _stage_inputs_and_execute(**kwargs)
        else:
            background_tasks.add_task(_stage_inputs_and_execute, **kwargs)
        return _public_job(job_doc)

    @router.get("/api/jobs/{job_id}")
    def get_job_v2(job_id: str, user_id: str = Depends(require_user)) -> dict[str, Any]:
        doc = db.jobs.find_one({"_id": job_id, "user_id": user_id})
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        return with_dynamic_eta(_public_job(doc))

    return router
