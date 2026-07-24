"""Account-era job routes (INFRA-1/2): Mongo-backed job records over the
existing GPU runner, with an INFRA-2 pull-worker queue branch.

`PICKLEBALL_QUEUE_ENABLED=0` (default): jobs still execute via the injected
`GpuRunner` in FastAPI `BackgroundTasks` -- the INFRA-1 behavior, unchanged.
`=1`: `create_job_v2` inserts a `queued` Mongo doc and returns immediately;
`server/worker/daemon.py` (a separate process, possibly on a different
machine) claims it via `server/routes/worker.py` and drives it to
completion. Either way the job document lives in Mongo (mirroring the
legacy JSON job shape, ETA logic included) so the record survives restarts
and is owner-scoped. Inputs are staged by downloading the clip's raw bytes
from S3 into the same `upload_root` layout the legacy path uses (inline
path only -- the queue path downloads on the worker instead), so
`_execute_job` (injected from `server.render_app`) runs unchanged.

Queue-era fields (`attempts`, `worker_id`, `heartbeat_at`) were written from
INFRA-1 onward so INFRA-2's claim/heartbeat logic lands on documents that
already carry them. `get_job_v2` runs the SHARED stale-heartbeat sweep
(`_reclaim_stale_jobs`, defined in `server/routes/worker.py`) before every
read so status stays honest even when zero workers are polling.
"""

from __future__ import annotations

import json
import posixpath
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from ..gpu_runner import GpuRunner, GpuRunProgress, GpuRunRequest
from ..s3 import presign_get
from ..security import AuthConfig, bearer_auth_dependency
from .worker import _reclaim_stale_jobs

ProgressPayload = Callable[..., dict[str, Any]]
ExecuteJob = Callable[[Any, GpuRunner, GpuRunRequest], None]
WithDynamicEta = Callable[[dict[str, Any]], dict[str, Any]]

_PRIVATE_JOB_FIELDS = ("_id", "user_id")


class CreateJobBody(BaseModel):
    clip_id: str
    max_frames: int | None = None
    pipeline_preset: Literal["full", "court_skeletons"] = "court_skeletons"


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
    queue_enabled: bool,
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
        pipeline_preset: Literal["full", "court_skeletons"],
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
            pipeline_preset=pipeline_preset,
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
            "pipeline_preset": body.pipeline_preset,
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
        if queue_enabled:
            # INFRA-2: the pull-worker daemon claims this via
            # GET /api/worker/next-job -- nothing to schedule here.
            return _public_job(job_doc)

        kwargs = {
            "job_id": job_id,
            "clip_slug": job_doc["clip"],
            "video_filename": clip["filename"],
            "video_key": clip["video_key"],
            "sidecar_key": clip.get("sidecar_key"),
            "max_frames": body.max_frames,
            "pipeline_preset": body.pipeline_preset,
        }
        if run_jobs_inline:
            _stage_inputs_and_execute(**kwargs)
        else:
            background_tasks.add_task(_stage_inputs_and_execute, **kwargs)
        return _public_job(job_doc)

    @router.get("/api/jobs/{job_id}")
    def get_job_v2(job_id: str, user_id: str = Depends(require_user)) -> dict[str, Any]:
        _reclaim_stale_jobs(db)
        doc = db.jobs.find_one({"_id": job_id, "user_id": user_id})
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        return with_dynamic_eta(_public_job(doc))

    @router.get("/api/jobs/{job_id}/manifest", response_model=None)
    def get_job_manifest_v2(job_id: str, user_id: str = Depends(require_user)) -> JSONResponse:
        doc = _published_job(db, job_id=job_id, user_id=user_id)
        prefix = _bundle_prefix(doc)
        payload = _load_s3_json(s3_client, bucket=bucket, key=f"{prefix}replay_viewer_manifest.json")
        rewritten = _rewrite_bundle_urls(
            payload,
            job_id=job_id,
            bundle_prefix=prefix,
            document_relative="replay_viewer_manifest.json",
        )
        return JSONResponse(rewritten)

    @router.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}", response_model=None)
    def get_job_artifact_v2(
        job_id: str,
        artifact_path: str,
        user_id: str = Depends(require_user),
    ) -> JSONResponse | RedirectResponse:
        doc = _published_job(db, job_id=job_id, user_id=user_id)
        relative = _safe_bundle_relative_path(artifact_path)
        prefix = _bundle_prefix(doc)
        key = f"{prefix}{relative}"
        if PurePosixPath(relative).name in {"body_mesh_index.json", "replay_scene.json"}:
            payload = _load_s3_json(s3_client, bucket=bucket, key=key)
            return JSONResponse(
                _rewrite_bundle_urls(
                    payload,
                    job_id=job_id,
                    bundle_prefix=prefix,
                    document_relative=relative,
                )
            )
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - absent bundle object is API state
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return RedirectResponse(presign_get(s3_client, bucket=bucket, key=key), status_code=307)

    return router


def _published_job(db: Any, *, job_id: str, user_id: str) -> dict[str, Any]:
    doc = db.jobs.find_one({"_id": job_id, "user_id": user_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="job not found")
    if doc.get("status") not in {"complete", "partial"}:
        raise HTTPException(status_code=404, detail="manifest not ready")
    _bundle_prefix(doc)
    return doc


def _bundle_prefix(doc: dict[str, Any]) -> str:
    result = doc.get("result")
    prefix = result.get("s3_bundle_prefix") if isinstance(result, dict) else None
    if not isinstance(prefix, str) or not prefix.endswith("/"):
        raise HTTPException(status_code=404, detail="published bundle not found")
    return prefix


def _load_s3_json(s3_client: Any, *, bucket: str, key: str) -> dict[str, Any]:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        payload = json.loads(response["Body"].read())
    except Exception as exc:  # noqa: BLE001 - absent/invalid object is API state
        raise HTTPException(status_code=404, detail="bundle JSON not found") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="published bundle JSON must be an object")
    return payload


def _safe_bundle_relative_path(value: str) -> str:
    path = PurePosixPath(unquote(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HTTPException(status_code=404, detail="artifact not found")
    return path.as_posix()


def _rewrite_bundle_urls(
    value: Any,
    *,
    job_id: str,
    bundle_prefix: str,
    document_relative: str,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _rewrite_bundle_urls(
                child_value,
                job_id=job_id,
                bundle_prefix=bundle_prefix,
                document_relative=document_relative,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_bundle_urls(
                child,
                job_id=job_id,
                bundle_prefix=bundle_prefix,
                document_relative=document_relative,
                key=key,
            )
            for child in value
        ]
    if not (
        isinstance(value, str)
        and key is not None
        and (key == "url" or key.endswith("_url") or key == "court_glb")
    ):
        return value
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return value
    raw_path = unquote(parsed.path)
    if raw_path.startswith(bundle_prefix):
        relative = raw_path.removeprefix(bundle_prefix)
    elif raw_path.startswith("/"):
        return value
    else:
        relative = posixpath.normpath(
            posixpath.join(posixpath.dirname(document_relative), raw_path)
        )
    relative = _safe_bundle_relative_path(relative)
    return f"/api/jobs/{job_id}/artifacts/{relative}"
