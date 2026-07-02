from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gpu_runner import GpuRunner, GpuRunProgress, GpuRunRequest, GpuRunResult, runner_from_env, safe_slug

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_ROOT = Path(os.environ.get("PICKLEBALL_UPLOAD_ROOT", "/tmp/pickleball_render_uploads"))
DEFAULT_STATIC_DIR = ROOT / "web" / "replay" / "dist"

PIPELINE_STEPS: tuple[tuple[str, str, int], ...] = (
    ("queued", "Queued", 0),
    ("uploaded", "Inputs saved", 8),
    ("gpu_prepare", "GPU setup", 12),
    ("gpu_upload", "Input transfer", 20),
    ("gpu_pipeline", "GPU pipeline", 36),
    ("artifacts", "Artifact sync", 88),
    ("ready", "Replay ready", 100),
)


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def create(self, *, clip: str, video_name: str) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        now = time.time()
        payload: dict[str, Any] = {
            "id": job_id,
            "clip": clip,
            "video_name": video_name,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "progress": _progress_payload(
                GpuRunProgress(percent=0, stage="Queued", message="Waiting for the GPU worker."),
                status="queued",
                updated_at=now,
            ),
            "result": None,
            "error": None,
            "links": {
                "status": f"/api/jobs/{job_id}",
                "manifest": f"/api/jobs/{job_id}/manifest",
            },
        }
        self._write(job_id, payload)
        return payload

    def get(self, job_id: str) -> dict[str, Any]:
        safe_slug(job_id)
        path = self._job_path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return _with_dynamic_eta(payload)

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            payload = self.get(job_id)
            payload.update(changes)
            payload["updated_at"] = time.time()
            self._write(job_id, payload)
            return payload

    def _write(self, job_id: str, payload: dict[str, Any]) -> None:
        import json

        job_dir = self.root / safe_slug(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._job_path(job_id).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _job_path(self, job_id: str) -> Path:
        return self.root / safe_slug(job_id) / "job.json"


def create_app(
    *,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
    runner: GpuRunner | None = None,
    run_jobs_inline: bool = False,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> FastAPI:
    app = FastAPI(title="Pickleball Render Gateway")
    store = JobStore(upload_root)
    gpu_runner = runner if runner is not None else runner_from_env()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "runner": gpu_runner.describe()}

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        clip: str | None = Form(default=None),
        max_frames: int | None = Form(default=None),
        capture_sidecar: UploadFile | None = File(default=None),
        court_corners: UploadFile | None = File(default=None),
        court_calibration: UploadFile | None = File(default=None),
    ) -> dict[str, Any]:
        try:
            clip_id = _clip_id_from_upload(clip, video.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        job = store.create(clip=clip_id, video_name=video.filename or "video")
        job_dir = upload_root / job["id"]
        input_dir = job_dir / "input"
        artifacts_dir = job_dir / "artifacts"
        input_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        video_path = input_dir / _safe_upload_filename(video.filename, fallback=f"{clip_id}.mp4")
        await _save_upload(video, video_path)
        capture_sidecar_path = await _save_optional_upload(capture_sidecar, input_dir, "capture_sidecar.json")
        court_corners_path = await _save_optional_upload(court_corners, input_dir, "court_corners.json")
        court_calibration_path = await _save_optional_upload(court_calibration, input_dir, "court_calibration.json")
        store.update(
            job["id"],
            progress=_progress_payload(
                GpuRunProgress(percent=8, stage="Inputs saved", message="Upload received by Render."),
                status="queued",
            ),
        )

        request = GpuRunRequest(
            job_id=job["id"],
            clip=clip_id,
            input_dir=input_dir,
            video_path=video_path,
            artifacts_dir=artifacts_dir,
            capture_sidecar_path=capture_sidecar_path,
            court_corners_path=court_corners_path,
            court_calibration_path=court_calibration_path,
            max_frames=max_frames,
        )
        if run_jobs_inline:
            _execute_job(store, gpu_runner, request)
        else:
            background_tasks.add_task(_execute_job, store, gpu_runner, request)
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="job not found") from None

    @app.get("/api/jobs/{job_id}/manifest")
    def get_manifest(job_id: str) -> FileResponse:
        try:
            store.get(job_id)
            manifest = _job_artifacts_dir(upload_root, job_id) / "replay_viewer_manifest.json"
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="job not found") from None
        if not manifest.is_file():
            raise HTTPException(status_code=404, detail="manifest not ready")
        return FileResponse(manifest, media_type="application/json")

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
    def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
        try:
            store.get(job_id)
            safe_path = _safe_artifact_path(_job_artifacts_dir(upload_root, job_id), artifact_path)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="artifact not found") from None
        if not safe_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(safe_path)

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="site")

    return app


def _execute_job(store: JobStore, runner: GpuRunner, request: GpuRunRequest) -> None:
    started_at = time.time()
    estimated_total_seconds = _estimated_total_seconds(request)

    def handle_progress(progress: GpuRunProgress) -> None:
        store.update(
            request.job_id,
            progress=_progress_payload(
                progress,
                status="running",
                started_at=started_at,
                estimated_total_seconds=estimated_total_seconds,
            ),
        )

    store.update(
        request.job_id,
        status="running",
        error=None,
        progress=_progress_payload(
            GpuRunProgress(
                percent=10,
                stage="Starting GPU job",
                message="Render is handing the video package to the GPU runner.",
            ),
            status="running",
            started_at=started_at,
            estimated_total_seconds=estimated_total_seconds,
        ),
    )
    tracked_request = replace(request, progress_callback=handle_progress)
    try:
        result = runner.run(tracked_request)
    except Exception as exc:  # noqa: BLE001 - job failures are API state, not process crashes
        current = store.get(request.job_id)
        current_percent = int(current.get("progress", {}).get("percent", 10))
        store.update(
            request.job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            result=None,
            progress=_progress_payload(
                GpuRunProgress(percent=current_percent, stage="Failed", message=str(exc), eta_seconds=None),
                status="failed",
                started_at=started_at,
                completed_at=time.time(),
            ),
        )
        return

    final_progress = (
        GpuRunProgress(percent=100, stage="Replay ready", message="Replay artifacts are ready.", eta_seconds=0)
        if result.status == "complete"
        else GpuRunProgress(percent=55, stage="Submitted", message="GPU worker accepted the job.")
    )
    store.update(
        request.job_id,
        status=result.status,
        error=None,
        result=_result_payload(request.job_id, result),
        progress=_progress_payload(
            final_progress,
            status=result.status,
            started_at=started_at,
            completed_at=time.time() if result.status in {"complete", "failed"} else None,
        ),
    )


def _result_payload(job_id: str, result: GpuRunResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "notes": result.notes,
        "remote_run_dir": result.remote_run_dir,
        "manifest_url": f"/api/jobs/{job_id}/manifest" if result.manifest_path else None,
    }
    if result.raw:
        payload["raw"] = result.raw
    return payload


def _estimated_total_seconds(request: GpuRunRequest) -> int:
    override = os.environ.get("PICKLEBALL_GPU_DEFAULT_ETA_SECONDS", "").strip()
    if override:
        try:
            return max(30, int(override))
        except ValueError:
            pass
    if request.max_frames is not None:
        return max(45, min(900, request.max_frames * 8))
    try:
        size_mb = max(1.0, request.video_path.stat().st_size / (1024 * 1024))
    except OSError:
        size_mb = 100.0
    return int(max(180, min(5400, size_mb * 18)))


def _progress_payload(
    progress: GpuRunProgress,
    *,
    status: str,
    updated_at: float | None = None,
    started_at: float | None = None,
    estimated_total_seconds: int | None = None,
    completed_at: float | None = None,
) -> dict[str, Any]:
    now = time.time() if updated_at is None else updated_at
    percent = max(0, min(100, int(progress.percent)))
    payload: dict[str, Any] = {
        "percent": percent,
        "stage": progress.stage,
        "message": progress.message,
        "eta_seconds": progress.eta_seconds,
        "updated_at": now,
        "started_at": started_at,
        "completed_at": completed_at,
        "steps": _progress_steps(status=status, percent=percent),
    }
    if estimated_total_seconds is not None:
        payload["estimated_total_seconds"] = estimated_total_seconds
        if status in {"queued", "running", "submitted"} and started_at is not None:
            payload["eta_seconds"] = max(0, int(estimated_total_seconds - max(0.0, now - started_at)))
    if status == "complete":
        payload["eta_seconds"] = 0
    if status == "failed":
        payload["eta_seconds"] = None
    return payload


def _progress_steps(*, status: str, percent: int) -> list[dict[str, str]]:
    if status == "complete":
        return [{"id": step_id, "label": label, "status": "complete"} for step_id, label, _ in PIPELINE_STEPS]

    active_index = 0
    for index, (_, _, threshold) in enumerate(PIPELINE_STEPS):
        if percent >= threshold:
            active_index = index

    steps: list[dict[str, str]] = []
    for index, (step_id, label, _) in enumerate(PIPELINE_STEPS):
        if status == "failed" and index == active_index:
            step_status = "failed"
        elif index < active_index:
            step_status = "complete"
        elif index == active_index:
            step_status = "active"
        else:
            step_status = "pending"
        steps.append({"id": step_id, "label": label, "status": step_status})
    return steps


def _with_dynamic_eta(payload: dict[str, Any]) -> dict[str, Any]:
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        return payload
    if payload.get("status") not in {"queued", "running", "submitted"}:
        return payload
    started_at = progress.get("started_at")
    estimated_total_seconds = progress.get("estimated_total_seconds")
    if not isinstance(started_at, (int, float)) or not isinstance(estimated_total_seconds, (int, float)):
        return payload

    updated_payload = dict(payload)
    updated_progress = dict(progress)
    updated_progress["eta_seconds"] = max(0, int(estimated_total_seconds - max(0.0, time.time() - started_at)))
    updated_payload["progress"] = updated_progress
    return updated_payload


def _clip_id_from_upload(clip: str | None, filename: str | None) -> str:
    if clip and clip.strip():
        return safe_slug(clip.strip())
    fallback = Path(filename or "clip").stem or "clip"
    return safe_slug(_normalize_slug(fallback))


def _normalize_slug(value: str) -> str:
    normalized = re_sub_unsafe(value)
    return normalized if normalized else "clip"


def re_sub_unsafe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def _safe_upload_filename(filename: str | None, *, fallback: str) -> str:
    raw = Path(filename or fallback).name
    stem = re_sub_unsafe(Path(raw).stem) or Path(fallback).stem
    suffix = Path(raw).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".json"}:
        suffix = Path(fallback).suffix or ".mp4"
    return f"{stem}{suffix}"


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


async def _save_optional_upload(upload: UploadFile | None, input_dir: Path, filename: str) -> Path | None:
    if upload is None:
        return None
    path = input_dir / filename
    await _save_upload(upload, path)
    return path


def _job_artifacts_dir(upload_root: Path, job_id: str) -> Path:
    return upload_root / safe_slug(job_id) / "artifacts"


def _safe_artifact_path(root: Path, artifact_path: str) -> Path:
    candidate = (root / artifact_path).resolve()
    root_resolved = root.resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise ValueError("unsafe artifact path")
    return candidate


app = create_app()
