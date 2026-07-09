"""Pull-worker daemon core (INFRA-2).

`run_once(api_client, s3_client, process_runner, config)` is the tested,
pure/injectable heart of this module: claim a job from the API, download
its raw inputs from S3, run the pipeline (via the injected `process_runner`
-- never a real subprocess in tests), and report back. `main()` wires up
the REAL collaborators (`HttpApiClient` over the render-service API,
`boto3.client("s3")`, `_real_process_runner` shelling out to the heavy
pipeline venv) and loops, exactly mirroring what `SshGpuRunner` does today
except the daemon *is* the GPU box instead of SSHing into one.

Usage: `python -m server.worker.daemon` (loop) or `--check-config` /
`--help` (exit 0, no network -- see `docs` note on `main()`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from threed.racketsport.best_stack import server_override_value

from .config import WorkerConfig, worker_config_from_env
from ..pipeline_invocation import (
    PIPELINE_SUMMARY_ARTIFACT,
    build_process_video_args,
    copy_source_video_artifact,
    remote_model_root,
    rewrite_manifest_urls,
)

HEARTBEAT_STAGE_STARTING = "Running pipeline on GPU"
PREEMPTION_FLAG_PATH = Path("/tmp/PREEMPTED")


def default_allow_auto_court_corners_preview() -> bool:
    return bool(server_override_value("allow_auto_court_corners_preview"))


@dataclass(frozen=True)
class WorkerJob:
    job_id: str
    clip_id: str
    s3_raw_key: str
    s3_sidecar_key: str | None
    video_filename: str
    max_frames: int | None
    attempts: int


@dataclass(frozen=True)
class RunResult:
    status: str  # "succeeded" | "failed"
    error: str | None = None
    pipeline_stage_summary: list[dict[str, Any]] | None = None


ProcessRunner = Callable[[WorkerJob, Path, "Path | None", Path], RunResult]


class ApiClient(Protocol):
    def claim_next_job(self) -> dict[str, Any] | None: ...

    def send_heartbeat(self, job_id: str, *, stage: str, percent: int, message: str) -> None: ...

    def complete_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        pipeline_stage_summary: list[dict[str, Any]] | None = None,
        s3_artifacts_prefix: str | None = None,
        s3_bundle_prefix: str | None = None,
    ) -> None: ...


class S3Client(Protocol):
    def download_file(self, bucket: str, key: str, filename: str) -> None: ...

    def upload_file(self, filename: str, bucket: str, key: str) -> None: ...


def run_once(
    api_client: ApiClient,
    s3_client: S3Client,
    process_runner: ProcessRunner,
    config: WorkerConfig,
) -> bool:
    """Claim and fully drive at most one job. Returns True iff a job was
    claimed (regardless of whether it ultimately succeeded or failed) --
    `main()` uses this to decide whether a `git pull --ff-only` is due.
    """
    job_payload = api_client.claim_next_job()
    if job_payload is None:
        return False

    job = WorkerJob(
        job_id=str(job_payload["job_id"]),
        clip_id=str(job_payload["clip_id"]),
        s3_raw_key=str(job_payload["s3_raw_key"]),
        s3_sidecar_key=(str(job_payload["s3_sidecar_key"]) if job_payload.get("s3_sidecar_key") else None),
        video_filename=str(job_payload["video_filename"]),
        max_frames=job_payload.get("max_frames"),
        attempts=int(job_payload.get("attempts", 1)),
    )

    job_dir = Path(config.work_dir) / job.job_id
    input_dir = job_dir / "input"
    out_dir = job_dir / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = input_dir / job.video_filename
    try:
        s3_client.download_file(config.s3_bucket, job.s3_raw_key, str(video_path))
    except Exception as exc:  # noqa: BLE001 - a bad/missing raw key is job state, not a daemon crash
        # Mirror the inline path (server/routes/jobs_v2.py): a failed input
        # download fails THIS job, it must not escape into main()'s loop and
        # crash the worker (which would block every other queued job).
        api_client.complete_job(
            job.job_id,
            status="failed",
            error=f"input download failed: {type(exc).__name__}: {exc}",
            pipeline_stage_summary=None,
            s3_artifacts_prefix=None,
            s3_bundle_prefix=None,
        )
        return True
    sidecar_path: Path | None = None
    if job.s3_sidecar_key:
        candidate = input_dir / "capture_sidecar.json"
        try:
            s3_client.download_file(config.s3_bucket, job.s3_sidecar_key, str(candidate))
            sidecar_path = candidate
        except Exception:  # noqa: BLE001 - sidecar is optional; absence is normal
            sidecar_path = None

    stop_heartbeat = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop_heartbeat.wait(config.heartbeat_interval_s):
            try:
                api_client.send_heartbeat(
                    job.job_id, stage=HEARTBEAT_STAGE_STARTING, percent=50, message="processing"
                )
            except Exception:  # noqa: BLE001 - a missed heartbeat must not kill the run
                pass

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    try:
        api_client.send_heartbeat(job.job_id, stage=HEARTBEAT_STAGE_STARTING, percent=15, message="starting process_video")
    except Exception:  # noqa: BLE001 - same as above
        pass

    try:
        result = process_runner(job, video_path, sidecar_path, out_dir)
    except Exception as exc:  # noqa: BLE001 - a crashing runner is job state, not a daemon crash
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=5)
        api_client.complete_job(
            job.job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            pipeline_stage_summary=None,
            s3_artifacts_prefix=None,
            s3_bundle_prefix=None,
        )
        return True

    stop_heartbeat.set()
    heartbeat_thread.join(timeout=5)

    if result.status != "succeeded":
        api_client.complete_job(
            job.job_id,
            status="failed",
            error=result.error or "pipeline reported failure",
            pipeline_stage_summary=result.pipeline_stage_summary,
            s3_artifacts_prefix=None,
            s3_bundle_prefix=None,
        )
        return True

    bundle_dir = out_dir / job.clip_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copy_source_video_artifact(video_path=video_path, artifacts_dir=bundle_dir)
    bundle_prefix = f"bundles/{job.clip_id}/"
    rewrite_manifest_urls(
        artifacts_dir=bundle_dir,
        video_path=video_path,
        resolve=lambda name, _prefix=bundle_prefix: f"{_prefix}{name}",
    )

    artifacts_prefix = f"artifacts/{job.job_id}/"
    _upload_dir(s3_client, out_dir, config.s3_bucket, artifacts_prefix)
    _upload_dir(s3_client, bundle_dir, config.s3_bucket, bundle_prefix)

    api_client.complete_job(
        job.job_id,
        status="succeeded",
        error=None,
        pipeline_stage_summary=result.pipeline_stage_summary,
        s3_artifacts_prefix=artifacts_prefix,
        s3_bundle_prefix=bundle_prefix,
    )
    return True


def _upload_dir(s3_client: S3Client, local_dir: Path, bucket: str, prefix: str) -> None:
    if not local_dir.is_dir():
        return
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        s3_client.upload_file(str(path), bucket, f"{prefix}{relative}")


def _load_pipeline_summary(clip_dir: Path) -> list[dict[str, Any]] | None:
    summary_path = clip_dir / PIPELINE_SUMMARY_ARTIFACT
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stages = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(stages, list):
        return None
    return [stage for stage in stages if isinstance(stage, dict)]


def _real_process_runner(
    job: WorkerJob,
    video_path: Path,
    sidecar_path: Path | None,
    out_dir: Path,
    *,
    config: WorkerConfig,
) -> RunResult:
    """Real implementation: shells out to the heavy pipeline venv exactly as
    `SshGpuRunner` does remotely today (monitor_process_resources wrapper,
    `--body-local --device cuda:0 --json`), except locally -- the worker
    *is* the GPU box. Not exercised by unit tests (no real subprocess/GPU
    per the lane fence); `run_once` takes this in as an injectable so tests
    supply a fake instead.
    """
    clip = job.clip_id
    process_args = build_process_video_args(
        python=config.pipeline_python,
        script="scripts/racketsport/process_video.py",
        video=str(video_path),
        out=str(out_dir),
        clip=clip,
        model_root=remote_model_root(config.pipeline_python),
        sidecar=str(sidecar_path) if sidecar_path is not None else None,
        max_frames=job.max_frames,
        allow_auto_court=default_allow_auto_court_corners_preview(),
    )
    clip_out_dir = out_dir / clip
    clip_out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = clip_out_dir / "gpu_resource_usage.json"
    monitor_args = [
        config.pipeline_python,
        str(Path(config.repo_dir) / "scripts" / "racketsport" / "monitor_process_resources.py"),
        "--out",
        str(telemetry_path),
        "--sample-interval",
        "5",
        "--",
        *process_args,
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = config.repo_dir + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    completed = subprocess.run(
        monitor_args,
        cwd=config.repo_dir or None,
        env=env,
        capture_output=True,
        text=True,
        timeout=config.command_timeout_s,
    )
    summary = _load_pipeline_summary(clip_out_dir)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return RunResult(
            status="failed",
            error=f"process_video exit {completed.returncode}: {detail}",
            pipeline_stage_summary=summary,
        )
    return RunResult(status="succeeded", pipeline_stage_summary=summary)


class HttpApiClient:
    """Real `ApiClient` over the render-service worker API (httpx)."""

    def __init__(self, *, base_url: str, token: str, worker_id: str, poll_wait_s: int, timeout_s: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "X-Worker-Id": worker_id}
        self._poll_wait_s = poll_wait_s
        self._timeout_s = timeout_s

    def claim_next_job(self) -> dict[str, Any] | None:
        import httpx

        response = httpx.get(
            f"{self._base_url}/api/worker/next-job",
            params={"wait_s": self._poll_wait_s},
            headers=self._headers,
            timeout=self._poll_wait_s + self._timeout_s,
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    def send_heartbeat(self, job_id: str, *, stage: str, percent: int, message: str) -> None:
        import httpx

        response = httpx.post(
            f"{self._base_url}/api/worker/jobs/{job_id}/heartbeat",
            json={"stage": stage, "percent": percent, "message": message},
            headers=self._headers,
            timeout=self._timeout_s,
        )
        response.raise_for_status()

    def complete_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        pipeline_stage_summary: list[dict[str, Any]] | None = None,
        s3_artifacts_prefix: str | None = None,
        s3_bundle_prefix: str | None = None,
    ) -> None:
        import httpx

        response = httpx.post(
            f"{self._base_url}/api/worker/jobs/{job_id}/complete",
            json={
                "status": status,
                "error": error,
                "pipeline_stage_summary": pipeline_stage_summary,
                "s3_artifacts_prefix": s3_artifacts_prefix,
                "s3_bundle_prefix": s3_bundle_prefix,
            },
            headers=self._headers,
            timeout=self._timeout_s,
        )
        response.raise_for_status()


def _git_pull_ff_only(repo_dir: str) -> None:
    if not repo_dir:
        return
    try:
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:  # noqa: BLE001 - a failed code refresh must not kill the loop
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.worker.daemon",
        description=(
            "Pull-based GPU worker daemon (INFRA-2): claims jobs from the "
            "render-service queue and runs scripts/racketsport/process_video.py."
        ),
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Resolve worker config from the environment and print it, then exit 0 without any network calls.",
    )
    return parser


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = worker_config_from_env(env)

    if args.check_config:
        print(
            "worker config ok: "
            f"api_base_url={config.api_base_url!r} worker_id={config.worker_id!r} "
            f"s3_bucket={config.s3_bucket!r} s3_region={config.s3_region!r} "
            f"repo_dir={config.repo_dir!r} pipeline_python={config.pipeline_python!r} "
            f"poll_wait_s={config.poll_wait_s} heartbeat_interval_s={config.heartbeat_interval_s} "
            f"work_dir={config.work_dir!r}"
        )
        return 0

    import boto3

    api_client = HttpApiClient(
        base_url=config.api_base_url,
        token=config.worker_bearer_token,
        worker_id=config.worker_id,
        poll_wait_s=config.poll_wait_s,
    )
    s3_kwargs: dict[str, Any] = {"region_name": config.s3_region}
    if config.aws_access_key_id and config.aws_secret_access_key:
        s3_kwargs["aws_access_key_id"] = config.aws_access_key_id
        s3_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
    s3_client = boto3.client("s3", **s3_kwargs)

    def process_runner(job: WorkerJob, video_path: Path, sidecar_path: Path | None, out_dir: Path) -> RunResult:
        return _real_process_runner(job, video_path, sidecar_path, out_dir, config=config)

    while True:
        if PREEMPTION_FLAG_PATH.exists():
            return 0
        processed = run_once(api_client, s3_client, process_runner, config)
        if processed:
            _git_pull_ff_only(config.repo_dir)


if __name__ == "__main__":
    sys.exit(main())
