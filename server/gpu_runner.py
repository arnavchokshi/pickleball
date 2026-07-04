from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import httpx

SAFE_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
RESOURCE_USAGE_ARTIFACT = "gpu_resource_usage.json"
PIPELINE_SUMMARY_ARTIFACT = "PIPELINE_SUMMARY.json"
RESOURCE_MONITOR_SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "racketsport" / "monitor_process_resources.py"


class MissingGpuRunnerConfig(RuntimeError):
    """Raised when no real GPU execution path is configured."""


def safe_slug(value: str) -> str:
    if not value or not SAFE_SLUG_PATTERN.match(value):
        raise ValueError(f"unsafe slug {value!r}; expected {SAFE_SLUG_PATTERN.pattern}")
    return value


@dataclass(frozen=True)
class GpuRunProgress:
    percent: int
    stage: str
    message: str = ""
    eta_seconds: int | None = None


ProgressCallback = Callable[[GpuRunProgress], None]


@dataclass(frozen=True)
class GpuRunRequest:
    job_id: str
    clip: str
    input_dir: Path
    video_path: Path
    artifacts_dir: Path
    capture_sidecar_path: Path | None = None
    court_corners_path: Path | None = None
    court_calibration_path: Path | None = None
    court_review_path: Path | None = None
    max_frames: int | None = None
    allow_auto_court_corners_preview: bool = True
    progress_callback: ProgressCallback | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class GpuRunResult:
    status: str
    notes: list[str] = field(default_factory=list)
    artifacts_dir: Path | None = None
    manifest_path: Path | None = None
    remote_run_dir: str | None = None
    raw: dict[str, object] = field(default_factory=dict)


class GpuRunner:
    name = "base"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        raise NotImplementedError

    @staticmethod
    def emit_progress(
        request: GpuRunRequest,
        *,
        percent: int,
        stage: str,
        message: str = "",
        eta_seconds: int | None = None,
    ) -> None:
        if request.progress_callback is None:
            return
        request.progress_callback(
            GpuRunProgress(
                percent=max(0, min(100, percent)),
                stage=stage,
                message=message,
                eta_seconds=eta_seconds,
            )
        )


class UnconfiguredGpuRunner(GpuRunner):
    name = "unconfigured"

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        raise MissingGpuRunnerConfig(
            "No GPU runner is configured. Set PICKLEBALL_GPU_SSH_HOST + "
            "PICKLEBALL_GPU_SSH_KEY_PATH for GCP SSH execution, or set "
            "PICKLEBALL_GPU_WORKER_URL for an HTTP GPU worker."
        )


RunCommand = Callable[[list[str], int | None], subprocess.CompletedProcess[str]]


def _run_command(cmd: list[str], timeout_s: int | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


class SshGpuRunner(GpuRunner):
    name = "gcp-ssh"

    def __init__(
        self,
        *,
        host: str,
        key_path: str,
        remote_repo: str,
        remote_python: str,
        known_hosts_path: str | None = None,
        connect_timeout_s: int = 20,
        command_timeout_s: int = 7200,
        supports_court_calibration: bool = False,
        extra_pythonpath: str | None = None,
        wasb_repo: str | None = None,
        wasb_checkpoint: str | None = None,
        run: RunCommand = _run_command,
    ) -> None:
        self.host = host
        self.key_path = key_path
        self.remote_repo = remote_repo.rstrip("/")
        self.remote_python = remote_python
        self.known_hosts_path = known_hosts_path
        self.connect_timeout_s = connect_timeout_s
        self.command_timeout_s = command_timeout_s
        self.supports_court_calibration = supports_court_calibration
        self.extra_pythonpath = extra_pythonpath
        self.wasb_repo = wasb_repo
        self.wasb_checkpoint = wasb_checkpoint
        self._run = run

    def describe(self) -> dict[str, str]:
        return {
            "mode": self.name,
            "host": self.host,
            "remote_repo": self.remote_repo,
            "remote_python": self.remote_python,
        }

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        job_id = safe_slug(request.job_id)
        clip = safe_slug(request.clip)
        request.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if request.court_calibration_path is not None and not self.supports_court_calibration:
            raise RuntimeError(
                "court_calibration upload is not supported by the configured GPU process_video snapshot; "
                "upload capture_sidecar or court_corners instead."
            )

        remote_job_dir = f"{self.remote_repo}/runs/render_jobs/{job_id}"
        remote_input_dir = f"{remote_job_dir}/input"
        remote_out_dir = f"{remote_job_dir}/out"
        remote_artifacts_dir = f"{remote_out_dir}/{clip}"

        _copy_resource_monitor_to_input(request.input_dir)
        self.emit_progress(
            request,
            percent=12,
            stage="Preparing GPU workspace",
            message="Creating the remote job directory.",
        )
        self._checked_run(
            [
                *self._ssh_base(),
                f"mkdir -p {shlex.quote(remote_input_dir)} {shlex.quote(remote_out_dir)} {shlex.quote(remote_artifacts_dir)}",
            ]
        )
        self.emit_progress(
            request,
            percent=20,
            stage="Uploading inputs to GPU",
            message="Copying video and sidecar files to the GCP host.",
        )
        self._checked_run(
            [
                "rsync",
                "-az",
                "-e",
                self._rsync_ssh_command(),
                f"{request.input_dir}/",
                f"{self.host}:{remote_input_dir}/",
            ]
        )
        self.emit_progress(
            request,
            percent=36,
            stage="Running pipeline on GPU",
            message="GPU processing is running process_video.py.",
        )
        self._checked_run([*self._ssh_base(), self._remote_process_command(request, remote_input_dir, remote_out_dir)])
        self.emit_progress(
            request,
            percent=88,
            stage="Syncing replay artifacts",
            message="Copying replay outputs back to Render.",
        )
        self._checked_run(
            [
                "rsync",
                "-az",
                "-e",
                self._rsync_ssh_command(),
                f"{self.host}:{remote_artifacts_dir}/",
                f"{request.artifacts_dir}/",
            ]
        )
        prepare_render_artifacts(request)

        manifest_path = request.artifacts_dir / "replay_viewer_manifest.json"
        raw = _artifact_payloads(request.artifacts_dir)
        return GpuRunResult(
            status="complete",
            notes=["processed on configured GCP GPU host via SSH"],
            artifacts_dir=request.artifacts_dir,
            manifest_path=manifest_path if manifest_path.is_file() else None,
            remote_run_dir=remote_artifacts_dir,
            raw=raw,
        )

    def _ssh_options(self) -> list[str]:
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self.known_hosts_path:
            options.extend(["-o", f"UserKnownHostsFile={self.known_hosts_path}"])
        return options

    def _ssh_base(self) -> list[str]:
        return ["ssh", "-i", self.key_path, *self._ssh_options(), self.host]

    def _rsync_ssh_command(self) -> str:
        return " ".join(shlex.quote(part) for part in ["ssh", "-i", self.key_path, *self._ssh_options()])

    def _remote_process_command(self, request: GpuRunRequest, remote_input_dir: str, remote_out_dir: str) -> str:
        video_name = request.video_path.name
        process_args = [
            self.remote_python,
            "scripts/racketsport/process_video.py",
            "--video",
            f"{remote_input_dir}/{video_name}",
            "--out",
            remote_out_dir,
            "--clip",
            safe_slug(request.clip),
            "--body-local",
            "--device",
            "cuda:0",
            "--json",
        ]
        if request.allow_auto_court_corners_preview:
            process_args.append("--allow-auto-court-corners-preview")
        if self.wasb_repo:
            process_args.extend(["--wasb-repo", self.wasb_repo])
        if self.wasb_checkpoint:
            process_args.extend(["--wasb-checkpoint", self.wasb_checkpoint])
        if request.max_frames is not None:
            process_args.extend(["--max-frames", str(request.max_frames)])
        if request.capture_sidecar_path is not None:
            process_args.extend(["--capture-sidecar", f"{remote_input_dir}/{request.capture_sidecar_path.name}"])
        if request.court_corners_path is not None:
            process_args.extend(["--court-corners", f"{remote_input_dir}/{request.court_corners_path.name}"])
        if request.court_calibration_path is not None and self.supports_court_calibration:
            process_args.extend(["--court-calibration", f"{remote_input_dir}/{request.court_calibration_path.name}"])

        telemetry_path = f"{remote_out_dir}/{safe_slug(request.clip)}/{RESOURCE_USAGE_ARTIFACT}"
        monitor_args = [
            self.remote_python,
            f"{remote_input_dir}/{RESOURCE_MONITOR_SOURCE.name}",
            "--out",
            telemetry_path,
            "--sample-interval",
            "5",
            "--",
            *process_args,
        ]
        quoted = " ".join(shlex.quote(arg) for arg in monitor_args)
        env_prefix = self._remote_env_prefix()
        return f"cd {shlex.quote(self.remote_repo)} && {env_prefix}{quoted}"

    def _remote_env_prefix(self) -> str:
        if not self.extra_pythonpath:
            return ""
        return f"PYTHONPATH={shlex.quote(self.extra_pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}} "

    def _checked_run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._run(cmd, self.command_timeout_s)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"GPU SSH command failed ({cmd[0]} exit {completed.returncode}): {detail}")
        return completed


class HttpGpuRunner(GpuRunner):
    name = "gpu-http-worker"

    def __init__(self, *, base_url: str, token: str | None = None, timeout_s: int = 7200) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def describe(self) -> dict[str, str]:
        return {"mode": self.name, "base_url": self.base_url}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        self.emit_progress(
            request,
            percent=20,
            stage="Uploading inputs to GPU worker",
            message="Submitting the video package to the configured GPU worker.",
        )
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        data: dict[str, str] = {"job_id": request.job_id, "clip": request.clip}
        if request.max_frames is not None:
            data["max_frames"] = str(request.max_frames)
        if request.allow_auto_court_corners_preview:
            data["allow_auto_court_corners_preview"] = "1"

        files = {"video": (request.video_path.name, request.video_path.open("rb"), "application/octet-stream")}
        optional_files = {
            "capture_sidecar": request.capture_sidecar_path,
            "court_corners": request.court_corners_path,
            "court_calibration": request.court_calibration_path,
            "court_review": request.court_review_path,
        }
        try:
            for field, path in optional_files.items():
                if path is not None:
                    files[field] = (path.name, path.open("rb"), "application/json")
            with httpx.Client(timeout=self.timeout_s) as client:
                self.emit_progress(
                    request,
                    percent=36,
                    stage="Running pipeline on GPU",
                    message="Waiting for the GPU worker to return.",
                )
                response = client.post(f"{self.base_url}/jobs", data=data, files=files, headers=headers)
                response.raise_for_status()
                payload = response.json()
        finally:
            for file_tuple in files.values():
                file_tuple[1].close()

        return GpuRunResult(
            status=str(payload.get("status", "submitted")),
            notes=[str(note) for note in payload.get("notes", []) if isinstance(note, str)],
            artifacts_dir=request.artifacts_dir,
            remote_run_dir=str(payload.get("remote_run_dir")) if payload.get("remote_run_dir") else None,
            raw=payload,
        )


class LocalPipelineRunner(GpuRunner):
    name = "local-pipeline"

    def __init__(
        self,
        *,
        enabled: bool,
        python: str = "python",
        command_timeout_s: int = 7200,
        run: RunCommand = _run_command,
    ) -> None:
        self.enabled = enabled
        self.python = python
        self.command_timeout_s = command_timeout_s
        self._run = run

    def describe(self) -> dict[str, str]:
        return {"mode": self.name, "enabled": str(self.enabled).lower()}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        if not self.enabled:
            raise MissingGpuRunnerConfig("Local pipeline execution is disabled; set PICKLEBALL_ALLOW_LOCAL_PIPELINE=1 to enable it.")

        request.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_dir = request.artifacts_dir.parent / "process_video"
        self.emit_progress(
            request,
            percent=36,
            stage="Running local pipeline",
            message="process_video.py is running on the local machine.",
        )
        args = [
            self.python,
            "scripts/racketsport/process_video.py",
            "--video",
            str(request.video_path),
            "--out",
            str(out_dir),
            "--clip",
            safe_slug(request.clip),
            "--vite-allow-root",
            str(request.input_dir.parent),
            "--json",
        ]
        if request.allow_auto_court_corners_preview:
            args.append("--allow-auto-court-corners-preview")
        if request.max_frames is not None:
            args.extend(["--max-frames", str(request.max_frames)])
        if request.capture_sidecar_path is not None:
            args.extend(["--capture-sidecar", str(request.capture_sidecar_path)])
        if request.court_corners_path is not None:
            args.extend(["--court-corners", str(request.court_corners_path)])
        if request.court_calibration_path is not None:
            args.extend(["--court-calibration", str(request.court_calibration_path)])

        completed = self._run(args, self.command_timeout_s)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"local process_video failed ({completed.returncode}): {detail}")

        self.emit_progress(
            request,
            percent=88,
            stage="Collecting replay artifacts",
            message="Copying process_video outputs into the job artifacts directory.",
        )
        produced_dir = out_dir / safe_slug(request.clip)
        if produced_dir.is_dir() and produced_dir != request.artifacts_dir:
            for path in produced_dir.iterdir():
                target = request.artifacts_dir / path.name
                if path.is_file():
                    target.write_bytes(path.read_bytes())

        prepare_render_artifacts(request)
        manifest_path = request.artifacts_dir / "replay_viewer_manifest.json"
        return GpuRunResult(
            status="complete",
            notes=["processed by local pipeline"],
            artifacts_dir=request.artifacts_dir,
            manifest_path=manifest_path if manifest_path.is_file() else None,
        )


def runner_from_env(env: Mapping[str, str] | None = None) -> GpuRunner:
    env = os.environ if env is None else env
    worker_url = env.get("PICKLEBALL_GPU_WORKER_URL", "").strip()
    if worker_url:
        return HttpGpuRunner(base_url=worker_url, token=env.get("PICKLEBALL_GPU_WORKER_TOKEN") or None)

    ssh_host = env.get("PICKLEBALL_GPU_SSH_HOST", "").strip()
    ssh_key = env.get("PICKLEBALL_GPU_SSH_KEY_PATH", "").strip()
    if ssh_host and ssh_key:
        return SshGpuRunner(
            host=ssh_host,
            key_path=ssh_key,
            remote_repo=env.get("PICKLEBALL_GPU_REPO", "/home/arnavchokshi/pickleball_git"),
            remote_python=env.get("PICKLEBALL_GPU_PYTHON", "/home/arnavchokshi/pickleball_git/.venv/bin/python"),
            known_hosts_path=env.get("PICKLEBALL_GPU_KNOWN_HOSTS_PATH") or None,
            connect_timeout_s=int(env.get("PICKLEBALL_GPU_CONNECT_TIMEOUT_S", "20")),
            command_timeout_s=int(env.get("PICKLEBALL_GPU_COMMAND_TIMEOUT_S", "7200")),
            supports_court_calibration=env.get("PICKLEBALL_GPU_SUPPORTS_COURT_CALIBRATION", "").strip() == "1",
            extra_pythonpath=env.get("PICKLEBALL_GPU_EXTRA_PYTHONPATH") or None,
            wasb_repo=env.get("PICKLEBALL_GPU_WASB_REPO") or None,
            wasb_checkpoint=env.get("PICKLEBALL_GPU_WASB_CHECKPOINT") or None,
        )

    return LocalPipelineRunner(
        enabled=env.get("PICKLEBALL_ALLOW_LOCAL_PIPELINE", "").strip() == "1",
        python=env.get("PICKLEBALL_LOCAL_PYTHON", "python"),
        command_timeout_s=int(env.get("PICKLEBALL_GPU_COMMAND_TIMEOUT_S", "7200")),
    )


def prepare_render_artifacts(request: GpuRunRequest) -> None:
    request.artifacts_dir.mkdir(parents=True, exist_ok=True)
    _copy_source_video_artifact(request)
    _rewrite_manifest_urls_for_render(request)


def _copy_resource_monitor_to_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESOURCE_MONITOR_SOURCE, input_dir / RESOURCE_MONITOR_SOURCE.name)


def _artifact_payloads(artifacts_dir: Path) -> dict[str, object]:
    payloads: dict[str, object] = {}
    resource_usage = _load_json_artifact(artifacts_dir / RESOURCE_USAGE_ARTIFACT)
    if resource_usage is not None:
        payloads["resource_usage"] = resource_usage
    pipeline_summary = _load_json_artifact(artifacts_dir / PIPELINE_SUMMARY_ARTIFACT)
    if pipeline_summary is not None:
        payloads["pipeline_summary"] = pipeline_summary
    return payloads


def _load_json_artifact(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _copy_source_video_artifact(request: GpuRunRequest) -> None:
    suffix = request.video_path.suffix.lower() or ".mp4"
    target = request.artifacts_dir / f"source{suffix}"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        target.unlink()
    if not target.is_file():
        shutil.copy2(request.video_path, target)


def _rewrite_manifest_urls_for_render(request: GpuRunRequest) -> None:
    manifest_path = request.artifacts_dir / "replay_viewer_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    source_artifact_name = f"source{request.video_path.suffix.lower() or '.mp4'}"
    rewritten = _rewrite_manifest_value(payload, job_id=request.job_id, source_artifact_name=source_artifact_name)
    manifest_path.write_text(json.dumps(rewritten, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_manifest_value(
    value: object,
    *,
    job_id: str,
    source_artifact_name: str,
    key: str | None = None,
) -> object:
    if isinstance(value, dict):
        return {
            str(child_key): _rewrite_manifest_value(
                child_value,
                job_id=job_id,
                source_artifact_name=source_artifact_name,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_manifest_value(item, job_id=job_id, source_artifact_name=source_artifact_name, key=key) for item in value]
    if isinstance(value, str) and key is not None and key.endswith("_url"):
        artifact_name = _artifact_name_from_pipeline_url(value)
        if artifact_name is not None:
            if key == "video_url":
                artifact_name = source_artifact_name
            return f"/api/jobs/{safe_slug(job_id)}/artifacts/{artifact_name}"
    return value


def _artifact_name_from_pipeline_url(value: str) -> str | None:
    if value.startswith("/@fs//"):
        return Path(value.removeprefix("/@fs//")).name
    if value.startswith("/@fs/"):
        return Path(value.removeprefix("/@fs/")).name
    if value.startswith("/"):
        path = Path(value)
        if "runs" in path.parts and path.name:
            return path.name
    return None
