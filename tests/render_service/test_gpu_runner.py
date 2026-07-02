import subprocess
import json
from pathlib import Path

import pytest

from server.gpu_runner import (
    GpuRunProgress,
    GpuRunRequest,
    LocalPipelineRunner,
    MissingGpuRunnerConfig,
    SshGpuRunner,
    runner_from_env,
    safe_slug,
)


def test_safe_slug_rejects_shell_metacharacters() -> None:
    assert safe_slug("match_01-clip.a") == "match_01-clip.a"

    with pytest.raises(ValueError):
        safe_slug("match; rm -rf /")


def test_runner_from_env_prefers_ssh_when_configured(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("not-a-real-key", encoding="utf-8")

    runner = runner_from_env(
        {
            "PICKLEBALL_GPU_SSH_HOST": "arnav@example-gpu",
            "PICKLEBALL_GPU_SSH_KEY_PATH": str(key),
            "PICKLEBALL_GPU_REPO": "/srv/pickleball",
            "PICKLEBALL_GPU_PYTHON": "/srv/pickleball/.venv/bin/python",
        }
    )

    assert isinstance(runner, SshGpuRunner)


def test_runner_from_env_fails_closed_without_gpu_config() -> None:
    runner = runner_from_env({})

    with pytest.raises(MissingGpuRunnerConfig):
        runner.run(
            GpuRunRequest(
                job_id="job_1",
                clip="clip_1",
                input_dir=Path("/tmp/input"),
                video_path=Path("/tmp/input/clip.mp4"),
                artifacts_dir=Path("/tmp/artifacts"),
            )
        )


def test_ssh_runner_uploads_runs_body_local_and_syncs_artifacts(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout_s: int | None) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[0] == "rsync" and "gpu.example:/srv/pickleball/runs/render_jobs/job_1/out/clip_1/" in cmd[-2]:
            (tmp_path / "artifacts" / "confidence_gated_world.json").write_text("{}", encoding="utf-8")
            (tmp_path / "artifacts" / "replay_viewer_manifest.json").write_text(
                json.dumps(
                    {
                        "video_url": "/@fs//srv/pickleball/runs/render_jobs/job_1/input/clip.mp4",
                        "virtual_world_url": "/@fs//srv/pickleball/runs/render_jobs/job_1/out/clip_1/confidence_gated_world.json",
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    request = GpuRunRequest(
        job_id="job_1",
        clip="clip_1",
        input_dir=tmp_path / "input",
        video_path=tmp_path / "input" / "clip.mp4",
        capture_sidecar_path=tmp_path / "input" / "capture_sidecar.json",
        artifacts_dir=tmp_path / "artifacts",
        max_frames=12,
    )
    request.input_dir.mkdir()
    request.video_path.write_bytes(b"video")
    request.capture_sidecar_path.write_text("{}", encoding="utf-8")
    progress_events: list[GpuRunProgress] = []
    request = GpuRunRequest(
        job_id=request.job_id,
        clip=request.clip,
        input_dir=request.input_dir,
        video_path=request.video_path,
        capture_sidecar_path=request.capture_sidecar_path,
        artifacts_dir=request.artifacts_dir,
        max_frames=request.max_frames,
        progress_callback=progress_events.append,
    )

    runner = SshGpuRunner(
        host="gpu.example",
        key_path="/etc/secrets/gcp_ssh_key",
        remote_repo="/srv/pickleball",
        remote_python="/srv/pickleball/.venv/bin/python",
        known_hosts_path="/etc/secrets/gcp_known_hosts",
        run=fake_run,
    )

    result = runner.run(request)

    assert result.status == "complete"
    assert result.manifest_path == tmp_path / "artifacts" / "replay_viewer_manifest.json"
    assert calls[0][0] == "ssh"
    assert calls[1][0] == "rsync"
    assert calls[2][0] == "ssh"
    assert calls[3][0] == "rsync"

    remote_command = calls[2][-1]
    assert "scripts/racketsport/process_video.py" in remote_command
    assert "--body-local" in remote_command
    assert "--device cuda:0" in remote_command
    assert "--allow-auto-court-corners-preview" in remote_command
    assert "--capture-sidecar" in remote_command
    assert "--max-frames 12" in remote_command
    assert [event.stage for event in progress_events] == [
        "Preparing GPU workspace",
        "Uploading inputs to GPU",
        "Running pipeline on GPU",
        "Syncing replay artifacts",
    ]
    rewritten_manifest = json.loads((tmp_path / "artifacts" / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert rewritten_manifest["video_url"] == "/api/jobs/job_1/artifacts/source.mp4"
    assert rewritten_manifest["virtual_world_url"] == "/api/jobs/job_1/artifacts/confidence_gated_world.json"
    assert (tmp_path / "artifacts" / "source.mp4").read_bytes() == b"video"


def test_local_pipeline_runner_requires_explicit_enablement(tmp_path: Path) -> None:
    runner = LocalPipelineRunner(enabled=False)

    with pytest.raises(MissingGpuRunnerConfig):
        runner.run(
            GpuRunRequest(
                job_id="job_1",
                clip="clip_1",
                input_dir=tmp_path,
                video_path=tmp_path / "clip.mp4",
                artifacts_dir=tmp_path / "artifacts",
            )
        )
