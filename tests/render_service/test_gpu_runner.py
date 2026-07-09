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
                        "clip": "clip_1",
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
        extra_pythonpath="/srv/pickleball/extra_pythonpath",
        wasb_repo="/srv/pickleball_git/third_party/WASB-SBDT",
        wasb_checkpoint="/srv/pickleball_git/models/checkpoints/wasb/wasb_tennis_best.pth.tar",
        run=fake_run,
    )

    result = runner.run(request)

    assert result.status == "complete"
    assert result.manifest_path == tmp_path / "artifacts" / "replay_viewer_manifest.json"
    assert calls[0][0] == "ssh"
    assert calls[1][0] == "rsync"
    assert calls[2][0] == "rsync"
    assert calls[3][0] == "ssh"
    assert calls[4][0] == "rsync"
    assert str(Path.cwd() / "scripts") in calls[2]
    assert str(Path.cwd() / "threed") in calls[2]
    assert str(Path.cwd() / "configs") in calls[2]

    remote_command = calls[3][-1]
    assert "cd /srv/pickleball/runs/render_jobs/job_1/code" in remote_command
    assert "/srv/pickleball/runs/render_jobs/job_1/input/monitor_process_resources.py" in remote_command
    assert "--out /srv/pickleball/runs/render_jobs/job_1/out/clip_1/gpu_resource_usage.json" in remote_command
    assert " -- /srv/pickleball/.venv/bin/python scripts/racketsport/process_video.py " in remote_command
    assert "scripts/racketsport/process_video.py" in remote_command
    assert "--body-local" in remote_command
    assert "--device cuda:0" in remote_command
    assert "--manifest /srv/pickleball/models/MANIFEST.json" in remote_command
    assert "--reid-model /srv/pickleball/models/checkpoints/osnet_x1_0_market1501.pt" in remote_command
    assert "--allow-auto-court-corners-preview" in remote_command
    assert (
        "PYTHONPATH=/srv/pickleball/runs/render_jobs/job_1/code:"
        "/srv/pickleball/extra_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
    ) in remote_command
    assert "--wasb-repo /srv/pickleball_git/third_party/WASB-SBDT" in remote_command
    assert "--wasb-checkpoint /srv/pickleball_git/models/checkpoints/wasb/wasb_tennis_best.pth.tar" in remote_command
    assert "--capture-sidecar" in remote_command
    assert "--max-frames 12" in remote_command
    assert [event.stage for event in progress_events] == [
        "Preparing GPU workspace",
        "Uploading inputs to GPU",
        "Syncing current pipeline code",
        "Running pipeline on GPU",
        "Syncing replay artifacts",
    ]
    rewritten_manifest = json.loads((tmp_path / "artifacts" / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert rewritten_manifest["video_url"] == "/api/jobs/job_1/artifacts/source.mp4"
    assert rewritten_manifest["virtual_world_url"] == "/api/jobs/job_1/artifacts/confidence_gated_world.json"
    assert (tmp_path / "artifacts" / "source.mp4").read_bytes() == b"video"
    assert (tmp_path / "input" / "monitor_process_resources.py").is_file()


def test_ssh_runner_exposes_resource_usage_artifact(tmp_path: Path) -> None:
    def fake_run(cmd: list[str], timeout_s: int | None) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "rsync" and "gpu.example:/srv/pickleball/runs/render_jobs/job_1/out/clip_1/" in cmd[-2]:
            (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "artifacts" / "replay_viewer_manifest.json").write_text("{}", encoding="utf-8")
            (tmp_path / "artifacts" / "gpu_resource_usage.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "racketsport_resource_usage",
                        "summary": {"gpu_utilization_avg_pct": 64.2, "gpu_memory_used_max_mb": 22118},
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "clip.mp4"
    video.write_bytes(b"video")
    request = GpuRunRequest(
        job_id="job_1",
        clip="clip_1",
        input_dir=input_dir,
        video_path=video,
        artifacts_dir=tmp_path / "artifacts",
    )
    runner = SshGpuRunner(
        host="gpu.example",
        key_path="/etc/secrets/gcp_ssh_key",
        remote_repo="/srv/pickleball",
        remote_python="/srv/pickleball/.venv/bin/python",
        run=fake_run,
    )

    result = runner.run(request)

    assert result.raw["resource_usage"]["summary"]["gpu_utilization_avg_pct"] == 64.2


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


def test_local_pipeline_runner_passes_reviewed_calibration_to_process_video(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout_s: int | None) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        out_dir = Path(cmd[cmd.index("--out") + 1])
        clip = cmd[cmd.index("--clip") + 1]
        produced_dir = out_dir / clip
        produced_dir.mkdir(parents=True, exist_ok=True)
        (produced_dir / "replay_viewer_manifest.json").write_text(
            json.dumps(
                    {
                        "clip": "clip_1",
                        "video_url": str(tmp_path / "input" / "clip.mp4"),
                    "virtual_world_url": str(produced_dir / "confidence_gated_world.json"),
                    "body_mesh_index_url": str(produced_dir / "body_mesh_index" / "body_mesh_index.json"),
                }
            ),
            encoding="utf-8",
        )
        (produced_dir / "confidence_gated_world.json").write_text("{}", encoding="utf-8")
        body_index_dir = produced_dir / "body_mesh_index"
        body_index_dir.mkdir()
        (body_index_dir / "body_mesh_index.json").write_text(
            json.dumps({"windows": [{"url": "body_mesh_chunks/window_000.bin.gz"}]}),
            encoding="utf-8",
        )
        chunk = body_index_dir / "body_mesh_chunks" / "window_000.bin.gz"
        chunk.parent.mkdir()
        chunk.write_bytes(b"mesh-chunk")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "clip.mp4"
    calibration = input_dir / "court_calibration.json"
    review = input_dir / "reviewed_court_calibration.json"
    video.write_bytes(b"video")
    calibration.write_text('{"schema_version":1}', encoding="utf-8")
    review.write_text('{"review_status":"human_reviewed"}', encoding="utf-8")
    request = GpuRunRequest(
        job_id="job_1",
        clip="clip_1",
        input_dir=input_dir,
        video_path=video,
        artifacts_dir=tmp_path / "artifacts",
        court_calibration_path=calibration,
        court_review_path=review,
    )

    result = LocalPipelineRunner(enabled=True, python="/repo/.venv/bin/python", run=fake_run).run(request)

    assert result.status == "complete"
    command = calls[0]
    assert command[:2] == ["/repo/.venv/bin/python", "scripts/racketsport/process_video.py"]
    assert command[command.index("--vite-allow-root") + 1] == str(input_dir.parent)
    assert command[command.index("--court-calibration") + 1] == str(calibration)
    assert "--court-review" not in command
    assert (request.artifacts_dir / "body_mesh_index" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes() == b"mesh-chunk"
