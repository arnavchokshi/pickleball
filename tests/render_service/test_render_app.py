from pathlib import Path

from fastapi.testclient import TestClient

from server.gpu_runner import GpuRunRequest, GpuRunResult
from server.render_app import create_app


class CompletingRunner:
    name = "test-completing"

    def __init__(self) -> None:
        self.requests: list[GpuRunRequest] = []

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        self.requests.append(request)
        request.artifacts_dir.mkdir(parents=True, exist_ok=True)
        manifest = request.artifacts_dir / "replay_viewer_manifest.json"
        manifest.write_text('{"artifact_type":"replay_viewer_manifest"}', encoding="utf-8")
        return GpuRunResult(
            status="complete",
            notes=["fake runner complete"],
            artifacts_dir=request.artifacts_dir,
            manifest_path=manifest,
        )


class FailingRunner:
    name = "test-failing"

    def describe(self) -> dict[str, str]:
        return {"mode": self.name}

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        raise RuntimeError("gpu unavailable")


def test_health_reports_runner_mode(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["runner"]["mode"] == "test-completing"


def test_upload_job_saves_inputs_runs_gpu_and_exposes_manifest(tmp_path: Path) -> None:
    runner = CompletingRunner()
    app = create_app(upload_root=tmp_path, runner=runner, run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "drill_01", "max_frames": "8"},
        files={
            "video": ("drill.mp4", b"fake-video", "video/mp4"),
            "capture_sidecar": ("capture_sidecar.json", b"{}", "application/json"),
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["links"]["status"].startswith("/api/jobs/")

    status = client.get(payload["links"]["status"]).json()
    assert status["status"] == "complete"
    assert status["clip"] == "drill_01"
    assert status["result"]["manifest_url"].endswith("/manifest")

    assert runner.requests[0].clip == "drill_01"
    assert runner.requests[0].max_frames == 8
    assert runner.requests[0].video_path.read_bytes() == b"fake-video"
    assert runner.requests[0].capture_sidecar_path is not None
    assert runner.requests[0].capture_sidecar_path.read_text(encoding="utf-8") == "{}"

    manifest = client.get(status["result"]["manifest_url"])
    assert manifest.status_code == 200
    assert manifest.json()["artifact_type"] == "replay_viewer_manifest"


def test_upload_job_records_runner_failures(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=FailingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 202
    status = client.get(response.json()["links"]["status"]).json()

    assert status["status"] == "failed"
    assert "gpu unavailable" in status["error"]


def test_upload_rejects_unsafe_clip_names(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "../escape"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 400
