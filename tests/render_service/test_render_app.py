import base64
from pathlib import Path

from fastapi.testclient import TestClient

from server.gpu_runner import GpuRunProgress, GpuRunRequest, GpuRunResult
from server.render_app import create_app
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


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
        (request.artifacts_dir / "gpu_resource_usage.json").write_text(
            '{"artifact_type":"racketsport_resource_usage","summary":{"gpu_utilization_avg_pct":55.5,"gpu_memory_used_max_mb":12345}}',
            encoding="utf-8",
        )
        (request.artifacts_dir / "PIPELINE_SUMMARY.json").write_text(
            '{"status":"complete","stages":[{"stage":"ingest","wall_seconds":1.25},{"stage":"body","wall_seconds":9.5}]}',
            encoding="utf-8",
        )
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


class ProgressRunner(CompletingRunner):
    name = "test-progress"

    def run(self, request: GpuRunRequest) -> GpuRunResult:
        assert request.progress_callback is not None
        request.progress_callback(
            GpuRunProgress(
                percent=42,
                stage="Running pipeline on GPU",
                message="Tracking and body stages are active.",
                eta_seconds=118,
            )
        )
        return super().run(request)


def _prediction_points() -> dict[str, dict[str, object]]:
    xs = {
        "near_left_corner": 180,
        "near_baseline_center": 500,
        "near_right_corner": 820,
        "far_right_corner": 780,
        "far_baseline_center": 500,
        "far_left_corner": 220,
        "near_nvz_left": 220,
        "near_nvz_center": 500,
        "near_nvz_right": 780,
        "net_left_sideline": 230,
        "net_center": 500,
        "net_right_sideline": 770,
        "far_nvz_left": 240,
        "far_nvz_center": 500,
        "far_nvz_right": 760,
    }
    ys = {
        "near_left_corner": 520,
        "near_baseline_center": 520,
        "near_right_corner": 520,
        "far_right_corner": 180,
        "far_baseline_center": 180,
        "far_left_corner": 180,
        "near_nvz_left": 400,
        "near_nvz_center": 400,
        "near_nvz_right": 400,
        "net_left_sideline": 330,
        "net_center": 330,
        "net_right_sideline": 330,
        "far_nvz_left": 260,
        "far_nvz_center": 260,
        "far_nvz_right": 260,
    }
    return {name: {"xy": [float(xs[name]), float(ys[name])], "confidence": 0.7} for name in PICKLEBALL_COURT_KEYPOINT_NAMES}


def _review_payload() -> dict[str, object]:
    adjusted = {name: list(point["xy"]) for name, point in _prediction_points().items()}
    adjusted["near_left_corner"] = [96.0, 203.0]
    return {
        "video_id": "drill_01",
        "video_path": "/tmp/drill.mp4",
        "video_sha256": "c" * 64,
        "image_size": [1000, 600],
        "frame_index": 12,
        "frame_time_s": 0.4,
        "auto_prediction_source": "court_detector_v2:selected_hypothesis=hypothesis_0001",
        "predicted_points": _prediction_points(),
        "adjusted_points": adjusted,
        "created_at": "2026-07-04T12:00:00Z",
    }


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
    assert status["result"]["resource_usage_url"].endswith("/artifacts/gpu_resource_usage.json")
    assert status["result"]["pipeline_summary_url"].endswith("/artifacts/PIPELINE_SUMMARY.json")
    assert status["result"]["resource_summary"]["gpu_utilization_avg_pct"] == 55.5

    assert runner.requests[0].clip == "drill_01"
    assert runner.requests[0].max_frames == 8
    assert runner.requests[0].video_path.read_bytes() == b"fake-video"
    assert runner.requests[0].capture_sidecar_path is not None
    assert runner.requests[0].capture_sidecar_path.read_text(encoding="utf-8") == "{}"

    manifest = client.get(status["result"]["manifest_url"])
    assert manifest.status_code == 200
    assert manifest.json()["artifact_type"] == "replay_viewer_manifest"


def test_court_prediction_endpoint_uses_configured_predictor_and_returns_preview_status(tmp_path: Path) -> None:
    def fake_predictor(*, video_path: Path, clip: str, frame_index: int | None) -> dict[str, object]:
        assert video_path.read_bytes() == b"fake-video"
        assert clip == "drill_01"
        assert frame_index == 12
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_court_layout_prediction",
            "clip": clip,
            "image_size": [1000, 600],
            "frame_index": 12,
            "frame_time_s": 0.4,
            "prediction_source": "court_detector_v2:selected_hypothesis=hypothesis_0001",
            "verified": False,
            "not_cal3_verified": True,
            "points": _prediction_points(),
            "lines": [],
            "warnings": ["auto_court_detection_preview_not_verified"],
        }

    app = create_app(
        upload_root=tmp_path,
        runner=CompletingRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        court_predictor=fake_predictor,
    )
    client = TestClient(app)

    response = client.post(
        "/api/court/predict",
        data={"clip": "drill_01", "frame_index": "12"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["video"]["sha256"]
    assert payload["points"]["near_left_corner"]["xy"] == [180.0, 520.0]
    assert payload["preview_frame_url"] is None
    assert "preview_frame_jpeg_base64" not in payload


def test_court_prediction_endpoint_persists_preview_frame_and_exposes_follow_up_url(tmp_path: Path) -> None:
    frame_bytes = b"\xff\xd8\xff\xdb-fake-jpeg-bytes"

    def fake_predictor(*, video_path: Path, clip: str, frame_index: int | None) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_court_layout_prediction",
            "clip": clip,
            "image_size": [1000, 600],
            "frame_index": 12,
            "frame_time_s": 0.4,
            "prediction_source": "court_proposals:selected_proposal=proposal_0001",
            "verified": False,
            "not_cal3_verified": True,
            "points": _prediction_points(),
            "lines": [],
            "warnings": ["auto_court_detection_preview_not_verified"],
            "needs_user_input": [],
            "assist": {"mode": "none", "tap_points": [], "line_label": None},
            "preview_frame_index": 12,
            "preview_frame_jpeg_base64": base64.b64encode(frame_bytes).decode("ascii"),
        }

    app = create_app(
        upload_root=tmp_path,
        runner=CompletingRunner(),
        run_jobs_inline=True,
        static_dir=tmp_path / "dist",
        court_predictor=fake_predictor,
    )
    client = TestClient(app)

    response = client.post(
        "/api/court/predict",
        data={"clip": "drill_01"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "preview_frame_jpeg_base64" not in payload
    assert payload["preview_frame_index"] == 12
    preview_frame_url = payload["preview_frame_url"]
    assert preview_frame_url is not None
    assert preview_frame_url.startswith("/api/court/predict/")
    assert preview_frame_url.endswith("/frame")

    frame_response = client.get(preview_frame_url)
    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"] == "image/jpeg"
    assert frame_response.content == frame_bytes


def test_court_prediction_frame_endpoint_404s_for_unknown_prediction_id(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.get("/api/court/predict/pred_does_not_exist/frame")

    assert response.status_code == 404


def test_court_review_endpoint_saves_training_artifacts_and_returns_pipeline_calibration(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post("/api/court/reviews", json=_review_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["review"]["artifact_type"] == "racketsport_reviewed_court_calibration"
    assert payload["review"]["review_status"] == "human_reviewed"
    assert payload["court_calibration"]["schema_version"] == 1
    assert payload["saved"]["review_path"].endswith("reviewed_court_calibration.json")
    assert Path(payload["saved"]["index_path"]).is_file()


def test_court_review_endpoint_keeps_auto_predictions_out_of_training_index(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)
    payload = _review_payload()
    payload["review_status"] = "auto_predicted_unreviewed"
    payload["adjusted_points"] = {name: list(point["xy"]) for name, point in _prediction_points().items()}

    response = client.post("/api/court/reviews", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["review"]["review_status"] == "auto_predicted_unreviewed"
    assert body["review"]["training"]["usable_for_court_detector_training"] is False
    assert body["review"]["training"]["training_policy"] == "auto_prediction_not_training_ready"
    assert "auto_predicted_court_layout_unreviewed" in body["court_calibration"]["capture_quality"]["reasons"]

    index = Path(body["saved"]["index_path"]).read_text(encoding="utf-8")
    assert "auto_predicted_unreviewed" in index
    assert "auto_prediction_not_training_ready" in index


def test_upload_job_saves_reviewed_court_artifact_and_passes_derived_calibration(tmp_path: Path) -> None:
    runner = CompletingRunner()
    app = create_app(upload_root=tmp_path, runner=runner, run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "drill_01"},
        files={
            "video": ("drill.mp4", b"fake-video", "video/mp4"),
            "court_review": ("reviewed_court_calibration.json", b'{"review_status":"human_reviewed"}', "application/json"),
            "court_calibration": ("court_calibration.json", b'{"schema_version":1}', "application/json"),
        },
    )

    assert response.status_code == 202
    assert runner.requests[0].court_calibration_path is not None
    assert runner.requests[0].court_review_path is not None
    assert runner.requests[0].court_review_path.read_text(encoding="utf-8") == '{"review_status":"human_reviewed"}'


def test_upload_job_accepts_court_assist_seed_and_surfaces_it_in_job_status(tmp_path: Path) -> None:
    """iOS parity: the RenderGatewayClient multipart field is named "court_assist_seed"
    (RenderGatewayClient.swift:75-77). Today FastAPI silently drops unknown fields; this
    field must now be persisted to the job input dir and surfaced in job status so a later
    pipeline session can consume it. It is intentionally NOT forwarded into GpuRunRequest
    this wave."""

    runner = CompletingRunner()
    app = create_app(upload_root=tmp_path, runner=runner, run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    assist_seed_body = b'{"mode":"one_inside_tap","tap_points":[[500.0,300.0]],"line_label":null,"trusted_calibration":false}'
    response = client.post(
        "/api/jobs",
        data={"clip": "drill_01"},
        files={
            "video": ("drill.mp4", b"fake-video", "video/mp4"),
            "court_assist_seed": ("court_assist_seed.json", assist_seed_body, "application/json"),
        },
    )

    assert response.status_code == 202
    job_id = response.json()["id"]

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["court_assist_seed"]["present"] is True

    saved_path = Path(status["court_assist_seed"]["path"])
    assert saved_path.is_file()
    assert saved_path.read_bytes() == assist_seed_body
    assert saved_path.name == "court_assist_seed.json"

    # gpu_runner forwarding is explicitly untouched this wave (owned by another lane).
    assert not hasattr(runner.requests[0], "court_assist_seed_path")


def test_upload_job_without_court_assist_seed_reports_it_as_absent(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "drill_01"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    status = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert status["court_assist_seed"] is None


def test_upload_job_reports_progress_and_eta(tmp_path: Path) -> None:
    runner = ProgressRunner()
    app = create_app(upload_root=tmp_path, runner=runner, run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "progress_01", "max_frames": "8"},
        files={"video": ("progress.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 202
    initial = response.json()
    assert initial["progress"]["percent"] >= 0

    status = client.get(initial["links"]["status"]).json()

    assert status["status"] == "complete"
    assert status["progress"]["percent"] == 100
    assert status["progress"]["stage"] == "Replay ready"
    assert status["progress"]["eta_seconds"] == 0
    assert any(step["id"] == "gpu_pipeline" and step["status"] == "complete" for step in status["progress"]["steps"])


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
    assert status["progress"]["stage"] == "Failed"
    assert status["progress"]["eta_seconds"] is None


def test_upload_rejects_unsafe_clip_names(tmp_path: Path) -> None:
    app = create_app(upload_root=tmp_path, runner=CompletingRunner(), run_jobs_inline=True, static_dir=tmp_path / "dist")
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        data={"clip": "../escape"},
        files={"video": ("drill.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 400
