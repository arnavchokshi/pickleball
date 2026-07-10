from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.pipeline_invocation as pipeline_invocation
from server.bundle_policy import evaluate_bundle, gate_reported_status
from server.pipeline_invocation import stage_manifest_delivery_bundle
from server.routes.worker import build_worker_router
from server.worker.config import WorkerConfig
from server.worker.daemon import RunResult, run_once


WORKER_TOKEN = "ns015-worker-token"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    status: str = "complete",
    missing_capabilities: list[dict[str, str]] | None = None,
    omit: str | None = None,
    missing_extra_url: bool = False,
) -> dict[str, Any]:
    trust_bands = {
        "court": {"badge": "preview", "stage": "CAL"},
        "body": {"badge": "preview", "stage": "BODY"},
        "ball": {"badge": "low_confidence", "stage": "BALL"},
        "paddle": {"badge": "preview", "stage": "RKT"},
    }
    summary = {
        "status": status,
        "missing_capabilities": missing_capabilities or [],
        "trust_bands": trust_bands,
        "video": {"sha256": "a" * 64, "size_bytes": 11},
        "stages": [{"stage": "manifest", "status": "ran"}],
    }
    _write_json(root / "PIPELINE_SUMMARY.json", summary)
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.mp4").write_bytes(b"video-bytes")
    _write_json(root / "source_identity.json", {"sha256": "a" * 64, "size_bytes": 11})
    _write_json(root / "capture_sidecar.json", {"schema_version": 1})
    _write_json(root / "court_calibration.json", {"coordinate_space": "encoded_pixels"})
    _write_json(root / "tracks.json", {"tracks": []})
    _write_json(root / "body_full_clip_gate.json", {"status": "preview"})
    _write_json(root / "ball_track.json", {"frames": []})
    _write_json(root / "ball_track_arc_solved.json", {"arcs": []})
    _write_json(root / "contact_windows.json", {"windows": []})
    _write_json(root / "racket_pose_estimate.json", {"poses": []})
    _write_json(root / "confidence_gated_world.json", {"players": [], "ball": {}})
    _write_json(root / "match_stats.json", {"facts": []})
    _write_json(root / "coaching_card_facts.json", {"facts": []})
    _write_json(root / "trust_bands.json", trust_bands)
    _write_json(
        root / "body_mesh_index" / "body_mesh_index.json",
        {"faces_url": "body_mesh_faces.json", "windows": [{"url": "chunks/window_000.bin.gz"}]},
    )
    _write_json(root / "body_mesh_index" / "body_mesh_faces.json", {"faces": []})
    chunk = root / "body_mesh_index" / "chunks" / "window_000.bin.gz"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"body-chunk")
    _write_json(root / "replay_scene.json", {"court_glb": "assets/court.glb"})
    court = root / "assets" / "court.glb"
    court.parent.mkdir(parents=True, exist_ok=True)
    court.write_bytes(b"court")

    manifest: dict[str, Any] = {
        "clip": "clip_1",
        "video_url": "source.mp4",
        "body_mesh_index_url": "body_mesh_index/body_mesh_index.json",
        "ball_url": "ball_track.json",
        "paddle_url": "racket_pose_estimate.json",
        "virtual_world_url": "confidence_gated_world.json",
        "replay_scene_url": "replay_scene.json",
    }
    if missing_extra_url:
        manifest["label_overlays"] = [{"url": "assets/missing_overlay.png"}]

    if omit == "body":
        shutil.rmtree(root / "body_mesh_index")
        manifest["body_mesh_index_url"] = None
    elif omit == "ball":
        (root / "ball_track.json").unlink()
        manifest["ball_url"] = None
    elif omit == "paddle":
        (root / "racket_pose_estimate.json").unlink()
        manifest["paddle_url"] = None
    elif omit == "assets":
        (root / "assets" / "court.glb").unlink()
        (root / "assets").rmdir()
        (root / "replay_scene.json").unlink()
        manifest["replay_scene_url"] = None

    _write_json(root / "replay_viewer_manifest.json", manifest)
    return summary


class _FakeS3:
    def __init__(self) -> None:
        self.inputs = {
            "raw/u/clip_1/video.mp4": b"video-bytes",
            "raw/u/clip_1/capture_sidecar.json": b"{}",
        }
        self.uploaded: dict[str, bytes] = {}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.inputs[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploaded[key] = Path(filename).read_bytes()

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = str(kwargs["Prefix"])
        keys = sorted(key for key in self.uploaded if key.startswith(prefix))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        for item in kwargs["Delete"]["Objects"]:
            self.uploaded.pop(str(item["Key"]), None)
        return {"Errors": []}


class _InProcessApi:
    def __init__(self, client: TestClient, payload: dict[str, Any]) -> None:
        self.client = client
        self.payload = payload
        self.claimed = False
        self.completion: dict[str, Any] | None = None

    def claim_next_job(self) -> dict[str, Any] | None:
        if self.claimed:
            return None
        self.claimed = True
        return self.payload

    def send_heartbeat(self, job_id: str, *, stage: str, percent: int, message: str) -> None:
        response = self.client.post(
            f"/api/worker/jobs/{job_id}/heartbeat",
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
            json={"stage": stage, "percent": percent, "message": message},
        )
        assert response.status_code == 204, response.text

    def complete_job(self, job_id: str, **payload: Any) -> None:
        response = self.client.post(
            f"/api/worker/jobs/{job_id}/complete",
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
            json=payload,
        )
        assert response.status_code == 200, response.text
        self.completion = response.json()


def _worker_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        api_base_url="in-process",
        worker_bearer_token=WORKER_TOKEN,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        s3_bucket="bucket",
        s3_region="us-east-1",
        pipeline_python="/repo/.venv/bin/python",
        repo_dir="/repo",
        worker_id="worker-1",
        poll_wait_s=0,
        heartbeat_interval_s=9999,
        command_timeout_s=60,
        work_dir=str(tmp_path / "worker"),
    )


def _worker_api(tmp_path: Path) -> tuple[_InProcessApi, Any]:
    db = mongomock.MongoClient()["ns015"]
    db.jobs.insert_one(
        {
            "_id": "job_1",
            "id": "job_1",
            "user_id": "user_1",
            "status": "claimed",
            "progress": {},
            "created_at": 1.0,
        }
    )
    app = FastAPI()
    app.include_router(build_worker_router(db=db, worker_token=WORKER_TOKEN, with_dynamic_eta=lambda doc: doc))
    payload = {
        "job_id": "job_1",
        "clip_id": "clip_1",
        "s3_raw_key": "raw/u/clip_1/video.mp4",
        "s3_sidecar_key": "raw/u/clip_1/capture_sidecar.json",
        "video_filename": "video.mp4",
        "max_frames": None,
        "attempts": 1,
    }
    return _InProcessApi(TestClient(app), payload), db


def test_complete_fixture_passes_policy(tmp_path: Path) -> None:
    root = tmp_path / "complete"
    _make_run(root)

    result = evaluate_bundle(root)

    assert result.status == "complete"
    assert result.missing_capabilities == ()
    assert result.missing_urls == ()


def test_legacy_exit_success_cannot_upgrade_without_bundle_policy() -> None:
    assert (
        gate_reported_status(
            status="succeeded",
            missing_capabilities=None,
            trust_bands=None,
            bundle_policy=None,
        )
        == "partial"
    )


def test_complete_worker_publish_reaches_api_only_after_every_url_check(tmp_path: Path) -> None:
    api, db = _worker_api(tmp_path)
    s3 = _FakeS3()

    def process_runner(job: Any, video_path: Path, sidecar_path: Path | None, out_dir: Path) -> RunResult:
        summary = _make_run(out_dir / job.clip_id)
        return RunResult(status="succeeded", pipeline_summary=summary)

    assert run_once(api, s3, process_runner, _worker_config(tmp_path)) is True
    assert api.completion is not None
    assert api.completion["status"] == "complete"
    assert api.completion["progress"]["stage"] == "Replay ready"
    assert api.completion["result"]["bundle_policy"]["missing_urls"] == []
    assert db.jobs.find_one({"_id": "job_1"})["status"] == "complete"


@pytest.mark.parametrize("capability", ["body", "ball", "paddle", "assets"])
def test_missing_capability_stays_partial_worker_db_api_unchanged(tmp_path: Path, capability: str) -> None:
    api, db = _worker_api(tmp_path)
    s3 = _FakeS3()
    missing = [{"capability": capability, "reason": f"fixture missing {capability}"}]

    def process_runner(job: Any, video_path: Path, sidecar_path: Path | None, out_dir: Path) -> RunResult:
        run_dir = out_dir / job.clip_id
        summary = _make_run(
            run_dir,
            status="partial",
            missing_capabilities=missing,
            omit=capability,
        )
        return RunResult(
            status="succeeded",
            pipeline_stage_summary=summary["stages"],
            pipeline_summary=summary,
        )

    assert run_once(api, s3, process_runner, _worker_config(tmp_path)) is True

    assert api.completion is not None
    assert api.completion["status"] == "partial"
    assert api.completion["missing_capabilities"] == missing
    assert api.completion["result"]["missing_capabilities"] == missing
    assert api.completion["trust_bands"] == api.completion["result"]["trust_bands"]
    assert api.completion["progress"]["stage"] == "Partial result"
    assert "Replay ready" not in api.completion["progress"]["stage"]
    stored = db.jobs.find_one({"_id": "job_1"})
    assert stored["status"] == "partial"
    assert stored["missing_capabilities"] == missing


def test_missing_advertised_url_degrades_after_publish(tmp_path: Path) -> None:
    api, _db = _worker_api(tmp_path)
    s3 = _FakeS3()

    def process_runner(job: Any, video_path: Path, sidecar_path: Path | None, out_dir: Path) -> RunResult:
        summary = _make_run(out_dir / job.clip_id, missing_extra_url=True)
        return RunResult(status="succeeded", pipeline_summary=summary)

    assert run_once(api, s3, process_runner, _worker_config(tmp_path)) is True

    assert api.completion is not None
    assert api.completion["status"] == "partial"
    policy = api.completion["result"]["bundle_policy"]
    assert policy["missing_urls"] == ["assets/missing_overlay.png"]
    assert any(item["capability"] == "advertised_urls" for item in api.completion["missing_capabilities"])


def test_simulated_kill_before_atomic_publish_leaves_no_visible_bundle(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    _make_run(source)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "published"
    real_replace = pipeline_invocation.os.replace

    def killed_replace(source_path: Any, destination_path: Any) -> None:
        if Path(destination_path) == destination:
            raise KeyboardInterrupt("simulated worker kill")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(pipeline_invocation.os, "replace", killed_replace)
    with pytest.raises(KeyboardInterrupt, match="simulated worker kill"):
        stage_manifest_delivery_bundle(
            source_dir=source,
            bundle_dir=destination,
            video_path=video,
            resolve=lambda path: f"bundles/clip_1/{path}",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".published.delivery-*"))


def test_stats_and_coaching_are_staged_before_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_run(source)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "published"

    order = stage_manifest_delivery_bundle(
        source_dir=source,
        bundle_dir=destination,
        video_path=video,
        resolve=lambda path: f"bundles/clip_1/{path}",
    )

    assert order.index(Path("match_stats.json")) < order.index(Path("replay_viewer_manifest.json"))
    assert order.index(Path("coaching_card_facts.json")) < order.index(Path("replay_viewer_manifest.json"))
    manifest = json.loads((destination / "replay_viewer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["match_stats_url"] == "bundles/clip_1/match_stats.json"
    assert manifest["coaching_card_facts_url"] == "bundles/clip_1/coaching_card_facts.json"


def test_stats_created_after_staging_snapshot_cannot_yield_complete(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    _make_run(source)
    (source / "match_stats.json").unlink()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "published"
    original_compact = pipeline_invocation._compact_json_file
    created = False

    def create_late(source_path: Path, destination_path: Path, **kwargs: Any) -> None:
        nonlocal created
        if not created:
            created = True
            _write_json(source / "match_stats.json", {"late": True})
        original_compact(source_path, destination_path, **kwargs)

    monkeypatch.setattr(pipeline_invocation, "_compact_json_file", create_late)
    stage_manifest_delivery_bundle(
        source_dir=source,
        bundle_dir=destination,
        video_path=video,
        resolve=lambda path: f"bundles/clip_1/{path}",
    )

    result = evaluate_bundle(destination)
    assert result.status == "partial"
    assert any(item["capability"] == "stats" for item in result.missing_capabilities)
    assert not (destination / "match_stats.json").exists()
