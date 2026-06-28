from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from threed.racketsport import person_tracking_benchmark
from threed.racketsport.person_tracking_benchmark import (
    PersonTrackerCandidate,
    build_person_tracking_report,
    parse_candidate_spec,
    run_person_tracking_candidate,
    score_track_presence,
    write_person_tracking_report,
)
from threed.racketsport.schemas import PlayerTrack, TrackFrame, Tracks


def test_parse_candidate_spec_requires_name_model_and_tracker() -> None:
    candidate = parse_candidate_spec("yolo26n_botsort=yolo26n.pt,configs/racketsport/botsort_reid.yaml")

    assert candidate == PersonTrackerCandidate(
        name="yolo26n_botsort",
        model="yolo26n.pt",
        tracker_config=Path("configs/racketsport/botsort_reid.yaml"),
    )

    with pytest.raises(ValueError, match="name=model,tracker"):
        parse_candidate_spec("missing_parts")


def test_write_person_tracking_report_outputs_markdown_json_and_timing_chart(tmp_path: Path) -> None:
    rows = [
        {
            "clip": "clip_a",
            "variant": "yolo26n_bytetrack",
            "model": "yolo26n.pt",
            "tracker_config": "configs/racketsport/bytetrack.yaml",
            "max_players": 4,
            "wall_time_s": 8.25,
            "effective_fps": 36.36,
            "player_count": 4,
            "track_frame_count": 240,
            "counts": {"accepted": 240, "extra_players_dropped": 1},
            "canonical_safety_audit": {
                "status": "canonical_candidate_not_gate_verified",
                "safe_for_canonical_review": True,
                "diagnostic_only": False,
                "trusted_for_trk_promotion": False,
                "promotion_blockers": ["labeled_idf1_spectator_gate_missing"],
                "safety_blockers": [],
            },
            "overlay_path": "clip_a/yolo26n_bytetrack/track_overlay.mp4",
            "tracks_path": "clip_a/yolo26n_bytetrack/tracks.json",
            "adaptive_body_schedule": {
                "body_execution_summary": {
                    "scheduled_frame_count": 2,
                    "scheduled_player_frame_count": 8,
                }
            },
            "status": "ok",
        },
        {
            "clip": "clip_a",
            "variant": "yolo26n_botsort_reid",
            "model": "yolo26n.pt",
            "tracker_config": "configs/racketsport/botsort_reid.yaml",
            "max_players": 4,
            "wall_time_s": 9.5,
            "effective_fps": 31.58,
            "player_count": 4,
            "track_frame_count": 240,
            "counts": {"accepted": 240, "extra_players_dropped": 0},
            "overlay_path": "clip_a/yolo26n_botsort_reid/track_overlay.mp4",
            "tracks_path": "clip_a/yolo26n_botsort_reid/tracks.json",
            "status": "ok",
        },
    ]

    summary = build_person_tracking_report(rows, device="mps", max_frames=300)
    write_person_tracking_report(summary, out_dir=tmp_path)

    payload = json.loads((tmp_path / "person_tracking_benchmark.json").read_text(encoding="utf-8"))
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert payload["candidate_count"] == 2
    assert payload["device"] == "mps"
    assert "yolo26n_bytetrack" in report
    assert "BODY frames" in report
    assert "Safety" in report
    assert "canonical candidate" in report
    assert "2 / 8" in report
    assert "track_overlay.mp4" in report
    assert (tmp_path / "timing_chart.png").is_file()


def test_score_track_presence_reports_four_player_frame_coverage() -> None:
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[
                    TrackFrame(t=0.0, bbox=(0, 0, 10, 10), world_xy=[0, 0], conf=0.9),
                    TrackFrame(t=1 / 30, bbox=(0, 0, 10, 10), world_xy=[0, 0], conf=0.9),
                ],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                frames=[
                    TrackFrame(t=0.0, bbox=(10, 0, 20, 10), world_xy=[1, 0], conf=0.8),
                    TrackFrame(t=1 / 30, bbox=(10, 0, 20, 10), world_xy=[1, 0], conf=0.8),
                ],
            ),
            PlayerTrack(
                id=3,
                side="far",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(0, 10, 10, 20), world_xy=[0, 1], conf=0.7)],
            ),
            PlayerTrack(
                id=4,
                side="far",
                role="right",
                frames=[TrackFrame(t=0.0, bbox=(10, 10, 20, 20), world_xy=[1, 1], conf=0.7)],
            ),
        ],
        rally_spans=[],
    )

    score = score_track_presence(tracks, total_frames=3, target_players=4)

    assert score == {
        "target_players": 4,
        "total_frames": 3,
        "frames_with_any_player": 2,
        "target_player_frames": 1,
        "target_player_frame_rate": pytest.approx(1 / 3),
        "mean_active_players": pytest.approx(2.0),
        "active_player_histogram": {"0": 1, "2": 1, "4": 1},
        "id_fragmentation": {
            "selected_player_count": 4,
            "short_track_count": 2,
            "min_track_frames": 1,
            "max_track_frames": 2,
        },
    }


def test_tiled_candidate_uses_batched_predict_path_and_records_gpu_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeYolo:
        def __init__(self, model: str) -> None:
            self.model = model

        def track(self, **_kwargs: object) -> object:
            raise AssertionError("tiled candidates must not use model.track")

    def fake_tiled_payload(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "fps": 30.0,
            "frames": [
                {
                    "frame": 0,
                    "detections": [
                        {
                            "bbox": [10.0, 10.0, 30.0, 80.0],
                            "conf": 0.9,
                            "class": "person",
                            "track_id": 1,
                        }
                    ],
                }
            ],
        }

    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[PlayerTrack(id=1, side="near", role="left", frames=[TrackFrame(t=0.0, bbox=(10, 10, 30, 80), world_xy=[0, 0], conf=0.9)])],
        rally_spans=[],
    )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setattr(person_tracking_benchmark, "_load_calibration", lambda _path: object())
    monkeypatch.setattr(person_tracking_benchmark, "_video_fps", lambda _path: 30.0)
    monkeypatch.setattr(person_tracking_benchmark, "_video_size", lambda _path: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "_calibration_resolution", lambda _calibration: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "yolo_tiled_detections_payload", fake_tiled_payload)
    monkeypatch.setattr(person_tracking_benchmark, "build_tracks", lambda *_args, **_kwargs: (tracks, {"accepted": 1}))
    monkeypatch.setattr(person_tracking_benchmark, "render_player_track_overlay", lambda **_kwargs: {"status": "ok"})

    metrics = run_person_tracking_candidate(
        candidate=PersonTrackerCandidate("yolo26n_tiled", "yolo26n.pt", Path("tiled_predict_role_lock")),
        clip="clip_a",
        video_path=tmp_path / "clip.mp4",
        calibration_path=tmp_path / "court_calibration.json",
        out_dir=tmp_path / "out",
        max_players=4,
        max_frames=1,
        device="0",
        imgsz=1280,
        conf=0.17,
        iou=0.6,
        court_margin_m=8.0,
        id_strategy="role_lock",
        batch_size=64,
        half=True,
        crop_regions="full_lr3",
    )

    assert captured["batch_size"] == 64
    assert captured["half"] is True
    assert captured["device"] == "0"
    assert len(captured["crop_regions"]) == 3
    assert metrics["tracker_config"] == "tiled_predict_role_lock"
    assert metrics["batch_size"] == 64
    assert metrics["half"] is True
    assert metrics["crop_region_count"] == 3
    assert metrics["counts"]["tracker_frames"] == 1
    assert metrics["canonical_safety_audit"]["status"] == "diagnostic_only"
    assert metrics["canonical_safety_audit"]["safety_blockers"] == [
        "widened_court_margin_diagnostic_only",
        "missing_expected_players",
    ]
    assert metrics["canonical_safety_audit"]["promotion_blockers"] == [
        "widened_court_margin_diagnostic_only",
        "missing_expected_players",
        "labeled_idf1_spectator_gate_missing",
    ]


def test_tiled_candidate_uses_adaptive_crop_preset_and_records_fallback_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeYolo:
        def __init__(self, model: str) -> None:
            self.model = model

        def track(self, **_kwargs: object) -> object:
            raise AssertionError("adaptive tiled candidates must not use model.track")

    def fake_tiled_payload(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("adaptive crop presets must use the adaptive payload path")

    def fake_adaptive_payload(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "fps": 30.0,
            "frames": [
                {
                    "frame": 0,
                    "detections": [
                        {
                            "bbox": [10.0, 10.0, 30.0, 80.0],
                            "conf": 0.9,
                            "class": "person",
                            "track_id": 1,
                        }
                    ],
                }
            ],
            "crop_eval_count": 3,
            "fallback_frame_count": 1,
            "primary_crop_region_count": 1,
            "fallback_crop_region_count": 2,
            "adaptive_min_detections": 4,
        }

    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[PlayerTrack(id=1, side="near", role="left", frames=[TrackFrame(t=0.0, bbox=(10, 10, 30, 80), world_xy=[0, 0], conf=0.9)])],
        rally_spans=[],
    )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setattr(person_tracking_benchmark, "_load_calibration", lambda _path: object())
    monkeypatch.setattr(person_tracking_benchmark, "_video_fps", lambda _path: 30.0)
    monkeypatch.setattr(person_tracking_benchmark, "_video_size", lambda _path: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "_calibration_resolution", lambda _calibration: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "yolo_tiled_detections_payload", fake_tiled_payload)
    monkeypatch.setattr(person_tracking_benchmark, "yolo_adaptive_tiled_detections_payload", fake_adaptive_payload)
    monkeypatch.setattr(person_tracking_benchmark, "build_tracks", lambda *_args, **_kwargs: (tracks, {"accepted": 1}))
    monkeypatch.setattr(person_tracking_benchmark, "render_player_track_overlay", lambda **_kwargs: {"status": "ok"})

    metrics = run_person_tracking_candidate(
        candidate=PersonTrackerCandidate("yolo26n_adaptive", "yolo26n.pt", Path("tiled_predict_role_lock")),
        clip="clip_a",
        video_path=tmp_path / "clip.mp4",
        calibration_path=tmp_path / "court_calibration.json",
        out_dir=tmp_path / "out",
        max_players=4,
        max_frames=1,
        device="0",
        imgsz=1280,
        conf=0.17,
        iou=0.6,
        court_margin_m=8.0,
        id_strategy="role_lock",
        batch_size=64,
        half=True,
        crop_regions="adaptive_full_tb3",
        adaptive_min_detections=6,
    )

    assert captured["primary_crop_regions"] == ((0.0, 0.0, 1.0, 1.0),)
    assert captured["fallback_crop_regions"] == ((0.0, 0.0, 1.0, 0.58), (0.0, 0.42, 1.0, 1.0))
    assert captured["min_detections"] == 6
    assert metrics["crop_region_count"] == 3
    assert metrics["adaptive_min_detections"] == 6
    assert metrics["counts"]["crop_eval_count"] == 3
    assert metrics["counts"]["fallback_frame_count"] == 1


def test_candidate_can_emit_adaptive_body_schedule_from_candidate_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeYolo:
        def __init__(self, model: str) -> None:
            self.model = model

        def track(self, **_kwargs: object) -> object:
            raise AssertionError("test uses the tiled predict path")

    def fake_tiled_payload(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "fps": 30.0,
            "frames": [
                {
                    "frame": 0,
                    "detections": [
                        {
                            "bbox": [10.0, 10.0, 30.0, 80.0],
                            "conf": 0.9,
                            "class": "person",
                            "track_id": 1,
                        }
                    ],
                }
            ],
        }

    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[TrackFrame(t=0.0, bbox=(10, 10, 30, 80), world_xy=[0, 0], conf=0.9)],
            )
        ],
        rally_spans=[],
    )
    ball_track = tmp_path / "ball_track.json"
    ball_track.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "tracknet",
                "frames": [{"t": 0.0, "xy": [40.0, 50.0], "conf": 0.95, "visible": True}],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )
    contact_windows = tmp_path / "contact_windows.json"
    contact_windows.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [
                    {
                        "type": "contact",
                        "t": 0.0,
                        "frame": 0,
                        "player_id": 1,
                        "confidence": 1.0,
                        "sources": {"audio": 0.0, "wrist_vel": 0.0, "ball_inflection": 0.0, "human_review": 1.0},
                        "window": {"t0": 0.0, "t1": 0.02, "importance": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setattr(person_tracking_benchmark, "_load_calibration", lambda _path: object())
    monkeypatch.setattr(person_tracking_benchmark, "_video_fps", lambda _path: 30.0)
    monkeypatch.setattr(person_tracking_benchmark, "_video_size", lambda _path: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "_calibration_resolution", lambda _calibration: (200.0, 100.0))
    monkeypatch.setattr(person_tracking_benchmark, "yolo_tiled_detections_payload", fake_tiled_payload)
    monkeypatch.setattr(person_tracking_benchmark, "build_tracks", lambda *_args, **_kwargs: (tracks, {"accepted": 1}))
    monkeypatch.setattr(person_tracking_benchmark, "render_player_track_overlay", lambda **_kwargs: {"status": "ok"})

    metrics = run_person_tracking_candidate(
        candidate=PersonTrackerCandidate("yolo26n_adaptive_audit", "yolo26n.pt", Path("tiled_predict_role_lock")),
        clip="clip_a",
        video_path=tmp_path / "clip.mp4",
        calibration_path=tmp_path / "court_calibration.json",
        out_dir=tmp_path / "out",
        max_players=4,
        max_frames=1,
        device="cpu",
        imgsz=1280,
        conf=0.17,
        iou=0.6,
        court_margin_m=8.0,
        id_strategy="role_lock",
        batch_size=64,
        crop_regions="full_lr3",
        ball_track_path=ball_track,
        contact_windows_path=contact_windows,
        expected_players=4,
    )

    frame_plan_path = tmp_path / "out" / "frame_compute_plan.json"
    body_execution_path = tmp_path / "out" / "body_compute_execution.json"
    frame_plan = json.loads(frame_plan_path.read_text(encoding="utf-8"))
    body_execution = json.loads(body_execution_path.read_text(encoding="utf-8"))

    assert captured["device"] == "cpu"
    assert metrics["frame_compute_plan_path"] == str(frame_plan_path)
    assert metrics["body_compute_execution_path"] == str(body_execution_path)
    assert metrics["adaptive_body_schedule"]["frame_plan_summary"]["deep_mesh_frame_count"] == 1
    assert metrics["adaptive_body_schedule"]["body_execution_summary"]["scheduled_player_frame_count"] == 1
    assert frame_plan["frames"][0]["reasons"] == [
        "contact_window",
        "missing_expected_players",
        "reviewed_contact_targeted_body",
    ]
    assert body_execution["scheduled_frames"][0]["target_player_ids"] == [1]
    assert body_execution["summary"]["scheduled_targeted_reviewed_contact_frame_count"] == 1
