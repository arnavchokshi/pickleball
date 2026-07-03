from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.player_source_selection import (
    SourceSelectionConfig,
    source_select_global_four_player_tracks,
    source_select_four_player_tracks,
)
from threed.racketsport.schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks


def _calibration() -> CourtCalibration:
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 50.0, "cy": 50.0, "source": "synthetic"},
            "image_size": [100, 100],
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 10.0],
                "camera_height_m": 10.0,
            },
            "homography": [[1.0, 0.0, 50.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]],
            "reprojection_error_px": {"median": 1.0, "p95": 2.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            "world_pts": [[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, 50.0, 0.0], [-50.0, 50.0, 0.0]],
        }
    )


def _frame(frame_idx: int, x: float, y: float) -> TrackFrame:
    return TrackFrame(t=frame_idx / 10.0, bbox=(x + 49.5, y + 48.0, x + 50.5, y + 50.0), world_xy=(x, y), conf=0.9)


def _seed_tracks() -> Tracks:
    return Tracks(
        schema_version=1,
        fps=10.0,
        players=[
            PlayerTrack(id=1, side="near", role="left", frames=[_frame(0, -1.0, -2.0), _frame(2, 1.0, -2.0)]),
            PlayerTrack(id=2, side="near", role="right", frames=[_frame(0, 2.0, -2.0), _frame(1, 2.0, -2.0), _frame(2, 2.0, -2.0)]),
        ],
        rally_spans=[],
    )


def _detections_payload(*detections: dict) -> dict:
    frames = []
    for frame_idx in range(3):
        frames.append(
            {
                "frame": frame_idx,
                "detections": [detection for detection in detections if detection["frame"] == frame_idx],
            }
        )
    return {"schema_version": 1, "fps": 10.0, "frames": frames}


def _detections_payload_for_frame_count(frame_count: int, *detections: dict) -> dict:
    frames = []
    for frame_idx in range(frame_count):
        frames.append(
            {
                "frame": frame_idx,
                "detections": [detection for detection in detections if detection["frame"] == frame_idx],
            }
        )
    return {"schema_version": 1, "fps": 10.0, "frames": frames}


def _embedding_payload(*rows: dict) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_reid_embeddings",
        "status": "source_only_learned_embedding_export",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "feature_dim": 2,
        "l2_normalized": True,
        "detections": list(rows),
    }


def _emb_row(frame_idx: int, source_track_id: int, detection_index: int, embedding: list[float], bbox: list[float]) -> dict:
    return {
        "frame": frame_idx,
        "source_track_id": source_track_id,
        "track_id": source_track_id,
        "detection_index": detection_index,
        "bbox": bbox,
        "embedding": embedding,
        "feature_dim": 2,
    }


def _det(frame_idx: int, x: float, y: float, *, conf: float = 0.8, track_id: int = 99) -> dict:
    return {
        "frame": frame_idx,
        "bbox": [x + 49.5, y + 48.0, x + 50.5, y + 50.0],
        "class": "person",
        "conf": conf,
        "track_id": track_id,
    }


def test_source_selection_fills_short_seeded_gap_from_tracked_detections() -> None:
    repaired, summary = source_select_four_player_tracks(
        _detections_payload(_det(1, 0.0, -2.0)),
        _calibration(),
        seed_tracks=_seed_tracks(),
        config=SourceSelectionConfig(expected_players=2, max_gap_fill_frames=3, max_fill_distance_m=0.25),
    )

    player = next(player for player in repaired.players if player.id == 1)

    assert [int(round(frame.t * repaired.fps)) for frame in player.frames] == [0, 1, 2]
    assert summary.filled_frame_count == 1
    assert summary.output_player_count == 2
    assert summary.uses_cvat_labels is False


def test_source_selection_skips_fill_when_candidate_overlaps_existing_player() -> None:
    repaired, summary = source_select_four_player_tracks(
        _detections_payload(_det(1, 2.0, -2.0)),
        _calibration(),
        seed_tracks=_seed_tracks(),
        config=SourceSelectionConfig(expected_players=2, max_gap_fill_frames=3, max_fill_distance_m=4.0),
    )

    player = next(player for player in repaired.players if player.id == 1)

    assert [int(round(frame.t * repaired.fps)) for frame in player.frames] == [0, 2]
    assert summary.filled_frame_count == 0
    assert summary.skipped_overlap_count == 1
    assert summary.status == "ok"


def test_global_source_selection_controls_per_frame_cardinality_from_crowded_source_pool() -> None:
    detections = []
    court_players = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)]
    for frame_idx in range(3):
        for source_id, (x, y) in enumerate(court_players, start=1):
            detections.append(_det(frame_idx, x + frame_idx * 0.1, y, conf=0.9, track_id=source_id))
        detections.append(_det(frame_idx, -2.0 + frame_idx * 0.1, -2.0, conf=0.95, track_id=99))
        detections.append(_det(frame_idx, 0.0, 0.0, conf=0.1, track_id=100))

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(3, *detections),
        _calibration(),
        seed_tracks=None,
        config=SourceSelectionConfig(expected_players=4, overlap_iou_threshold=0.5),
    )

    per_frame_counts = {}
    for player in selected.players:
        for frame in player.frames:
            frame_idx = int(round(frame.t * selected.fps))
            per_frame_counts[frame_idx] = per_frame_counts.get(frame_idx, 0) + 1

    assert len(selected.players) == 4
    assert per_frame_counts == {0: 4, 1: 4, 2: 4}
    assert summary.global_selected_frame_count == 12
    assert summary.exact_cardinality_frame_count == 3
    assert summary.source_only is True
    assert summary.uses_cvat_labels is False


def test_global_source_selection_preserves_source_identity_through_crossing() -> None:
    detections = []
    crossing_tracks = {
        1: [(-2.0, -1.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (2.0, -1.0)],
        2: [(2.0, 1.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-2.0, 1.0)],
        3: [(-2.0, 2.5)] * 5,
        4: [(2.0, 2.5)] * 5,
    }
    for source_id, points in crossing_tracks.items():
        for frame_idx, (x, y) in enumerate(points):
            detections.append(_det(frame_idx, x, y, conf=0.9, track_id=source_id))

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(5, *detections),
        _calibration(),
        seed_tracks=None,
        config=SourceSelectionConfig(expected_players=4, overlap_iou_threshold=0.5),
    )

    player_1 = next(player for player in selected.players if player.id == 1)
    player_2 = next(player for player in selected.players if player.id == 2)

    assert [round(frame.world_xy[0], 1) for frame in player_1.frames] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert [round(frame.world_xy[0], 1) for frame in player_2.frames] == [2.0, 1.0, 0.0, -1.0, -2.0]
    assert summary.identity_reassignment_count == 0
    assert summary.exact_cardinality_frame_count == 5


def test_global_source_selection_fails_closed_for_overlapping_and_off_court_candidates() -> None:
    detections = [
        _det(0, -2.0, -2.0, conf=0.9, track_id=1),
        _det(0, 2.0, -2.0, conf=0.9, track_id=2),
        _det(0, -2.0, 2.0, conf=0.9, track_id=3),
        _det(0, -2.0, 2.0, conf=0.88, track_id=4),
        _det(0, 60.0, 0.0, conf=0.99, track_id=5),
    ]

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(1, *detections),
        _calibration(),
        seed_tracks=None,
        config=SourceSelectionConfig(expected_players=4, overlap_iou_threshold=0.5),
    )

    frame_count = sum(len(player.frames) for player in selected.players)

    assert frame_count == 3
    assert summary.exact_cardinality_frame_count == 0
    assert summary.skipped_overlap_count >= 1
    assert summary.source_candidate_detections == 5
    assert summary.source_candidate_kept == 3


def test_global_source_selection_penalizes_margin_band_candidates_when_strict_court_alternative_exists() -> None:
    detections = [
        _det(0, -2.0, -2.0, conf=0.9, track_id=1),
        _det(0, 2.0, -2.0, conf=0.9, track_id=2),
        _det(0, -2.0, 2.0, conf=0.9, track_id=3),
        _det(0, 4.0, 0.0, conf=0.99, track_id=4),
        _det(0, 2.0, 2.0, conf=0.55, track_id=5),
    ]

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(1, *detections),
        _calibration(),
        seed_tracks=None,
        config=SourceSelectionConfig(
            expected_players=4,
            court_margin_m=2.0,
            overlap_iou_threshold=0.5,
            confidence_reward_weight=2.0,
        ),
    )

    selected_points = {
        (round(frame.world_xy[0], 1), round(frame.world_xy[1], 1))
        for player in selected.players
        for frame in player.frames
    }

    assert (2.0, 2.0) in selected_points
    assert (4.0, 0.0) not in selected_points
    assert summary.exact_cardinality_frame_count == 1


def test_global_source_selection_can_leave_gap_instead_of_forcing_penalized_margin_band_candidate() -> None:
    detections = [
        _det(0, -2.0, -2.0, conf=0.9, track_id=1),
        _det(0, 2.0, -2.0, conf=0.9, track_id=2),
        _det(0, -2.0, 2.0, conf=0.9, track_id=3),
        _det(0, 4.0, 0.0, conf=0.99, track_id=4),
    ]

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(1, *detections),
        _calibration(),
        seed_tracks=None,
        config=SourceSelectionConfig(
            expected_players=4,
            court_margin_m=2.0,
            margin_band_penalty=20.0,
            overlap_iou_threshold=0.5,
            confidence_reward_weight=2.0,
        ),
    )

    selected_points = {
        (round(frame.world_xy[0], 1), round(frame.world_xy[1], 1))
        for player in selected.players
        for frame in player.frames
    }

    assert (4.0, 0.0) not in selected_points
    assert len(selected_points) == 3
    assert summary.exact_cardinality_frame_count == 0
    assert summary.global_selected_frame_count == 3


def test_global_source_selection_can_use_embeddings_as_weak_ambiguous_tiebreaker() -> None:
    detections = [
        _det(0, -1.0, 0.0, conf=0.9, track_id=10),
        _det(0, 1.0, 0.0, conf=0.9, track_id=20),
        _det(1, 0.0, 0.0, conf=0.9, track_id=30),
        _det(1, 1.2, 0.0, conf=0.9, track_id=40),
    ]
    embeddings = _embedding_payload(
        _emb_row(0, 10, 0, [1.0, 0.0], detections[0]["bbox"]),
        _emb_row(0, 20, 1, [0.0, 1.0], detections[1]["bbox"]),
        _emb_row(1, 30, 0, [0.0, 1.0], detections[2]["bbox"]),
        _emb_row(1, 40, 1, [1.0, 0.0], detections[3]["bbox"]),
    )

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(2, *detections),
        _calibration(),
        seed_tracks=None,
        embedding_payload=embeddings,
        config=SourceSelectionConfig(
            expected_players=2,
            overlap_iou_threshold=0.0,
            max_global_step_m=5.0,
            source_id_switch_penalty=0.0,
            continuity_weight=0.05,
            confidence_reward_weight=0.0,
            embedding_weight=0.5,
        ),
    )

    player_10 = next(player for player in selected.players if player.id == 10)
    player_20 = next(player for player in selected.players if player.id == 20)

    assert [round(frame.world_xy[0], 1) for frame in player_10.frames] == [-1.0, 1.2]
    assert [round(frame.world_xy[0], 1) for frame in player_20.frames] == [1.0, 0.0]
    assert summary.embedding_joined_count == 4
    assert summary.embedding_cost_applied_count > 0
    assert summary.uses_embeddings is True
    assert summary.uses_cvat_labels is False
    assert summary.source_only is True


def test_global_source_selection_ignores_embedding_export_until_weight_opted_in() -> None:
    detections = [
        _det(0, -1.0, 0.0, conf=0.9, track_id=10),
        _det(0, 1.0, 0.0, conf=0.9, track_id=20),
        _det(1, 0.0, 0.0, conf=0.9, track_id=30),
        _det(1, 1.2, 0.0, conf=0.9, track_id=40),
    ]
    embeddings = _embedding_payload(
        _emb_row(0, 10, 0, [1.0, 0.0], detections[0]["bbox"]),
        _emb_row(0, 20, 1, [0.0, 1.0], detections[1]["bbox"]),
        _emb_row(1, 30, 0, [0.0, 1.0], detections[2]["bbox"]),
        _emb_row(1, 40, 1, [1.0, 0.0], detections[3]["bbox"]),
    )

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(2, *detections),
        _calibration(),
        seed_tracks=None,
        embedding_payload=embeddings,
        config=SourceSelectionConfig(
            expected_players=2,
            overlap_iou_threshold=0.0,
            max_global_step_m=5.0,
            source_id_switch_penalty=0.0,
            continuity_weight=0.05,
            confidence_reward_weight=0.0,
        ),
    )

    player_10 = next(player for player in selected.players if player.id == 10)
    player_20 = next(player for player in selected.players if player.id == 20)

    assert SourceSelectionConfig().embedding_weight == 0.0
    assert [round(frame.world_xy[0], 1) for frame in player_10.frames] == [-1.0, 0.0]
    assert [round(frame.world_xy[0], 1) for frame in player_20.frames] == [1.0, 1.2]
    assert summary.uses_embeddings is False
    assert summary.embedding_cost_applied_count == 0


def test_global_source_selection_can_join_raw_scale_embedding_bboxes() -> None:
    detections = [
        _det(0, -1.0, 0.0, conf=0.9, track_id=10),
        _det(1, -0.8, 0.0, conf=0.9, track_id=10),
    ]
    embeddings = _embedding_payload(
        _emb_row(0, 10, 0, [1.0, 0.0], [value * 2.0 for value in detections[0]["bbox"]]),
        _emb_row(1, 10, 0, [1.0, 0.0], [value * 2.0 for value in detections[1]["bbox"]]),
    )

    selected, summary = source_select_global_four_player_tracks(
        _detections_payload_for_frame_count(2, *detections),
        _calibration(),
        seed_tracks=None,
        embedding_payload=embeddings,
        config=SourceSelectionConfig(
            expected_players=1,
            embedding_weight=0.5,
            embedding_bbox_scale=2.0,
        ),
    )

    assert len(selected.players) == 1
    assert summary.embedding_joined_count == 2
    assert summary.embedding_missing_count == 0
    assert summary.embedding_cost_applied_count == 1


def test_select_source_person_tracks_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/select_source_person_tracks.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Source-only four-player selection" in completed.stdout
    assert "--embedding-export" in completed.stdout
    assert "--embedding-weight" in completed.stdout
    assert "--margin-band-penalty" in completed.stdout


def test_select_source_person_tracks_cli_keeps_embedding_weight_opt_in(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    out_dir = tmp_path / "out"
    source_dir.mkdir()
    detections = [
        _det(0, -1.0, 0.0, conf=0.9, track_id=10),
        _det(0, 1.0, 0.0, conf=0.9, track_id=20),
        _det(1, 0.0, 0.0, conf=0.9, track_id=30),
        _det(1, 1.2, 0.0, conf=0.9, track_id=40),
    ]
    embeddings = _embedding_payload(
        _emb_row(0, 10, 0, [1.0, 0.0], detections[0]["bbox"]),
        _emb_row(0, 20, 1, [0.0, 1.0], detections[1]["bbox"]),
        _emb_row(1, 30, 0, [0.0, 1.0], detections[2]["bbox"]),
        _emb_row(1, 40, 1, [1.0, 0.0], detections[3]["bbox"]),
    )
    (source_dir / "tracked_detections.json").write_text(
        json.dumps(_detections_payload_for_frame_count(2, *detections)),
        encoding="utf-8",
    )
    (source_dir / "tracks.json").write_text(json.dumps(_seed_tracks().model_dump(mode="json")), encoding="utf-8")
    calibration_path = tmp_path / "court_calibration.json"
    embedding_path = tmp_path / "embeddings.json"
    calibration_path.write_text(json.dumps(_calibration().model_dump(mode="json")), encoding="utf-8")
    embedding_path.write_text(json.dumps(embeddings), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/select_source_person_tracks.py",
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(out_dir),
            "--calibration",
            str(calibration_path),
            "--embedding-export",
            str(embedding_path),
            "--expected-players",
            "2",
            "--overlap-iou-threshold",
            "0.0",
            "--max-global-step-m",
            "5.0",
            "--source-id-switch-penalty",
            "0.0",
            "--continuity-weight",
            "0.05",
            "--confidence-reward-weight",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((out_dir / "source_selection_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["embedding_weight"] == 0.0
    assert summary["summary"]["uses_embeddings"] is False
    assert summary["summary"]["embedding_cost_applied_count"] == 0

    opt_in_dir = tmp_path / "out_opt_in"
    opt_in = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/select_source_person_tracks.py",
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(opt_in_dir),
            "--calibration",
            str(calibration_path),
            "--embedding-export",
            str(embedding_path),
            "--embedding-weight",
            "0.5",
            "--expected-players",
            "2",
            "--overlap-iou-threshold",
            "0.0",
            "--max-global-step-m",
            "5.0",
            "--source-id-switch-penalty",
            "0.0",
            "--continuity-weight",
            "0.05",
            "--confidence-reward-weight",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert opt_in.returncode == 0, opt_in.stderr
    opt_in_summary = json.loads((opt_in_dir / "source_selection_summary.json").read_text(encoding="utf-8"))
    assert opt_in_summary["config"]["embedding_weight"] == 0.5
    assert opt_in_summary["summary"]["uses_embeddings"] is True
    assert opt_in_summary["summary"]["embedding_cost_applied_count"] > 0
