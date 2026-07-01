from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.person_track_gt_scoring import (
    build_source_promotion_decision,
    build_scoring_report,
    derive_track_source_id,
    render_scoring_report_markdown,
    score_tracks_against_person_ground_truth,
)
from threed.racketsport.schemas import PersonGroundTruth, PlayerTrack, TrackFrame, Tracks


def _person_label(track_id: int, frame_index: int, x: float) -> dict:
    return {
        "track_id": track_id,
        "bbox_xywh": [x, 0.0, 10.0, 10.0],
        "ignored": False,
        "visibility": 1.0,
        "confidence": 1.0,
        "class_id": None,
        "class_name": "player",
        "person_class": True,
    }


def _ground_truth() -> PersonGroundTruth:
    return PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_a",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [
                {"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 0.0)]},
                {"frame_index": 1, "source_frame_id": 2, "labels": [_person_label(1, 1, 1.0)]},
            ],
            "summary": {
                "frame_count": 2,
                "valid_label_count": 2,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )


def _track_frame(frame_index: int, bbox: tuple[float, float, float, float], world_xy: list[float]) -> TrackFrame:
    return TrackFrame(t=frame_index / 30.0, bbox=bbox, world_xy=world_xy, conf=0.9)


def _tracks_with_switch_fp_and_tail() -> Tracks:
    return Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=1,
                side="near",
                role="left",
                frames=[_track_frame(0, (0.0, 0.0, 10.0, 10.0), [0.0, 0.0])],
            ),
            PlayerTrack(
                id=2,
                side="near",
                role="right",
                frames=[_track_frame(1, (1.0, 0.0, 11.0, 10.0), [0.0, 0.0])],
            ),
            PlayerTrack(
                id=3,
                side="far",
                role="left",
                frames=[_track_frame(1, (50.0, 50.0, 60.0, 60.0), [8.0, 8.0])],
            ),
            PlayerTrack(
                id=4,
                side="far",
                role="right",
                frames=[_track_frame(3, (0.0, 0.0, 10.0, 10.0), [0.0, 0.0])],
            ),
        ],
        rally_spans=[],
    )


def test_score_tracks_against_person_ground_truth_catches_switches_false_positives_and_caps_tail() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_ground_truth(),
        tracks=_tracks_with_switch_fp_and_tail(),
        candidate="candidate_a",
        tracks_path="runs/example/clip_a/candidate_a/tracks.json",
        iou_threshold=0.5,
    )

    assert score["idf1"] == pytest.approx(0.4)
    assert score["id_switches"] == 1
    assert score["false_positives"] == 1
    assert score["false_negatives"] == 0
    assert score["four_player_coverage"] == pytest.approx(0.5)
    assert score["spectator_or_background_false_positives"] == 1
    assert score["off_court_false_positive_frames"] == 1
    assert score["off_court_false_positive_track_ids"] == [3]
    assert score["outside_gt_prediction_count"] == 1


def test_score_tracks_against_person_ground_truth_reports_switch_events_and_temporal_coverage() -> None:
    score = score_tracks_against_person_ground_truth(
        ground_truth=_ground_truth(),
        tracks=_tracks_with_switch_fp_and_tail(),
        candidate="candidate_a",
        tracks_path="runs/example/clip_a/candidate_a/tracks.json",
        iou_threshold=0.5,
    )

    assert score["identity_switch_event_count"] == 1
    assert score["identity_switch_events"] == [
        {
            "frame_index": 1,
            "gt_track_id": 1,
            "previous_pred_track_id": 1,
            "new_pred_track_id": 2,
            "previous_match_frame_index": 0,
            "frames_since_previous_match": 1,
            "iou": pytest.approx(1.0),
        }
    ]
    assert score["identity_switch_transitions"] == [
        {
            "gt_track_id": 1,
            "previous_pred_track_id": 1,
            "new_pred_track_id": 2,
            "count": 1,
            "first_frame_index": 1,
            "last_frame_index": 1,
        }
    ]
    assert score["temporal_coverage"]["gt_frame_range"] == {"first": 0, "last": 1}
    assert score["temporal_coverage"]["prediction_frame_range"] == {"first": 0, "last": 1}
    assert score["temporal_coverage"]["gt_detections_after_last_prediction"] == 0


def test_score_tracks_against_person_ground_truth_scales_track_boxes_to_gt_pixels() -> None:
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_scaled",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [
                {"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 20.0)]},
            ],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[_track_frame(0, (10.0, 0.0, 20.0, 5.0), [0.0, 0.0])],
            )
        ],
        rally_spans=[],
    )

    score = score_tracks_against_person_ground_truth(
        ground_truth=gt,
        tracks=tracks,
        candidate="half_scale_tracks",
        tracks_path="runs/example/clip_scaled/half_scale_tracks/tracks.json",
        bbox_scale_x=2.0,
        bbox_scale_y=2.0,
    )

    assert score["idf1"] == pytest.approx(1.0)
    assert score["false_positives"] == 0
    assert score["false_negatives"] == 0


def test_build_source_promotion_decision_requires_all_clips_and_clean_identity_gate() -> None:
    rows = [
        {
            "clip_id": "clip_a",
            "idf1": 0.91,
            "id_switches": 0,
            "spectator_or_background_false_positives": 0,
            "off_court_false_positive_frames": 0,
            "four_player_coverage": 0.96,
        },
        {
            "clip_id": "clip_b",
            "idf1": 0.84,
            "id_switches": 0,
            "spectator_or_background_false_positives": 0,
            "off_court_false_positive_frames": 0,
            "four_player_coverage": 0.97,
        },
    ]

    decision = build_source_promotion_decision(rows, required_clip_ids=["clip_a", "clip_b", "clip_c"])

    assert decision["promote"] is False
    assert "missing_required_clips:clip_c" in decision["blockers"]
    assert "clip_b:idf1_below_0.85" in decision["blockers"]


def test_scoring_report_adds_failure_mode_summary_per_row_and_source() -> None:
    rows = [
        {
            "clip_id": "clip_a",
            "track_source_id": "source_a",
            "idf1": 0.5,
            "mota": 0.2,
            "id_switches": 3,
            "spectator_or_background_false_positives": 8,
            "off_court_false_positive_frames": 2,
            "false_positives": 8,
            "false_negatives": 40,
            "gt_detections": 100,
            "pred_detections": 68,
            "matches": 60,
            "four_player_coverage": 0.75,
            "expected_four_player_frames": 20,
            "exact_four_player_frames": 15,
            "track_count": 4,
            "tracks_path": "runs/source_a/clip_a/tracks.json",
        },
        {
            "clip_id": "clip_b",
            "track_source_id": "source_a",
            "idf1": 0.7,
            "mota": 0.3,
            "id_switches": 1,
            "spectator_or_background_false_positives": 4,
            "off_court_false_positive_frames": 3,
            "false_positives": 4,
            "false_negatives": 6,
            "gt_detections": 50,
            "pred_detections": 48,
            "matches": 44,
            "four_player_coverage": 0.9,
            "expected_four_player_frames": 10,
            "exact_four_player_frames": 9,
            "track_count": 4,
            "tracks_path": "runs/source_a/clip_b/tracks.json",
        },
    ]

    report = build_scoring_report(rows, required_clip_ids=["clip_a", "clip_b"], iou_threshold=0.5)
    source = report["sources"][0]

    assert source["failure_analysis"]["primary_failure_mode"] == "missing_gt_detections"
    assert source["failure_analysis"]["modes"][0]["count"] == 46
    assert source["rows"][0]["primary_failure_mode"] == "missing_gt_detections"
    assert {mode["mode"] for mode in source["rows"][0]["failure_modes"]} >= {
        "id_switches",
        "spectator_or_background_false_positives",
        "off_court_false_positives",
        "four_player_coverage_gap",
    }
    markdown = render_scoring_report_markdown(report)
    source_line = next(line for line in markdown.splitlines() if line.startswith("| `source_a` |"))
    assert source_line.count("|") == 14
    assert "| missing_gt_detections |" in source_line


def test_derive_track_source_id_groups_phase2_and_canonical_paths() -> None:
    clip_ids = ["burlington_gold_0300_low_steep_corner", "wolverine_mixed_0200_mid_steep_corner"]

    phase2 = derive_track_source_id(
        "runs/phase2/person_tracking_h100_final_modes_fullclips/yolo26n_fulltb3/"
        "burlington_gold_0300_low_steep_corner/yolo26n_fulltb3/tracks.json",
        clip_ids=clip_ids,
    )
    canonical = derive_track_source_id(
        "runs/eval0/prototype_gate_h100_v2/burlington_gold_0300_low_steep_corner/tracks.json",
        clip_ids=clip_ids,
    )

    assert phase2 == "phase2/person_tracking_h100_final_modes_fullclips/yolo26n_fulltb3"
    assert canonical == "eval0/prototype_gate_h100_v2/canonical_tracks"


def test_score_person_track_sources_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_a"
    gt_dir.mkdir(parents=True)
    (gt_dir / "person_ground_truth.json").write_text(
        json.dumps(_ground_truth().model_dump(mode="json")),
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    track_dir = runs_root / "phase2" / "source_a" / "clip_a" / "candidate_a"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(
        json.dumps(_tracks_with_switch_fp_and_tail().model_dump(mode="json")),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(runs_root),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    assert payload["track_file_count"] == 1
    assert payload["sources"][0]["decision"]["status"] == "do_not_promote"
    assert (out_dir / "PERSON_TRACK_GT_SCORING_REPORT.md").exists()


def test_score_person_track_sources_cli_applies_metrics_coordinate_scale(tmp_path: Path) -> None:
    cvat_root = tmp_path / "cvat"
    gt_dir = cvat_root / "clip_scaled"
    gt_dir.mkdir(parents=True)
    gt = PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "clip_scaled",
            "source_format": "cvat_video_1_1",
            "source_path": "synthetic.zip",
            "fps": 30.0,
            "frames": [{"frame_index": 0, "source_frame_id": 1, "labels": [_person_label(1, 0, 20.0)]}],
            "summary": {
                "frame_count": 1,
                "valid_label_count": 1,
                "ignored_label_count": 0,
                "track_ids": [1],
                "max_valid_players_per_frame": 1,
            },
        }
    )
    (gt_dir / "person_ground_truth.json").write_text(json.dumps(gt.model_dump(mode="json")), encoding="utf-8")
    tracks = Tracks(
        schema_version=1,
        fps=30.0,
        players=[
            PlayerTrack(
                id=7,
                side="near",
                role="left",
                frames=[_track_frame(0, (10.0, 0.0, 20.0, 5.0), [0.0, 0.0])],
            )
        ],
        rally_spans=[],
    )
    track_dir = tmp_path / "runs" / "phase2" / "source_scaled" / "clip_scaled" / "candidate_scaled"
    track_dir.mkdir(parents=True)
    (track_dir / "tracks.json").write_text(json.dumps(tracks.model_dump(mode="json")), encoding="utf-8")
    (track_dir / "metrics.json").write_text(
        json.dumps(
            {
                "counts": {
                    "source_width": 1920,
                    "source_height": 1080,
                    "calibration_width": 960,
                    "calibration_height": 540,
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_person_track_sources.py",
            "--cvat-root",
            str(cvat_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "person_track_gt_scoring_report.json").read_text(encoding="utf-8"))
    row = payload["sources"][0]["rows"][0]
    assert row["bbox_scale_x"] == pytest.approx(2.0)
    assert row["bbox_scale_y"] == pytest.approx(2.0)
    assert row["idf1"] == pytest.approx(1.0)
