from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import CourtCalibration, PersonGroundTruth, Tracks, validate_artifact_file


def _ground_truth() -> PersonGroundTruth:
    return PersonGroundTruth.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "outdoor_fixture",
            "source_format": "cvat_video_1_1",
            "source_path": "cvat.zip",
            "fps": 30.0,
            "frames": [
                {
                    "frame_index": 0,
                    "source_frame_id": 1,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [10.0, 20.0, 5.0, 10.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        },
                        {
                            "track_id": 2,
                            "bbox_xywh": [100.0, 20.0, 8.0, 12.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        },
                    ],
                },
                {"frame_index": 1, "source_frame_id": 2, "labels": []},
                {
                    "frame_index": 2,
                    "source_frame_id": 3,
                    "labels": [
                        {
                            "track_id": 1,
                            "bbox_xywh": [12.0, 21.0, 5.0, 10.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        },
                        {
                            "track_id": 2,
                            "bbox_xywh": [101.0, 21.0, 8.0, 12.0],
                            "ignored": False,
                            "visibility": 1.0,
                            "confidence": 1.0,
                            "class_id": None,
                            "class_name": "player",
                            "person_class": True,
                        },
                    ],
                },
            ],
            "summary": {
                "frame_count": 3,
                "valid_label_count": 4,
                "ignored_label_count": 0,
                "track_ids": [1, 2],
                "max_valid_players_per_frame": 2,
            },
        }
    )


def _calibration() -> CourtCalibration:
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 500.0, "dist": [], "source": "test"},
            "image_size": [1000, 1000],
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 10.0],
                "camera_height_m": 10.0,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "world_pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        }
    )


def test_build_cvat_player_review_proxy_tracks_preserves_full_horizon_and_refuses_promotion() -> None:
    from threed.racketsport.cvat_player_review_proxy import build_cvat_player_review_proxy

    result = build_cvat_player_review_proxy(
        ground_truth=_ground_truth(),
        calibration=_calibration(),
        source_ground_truth_path="runs/cvat/outdoor_fixture/person_ground_truth.json",
        source_calibration_path="runs/eval/outdoor_fixture/court_calibration.json",
        output_tracks_path="runs/phase2/outdoor_fixture/cvat_player_review_proxy_tracks.json",
    )

    assert isinstance(result.tracks, Tracks)
    assert result.tracks.fps == pytest.approx(30.0)
    assert [player.id for player in result.tracks.players] == [1, 2]
    assert [len(player.frames) for player in result.tracks.players] == [2, 2]
    assert result.tracks.players[0].frames[0].bbox == pytest.approx((10.0, 20.0, 15.0, 30.0))
    assert result.tracks.players[0].frames[0].world_xy == pytest.approx([12.5, 30.0])

    report = result.report
    assert report["status"] == "review_only_not_ground_truth"
    assert report["review_only"] is True
    assert report["not_ground_truth"] is True
    assert report["not_gate_verified"] is True
    assert report["promote_trk"] is False
    assert report["coverage"]["gt_frame_range"] == {"first": 0, "last": 2}
    assert report["coverage"]["proxy_prediction_frame_range"] == {"first": 0, "last": 2}
    assert report["coverage"]["full_horizon_label_span"] is True
    assert report["coverage"]["gt_labeled_frame_count"] == 2
    assert report["coverage"]["expected_player_frame_count"] == 2
    assert "label-derived proxy" in report["blocker"]


def test_build_cvat_player_review_proxy_scales_cvat_pixels_to_calibration_pixels() -> None:
    from threed.racketsport.cvat_player_review_proxy import build_cvat_player_review_proxy

    result = build_cvat_player_review_proxy(
        ground_truth=_ground_truth(),
        calibration=_calibration().model_copy(update={"image_size": (500, 500)}),
        source_ground_truth_path="runs/cvat/outdoor_fixture/person_ground_truth.json",
        source_calibration_path="runs/eval/outdoor_fixture/court_calibration.json",
        output_tracks_path="runs/phase2/outdoor_fixture/cvat_player_review_proxy_tracks.json",
        source_image_size=(1000, 1000),
    )

    assert result.tracks.players[0].frames[0].world_xy == pytest.approx([6.25, 15.0])
    assert result.report["coordinate_mapping"] == {
        "source_image_size": [1000.0, 1000.0],
        "calibration_image_size": [500.0, 500.0],
        "image_to_calibration_scale_x": 0.5,
        "image_to_calibration_scale_y": 0.5,
    }


def test_cvat_player_review_proxy_cli_writes_noncanonical_tracks_and_report(tmp_path: Path) -> None:
    gt_path = tmp_path / "person_ground_truth.json"
    cal_path = tmp_path / "court_calibration.json"
    out_dir = tmp_path / "proxy"
    gt_path.write_text(json.dumps(_ground_truth().model_dump(mode="json")), encoding="utf-8")
    cal_path.write_text(json.dumps(_calibration().model_dump(mode="json")), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_cvat_player_review_proxy.py",
            "--person-ground-truth",
            str(gt_path),
            "--court-calibration",
            str(cal_path),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert not (out_dir / "tracks.json").exists()
    parsed = validate_artifact_file("tracks", out_dir / "cvat_player_review_proxy_tracks.json")
    assert isinstance(parsed, Tracks)
    report = json.loads((out_dir / "cvat_player_review_proxy_report.json").read_text(encoding="utf-8"))
    assert report["output_tracks_path"].endswith("cvat_player_review_proxy_tracks.json")
    assert report["promote_trk"] is False
    assert report["safe_outputs"]["can_overwrite_canonical_tracks"] is False
    markdown = (out_dir / "CVAT_PLAYER_REVIEW_PROXY_REPORT.md").read_text(encoding="utf-8")
    assert "not TRK promotion" in markdown
