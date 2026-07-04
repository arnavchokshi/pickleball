import json
from pathlib import Path

import pytest

from threed.racketsport.court_review_artifacts import (
    COURT_REVIEW_ARTIFACT_TYPE,
    COURT_REVIEW_SCHEMA_VERSION,
    build_reviewed_court_artifacts,
    save_reviewed_court_artifacts,
    validate_reviewed_court_points,
)
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES, CourtCalibration


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
    return {name: {"xy": [float(xs[name]), float(ys[name])], "confidence": 0.74} for name in PICKLEBALL_COURT_KEYPOINT_NAMES}


def _adjusted_points() -> dict[str, list[float]]:
    adjusted = {name: list(point["xy"]) for name, point in _prediction_points().items()}
    adjusted["near_left_corner"] = [174.0, 526.0]
    return adjusted


def test_build_reviewed_court_artifacts_preserves_prediction_manual_moves_and_pipeline_calibration() -> None:
    artifact, calibration = build_reviewed_court_artifacts(
        video_id="match_01",
        video_path="/uploads/match_01.mp4",
        video_sha256="a" * 64,
        image_size=(1000, 600),
        frame_index=42,
        frame_time_s=1.4,
        auto_prediction_source="court_detector_v2:selected_hypothesis=hypothesis_0001",
        predicted_points=_prediction_points(),
        adjusted_points=_adjusted_points(),
        created_at="2026-07-04T12:00:00Z",
    )

    assert artifact["schema_version"] == COURT_REVIEW_SCHEMA_VERSION
    assert artifact["artifact_type"] == COURT_REVIEW_ARTIFACT_TYPE
    assert artifact["review_status"] == "human_reviewed"
    assert artifact["auto_prediction"]["verified"] is False
    assert artifact["source_video"]["sha256"] == "a" * 64
    assert artifact["points"]["near_left_corner"]["manual_moved"] is True
    assert artifact["points"]["near_right_corner"]["manual_moved"] is False
    assert artifact["validation"]["status"] == "pass"
    assert artifact["training"]["usable_for_court_detector_training"] is True

    assert calibration["schema_version"] == 1
    assert calibration["sport"] == "pickleball"
    assert len(calibration["image_pts"]) == len(PICKLEBALL_COURT_KEYPOINT_NAMES)
    assert "human_reviewed_court_correction" in calibration["capture_quality"]["reasons"]
    CourtCalibration.model_validate(calibration)


def test_build_auto_predicted_court_artifacts_are_not_training_ready() -> None:
    predicted = _prediction_points()
    adjusted = {name: list(point["xy"]) for name, point in predicted.items()}

    artifact, calibration = build_reviewed_court_artifacts(
        video_id="match_01",
        video_path="/uploads/match_01.mp4",
        video_sha256="a" * 64,
        image_size=(1000, 600),
        frame_index=42,
        frame_time_s=1.4,
        auto_prediction_source="court_detector_v2:selected_hypothesis=hypothesis_0001",
        predicted_points=predicted,
        adjusted_points=adjusted,
        created_at="2026-07-04T12:00:00Z",
        review_status="auto_predicted_unreviewed",
    )

    assert artifact["review_status"] == "auto_predicted_unreviewed"
    assert artifact["pipeline"]["trust"] == "auto_predicted_unreviewed_court_layout"
    assert artifact["training"]["usable_for_court_detector_training"] is False
    assert artifact["training"]["training_policy"] == "auto_prediction_not_training_ready"
    assert all(point["manual_moved"] is False for point in artifact["points"].values())
    assert "auto_predicted_court_layout_unreviewed" in calibration["capture_quality"]["reasons"]


def test_validate_reviewed_court_points_warns_for_missing_low_confidence_and_bad_geometry() -> None:
    predicted = _prediction_points()
    adjusted = _adjusted_points()
    del adjusted["near_nvz_center"]
    predicted["near_nvz_left"]["confidence"] = 0.1
    adjusted["far_left_corner"] = [900.0, 560.0]

    report = validate_reviewed_court_points(
        adjusted_points=adjusted,
        predicted_points=predicted,
        image_size=(1000, 600),
    )

    codes = {warning["code"] for warning in report["warnings"]}
    assert {"missing_point", "low_prediction_confidence", "bad_geometry"} <= codes
    assert report["status"] == "warn"


def test_save_reviewed_court_artifacts_writes_training_index_and_calibration(tmp_path: Path) -> None:
    artifact, calibration = build_reviewed_court_artifacts(
        video_id="match_01",
        video_path="/uploads/match_01.mp4",
        video_sha256="b" * 64,
        image_size=(1000, 600),
        frame_index=42,
        frame_time_s=1.4,
        auto_prediction_source="court_detector_v2:selected_hypothesis=hypothesis_0001",
        predicted_points=_prediction_points(),
        adjusted_points=_adjusted_points(),
        created_at="2026-07-04T12:00:00Z",
    )

    saved = save_reviewed_court_artifacts(
        artifact=artifact,
        court_calibration=calibration,
        root=tmp_path,
    )

    review_path = Path(saved["review_path"])
    calibration_path = Path(saved["court_calibration_path"])
    index_path = Path(saved["index_path"])
    assert review_path.is_file()
    assert calibration_path.is_file()
    assert index_path.is_file()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["artifact_type"] == "racketsport_reviewed_court_calibration_index"
    assert index["entries"][0]["review_path"] == str(review_path)
    assert index["entries"][0]["video_sha256"] == "b" * 64
    assert index["entries"][0]["review_status"] == "human_reviewed"
    assert index["entries"][0]["training_policy"] == "human_reviewed_not_eval_promoted"
