from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_keypoint_labelspace_diagnostic import (
    analyze_court_keypoint_labelspace,
    build_court_keypoint_labelspace_report,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibration_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[36.0, 2.0, 320.0], [1.0, 18.0, 240.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0, "dist": [], "source": "test"},
        "image_size": [1280, 720],
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 2.5],
            "camera_height_m": 2.5,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[210.0, 120.0], [970.0, 120.0], [990.0, 610.0], [190.0, 610.0]],
        "world_pts": [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0], [3.048, 6.7056, 0.0], [-3.048, 6.7056, 0.0]],
    }


def _trusted_projection(calibration: dict) -> dict[str, list[float]]:
    return {
        point.name: project_planar_points(calibration["homography"], [point.world_xyz_m[:2]])[0]
        for point in PICKLEBALL_KEYPOINTS
    }


def _prediction_payload(
    calibration: dict,
    *,
    mode: str,
    confidence: float = 0.9,
) -> dict:
    trusted = _trusted_projection(calibration)
    names = [point.name for point in PICKLEBALL_KEYPOINTS]
    keypoints = {}
    for index, name in enumerate(names):
        x, y = trusted[name]
        if mode == "scale":
            xy = [(x - 12.0) / 0.5, (y + 7.0) / 1.8]
        elif mode == "flip_x":
            xy = [1280.0 - x, y]
        elif mode == "affine":
            xy = [x + 0.25 * y + 21.0, -0.15 * x + y - 33.0]
        elif mode == "rotated_labels":
            xy = trusted[names[(index + 1) % len(names)]]
        else:
            raise AssertionError(f"unknown mode: {mode}")
        keypoints[name] = {"xy": xy, "confidence": confidence, "heatmap_score": confidence}
    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_no_tap_predictions",
        "clip": f"clip_{mode}",
        "video": "input.mp4",
        "coordinate_space": "source_video_pixels",
        "model_input_size": [160, 90],
        "source_size": [1280, 720],
        "min_confidence": 0.5,
        "frames": [{"frame_index": 0, "keypoints": keypoints}],
        "verified": False,
        "not_cal3_verified": True,
    }


def _by_name(results: list[dict]) -> dict[str, dict]:
    return {str(result["name"]): result for result in results}


def test_labelspace_diagnostic_scores_scale_flip_affine_and_keeps_scope_diagnostic_only() -> None:
    calibration = _calibration_payload()

    scale = analyze_court_keypoint_labelspace(
        prediction_payload=_prediction_payload(calibration, mode="scale"),
        calibration_payload=calibration,
        threshold=0.5,
    )
    scale_results = _by_name(scale["transform_results"])
    assert scale["status"] == "diagnostic_only"
    assert scale["verified"] is False
    assert scale["promote_cal"] is False
    assert scale["cal3_verified"] is False
    assert scale_results["identity"]["residual_px"]["p95"] > 100.0
    assert scale_results["scale_translate_xy"]["residual_px"]["p95"] < 1e-6
    assert scale["best_label_preserving_transform"] == "scale_translate_xy"

    flip = analyze_court_keypoint_labelspace(
        prediction_payload=_prediction_payload(calibration, mode="flip_x"),
        calibration_payload=calibration,
        threshold=0.5,
    )
    flip_results = _by_name(flip["transform_results"])
    assert flip_results["flip_x_scale_translate_xy"]["residual_px"]["p95"] < 1e-6

    affine = analyze_court_keypoint_labelspace(
        prediction_payload=_prediction_payload(calibration, mode="affine"),
        calibration_payload=calibration,
        threshold=0.5,
    )
    affine_results = _by_name(affine["transform_results"])
    assert affine_results["affine"]["residual_px"]["p95"] < 1e-6


def test_labelspace_diagnostic_surfaces_nearest_keypoint_confusion_without_promotion() -> None:
    calibration = _calibration_payload()

    clip = analyze_court_keypoint_labelspace(
        prediction_payload=_prediction_payload(calibration, mode="rotated_labels"),
        calibration_payload=calibration,
        threshold=0.5,
    )

    results = _by_name(clip["transform_results"])
    assert results["identity"]["residual_px"]["p95"] > 100.0
    assert results["nearest_keypoint_permutation"]["residual_px"]["p95"] < 1e-6
    assert clip["nearest_keypoint_same_label_ratio"] < 0.2
    assert clip["likely_failure_mode"] == "label_permutation_or_confusion"
    assert clip["promote_cal"] is False


def test_report_builder_and_cli_write_blocked_artifacts_for_missing_inputs(tmp_path: Path) -> None:
    out = tmp_path / "court_keypoint_labelspace_diagnostic.json"
    markdown = tmp_path / "court_keypoint_labelspace_diagnostic.md"

    report = build_court_keypoint_labelspace_report(
        eval_report_path=tmp_path / "missing_eval_report.json",
        out=out,
        markdown_out=markdown,
        threshold=0.05,
    )

    assert report["status"] == "blocked"
    assert report["verified"] is False
    assert report["promote_cal"] is False
    assert "missing_eval_report" in report["blockers"]
    assert out.is_file()
    assert "missing_eval_report" in markdown.read_text(encoding="utf-8")

    cli_out = tmp_path / "cli" / "diagnostic.json"
    cli_md = tmp_path / "cli" / "diagnostic.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_court_keypoint_labelspace_diagnostic.py",
            "--eval-report",
            str(tmp_path / "still_missing.json"),
            "--out",
            str(cli_out),
            "--markdown-out",
            str(cli_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    cli_payload = json.loads(cli_out.read_text(encoding="utf-8"))
    assert cli_payload["status"] == "blocked"
    assert "missing_eval_report" in cli_payload["blockers"]
    assert "not CAL-3 verified" in cli_md.read_text(encoding="utf-8")
