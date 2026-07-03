from __future__ import annotations

import json
import subprocess
import sys

from threed.racketsport.overlapping_court_calibration import build_lm_homography_reviewed_label_report


def test_lm_homography_report_scores_reviewed_full_labels_without_claiming_verified(tmp_path) -> None:
    report = build_lm_homography_reviewed_label_report(
        eval_root="eval_clips/ball",
        out_path=tmp_path / "overlap_calibration_eval.json",
    )

    assert report["artifact_type"] == "racketsport_overlapping_court_calibration_eval"
    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert report["summary"]["sample_count"] == 5
    assert report["summary"]["full_15pt_clip_count"] == 4
    assert report["summary"]["partial_excluded_count"] == 1
    assert report["summary"]["lm_target_mean_residual_ft"] == 0.2
    assert report["summary"]["lm_optimized_mean_residual_ft_mean"] <= report["summary"]["corner_seed_mean_residual_ft_mean"]
    assert report["summary"]["distorted_camera_rmse_px_mean"] is not None
    assert report["summary"]["distorted_camera_mean_residual_ft_mean"] is not None
    assert report["summary"]["all15_camera_floor_mean_residual_ft_mean"] is not None
    assert report["summary"]["metric_plane_camera_mean_residual_ft_mean"] is not None
    assert (
        report["summary"]["metric_plane_camera_mean_residual_ft_mean"]
        < report["summary"]["distorted_camera_mean_residual_ft_mean"]
    )
    assert report["summary"]["metric_plane_global_trimmed_worst8_mean_residual_ft"] <= 0.2
    assert report["summary"]["metric_plane_global_trimmed_worst8_diagnostic_only"] is True
    assert "point_line_fit_clip_count" in report["summary"]
    assert report["summary"]["point_line_fit_clip_count"] >= 3
    assert report["summary"]["point_line_camera_mean_residual_ft_mean"] is not None
    assert report["summary"]["point_line_weight_sweep_best_mean_residual_ft_mean"] is not None
    assert report["summary"]["point_line_pair_subset_oracle_mean_residual_ft_mean"] is not None
    assert (
        report["summary"]["safe_selected_camera_mean_residual_ft_mean"]
        <= report["summary"]["point_line_pair_subset_oracle_mean_residual_ft_mean"]
    )
    assert (
        report["summary"]["safe_selected_camera_mean_residual_ft_mean"]
        <= report["summary"]["distorted_camera_mean_residual_ft_mean"]
    )
    assert report["summary"]["safe_selected_camera_source_counts"]["metric_plane_camera"] >= 1
    assert len(report["results"]) == 4
    first = report["results"][0]
    assert first["distorted_camera"]["method"] == "joint_focal_pose_radial_lm"
    assert first["distorted_camera"]["optimized_reprojection_rmse_px"] >= 0.0
    assert first["distorted_camera"]["mean_residual_ft"] >= 0.0
    assert first["all15_camera"]["method"] == "joint_focal_pose_radial_lm"
    assert first["all15_camera"]["point_count"] == 15
    assert first["all15_camera"]["floor_mean_residual_ft"] >= 0.0
    assert first["all15_camera"]["net_keypoint_count"] == 3
    assert first["metric_plane_camera"]["method"] == "metric_plane_focal_pose_radial_soft_l1_lm"
    assert first["metric_plane_camera"]["mean_residual_ft"] < first["distorted_camera"]["mean_residual_ft"]
    assert len(first["metric_plane_camera"]["per_keypoint_residual_ft"]) == 12
    assert first["metric_plane_camera"]["trimmed_mean_residual_ft_drop_worst_3"] < first["metric_plane_camera"]["mean_residual_ft"]
    assert "point_line_camera" in first
    assert first["point_line_camera"]["line_candidate_technology_id"] == "opencv_hough_lsd_temporal_persistent"
    assert first["point_line_camera"]["mean_residual_ft"] >= 0.0
    assert len(first["point_line_camera"]["weight_sweep"]) >= 6
    assert first["point_line_camera"]["best_weighted_camera"]["mean_residual_ft"] <= first["point_line_camera"]["mean_residual_ft"]
    assert first["point_line_camera"]["pair_subset_oracle"]["diagnostic_only"] is True
    assert len(first["point_line_camera"]["pair_subset_oracle"]["line_names"]) == 2
    assert first["safe_selected_camera"]["mean_residual_ft"] <= first["distorted_camera"]["mean_residual_ft"]
    assert first["safe_selected_camera"]["selection_mode"] == "reviewed_label_diagnostic_only"


def test_evaluate_overlapping_court_calibration_cli_writes_report(tmp_path) -> None:
    out = tmp_path / "overlap_calibration_eval.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_overlapping_court_calibration.py",
            "--eval-root",
            "eval_clips/ball",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["artifact_type"] == "racketsport_overlapping_court_calibration_eval"
    assert payload["summary"]["full_15pt_clip_count"] == 4
