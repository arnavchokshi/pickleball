from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.overlapping_court_calibration import (
    build_lm_homography_reviewed_label_report,
    render_metric_plane_outlier_review_packet,
)


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
    assert report["summary"]["metric_plane_line_intersection_override_candidate_count"] >= 1
    assert report["summary"]["metric_plane_line_intersection_override_mean_residual_ft_mean"] is not None
    assert report["summary"]["metric_plane_line_intersection_override_diagnostic_only"] is True
    assert "point_line_fit_clip_count" in report["summary"]
    assert report["summary"]["point_line_fit_clip_count"] >= 3
    assert report["summary"]["point_line_camera_mean_residual_ft_mean"] is not None
    assert report["summary"]["point_line_weight_sweep_best_mean_residual_ft_mean"] is not None
    assert report["summary"]["point_line_pair_subset_oracle_mean_residual_ft_mean"] is not None
    assert report["summary"]["neural_keypoint_checkpoint_candidate_count"] >= 1
    assert report["summary"]["neural_keypoint_real_label_candidate_count"] >= 1
    assert report["summary"]["neural_keypoint_gate_pass_count"] == 0
    assert report["summary"]["neural_keypoint_best_real_median_px"] == 27.538259
    assert report["summary"]["neural_keypoint_diagnostic_only"] is True
    assert (
        report["summary"]["safe_selected_camera_mean_residual_ft_mean"]
        <= report["summary"]["point_line_pair_subset_oracle_mean_residual_ft_mean"]
    )
    assert (
        report["summary"]["safe_selected_camera_mean_residual_ft_mean"]
        <= report["summary"]["distorted_camera_mean_residual_ft_mean"]
    )
    assert report["summary"]["safe_selected_camera_source_counts"]["metric_plane_camera"] >= 1
    neural_evidence = report["neural_keypoint_checkpoint_evidence"]
    assert neural_evidence["diagnostic_only"] is True
    assert neural_evidence["promotes_calibration"] is False
    assert neural_evidence["candidate_count"] >= 1
    assert neural_evidence["real_label_candidate_count"] >= 1
    assert neural_evidence["best_real_label_candidate"]["candidate_metric_name"] == "after.real_keypoint_median_px"
    assert neural_evidence["best_real_label_candidate"]["candidate_metric_value_px"] == 27.538259
    assert neural_evidence["best_real_label_candidate"]["gate_passed"] is False
    assert "gate_failed" in neural_evidence["best_real_label_candidate"]["promotion_blockers"]
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
    assert first["metric_plane_camera"]["outlier_review_candidates"][0]["diagnostic_only"] is True
    assert first["metric_plane_camera"]["outlier_review_candidates"][0]["keypoint"] == "far_right_corner"
    assert first["metric_plane_camera"]["outlier_review_candidates"][0]["residual_ft"] >= 0.45
    assert len(first["metric_plane_camera"]["outlier_review_candidates"][0]["reviewed_image_px"]) == 2
    assert len(first["metric_plane_camera"]["outlier_review_candidates"][0]["model_projected_image_px"]) == 2
    assert first["metric_plane_camera"]["outlier_review_candidates"][0]["line_intersection_available"] is True
    assert len(first["metric_plane_camera"]["outlier_review_candidates"][0]["line_intersection_image_px"]) == 2
    first_outlier_by_name = {
        candidate["keypoint"]: candidate
        for candidate in first["metric_plane_camera"]["outlier_review_candidates"]
    }
    assert first_outlier_by_name["far_nvz_center"]["line_intersection_available"] is True
    assert "centerline_collinear_segment" in first_outlier_by_name["far_nvz_center"]["line_support_modes"]
    assert first_outlier_by_name["near_nvz_center"]["line_intersection_available"] is True
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["diagnostic_only"] is True
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["mutates_reviewed_labels"] is False
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["status"] == "scored"
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["default_strategy"] == "endpoint_intersections_only"
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["override_candidate_count"] >= 1
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["strict_support_required"] is True
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["skipped_non_strict_line_intersection_count"] >= 1
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["skipped_center_keypoint_count"] >= 1
    assert first["metric_plane_camera"]["line_intersection_override_oracle"]["mean_residual_ft"] >= 0.0
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


def test_render_overlapping_court_outlier_review_packet_cli_exposes_direct_help_reference() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_overlapping_court_outlier_review_packet.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--report" in completed.stdout
    assert "--out-dir" in completed.stdout


def test_metric_plane_outlier_review_packet_renders_diagnostic_crops(tmp_path) -> None:
    eval_root = tmp_path / "eval"
    clip_dir = eval_root / "clip_a"
    labels_dir = clip_dir / "labels"
    labels_dir.mkdir(parents=True)
    _write_tiny_video(clip_dir / "source.mp4", width=120, height=90)
    label_path = labels_dir / "court_keypoints.json"
    label_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (labels_dir / "court_calibration_metric15pt.json").write_text(
        json.dumps({"schema_version": 1, "solved_over_frames": [0]}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_overlapping_court_calibration_eval",
                "results": [
                    {
                        "clip": "clip_a",
                        "label_path": str(label_path),
                        "metric_plane_camera": {
                            "outlier_review_candidates": [
                                {
                                    "diagnostic_only": True,
                                    "keypoint": "far_right_corner",
                                    "residual_ft": 0.9,
                                    "reviewed_image_px": [52.0, 46.0],
                                    "model_projected_image_px": [62.0, 47.0],
                                    "model_delta_px": 10.05,
                                    "line_intersection_available": True,
                                    "line_intersection_image_px": [63.0, 47.0],
                                    "line_intersection_delta_px": 11.0,
                                    "model_to_line_intersection_delta_px": 1.0,
                                },
                                {
                                    "diagnostic_only": True,
                                    "keypoint": "near_nvz_center",
                                    "residual_ft": 0.5,
                                    "reviewed_image_px": [35.0, 52.0],
                                    "model_projected_image_px": [40.0, 56.0],
                                    "model_delta_px": 6.4,
                                    "line_intersection_available": False,
                                },
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    packet = render_metric_plane_outlier_review_packet(
        report_path=report_path,
        eval_root=eval_root,
        out_dir=tmp_path / "packet",
        crop_radius_px=24,
    )

    assert packet["artifact_type"] == "racketsport_metric_plane_outlier_review_packet"
    assert packet["status"] == "needs_human_review"
    assert packet["verified"] is False
    assert packet["not_cal3_verified"] is True
    assert packet["diagnostic_only"] is True
    assert packet["item_count"] == 2
    assert packet["line_intersection_item_count"] == 1
    assert packet["line_intersection_support_counts"]["model_projection_closer_to_line"] == 1
    assert packet["line_intersection_support_counts"]["missing_line_intersection"] == 1
    assert Path(packet["contact_sheet"]).is_file()
    assert Path(packet["items"][0]["image"]).is_file()
    assert (tmp_path / "packet" / "metric_plane_outlier_review_packet.json").is_file()


def _write_tiny_video(path: Path, *, width: int, height: int) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    assert writer.isOpened()
    frame = np.full((height, width, 3), 48, dtype=np.uint8)
    cv2.line(frame, (0, height - 20), (width, height - 20), (255, 255, 255), 2)
    writer.write(frame)
    writer.release()
