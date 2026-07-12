from __future__ import annotations

import json
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.court_precision_metrics import (
    SCORER_VERSION,
    calibration_sensitivity_diagnostics,
    image_to_world_scale_jacobian,
    line_evidence_residual,
    net_consistency,
    pbvision_court_comparison,
    score_line_evidence_frame,
    temporal_stability,
)
from threed.racketsport.court_templates import get_court_template


def _calibration(*, x_shift: float = 0.0) -> dict:
    homography = [[50.0, 0.0, 320.0 + x_shift], [0.0, 25.0, 180.0], [0.0, 0.0, 1.0]]
    template = get_court_template("pickleball")
    world = []
    for endpoints in template.line_segments_m.values():
        for point in endpoints:
            if list(point) not in world:
                world.append(list(point))
    world = world[:15]
    image = []
    for x, y, _z in world:
        image.append([50.0 * x + 320.0 + x_shift, 25.0 * y + 180.0])
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "image_size": [640, 360],
        "homography": homography,
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 180.0, "dist": [0.0, 0.0, 0.0, 0.0]},
        "extrinsics": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 10.0]},
        "image_pts": image,
        "world_pts": world,
    }


def _write_court_video(path: Path, calibration: dict, *, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360))
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(_render_court_frame())
    writer.release()


def _render_court_frame(*, distractor_offset_px: int | None = None) -> np.ndarray:
    frame = np.full((360, 640, 3), (70, 120, 55), dtype=np.uint8)
    template = get_court_template("pickleball")
    for line_id, endpoints in template.line_segments_m.items():
        if line_id == "net":
            continue
        points = [
            (int(round(50.0 * x + 320.0)), int(round(25.0 * y + 180.0)))
            for x, y, _z in endpoints
        ]
        cv2.line(frame, points[0], points[1], (255, 255, 255), 4, cv2.LINE_AA)
        if line_id == "near_baseline" and distractor_offset_px is not None:
            shifted = [(x, y + distractor_offset_px) for x, y in points]
            cv2.line(frame, shifted[0], shifted[1], (255, 255, 255), 7, cv2.LINE_AA)
    return frame


def test_m1_scores_existing_white_line_evidence_without_recalibration(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    calibration = _calibration()
    _write_court_video(video, calibration)

    metric = line_evidence_residual(video, calibration, [0, 1, 2], search_window_px=8)

    assert metric["status"] == "present"
    assert metric["residual_px"]["median"] < 0.75
    assert metric["evidence_coverage_fraction"] > 0.5
    assert metric["overflow_count"] == metric["visible_sample_count"] - metric["evidence_sample_count"]
    assert metric["frozen_visible_sample_set"]
    assert len(metric["frozen_visible_sample_set_sha256"]) == 64
    assert len(metric["frozen_evidence_fits_sha256"]) == 64
    assert set(metric["buckets"]) == {"near_far_side", "edge_center", "occlusion"}
    assert metric["coverage_weighted_rollup"]["overflow_count"] == metric["overflow_count"]
    assert [row["frame_index"] for row in metric["per_frame"]] == [0, 1, 2]


def test_m1_exact_h_synthetic_render_is_near_zero_and_visibility_is_frozen() -> None:
    calibration = _calibration()
    frame = _render_court_frame()

    metric = score_line_evidence_frame(frame, calibration, seed_calibration=calibration, frame_index=0, search_window_px=14)

    assert metric["status"] == "present"
    assert metric["residual_px"]["median"] < 0.5
    assert metric["evidence_coverage_fraction"] > 0.95


def test_m1_known_two_pixel_template_warp_scores_two_pixels_on_named_line() -> None:
    seed = _calibration()
    candidate = copy.deepcopy(seed)
    candidate["homography"][1][2] += 2.0
    frame = _render_court_frame()

    exact = score_line_evidence_frame(frame, seed, seed_calibration=seed, frame_index=0, search_window_px=14)
    warped = score_line_evidence_frame(frame, candidate, seed_calibration=seed, frame_index=0, search_window_px=14)
    exact_near = next(row for row in exact["lines"] if row["line_id"] == "near_baseline")
    warped_near = next(row for row in warped["lines"] if row["line_id"] == "near_baseline")

    assert warped["frozen_visible_sample_set_sha256"] == exact["frozen_visible_sample_set_sha256"]
    assert warped_near["residual_px"]["median"] - exact_near["residual_px"]["median"] == pytest.approx(2.0, abs=0.15)


def test_m1_blur_monotonically_reduces_coverage_without_residual_bias() -> None:
    calibration = _calibration()
    frame = _render_court_frame()
    frames = [frame, cv2.GaussianBlur(frame, (23, 23), 0), cv2.GaussianBlur(frame, (29, 29), 0), cv2.GaussianBlur(frame, (33, 33), 0)]
    metrics = [
        score_line_evidence_frame(value, calibration, seed_calibration=calibration, frame_index=0, search_window_px=14)
        for value in frames
    ]
    coverage = [metric["evidence_coverage_fraction"] for metric in metrics]
    medians = [metric["residual_px"]["median"] for metric in metrics]

    assert coverage == sorted(coverage, reverse=True)
    assert coverage[-1] < coverage[0]
    assert max(medians) - min(medians) < 0.15
    assert metrics[-1]["overflow_count"] > metrics[0]["overflow_count"]


def test_m1_dual_parallel_distractor_does_not_capture_center_fit() -> None:
    calibration = _calibration()
    clean = score_line_evidence_frame(_render_court_frame(), calibration, frame_index=0, search_window_px=14)
    distracted = score_line_evidence_frame(
        _render_court_frame(distractor_offset_px=12), calibration, frame_index=0, search_window_px=14
    )
    clean_near = next(row for row in clean["lines"] if row["line_id"] == "near_baseline")
    distracted_near = next(row for row in distracted["lines"] if row["line_id"] == "near_baseline")

    assert distracted_near["residual_px"]["median"] == pytest.approx(clean_near["residual_px"]["median"], abs=0.15)
    assert distracted_near["residual_px"]["median"] < 1.0


def test_m2_is_honestly_absent_for_static_calibration(tmp_path: Path) -> None:
    static_path = tmp_path / "court_calibration.json"
    static_path.write_text("{}", encoding="utf-8")

    metric = temporal_stability({}, camera_motion=None, static_calibration_path=static_path)

    assert metric == {
        "status": "absent",
        "reason": "per_frame_calibration_missing",
        "detail": "only a static court_calibration.json is available; temporal precision is not inferred",
        "static_calibration": str(static_path),
        "per_frame_calibration_count": 0,
        "camera_motion_available": False,
    }


def test_m2_reports_keypoint_delta_when_per_frame_calibrations_exist(tmp_path: Path) -> None:
    metric = temporal_stability(
        {0: _calibration(x_shift=0.0), 1: _calibration(x_shift=2.0)},
        camera_motion=None,
        static_calibration_path=tmp_path / "court_calibration.json",
    )

    assert metric["status"] == "present"
    assert metric["frame_to_frame_keypoint_delta_px"]["median"] == pytest.approx(2.0)
    assert metric["drift_vs_robust_clip_median_px"]["median"] == pytest.approx(1.0)


def test_m5_names_local_scale_honestly_and_turns_one_pixel_into_centimetres() -> None:
    metric = image_to_world_scale_jacobian(_calibration())
    rows = {row["point"]: row for row in metric["table"]}

    assert metric["status"] == "present"
    assert metric["metric_name"] == "image_to_world_scale_jacobian"
    assert metric["calibration_uncertainty"] is False
    assert rows["near_baseline_center"]["plus_x_cm"] == pytest.approx(2.0)
    assert rows["near_baseline_center"]["plus_y_cm"] == pytest.approx(4.0)


def test_m5_bootstraps_seed_observations_through_existing_homography_solver() -> None:
    metric = calibration_sensitivity_diagnostics(_calibration(), bootstrap_draws=200)
    bootstrap = metric["observation_perturbation_bootstrap"]

    assert bootstrap["status"] == "present"
    assert bootstrap["solver"] == "court_calibration.homography_from_planar_points"
    assert bootstrap["draws_requested"] == 200
    assert bootstrap["draws_completed"] == 200
    assert bootstrap["sigma_px"] == 1.0
    assert all(row["boundary_normal_abs_displacement_cm"]["p95"] > 0.0 for row in bootstrap["table"])


def test_m3_rejects_fail_closed_net_anchor_segment_as_semantically_incomparable(tmp_path: Path) -> None:
    proposal = {
        "artifact_type": "racketsport_net_anchor_court_proposals",
        "source": {"image_size": [1280, 720], "frame_role": "player_suppressed_or_single_frame"},
        "solver": {"name": "net_anchor_court", "strategy": "multi_hypothesis_global_fit_v2"},
        "solver_confidence": 0.5,
        "needs_user_confirmation": True,
        "self_verification": {"status": "failed", "promotion_allowed": False, "reasons": ["global_fit_residual_too_high"]},
        "net": {
            "tape_line": [[100.0, 200.0], [500.0, 220.0]],
            "post_tops": [[100.0, 200.0], [500.0, 220.0]],
            "confidence": 0.6,
            "evidence": {"post_count": 1},
        },
    }
    path = tmp_path / "court_proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")

    metric = net_consistency(_calibration(), root=tmp_path, explicit_evidence_path=path)

    assert metric["status"] == "absent"
    assert metric["reason"] == "net_anchor_tape_segment_extents_are_not_verified_semantic_posts"
    assert metric["semantics_audit"]["coordinate_space"] == "source_image_pixels"
    assert metric["semantics_audit"]["segment_identity"].startswith("hough_tape_candidate_segment_extent")
    assert "residual_px" not in metric


def test_m4_reproduces_unique_12_point_wolverine_protocol(tmp_path: Path) -> None:
    calibration = _calibration()
    assert len(calibration["image_pts"]) >= 12
    points = calibration["image_pts"][:12]
    payload = {
        "camera": {
            "fps": 30,
            "cameraSegments": [
                {
                    "s": 0,
                    "e": 299,
                    "fov": 1.2,
                    "position": {"x": 10.0, "y": 50.0, "z": 6.0},
                    "orientation": {"yaw": 0.0, "pitch": -0.2, "roll": 0.0},
                    "court_points": [{"u": x / 640.0, "v": y / 360.0} for x, y in points],
                }
            ],
        }
    }
    export = tmp_path / "cv_export.json"
    export.write_text(json.dumps(payload), encoding="utf-8")

    metric = pbvision_court_comparison(calibration, export)

    assert metric["status"] == "present"
    assert metric["point_count"] == 12
    assert metric["frame_protocol"]["camera_segment_end"] == 299
    assert metric["pbvision_observation_to_our_reviewed_point_px"]["median"] == pytest.approx(0.0)
    assert metric["our_calibration_reprojection_to_our_reviewed_points_px"]["median"] == pytest.approx(0.0)
