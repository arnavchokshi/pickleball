from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import threed.racketsport.court_finding_technology_benchmark as court_finding_benchmark
from threed.racketsport.court_finding_technology_benchmark import (
    build_court_finding_technology_report,
    detect_line_candidates_for_technology,
    discover_court_finding_samples,
    score_line_candidates_against_reviewed_keypoints,
)


def test_discovers_four_full_clips_and_img1605_partial() -> None:
    samples = discover_court_finding_samples("eval_clips/ball")

    assert [sample.clip for sample in samples] == [
        "burlington_gold_0300_low_steep_corner",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "owner_IMG_1605_8a193402780b",
        "wolverine_mixed_0200_mid_steep_corner",
    ]
    assert sum(1 for sample in samples if sample.label_kind == "full_15pt") == 4
    img1605 = next(sample for sample in samples if sample.clip == "owner_IMG_1605_8a193402780b")
    assert img1605.label_kind == "partial_visible"
    assert img1605.label_path.name == "court_keypoints_partial.json"
    assert img1605.frame_input.name == "court_keypoint_partial_frames"


def test_opencv_lsd_line_adapter_reports_available_segments() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 70)
    cv2.line(image, (30, 190), (290, 175), (245, 245, 245), 3, cv2.LINE_AA)
    cv2.line(image, (80, 210), (125, 50), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_lsd")

    assert evidence["technology_id"] == "opencv_lsd"
    assert evidence["available"] is True
    assert evidence["candidate_count"] >= 2
    assert evidence["segments"][0]["length_px"] > 20


def test_skimage_probabilistic_hough_adapter_reports_available_segments() -> None:
    pytest.importorskip("skimage.transform")
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 70)
    cv2.line(image, (28, 188), (292, 168), (245, 245, 245), 3, cv2.LINE_AA)
    cv2.line(image, (82, 210), (128, 50), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "skimage_probabilistic_hough")

    assert evidence["technology_id"] == "skimage_probabilistic_hough"
    assert evidence["available"] is True
    assert evidence["candidate_count"] >= 2
    assert evidence["segments"][0]["length_px"] > 20
    assert all(segment["source"] == "skimage_probabilistic_hough" for segment in evidence["segments"])


def test_skimage_probabilistic_hough_adapter_is_deterministic() -> None:
    pytest.importorskip("skimage.transform")
    sample = next(
        sample
        for sample in discover_court_finding_samples("eval_clips/ball")
        if sample.clip == "owner_IMG_1605_8a193402780b"
    )
    from threed.racketsport.net_anchor_court import load_player_suppressed_frame

    image, _meta = load_player_suppressed_frame(sample.frame_input)

    first = detect_line_candidates_for_technology(image, "skimage_probabilistic_hough")
    second = detect_line_candidates_for_technology(image, "skimage_probabilistic_hough")

    assert first["available"] is True
    assert second["available"] is True
    assert first["segments"] == second["segments"]


def test_opencv_fast_line_detector_adapter_reports_available_segments() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 70)
    cv2.line(image, (28, 188), (292, 168), (245, 245, 245), 3, cv2.LINE_AA)
    cv2.line(image, (82, 210), (128, 50), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_fast_line_detector")

    assert evidence["technology_id"] == "opencv_fast_line_detector"
    if evidence["available"]:
        assert evidence["candidate_count"] >= 2
        assert evidence["segments"][0]["length_px"] > 20
        assert all(segment["source"] == "opencv_fast_line_detector" for segment in evidence["segments"])
    else:
        assert evidence["candidate_count"] == 0
        assert evidence["reason"] == "opencv_ximgproc_fast_line_detector_unavailable"


def test_elsed_adapter_fails_closed_when_pyelsed_is_unavailable() -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    image[:] = (40, 95, 55)
    cv2.line(image, (25, 120), (195, 118), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "elsed")

    assert evidence["technology_id"] == "elsed"
    if evidence["available"]:
        assert evidence["candidate_count"] >= 1
        assert all(segment["source"] == "elsed" for segment in evidence["segments"])
    else:
        assert evidence["candidate_count"] == 0
        assert evidence["reason"] == "pyelsed_unavailable"
        assert "git+https://github.com/iago-suarez/ELSED.git" in evidence["install_hint"]


def test_shadow_normalized_hough_adapter_records_preprocess_evidence() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (50, 120, 70)
    image[:, :155] = (20, 55, 35)
    cv2.line(image, (30, 188), (292, 168), (245, 245, 245), 3, cv2.LINE_AA)
    cv2.line(image, (80, 210), (125, 50), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_hough_shadow_normalized")

    assert evidence["technology_id"] == "opencv_hough_shadow_normalized"
    assert evidence["available"] is True
    assert evidence["candidate_count"] >= 2
    assert evidence["shadow_preprocess"]["method"] == "lab_luminance_local_illumination_compensation"
    assert evidence["shadow_preprocess"]["pretrained_model_used"] is False
    assert all(segment["source"] == "opencv_hough_shadow_normalized" for segment in evidence["segments"])


def test_pretrained_shadow_removed_hough_fails_closed_without_model(monkeypatch) -> None:
    monkeypatch.delenv("PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT", raising=False)
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    image[:] = (40, 95, 55)
    cv2.line(image, (25, 120), (195, 118), (245, 245, 245), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_hough_pretrained_shadow_removed")

    assert evidence["technology_id"] == "opencv_hough_pretrained_shadow_removed"
    assert evidence["available"] is False
    assert evidence["candidate_count"] == 0
    assert evidence["shadow_preprocess"]["pretrained_model_used"] is False
    assert evidence["shadow_preprocess"]["reason"] == "missing_pretrained_shadow_removal_model"
    assert "ShadowFormer" in evidence["shadow_preprocess"]["candidate_model_families"]


def test_pretrained_shadow_removed_hough_runs_torchscript_model(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class BrightenShadowModel(torch.nn.Module):
        def forward(self, image):  # type: ignore[no-untyped-def]
            return torch.clamp(image * 2.4, 0.0, 1.0)

    model_path = tmp_path / "shadow_removal.pt"
    example = torch.zeros(1, 3, 32, 32)
    torch.jit.trace(BrightenShadowModel(), example).save(str(model_path))
    monkeypatch.setenv("PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT", str(model_path))

    image = np.zeros((160, 220, 3), dtype=np.uint8)
    image[:] = (42, 95, 55)
    cv2.line(image, (25, 120), (195, 118), (120, 120, 120), 3, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_hough_pretrained_shadow_removed")

    assert evidence["technology_id"] == "opencv_hough_pretrained_shadow_removed"
    assert evidence["available"] is True
    assert evidence["shadow_preprocess"]["pretrained_model_used"] is True
    assert evidence["shadow_preprocess"]["framework"] == "torchscript"
    assert evidence["shadow_preprocess"]["model_path"].endswith("shadow_removal.pt")
    assert evidence["candidate_count"] >= 1
    assert all(segment["source"] == "opencv_hough_pretrained_shadow_removed" for segment in evidence["segments"])


def test_hsv_paint_hough_line_adapter_ignores_white_tennis_lines() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 70)
    cv2.line(image, (20, 45), (300, 45), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (20, 205), (300, 205), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (70, 90), (250, 90), (40, 230, 230), 4, cv2.LINE_AA)
    cv2.line(image, (70, 190), (250, 190), (40, 230, 230), 4, cv2.LINE_AA)

    evidence = detect_line_candidates_for_technology(image, "opencv_hsv_paint_hough")

    assert evidence["technology_id"] == "opencv_hsv_paint_hough"
    assert evidence["available"] is True
    assert evidence["candidate_count"] >= 2
    assert evidence["paint_mask"]["mode"] == "strict_hsv_ranges"
    assert evidence["paint_mask"]["support_ratio"] > 0.001
    assert all(segment["source"] == "opencv_hsv_paint_hough" for segment in evidence["segments"])


def test_line_support_scoring_uses_reviewed_floor_lines() -> None:
    segments = [
        {
            "p1": [553.0, 520.0],
            "p2": [1361.0, 518.0],
            "length_px": 808.0,
            "angle_deg": 0.0,
            "source": "synthetic_true_near_nvz",
        },
        {
            "p1": [100.0, 100.0],
            "p2": [140.0, 130.0],
            "length_px": 50.0,
            "angle_deg": 36.0,
            "source": "synthetic_noise",
        },
    ]

    support = score_line_candidates_against_reviewed_keypoints(
        reviewed_keypoints_path=(
            "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_keypoints.json"
        ),
        line_candidates=segments,
    )

    assert support["evaluated_line_count"] == 8
    assert support["supported_line_count"] >= 1
    assert support["per_line"]["near_nvz"]["status"] == "supported"


def test_line_support_scoring_scales_candidate_frame_to_native_space() -> None:
    segments = [
        {
            "p1": [368.7, 346.7],
            "p2": [907.4, 345.3],
            "length_px": 538.7,
            "angle_deg": 0.0,
            "source": "synthetic_downscaled_near_nvz",
        },
    ]

    support = score_line_candidates_against_reviewed_keypoints(
        reviewed_keypoints_path=(
            "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_keypoints.json"
        ),
        line_candidates=segments,
        candidate_image_size=(1280, 720),
    )

    assert support["per_line"]["near_nvz"]["status"] == "supported"


def test_opencv_fast_line_detector_reports_real_sample_line_support(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_fast_line_detector"],
        out_dir=tmp_path,
    )

    fast_line = report["summary"]["by_technology"]["opencv_fast_line_detector"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert fast_line["scored_clip_count"] == 0
    assert fast_line["line_support_ratio_mean"] is not None


def test_template_competition_penalizes_tennis_service_spacing() -> None:
    pickleball_assignment = {
        "far_baseline": {"p1": [0.0, 100.0], "p2": [1000.0, 100.0]},
        "far_nvz": {"p1": [0.0, 250.0], "p2": [1000.0, 250.0]},
        "net": {"p1": [0.0, 320.0], "p2": [1000.0, 320.0]},
        "near_nvz": {"p1": [0.0, 390.0], "p2": [1000.0, 390.0]},
        "near_baseline": {"p1": [0.0, 540.0], "p2": [1000.0, 540.0]},
    }
    tennis_assignment = {
        "far_baseline": {"p1": [0.0, 100.0], "p2": [1000.0, 100.0]},
        "far_nvz": {"p1": [0.0, 280.0], "p2": [1000.0, 280.0]},
        "net": {"p1": [0.0, 490.0], "p2": [1000.0, 490.0]},
        "near_nvz": {"p1": [0.0, 700.0], "p2": [1000.0, 700.0]},
        "near_baseline": {"p1": [0.0, 880.0], "p2": [1000.0, 880.0]},
    }

    assert hasattr(court_finding_benchmark, "score_template_competition_for_line_assignment")
    pickleball_score = court_finding_benchmark.score_template_competition_for_line_assignment(
        pickleball_assignment,
        image_size=(1000, 1000),
    )
    tennis_score = court_finding_benchmark.score_template_competition_for_line_assignment(
        tennis_assignment,
        image_size=(1000, 1000),
    )

    assert pickleball_score["cross_template"]["pickleball_spacing_error"] < 0.001
    assert pickleball_score["cross_template"]["tennis_service_spacing_error"] > 0.30
    assert pickleball_score["cross_template"]["tennis_better_than_pickleball"] is False
    assert pickleball_score["tennis_template_penalty"] == 0.0
    assert tennis_score["cross_template"]["tennis_service_spacing_error"] < 0.001
    assert tennis_score["cross_template"]["pickleball_spacing_error"] > 0.30
    assert tennis_score["cross_template"]["tennis_better_than_pickleball"] is True
    assert tennis_score["tennis_template_penalty"] > 25.0


def test_projected_line_pixel_support_prefers_lines_on_image_mask() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 95, 55)
    keypoints = {
        "near_left_corner": (60.0, 190.0),
        "near_right_corner": (260.0, 190.0),
        "far_left_corner": (90.0, 60.0),
        "far_right_corner": (230.0, 60.0),
        "near_nvz_left": (70.0, 150.0),
        "near_nvz_right": (250.0, 150.0),
        "far_nvz_left": (85.0, 100.0),
        "far_nvz_right": (235.0, 100.0),
        "near_baseline_center": (160.0, 190.0),
        "near_nvz_center": (160.0, 150.0),
        "far_baseline_center": (160.0, 60.0),
        "far_nvz_center": (160.0, 100.0),
    }
    for p1_name, p2_name in [
        ("near_left_corner", "near_right_corner"),
        ("far_left_corner", "far_right_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
        ("near_left_corner", "far_left_corner"),
        ("near_right_corner", "far_right_corner"),
        ("near_baseline_center", "near_nvz_center"),
        ("far_baseline_center", "far_nvz_center"),
    ]:
        p1 = keypoints[p1_name]
        p2 = keypoints[p2_name]
        cv2.line(image, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), (40, 235, 235), 4)

    shifted_keypoints = {name: (xy[0], xy[1] - 23.0) for name, xy in keypoints.items()}

    aligned = court_finding_benchmark.score_projected_line_pixels_against_image(image, keypoints)
    shifted = court_finding_benchmark.score_projected_line_pixels_against_image(image, shifted_keypoints)

    assert aligned["available"] is True
    assert aligned["supported_line_pixel_count"] >= 7
    assert aligned["mean_line_pixel_support_ratio"] > 0.60
    assert aligned["mean_line_pixel_support_ratio"] > shifted["mean_line_pixel_support_ratio"] + 0.30


def test_projected_line_distance_transform_prefers_nearby_line_pixels() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 95, 55)
    keypoints = {
        "near_left_corner": (60.0, 190.0),
        "near_right_corner": (260.0, 190.0),
        "far_left_corner": (90.0, 60.0),
        "far_right_corner": (230.0, 60.0),
        "near_nvz_left": (70.0, 150.0),
        "near_nvz_right": (250.0, 150.0),
        "far_nvz_left": (85.0, 100.0),
        "far_nvz_right": (235.0, 100.0),
        "near_baseline_center": (160.0, 190.0),
        "near_nvz_center": (160.0, 150.0),
        "far_baseline_center": (160.0, 60.0),
        "far_nvz_center": (160.0, 100.0),
    }
    for p1_name, p2_name in [
        ("near_left_corner", "near_right_corner"),
        ("far_left_corner", "far_right_corner"),
        ("near_nvz_left", "near_nvz_right"),
        ("far_nvz_left", "far_nvz_right"),
        ("near_left_corner", "far_left_corner"),
        ("near_right_corner", "far_right_corner"),
        ("near_baseline_center", "near_nvz_center"),
        ("far_baseline_center", "far_nvz_center"),
    ]:
        p1 = keypoints[p1_name]
        p2 = keypoints[p2_name]
        cv2.line(image, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), (40, 235, 235), 4)

    shifted_keypoints = {name: (xy[0], xy[1] - 18.0) for name, xy in keypoints.items()}

    aligned = court_finding_benchmark.score_projected_line_distance_transform_against_image(image, keypoints)
    shifted = court_finding_benchmark.score_projected_line_distance_transform_against_image(image, shifted_keypoints)

    assert aligned["available"] is True
    assert aligned["mean_projected_line_distance_px"] < 1.5
    assert shifted["mean_projected_line_distance_px"] > aligned["mean_projected_line_distance_px"] + 6.0
    assert aligned["distance_supported_line_count"] >= 7


def test_line_color_consistency_detects_mixed_overlay_line_layers() -> None:
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[:] = (35, 100, 55)
    assignment = {
        "far_baseline": {"p1": [55.0, 50.0], "p2": [265.0, 50.0]},
        "far_nvz": {"p1": [70.0, 90.0], "p2": [250.0, 90.0]},
        "near_nvz": {"p1": [70.0, 145.0], "p2": [250.0, 145.0]},
        "near_baseline": {"p1": [55.0, 185.0], "p2": [265.0, 185.0]},
        "left_sideline": {"p1": [55.0, 185.0], "p2": [55.0, 50.0]},
        "right_sideline": {"p1": [265.0, 185.0], "p2": [265.0, 50.0]},
    }
    for name, item in assignment.items():
        color = (245, 245, 245) if name in {"far_baseline", "near_baseline"} else (35, 235, 235)
        cv2.line(
            image,
            (round(item["p1"][0]), round(item["p1"][1])),
            (round(item["p2"][0]), round(item["p2"][1])),
            color,
            5,
        )

    score = court_finding_benchmark.score_line_color_consistency_for_assignment(image, assignment)

    assert score["available"] is True
    assert score["sampled_line_count"] == 6
    assert score["distinct_color_cluster_count"] >= 2
    assert score["mixed_layer_penalty"] > 0.0
    assert {"far_baseline", "near_baseline"}.issubset(score["per_line"])


def test_regulation_proposal_records_projected_pixel_and_color_evidence(tmp_path) -> None:
    build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_regulation"],
        out_dir=tmp_path,
    )

    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_regulation"
            / "court_proposal.json"
        ).read_text()
    )
    components = proposal["score_components"]

    assert components["projected_pixel_support"]["available"] is True
    assert "mean_line_pixel_support_ratio" in components["projected_pixel_support"]
    assert "supported_line_pixel_count" in components["projected_pixel_support"]
    assert components["line_color_consistency"]["available"] is True
    assert "distinct_color_cluster_count" in components["line_color_consistency"]
    assert proposal["hypotheses"][0]["score_components"]["projected_pixel_support"] == components["projected_pixel_support"]
    assert proposal["hypotheses"][0]["score_components"]["line_color_consistency"] == components["line_color_consistency"]


def test_distance_mask_regulation_records_distance_transform_evidence(tmp_path) -> None:
    build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_regulation_distance_mask"],
        out_dir=tmp_path,
    )

    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_regulation_distance_mask"
            / "court_proposal.json"
        ).read_text()
    )
    components = proposal["score_components"]

    assert proposal["solver"]["name"] == "opencv_hough_lsd_regulation_distance_mask"
    assert proposal["solver"]["writes_court_calibration"] is False
    assert proposal["needs_user_confirmation"] is True
    assert components["image_evidence_mode"] == "distance_mask"
    assert components["projected_distance_support"]["available"] is True
    assert "mean_projected_line_distance_px" in components["projected_distance_support"]
    assert "distance_supported_line_count" in components["projected_distance_support"]


def test_line_refined_regulation_reduces_synthetic_assignment_residual() -> None:
    from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points

    true_h = homography_from_planar_points(
        [[-10.0, -22.0, 0.0], [10.0, -22.0, 0.0], [10.0, 22.0, 0.0], [-10.0, 22.0, 0.0]],
        [[180.0, 650.0], [1110.0, 640.0], [860.0, 120.0], [405.0, 130.0]],
    )
    seed_h = homography_from_planar_points(
        [[-10.0, -22.0, 0.0], [10.0, -22.0, 0.0], [10.0, 22.0, 0.0], [-10.0, 22.0, 0.0]],
        [[205.0, 636.0], [1080.0, 656.0], [884.0, 142.0], [382.0, 111.0]],
    )
    line_world_endpoints = {
        "near_baseline": ((-10.0, -22.0, 0.0), (10.0, -22.0, 0.0)),
        "far_baseline": ((-10.0, 22.0, 0.0), (10.0, 22.0, 0.0)),
        "near_nvz": ((-10.0, -7.0, 0.0), (10.0, -7.0, 0.0)),
        "far_nvz": ((-10.0, 7.0, 0.0), (10.0, 7.0, 0.0)),
        "left_sideline": ((-10.0, -22.0, 0.0), (-10.0, 22.0, 0.0)),
        "right_sideline": ((10.0, -22.0, 0.0), (10.0, 22.0, 0.0)),
    }
    line_assignment = {}
    for name, endpoints in line_world_endpoints.items():
        p1, p2 = project_planar_points(true_h, endpoints)
        line_assignment[name] = {
            "p1": [p1[0], p1[1]],
            "p2": [p2[0], p2[1]],
            "angle_deg": 0.0,
            "support_length_px": 500.0,
            "source_segment_count": 1,
        }
    keypoints = {
        name: tuple(project_planar_points(seed_h, [[xy[0], xy[1], 0.0]])[0])
        for name, xy in court_finding_benchmark._FLOOR_WORLD_XY.items()
    }
    hypothesis = {
        "score": 0.0,
        "keypoints": keypoints,
        "supported_line_count": 6,
        "line_assignment": line_assignment,
        "score_components": {},
    }

    refined = court_finding_benchmark.refine_regulation_homography_for_line_assignment(
        hypothesis,
        image_size=(1280, 720),
    )

    refinement = refined["score_components"]["line_refinement"]
    assert refinement["method"] == "scipy_least_squares_point_line_homography"
    assert refinement["accepted"] is True
    assert refinement["optimized_line_rmse_px"] < refinement["initial_line_rmse_px"] * 0.75
    assert refined["keypoints"]["near_left_corner"] != hypothesis["keypoints"]["near_left_corner"]


def test_line_refined_regulation_scores_real_samples_without_claiming_verified(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_regulation", "opencv_hough_lsd_regulation_line_refined"],
        out_dir=tmp_path,
    )

    refined = report["summary"]["by_technology"]["opencv_hough_lsd_regulation_line_refined"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert refined["scored_clip_count"] == 5
    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_regulation_line_refined"
            / "court_proposal.json"
        ).read_text()
    )
    assert proposal["solver"]["writes_court_calibration"] is False
    assert proposal["needs_user_confirmation"] is True
    assert proposal["score_components"]["line_refinement"]["available"] is True
    assert proposal["hypotheses"][0]["score_components"]["line_refinement"]["available"] is True


def test_hough_or_distance_mask_selector_scores_candidate_combination(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_keypoints",
            "opencv_hough_lsd_regulation",
            "opencv_hough_lsd_regulation_distance_mask",
            "hough_or_regulation_distance_mask_selector",
        ],
        out_dir=tmp_path,
    )

    selector = report["summary"]["by_technology"]["hough_or_regulation_distance_mask_selector"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert selector["scored_clip_count"] == 5
    assert selector["floor_visible_median_px_mean"] < report["summary"]["by_technology"]["hough_keypoints"]["floor_visible_median_px_mean"]
    assert selector["floor_visible_median_px_mean"] < 305.0
    assert selector["floor_visible_p95_px_mean"] < 540.0

    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "hough_or_regulation_distance_mask_selector"
            / "court_proposal.json"
        ).read_text()
    )
    assert set(proposal["selector"]["candidate_scores"]) == {
        "hough_keypoints",
        "opencv_hough_lsd_regulation",
        "opencv_hough_lsd_regulation_distance_mask",
    }
    assert proposal["needs_user_confirmation"] is True

    wolverine = json.loads(
        (
            tmp_path
            / "wolverine_mixed_0200_mid_steep_corner"
            / "hough_or_regulation_distance_mask_selector"
            / "court_proposal.json"
        ).read_text()
    )
    assert wolverine["selector"]["selected_technology_id"] == "opencv_hough_lsd_regulation"

    indoor = json.loads(
        (
            tmp_path
            / "indoor_doubles_fwuks_0500_long_mid_baseline"
            / "hough_or_regulation_distance_mask_selector"
            / "court_proposal.json"
        ).read_text()
    )
    assert indoor["selector"]["selected_technology_id"] == "opencv_hough_lsd_regulation_distance_mask"


def test_benchmark_report_scores_existing_adapters_without_claiming_verified(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["net_anchor", "hough_keypoints", "opencv_lsd"],
        out_dir=tmp_path,
    )

    assert report["artifact_type"] == "racketsport_court_finding_technology_benchmark"
    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert report["summary"]["sample_count"] == 5
    assert set(report["summary"]["technology_ids"]) == {"net_anchor", "hough_keypoints", "opencv_lsd"}
    assert report["summary"]["scored_result_count"] >= 1
    assert any(result["label_kind"] == "partial_visible" for result in report["results"])
    lsd_results = [result for result in report["results"] if result["technology_id"] == "opencv_lsd"]
    assert lsd_results
    assert all(result["status"] in {"line_candidates_only", "no_line_candidates"} for result in lsd_results)
    assert all("line_support_ratio" in result for result in lsd_results)


def test_benchmark_report_runs_hsv_paint_hough_as_line_candidate_only(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hsv_paint_hough"],
        out_dir=tmp_path,
    )

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert report["summary"]["sample_count"] == 5
    hsv_results = [result for result in report["results"] if result["technology_id"] == "opencv_hsv_paint_hough"]
    assert len(hsv_results) == 5
    assert all(result["status"] in {"line_candidates_only", "no_line_candidates"} for result in hsv_results)
    assert all("line_support_ratio" in result for result in hsv_results)


def test_benchmark_report_records_pretrained_shadow_removal_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT", raising=False)

    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_pretrained_shadow_removed"],
        out_dir=tmp_path,
    )

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    results = [
        result
        for result in report["results"]
        if result["technology_id"] == "opencv_hough_pretrained_shadow_removed"
    ]
    assert len(results) == 5
    assert all(result["status"] == "no_line_candidates" for result in results)
    evidence = json.loads((tmp_path / results[0]["line_candidate_path"]).read_text())
    assert evidence["available"] is False
    assert evidence["shadow_preprocess"]["reason"] == "missing_pretrained_shadow_removal_model"
    assert evidence["shadow_preprocess"]["pretrained_model_used"] is False


def test_hough_lsd_regulation_selector_scores_and_improves_mean_floor_error(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["hough_keypoints", "opencv_hough_lsd_regulation", "hough_or_regulation_line_selector"],
        out_dir=tmp_path,
    )

    baseline = report["summary"]["by_technology"]["hough_keypoints"]
    regulation = report["summary"]["by_technology"]["opencv_hough_lsd_regulation"]
    selector = report["summary"]["by_technology"]["hough_or_regulation_line_selector"]

    assert regulation["scored_clip_count"] == 5
    assert selector["scored_clip_count"] == 5
    assert selector["floor_visible_median_px_mean"] < 350.0
    assert selector["floor_visible_median_px_mean"] < baseline["floor_visible_median_px_mean"]
    selector_results = {
        result["clip"]: result
        for result in report["results"]
        if result["technology_id"] == "hough_or_regulation_line_selector"
    }
    burlington_selector = json.loads(
        (tmp_path / "burlington_gold_0300_low_steep_corner" / "hough_or_regulation_line_selector" / "court_proposal.json").read_text()
    )
    assert burlington_selector["selector"]["selected_technology_id"] == "hough_keypoints"
    assert selector_results["burlington_gold_0300_low_steep_corner"]["floor_visible_median_px"] < 500.0
    assert all(
        result["status"] == "scored"
        for result in report["results"]
        if result["technology_id"] == "hough_or_regulation_line_selector"
    )


def test_temporal_hough_lsd_line_adapter_uses_available_frame_sequence(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_temporal"],
        out_dir=tmp_path,
    )

    assert report["summary"]["sample_count"] == 5
    temporal_results = [result for result in report["results"] if result["technology_id"] == "opencv_hough_lsd_temporal"]
    assert len(temporal_results) == 5
    assert all(result["status"] == "line_candidates_only" for result in temporal_results)
    assert all(result["line_candidate_count"] > 0 for result in temporal_results)

    burlington_evidence = json.loads(
        (
            tmp_path
            / "burlington_gold_0300_low_steep_corner"
            / "opencv_hough_lsd_temporal"
            / "line_candidates.json"
        ).read_text()
    )
    img1605_evidence = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_temporal"
            / "line_candidates.json"
        ).read_text()
    )

    assert burlington_evidence["temporal_frame_count"] >= 2
    assert img1605_evidence["temporal_frame_count"] == 1
    assert burlington_evidence["candidate_count"] >= 1
    assert all("temporal_frame_index" in segment for segment in burlington_evidence["segments"][:5])


def test_temporal_persistent_line_adapter_reports_frame_support(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_temporal_persistent"],
        out_dir=tmp_path,
    )

    assert report["summary"]["sample_count"] == 5
    persistent_results = [
        result for result in report["results"] if result["technology_id"] == "opencv_hough_lsd_temporal_persistent"
    ]
    assert len(persistent_results) == 5
    assert all(result["status"] == "line_candidates_only" for result in persistent_results)

    burlington_evidence = json.loads(
        (
            tmp_path
            / "burlington_gold_0300_low_steep_corner"
            / "opencv_hough_lsd_temporal_persistent"
            / "line_candidates.json"
        ).read_text()
    )
    img1605_evidence = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_temporal_persistent"
            / "line_candidates.json"
        ).read_text()
    )

    assert burlington_evidence["temporal_frame_count"] >= 2
    assert burlington_evidence["persistence_min_frame_count"] >= 2
    assert burlington_evidence["candidate_count"] > 0
    assert all(
        segment["temporal_support_frame_count"] >= 2
        for segment in burlington_evidence["segments"][: min(10, burlington_evidence["candidate_count"])]
    )
    assert all(
        0.0 < segment["temporal_persistence_ratio"] <= 1.0
        for segment in burlington_evidence["segments"][: min(10, burlington_evidence["candidate_count"])]
    )
    assert img1605_evidence["temporal_frame_count"] == 1
    assert img1605_evidence["persistence_min_frame_count"] == 1


def test_temporal_hough_lsd_regulation_scores_without_claiming_verified(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_temporal_regulation"],
        out_dir=tmp_path,
    )

    temporal = report["summary"]["by_technology"]["opencv_hough_lsd_temporal_regulation"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert temporal["scored_clip_count"] == 5
    assert all(
        result["status"] == "scored"
        for result in report["results"]
        if result["technology_id"] == "opencv_hough_lsd_temporal_regulation"
    )
    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_temporal_regulation"
            / "court_proposal.json"
        ).read_text()
    )
    assert proposal["needs_user_confirmation"] is True
    assert proposal["solver"]["writes_court_calibration"] is False
    assert proposal["line_evidence"]["temporal_frame_count"] == 1


def test_regulation_proposal_records_tennis_template_competition(tmp_path) -> None:
    build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_regulation"],
        out_dir=tmp_path,
    )

    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_regulation"
            / "court_proposal.json"
        ).read_text()
    )
    template = proposal["score_components"]["template_competition"]

    assert template["available"] is True
    assert "cross_template" in template
    assert "pickleball_spacing_error" in template["cross_template"]
    assert "tennis_service_spacing_error" in template["cross_template"]
    assert "tennis_template_penalty" in template
    assert proposal["hypotheses"][0]["score_components"]["template_competition"] == template


def test_temporal_persistent_regulation_scores_without_claiming_verified(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["opencv_hough_lsd_temporal_persistent_regulation"],
        out_dir=tmp_path,
    )

    persistent = report["summary"]["by_technology"]["opencv_hough_lsd_temporal_persistent_regulation"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert persistent["scored_clip_count"] == 5
    assert all(
        result["status"] == "scored"
        for result in report["results"]
        if result["technology_id"] == "opencv_hough_lsd_temporal_persistent_regulation"
    )
    proposal = json.loads(
        (
            tmp_path
            / "burlington_gold_0300_low_steep_corner"
            / "opencv_hough_lsd_temporal_persistent_regulation"
            / "court_proposal.json"
        ).read_text()
    )
    assert proposal["needs_user_confirmation"] is True
    assert proposal["solver"]["writes_court_calibration"] is False
    assert proposal["line_evidence"]["technology_id"] == "opencv_hough_lsd_temporal_persistent"
    assert proposal["line_evidence"]["persistence_min_frame_count"] >= 2


def test_skimage_merged_regulation_scores_real_samples_without_claiming_verified(tmp_path) -> None:
    pytest.importorskip("skimage.transform")
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "skimage_probabilistic_hough",
            "opencv_hough_lsd_skimage",
            "opencv_hough_lsd_skimage_regulation",
        ],
        out_dir=tmp_path,
    )

    skimage_lines = report["summary"]["by_technology"]["skimage_probabilistic_hough"]
    merged_lines = report["summary"]["by_technology"]["opencv_hough_lsd_skimage"]
    merged_regulation = report["summary"]["by_technology"]["opencv_hough_lsd_skimage_regulation"]

    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert skimage_lines["line_support_ratio_mean"] is not None
    assert merged_lines["line_support_ratio_mean"] is not None
    assert merged_regulation["scored_clip_count"] == 5
    proposal = json.loads(
        (
            tmp_path
            / "owner_IMG_1605_8a193402780b"
            / "opencv_hough_lsd_skimage_regulation"
            / "court_proposal.json"
        ).read_text()
    )
    assert proposal["needs_user_confirmation"] is True
    assert proposal["solver"]["writes_court_calibration"] is False
    assert proposal["line_evidence"]["technology_id"] == "opencv_hough_lsd_skimage"


def test_hough_regulation_temporal_selector_improves_deployable_mean(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_keypoints",
            "opencv_hough_lsd_regulation",
            "opencv_hough_lsd_temporal_regulation",
            "hough_regulation_temporal_line_selector",
        ],
        out_dir=tmp_path,
    )

    hough = report["summary"]["by_technology"]["hough_keypoints"]
    old_regulation = report["summary"]["by_technology"]["opencv_hough_lsd_regulation"]
    temporal_regulation = report["summary"]["by_technology"]["opencv_hough_lsd_temporal_regulation"]
    selector = report["summary"]["by_technology"]["hough_regulation_temporal_line_selector"]

    assert selector["scored_clip_count"] == 5
    assert selector["floor_visible_median_px_mean"] < hough["floor_visible_median_px_mean"]
    assert selector["floor_visible_median_px_mean"] < old_regulation["floor_visible_median_px_mean"]
    assert selector["floor_visible_median_px_mean"] < temporal_regulation["floor_visible_median_px_mean"]
    assert selector["floor_visible_median_px_mean"] < 330.0

    burlington_selector = json.loads(
        (
            tmp_path
            / "burlington_gold_0300_low_steep_corner"
            / "hough_regulation_temporal_line_selector"
            / "court_proposal.json"
        ).read_text()
    )
    assert burlington_selector["selector"]["selected_technology_id"] == "hough_keypoints"


def test_summary_reports_floor_visible_p95_tail_metrics_after_geometry_guard(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=["hough_regulation_temporal_line_selector"],
        out_dir=tmp_path,
    )

    selector = report["summary"]["by_technology"]["hough_regulation_temporal_line_selector"]

    assert selector["scored_clip_count"] == 5
    assert selector["floor_visible_p95_px_mean"] < 620.0
    assert selector["floor_visible_p95_px_max"] < 900.0


def test_balanced_temporal_selector_reduces_tail_risk_without_losing_median_gain(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_or_regulation_line_selector",
            "hough_regulation_temporal_line_selector",
            "hough_regulation_temporal_balanced_selector",
        ],
        out_dir=tmp_path,
    )

    old_selector = report["summary"]["by_technology"]["hough_or_regulation_line_selector"]
    median_selector = report["summary"]["by_technology"]["hough_regulation_temporal_line_selector"]
    balanced = report["summary"]["by_technology"]["hough_regulation_temporal_balanced_selector"]

    assert balanced["scored_clip_count"] == 5
    assert balanced["floor_visible_median_px_mean"] <= old_selector["floor_visible_median_px_mean"]
    assert balanced["floor_visible_p95_px_mean"] < median_selector["floor_visible_p95_px_mean"]
    assert balanced["floor_visible_median_px_mean"] < 300.0
    assert balanced["floor_visible_p95_px_mean"] < 560.0

    choices = {}
    for clip in [
        "burlington_gold_0300_low_steep_corner",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "owner_IMG_1605_8a193402780b",
        "wolverine_mixed_0200_mid_steep_corner",
    ]:
        proposal = json.loads((tmp_path / clip / "hough_regulation_temporal_balanced_selector" / "court_proposal.json").read_text())
        choices[clip] = proposal["selector"]["selected_technology_id"]

    assert choices == {
        "burlington_gold_0300_low_steep_corner": "hough_keypoints",
        "indoor_doubles_fwuks_0500_long_mid_baseline": "opencv_hough_lsd_regulation",
        "outdoor_webcam_iynbd_1500_long_high_baseline": "opencv_hough_lsd_regulation",
        "owner_IMG_1605_8a193402780b": "opencv_hough_lsd_regulation",
        "wolverine_mixed_0200_mid_steep_corner": "opencv_hough_lsd_regulation",
    }


def test_persistent_tail_selector_uses_temporal_persistence_when_it_reduces_risk(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_or_regulation_line_selector",
            "hough_regulation_temporal_balanced_selector",
            "hough_regulation_temporal_persistent_tail_selector",
        ],
        out_dir=tmp_path,
    )

    old_selector = report["summary"]["by_technology"]["hough_or_regulation_line_selector"]
    balanced = report["summary"]["by_technology"]["hough_regulation_temporal_balanced_selector"]
    persistent_tail = report["summary"]["by_technology"]["hough_regulation_temporal_persistent_tail_selector"]

    assert persistent_tail["scored_clip_count"] == 5
    assert persistent_tail["floor_visible_median_px_mean"] < 340.0
    assert persistent_tail["floor_visible_p95_px_mean"] < old_selector["floor_visible_p95_px_mean"]
    assert persistent_tail["floor_visible_p95_px_mean"] < balanced["floor_visible_p95_px_mean"]
    assert persistent_tail["floor_visible_median_px_mean"] < 340.0
    assert persistent_tail["floor_visible_p95_px_mean"] < 525.0

    choices = {}
    for clip in [
        "burlington_gold_0300_low_steep_corner",
        "indoor_doubles_fwuks_0500_long_mid_baseline",
        "outdoor_webcam_iynbd_1500_long_high_baseline",
        "owner_IMG_1605_8a193402780b",
        "wolverine_mixed_0200_mid_steep_corner",
    ]:
        proposal = json.loads(
            (tmp_path / clip / "hough_regulation_temporal_persistent_tail_selector" / "court_proposal.json").read_text()
        )
        choices[clip] = proposal["selector"]["selected_technology_id"]

    assert choices == {
        "burlington_gold_0300_low_steep_corner": "hough_keypoints",
        "indoor_doubles_fwuks_0500_long_mid_baseline": "opencv_hough_lsd_temporal_persistent_regulation",
        "outdoor_webcam_iynbd_1500_long_high_baseline": "opencv_hough_lsd_regulation",
        "owner_IMG_1605_8a193402780b": "opencv_hough_lsd_regulation",
        "wolverine_mixed_0200_mid_steep_corner": "opencv_hough_lsd_temporal_persistent_regulation",
    }


def test_reviewed_oracle_selector_reports_candidate_pool_upper_bound(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_keypoints",
            "opencv_hough_lsd_regulation",
            "reviewed_oracle_hough_or_regulation",
        ],
        out_dir=tmp_path,
    )

    hough = report["summary"]["by_technology"]["hough_keypoints"]
    regulation = report["summary"]["by_technology"]["opencv_hough_lsd_regulation"]
    oracle = report["summary"]["by_technology"]["reviewed_oracle_hough_or_regulation"]

    assert oracle["scored_clip_count"] == 5
    assert oracle["floor_visible_median_px_mean"] <= hough["floor_visible_median_px_mean"]
    assert oracle["floor_visible_median_px_mean"] <= regulation["floor_visible_median_px_mean"]
    assert all(
        "reviewed_label_oracle_not_deployable" in result.get("rejection_reasons", [])
        for result in report["results"]
        if result["technology_id"] == "reviewed_oracle_hough_or_regulation"
    )


def test_temporal_reviewed_oracle_reports_candidate_pool_upper_bound(tmp_path) -> None:
    report = build_court_finding_technology_report(
        eval_root="eval_clips/ball",
        technologies=[
            "hough_keypoints",
            "opencv_hough_lsd_regulation",
            "opencv_hough_lsd_temporal_regulation",
            "hough_regulation_temporal_line_selector",
            "reviewed_oracle_hough_regulation_temporal",
        ],
        out_dir=tmp_path,
    )

    selector = report["summary"]["by_technology"]["hough_regulation_temporal_line_selector"]
    temporal_oracle = report["summary"]["by_technology"]["reviewed_oracle_hough_regulation_temporal"]

    assert temporal_oracle["scored_clip_count"] == 5
    assert temporal_oracle["floor_visible_median_px_mean"] <= selector["floor_visible_median_px_mean"]
    assert all(
        "reviewed_label_oracle_not_deployable" in result.get("rejection_reasons", [])
        for result in report["results"]
        if result["technology_id"] == "reviewed_oracle_hough_regulation_temporal"
    )


def test_evaluate_court_finding_technologies_cli_writes_report(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_finding_technologies.py",
            "--eval-root",
            "eval_clips/ball",
            "--out-dir",
            str(tmp_path),
            "--technology",
            "opencv_lsd",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "court_finding_technology_benchmark.json").read_text())
    assert payload["artifact_type"] == "racketsport_court_finding_technology_benchmark"
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["summary"]["sample_count"] == 5
    assert payload["summary"]["technology_ids"] == ["opencv_lsd"]
