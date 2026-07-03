from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from threed.racketsport.net_anchor_court import (
    DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
    LineEvidence,
    NetDetection,
    ProposalPoint,
    _apply_hypothesis_confidence_cap,
    _build_global_fit_hypotheses,
    _confidence_cap_for_hypothesis,
    _homography_from_named_image_points,
    accept_candidate_correspondences,
    cluster_ground_line_directions,
    build_proposals_from_homography,
    draw_net_anchor_overlay,
    load_player_foot_points_from_tracks,
    project_standard_keypoints,
    solve_net_anchor_court_from_frame,
)


SYNTHETIC_IMAGE_SIZE = (960, 540)


def _synthetic_homography() -> list[list[float]]:
    world = np.asarray(
        [
            [-3.048, -6.7056],
            [3.048, -6.7056],
            [3.048, 6.7056],
            [-3.048, 6.7056],
        ],
        dtype=np.float64,
    )
    image = np.asarray(
        [
            [170.0, 500.0],
            [810.0, 468.0],
            [665.0, 126.0],
            [294.0, 136.0],
        ],
        dtype=np.float64,
    )
    homography, _ = cv2.findHomography(world, image)
    return homography.tolist()


def _make_synthetic_frame() -> tuple[np.ndarray, list[list[float]]]:
    homography = _synthetic_homography()
    frame = np.zeros((SYNTHETIC_IMAGE_SIZE[1], SYNTHETIC_IMAGE_SIZE[0], 3), dtype=np.uint8)
    frame[:] = (36, 108, 66)
    keypoints = project_standard_keypoints(homography)

    def point(name: str) -> tuple[int, int]:
        proposal = keypoints[name]
        return int(round(proposal[0])), int(round(proposal[1]))

    blue = (235, 185, 65)
    white = (248, 248, 248)
    lines = [
        ("near_left_corner", "near_right_corner", blue),
        ("near_nvz_left", "near_nvz_right", blue),
        ("net_left_sideline", "net_right_sideline", white),
        ("far_nvz_left", "far_nvz_right", blue),
        ("far_left_corner", "far_right_corner", blue),
        ("near_left_corner", "far_left_corner", blue),
        ("near_right_corner", "far_right_corner", blue),
        ("near_baseline_center", "near_nvz_center", blue),
        ("far_nvz_center", "far_baseline_center", blue),
    ]
    for start, end, color in lines:
        cv2.line(frame, point(start), point(end), color, 5, cv2.LINE_AA)

    net_left = point("net_left_sideline")
    net_right = point("net_right_sideline")
    cv2.line(frame, (net_left[0], net_left[1] + 8), (net_right[0], net_right[1] + 8), (28, 28, 28), 16, cv2.LINE_AA)
    cv2.line(frame, net_left, net_right, white, 7, cv2.LINE_AA)
    cv2.line(frame, (net_left[0], net_left[1] - 35), (net_left[0], net_left[1] + 42), white, 5, cv2.LINE_AA)
    cv2.line(frame, (net_right[0], net_right[1] - 35), (net_right[0], net_right[1] + 42), white, 5, cv2.LINE_AA)
    return frame, homography


def test_project_standard_keypoints_uses_regulation_pickleball_geometry() -> None:
    projected = project_standard_keypoints(_synthetic_homography())

    assert set(projected) == {
        "near_left_corner",
        "near_baseline_center",
        "near_right_corner",
        "far_right_corner",
        "far_baseline_center",
        "far_left_corner",
        "near_nvz_left",
        "near_nvz_center",
        "near_nvz_right",
        "net_left_sideline",
        "net_center",
        "net_right_sideline",
        "far_nvz_left",
        "far_nvz_center",
        "far_nvz_right",
    }
    assert projected["net_center"] == pytest.approx([480.0, 278.0], abs=25.0)
    assert projected["near_left_corner"][1] > projected["far_left_corner"][1]


def test_refinement_rejects_candidate_that_worsens_existing_residual() -> None:
    homography = _synthetic_homography()
    projected = project_standard_keypoints(homography)
    base = {
        "near_left_corner": projected["near_left_corner"],
        "near_right_corner": projected["near_right_corner"],
        "far_left_corner": projected["far_left_corner"],
        "far_right_corner": projected["far_right_corner"],
    }
    good_candidate = {"near_nvz_left": projected["near_nvz_left"], "near_nvz_right": projected["near_nvz_right"]}
    bad_candidate = {"near_nvz_left": [40.0, 40.0], "near_nvz_right": [900.0, 500.0]}

    good = accept_candidate_correspondences(base, good_candidate, residual_gate_px=2.0)
    bad = accept_candidate_correspondences(base, bad_candidate, residual_gate_px=2.0)

    assert good.accepted is True
    assert good.reason == "accepted"
    assert good.median_residual_px <= 0.5
    assert bad.accepted is False
    assert bad.reason == "candidate_worsens_residual"


def test_top_net_points_are_excluded_from_planar_homography_fits() -> None:
    projected = project_standard_keypoints(_synthetic_homography())
    with pytest.raises(ValueError, match="at least 4 named court correspondences"):
        _homography_from_named_image_points(
            {
                "net_left_sideline": projected["net_left_sideline"],
                "net_center": projected["net_center"],
                "net_right_sideline": projected["net_right_sideline"],
                "near_left_corner": projected["near_left_corner"],
            }
        )

    net = NetDetection(
        tape_line=((290.0, 278.0), (670.0, 278.0)),
        post_tops=((290.0, 260.0), (670.0, 260.0)),
        post_bases=((290.0, 315.0), (670.0, 315.0)),
        confidence=0.8,
        evidence={"support_px": 380, "mesh_band_contrast": 0.4, "post_count": 2},
    )
    detected = {
        name: projected[name]
        for name in (
            "near_left_corner",
            "near_right_corner",
            "far_right_corner",
            "far_left_corner",
            "net_left_sideline",
            "net_center",
            "net_right_sideline",
        )
    }

    hypotheses = _build_global_fit_hypotheses(
        detected_points=detected,
        net=net,
        line_evidence={},
        image_size=SYNTHETIC_IMAGE_SIZE,
        player_prior={"point_count": 0},
    )

    all_detected = next(item for item in hypotheses if item["name"] == "all_detected_correspondences")
    assert set(all_detected["accepted_correspondence_names"]) == {
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
    }
    assert not set(all_detected["residuals_by_name"]) & {"net_left_sideline", "net_center", "net_right_sideline"}


def test_build_proposals_marks_low_confidence_points_for_user_input() -> None:
    net = NetDetection(
        tape_line=((290.0, 278.0), (670.0, 278.0)),
        post_tops=((290.0, 260.0), (670.0, 260.0)),
        post_bases=((290.0, 315.0), (670.0, 315.0)),
        confidence=0.72,
        evidence={"support_px": 380, "mesh_band_contrast": 0.4, "post_count": 2},
    )
    evidence = {
        "net": LineEvidence("net", ((290.0, 278.0), (670.0, 278.0)), 0.72, 380, "net", [248.0, 248.0, 248.0]),
        "near_nvz": LineEvidence("near_nvz", ((224.0, 368.0), (744.0, 356.0)), 0.25, 22, "kitchen", [235.0, 185.0, 65.0]),
    }

    artifact = build_proposals_from_homography(
        _synthetic_homography(),
        image_size=SYNTHETIC_IMAGE_SIZE,
        net=net,
        line_evidence=evidence,
        accepted_correspondence_names=["net_left_sideline", "net_right_sideline"],
        residuals_by_name={"near_left_corner": 24.0, "near_right_corner": 24.0},
        confidence_threshold=DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
    )

    near_left = ProposalPoint(**artifact["corners"]["near_left"])
    assert 0.0 <= near_left.confidence <= 1.0
    assert artifact["solver_confidence"] < DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD
    assert "near_left" in artifact["needs_user_input"]
    assert artifact["corners"]["near_left"]["evidence"]["net"] >= 1
    assert artifact["keypoints"]["net_center"]["evidence"]["net"] >= 1


def test_hypothesis_confidence_cap_forces_user_confirmation_on_seed_only_fit() -> None:
    net = NetDetection(
        tape_line=((290.0, 278.0), (670.0, 278.0)),
        post_tops=((290.0, 260.0), (670.0, 260.0)),
        post_bases=((290.0, 315.0), (670.0, 315.0)),
        confidence=0.8,
        evidence={"support_px": 380, "mesh_band_contrast": 0.4, "post_count": 2},
    )
    evidence = {
        name: LineEvidence(name, endpoints, 0.8, 800, stage, [248.0, 248.0, 248.0])
        for name, endpoints, stage in [
            ("net", ((290.0, 278.0), (670.0, 278.0)), "net"),
            ("near_baseline", ((180.0, 470.0), (780.0, 470.0)), "baseline"),
            ("far_baseline", ((260.0, 120.0), (700.0, 120.0)), "baseline"),
            ("left_sideline", ((180.0, 470.0), (260.0, 120.0)), "sideline"),
            ("right_sideline", ((780.0, 470.0), (700.0, 120.0)), "sideline"),
        ]
    }
    artifact = build_proposals_from_homography(
        _synthetic_homography(),
        image_size=SYNTHETIC_IMAGE_SIZE,
        net=net,
        line_evidence=evidence,
        accepted_correspondence_names=[],
        residuals_by_name={},
        confidence_threshold=DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
    )

    _apply_hypothesis_confidence_cap(artifact, max_confidence=0.64, reason="seed_only_hypothesis_not_globally_verified")

    assert artifact["solver_confidence"] == pytest.approx(0.64)
    assert set(artifact["needs_user_input"]) == {"near_left", "near_right", "far_right", "far_left"}
    assert artifact["needs_user_confirmation"] is True


def test_seed_only_hypothesis_caps_at_low_confidence_and_fails_self_verification() -> None:
    net = NetDetection(
        tape_line=((290.0, 278.0), (670.0, 278.0)),
        post_tops=((290.0, 260.0), (670.0, 260.0)),
        post_bases=((290.0, 315.0), (670.0, 315.0)),
        confidence=0.8,
        evidence={"support_px": 380, "mesh_band_contrast": 0.4, "post_count": 2},
    )
    artifact = build_proposals_from_homography(
        _synthetic_homography(),
        image_size=SYNTHETIC_IMAGE_SIZE,
        net=net,
        line_evidence={},
        accepted_correspondence_names=[],
        residuals_by_name={},
        confidence_threshold=DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
    )

    cap, reason = _confidence_cap_for_hypothesis(
        {"name": "corner_subset_refinement_seed", "median_residual_px": 42.0}
    )
    _apply_hypothesis_confidence_cap(artifact, max_confidence=cap, reason=reason)

    assert cap == pytest.approx(0.30)
    assert artifact["solver_confidence"] <= 0.30
    assert artifact["needs_user_confirmation"] is True
    assert artifact["self_verification"]["status"] == "failed"
    assert artifact["self_verification"]["promotion_allowed"] is False
    assert "seed_only_hypothesis_not_globally_verified" in artifact["self_verification"]["reasons"]


def test_solver_recovers_synthetic_court_and_draws_overlay() -> None:
    frame, homography = _make_synthetic_frame()
    truth = project_standard_keypoints(homography)

    artifact = solve_net_anchor_court_from_frame(frame, clip_id="synthetic")

    assert artifact["solver"]["strategy"] == "multi_hypothesis_global_fit_v2"
    assert artifact["hypotheses"]
    assert artifact["hypotheses"][0]["evidence_mass"] > 0.0
    assert artifact["net"]["confidence"] >= 0.6
    assert artifact["solver_confidence"] >= 0.6
    for name in ("near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner"):
        proposal = artifact["keypoints"][name]["xy"]
        expected = truth[name]
        assert math.hypot(proposal[0] - expected[0], proposal[1] - expected[1]) < 25.0

    overlay = draw_net_anchor_overlay(frame, artifact)
    assert overlay.shape == frame.shape
    assert np.abs(overlay.astype(np.int16) - frame.astype(np.int16)).sum() > 0


def test_cluster_ground_line_directions_finds_two_dominant_families() -> None:
    raw_segments = [
        ((100.0, 100.0), (900.0, 104.0)),
        ((110.0, 250.0), (890.0, 254.0)),
        ((120.0, 390.0), (880.0, 394.0)),
        ((250.0, 80.0), (320.0, 620.0)),
        ((650.0, 90.0), (720.0, 630.0)),
        ((500.0, 120.0), (548.0, 500.0)),
    ]

    clusters = cluster_ground_line_directions(raw_segments)

    assert len(clusters) == 2
    assert clusters[0]["support_px"] > clusters[1]["support_px"]
    assert abs(clusters[0]["angle_deg"] - clusters[1]["angle_deg"]) > 25.0


def test_player_feet_prior_loads_bbox_bottom_points_from_tracks(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "players": [
                    {
                        "id": 1,
                        "frames": [
                            {"t": 0.0, "bbox": [100.0, 50.0, 180.0, 250.0], "conf": 0.9},
                            {"t": 0.1, "bbox": [120.0, 60.0, 200.0, 270.0], "conf": 0.8},
                        ],
                    },
                    {"id": 2, "frames": [{"t": 0.0, "bbox": [500.0, 80.0, 620.0, 360.0], "conf": 0.7}]},
                ],
            }
        ),
        encoding="utf-8",
    )

    points = load_player_foot_points_from_tracks(tracks_path)

    assert points == [[140.0, 250.0], [160.0, 270.0], [560.0, 360.0]]


def test_solver_records_player_feet_prior_when_supplied() -> None:
    frame, _ = _make_synthetic_frame()
    foot_points = [[240.0, 420.0], [450.0, 430.0], [700.0, 410.0]]

    artifact = solve_net_anchor_court_from_frame(frame, clip_id="synthetic", player_foot_points=foot_points)

    assert artifact["player_feet_prior"]["point_count"] == 3
    assert artifact["player_feet_prior"]["ground_region_bbox"] == pytest.approx([240.0, 410.0, 700.0, 430.0])
    assert "player_feet_ground_prior" in artifact["notes"]


def test_solve_net_anchor_court_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/solve_net_anchor_court.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--input" in completed.stdout
    assert "--out-dir" in completed.stdout
    assert "--gt-corners" in completed.stdout
    assert "--tracks-json" in completed.stdout
    assert "--allow-internal-val-labels" in completed.stdout


def test_solve_net_anchor_court_cli_runs_on_synthetic_image(tmp_path: Path) -> None:
    frame, _ = _make_synthetic_frame()
    image_path = tmp_path / "synthetic.jpg"
    out_dir = tmp_path / "out"
    cv2.imwrite(str(image_path), frame)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/solve_net_anchor_court.py",
            "--input",
            str(image_path),
            "--clip-id",
            "synthetic_cli",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((out_dir / "court_corner_proposals.json").read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["artifact_type"] == "racketsport_net_anchor_court_proposals"
    assert (out_dir / "court_corner_proposals_overlay.jpg").exists()


def test_solve_net_anchor_court_cli_rejects_protected_heldout_gt(tmp_path: Path) -> None:
    frame, _ = _make_synthetic_frame()
    image_path = tmp_path / "synthetic.jpg"
    cv2.imwrite(str(image_path), frame)
    protected = tmp_path / "outdoor_webcam_iynbd_1500_long_high_baseline" / "labels" / "court_corners.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/solve_net_anchor_court.py",
            "--input",
            str(image_path),
            "--clip-id",
            "synthetic_cli",
            "--out-dir",
            str(tmp_path / "out"),
            "--gt-corners",
            str(protected),
            "--allow-internal-val-labels",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Outdoor/Indoor labels are never allowed" in completed.stderr
