from __future__ import annotations

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.calibration_overlay import build_calibration_overlay
from threed.racketsport.court_auto_evidence import (
    _merge_line_observations,
    _merge_net_observations,
    build_auto_court_line_evidence_from_image,
    calibration_for_image_size,
    select_top_net_observation,
)
from threed.racketsport.net_plane import project_net_plane
from threed.racketsport.net_plane import build_net_plane
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    CourtLineObservation,
    NetLineObservation,
    ReprojectionError,
)


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _synthetic_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[20.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def _half_resolution_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[10.0, 0.0, 480.0], [0.0, -10.0, 270.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=480.0, cy=270.0, dist=[], source="synthetic_half"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[[380.0, 490.0], [580.0, 490.0], [580.0, 50.0], [380.0, 50.0]],
        world_pts=minimal_calibration_world_pts(),
    )


def _untrusted_top_net_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 500.0], [0.0, 100.0, 300.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=300.0, dist=[], source="synthetic"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.35, 0.0, 1.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def test_calibration_for_image_size_scales_homography_and_intrinsics_to_frame_resolution():
    calibration = _half_resolution_calibration()

    scaled = calibration_for_image_size(calibration, width=1920, height=1080)

    assert scaled.homography[0] == pytest.approx([20.0, 0.0, 960.0])
    assert scaled.homography[1] == pytest.approx([0.0, -20.0, 540.0])
    assert scaled.homography[2] == pytest.approx([0.0, 0.0, 1.0])
    assert scaled.intrinsics.fx == pytest.approx(1000.0)
    assert scaled.intrinsics.fy == pytest.approx(1000.0)
    assert scaled.intrinsics.cx == pytest.approx(960.0)
    assert scaled.intrinsics.cy == pytest.approx(540.0)
    assert scaled.image_pts[0] == pytest.approx([760.0, 980.0])
    assert scaled.image_size == (1920, 1080)

    rescaled = calibration_for_image_size(scaled, width=1920, height=1080)

    assert rescaled.homography == scaled.homography
    assert rescaled.image_pts == scaled.image_pts


def test_calibration_for_image_size_does_not_rescale_off_center_intrinsics():
    calibration = _synthetic_calibration().model_copy(
        update={
            "intrinsics": CameraIntrinsics(
                fx=1000.0,
                fy=1000.0,
                cx=950.0,
                cy=535.0,
                dist=[],
                source="synthetic_off_center",
            ),
            "image_size": None,
        }
    )

    scaled = calibration_for_image_size(calibration, width=1920, height=1080)

    assert scaled.homography == calibration.homography
    assert scaled.intrinsics.cx == pytest.approx(950.0)
    assert scaled.intrinsics.cy == pytest.approx(535.0)


def _blank_frame() -> object:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _draw_overlay_line(image: object, overlay: dict, line_id: str) -> None:
    line = next(item for item in overlay["court_lines"] if item["id"] == line_id)
    start, end = line["image"]
    cv2.line(image, (round(start[0]), round(start[1])), (round(end[0]), round(end[1])), (255, 255, 255), 5)


def _draw_top_net(image: object, overlay: dict) -> None:
    left = overlay["net_points"]["left_post"]
    center = overlay["net_points"]["center"]
    right = overlay["net_points"]["right_post"]
    cv2.line(image, (round(left[0]), round(left[1])), (round(center[0]), round(center[1])), (255, 255, 255), 5)
    cv2.line(image, (round(center[0]), round(center[1])), (round(right[0]), round(right[1])), (255, 255, 255), 5)


def test_merge_line_observations_aligns_reversed_segment_endpoints():
    observations = [
        CourtLineObservation(
            line_id="near_nvz",
            image_segment=[[0.0, 10.0], [100.0, 10.0]],
            confidence=0.9,
            frame_indexes=[0],
            residual_px={"mean": 1.0, "p95": 1.0},
            visible_fraction=1.0,
            source="test",
        ),
        CourtLineObservation(
            line_id="near_nvz",
            image_segment=[[101.0, 10.0], [1.0, 10.0]],
            confidence=0.8,
            frame_indexes=[1],
            residual_px={"mean": 2.0, "p95": 2.0},
            visible_fraction=1.0,
            source="test",
        ),
    ]

    [merged] = _merge_line_observations(observations)

    assert merged.image_segment[0] == pytest.approx([0.5, 10.0])
    assert merged.image_segment[1] == pytest.approx([100.5, 10.0])
    assert merged.residual_px.mean == pytest.approx(1.5)
    assert merged.frame_indexes == [0, 1]


def test_merge_net_observations_aligns_reversed_top_net_points():
    observations = [
        NetLineObservation(
            net_id="top_net",
            image_points=[[0.0, 200.0], [500.0, 195.0], [1000.0, 200.0]],
            confidence=0.9,
            frame_indexes=[0],
            residual_px={"mean": 1.0, "p95": 1.0},
            source="test",
        ),
        NetLineObservation(
            net_id="top_net",
            image_points=[[1002.0, 200.0], [501.0, 195.0], [2.0, 200.0]],
            confidence=0.8,
            frame_indexes=[1],
            residual_px={"mean": 2.0, "p95": 2.0},
            source="test",
        ),
    ]

    [merged] = _merge_net_observations(observations)

    assert merged.image_points[0] == pytest.approx([1.0, 200.0])
    assert merged.image_points[1] == pytest.approx([500.5, 195.0])
    assert merged.image_points[2] == pytest.approx([1001.0, 200.0])
    assert merged.frame_indexes == [0, 1]


def test_auto_evidence_accepts_observed_kitchen_centerlines_and_top_net():
    calibration = _synthetic_calibration()
    net_plane = build_net_plane("pickleball")
    overlay = build_calibration_overlay(calibration, net_plane=net_plane)
    image = _blank_frame()
    for line_id in ("near_nvz", "far_nvz", "near_centerline", "far_centerline"):
        _draw_overlay_line(image, overlay, line_id)
    _draw_top_net(image, overlay)

    evidence = build_auto_court_line_evidence_from_image(image, calibration, net_plane=net_plane)

    assert evidence.aggregate.auto_calibration_ready is True
    assert set(evidence.aggregate.accepted_line_ids) >= {"near_nvz", "far_nvz", "near_centerline", "far_centerline"}
    assert evidence.aggregate.missing_required_net_ids == []
    assert evidence.net_observations[0].net_id == "top_net"


def test_auto_evidence_refuses_readiness_when_centerlines_are_not_seen():
    calibration = _synthetic_calibration()
    net_plane = build_net_plane("pickleball")
    overlay = build_calibration_overlay(calibration, net_plane=net_plane)
    image = _blank_frame()
    for line_id in ("near_nvz", "far_nvz"):
        _draw_overlay_line(image, overlay, line_id)
    _draw_top_net(image, overlay)

    evidence = build_auto_court_line_evidence_from_image(image, calibration, net_plane=net_plane)

    assert evidence.aggregate.auto_calibration_ready is False
    assert evidence.aggregate.missing_required_line_ids == ["near_centerline", "far_centerline"]
    assert "missing_near_centerline" in evidence.aggregate.reasons


def test_top_net_observation_rejects_parallel_non_overlapping_candidate():
    calibration = _synthetic_calibration()
    net_plane = build_net_plane("pickleball")
    net_points = project_net_plane(calibration, net_plane)
    right = net_points["right_post"]
    shifted_segment = (
        (right[0] + 200.0, right[1]),
        (right[0] + 360.0, right[1]),
    )

    observation = select_top_net_observation(calibration, net_plane, [shifted_segment])

    assert observation is None


def test_top_net_observation_rejects_untrusted_pnp_projection_even_with_matching_candidate():
    calibration = _untrusted_top_net_calibration()
    net_plane = build_net_plane("pickleball")
    net_points = project_net_plane(calibration, net_plane)
    candidate_segment = (tuple(net_points["left_post"]), tuple(net_points["right_post"]))

    observation = select_top_net_observation(calibration, net_plane, [candidate_segment])

    assert observation is None


def test_top_net_observation_rejects_projection_with_implausible_top_to_ground_length_ratio():
    calibration = _synthetic_calibration().model_copy(
        deep=True,
        update={"homography": [[1.0, 0.0, 960.0], [0.0, -20.0, 540.0], [0.0, 0.0, 1.0]]},
    )
    net_plane = build_net_plane("pickleball")
    net_points = project_net_plane(calibration, net_plane)
    candidate_segment = (tuple(net_points["left_post"]), tuple(net_points["right_post"]))

    observation = select_top_net_observation(calibration, net_plane, [candidate_segment])

    assert observation is None


def test_top_net_observation_rejects_estimated_review_frame_intrinsics_even_with_matching_candidate():
    calibration = _synthetic_calibration()
    calibration = calibration.model_copy(
        deep=True,
        update={"intrinsics": calibration.intrinsics.model_copy(update={"source": "estimated_from_review_frame"})},
    )
    net_plane = build_net_plane("pickleball")
    net_points = project_net_plane(calibration, net_plane)
    candidate_segment = (tuple(net_points["left_post"]), tuple(net_points["right_post"]))

    observation = select_top_net_observation(calibration, net_plane, [candidate_segment])

    assert observation is None
