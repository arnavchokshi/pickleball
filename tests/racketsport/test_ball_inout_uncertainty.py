from __future__ import annotations

import math

import pytest

from threed.racketsport.ball_inout_uncertainty import (
    BALL_RADIUS_UNCERTAINTY_M,
    BOUNCE_DETECTION_FRAME_WINDOW,
    METHOD_CAMERA_GEOMETRY,
    METHOD_FIXED_OVERRIDE,
    PICKLEBALL_DROP_TEST_HEIGHT_M,
    STANDARD_GRAVITY_MPS2,
    CameraPose,
    binding_boundary_axis,
    bounce_geometric_uncertainty_m,
    fixed_override_breakdown,
    ground_point_at_height,
    physics_constants_manifest,
    reference_vertical_impact_speed_mps,
    solve_manual_corner_camera_pose,
)
from threed.racketsport.court_calibration import project_image_points_to_world
from threed.racketsport.court_templates import get_court_template

# Real reviewed manual court corners (burlington_gold_0300_low_steep_corner),
# a "low, steep corner" clip -- exactly the camera geometry the M4/M5 owner
# review flagged as producing dishonest confident "out" calls.
_BURLINGTON_IMAGE_PTS = [[146.0, 328.0], [555.0, 522.0], [924.0, 259.0], [688.0, 248.0]]


def _burlington_world_pts() -> list[list[float]]:
    return [list(point) for point in get_court_template("pickleball").corners_m]


def _synthetic_overhead_pose(height_m: float = 6.0, focal_px: float = 1000.0) -> CameraPose:
    """A camera directly above the court origin looking straight down."""

    return CameraPose(
        fx=focal_px,
        fy=focal_px,
        cx=500.0,
        cy=300.0,
        R=[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        t=[0.0, 0.0, height_m],
        camera_center_world=[0.0, 0.0, height_m],
        camera_height_m=height_m,
        reprojection_error_px_median=0.0,
        reprojection_error_px_p95=0.0,
        source="synthetic_test_overhead",
    )


def test_reference_vertical_impact_speed_is_pure_freefall_physics() -> None:
    expected = math.sqrt(2.0 * STANDARD_GRAVITY_MPS2 * PICKLEBALL_DROP_TEST_HEIGHT_M)
    assert reference_vertical_impact_speed_mps() == pytest.approx(expected)
    # Sanity: USA Pickleball's 78 in drop test height, not a fitted number.
    assert PICKLEBALL_DROP_TEST_HEIGHT_M == pytest.approx(1.9812, abs=1e-6)


def test_physics_constants_manifest_has_no_free_parameters_tied_to_labels() -> None:
    manifest = physics_constants_manifest()
    # Every entry is a (value, justification) pair; none are described as
    # fit/tuned to the reviewed human in/out labels.
    forbidden_phrases = ("fit to", "tuned to", "reviewed label", "human review label")
    for name, entry in manifest.items():
        assert "value" in entry and "justification" in entry, name
        justification = entry["justification"].lower()
        assert not any(phrase in justification for phrase in forbidden_phrases), name


def test_ground_point_at_height_matches_closed_form_pinhole_ray() -> None:
    height_m = 6.0
    pose = _synthetic_overhead_pose(height_m=height_m)
    u, v, h_target = 620.0, 380.0, 0.35

    point = ground_point_at_height([u, v], pose, h_target)

    s = height_m - h_target
    expected_x = s * (u - pose.cx) / pose.fx
    expected_y = -s * (v - pose.cy) / pose.fy
    assert point[0] == pytest.approx(expected_x, abs=1e-9)
    assert point[1] == pytest.approx(expected_y, abs=1e-9)
    assert point[2] == pytest.approx(h_target, abs=1e-9)


def test_elevation_displacement_is_zero_at_the_image_center_nadir() -> None:
    """Directly under an overhead camera, residual ball height causes no
    horizontal ground-plane error: the viewing ray is vertical there."""

    pose = _synthetic_overhead_pose()
    point0 = ground_point_at_height([pose.cx, pose.cy], pose, 0.0)
    point_h = ground_point_at_height([pose.cx, pose.cy], pose, 0.4)
    assert point0[0] == pytest.approx(0.0, abs=1e-9)
    assert point0[1] == pytest.approx(0.0, abs=1e-9)
    assert point_h[0] == pytest.approx(0.0, abs=1e-9)
    assert point_h[1] == pytest.approx(0.0, abs=1e-9)


def test_elevation_displacement_grows_with_distance_from_nadir_closed_form() -> None:
    height_m = 6.0
    pose = _synthetic_overhead_pose(height_m=height_m)
    h_max = 0.4

    near_pixel = [pose.cx + 20.0, pose.cy]
    far_pixel = [pose.cx + 400.0, pose.cy]

    def displacement_x(pixel: list[float]) -> float:
        p0 = ground_point_at_height(pixel, pose, 0.0)
        ph = ground_point_at_height(pixel, pose, h_max)
        return abs(ph[0] - p0[0])

    near_disp = displacement_x(near_pixel)
    far_disp = displacement_x(far_pixel)

    # Closed form: |dX| = h_max * |u - cx| / fx (independent of camera height).
    assert near_disp == pytest.approx(h_max * 20.0 / pose.fx, abs=1e-9)
    assert far_disp == pytest.approx(h_max * 400.0 / pose.fx, abs=1e-9)
    assert far_disp > near_disp


def test_binding_boundary_axis_picks_the_nearer_line_direction() -> None:
    template = get_court_template("pickleball")
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0

    # Near a sideline (small x-margin) -> axis "x".
    assert binding_boundary_axis([half_width - 0.05, 0.0], sport="pickleball") == "x"
    # Near a baseline (small y-margin) -> axis "y".
    assert binding_boundary_axis([0.0, half_length - 0.05], sport="pickleball") == "y"


def test_solve_manual_corner_camera_pose_reproduces_the_four_corners_closely() -> None:
    pose = solve_manual_corner_camera_pose(_BURLINGTON_IMAGE_PTS, _burlington_world_pts())

    assert pose.camera_height_m > 0.0
    assert pose.reprojection_error_px_median < 8.0
    assert pose.reprojection_error_px_p95 < 15.0
    assert pose.source == "manual_corner_focal_length_search_v1"


def test_pose_ray_cast_at_zero_height_matches_the_ground_homography() -> None:
    """The new pose is solved independently of the DLT homography, but at
    height 0 both must agree closely -- they describe the same ground plane
    for the same 4 corner correspondences."""

    from threed.racketsport.court_calibration import homography_from_planar_points

    world_pts = _burlington_world_pts()
    homography = homography_from_planar_points(world_pts, _BURLINGTON_IMAGE_PTS)
    pose = solve_manual_corner_camera_pose(_BURLINGTON_IMAGE_PTS, world_pts)

    probe_pixels = [[1090.0, 472.0], [1458.0, 483.0], [1042.0, 577.0]]
    homography_points = project_image_points_to_world(homography, probe_pixels)
    for pixel, expected in zip(probe_pixels, homography_points, strict=True):
        pose_point = ground_point_at_height(pixel, pose, 0.0)
        assert pose_point[0] == pytest.approx(expected[0], abs=0.15)
        assert pose_point[1] == pytest.approx(expected[1], abs=0.15)


def test_bounce_geometric_uncertainty_is_wider_on_a_steep_low_camera_than_overhead() -> None:
    """The whole point of the model: the same residual-height physics
    produces a small band on a near-overhead camera and a large band on a
    steep, low, far-court camera (burlington's real geometry)."""

    overhead_pose = _synthetic_overhead_pose(height_m=8.0)
    overhead_pixel = [overhead_pose.cx + 150.0, overhead_pose.cy + 100.0]
    overhead_world = ground_point_at_height(overhead_pixel, overhead_pose, 0.0)[:2]
    overhead_result = bounce_geometric_uncertainty_m(
        contact_xy_img=overhead_pixel,
        world_xy=overhead_world,
        pose=overhead_pose,
        sport="pickleball",
        fps=60.0,
    )

    steep_pose = solve_manual_corner_camera_pose(_BURLINGTON_IMAGE_PTS, _burlington_world_pts())
    # A far-court contact point, well outside the calibrated corner box --
    # exactly the regime the owner review flagged.
    steep_pixel = [1458.0, 483.0]
    steep_world = ground_point_at_height(steep_pixel, steep_pose, 0.0)[:2]
    steep_result = bounce_geometric_uncertainty_m(
        contact_xy_img=steep_pixel,
        world_xy=steep_world,
        pose=steep_pose,
        sport="pickleball",
        fps=60.0,
    )

    assert steep_result["breakdown"]["sigma_depth_m"] > overhead_result["breakdown"]["sigma_depth_m"]
    assert steep_result["uncertainty_m"] > overhead_result["uncertainty_m"]
    # The fixed 0.05 m radius the old model used is far too tight for the
    # steep/far case; the new band should honestly be much wider.
    assert steep_result["uncertainty_m"] > 0.05


def test_bounce_geometric_uncertainty_breakdown_sums_in_quadrature() -> None:
    pose = _synthetic_overhead_pose()
    pixel = [pose.cx + 250.0, pose.cy - 60.0]
    world = ground_point_at_height(pixel, pose, 0.0)[:2]

    result = bounce_geometric_uncertainty_m(
        contact_xy_img=pixel,
        world_xy=world,
        pose=pose,
        sport="pickleball",
        fps=30.0,
        reprojection_error_px_p95=3.0,
    )
    breakdown = result["breakdown"]
    expected_total = math.sqrt(
        breakdown["sigma_reproj_m"] ** 2
        + breakdown["sigma_depth_m"] ** 2
        + breakdown["sigma_ballradius_m"] ** 2
        + breakdown["sigma_localization_m"] ** 2
    )
    assert result["uncertainty_m"] == pytest.approx(expected_total)
    assert breakdown["sigma_ballradius_m"] == pytest.approx(BALL_RADIUS_UNCERTAINTY_M)
    assert breakdown["frames_window"] == pytest.approx(BOUNCE_DETECTION_FRAME_WINDOW)
    assert breakdown["method"] == METHOD_CAMERA_GEOMETRY
    assert breakdown["dt_s"] * 30.0 == pytest.approx(BOUNCE_DETECTION_FRAME_WINDOW)


def test_bounce_geometric_uncertainty_shorter_fps_window_gives_smaller_band() -> None:
    """Higher fps -> smaller dt for the same +/-2 frame detector tolerance
    -> smaller h_max -> tighter band, holding geometry fixed."""

    pose = solve_manual_corner_camera_pose(_BURLINGTON_IMAGE_PTS, _burlington_world_pts())
    pixel = [1458.0, 483.0]
    world = ground_point_at_height(pixel, pose, 0.0)[:2]

    slow = bounce_geometric_uncertainty_m(contact_xy_img=pixel, world_xy=world, pose=pose, fps=30.0)
    fast = bounce_geometric_uncertainty_m(contact_xy_img=pixel, world_xy=world, pose=pose, fps=120.0)

    assert fast["breakdown"]["h_max_m"] < slow["breakdown"]["h_max_m"]
    assert fast["uncertainty_m"] < slow["uncertainty_m"]


def test_fixed_override_breakdown_marks_method_and_carries_the_override_value() -> None:
    breakdown = fixed_override_breakdown(0.05)
    assert breakdown["method"] == METHOD_FIXED_OVERRIDE
    assert breakdown["sigma_depth_m"] == 0.0
    assert breakdown["sigma_localization_m"] == pytest.approx(0.05)


def test_solve_manual_corner_camera_pose_requires_at_least_four_points() -> None:
    with pytest.raises(ValueError):
        solve_manual_corner_camera_pose(_BURLINGTON_IMAGE_PTS[:3], _burlington_world_pts()[:3])
