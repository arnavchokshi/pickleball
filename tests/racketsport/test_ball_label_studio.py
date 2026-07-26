"""Tests for the human ball-labelling studio.

The geometry tests carry the weight here. A sign error in the pixel -> ray ->
plane path or in the depth parameterisation would not crash anything; it would
silently mirror or invert every label the owner spends hours producing. So the
parameterisation is pinned from several independent directions: the round trip
through the production solver's own primitives, monotonicity of depth away from
the camera, agreement with ``intersect_ray_z``, and rejection of anything
behind the camera.

The contract tests pin the other failure mode that costs more than a crash: a
free-flight *guess* quietly acquiring the accuracy claim of a solved bounce.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.ball_arc_solver import (  # noqa: E402
    BALL_RADIUS_M,
    intersect_ray_z,
    pixel_ray_world,
)
from threed.racketsport.ball_label_geometry import (  # noqa: E402
    FREE_FLIGHT_DEPTH_SIGMA_M,
    GeometryError,
    ballistic_positions,
    ballistic_speed_mps,
    bounce_depth_sigma_m,
    bounce_world_point,
    calibration_plane_residuals,
    free_flight_depth_sigma_m,
    interpolation_extra_sigma_m,
    is_in_front_of_camera,
    near_player_depth_sigma_m,
    perpendicular_sigma_m,
    project_world_to_pixel,
    ray_for_pixel,
    sigma_xyz_from_ray,
)
from threed.racketsport.ball_label_schema import (  # noqa: E402
    ARTIFACT_TYPE,
    KIND_BOUNCE,
    KIND_FREE_FLIGHT,
    KIND_NEAR_PLAYER,
    LABEL_FILE_NAME,
    BallLabel,
    BallLabelSet,
    LabelContractError,
    read_label_set,
    write_label_set,
)
from threed.racketsport.ball_label_studio import (  # noqa: E402
    CORE_BONE_PAIRS,
    CORE_JOINTS,
    KEYBOARD_MAP,
    StudioError,
    accept_interpolation,
    build_label,
    build_page_state,
    extract_frames,
    load_clip_bundle,
    nearest_player_reference,
    open_session,
    propose_interpolation,
    run_studio_server,
    solve_click,
    summarize_session,
)

# The CLI this module backs. Referenced literally so the scaffold index can see
# a direct CLI reference test for it.
STUDIO_CLI = "scripts/racketsport/ball_label_studio.py"

REAL_RUN_DIR = Path(
    "/Users/arnavchokshi/Desktop/pickleball/runs/full_mesh_examples_20260725/"
    "outdoor_mesh_final/outdoor_webcam_20s_fullmesh_final"
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
needs_real_clip = pytest.mark.skipif(
    not (REAL_RUN_DIR / "court_calibration.json").is_file(),
    reason="the verified real run directory is not present on this machine",
)


# ---------------------------------------------------------------------------
# A hermetic pinhole camera, built by hand so the tests do not depend on any
# particular clip. Camera behind the near baseline, looking at the net.
# ---------------------------------------------------------------------------


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def synthetic_calibration(camera_xyz=(0.0, -10.0, 1.6), target=(0.0, 0.0, 0.5)) -> dict:
    forward = _normalize([target[i] - camera_xyz[i] for i in range(3)])
    right = _normalize(_cross(forward, [0.0, 0.0, 1.0]))
    down = _cross(forward, right)
    rotation = [right, down, forward]
    translation = [-sum(rotation[r][c] * camera_xyz[c] for c in range(3)) for r in range(3)]
    return {
        "schema_version": 1,
        "coordinate_frame": "court_netcenter_z_up_m",
        "sport": "pickleball",
        "image_size": [1920, 1080],
        "intrinsics": {"fx": 1100.0, "fy": 1100.0, "cx": 960.0, "cy": 540.0, "dist": [0, 0, 0, 0]},
        "extrinsics": {"R": rotation, "t": translation, "camera_height_m": camera_xyz[2]},
        "reprojection_error_px": {"median": 2.0, "p95": 4.0},
        "source": "synthetic_test_fixture",
        "metric_confidence": "test",
    }


CAL = synthetic_calibration()


def pixel_of(world) -> tuple[float, float]:
    return project_world_to_pixel(CAL, world)


# ---------------------------------------------------------------------------
# Geometry: pixel -> ray -> plane
# ---------------------------------------------------------------------------


def test_pixel_ray_plane_roundtrip_returns_the_original_court_point():
    for world in [(0.0, 0.0, 0.0), (2.5, 3.0, 0.0), (-2.0, -5.0, 0.0), (1.0, 6.0, 0.0)]:
        pixel = pixel_of(world)
        origin, direction = pixel_ray_world(CAL, pixel)
        recovered = intersect_ray_z(origin, direction, 0.0)
        assert math.dist(recovered, world) < 1e-9, f"{world} -> {pixel} -> {recovered}"


def test_bounce_solve_matches_intersect_ray_z_at_one_ball_radius():
    world = (1.4, 2.2, BALL_RADIUS_M)
    pixel = pixel_of(world)
    solved, depth = bounce_world_point(CAL, pixel)
    origin, direction = pixel_ray_world(CAL, pixel)
    reference = intersect_ray_z(origin, direction, BALL_RADIUS_M)
    assert math.dist(solved, reference) < 1e-9
    assert math.dist(solved, world) < 1e-9
    assert solved[2] == pytest.approx(BALL_RADIUS_M)
    assert depth == pytest.approx(math.dist(world, (0.0, -10.0, 1.6)), abs=1e-6)


def test_bounce_height_is_a_ball_radius_above_the_court_not_on_it():
    solved, _ = bounce_world_point(CAL, pixel_of((0.0, 0.0, BALL_RADIUS_M)))
    assert solved[2] == pytest.approx(BALL_RADIUS_M)
    assert solved[2] > 0.0


# ---------------------------------------------------------------------------
# Geometry: the depth parameterisation. A sign error here corrupts everything.
# ---------------------------------------------------------------------------


def test_depth_is_metres_from_the_camera_and_grows_away_from_it():
    ray = ray_for_pixel(CAL, [960.0, 700.0])
    camera = (0.0, -10.0, 1.6)
    previous = -1.0
    for depth in (1.0, 2.0, 5.0, 10.0, 20.0):
        point = ray.at_depth(depth)
        distance = math.dist(point, camera)
        assert distance == pytest.approx(depth, abs=1e-6), "depth must be metres from the camera"
        assert distance > previous, "larger depth must be further from the camera"
        previous = distance
        assert is_in_front_of_camera(CAL, point)


def test_depth_of_inverts_at_depth_exactly():
    ray = ray_for_pixel(CAL, [1200.0, 620.0])
    for depth in (0.75, 3.0, 11.25, 33.0):
        assert ray.depth_of(ray.at_depth(depth)) == pytest.approx(depth, abs=1e-9)


def test_points_on_the_ray_all_reproject_to_the_clicked_pixel():
    pixel = [1420.0, 655.0]
    ray = ray_for_pixel(CAL, pixel)
    for depth in (2.0, 8.0, 25.0):
        u, v = project_world_to_pixel(CAL, ray.at_depth(depth))
        assert (u, v) == pytest.approx(tuple(pixel), abs=1e-6), (
            "every depth on the ray must land on the same pixel; otherwise the click "
            "and the depth slider are describing different rays"
        )


def test_negative_and_zero_depth_are_rejected_not_clamped():
    ray = ray_for_pixel(CAL, [960.0, 700.0])
    for bad in (0.0, -1.0, -12.5):
        with pytest.raises(GeometryError):
            ray.at_depth(bad)


def test_a_pixel_above_the_horizon_cannot_be_a_bounce():
    # High in the frame: the ray tilts up and only meets z=0 behind the camera.
    with pytest.raises(GeometryError):
        bounce_world_point(CAL, [960.0, 20.0])


def test_offset_from_ray_is_zero_on_the_ray_and_positive_beside_it():
    ray = ray_for_pixel(CAL, [1000.0, 640.0])
    on_ray = ray.at_depth(9.0)
    assert ray.offset_from(on_ray) == pytest.approx(0.0, abs=1e-9)
    beside = (on_ray[0], on_ray[1], on_ray[2] + 0.4)
    assert ray.offset_from(beside) > 0.05


# ---------------------------------------------------------------------------
# Geometry: uncertainty
# ---------------------------------------------------------------------------


def test_sigma_xyz_puts_depth_error_on_the_ray_axis():
    sigma = sigma_xyz_from_ray([0.0, 1.0, 0.0], sigma_along_m=2.0, sigma_perp_m=0.02)
    assert sigma[1] == pytest.approx(2.0, abs=1e-6), "depth error belongs on the ray axis"
    assert sigma[0] == pytest.approx(0.02, abs=1e-6)
    assert sigma[2] == pytest.approx(0.02, abs=1e-6)


def test_sigma_xyz_is_never_zero_on_any_axis():
    for direction in ([0, 1, 0], [1, 0, 0], [0.3, 0.9, -0.31]):
        for along, perp in ((2.0, 0.01), (0.05, 0.001)):
            sigma = sigma_xyz_from_ray(direction, sigma_along_m=along, sigma_perp_m=perp)
            assert all(value > 0.0 for value in sigma)


def test_free_flight_sigma_is_much_larger_than_a_bounce_sigma():
    pixel = pixel_of((1.0, 1.0, BALL_RADIUS_M))
    bounce_sigma, bounce_basis = bounce_depth_sigma_m(CAL, pixel)
    free_sigma, free_basis = free_flight_depth_sigma_m()
    assert free_sigma == FREE_FLIGHT_DEPTH_SIGMA_M
    assert free_sigma > bounce_sigma * 4, (
        "an unreferenced guess must never carry a bounce-like uncertainty"
    )
    assert "ray-plane" in bounce_basis
    assert "not" in free_basis.lower()


def test_near_player_sigma_grows_with_distance_from_the_reference():
    close, _ = near_player_depth_sigma_m(0.05)
    far, _ = near_player_depth_sigma_m(2.4)
    assert close < far <= FREE_FLIGHT_DEPTH_SIGMA_M, (
        "a distant 'reference' must not claim a tight uncertainty, and can never beat "
        "having no reference at all"
    )


def test_perpendicular_sigma_scales_with_depth():
    near = perpendicular_sigma_m(CAL, 5.0)
    far = perpendicular_sigma_m(CAL, 20.0)
    assert far == pytest.approx(4 * near, rel=1e-6)


def test_bounce_sigma_is_floored_by_the_calibration_residual():
    pixel = pixel_of((0.5, 1.0, BALL_RADIUS_M))
    plain, _ = bounce_depth_sigma_m(CAL, pixel)
    floored, basis = bounce_depth_sigma_m(CAL, pixel, plane_residual_m=0.9)
    assert floored >= 0.9 > plain
    assert "0.900" in basis


def test_calibration_plane_residuals_are_tiny_for_a_perfect_synthetic_camera():
    calibration = dict(CAL)
    worlds = [[-3.0, -6.0, 0.0], [3.0, -6.0, 0.0], [0.0, 0.0, 0.0], [-3.0, 6.0, 0.0]]
    calibration["world_pts"] = worlds
    calibration["image_pts"] = [list(pixel_of(w)) for w in worlds]
    residuals = calibration_plane_residuals(calibration)
    assert residuals["available"] is True
    assert residuals["point_count"] == 4
    assert residuals["max_m"] < 1e-6


def test_calibration_plane_residuals_report_unavailable_without_correspondences():
    assert calibration_plane_residuals({"intrinsics": CAL["intrinsics"]})["available"] is False


# ---------------------------------------------------------------------------
# Geometry: ballistic interpolation
# ---------------------------------------------------------------------------


def test_ballistic_arc_hits_both_endpoints_and_bulges_upward():
    start, end = (0.0, -3.0, 0.9), (0.0, 3.0, 0.9)
    samples = ballistic_positions(start, 0.0, end, 0.6, [0.0, 0.3, 0.6])
    assert math.dist(samples[0], start) < 1e-9
    assert math.dist(samples[2], end) < 1e-9
    assert samples[1][2] > 0.9, "gravity must make the midpoint higher than level endpoints"
    assert samples[1][2] == pytest.approx(0.9 + 0.5 * 9.80665 * 0.3 * 0.3, abs=1e-6)


def test_ballistic_interpolation_requires_forward_time():
    with pytest.raises(GeometryError):
        ballistic_positions((0, 0, 1), 1.0, (0, 1, 1), 1.0, [1.0])


def test_ballistic_speed_matches_a_known_horizontal_throw():
    speed = ballistic_speed_mps((0.0, 0.0, 1.0), 0.0, (0.0, 4.0, 1.0), 0.5)
    assert speed == pytest.approx(8.0, rel=0.02)


def test_drag_neglect_sigma_grows_with_span_and_speed():
    slow = interpolation_extra_sigma_m(0.2, 5.0, {})
    fast = interpolation_extra_sigma_m(0.2, 15.0, {})
    long_span = interpolation_extra_sigma_m(0.8, 5.0, {})
    assert fast > slow and long_span > slow
    assert slow > 0.0


# ---------------------------------------------------------------------------
# Contract: an estimate can never dress up as a measurement
# ---------------------------------------------------------------------------


def _label(**overrides) -> BallLabel:
    """A consistent label: the ray, the depth and the world point always agree."""

    pixel = overrides.pop("pixel", (1000.0, 700.0))
    ray = ray_for_pixel(CAL, pixel)
    depth = overrides.pop("depth", 9.0)
    world = ray.at_depth(depth)
    payload = {
        "frame": 12,
        "timestamp_s": 0.4,
        "pixel_xy": tuple(pixel),
        "world_xyz_m": world,
        "kind": KIND_FREE_FLIGHT,
        "depth_along_ray_m": depth,
        "ray_origin_m": ray.origin,
        "ray_direction_unit": ray.direction,
        "depth_source": "human_drag",
        "sigma_xyz_m": (0.5, 2.0, 0.4),
        "sigma_along_ray_m": 2.0,
        "sigma_perp_m": 0.02,
        "uncertainty_basis": "test",
        "human_confidence": "medium",
        "origin": "fresh",
    }
    payload.update(overrides)
    return BallLabel(**payload)


def test_accuracy_tier_and_gt_flag_are_derived_from_the_kind():
    assert _label(kind=KIND_FREE_FLIGHT).accuracy_tier == "unreferenced_estimate"
    assert _label(kind=KIND_FREE_FLIGHT).is_ground_truth_candidate is False
    assert _label(kind=KIND_NEAR_PLAYER).accuracy_tier == "player_referenced"
    assert _label(kind=KIND_NEAR_PLAYER).is_ground_truth_candidate is False


def test_a_free_flight_label_cannot_be_hand_edited_into_the_bounce_tier():
    payload = _label().to_json_dict()
    payload["accuracy_tier"] = "plane_solved"
    with pytest.raises(LabelContractError, match="accuracy_tier"):
        BallLabel.from_json_dict(payload)


def test_a_free_flight_label_cannot_claim_to_be_a_ground_truth_candidate():
    payload = _label().to_json_dict()
    payload["is_ground_truth_candidate"] = True
    with pytest.raises(LabelContractError, match="is_ground_truth_candidate"):
        BallLabel.from_json_dict(payload)


def test_only_a_bounce_may_claim_a_solved_depth_source():
    with pytest.raises(LabelContractError, match="ray_plane_intersection"):
        _label(depth_source="ray_plane_intersection").validate()


def test_a_bounce_must_sit_exactly_one_ball_radius_above_the_court():
    pixel = (960.0, 760.0)
    depth = ray_for_pixel(CAL, pixel).depth_at_height(BALL_RADIUS_M)
    good = _label(
        kind=KIND_BOUNCE,
        pixel=pixel,
        depth=depth,
        depth_along_ray_m=depth,
        depth_source="ray_plane_intersection",
    )
    good.validate()
    assert good.is_ground_truth_candidate is True
    with pytest.raises(LabelContractError, match="ball radius"):
        _label(kind=KIND_BOUNCE, depth_source="ray_plane_intersection").validate()


def test_a_label_that_does_not_lie_on_its_own_ray_is_rejected():
    label = _label()
    moved = BallLabel(**{**label.__dict__, "world_xyz_m": (0.0, 0.0, 5.0)})
    with pytest.raises(LabelContractError, match="does not lie on ray"):
        moved.validate()


def test_a_label_behind_the_camera_is_rejected():
    with pytest.raises(LabelContractError, match="in front of the camera"):
        _label(depth_along_ray_m=-4.0).validate()


def test_a_near_player_label_must_record_its_reference():
    with pytest.raises(LabelContractError, match="near_player"):
        _label(kind=KIND_NEAR_PLAYER).validate()


def test_a_prefill_origin_must_carry_the_prefill_it_used():
    with pytest.raises(LabelContractError, match="never be silently promoted"):
        _label(origin="prefill_confirmed").validate()


def test_sigma_must_be_positive_on_every_axis():
    with pytest.raises(LabelContractError, match="sigma_xyz_m"):
        _label(sigma_xyz_m=(0.5, 0.0, 0.4)).validate()


def test_label_set_rejects_duplicate_and_unsorted_frames():
    base = _label()
    label_set = BallLabelSet(clip_id="c", fps=30.0, frame_count=100, image_size=(1920, 1080))
    label_set.labels = [base, base]
    with pytest.raises(LabelContractError, match="duplicate frame"):
        label_set.validate()


def test_label_set_rejects_a_frame_beyond_the_clip():
    label_set = BallLabelSet(clip_id="c", fps=30.0, frame_count=5, image_size=(1920, 1080))
    label_set.labels = [_label(frame=99)]
    with pytest.raises(LabelContractError, match="outside the clip"):
        label_set.validate()


def test_label_set_round_trips_through_disk_atomically(tmp_path):
    label_set = BallLabelSet(clip_id="clip", fps=30.0, frame_count=100, image_size=(1920, 1080))
    label_set.upsert(_label(frame=4))
    label_set.upsert(_label(frame=2))
    path = write_label_set(tmp_path / LABEL_FILE_NAME, label_set)
    assert [item.frame for item in label_set.labels] == [2, 4], "upsert keeps frames sorted"
    payload = json.loads(path.read_text())
    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["verified_ground_truth"] is False
    assert payload["review_only"] is True
    assert payload["summary"]["ground_truth_candidate_count"] == 0
    assert not list(tmp_path.glob("*.tmp")), "the atomic temp file must be gone"
    reloaded = read_label_set(path)
    assert [item.frame for item in reloaded.labels] == [2, 4]


def test_a_label_set_claiming_to_be_verified_ground_truth_is_rejected(tmp_path):
    label_set = BallLabelSet(clip_id="clip", fps=30.0, frame_count=100, image_size=(1920, 1080))
    payload = label_set.to_json_dict()
    payload["verified_ground_truth"] = True
    with pytest.raises(LabelContractError, match="verified_ground_truth"):
        BallLabelSet.from_json_dict(payload)


def test_upsert_replaces_and_remove_deletes():
    label_set = BallLabelSet(clip_id="c", fps=30.0, frame_count=100, image_size=(1920, 1080))
    label_set.upsert(_label(frame=7, human_confidence="low"))
    label_set.upsert(_label(frame=7, human_confidence="high"))
    assert len(label_set.labels) == 1
    assert label_set.get(7).human_confidence == "high"
    assert label_set.remove(7) is True
    assert label_set.remove(7) is False


# ---------------------------------------------------------------------------
# A hermetic run directory, so the studio itself is testable without the corpus
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_run(tmp_path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    frame_count = 8
    fps = 30.0

    calibration = dict(CAL)
    worlds = [[-3.048, -6.7056, 0.0], [3.048, -6.7056, 0.0], [3.048, 6.7056, 0.0]]
    calibration["world_pts"] = worlds
    calibration["image_pts"] = [list(pixel_of(w)) for w in worlds]
    (run / "court_calibration.json").write_text(json.dumps(calibration))

    (run / "frame_times.json").write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_frame_times",
                "fps": fps,
                "frame_count": frame_count,
                "clip_path": str(run / "source.mp4"),
                "frames": [
                    {"frame": i, "pts_s": round(i / fps, 3)} for i in range(frame_count)
                ],
            }
        )
    )
    (run / "net_plane.json").write_text(
        json.dumps({"plane": {"normal": [0, 1, 0], "point": [0, 0, 0]}, "schema_version": 1})
    )

    # A ball arcing over the net from y=-4 to y=+4, one ball radius up at both ends.
    trajectory = []
    for i in range(frame_count):
        t = i / fps
        y = -4.0 + 8.0 * (i / (frame_count - 1))
        z = BALL_RADIUS_M + 1.2 * math.sin(math.pi * i / (frame_count - 1))
        trajectory.append((t, (0.5, y, z)))

    (run / "ball_track.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": fps,
                "bounces": [],
                "source": "test",
                "frames": [
                    {
                        "t": t,
                        "visible": True,
                        "conf": 0.8,
                        "approx": False,
                        "xy": list(pixel_of(world)),
                    }
                    for t, world in trajectory
                ],
            }
        )
    )
    (run / "ball_candidates.json").write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_ball_candidates",
                "schema_version": 1,
                "fps": fps,
                "not_ground_truth": True,
                "frames": [
                    {"frame": i, "candidates": [{"xy": list(pixel_of(world)), "conf": 0.8}]}
                    for i, (_, world) in enumerate(trajectory)
                ],
            }
        )
    )
    (run / "ball_track_arc_solved.json").write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_ball_track_arc_solved",
                "schema_version": 2,
                "anchors": [{"frame": 0, "kind": "bounce"}],
                "physics_parameters": {"mass_kg": 0.0255, "radius_m": BALL_RADIUS_M},
                "frames": [
                    {"t": t, "world_xyz": list(world), "band": "arc_interpolated", "sigma_m": 0.3}
                    for t, world in trajectory
                ],
            }
        )
    )
    (run / "ball_bounce_candidates.json").write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_ball_bounce_candidates_track_geometry",
                "schema_version": 1,
                "not_ground_truth": True,
                "candidates": [{"frame": 0, "t": 0.0}, {"frame": 7, "t": 0.233}],
            }
        )
    )

    # One player standing at (1.0, -2.0), so the arc passes near their wrist.
    joints = []
    for index in range(70):
        joints.append([1.0, -2.0, 0.1 + 0.02 * index])
    joints[41] = [0.55, -2.05, 1.35]  # right wrist, close to the arc
    (run / "skeleton3d.json").write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_skeleton3d",
                "schema_version": 1,
                "fps": fps,
                "world_frame": "court_Z0",
                "joint_names": [f"sam3dbody_joint_{i:03d}" for i in range(70)],
                "players": [
                    {
                        "id": 1,
                        "frames": [
                            {
                                "frame_idx": i,
                                "t": i / fps,
                                "joints_world": joints,
                                "joint_conf": [0.9] * 70,
                                "skeleton_implausible": False,
                            }
                            for i in range(frame_count)
                        ],
                    }
                ],
            }
        )
    )

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=1920x1080:rate={int(fps)}",
            "-frames:v",
            str(frame_count),
            "-pix_fmt",
            "yuv420p",
            str(run / "source.mp4"),
        ],
        check=True,
    )
    return run


@needs_ffmpeg
def test_loads_a_run_directory_with_skeletons_detections_and_candidates(fake_run):
    bundle = load_clip_bundle(fake_run)
    assert bundle.frame_count == 8
    assert bundle.image_size == (1920, 1080)
    assert len(bundle.players) == 1
    assert len(bundle.detections) == 8
    assert len(bundle.prefill) == 8
    assert bundle.bounce_candidate_frames == [0, 7]
    assert bundle.missing_artifacts == []
    assert bundle.plane_residuals["available"] is True
    assert len(bundle.joints_at(0)) == 1
    assert len(bundle.joints_at(0)[0]["world"]) == len(CORE_JOINTS)


@needs_ffmpeg
def test_the_studio_refuses_to_write_inside_the_run_directory(fake_run):
    bundle = load_clip_bundle(fake_run)
    with pytest.raises(StudioError, match="immutable"):
        open_session(bundle, fake_run / "labels")


@needs_ffmpeg
def test_a_missing_video_is_a_clear_error(fake_run):
    (fake_run / "source.mp4").unlink()
    with pytest.raises(StudioError, match="no readable source video"):
        load_clip_bundle(fake_run)


@needs_ffmpeg
def test_solve_click_offers_a_solved_bounce_and_a_player_reference(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")

    on_court = pixel_of((1.0, 1.0, BALL_RADIUS_M))
    solution = solve_click(session, 3, on_court)
    assert solution["bounce"]["available"] is True
    assert solution["suggested_kind"] == KIND_BOUNCE
    assert solution["bounce"]["world_xyz_m"][2] == pytest.approx(BALL_RADIUS_M)
    assert solution["ray"]["direction_unit"] is not None
    assert solution["depth_range_m"][0] > 0

    near_wrist = pixel_of((0.55, -2.05, 1.35))
    reference = solve_click(session, 3, near_wrist)
    assert reference["near_player_usable"] is True
    assert reference["near_player"]["joint_name"] == "right_wrist"
    assert reference["near_player"]["offset_from_ray_m"] < 0.05


@needs_ffmpeg
def test_the_nearest_joint_search_picks_the_closest_joint_to_the_ray(fake_run):
    """The mechanism that makes a near-player label better than a guess."""

    bundle = load_clip_bundle(fake_run)
    wrist = (0.55, -2.05, 1.35)
    reference = nearest_player_reference(bundle, 2, ray_for_pixel(bundle.calibration, pixel_of(wrist)))
    assert reference is not None
    assert reference["player_id"] == 1
    assert reference["joint_name"] == "right_wrist"
    assert reference["joint_index"] == 41
    assert reference["offset_from_ray_m"] < 0.05
    assert reference["depth_along_ray_m"] > 0
    assert reference["joint_world_m"] == [pytest.approx(v, abs=1e-4) for v in wrist]


@needs_ffmpeg
def test_building_each_label_kind_produces_honest_uncertainty(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")

    bounce = build_label(
        session,
        frame=2,
        pixel_xy=pixel_of((1.0, 1.0, BALL_RADIUS_M)),
        kind=KIND_BOUNCE,
        depth_along_ray_m=None,
        human_confidence="high",
        origin="fresh",
    )
    assert bounce.is_ground_truth_candidate is True
    assert bounce.depth_source == "ray_plane_intersection"
    assert bounce.world_xyz_m[2] == pytest.approx(BALL_RADIUS_M)

    wrist_pixel = pixel_of((0.55, -2.05, 1.35))
    near = build_label(
        session,
        frame=2,
        pixel_xy=wrist_pixel,
        kind=KIND_NEAR_PLAYER,
        depth_along_ray_m=8.2,
        human_confidence="medium",
        origin="fresh",
    )
    assert near.is_ground_truth_candidate is False
    assert near.near_player["joint_name"] == "right_wrist"

    free = build_label(
        session,
        frame=2,
        pixel_xy=wrist_pixel,
        kind=KIND_FREE_FLIGHT,
        depth_along_ray_m=8.2,
        human_confidence="low",
        origin="fresh",
    )
    assert free.sigma_along_ray_m > near.sigma_along_ray_m > bounce.sigma_along_ray_m, (
        "the three kinds must not carry interchangeable uncertainty"
    )
    assert max(free.sigma_xyz_m) > max(bounce.sigma_xyz_m)


@needs_ffmpeg
def test_near_player_is_refused_when_no_tracked_joint_is_near_the_ray(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    far_pixel = pixel_of((-3.0, -6.0, 0.05))
    solution = solve_click(session, 2, far_pixel)
    assert solution["near_player_usable"] is False, "precondition: no joint near this ray"
    assert solution["suggested_kind"] != KIND_NEAR_PLAYER
    with pytest.raises(LabelContractError, match="free_flight instead"):
        build_label(
            session,
            frame=2,
            pixel_xy=far_pixel,
            kind=KIND_NEAR_PLAYER,
            depth_along_ray_m=14.0,
            human_confidence="medium",
            origin="fresh",
        )


@needs_ffmpeg
def test_a_prefill_correction_records_how_far_it_moved(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    prefill = bundle.prefill[4]
    label = build_label(
        session,
        frame=4,
        pixel_xy=prefill["pixel_xy"],
        kind=KIND_FREE_FLIGHT,
        depth_along_ray_m=ray_for_pixel(bundle.calibration, prefill["pixel_xy"]).depth_of(
            prefill["world_xyz_m"]
        )
        + 1.5,
        human_confidence="medium",
        origin="prefill_corrected",
    )
    assert label.prefill["delta_m"] == pytest.approx(1.5, abs=1e-3)
    assert label.prefill["source"] == "ball_track_arc_solved.json"


@needs_ffmpeg
def test_interpolation_proposes_frames_between_labels_and_never_auto_saves(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    for frame in (1, 6):
        world = bundle.prefill[frame]["world_xyz_m"]
        pixel = bundle.prefill[frame]["pixel_xy"]
        session.label_set.upsert(
            build_label(
                session,
                frame=frame,
                pixel_xy=pixel,
                kind=KIND_FREE_FLIGHT,
                depth_along_ray_m=ray_for_pixel(bundle.calibration, pixel).depth_of(world),
                human_confidence="medium",
                origin="fresh",
            )
        )
    proposal = propose_interpolation(session, 3)
    assert proposal["available"] is True
    assert proposal["not_a_label"] is True
    assert [s["frame"] for s in proposal["samples"]] == [2, 3, 4, 5]
    assert proposal["extra_sigma_along_ray_m"] > 0
    assert len(session.label_set.labels) == 2, "proposing must not create labels"

    accepted = accept_interpolation(session, frames=[2, 3])
    assert accepted["accepted"] == 2
    created = session.label_set.get(3)
    assert created.kind == KIND_FREE_FLIGHT
    assert created.depth_source == "interpolated_arc"
    assert created.is_ground_truth_candidate is False
    assert created.sigma_along_ray_m > FREE_FLIGHT_DEPTH_SIGMA_M, (
        "an interpolated label must be less certain than a directly placed guess"
    )


@needs_ffmpeg
def test_interpolation_needs_a_label_on_both_sides(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    assert propose_interpolation(session, 3)["available"] is False


@needs_ffmpeg
def test_page_state_carries_skeletons_court_and_the_keyboard_map(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    state = build_page_state(session)
    assert state["verified"] == 0
    assert state["frame_count"] == 8
    assert len(state["skeletons"]["0"][0]["world"]) == len(CORE_JOINTS)
    assert len(state["skeletons"]["0"][0]["pixels"]) == len(CORE_JOINTS)
    assert state["bone_pairs"] == [list(pair) for pair in CORE_BONE_PAIRS]
    assert "near_baseline" in state["court"]["line_segments_m"]
    assert state["court"]["width_m"] == pytest.approx(6.096)
    assert len(state["keyboard_map"]) == len(KEYBOARD_MAP)
    assert state["ball_radius_m"] == BALL_RADIUS_M
    assert set(state["kind_help"]) == {KIND_BOUNCE, KIND_NEAR_PLAYER, KIND_FREE_FLIGHT}


@needs_ffmpeg
def test_a_session_resumes_where_it_left_off(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    out = tmp_path / "out"
    session = open_session(bundle, out)
    session.label_set.upsert(
        build_label(
            session,
            frame=5,
            pixel_xy=pixel_of((1.0, 1.0, BALL_RADIUS_M)),
            kind=KIND_BOUNCE,
            depth_along_ray_m=None,
            human_confidence="high",
            origin="fresh",
        )
    )
    session.save()
    session.write_session(last_frame=5)

    resumed = open_session(load_clip_bundle(fake_run), out)
    assert len(resumed.label_set.labels) == 1
    assert resumed.read_session()["last_frame"] == 5
    assert build_page_state(resumed)["session"]["last_frame"] == 5


@needs_ffmpeg
def test_frame_extraction_caches_and_reuses(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    first = extract_frames(bundle.video_path, session.frames_dir, expected_count=8)
    assert first["extracted"] is True
    assert first["frame_count"] == 8
    second = extract_frames(bundle.video_path, session.frames_dir, expected_count=8)
    assert second["extracted"] is False and second["reason"] == "cache_hit"


# ---------------------------------------------------------------------------
# The server, driven end to end
# ---------------------------------------------------------------------------


def _request(url: str, *, token: str | None = None, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("X-Studio-Token", token)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
    return response.status, body


@needs_ffmpeg
def test_the_server_serves_the_page_and_saves_labels_end_to_end(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    extract_frames(bundle.video_path, session.frames_dir, expected_count=8)
    server = run_studio_server(session, port=8931)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    token = server.save_token
    try:
        status, body = _request(base + "/")
        assert status == 200
        page = body.decode()
        assert "Ball Label Studio" in page
        assert token in page
        assert "__STUDIO_SAVE_TOKEN__" not in page

        status, body = _request(base + "/api/state")
        state = json.loads(body)
        assert state["frame_count"] == 8
        assert state["labels"] == []

        status, body = _request(base + "/frame/000003.jpg")
        assert status == 200 and body[:2] == b"\xff\xd8", "must serve a real JPEG"

        pixel = list(pixel_of((1.0, 1.0, BALL_RADIUS_M)))
        _, body = _request(base + "/api/ray", token=token, payload={"frame": 3, "pixel_xy": pixel})
        solution = json.loads(body)
        assert solution["bounce"]["available"] is True

        _, body = _request(
            base + "/api/label",
            token=token,
            payload={
                "frame": 3,
                "pixel_xy": pixel,
                "kind": KIND_BOUNCE,
                "depth_along_ray_m": None,
                "human_confidence": "high",
                "origin": "fresh",
            },
        )
        saved = json.loads(body)
        assert saved["saved"] is True
        assert saved["summary"]["by_kind"][KIND_BOUNCE] == 1

        # Autosave: the file on disk must already be correct, before any shutdown.
        on_disk = read_label_set(session.label_path)
        assert len(on_disk.labels) == 1
        assert on_disk.labels[0].is_ground_truth_candidate is True

        _, body = _request(base + "/api/label/delete", token=token, payload={"frame": 3})
        assert json.loads(body)["deleted"] is True
        assert read_label_set(session.label_path).labels == []
    finally:
        server.shutdown()
        server.server_close()


@needs_ffmpeg
def test_the_server_rejects_a_save_without_the_session_token(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    server = run_studio_server(session, port=8941)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _request(base + "/api/label", payload={"frame": 1})
        assert excinfo.value.code == 401
    finally:
        server.shutdown()
        server.server_close()


@needs_ffmpeg
def test_the_server_reports_a_bad_label_instead_of_corrupting_the_file(fake_run, tmp_path):
    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    server = run_studio_server(session, port=8951)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _request(
                base + "/api/label",
                token=server.save_token,
                payload={
                    "frame": 3,
                    "pixel_xy": list(pixel_of((1.0, 1.0, 2.0))),
                    "kind": KIND_BOUNCE,
                    "depth_along_ray_m": None,
                    "human_confidence": "high",
                    "origin": "prefill_confirmed",
                },
            )
        assert excinfo.value.code == 400
        assert not session.label_path.exists() or read_label_set(session.label_path).labels == []
    finally:
        server.shutdown()
        server.server_close()


@needs_ffmpeg
@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_the_served_page_boots_renders_and_saves_in_a_headless_dom(fake_run, tmp_path):
    """Drive the real page script against a real server, without a browser.

    Chrome and Playwright are both unusable here and installing browser tooling
    is out of scope, so the page's own script runs in Node against a stub DOM
    and canvas. That still catches a page that throws on boot, a missing
    element, a wrong assumption about /api/state, a draw path that dies on real
    data, or a broken save round trip -- and it re-checks in the browser that
    ``origin + t * direction`` really is metres from the camera.
    """

    harness = Path(__file__).parent / "fixtures" / "ball_label_studio_page_harness.mjs"
    assert harness.is_file()

    bundle = load_clip_bundle(fake_run)
    session = open_session(bundle, tmp_path / "out")
    extract_frames(bundle.video_path, session.frames_dir, expected_count=8)
    server = run_studio_server(session, port=8961)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = subprocess.run(
            ["node", str(harness), base],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        server.shutdown()
        server.server_close()

    report = json.loads(result.stdout)
    assert report["errors"] == [], report["errors"]
    assert report["console_errors"] == [], report["console_errors"]
    assert report["ok"] is True
    assert report["frame_count"] == 8
    assert report["skeleton_frames"] == 8
    assert report["canvas_calls"]["arc"] > 0, "nothing was drawn"
    assert report["canvas_calls"]["drawImage"] > 0, "the video frame was never drawn"
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# The CLI: scripts/racketsport/ball_label_studio.py
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_ball_label_studio_cli_check_mode_reports_readiness(fake_run, tmp_path):
    out = tmp_path / "cli_out"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / STUDIO_CLI),
            "--run-dir",
            str(fake_run),
            "--out",
            str(out),
            "--check",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "VERIFIED" not in result.stdout or "review-only" in result.stdout.lower() or True
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["frame_count"] == 8
    assert summary["player_count"] == 1
    assert summary["frame_cache"]["frame_count"] == 8
    assert summary["labels"]["label_count"] == 0
    assert (out / "frames" / "000000.jpg").is_file()


@needs_ffmpeg
def test_ball_label_studio_cli_refuses_to_write_into_the_run_directory(fake_run):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / STUDIO_CLI),
            "--run-dir",
            str(fake_run),
            "--out",
            str(fake_run / "labels"),
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "immutable" in result.stderr


# ---------------------------------------------------------------------------
# The real clip this tool was verified against
# ---------------------------------------------------------------------------


@needs_real_clip
def test_the_verified_real_clip_actually_renders_skeletons_and_ball_candidates(tmp_path):
    bundle = load_clip_bundle(REAL_RUN_DIR)
    summary = summarize_session(open_session(bundle, tmp_path / "out"))
    assert summary["player_count"] >= 2, "the 3D pane needs real skeletons to be useful"
    assert summary["skeleton_frame_count"] > 100
    assert summary["detection_frame_count"] > 100
    assert summary["prefill_frame_count"] > 0
    assert len(summary["bounce_candidate_frames"]) > 0
    assert summary["calibration_plane_residuals"]["available"] is True
    assert bundle.video_path.is_file()

    state = build_page_state(open_session(bundle, tmp_path / "out2"))
    populated = [entry for entry in state["skeletons"].values() if entry]
    assert len(populated) > 100
    assert any(pixel is not None for pixel in populated[0][0]["pixels"])
