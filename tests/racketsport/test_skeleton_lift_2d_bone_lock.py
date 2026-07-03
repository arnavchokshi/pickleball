"""LIFT-2D-BONE-LOCK: kinematic-chain depth solve from the hip root outward.

Covers the specific failure mode the bone-calibration measurement traced
(runs/bone_calib_20260703T0102Z/REPORT.md): hip_width/torso_length p90 error
300-800% because each joint's depth was solved independently along its own
camera ray. These tests exercise the shared-depth lateral-pair solve
(`_lateral_pair_shared_depth_solution`), the outward chain
(`_solve_body_core_chain`), the HARD-vs-self-measured leg length resolution
(`_resolve_player_body_scale` / `_self_measure_leg_lengths`), and the
bone_calib loader (`load_player_bone_lengths`).
"""

from __future__ import annotations

import math

import pytest

from threed.racketsport.skeleton_lift_2d import (
    BODY_CORE_REQUIRED_NAMES,
    Lift2DConfig,
    _BodyScale,
    _Ray,
    _solve_body_core_chain,
    lift_skeleton_from_2d,
    load_player_bone_lengths,
)


FX = FY = 800.0
CX, CY = 320.0, 240.0

# Target hard/anthropometric bone lengths (meters) for the full-body fixture.
THIGH = 0.40
SHIN = 0.38
HIP_WIDTH = 0.18
TORSO_LENGTH = 0.50
SHOULDER_WIDTH = 0.36
UPPER_ARM = 0.28
LOWER_ARM = 0.24
LEG_SCALE = THIGH + SHIN

JOINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "dist": []},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "homography": [[80.0, 0.0, 320.0], [0.0, 80.0, 240.0], [0.0, 0.0, 1.0]],
        "image_size": [640, 480],
    }


def _project(point: tuple[float, float, float]) -> list[float]:
    x, y, z = point
    depth = z + 10.0
    return [FX * x / depth + CX, FY * y / depth + CY]


def _unit_scaled(parent: tuple[float, float, float], guess: tuple[float, float, float], length: float) -> tuple[float, float, float]:
    delta = tuple(guess[i] - parent[i] for i in range(3))
    norm = math.sqrt(sum(c * c for c in delta))
    return tuple(parent[i] + (delta[i] / norm) * length for i in range(3))


def _body_points(*, x_shift: float = 0.0) -> dict[str, tuple[float, float, float]]:
    """A geometrically self-consistent standing pose honoring the module constants.

    Height varies along world Y (image-vertical) and lateral offset along
    world X, matching a realistic mostly-frontal court camera; world Z
    ("depth" in this fixture's simplified R=identity camera, matching the
    convention already used by test_skeleton_lift_2d.py) only varies
    slightly, the way a real standing player's depth extent is small
    relative to their height. Both hips share one Z exactly, so hip_width is
    exactly recoverable by the shared-depth linear solve. `x_shift`
    translates the whole pose laterally (used to build a short multi-frame
    sequence for the self-measurement bootstrap test).
    """

    left_ankle = (-0.15 + x_shift, 0.0, 0.0)
    right_ankle = (0.15 + x_shift, 0.0, 0.0)
    knee_z = 0.02
    knee_y = math.sqrt(SHIN**2 - knee_z**2)  # exact shin length despite the small depth offset
    left_knee = (-0.15 + x_shift, knee_y, knee_z)
    right_knee = (0.15 + x_shift, knee_y, knee_z)

    def _hip_y(knee: tuple[float, float, float], hip_x: float, hip_z: float) -> float:
        dx = hip_x - knee[0]
        dz = hip_z - knee[2]
        return knee[1] + math.sqrt(max(THIGH**2 - dx * dx - dz * dz, 0.0))

    hip_z = 0.05
    left_hip = (-HIP_WIDTH / 2 + x_shift, _hip_y(left_knee, -HIP_WIDTH / 2 + x_shift, hip_z), hip_z)
    right_hip = (HIP_WIDTH / 2 + x_shift, _hip_y(right_knee, HIP_WIDTH / 2 + x_shift, hip_z), hip_z)
    pelvis = tuple((left_hip[i] + right_hip[i]) / 2 for i in range(3))
    shoulder_z = 0.03
    shoulder_y = pelvis[1] + math.sqrt(max(TORSO_LENGTH**2 - (shoulder_z - pelvis[2]) ** 2, 0.0))
    left_shoulder = (-SHOULDER_WIDTH / 2 + x_shift, shoulder_y, shoulder_z)
    right_shoulder = (SHOULDER_WIDTH / 2 + x_shift, shoulder_y, shoulder_z)
    left_elbow = _unit_scaled(
        left_shoulder, (left_shoulder[0] - 0.05, left_shoulder[1] - UPPER_ARM, left_shoulder[2] + 0.02), UPPER_ARM
    )
    right_elbow = _unit_scaled(
        right_shoulder, (right_shoulder[0] + 0.05, right_shoulder[1] - UPPER_ARM, right_shoulder[2] + 0.02), UPPER_ARM
    )
    left_wrist = _unit_scaled(
        left_elbow, (left_elbow[0] - 0.05, left_elbow[1] - LOWER_ARM, left_elbow[2] + 0.02), LOWER_ARM
    )
    right_wrist = _unit_scaled(
        right_elbow, (right_elbow[0] + 0.05, right_elbow[1] - LOWER_ARM, right_elbow[2] + 0.02), LOWER_ARM
    )
    nose = (0.0 + x_shift, shoulder_y + 0.25, shoulder_z)
    return {
        "nose": nose,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
    }


def _keypoints_payload(frames_points: list[dict[str, tuple[float, float, float]]], *, conf: float = 0.95) -> dict:
    frames = []
    for frame_idx, points in enumerate(frames_points):
        keypoints = [
            {"joint": name, "x_px": _project(points[name])[0], "y_px": _project(points[name])[1], "conf": conf}
            for name in JOINT_NAMES
        ]
        frames.append({"frame_idx": frame_idx, "t": frame_idx / 30.0, "keypoints": keypoints})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_keypoints_2d",
        "fps": 30.0,
        "convention": "synthetic_coco_wholebody",
        "joint_names": JOINT_NAMES,
        "players": [{"id": "p1", "height_m": 1.72, "frames": frames}],
    }


def _tracks_payload(n_frames: int) -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": "p1",
                "frames": [{"frame_idx": idx, "t": idx / 30.0, "world_xy": [0.0, 0.0]} for idx in range(n_frames)],
            }
        ],
    }


def _leg_derived_ratios() -> dict[str, float]:
    return {
        "hip_width": HIP_WIDTH / LEG_SCALE,
        "torso_length": TORSO_LENGTH / LEG_SCALE,
        "shoulder_width": SHOULDER_WIDTH / LEG_SCALE,
        "upper_arm": UPPER_ARM / LEG_SCALE,
        "lower_arm": LOWER_ARM / LEG_SCALE,
    }


def _hard_leg_lengths() -> dict[str, dict[str, float]]:
    return {"p1": {"left_upper_leg": THIGH, "right_upper_leg": THIGH, "left_lower_leg": SHIN, "right_lower_leg": SHIN}}


def test_hip_width_and_torso_length_solved_exactly_with_hard_leg_lengths() -> None:
    """The dominant bone_calib failure: hip_width/torso_length are the two GATED
    bones (p90 error <10% vs canonical). With hard leg lengths + population
    ratios pinned to the leg-derived scale, both are recovered essentially
    exactly (analytic shared-depth solve), unlike the old independent-ray
    approach that produced 300-800% p90 error on these exact two bones.
    """

    points = _body_points()
    skeleton, report = lift_skeleton_from_2d(
        _keypoints_payload([points]),
        tracks_payload=_tracks_payload(1),
        calibration_payload=_calibration(),
        config=Lift2DConfig(
            root_smoothing_radius=0,
            player_bone_lengths=_hard_leg_lengths(),
            leg_derived_ratios=_leg_derived_ratios(),
        ),
    )

    frame0 = skeleton["players"][0]["frames"][0]
    by_name = {name: frame0["joints_world"][idx] for idx, name in enumerate(JOINT_NAMES)}

    hip_width_solved = math.dist(by_name["left_hip"], by_name["right_hip"])
    pelvis_solved = tuple((by_name["left_hip"][i] + by_name["right_hip"][i]) / 2 for i in range(3))
    shoulder_mid_solved = tuple((by_name["left_shoulder"][i] + by_name["right_shoulder"][i]) / 2 for i in range(3))
    torso_solved = math.dist(pelvis_solved, shoulder_mid_solved)

    assert hip_width_solved == pytest.approx(HIP_WIDTH, abs=1e-6)
    assert torso_solved == pytest.approx(TORSO_LENGTH, abs=1e-6)

    # Hard leg lengths are honored (this is the "HARD" requirement, not a ratio).
    assert math.dist(by_name["left_hip"], by_name["left_knee"]) == pytest.approx(THIGH, abs=1e-5)
    assert math.dist(by_name["right_hip"], by_name["right_knee"]) == pytest.approx(THIGH, abs=1e-5)
    assert math.dist(by_name["left_knee"], by_name["left_ankle"]) == pytest.approx(SHIN, abs=1e-5)
    assert math.dist(by_name["right_knee"], by_name["right_ankle"]) == pytest.approx(SHIN, abs=1e-5)

    # Arms honor the leg-derived anthropometric ratio lengths (not the raw
    # unstable measured values -- the report's explicit instruction).
    assert math.dist(by_name["left_shoulder"], by_name["left_elbow"]) == pytest.approx(UPPER_ARM, abs=1e-5)
    assert math.dist(by_name["left_elbow"], by_name["left_wrist"]) == pytest.approx(LOWER_ARM, abs=1e-5)

    body_scale_report = report["players"]["p1"]["body_scale"]
    assert body_scale_report["leg_source"] == "hard_canonical_leg"
    assert body_scale_report["hip_width_m"] == pytest.approx(HIP_WIDTH, abs=1e-4)
    assert body_scale_report["torso_length_m"] == pytest.approx(TORSO_LENGTH, abs=1e-4)


def test_every_lifted_joint_still_reprojects_onto_its_detected_keypoint() -> None:
    """The lane-B invariant (every joint stays ON its own camera ray) must hold
    for the new staged solver exactly as it did for the old generic one.
    """

    points = _body_points()
    payload = _keypoints_payload([points])
    skeleton, _ = lift_skeleton_from_2d(
        payload,
        tracks_payload=_tracks_payload(1),
        calibration_payload=_calibration(),
        config=Lift2DConfig(
            root_smoothing_radius=0,
            player_bone_lengths=_hard_leg_lengths(),
            leg_derived_ratios=_leg_derived_ratios(),
        ),
    )
    frame0 = skeleton["players"][0]["frames"][0]
    source_by_joint = {item["joint"]: item for item in payload["players"][0]["frames"][0]["keypoints"]}
    for joint_name, joint_world in zip(JOINT_NAMES, frame0["joints_world"], strict=True):
        projected = _project(tuple(joint_world))
        detected = source_by_joint[joint_name]
        assert math.hypot(projected[0] - detected["x_px"], projected[1] - detected["y_px"]) < 1e-5


def test_falls_back_to_anthropometric_default_leg_when_no_canonical_supplied() -> None:
    """Outdoor/Indoor/Burlington have no bone_calib canonical file (only
    Wolverine does). Without one, the chain must fall back to the
    pre-existing height-ratio anthropometric default -- honestly labeled as
    a default, not a measurement (a bootstrap "self-measurement" pass would
    be circular: a sphere-ray solve enforces its input length by
    construction, so re-aggregating its own output just restates the input
    ratio; see `_self_measure_leg_lengths`'s docstring). hip_width/torso
    still come out internally consistent (exactly on the population ratio
    times this default leg scale), even though the leg scale itself carries
    whatever bias the default ratio has relative to this player's true legs.
    """

    frames_points = [_body_points(x_shift=0.01 * idx) for idx in range(6)]
    default_height_m = 1.72
    skeleton, report = lift_skeleton_from_2d(
        _keypoints_payload(frames_points),
        tracks_payload=_tracks_payload(len(frames_points)),
        calibration_payload=_calibration(),
        config=Lift2DConfig(
            root_smoothing_radius=0, leg_derived_ratios=_leg_derived_ratios(), default_player_height_m=default_height_m
        ),
    )

    body_scale_report = report["players"]["p1"]["body_scale"]
    assert body_scale_report["leg_source"] == "default_anthropometric_leg"
    assert body_scale_report["left_upper_leg_m"] == pytest.approx(0.245 * default_height_m, abs=1e-4)
    assert body_scale_report["left_lower_leg_m"] == pytest.approx(0.246 * default_height_m, abs=1e-4)

    # hip_width is exactly the population ratio times this (biased) leg
    # scale -- internally consistent even though the leg scale itself is a
    # generic default here, not this specific player's true leg length.
    default_leg_scale = 0.245 * default_height_m + 0.246 * default_height_m
    expected_hip_width = _leg_derived_ratios()["hip_width"] * default_leg_scale
    last_frame = skeleton["players"][0]["frames"][-1]
    by_name = {name: last_frame["joints_world"][idx] for idx, name in enumerate(JOINT_NAMES)}
    hip_width_solved = math.dist(by_name["left_hip"], by_name["right_hip"])
    assert hip_width_solved == pytest.approx(expected_hip_width, abs=1e-6)


def test_body_core_chain_is_a_noop_without_required_joint_names() -> None:
    """Backward compatibility: callers using a non-COCO joint set (e.g. the
    existing generic ankle->pelvis->nose bone_priors tests) must see zero
    behavior change -- the staged solver should not engage at all.
    """

    assert not BODY_CORE_REQUIRED_NAMES.issubset({"left_ankle", "right_ankle", "pelvis", "nose"})
    locked = _solve_body_core_chain(
        rays={},
        solved={},
        solved_depths={},
        previous_depths={},
        root_depth=10.0,
        body_scale=None,
    )
    assert locked == frozenset()


def test_knee_two_solution_ambiguity_prefers_temporal_continuity() -> None:
    """A thigh-length sphere-ray intersection can have two valid depths (the
    knee bone pointed nearly along the viewing ray). Absent history the
    solver takes a joint-limit-flavored default (nearest the parent's own
    depth); given a previous-frame depth it must track that continuation
    instead of jumping to the other root, per the spec's disambiguation rule.
    """

    hip_point = [0.0, 0.0, 5.0]
    hip_depth = 5.0
    knee_ray = _Ray(x_px=0.0, y_px=0.0, center=(0.0, 0.0, 0.0), direction_per_depth=(0.02, 0.0, 1.0))
    body_scale = _BodyScale(
        thigh_l=0.40,
        thigh_r=0.40,
        shin_l=0.38,
        shin_r=0.38,
        leg_scale_m=0.78,
        hip_width=0.18,
        torso_length=0.50,
        shoulder_width=0.36,
        upper_arm=0.28,
        lower_arm=0.24,
        head_offset=0.20,
        leg_source="test",
        ratio_source="test",
    )

    # Establish that this ray really does yield two distinct valid depths for
    # the same thigh length (otherwise this test wouldn't be exercising the
    # ambiguity path at all).
    a = sum(c * c for c in knee_ray.direction_per_depth)
    b = -2 * sum(knee_ray.direction_per_depth[i] * hip_point[i] for i in range(3))
    c = sum(v * v for v in hip_point) - body_scale.thigh_l**2
    discriminant = b * b - 4 * a * c
    assert discriminant > 0
    root = math.sqrt(discriminant)
    near_root, far_root = sorted([(-b - root) / (2 * a), (-b + root) / (2 * a)])
    assert far_root - near_root > 0.5  # genuinely far apart, not a numerical wash

    # No previous depth: joint-limit default (closest to hip's own depth).
    solved_a: dict[str, list[float]] = {"left_hip": list(hip_point)}
    depths_a: dict[str, float] = {"left_hip": hip_depth}
    locked_a = _solve_body_core_chain(
        rays={"left_knee": knee_ray},
        solved=solved_a,
        solved_depths=depths_a,
        previous_depths={},
        root_depth=hip_depth,
        body_scale=body_scale,
    )
    assert "left_knee" in locked_a
    default_depth = depths_a["left_knee"]
    assert default_depth == pytest.approx(min(near_root, far_root, key=lambda d: abs(d - hip_depth)))

    # Previous frame tracked the FAR root: continuity must keep tracking it,
    # even though it's further from the joint-limit default's preference.
    solved_b: dict[str, list[float]] = {"left_hip": list(hip_point)}
    depths_b: dict[str, float] = {"left_hip": hip_depth}
    locked_b = _solve_body_core_chain(
        rays={"left_knee": knee_ray},
        solved=solved_b,
        solved_depths=depths_b,
        previous_depths={"left_knee": far_root},
        root_depth=hip_depth,
        body_scale=body_scale,
    )
    assert "left_knee" in locked_b
    assert depths_b["left_knee"] == pytest.approx(far_root, abs=1e-6)


def test_lateral_pair_hip_width_rejects_degenerate_or_out_of_bounds_geometry() -> None:
    """When the two hip rays are (nearly) parallel -- e.g. hips at the same
    pixel, which can't happen for real detections but guards the formula's
    division -- or the exact-solve depth falls outside a sane physical
    bound, the chain must fall back gracefully (no lock) rather than
    fabricate a wild position.
    """

    from threed.racketsport.skeleton_lift_2d import _lateral_pair_shared_depth_solution

    parallel_ray = _Ray(x_px=0.0, y_px=0.0, center=(0.0, 0.0, 0.0), direction_per_depth=(0.0, 0.0, 1.0))
    point_a, point_b, depth = _lateral_pair_shared_depth_solution(
        parallel_ray, parallel_ray, 0.18, min_depth=0.3, max_depth=50.0
    )
    assert point_a is None and point_b is None and depth is None

    ray_a = _Ray(x_px=0.0, y_px=0.0, center=(0.0, 0.0, 0.0), direction_per_depth=(-1e-6, 0.0, 1.0))
    ray_b = _Ray(x_px=0.0, y_px=0.0, center=(0.0, 0.0, 0.0), direction_per_depth=(1e-6, 0.0, 1.0))
    point_a, point_b, depth = _lateral_pair_shared_depth_solution(ray_a, ray_b, 0.18, min_depth=0.3, max_depth=50.0)
    assert point_a is None and point_b is None and depth is None  # would require an absurd 90000m depth


def test_load_player_bone_lengths_flattens_bone_calib_schema() -> None:
    payload = {
        "artifact_type": "bone_calib_canonical_lengths",
        "players": {
            "1": {
                "n_frames_accurate_tier": 22,
                "bones": {
                    "left_upper_leg": {"median_m": 0.3948, "stable_lt_5pct": True},
                    "right_upper_leg": {"median_m": 0.3948, "stable_lt_5pct": True},
                    "left_lower_leg": {"median_m": 0.3700, "stable_lt_5pct": True},
                    "right_lower_leg": {"median_m": 0.3691, "stable_lt_5pct": True},
                    "hip_width": {"median_m": 0.1622, "stable_lt_5pct": False},
                },
            },
            "2": {"bones": {"left_upper_leg": {"median_m": 0.3823}}},  # missing bones -> excluded
        },
    }
    flattened = load_player_bone_lengths(payload)
    assert set(flattened.keys()) == {"1"}
    assert flattened["1"] == {
        "left_upper_leg": 0.3948,
        "right_upper_leg": 0.3948,
        "left_lower_leg": 0.3700,
        "right_lower_leg": 0.3691,
    }
    # Unstable/pose-dependent bones (hip_width etc.) are deliberately not
    # extracted -- the chain derives them from anthropometric ratios instead.
    assert "hip_width" not in flattened["1"]
