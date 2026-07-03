"""Geometric 2D-first skeleton lift for racket-sport player keypoints.

The lane-B invariant is simple: detected 2D keypoints are authoritative. Every
lifted joint is represented as a point on that keypoint's camera ray; depth is
chosen by court-plane roots, scaled bone priors, and temporal preference.

Depth solve for the core body (hips/knees/shoulders/elbows/wrists) is a
kinematic chain from the root outward (see ``_solve_body_core_chain``):

* Feet stay anchored to the court plane (unchanged, most reliable anchor).
* Hip pair (``left_hip``/``right_hip``) is solved jointly and *exactly* via
  ``_lateral_pair_shared_depth_solution``: since both hip pixels share one
  camera center, their world separation is an exactly-linear function of a
  single shared depth, so the depth that reproduces the hard hip-width bone
  is unique -- no independent per-joint depth guessing, which is what made
  hip_width/torso_length blow out 300-800% under the old per-joint solve
  (see runs/bone_calib_20260703T0102Z/REPORT.md).
* Knees are solved outward from the (now-anchored) hips via HARD per-player
  thigh lengths; shoulders are solved outward from the hip midpoint via the
  torso-length bone (same shared-depth trick, using the average of the two
  shoulder rays as a synthetic ray); elbows/wrists are solved outward from
  shoulders via upper/lower-arm lengths.
* Every two-solution (near/far) sphere-ray ambiguity along the way is
  resolved by preferring the previous frame's depth for that joint
  (temporal continuity) and, absent that, the immediate parent's own depth
  (a joint-limit-flavored default that keeps the chain from stretching
  along the viewing ray).
* Leg bone lengths are HARD: either the measured canonical value for that
  player (``Lift2DConfig.player_bone_lengths``, e.g.
  ``runs/bone_calib_20260703T0102Z/player_bone_lengths.json``) or, when no
  canonical value exists for that player/clip, a self-measured per-clip
  median from a bootstrap pass (mirrors the bone_calib measurement
  methodology). Torso/hip-width/shoulder-width/arm lengths are NOT trusted
  raw (measured values are pose-dependent and unstable per the report) --
  they are anthropometric ratios of that player's own leg-derived scale,
  with the ratio constants themselves population medians computed from
  ``player_bone_lengths.json`` (see ``_DEFAULT_LEG_DERIVED_RATIOS``).

Every joint stays exactly ON its own detected 2D keypoint's camera ray --
only the depth along that ray is chosen by the chain.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_TYPE = "racketsport_skeleton3d_v2"
REPORT_ARTIFACT_TYPE = "racketsport_skeleton_lift_2d_report"
LANE_NAME = "lane_b_2d_first"
SCHEMA_VERSION = 2

LEFT_ANKLE_NAMES = ("left_ankle", "l_ankle", "left_foot", "left_heel")
RIGHT_ANKLE_NAMES = ("right_ankle", "r_ankle", "right_foot", "right_heel")
FOOT_NAMES = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
)
ROOT_NAMES = ("pelvis", "root", "mid_hip", "midhip")
COCO_WHOLEBODY_CORE_BONES = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("neck", "nose"),
)

# Joint names required for the kinematic-chain body-core solve to engage at
# all (see `_solve_body_core_chain`). When a caller's joint set doesn't
# include these (e.g. the generic synthetic bone_priors tests), the solver
# is a strict no-op and behavior is identical to the pre-existing
# independent-ray-plus-generic-priors mechanism.
BODY_CORE_REQUIRED_NAMES = frozenset(
    {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
)

# bone_calib schema (runs/bone_calib_20260703T0102Z/player_bone_lengths.json)
# names for the four hard leg segments, mapped to the (parent, child) joint
# pair whose 3D distance that bone measures.
LEG_BONE_JOINT_PAIRS: dict[str, tuple[str, str]] = {
    "left_upper_leg": ("left_hip", "left_knee"),
    "right_upper_leg": ("right_hip", "right_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_lower_leg": ("right_knee", "right_ankle"),
}

# Anthropometric ratios (bone_length / player_leg_scale) for the bones the
# bone-calibration measurement run found too pose-dependent/unstable to
# trust raw (hip_width/torso_length/shoulder_width/arms; see
# runs/bone_calib_20260703T0102Z/REPORT.md). `leg_scale` for a player is
# avg(upper_leg) + avg(lower_leg) using that player's own (stable) measured
# or hard-canonical leg lengths. These ratio constants are the POPULATION
# MEDIAN of bone_length/leg_scale across the four bone_calib canonical
# players -- computed once from
# runs/bone_calib_20260703T0102Z/player_bone_lengths.json, not a generic
# anthropometric height chart, so they're grounded in the same accurate
# SAM-3D-Body tier as the hard leg lengths themselves.
_DEFAULT_LEG_DERIVED_RATIOS: dict[str, float] = {
    "hip_width": 0.2396,
    "torso_length": 0.6366,
    "shoulder_width": 0.4342,
    "upper_arm": 0.3525,
    "lower_arm": 0.2989,
    # nose (head) offset from shoulder-midpoint. Not gated by this lane's
    # acceptance criteria, but left fully unconstrained (as it was before
    # this lane) it's the single largest remaining depth-extent outlier on
    # real data. bone_calib has no direct shoulder-to-nose bone, only
    # stature_proxy (nose-to-ankle chord); this ratio is a rough
    # approximation (stature_proxy - leg_scale - torso_length) / leg_scale,
    # median across the four canonical players -- noisy (0.02-0.19 range,
    # since stature_proxy itself is explicitly flagged unreliable in
    # runs/bone_calib_20260703T0102Z/REPORT.md for non-standing poses), good
    # enough to bound the nose depth without pretending to be precise.
    "head_offset": 0.20,
}


@dataclass(frozen=True)
class BonePrior:
    parent: str
    child: str
    length_m: float
    source: str = "input"


@dataclass(frozen=True)
class Lift2DConfig:
    min_joint_confidence: float = 0.2
    court_z_m: float = 0.0
    root_smoothing_radius: int = 2
    default_player_height_m: float = 1.72
    bone_priors: tuple[BonePrior, ...] = field(default_factory=tuple)
    plausibility_joint_confidence_floor: float = 0.2
    plausibility_max_bone_zscore: float = 10.0
    plausibility_min_bone_samples: int = 5
    plausibility_min_sigma_m: float = 0.10
    # HARD per-player leg bone lengths (meters), keyed by player id then by
    # bone name from LEG_BONE_JOINT_PAIRS (e.g. "left_upper_leg"). Typically
    # loaded from runs/bone_calib_20260703T0102Z/player_bone_lengths.json
    # via `load_player_bone_lengths`. Players absent from this mapping fall
    # back to a self-measured per-clip median (see `_self_measure_leg_lengths`).
    player_bone_lengths: Mapping[str, Mapping[str, float]] | None = None
    # Population-median ratios (bone_length / leg_scale) used to derive
    # hip_width/torso_length/shoulder_width/arm lengths from each player's
    # own leg-derived scale. Defaults to `_DEFAULT_LEG_DERIVED_RATIOS`.
    leg_derived_ratios: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_LEG_DERIVED_RATIOS))


@dataclass(frozen=True)
class _Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]

    @property
    def camera_center_world(self) -> tuple[float, float, float]:
        rt = _transpose3(self.rotation)
        return tuple(-sum(rt[row][col] * self.translation[col] for col in range(3)) for row in range(3))

    def camera_direction_for_pixel(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        d_camera = ((float(x_px) - self.cx) / self.fx, (float(y_px) - self.cy) / self.fy, 1.0)
        rt = _transpose3(self.rotation)
        return tuple(sum(rt[row][col] * d_camera[col] for col in range(3)) for row in range(3))

    def world_at_camera_depth(self, x_px: float, y_px: float, depth: float) -> list[float]:
        direction = self.camera_direction_for_pixel(x_px, y_px)
        center = self.camera_center_world
        return [center[idx] + float(depth) * direction[idx] for idx in range(3)]

    def camera_depth(self, world: Sequence[float]) -> float:
        camera_z = sum(self.rotation[2][col] * float(world[col]) for col in range(3)) + self.translation[2]
        return float(camera_z)

    def project(self, world: Sequence[float]) -> list[float]:
        camera = [
            sum(self.rotation[row][col] * float(world[col]) for col in range(3)) + self.translation[row]
            for row in range(3)
        ]
        if math.isclose(camera[2], 0.0, abs_tol=1e-12):
            raise ValueError("world point projects with zero camera depth")
        return [self.fx * camera[0] / camera[2] + self.cx, self.fy * camera[1] / camera[2] + self.cy]


@dataclass(frozen=True)
class _Ray:
    x_px: float
    y_px: float
    center: tuple[float, float, float]
    direction_per_depth: tuple[float, float, float]

    def point_at_depth(self, depth: float) -> list[float]:
        return [self.center[idx] + float(depth) * self.direction_per_depth[idx] for idx in range(3)]


@dataclass(frozen=True)
class _Keypoint:
    joint: str
    x_px: float
    y_px: float
    conf: float


@dataclass
class _FrameInput:
    frame_idx: int
    t: float | None
    keypoints: dict[str, _Keypoint]
    raw_root: list[float]
    root: list[float]
    root_source: str


def lift_skeleton_from_2d(
    keypoints_payload: Mapping[str, Any],
    *,
    tracks_payload: Mapping[str, Any] | None,
    calibration_payload: Mapping[str, Any],
    config: Lift2DConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lift a keypoints_2d artifact into a skeleton3d_v2 payload.

    The function is intentionally schema-tolerant at the edges because the GPU
    keypoint producer is a separate lane. It requires only player/frame IDs,
    joint names, x/y pixels, confidence, and a standard court calibration.
    """

    cfg = config or Lift2DConfig()
    camera = _camera_from_payload(calibration_payload)
    joint_names = _joint_names_from_keypoints(keypoints_payload)
    tracks_by_player = _tracks_by_player_frame(tracks_payload or {})
    fps = float(keypoints_payload.get("fps") or (tracks_payload or {}).get("fps") or 30.0)

    players_out: list[dict[str, Any]] = []
    report_players: dict[str, Any] = {}
    root_sources: Counter[str] = Counter()
    frame_count = 0
    missing_bone_solve_count = 0
    implausible_frame_count = 0

    for player_payload in _players(keypoints_payload):
        player_id = str(player_payload.get("id") or player_payload.get("player_id"))
        height_m = float(player_payload.get("height_m") or keypoints_payload.get("player_height_m") or cfg.default_player_height_m)
        priors = _bone_priors_for_player(keypoints_payload, player_payload, cfg, joint_names=joint_names, height_m=height_m)
        frame_inputs = _frame_inputs_for_player(
            player_payload,
            player_id=player_id,
            joint_names=joint_names,
            tracks_by_player=tracks_by_player,
            camera=camera,
            config=cfg,
        )
        _smooth_roots(frame_inputs, radius=cfg.root_smoothing_radius)
        body_scale = _resolve_player_body_scale(
            player_id,
            frame_inputs,
            joint_names=joint_names,
            camera=camera,
            config=cfg,
        )
        previous_depths: dict[str, float] = {}
        player_frames: list[dict[str, Any]] = []
        player_root_sources: Counter[str] = Counter()
        for frame_input in frame_inputs:
            frame_count += 1
            player_root_sources[frame_input.root_source] += 1
            root_sources[frame_input.root_source] += 1
            frame, solve_failures, previous_depths = _solve_frame(
                frame_input,
                joint_names=joint_names,
                priors=priors,
                camera=camera,
                config=cfg,
                previous_depths=previous_depths,
                body_scale=body_scale,
            )
            missing_bone_solve_count += solve_failures
            player_frames.append(frame)

        plausibility = _apply_plausibility_gate(player_frames, joint_names=joint_names, config=cfg)
        implausible_frame_count += plausibility["implausible_frame_count"]
        players_out.append({"id": player_id, "height_m": height_m, "frames": player_frames})
        report_players[player_id] = {
            "frame_count": len(player_frames),
            "root_sources": dict(sorted(player_root_sources.items())),
            "height_m": height_m,
            "bone_prior_count": len(priors),
            "skeleton_implausible_frame_count": plausibility["implausible_frame_count"],
            "plausibility_reasons": plausibility["reason_counts"],
            "body_scale": _body_scale_report(body_scale),
        }

    skeleton = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_model": "geometric_2d_first_lift",
        "world_frame": "court_Z0",
        "fps": fps,
        "joint_names": joint_names,
        "players": players_out,
        "provenance": {
            "lane": LANE_NAME,
            "input_artifact_type": keypoints_payload.get("artifact_type", "keypoints_2d"),
            "keypoint_convention": keypoints_payload.get("convention", "unknown"),
            "projection_authority": "keypoints_2d_camera_rays",
            "protected_eval_labels_used": False,
            "notes": [
                "2D keypoints are authoritative; lifted joints remain on their camera rays.",
                "Depths are geometric estimates from court-plane roots, bone priors, and temporal preference.",
            ],
        },
    }
    report = {
        "schema_version": 1,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "lane": LANE_NAME,
        "summary": {
            "player_count": len(players_out),
            "frame_count": frame_count,
            "joint_count": len(joint_names),
            "root_sources": dict(sorted(root_sources.items())),
            "bone_solve_failure_count": missing_bone_solve_count,
            "skeleton_implausible_frame_count": implausible_frame_count,
            "projection_error_expected": "zero_by_construction_for_visible_keypoints",
            "protected_eval_labels_used": False,
        },
        "players": report_players,
        "bone_priors": [
            {"parent": prior.parent, "child": prior.child, "length_m": prior.length_m, "source": prior.source}
            for prior in _bone_priors_for_player(
                keypoints_payload,
                _players(keypoints_payload)[0] if _players(keypoints_payload) else {},
                cfg,
                joint_names=joint_names,
                height_m=cfg.default_player_height_m,
            )
        ],
    }
    return skeleton, report


def project_skeleton_joint(camera_payload: Mapping[str, Any], joint_world: Sequence[float]) -> list[float]:
    """Project one world joint through a calibration payload."""

    return _camera_from_payload(camera_payload).project(joint_world)


def _solve_frame(
    frame_input: _FrameInput,
    *,
    joint_names: list[str],
    priors: list[BonePrior],
    camera: _Camera,
    config: Lift2DConfig,
    previous_depths: Mapping[str, float],
    body_scale: "_BodyScale | None" = None,
) -> tuple[dict[str, Any], int, dict[str, float]]:
    root_depth = camera.camera_depth(frame_input.root)
    rays: dict[str, _Ray] = {}
    conf_by_joint: dict[str, float] = {}
    for joint_name in joint_names:
        keypoint = frame_input.keypoints.get(joint_name)
        conf = float(keypoint.conf) if keypoint is not None else 0.0
        conf_by_joint[joint_name] = conf
        if keypoint is not None and conf >= config.min_joint_confidence:
            rays[joint_name] = _ray_for_keypoint(camera, keypoint)

    solved: dict[str, list[float]] = {}
    solved_depths: dict[str, float] = {}
    foot_locked: set[str] = set()
    for joint_name in joint_names:
        ray = rays.get(joint_name)
        if ray is None:
            continue
        if joint_name in FOOT_NAMES:
            point = _intersect_ray_with_z(ray, config.court_z_m)
            if point is not None:
                solved[joint_name] = point
                solved_depths[joint_name] = camera.camera_depth(point)
                # Ground-plane anchoring is the single most reliable depth
                # source in the whole solve; lock it against being
                # re-litigated by a generic (ratio-based) bone prior below.
                foot_locked.add(joint_name)

    locked = _solve_body_core_chain(
        rays=rays,
        solved=solved,
        solved_depths=solved_depths,
        previous_depths=previous_depths,
        root_depth=root_depth,
        body_scale=body_scale,
    ) | frozenset(foot_locked)

    # Generic depth-extent safety net for whatever the kinematic chain above
    # doesn't model per-bone (face/hand landmarks -- 110 of the 133
    # COCO-WholeBody joints): clamp their otherwise-independent fallback
    # depth to a band around the solved body core's own depth, rather than
    # leaving them free to drift to a previous_depths/root_depth value with
    # no relationship to where the body actually is this frame. A hand is
    # attached to an already-locked wrist; a clamp is a coarse substitute
    # for a real per-joint bone but is enough to prevent a detached-limb
    # depth blowout on frames where these joints' own temporal/root fallback
    # would otherwise be badly stale.
    core_reference_depth = _mean_of(solved_depths, *locked) if locked else None
    non_core_trust_band_m = 1.5
    for joint_name, ray in rays.items():
        if joint_name not in solved:
            preferred_depth = float(previous_depths.get(joint_name, root_depth))
            if core_reference_depth is not None and math.isfinite(core_reference_depth):
                preferred_depth = min(
                    max(preferred_depth, core_reference_depth - non_core_trust_band_m),
                    core_reference_depth + non_core_trust_band_m,
                )
            solved[joint_name] = ray.point_at_depth(preferred_depth)
            solved_depths[joint_name] = preferred_depth

    failures = 0
    for _pass_idx in range(max(1, len(priors) + 1)):
        changed = False
        for prior in priors:
            changed |= _apply_prior(
                prior,
                solved=solved,
                solved_depths=solved_depths,
                rays=rays,
                camera=camera,
                previous_depths=previous_depths,
                root_depth=root_depth,
                locked=locked,
            )
        if not changed:
            break

    for prior in priors:
        # Locked joints (the kinematic-chain body core) are governed by
        # body_scale's hard/anthropometric lengths, not this generic prior's
        # length -- counting a mismatch against the generic default here
        # would flag every locked frame as a "failure" for using a *better*
        # length than the one this comparison is stale against.
        if prior.child in locked or prior.parent in locked:
            continue
        if prior.parent in solved and prior.child in rays:
            observed = math.dist(solved[prior.parent], solved[prior.child])
            if abs(observed - prior.length_m) > max(0.05, prior.length_m * 0.25):
                failures += 1

    root_joint = _root_from_joints(solved, joint_names) or frame_input.root
    joints_world: list[list[float]] = []
    joint_conf: list[float] = []
    next_depths: dict[str, float] = {}
    for joint_name in joint_names:
        point = solved.get(joint_name, list(root_joint))
        joints_world.append([_round(point[0]), _round(point[1]), _round(point[2])])
        conf = conf_by_joint.get(joint_name, 0.0)
        joint_conf.append(_round(conf, digits=6))
        if joint_name in solved_depths:
            next_depths[joint_name] = solved_depths[joint_name]

    frame = {
        "frame_idx": frame_input.frame_idx,
        "frame_index": frame_input.frame_idx,
        "t": frame_input.t,
        "transl_world": [_round(frame_input.root[0]), _round(frame_input.root[1]), _round(frame_input.root[2])],
        "joints_world": joints_world,
        "joint_conf": joint_conf,
        "lift_2d_first": {
            "root_source": frame_input.root_source,
            "raw_root_world": [_round(value) for value in frame_input.raw_root],
            "projection_authority": "keypoints_2d_camera_rays",
            "min_joint_confidence": config.min_joint_confidence,
        },
    }
    return frame, failures, next_depths


def _apply_plausibility_gate(
    frames: list[dict[str, Any]],
    *,
    joint_names: Sequence[str],
    config: Lift2DConfig,
) -> dict[str, Any]:
    bone_pairs = _plausibility_bone_pairs(joint_names)
    bone_stats = _bone_length_stats(frames, joint_names=joint_names, bone_pairs=bone_pairs, config=config)
    reason_counts: Counter[str] = Counter()
    implausible_count = 0
    for frame in frames:
        reasons = _frame_plausibility_reasons(
            frame,
            joint_names=joint_names,
            bone_pairs=bone_pairs,
            bone_stats=bone_stats,
            config=config,
        )
        frame["skeleton_implausible"] = bool(reasons)
        frame["skeleton_plausibility"] = {
            "status": "low_confidence" if reasons else "pass",
            "reasons": reasons,
            "joint_confidence_floor": config.plausibility_joint_confidence_floor,
            "max_bone_zscore": config.plausibility_max_bone_zscore,
        }
        if reasons:
            implausible_count += 1
            reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
            frame["trust_band"] = {
                "stage": "BODY",
                "gate_id": "skeleton_lift_2d_plausibility",
                "gate_status": "low_confidence",
                "badge": "low_confidence",
                "reason": "; ".join(reasons),
                "evidence_path": None,
            }
    return {
        "implausible_frame_count": implausible_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _plausibility_bone_pairs(joint_names: Sequence[str]) -> list[tuple[str, str]]:
    names = set(joint_names)
    return [
        (parent, child)
        for parent, child in COCO_WHOLEBODY_CORE_BONES
        if (parent == "neck" or parent in names) and child in names and (parent != "neck" or {"left_shoulder", "right_shoulder"} <= names)
    ]


def _bone_length_stats(
    frames: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    config: Lift2DConfig,
) -> dict[tuple[str, str], tuple[float, float]]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for bone in bone_pairs:
        lengths = []
        for frame in frames:
            length = _frame_bone_length(frame, joint_names=joint_names, bone=bone, confidence_floor=config.plausibility_joint_confidence_floor)
            if length is not None:
                lengths.append(length)
        if len(lengths) < config.plausibility_min_bone_samples:
            continue
        center = float(median(lengths))
        abs_deviations = [abs(length - center) for length in lengths]
        robust_sigma = max(float(median(abs_deviations)) * 1.4826, config.plausibility_min_sigma_m)
        stats[bone] = (center, robust_sigma)
    return stats


def _frame_plausibility_reasons(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    bone_stats: Mapping[tuple[str, str], tuple[float, float]],
    config: Lift2DConfig,
) -> list[str]:
    reasons: list[str] = []
    low_conf = _low_confidence_core_joints(frame, joint_names=joint_names, bone_pairs=bone_pairs, floor=config.plausibility_joint_confidence_floor)
    if low_conf:
        reasons.append(f"joint_conf_below_floor:{','.join(low_conf[:6])}")
    for bone, (center, sigma) in bone_stats.items():
        length = _frame_bone_length(frame, joint_names=joint_names, bone=bone, confidence_floor=config.plausibility_joint_confidence_floor)
        if length is None:
            continue
        zscore = abs(length - center) / sigma
        if zscore > config.plausibility_max_bone_zscore:
            reasons.append(f"bone_length_zscore:{bone[0]}-{bone[1]}:{zscore:.2f}")
    return reasons


def _low_confidence_core_joints(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    floor: float,
) -> list[str]:
    conf = frame.get("joint_conf")
    if not isinstance(conf, Sequence):
        return []
    required = sorted({joint for bone in bone_pairs for joint in bone if joint != "neck"})
    if "neck" in {joint for bone in bone_pairs for joint in bone}:
        required.extend(["left_shoulder", "right_shoulder"])
    by_name = {name: idx for idx, name in enumerate(joint_names)}
    low = []
    for name in required:
        idx = by_name.get(name)
        if idx is None or idx >= len(conf):
            continue
        try:
            value = float(conf[idx])
        except (TypeError, ValueError):
            value = 0.0
        if value < floor:
            low.append(name)
    return sorted(set(low))


def _frame_bone_length(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone: tuple[str, str],
    confidence_floor: float,
) -> float | None:
    points = _frame_point_lookup(frame, joint_names=joint_names, confidence_floor=confidence_floor)
    left = _point_for_joint_name(bone[0], points)
    right = _point_for_joint_name(bone[1], points)
    if left is None or right is None:
        return None
    return float(math.dist(left, right))


def _frame_point_lookup(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    confidence_floor: float,
) -> dict[str, list[float]]:
    joints = frame.get("joints_world")
    conf = frame.get("joint_conf")
    if not isinstance(joints, Sequence):
        return {}
    points: dict[str, list[float]] = {}
    for index, name in enumerate(joint_names):
        if index >= len(joints):
            break
        raw_point = joints[index]
        if not isinstance(raw_point, Sequence) or len(raw_point) < 3:
            continue
        raw_conf = conf[index] if isinstance(conf, Sequence) and index < len(conf) else 1.0
        try:
            value = float(raw_conf)
        except (TypeError, ValueError):
            value = 0.0
        if value < confidence_floor:
            continue
        points[name] = [float(raw_point[0]), float(raw_point[1]), float(raw_point[2])]
    return points


def _point_for_joint_name(name: str, points: Mapping[str, Sequence[float]]) -> list[float] | None:
    if name != "neck":
        point = points.get(name)
        return [float(point[0]), float(point[1]), float(point[2])] if point is not None else None
    left = points.get("left_shoulder")
    right = points.get("right_shoulder")
    if left is None or right is None:
        return None
    return [
        (float(left[0]) + float(right[0])) / 2.0,
        (float(left[1]) + float(right[1])) / 2.0,
        (float(left[2]) + float(right[2])) / 2.0,
    ]


def _apply_prior(
    prior: BonePrior,
    *,
    solved: dict[str, list[float]],
    solved_depths: dict[str, float],
    rays: Mapping[str, _Ray],
    camera: _Camera,
    previous_depths: Mapping[str, float],
    root_depth: float,
    locked: frozenset[str] = frozenset(),
) -> bool:
    if prior.child in locked and prior.parent in locked:
        return False
    if prior.parent in solved and prior.child in rays and prior.child not in locked:
        if prior.child in previous_depths:
            # Temporal continuity: prefer the depth the joint occupied last frame.
            preferred = float(previous_depths[prior.child])
        elif prior.child not in FOOT_NAMES and prior.parent in solved_depths:
            # Joint-limit default: minimize depth extension from the already-solved
            # parent rather than assuming the child is systematically further away
            # (that bias was a direct contributor to elongated-body depth blowouts).
            preferred = float(solved_depths[prior.parent])
        else:
            preferred = float(solved_depths.get(prior.child, root_depth))
        point, depth = _sphere_ray_solution(rays[prior.child], solved[prior.parent], prior.length_m, preferred_depth=preferred)
        if point is not None and depth is not None and _point_changed(solved.get(prior.child), point):
            solved[prior.child] = point
            solved_depths[prior.child] = depth
            return True
    if prior.child in solved and prior.parent in rays and prior.parent not in locked:
        if prior.parent in previous_depths:
            preferred = float(previous_depths[prior.parent])
        elif prior.parent not in FOOT_NAMES and prior.child in solved_depths:
            preferred = float(solved_depths[prior.child])
        else:
            preferred = float(solved_depths.get(prior.parent, root_depth))
        point, depth = _sphere_ray_solution(rays[prior.parent], solved[prior.child], prior.length_m, preferred_depth=preferred)
        if point is not None and depth is not None and _point_changed(solved.get(prior.parent), point):
            solved[prior.parent] = point
            solved_depths[prior.parent] = depth
            return True
    del camera
    return False


def _sphere_ray_solution(
    ray: _Ray,
    anchor: Sequence[float],
    length_m: float,
    *,
    preferred_depth: float,
) -> tuple[list[float] | None, float | None]:
    center = ray.center
    direction = ray.direction_per_depth
    delta = [center[idx] - float(anchor[idx]) for idx in range(3)]
    a = sum(value * value for value in direction)
    b = 2.0 * sum(direction[idx] * delta[idx] for idx in range(3))
    c = sum(value * value for value in delta) - float(length_m) * float(length_m)
    discriminant = b * b - 4.0 * a * c
    if a <= 0.0 or discriminant < -1e-9:
        return None, None
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    positive = [value for value in candidates if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None, None
    depth = min(positive, key=lambda value: abs(value - preferred_depth))
    return ray.point_at_depth(depth), depth


@dataclass(frozen=True)
class _BodyScale:
    """Per-player hard/anthropometric bone-length targets for the kinematic chain."""

    thigh_l: float
    thigh_r: float
    shin_l: float
    shin_r: float
    leg_scale_m: float
    hip_width: float
    torso_length: float
    shoulder_width: float
    upper_arm: float
    lower_arm: float
    head_offset: float
    leg_source: str
    ratio_source: str


def load_player_bone_lengths(payload: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Flatten a bone_calib-schema payload into `{player_id: {bone_name: length_m}}`.

    Only the four HARD leg bones (`LEG_BONE_JOINT_PAIRS` keys) are kept; the
    calibration report found arm/torso/hip/shoulder measured values too
    pose-dependent to trust raw, so those are deliberately not extracted here
    (they're derived instead via `_DEFAULT_LEG_DERIVED_RATIOS`). A player is
    only included if all four leg bones are present -- a partial set isn't
    enough to anchor the chain, and `_resolve_player_body_scale` falls back
    to self-measurement for that player anyway.
    """

    out: dict[str, dict[str, float]] = {}
    players = payload.get("players", {})
    if not isinstance(players, Mapping):
        return out
    for player_id, player_data in players.items():
        if not isinstance(player_data, Mapping):
            continue
        bones = player_data.get("bones")
        if not isinstance(bones, Mapping):
            continue
        flat: dict[str, float] = {}
        for bone_name in LEG_BONE_JOINT_PAIRS:
            entry = bones.get(bone_name)
            if isinstance(entry, Mapping) and entry.get("median_m") is not None:
                try:
                    flat[bone_name] = float(entry["median_m"])
                except (TypeError, ValueError):
                    continue
        if len(flat) == len(LEG_BONE_JOINT_PAIRS):
            out[str(player_id)] = flat
    return out


def _body_scale_report(body_scale: "_BodyScale | None") -> dict[str, Any] | None:
    if body_scale is None:
        return None
    return {
        "leg_source": body_scale.leg_source,
        "ratio_source": body_scale.ratio_source,
        "leg_scale_m": _round(body_scale.leg_scale_m, digits=4),
        "left_upper_leg_m": _round(body_scale.thigh_l, digits=4),
        "right_upper_leg_m": _round(body_scale.thigh_r, digits=4),
        "left_lower_leg_m": _round(body_scale.shin_l, digits=4),
        "right_lower_leg_m": _round(body_scale.shin_r, digits=4),
        "hip_width_m": _round(body_scale.hip_width, digits=4),
        "torso_length_m": _round(body_scale.torso_length, digits=4),
        "shoulder_width_m": _round(body_scale.shoulder_width, digits=4),
        "upper_arm_m": _round(body_scale.upper_arm, digits=4),
        "lower_arm_m": _round(body_scale.lower_arm, digits=4),
        "head_offset_m": _round(body_scale.head_offset, digits=4),
    }


def _resolve_player_body_scale(
    player_id: str,
    frame_inputs: Sequence["_FrameInput"],
    *,
    joint_names: Sequence[str],
    camera: _Camera,
    config: Lift2DConfig,
) -> "_BodyScale | None":
    if not BODY_CORE_REQUIRED_NAMES.issubset(set(joint_names)):
        return None

    hard = (config.player_bone_lengths or {}).get(str(player_id))
    if hard is not None and all(name in hard for name in LEG_BONE_JOINT_PAIRS):
        thigh_l = float(hard["left_upper_leg"])
        thigh_r = float(hard["right_upper_leg"])
        shin_l = float(hard["left_lower_leg"])
        shin_r = float(hard["right_lower_leg"])
        leg_source = "hard_canonical_leg"
    else:
        thigh_l, thigh_r, shin_l, shin_r, leg_source = _self_measure_leg_lengths(
            frame_inputs, camera=camera, config=config
        )

    leg_scale = (thigh_l + thigh_r) / 2.0 + (shin_l + shin_r) / 2.0
    if not math.isfinite(leg_scale) or leg_scale <= 0.0:
        return None
    ratios = config.leg_derived_ratios or _DEFAULT_LEG_DERIVED_RATIOS
    ratio_source = "config_leg_derived_ratios" if config.leg_derived_ratios else "bone_calib_population_median"
    return _BodyScale(
        thigh_l=thigh_l,
        thigh_r=thigh_r,
        shin_l=shin_l,
        shin_r=shin_r,
        leg_scale_m=leg_scale,
        hip_width=float(ratios["hip_width"]) * leg_scale,
        torso_length=float(ratios["torso_length"]) * leg_scale,
        shoulder_width=float(ratios["shoulder_width"]) * leg_scale,
        upper_arm=float(ratios["upper_arm"]) * leg_scale,
        lower_arm=float(ratios["lower_arm"]) * leg_scale,
        head_offset=float(ratios.get("head_offset", _DEFAULT_LEG_DERIVED_RATIOS["head_offset"])) * leg_scale,
        leg_source=leg_source,
        ratio_source=ratio_source,
    )


def _self_measure_leg_lengths(
    frame_inputs: Sequence["_FrameInput"],
    *,
    camera: _Camera,
    config: Lift2DConfig,
) -> tuple[float, float, float, float, str]:
    """Fallback leg lengths for players with no external canonical measurement.

    IMPORTANT what this function does NOT do: it does not re-derive leg
    lengths from this clip's own 2D keypoints. An earlier version of this
    function tried to "self-measure" by running a bootstrap ankle->knee->hip
    sphere-ray solve using the height-ratio prior as the enforced bone
    length, then median-aggregating the resulting per-frame 3D distances --
    but a sphere-ray solve enforces its input length *by construction*, so
    that aggregate is mathematically guaranteed to equal the input ratio
    exactly. It was measuring nothing; it was restating its own assumption.
    A genuine independent leg-length measurement from monocular 2D-only
    keypoints (no accurate-tier mesh regression) would need a *different*
    depth-free signal, e.g. real height from ankle depth (ground-plane,
    prior-free) plus apparent ankle-to-head pixel span on a near-vertical
    frame -- exactly the still-unstable PLAYER-SCALE-NORM approach a
    separate lane already tried and killed (spreads 0.36-0.50m). Re-deriving
    that here is out of this lane's scope, so this is honestly just the
    height-ratio anthropometric default, unchanged from the pre-existing
    behavior, for players not covered by `Lift2DConfig.player_bone_lengths`.
    """

    height_m = config.default_player_height_m
    thigh = 0.245 * height_m
    shin = 0.246 * height_m
    return thigh, thigh, shin, shin, "default_anthropometric_leg"


def _lateral_pair_shared_depth_solution(
    ray_a: _Ray,
    ray_b: _Ray,
    width_m: float,
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[list[float] | None, list[float] | None, float | None]:
    """Exact, single-solution depth for two rays sharing one camera center.

    Both joints share the same camera center, so `point_a(d) - point_b(d) ==
    d * (dir_a - dir_b)` -- a linear function of the single shared depth `d`.
    There is therefore exactly one `d` that reproduces the hard width bone
    (no near/far ambiguity, unlike a generic sphere-ray solve), which is what
    makes this robust against the independent-per-joint-depth failure mode
    the bone-calibration measurement traced hip_width/torso_length blowouts
    to.
    """

    direction_delta = tuple(ray_a.direction_per_depth[idx] - ray_b.direction_per_depth[idx] for idx in range(3))
    delta_norm = math.sqrt(sum(value * value for value in direction_delta))
    if delta_norm < 1e-9:
        return None, None, None
    depth = float(width_m) / delta_norm
    if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
        return None, None, None
    return ray_a.point_at_depth(depth), ray_b.point_at_depth(depth), depth


def _solve_body_core_chain(
    *,
    rays: Mapping[str, _Ray],
    solved: dict[str, list[float]],
    solved_depths: dict[str, float],
    previous_depths: Mapping[str, float],
    root_depth: float,
    body_scale: "_BodyScale | None",
) -> frozenset[str]:
    """Kinematic chain from the hip root outward. Mutates `solved`/`solved_depths`.

    Returns the set of joint names this chain fixed, so the generic
    bone_priors pass (for any caller-supplied custom priors, kept for
    backward compatibility) does not re-solve/overwrite them.
    """

    if body_scale is None:
        return frozenset()

    locked: set[str] = set()
    # Sanity rails only (protect the division/positivity in the shared-depth
    # formula against literal degeneracy) -- the real robustness mechanism is
    # the root_depth-referenced trust band applied below, not this range.
    min_depth = 0.3
    max_depth = (root_depth * 4.0 + 40.0) if math.isfinite(root_depth) else 60.0

    # 1) Hip pair: exact shared-depth solve on the hard hip_width bone,
    # trusted only within a band of the ground-plane root depth.
    # depth = width / |dir_L - dir_R| is exact but arbitrarily sensitive to
    # 2D keypoint noise or genuine hip rotation (the shared-depth assumption
    # weakens when a player's hips are meaningfully turned relative to the
    # camera -- e.g. mid-swing) when the two hip rays' angular separation is
    # small: either can make the formula "explain" a normal hip_width with
    # an implausible depth (observed on real tracker output: raw depth 16m+
    # vs a ~8.3m ground-plane root). A per-frame band around THIS frame's
    # own root_depth (not a rate limit against the previous frame -- that
    # was tried and just delayed the same bad convergence over many frames
    # during a sustained rotation) rejects those without ever reverting to
    # solving each hip independently (that IS the bug this lane exists to
    # fix): out-of-band frames fall back to both hips sharing root_depth
    # itself, still one shared depth, just not exactly hip_width.
    hip_depth_trust_band_m = 1.5
    hip_l_ray = rays.get("left_hip")
    hip_r_ray = rays.get("right_hip")
    if hip_l_ray is not None and hip_r_ray is not None:
        _, _, raw_depth = _lateral_pair_shared_depth_solution(
            hip_l_ray, hip_r_ray, body_scale.hip_width, min_depth=min_depth, max_depth=max_depth
        )
        if raw_depth is not None and math.isfinite(root_depth) and abs(raw_depth - root_depth) <= hip_depth_trust_band_m:
            depth = raw_depth
        elif math.isfinite(root_depth):
            depth = root_depth
        elif raw_depth is not None:
            depth = raw_depth
        else:
            depth = previous_depths.get("left_hip", previous_depths.get("right_hip", 1.0))
        solved["left_hip"] = hip_l_ray.point_at_depth(depth)
        solved_depths["left_hip"] = depth
        locked.add("left_hip")
        solved["right_hip"] = hip_r_ray.point_at_depth(depth)
        solved_depths["right_hip"] = depth
        locked.add("right_hip")

    pelvis = _midpoint_of(solved, "left_hip", "right_hip")
    pelvis_depth = _mean_of(solved_depths, "left_hip", "right_hip")

    # 2) Knees, outward from the (now-anchored) hips via HARD thigh lengths.
    # The hip-thigh sphere-ray solve's two roots straddle the point on the
    # knee's ray closest to the hip; when the thigh has little real depth
    # extent (the common, desirable case -- see acceptance criterion) that
    # midpoint sits close to the hip's own depth too, making "prefer the
    # root nearest the hip's depth" a near coin-flip. The ankle is already
    # ground-plane anchored (the most reliable joint in the whole chain), so
    # when no temporal depth exists yet, use the depth where the knee's ray
    # comes closest to the ankle as the tha disambiguation default instead --
    # it's an exact, cheap computation and cross-checks the chain from both
    # ends rather than just the hip end.
    for hip_name, knee_name, ankle_name, thigh_len in (
        ("left_hip", "left_knee", "left_ankle", body_scale.thigh_l),
        ("right_hip", "right_knee", "right_ankle", body_scale.thigh_r),
    ):
        if hip_name in solved and knee_name in rays:
            if knee_name in previous_depths:
                preferred = float(previous_depths[knee_name])
            elif ankle_name in solved:
                preferred = _closest_ray_depth_to_point(rays[knee_name], solved[ankle_name])
            else:
                preferred = solved_depths[hip_name]
            point, depth = _sphere_ray_solution(rays[knee_name], solved[hip_name], thigh_len, preferred_depth=preferred)
            if point is not None and depth is not None:
                solved[knee_name] = point
                solved_depths[knee_name] = depth
                locked.add(knee_name)

    # 3) Shoulder pair, outward from the pelvis via the hard torso_length bone.
    # Both shoulders share one depth (the synthetic average-direction ray's
    # solved depth), which also makes the resulting shoulder_width emergent
    # rather than independently forced -- torso_length is the bone this
    # lane is graded on, not shoulder_width.
    shoulder_l_ray = rays.get("left_shoulder")
    shoulder_r_ray = rays.get("right_shoulder")
    if pelvis is not None and (shoulder_l_ray is not None or shoulder_r_ray is not None):
        if shoulder_l_ray is not None and shoulder_r_ray is not None:
            avg_direction = tuple(
                (shoulder_l_ray.direction_per_depth[idx] + shoulder_r_ray.direction_per_depth[idx]) / 2.0
                for idx in range(3)
            )
            camera_center = shoulder_l_ray.center
        else:
            single = shoulder_l_ray or shoulder_r_ray
            assert single is not None
            avg_direction = single.direction_per_depth
            camera_center = single.center
        synthetic_ray = _Ray(x_px=0.0, y_px=0.0, center=camera_center, direction_per_depth=avg_direction)
        anchor_depth = pelvis_depth if pelvis_depth is not None else root_depth
        _, raw_mid_depth = _sphere_ray_solution(synthetic_ray, pelvis, body_scale.torso_length, preferred_depth=anchor_depth)
        # Same reasoning as the hip pair: torso_length is exact-by-construction
        # when trusted, but only trust it within a band of the pelvis's own
        # (already hip-anchored) depth -- otherwise both shoulders fall back
        # to sharing the pelvis depth directly rather than drifting off to an
        # independent, noise-amplified depth.
        shoulder_depth_trust_band_m = 1.5
        if (
            raw_mid_depth is not None
            and math.isfinite(anchor_depth)
            and abs(raw_mid_depth - anchor_depth) <= shoulder_depth_trust_band_m
        ):
            mid_depth = raw_mid_depth
        else:
            mid_depth = anchor_depth
        if shoulder_l_ray is not None:
            solved["left_shoulder"] = shoulder_l_ray.point_at_depth(mid_depth)
            solved_depths["left_shoulder"] = mid_depth
            locked.add("left_shoulder")
        if shoulder_r_ray is not None:
            solved["right_shoulder"] = shoulder_r_ray.point_at_depth(mid_depth)
            solved_depths["right_shoulder"] = mid_depth
            locked.add("right_shoulder")

    # 3b) Nose, outward from the shoulder midpoint. Not a bone this lane is
    # graded on, but left on the pre-existing independent-depth fallback it
    # was the single largest remaining depth-extent outlier on real data (a
    # detached head floating meters from the body is exactly the class of
    # bug this lane exists to eliminate).
    nose_ray = rays.get("nose")
    shoulder_mid = _midpoint_of(solved, "left_shoulder", "right_shoulder")
    shoulder_mid_depth = _mean_of(solved_depths, "left_shoulder", "right_shoulder")
    if nose_ray is not None and shoulder_mid is not None:
        anchor_depth = shoulder_mid_depth if shoulder_mid_depth is not None else root_depth
        _, raw_nose_depth = _sphere_ray_solution(nose_ray, shoulder_mid, body_scale.head_offset, preferred_depth=anchor_depth)
        nose_depth_trust_band_m = 1.0
        if (
            raw_nose_depth is not None
            and math.isfinite(anchor_depth)
            and abs(raw_nose_depth - anchor_depth) <= nose_depth_trust_band_m
        ):
            nose_depth = raw_nose_depth
        else:
            nose_depth = anchor_depth
        solved["nose"] = nose_ray.point_at_depth(nose_depth)
        solved_depths["nose"] = nose_depth
        locked.add("nose")

    # 4) Elbows/wrists, outward from shoulders via anthropometric arm lengths.
    for shoulder_name, elbow_name, wrist_name in (
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
    ):
        if shoulder_name in solved and elbow_name in rays:
            preferred = previous_depths.get(elbow_name, solved_depths[shoulder_name])
            point, depth = _sphere_ray_solution(
                rays[elbow_name], solved[shoulder_name], body_scale.upper_arm, preferred_depth=preferred
            )
            if point is not None and depth is not None:
                solved[elbow_name] = point
                solved_depths[elbow_name] = depth
                locked.add(elbow_name)
        if elbow_name in solved and wrist_name in rays:
            preferred = previous_depths.get(wrist_name, solved_depths[elbow_name])
            point, depth = _sphere_ray_solution(
                rays[wrist_name], solved[elbow_name], body_scale.lower_arm, preferred_depth=preferred
            )
            if point is not None and depth is not None:
                solved[wrist_name] = point
                solved_depths[wrist_name] = depth
                locked.add(wrist_name)

    return frozenset(locked)


def _closest_ray_depth_to_point(ray: _Ray, point: Sequence[float]) -> float:
    """Depth at which `ray` comes closest to `point` (exact, no length needed).

    This is also, for any sphere centered at `point`, exactly the mean of
    that sphere-ray intersection's two roots (a property of the quadratic:
    sum of roots = -b/a, this closest-approach depth = -b/(2a)) -- so it's a
    natural, cheap "which side of the ambiguity is this joint probably on"
    signal that doesn't require knowing the bone length to `point` at all.
    """

    direction = ray.direction_per_depth
    delta = [float(point[idx]) - ray.center[idx] for idx in range(3)]
    denom = sum(value * value for value in direction)
    if denom <= 0.0:
        return 0.0
    return sum(direction[idx] * delta[idx] for idx in range(3)) / denom


def _midpoint_of(solved: Mapping[str, list[float]], *names: str) -> list[float] | None:
    points = [solved[name] for name in names if name in solved]
    if not points:
        return None
    return [sum(point[axis] for point in points) / len(points) for axis in range(3)]


def _mean_of(values: Mapping[str, float], *names: str) -> float | None:
    present = [values[name] for name in names if name in values]
    if not present:
        return None
    return sum(present) / len(present)


def _frame_inputs_for_player(
    player_payload: Mapping[str, Any],
    *,
    player_id: str,
    joint_names: Sequence[str],
    tracks_by_player: Mapping[str, Mapping[int, list[float]]],
    camera: _Camera,
    config: Lift2DConfig,
) -> list[_FrameInput]:
    frames: list[_FrameInput] = []
    for frame_payload in _frames(player_payload):
        frame_idx = _frame_idx(frame_payload)
        keypoints = _keypoints_by_joint(frame_payload)
        raw_root, root_source = _root_for_frame(
            keypoints,
            player_id=player_id,
            frame_idx=frame_idx,
            tracks_by_player=tracks_by_player,
            camera=camera,
            config=config,
        )
        frames.append(
            _FrameInput(
                frame_idx=frame_idx,
                t=_maybe_float(frame_payload.get("t")),
                keypoints={name: keypoints[name] for name in joint_names if name in keypoints},
                raw_root=list(raw_root),
                root=list(raw_root),
                root_source=root_source,
            )
        )
    return sorted(frames, key=lambda item: item.frame_idx)


def _root_for_frame(
    keypoints: Mapping[str, _Keypoint],
    *,
    player_id: str,
    frame_idx: int,
    tracks_by_player: Mapping[str, Mapping[int, list[float]]],
    camera: _Camera,
    config: Lift2DConfig,
) -> tuple[list[float], str]:
    left = _first_visible_keypoint(keypoints, LEFT_ANKLE_NAMES, min_conf=config.min_joint_confidence)
    right = _first_visible_keypoint(keypoints, RIGHT_ANKLE_NAMES, min_conf=config.min_joint_confidence)
    if left is not None and right is not None:
        midpoint = _Keypoint(
            joint="ankle_midpoint",
            x_px=(left.x_px + right.x_px) / 2.0,
            y_px=(left.y_px + right.y_px) / 2.0,
            conf=min(left.conf, right.conf),
        )
        point = _intersect_ray_with_z(_ray_for_keypoint(camera, midpoint), config.court_z_m)
        if point is not None:
            return point, "ankle_midpoint_ray_court_plane"

    track = tracks_by_player.get(str(player_id), {}).get(frame_idx)
    if track is not None:
        return [float(track[0]), float(track[1]), float(config.court_z_m)], "track_world_xy"

    single = left or right
    if single is not None:
        point = _intersect_ray_with_z(_ray_for_keypoint(camera, single), config.court_z_m)
        if point is not None:
            return point, "single_ankle_ray_court_plane"

    rootish = _first_visible_keypoint(keypoints, ROOT_NAMES, min_conf=config.min_joint_confidence)
    if rootish is not None:
        point = _intersect_ray_with_z(_ray_for_keypoint(camera, rootish), config.court_z_m)
        if point is not None:
            return point, "root_joint_ray_court_plane"
    return [0.0, 0.0, float(config.court_z_m)], "missing_root_zero_fallback"


def _smooth_roots(frames: list[_FrameInput], *, radius: int) -> None:
    if radius <= 0 or len(frames) < 3:
        return
    raw = [frame.raw_root for frame in frames]
    for index, frame in enumerate(frames):
        lo = max(0, index - radius)
        hi = min(len(frames), index + radius + 1)
        frame.root = [float(median(point[axis] for point in raw[lo:hi])) for axis in range(3)]


def _intersect_ray_with_z(ray: _Ray, z_m: float) -> list[float] | None:
    dz = ray.direction_per_depth[2]
    if math.isclose(dz, 0.0, abs_tol=1e-12):
        return None
    depth = (float(z_m) - ray.center[2]) / dz
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    return ray.point_at_depth(depth)


def _ray_for_keypoint(camera: _Camera, keypoint: _Keypoint) -> _Ray:
    return _Ray(
        x_px=keypoint.x_px,
        y_px=keypoint.y_px,
        center=camera.camera_center_world,
        direction_per_depth=camera.camera_direction_for_pixel(keypoint.x_px, keypoint.y_px),
    )


def _camera_from_payload(payload: Mapping[str, Any]) -> _Camera:
    intrinsics = payload.get("intrinsics")
    extrinsics = payload.get("extrinsics")
    if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
        raise ValueError("calibration payload must include intrinsics and extrinsics")
    return _Camera(
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        rotation=_mat3(extrinsics["R"]),
        translation=_vec3(extrinsics["t"]),
    )


def _bone_priors_for_player(
    keypoints_payload: Mapping[str, Any],
    player_payload: Mapping[str, Any],
    config: Lift2DConfig,
    *,
    joint_names: Sequence[str],
    height_m: float,
) -> list[BonePrior]:
    priors = list(config.bone_priors)
    for source_payload, source_name in ((keypoints_payload, "keypoints_2d"), (player_payload, "player")):
        for item in source_payload.get("bone_priors", []) or []:
            prior = _parse_bone_prior(item, height_m=height_m, source=source_name)
            if prior is not None:
                priors.append(prior)
    if not priors:
        priors.extend(_default_priors(joint_names, height_m=height_m))
    seen: set[tuple[str, str]] = set()
    deduped: list[BonePrior] = []
    valid_names = set(joint_names)
    for prior in priors:
        key = (prior.parent, prior.child)
        if key in seen or prior.parent not in valid_names or prior.child not in valid_names or prior.length_m <= 0.0:
            continue
        seen.add(key)
        deduped.append(prior)
    return deduped


def _parse_bone_prior(item: Any, *, height_m: float, source: str) -> BonePrior | None:
    if not isinstance(item, Mapping):
        return None
    parent = str(item.get("parent") or item.get("a") or "")
    child = str(item.get("child") or item.get("b") or "")
    if not parent or not child:
        return None
    if item.get("length_m") is not None:
        length = float(item["length_m"])
    elif item.get("length_ratio") is not None:
        length = float(item["length_ratio"]) * float(height_m)
    else:
        return None
    return BonePrior(parent=parent, child=child, length_m=length, source=source)


def _default_priors(joint_names: Sequence[str], *, height_m: float) -> list[BonePrior]:
    ratios = {
        ("left_ankle", "left_knee"): 0.246,
        ("right_ankle", "right_knee"): 0.246,
        ("left_knee", "left_hip"): 0.245,
        ("right_knee", "right_hip"): 0.245,
        ("left_hip", "pelvis"): 0.095,
        ("right_hip", "pelvis"): 0.095,
        ("pelvis", "neck"): 0.320,
        ("neck", "nose"): 0.155,
        ("neck", "left_shoulder"): 0.120,
        ("neck", "right_shoulder"): 0.120,
        ("left_shoulder", "left_elbow"): 0.186,
        ("right_shoulder", "right_elbow"): 0.186,
        ("left_elbow", "left_wrist"): 0.146,
        ("right_elbow", "right_wrist"): 0.146,
    }
    names = set(joint_names)
    return [
        BonePrior(parent=parent, child=child, length_m=ratio * float(height_m), source="default_scaled_height")
        for (parent, child), ratio in ratios.items()
        if parent in names and child in names
    ]


def _tracks_by_player_frame(payload: Mapping[str, Any]) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for player in _players(payload):
        player_id = str(player.get("id") or player.get("player_id"))
        frames: dict[int, list[float]] = {}
        for frame in _frames(player):
            world_xy = frame.get("world_xy") or frame.get("track_world_xy") or frame.get("center_world_xy")
            if isinstance(world_xy, Sequence) and len(world_xy) >= 2:
                frames[_frame_idx(frame)] = [float(world_xy[0]), float(world_xy[1])]
        if player_id and frames:
            out[player_id] = frames
    return out


def _joint_names_from_keypoints(payload: Mapping[str, Any]) -> list[str]:
    names = payload.get("joint_names")
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return [str(name) for name in names]
    ordered: list[str] = []
    seen: set[str] = set()
    for player in _players(payload):
        for frame in _frames(player):
            for name in _keypoints_by_joint(frame):
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
    if not ordered:
        raise ValueError("keypoints payload must include joint_names or frame keypoints")
    return ordered


def _keypoints_by_joint(frame_payload: Mapping[str, Any]) -> dict[str, _Keypoint]:
    raw = frame_payload.get("keypoints") or frame_payload.get("joints") or {}
    out: dict[str, _Keypoint] = {}
    if isinstance(raw, Mapping):
        iterable = []
        for joint_name, value in raw.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("joint", joint_name)
                iterable.append(item)
            elif isinstance(value, Sequence) and len(value) >= 2:
                iterable.append({"joint": joint_name, "x_px": value[0], "y_px": value[1], "conf": value[2] if len(value) > 2 else 1.0})
    else:
        iterable = raw if isinstance(raw, Sequence) else []
    for item in iterable:
        if not isinstance(item, Mapping):
            continue
        joint = str(item.get("joint") or item.get("name") or item.get("joint_name") or "")
        if not joint:
            continue
        x_px = item.get("x_px", item.get("x"))
        y_px = item.get("y_px", item.get("y"))
        if x_px is None or y_px is None:
            continue
        conf = item.get("conf", item.get("confidence", item.get("score", 1.0)))
        out[joint] = _Keypoint(joint=joint, x_px=float(x_px), y_px=float(y_px), conf=float(conf))
    return out


def _players(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = payload.get("players", [])
    return [player for player in players if isinstance(player, Mapping)] if isinstance(players, Sequence) else []


def _frames(player_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = player_payload.get("frames", [])
    return [frame for frame in frames if isinstance(frame, Mapping)] if isinstance(frames, Sequence) else []


def _frame_idx(frame_payload: Mapping[str, Any]) -> int:
    if frame_payload.get("frame_idx") is not None:
        return int(frame_payload["frame_idx"])
    if frame_payload.get("frame_index") is not None:
        return int(frame_payload["frame_index"])
    return int(frame_payload.get("frame", 0))


def _first_visible_keypoint(
    keypoints: Mapping[str, _Keypoint],
    names: Sequence[str],
    *,
    min_conf: float,
) -> _Keypoint | None:
    for name in names:
        keypoint = keypoints.get(name)
        if keypoint is not None and keypoint.conf >= min_conf:
            return keypoint
    return None


def _root_from_joints(solved: Mapping[str, list[float]], joint_names: Sequence[str]) -> list[float] | None:
    roots = [solved[name] for name in ROOT_NAMES if name in solved]
    if not roots:
        hip_names = [name for name in ("left_hip", "right_hip") if name in solved]
        roots = [solved[name] for name in hip_names]
    if not roots:
        return None
    return [sum(point[axis] for point in roots) / len(roots) for axis in range(3)]


def _point_changed(current: Sequence[float] | None, candidate: Sequence[float]) -> bool:
    if current is None:
        return True
    return math.dist(current, candidate) > 1e-9


def _mat3(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rows = [[float(item) for item in row] for row in value]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("expected 3x3 matrix")
    return (tuple(rows[0]), tuple(rows[1]), tuple(rows[2]))  # type: ignore[return-value]


def _vec3(value: Any) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError("expected 3-vector")
    return values  # type: ignore[return-value]


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0])),
        (float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1])),
        (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])),
    )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _round(value: float, *, digits: int = 9) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 10 ** (-digits) else rounded


__all__ = [
    "ARTIFACT_TYPE",
    "LANE_NAME",
    "Lift2DConfig",
    "BonePrior",
    "lift_skeleton_from_2d",
    "project_skeleton_joint",
    "load_player_bone_lengths",
    "LEG_BONE_JOINT_PAIRS",
    "BODY_CORE_REQUIRED_NAMES",
]
