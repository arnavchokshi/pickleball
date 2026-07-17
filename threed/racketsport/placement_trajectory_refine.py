"""Court-frame trajectory scoring and rigid placement refinement.

The refinement half of this module is intentionally added only after the
frozen scorer reproduces the immutable baseline.  Raw BODY/TRK payloads are
never modified in place.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.coordinates import (
    CoordinateSpace,
    homography_pixel_space,
    project_world_points,
    project_world_xy_points,
    resolve_homography_pixel_convention,
    resolve_world_coordinate_space,
)
from threed.racketsport.foot_contact import (
    build_body_skeleton_foot_contact_phases,
    contact_frames_from_skeleton3d,
    foot_contact_point,
    resolve_foot_joint_indices,
)
from threed.racketsport.schemas import CameraIntrinsics, CourtExtrinsics


PLACEMENT_REFINED_ARTIFACT_TYPE = "placement_trajectory_refined"
PLACEMENT_REFINED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlacementTrajectoryConfig:
    """One global, bounded configuration for all players and clips."""

    trk_weight: float = 24.0
    body_weight: float = 6.0
    plant_weight: float = 72.0
    smoothness_weight: float = 1.5
    huber_delta_m: float = 0.035
    max_xy_correction_m: float = 0.080
    z_plane_weight: float = 0.20
    z_body_weight: float = 20.0
    max_z_correction_m: float = 0.010
    irls_iterations: int = 8
    covariance_floor_m2: float = 0.000025
    covariance_ceiling_m2: float = 0.040

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def scaled(self, field: str, scale: float) -> "PlacementTrajectoryConfig":
        if field not in {"trk_weight", "plant_weight", "smoothness_weight"}:
            raise PlacementTrajectoryError(f"unsupported sensitivity weight: {field}")
        if not math.isfinite(scale) or scale <= 0.0:
            raise PlacementTrajectoryError("sensitivity scale must be finite and positive")
        return replace(self, **{field: float(getattr(self, field)) * scale})


class PlacementTrajectoryError(ValueError):
    """Base class for fail-closed placement trajectory errors."""


class MissingPlacementInputError(PlacementTrajectoryError):
    """A required placement input or field is absent."""


class MalformedPlacementInputError(PlacementTrajectoryError):
    """A placement input exists but violates its numeric/schema contract."""


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise MissingPlacementInputError(f"required input does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MalformedPlacementInputError(f"invalid JSON input {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedPlacementInputError(f"JSON input must be an object: {source}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_placement_slide(
    skeleton_payload: Mapping[str, Any],
    *,
    frozen_phases_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    tracks_payload: Mapping[str, Any],
    keypoints_2d_payload: Mapping[str, Any] | None = None,
    body_reference_payload: Mapping[str, Any] | None = None,
    clip: str | None = None,
) -> dict[str, Any]:
    """Score producer-rebuilt and frozen phase windows plus placement diagnostics."""

    _validate_skeleton(skeleton_payload)
    _validate_calibration(calibration_payload)
    _validate_tracks(tracks_payload)
    frozen_phases = _required_phases(frozen_phases_payload)
    producer = build_body_skeleton_foot_contact_phases(skeleton_payload, clip=clip)
    if producer.get("summary", {}).get("status") != "ran":
        raise MalformedPlacementInputError(
            f"frozen BODY phase producer failed: {producer.get('summary', {}).get('status')}"
        )
    accepted = _phase_window_metrics(skeleton_payload, _required_phases(producer))
    frozen = _phase_window_metrics(skeleton_payload, frozen_phases)
    reprojection = _reprojection_metrics(
        skeleton_payload,
        calibration_payload=calibration_payload,
        tracks_payload=tracks_payload,
        keypoints_2d_payload=keypoints_2d_payload,
    )
    disagreement = _disagreement_metrics(
        skeleton_payload,
        tracks_payload=tracks_payload,
        body_reference_payload=body_reference_payload,
        window_keys={
            (str(phase.get("player_id")), int(frame_index))
            for phase in frozen_phases
            for frame_index in phase["frame_indices"]
        },
    )
    return {
        "schema_version": 1,
        "artifact_type": "placement_slide_validation",
        "clip": str(clip or skeleton_payload.get("clip") or ""),
        "coordinate_space": CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M.value,
        "world_frame": "court_Z0",
        "distortion_state": _distortion_state(calibration_payload),
        "accepted_phase": accepted,
        "frozen_phase": frozen,
        "producer": {
            "phase_count": int(producer["phase_count"]),
            "rejected_phase_count": int(producer["rejected_phase_count"]),
            "rejected_reasons": dict(producer["summary"].get("rejected_reasons", {})),
            "accepted_phase_keys": [
                {
                    "player_id": str(phase.get("player_id")),
                    "foot": str(phase.get("foot")),
                    "start_frame_index": int(phase.get("start_frame_index")),
                    "end_frame_index": int(phase.get("end_frame_index")),
                }
                for phase in producer["phases"]
            ],
            "rejected_phases": [
                {
                    "player_id": str(phase.get("player_id")),
                    "foot": str(phase.get("foot")),
                    "start_frame_index": int(phase.get("start_frame_index")),
                    "end_frame_index": int(phase.get("end_frame_index")),
                    "rejection_reason": str(phase.get("rejection_reason")),
                }
                for phase in producer.get("rejected_phases", [])
            ],
            "source": "threed.racketsport.foot_contact.build_body_skeleton_foot_contact_phases",
        },
        "reprojection_px": reprojection,
        "disagreement_m": disagreement,
    }


def refine_placement_trajectory(
    skeleton_payload: Mapping[str, Any],
    *,
    tracks_payload: Mapping[str, Any],
    foot_contact_phases: Mapping[str, Any],
    config: PlacementTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Apply a robust, soft, rigid XYZ correction per BODY frame.

    XY factors combine TRK agreement, a BODY-placement prior, plant-window
    stationarity, and second-difference smoothness.  Z uses only a bounded
    soft sole-plane term and never clamps a joint or ankle to the floor.
    """

    cfg = config or PlacementTrajectoryConfig()
    _validate_config(cfg)
    _validate_skeleton(skeleton_payload)
    _validate_tracks(tracks_payload)
    phases = _required_phases(foot_contact_phases)
    output = copy.deepcopy(dict(skeleton_payload))
    output["schema_version"] = PLACEMENT_REFINED_SCHEMA_VERSION
    output["artifact_type"] = PLACEMENT_REFINED_ARTIFACT_TYPE
    output["coordinate_space"] = CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M.value
    output["world_frame"] = "court_Z0"
    output["preview_band"] = True
    output["VERIFIED"] = 0
    track_index = _track_index(tracks_payload)
    phase_index = _phase_index(phases)
    joint_names = [str(name) for name in output["joint_names"]]
    correction_values: list[float] = []
    covariance_values: list[float] = []
    plant_frame_count = 0
    player_summaries: dict[str, Any] = {}

    for player in output["players"]:
        player_id = str(player.get("id", player.get("player_id")))
        frames = sorted(player["frames"], key=lambda row: int(row.get("frame_idx", 0)))
        if not frames:
            raise MissingPlacementInputError(f"skeleton player {player_id} contains no frames")
        foot_indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0]["joints_world"]))
        factors, frame_terms = _build_xy_factors(
            player_id,
            frames,
            tracks=track_index,
            phases=phase_index.get(player_id, []),
            foot_indices=foot_indices,
            config=cfg,
        )
        corrections_xy, effective_weights, local_precision = _solve_robust_factors(
            len(frames), factors, config=cfg
        )
        correction_norms = np.linalg.norm(corrections_xy, axis=1)
        scale = np.ones_like(correction_norms)
        too_large = correction_norms > cfg.max_xy_correction_m
        scale[too_large] = cfg.max_xy_correction_m / correction_norms[too_large]
        corrections_xy = corrections_xy * scale[:, None]
        player_corrections: list[float] = []
        for ordinal, frame in enumerate(frames):
            frame_index = int(frame.get("frame_idx", ordinal))
            dz, z_provenance = _soft_z_correction(frame, foot_indices=foot_indices, config=cfg)
            dx, dy = (float(corrections_xy[ordinal, 0]), float(corrections_xy[ordinal, 1]))
            correction = [dx, dy, dz]
            _apply_rigid_translation(frame, correction)
            left_foot = _foot_point_from_payload(frame, foot_indices.left)
            right_foot = _foot_point_from_payload(frame, foot_indices.right)
            precision = max(float(local_precision[ordinal]), 1e-12)
            variance_xy = min(
                cfg.covariance_ceiling_m2,
                max(cfg.covariance_floor_m2, 1.0 / precision),
            )
            variance_z = min(
                cfg.covariance_ceiling_m2,
                max(cfg.covariance_floor_m2, 1.0 / (cfg.z_plane_weight + cfg.z_body_weight)),
            )
            covariance = [[variance_xy, 0.0, 0.0], [0.0, variance_xy, 0.0], [0.0, 0.0, variance_z]]
            phase_rows = frame_terms[ordinal]["plant_phases"]
            nominal = frame_terms[ordinal]["nominal_weights"]
            effective = {
                term: float(sum(effective_weights[index] for index in indices))
                for term, indices in frame_terms[ordinal]["factor_indices"].items()
            }
            magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
            correction_values.append(magnitude)
            player_corrections.append(magnitude)
            covariance_values.append(variance_xy)
            if phase_rows:
                plant_frame_count += 1
            frame["placement_trajectory_refinement"] = {
                "rigid_correction_xyz_m": correction,
                "correction_convention": "add_to_transl_world_and_every_joints_world_point",
                "correction_magnitude_m": magnitude,
                "refined_transl_world": list(frame["transl_world"]),
                "refined_foot_positions": {"left": list(left_foot), "right": list(right_foot)},
                "covariance_m2": covariance,
                "provenance": {
                    "plant_anchored": bool(phase_rows),
                    "plant_phases": phase_rows,
                    "evidence": {
                        term: {
                            "nominal_weight": float(nominal.get(term, 0.0)),
                            "effective_robust_weight": float(effective.get(term, 0.0)),
                        }
                        for term in ("trk", "body", "plant", "smoothness", "court_plane")
                    },
                    "z_soft_prior": z_provenance,
                },
            }
        player_summaries[player_id] = {
            "frame_count": len(frames),
            "plant_anchored_frame_count": sum(bool(row["plant_phases"]) for row in frame_terms),
            "correction_magnitude_m": _summary(player_corrections),
            "correction_max_m": max(player_corrections),
        }

    output["placement_trajectory_refinement"] = {
        "schema_version": PLACEMENT_REFINED_SCHEMA_VERSION,
        "artifact_type": PLACEMENT_REFINED_ARTIFACT_TYPE,
        "config": cfg.to_dict(),
        "coordinate_space": CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M.value,
        "world_frame": "court_Z0",
        "distortion_state": "not_applicable_no_image_transform_in_refiner",
        "preview_band": True,
        "VERIFIED": 0,
        "summary": {
            "player_count": len(output["players"]),
            "frame_count": len(correction_values),
            "plant_anchored_frame_count": plant_frame_count,
            "correction_magnitude_m": {
                **_summary(correction_values),
                "max": max(correction_values) if correction_values else 0.0,
            },
            "xy_covariance_variance_m2": {
                **_summary(covariance_values),
                "max": max(covariance_values) if covariance_values else 0.0,
            },
        },
        "players": player_summaries,
        "policy": {
            "raw_inputs_mutated": False,
            "rigid_translation_only": True,
            "root_relative_pose_unchanged": True,
            "plant_stationarity_soft_finite_weight": True,
            "robust_loss": "Huber IRLS on TRK, BODY, plant, and smoothness factors",
            "xy_correction_bounded": True,
            "court_plane_prior": "soft bounded residual reduction only",
            "sole_or_ankle_z_clamping": False,
            "covariance_method": "bounded diagonal inverse local robust factor precision approximation",
            "protected_eval_labels_used": False,
        },
    }
    return output


def _validate_config(config: PlacementTrajectoryConfig) -> None:
    weights = (
        config.trk_weight,
        config.body_weight,
        config.plant_weight,
        config.smoothness_weight,
        config.z_plane_weight,
        config.z_body_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise MalformedPlacementInputError("all placement weights must be finite and non-negative")
    if config.trk_weight <= 0.0 or config.body_weight <= 0.0 or config.plant_weight <= 0.0:
        raise MalformedPlacementInputError("TRK, BODY, and plant weights must be positive")
    if config.huber_delta_m <= 0.0 or config.max_xy_correction_m <= 0.0:
        raise MalformedPlacementInputError("Huber delta and max XY correction must be positive")
    if config.irls_iterations < 1:
        raise MalformedPlacementInputError("irls_iterations must be >= 1")


def _phase_index(phases: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for phase in phases:
        result.setdefault(str(phase.get("player_id")), []).append(phase)
    return result


def _build_xy_factors(
    player_id: str,
    frames: Sequence[Mapping[str, Any]],
    *,
    tracks: Mapping[tuple[str, int], Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    foot_indices: Any,
    config: PlacementTrajectoryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_indices = [int(frame.get("frame_idx", ordinal)) for ordinal, frame in enumerate(frames)]
    ordinal_by_frame = {frame_index: ordinal for ordinal, frame_index in enumerate(frame_indices)}
    roots = np.asarray([_finite_vector(frame["transl_world"], 3, "transl_world")[:2] for frame in frames])
    factors: list[dict[str, Any]] = []
    terms = [
        {
            "plant_phases": [],
            "nominal_weights": {"trk": 0.0, "body": 0.0, "plant": 0.0, "smoothness": 0.0, "court_plane": config.z_plane_weight},
            "factor_indices": {"trk": [], "body": [], "plant": [], "smoothness": []},
        }
        for _ in frames
    ]

    def add_factor(coefficients: Mapping[int, float], target: Sequence[float], weight: float, term: str) -> None:
        factor_index = len(factors)
        factors.append(
            {
                "coefficients": dict(coefficients),
                "target": np.asarray(
                    _finite_vector(target.tolist() if isinstance(target, np.ndarray) else target, 2, f"{term} factor target"),
                    dtype=float,
                ),
                "weight": float(weight),
                "term": term,
            }
        )
        for ordinal in coefficients:
            terms[ordinal]["factor_indices"][term].append(factor_index)
            terms[ordinal]["nominal_weights"][term] += float(weight)

    for ordinal, frame in enumerate(frames):
        key = (player_id, frame_indices[ordinal])
        track = tracks.get(key)
        if track is None:
            raise MissingPlacementInputError(f"missing track observation for skeleton frame {key}")
        track_conf = _finite_confidence(track.get("conf"), f"track {key}.conf")
        body_conf = float(np.mean([_finite_confidence(value, f"skeleton {key}.joint_conf") for value in frame["joint_conf"]]))
        target = np.asarray(_track_world_xy(track), dtype=float) - roots[ordinal]
        add_factor({ordinal: 1.0}, target, config.trk_weight * max(track_conf, 0.05) ** 2, "trk")
        add_factor({ordinal: 1.0}, [0.0, 0.0], config.body_weight * max(body_conf, 0.05) ** 2, "body")

    plant_frame_indices = {
        int(frame_index)
        for phase in phases
        for frame_index in phase["frame_indices"]
    }
    for ordinal in range(1, len(frames) - 1):
        if frame_indices[ordinal] - frame_indices[ordinal - 1] != 1 or frame_indices[ordinal + 1] - frame_indices[ordinal] != 1:
            continue
        if not all(frame_indices[index] in plant_frame_indices for index in (ordinal - 1, ordinal, ordinal + 1)):
            continue
        add_factor(
            {ordinal - 1: 1.0, ordinal: -2.0, ordinal + 1: 1.0},
            [0.0, 0.0],
            config.smoothness_weight,
            "smoothness",
        )

    for phase_ordinal, phase in enumerate(phases):
        phase_ordinals: list[int] = []
        points: list[list[float]] = []
        foot = str(phase["foot"])
        indices = foot_indices.for_foot(foot)
        for frame_index in phase["frame_indices"]:
            ordinal = ordinal_by_frame.get(int(frame_index))
            if ordinal is None:
                raise MissingPlacementInputError(
                    f"phase {player_id}:{foot}:{phase_ordinal} references missing frame {frame_index}"
                )
            phase_ordinals.append(ordinal)
            points.append(list(_foot_point_from_payload(frames[ordinal], indices)[:2]))
        point_array = np.asarray(points, dtype=float)
        anchor = np.median(point_array, axis=0)
        phase_conf = _finite_confidence(phase.get("min_confidence"), f"phase {phase_ordinal}.min_confidence")
        agreement = _finite_confidence(
            phase.get("assignment_evidence", {}).get("body_detector_agreement"),
            f"phase {phase_ordinal}.assignment_evidence.body_detector_agreement",
        )
        weight = config.plant_weight * phase_conf * agreement
        phase_descriptor = {
            "foot": foot,
            "start_frame_index": int(phase["frame_indices"][0]),
            "end_frame_index": int(phase["frame_indices"][-1]),
            "confidence": phase_conf,
            "soft_anchor_xy_m": anchor.tolist(),
        }
        for ordinal, point in zip(phase_ordinals, point_array, strict=True):
            add_factor({ordinal: 1.0}, anchor - point, weight, "plant")
            terms[ordinal]["plant_phases"].append(phase_descriptor)
    return factors, terms


def _solve_robust_factors(
    frame_count: int,
    factors: Sequence[Mapping[str, Any]],
    *,
    config: PlacementTrajectoryConfig,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr
    except ImportError as exc:
        raise MissingPlacementInputError("scipy is required for placement trajectory refinement") from exc
    solution = np.zeros((frame_count, 2), dtype=float)
    effective = [float(factor["weight"]) for factor in factors]
    for _ in range(config.irls_iterations):
        residual_norms = []
        for factor in factors:
            predicted = sum(float(coefficient) * solution[index] for index, coefficient in factor["coefficients"].items())
            residual_norms.append(float(np.linalg.norm(predicted - factor["target"])))
        effective = [
            float(factor["weight"]) * min(1.0, config.huber_delta_m / max(residual, 1e-12))
            for factor, residual in zip(factors, residual_norms, strict=True)
        ]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros((len(factors), 2), dtype=float)
        for row, (factor, weight) in enumerate(zip(factors, effective, strict=True)):
            root_weight = math.sqrt(max(weight, 1e-12))
            for index, coefficient in factor["coefficients"].items():
                rows.append(row)
                cols.append(int(index))
                data.append(root_weight * float(coefficient))
            rhs[row] = root_weight * factor["target"]
        matrix = coo_matrix((data, (rows, cols)), shape=(len(factors), frame_count)).tocsr()
        candidate = np.column_stack(
            [lsqr(matrix, rhs[:, axis], atol=1e-12, btol=1e-12, iter_lim=max(100, frame_count * 3))[0] for axis in range(2)]
        )
        if np.max(np.abs(candidate - solution)) < 1e-10:
            solution = candidate
            break
        solution = candidate
    local_precision = np.zeros(frame_count, dtype=float)
    for factor, weight in zip(factors, effective, strict=True):
        for index, coefficient in factor["coefficients"].items():
            local_precision[int(index)] += float(weight) * float(coefficient) ** 2
    return solution, effective, local_precision


def _soft_z_correction(
    frame: Mapping[str, Any],
    *,
    foot_indices: Any,
    config: PlacementTrajectoryConfig,
) -> tuple[float, dict[str, Any]]:
    joints = frame["joints_world"]
    sole_z = min(float(joints[index][2]) for index in foot_indices.all())
    gain = config.z_plane_weight / (config.z_plane_weight + config.z_body_weight)
    unbounded = -sole_z * gain
    correction = max(-config.max_z_correction_m, min(config.max_z_correction_m, unbounded))
    return correction, {
        "active": config.z_plane_weight > 0.0,
        "sole_z_before_m": sole_z,
        "sole_z_after_m": sole_z + correction,
        "gain": gain,
        "bounded": abs(unbounded) > config.max_z_correction_m,
        "clamped_to_plane": False,
    }


def _apply_rigid_translation(frame: dict[str, Any], correction: Sequence[float]) -> None:
    frame["transl_world"] = [float(frame["transl_world"][axis]) + float(correction[axis]) for axis in range(3)]
    frame["joints_world"] = [
        [float(joint[axis]) + float(correction[axis]) for axis in range(3)]
        for joint in frame["joints_world"]
    ]


def _foot_point_from_payload(frame: Mapping[str, Any], indices: Sequence[int]) -> tuple[float, float, float]:
    points = [_finite_vector(frame["joints_world"][index], 3, f"joints_world[{index}]") for index in indices]
    min_z = min(point[2] for point in points)
    low = [point for point in points if point[2] <= min_z + 0.025]
    return (
        float(sum(point[0] for point in low) / len(low)),
        float(sum(point[1] for point in low) / len(low)),
        float(sum(point[2] for point in low) / len(low)),
    )


def _validate_skeleton(payload: Mapping[str, Any]) -> None:
    try:
        resolve_world_coordinate_space(payload)
    except (TypeError, ValueError) as exc:
        raise MalformedPlacementInputError(f"invalid skeleton coordinate declaration: {exc}") from exc
    players = payload.get("players")
    joint_names = payload.get("joint_names")
    if not isinstance(players, list) or not players:
        raise MissingPlacementInputError("skeleton.players must be a non-empty list")
    if not isinstance(joint_names, list) or not joint_names:
        raise MissingPlacementInputError("skeleton.joint_names must be a non-empty list")
    for player_ordinal, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise MalformedPlacementInputError(f"skeleton.players[{player_ordinal}] must be an object")
        frames = player.get("frames")
        if not isinstance(frames, list) or not frames:
            raise MissingPlacementInputError(f"skeleton.players[{player_ordinal}].frames must be non-empty")
        for frame_ordinal, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise MalformedPlacementInputError(
                    f"skeleton.players[{player_ordinal}].frames[{frame_ordinal}] must be an object"
                )
            _finite_vector(frame.get("transl_world"), 3, f"skeleton frame {frame_ordinal}.transl_world")
            joints = frame.get("joints_world")
            if not isinstance(joints, list) or not joints:
                raise MissingPlacementInputError(f"skeleton frame {frame_ordinal}.joints_world must be non-empty")
            for joint_ordinal, joint in enumerate(joints):
                _finite_vector(joint, 3, f"skeleton frame {frame_ordinal}.joints_world[{joint_ordinal}]")
            conf = frame.get("joint_conf")
            if not isinstance(conf, list) or len(conf) != len(joints):
                raise MissingPlacementInputError(f"skeleton frame {frame_ordinal}.joint_conf must match joints_world")
            for value in conf:
                _finite_confidence(value, f"skeleton frame {frame_ordinal}.joint_conf")


def _validate_calibration(payload: Mapping[str, Any]) -> None:
    if "homography" not in payload or "extrinsics" not in payload or "intrinsics" not in payload:
        raise MissingPlacementInputError("calibration requires homography, extrinsics, and intrinsics")
    try:
        homography = np.asarray(payload["homography"], dtype=float)
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            raise ValueError("homography must be finite 3x3")
        extrinsics = payload["extrinsics"]
        CourtExtrinsics.model_validate(
            {"R": extrinsics["R"], "t": extrinsics["t"], "camera_height_m": extrinsics["camera_height_m"]}
        )
        CameraIntrinsics.model_validate(payload["intrinsics"])
        resolve_homography_pixel_convention(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedPlacementInputError(f"invalid calibration: {exc}") from exc


def _validate_tracks(payload: Mapping[str, Any]) -> None:
    players = payload.get("players")
    if not isinstance(players, list) or not players:
        raise MissingPlacementInputError("tracks.players must be a non-empty list")
    seen = 0
    for player in players:
        if not isinstance(player, Mapping):
            raise MalformedPlacementInputError("track player rows must be objects")
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                raise MalformedPlacementInputError("track frame rows must be objects")
            _track_world_xy(frame)
            _finite_confidence(frame.get("conf"), "track frame conf")
            seen += 1
    if seen == 0:
        raise MissingPlacementInputError("tracks contain no frames")


def _required_phases(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise MissingPlacementInputError("phase source must contain at least one phase")
    for ordinal, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise MalformedPlacementInputError(f"phases[{ordinal}] must be an object")
        frames = phase.get("frame_indices")
        if not isinstance(frames, list) or not frames:
            raise MissingPlacementInputError(f"phases[{ordinal}].frame_indices must be non-empty")
        if str(phase.get("foot")) not in {"left", "right"}:
            raise MalformedPlacementInputError(f"phases[{ordinal}].foot must be left or right")
        _finite_confidence(phase.get("min_confidence"), f"phases[{ordinal}].min_confidence")
    return phases


def _phase_window_metrics(
    skeleton_payload: Mapping[str, Any],
    phases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frames, joint_names = contact_frames_from_skeleton3d(skeleton_payload)
    if not frames:
        raise MissingPlacementInputError("skeleton contains no contact-compatible frames")
    try:
        indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    except ValueError as exc:
        raise MalformedPlacementInputError(f"cannot resolve foot joints: {exc}") from exc
    by_key = {(str(frame.player_id), int(frame.frame_index)): frame for frame in frames}
    rows: list[dict[str, Any]] = []
    for ordinal, phase in enumerate(phases):
        player_id = str(phase.get("player_id"))
        foot = str(phase["foot"])
        points: list[tuple[float, float, float]] = []
        for frame_index in phase["frame_indices"]:
            key = (player_id, int(frame_index))
            if key not in by_key:
                raise MissingPlacementInputError(f"phase {ordinal} references missing skeleton frame {key}")
            points.append(foot_contact_point(by_key[key], indices.for_foot(foot)))
        anchor = points[0]
        slide_m = max(math.hypot(point[0] - anchor[0], point[1] - anchor[1]) for point in points)
        rows.append(
            {
                "player_id": player_id,
                "foot": foot,
                "start_frame_index": int(phase["frame_indices"][0]),
                "end_frame_index": int(phase["frame_indices"][-1]),
                "frame_count": len(points),
                "slide_m": slide_m,
                "min_confidence": float(phase["min_confidence"]),
            }
        )
    values = [float(row["slide_m"]) for row in rows]
    return {
        "phase_count": len(rows),
        "phase_frame_count": sum(int(row["frame_count"]) for row in rows),
        "max_slide_m": max(values),
        "p95_slide_m": _percentile(values, 95.0),
        "median_phase_slide_m": float(median(values)),
        "per_phase": rows,
    }


def _reprojection_metrics(
    skeleton_payload: Mapping[str, Any],
    *,
    calibration_payload: Mapping[str, Any],
    tracks_payload: Mapping[str, Any],
    keypoints_2d_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    convention = resolve_homography_pixel_convention(calibration_payload)
    raster_space = homography_pixel_space(convention)
    homography = calibration_payload["homography"]
    extrinsics_payload = calibration_payload["extrinsics"]
    extrinsics = CourtExtrinsics.model_validate(
        {
            "R": extrinsics_payload["R"],
            "t": extrinsics_payload["t"],
            "camera_height_m": extrinsics_payload["camera_height_m"],
        }
    )
    intrinsics = CameraIntrinsics.model_validate(calibration_payload["intrinsics"])
    track_index = _track_index(tracks_payload)
    keypoint_index = _keypoint_index(keypoints_2d_payload)
    frames, joint_names = contact_frames_from_skeleton3d(skeleton_payload)
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    foot_errors: list[float] = []
    root_errors: list[float] = []
    fallback_count = 0
    for frame in frames:
        key = (str(frame.player_id), int(frame.frame_index))
        track = track_index.get(key)
        if track is None:
            continue
        left = foot_contact_point(frame, indices.left)
        right = foot_contact_point(frame, indices.right)
        foot_xy = [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5]
        projected_foot = project_world_xy_points(
            homography,
            [foot_xy],
            input_space=CoordinateSpace.WORLD_XY_HOMOGRAPHY_M,
            output_space=raster_space,
            homography_space=raster_space,
        )[0]
        source_frame = frame.source or {}
        root = _finite_vector(source_frame.get("transl_world"), 3, "skeleton transl_world")
        projected_root = project_world_points(
            extrinsics,
            intrinsics,
            [root],
            input_space=CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
            output_space=CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
            reference_space=raster_space,
        )[0]
        observed_foot = keypoint_index.get(key)
        bbox = track.get("bbox")
        if observed_foot is None:
            if not _valid_bbox(bbox):
                continue
            observed_foot = [(float(bbox[0]) + float(bbox[2])) * 0.5, float(bbox[3])]
            fallback_count += 1
        foot_errors.append(_distance2(projected_foot, observed_foot))
        if _valid_bbox(bbox):
            observed_root = [(float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5]
            root_errors.append(_distance2(projected_root, observed_root))
    combined = [*foot_errors, *root_errors]
    if not combined:
        raise MissingPlacementInputError("no reprojection evidence overlaps skeleton frames")
    return {
        "count": len(combined),
        "median": float(median(combined)),
        "p95": _percentile(combined, 95.0),
        "foot_midpoint": _summary(foot_errors),
        "root": _summary(root_errors),
        "foot_bbox_fallback_count": fallback_count,
        "projection_model": "typed court homography for foot midpoint; typed ideal pinhole for root",
        "evidence_space": raster_space.value,
    }


def _disagreement_metrics(
    skeleton_payload: Mapping[str, Any],
    *,
    tracks_payload: Mapping[str, Any],
    body_reference_payload: Mapping[str, Any] | None,
    window_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    track_index = _track_index(tracks_payload)
    current = _body_footpoint_index(skeleton_payload)
    reference = _body_footpoint_index(body_reference_payload) if body_reference_payload is not None else None
    trk_values: list[float] = []
    body_values: list[float] = []
    window_trk_values: list[float] = []
    window_body_values: list[float] = []
    for key, foot_xy in current.items():
        track = track_index.get(key)
        if track is None:
            continue
        trk_distance = _distance2(foot_xy, _track_world_xy(track))
        trk_values.append(trk_distance)
        if key in window_keys:
            window_trk_values.append(trk_distance)
        if reference is not None and key in reference:
            body_distance = _distance2(foot_xy, reference[key])
            body_values.append(body_distance)
            if key in window_keys:
                window_body_values.append(body_distance)
    if not trk_values:
        raise MissingPlacementInputError("no TRK/BODY frames overlap for disagreement metrics")
    return {
        "trk_vs_payload": _summary(trk_values),
        "payload_vs_body_reference": _summary(body_values) if reference is not None else None,
        "frozen_windows": {
            "trk_vs_payload": _summary(window_trk_values),
            "payload_vs_body_reference": _summary(window_body_values) if reference is not None else None,
        },
    }


def _body_footpoint_index(payload: Mapping[str, Any] | None) -> dict[tuple[str, int], list[float]]:
    if payload is None:
        return {}
    frames, names = contact_frames_from_skeleton3d(payload)
    if not frames:
        return {}
    indices = resolve_foot_joint_indices(names, joint_count=len(frames[0].joints_world))
    result: dict[tuple[str, int], list[float]] = {}
    for frame in frames:
        left = foot_contact_point(frame, indices.left)
        right = foot_contact_point(frame, indices.right)
        result[(str(frame.player_id), int(frame.frame_index))] = [
            (left[0] + right[0]) * 0.5,
            (left[1] + right[1]) * 0.5,
        ]
    return result


def _track_index(payload: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for player in payload.get("players", []):
        player_id = str(player.get("id", player.get("player_id")))
        for ordinal, frame in enumerate(player.get("frames", [])):
            frame_index = int(frame.get("frame_idx", ordinal))
            result[(player_id, frame_index)] = frame
    return result


def _keypoint_index(payload: Mapping[str, Any] | None) -> dict[tuple[str, int], list[float]]:
    result: dict[tuple[str, int], list[float]] = {}
    if payload is None:
        return result
    players = payload.get("players")
    if not isinstance(players, list):
        raise MalformedPlacementInputError("keypoints_2d.players must be a list")
    for player in players:
        player_id = str(player.get("id", player.get("player_id")))
        for ordinal, frame in enumerate(player.get("frames", [])):
            points = []
            weights = []
            for point in frame.get("keypoints", []):
                if not isinstance(point, Mapping) or point.get("name") not in {
                    "left_ankle", "right_ankle", "left_toe", "right_toe", "left_heel", "right_heel"
                }:
                    continue
                xy = _finite_vector(point.get("xy_px"), 2, "keypoint.xy_px")
                conf = _finite_confidence(point.get("conf"), "keypoint.conf")
                points.append(xy)
                weights.append(max(conf, 1e-6))
            if points:
                array = np.asarray(points, dtype=float)
                result[(player_id, int(frame.get("frame_idx", ordinal)))] = np.average(
                    array, axis=0, weights=np.asarray(weights, dtype=float)
                ).tolist()
    return result


def _track_world_xy(frame: Mapping[str, Any]) -> list[float]:
    value = frame.get("fused_world_xy")
    if value is None:
        value = frame.get("world_xy")
    if value is None:
        raise MissingPlacementInputError("track frame requires fused_world_xy or world_xy")
    return _finite_vector(value, 2, "track world_xy")


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise MissingPlacementInputError(f"{name} must be a length-{length} vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise MalformedPlacementInputError(f"{name} must contain finite values")
    return result


def _finite_confidence(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MissingPlacementInputError(f"{name} is required") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise MalformedPlacementInputError(f"{name} must be finite and in [0, 1]")
    return result


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return False
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in box) and box[2] > box[0] and box[3] > box[1]


def _distance2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile)) if values else 0.0


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": float(median(values)) if values else 0.0,
        "p95": _percentile(values, 95.0),
    }


def _distortion_state(calibration_payload: Mapping[str, Any]) -> dict[str, Any]:
    convention = resolve_homography_pixel_convention(calibration_payload)
    dist = calibration_payload.get("intrinsics", {}).get("dist", [])
    return {
        "homography_pixel_convention": convention,
        "declared_evidence_space": homography_pixel_space(convention).value,
        "coefficients_present": bool(dist),
        "homography_projection_applies_distortion": False,
        "root_projection": "ideal_pinhole_undistorted_native",
    }
