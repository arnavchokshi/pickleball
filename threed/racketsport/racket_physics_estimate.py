"""Preview-only racket pose estimates from ball impulse and wrist motion.

This module deliberately does not write or validate canonical ``racket_pose.json``.
It produces a physics-derived estimate artifact for review while the RKT gate
remains fail-closed until true paddle pose GT exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .court_calibration import project_world_points
from .schemas import CourtCalibration
from .skeleton3d import semanticize_skeleton_payload


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_racket_pose_estimate"
SOURCE = "physics_delta_v_wrist_swing_preview"
DEFAULT_RESTITUTION_RANGE = (0.55, 0.90)
DEFAULT_VELOCITY_WINDOW_S = 0.14
DEFAULT_MAX_SAMPLE_GAP_S = 0.12
DEFAULT_MIN_VELOCITY_VECTORS_PER_SIDE = 2
DEFAULT_MIN_DELTA_V_MPS = 0.35
DEFAULT_BALL_POSITION_NOISE_M = 0.05
DEFAULT_WRIST_TO_PADDLE_CENTER_M = 0.15
DEFAULT_MAX_CONTACT_TO_WRIST_M = 0.75
DEFAULT_MAX_WRIST_TIME_GAP_S = 0.18
DEFAULT_MIN_JOINT_CONFIDENCE = 0.20


@dataclass(frozen=True)
class BallSample:
    t: float
    frame: int | None
    world_xyz: tuple[float, float, float]
    confidence: float | None = None
    approximate: bool = False


@dataclass(frozen=True)
class VelocityEstimate:
    vector: tuple[float, float, float]
    vectors: tuple[tuple[float, float, float], ...]
    median_dt_s: float
    scatter_mps: float


@dataclass(frozen=True)
class WristSample:
    t: float
    frame: int | None
    player_id: int | str
    side: str
    wrist: tuple[float, float, float]
    elbow: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True)
class InterpolatedWrist:
    player_id: int | str
    side: str
    wrist: tuple[float, float, float]
    elbow: tuple[float, float, float]
    wrist_velocity: tuple[float, float, float] | None
    confidence: float
    time_gap_s: float
    hint: Mapping[str, Any] | None = None


def build_racket_physics_estimate(
    *,
    clip_id: str,
    contact_windows: Mapping[str, Any],
    ball_track: Mapping[str, Any],
    skeleton3d: Mapping[str, Any] | None = None,
    wrist_peaks: Mapping[str, Any] | None = None,
    contact_source_path: str | Path | None = None,
    ball_source_path: str | Path | None = None,
    skeleton_source_path: str | Path | None = None,
    wrist_peaks_source_path: str | Path | None = None,
    restitution_range: tuple[float, float] = DEFAULT_RESTITUTION_RANGE,
    velocity_window_s: float = DEFAULT_VELOCITY_WINDOW_S,
    max_sample_gap_s: float = DEFAULT_MAX_SAMPLE_GAP_S,
    min_velocity_vectors_per_side: int = DEFAULT_MIN_VELOCITY_VECTORS_PER_SIDE,
    min_delta_v_mps: float = DEFAULT_MIN_DELTA_V_MPS,
    ball_position_noise_m: float = DEFAULT_BALL_POSITION_NOISE_M,
    wrist_to_paddle_center_m: float = DEFAULT_WRIST_TO_PADDLE_CENTER_M,
    max_contact_to_wrist_m: float = DEFAULT_MAX_CONTACT_TO_WRIST_M,
    max_wrist_time_gap_s: float = DEFAULT_MAX_WRIST_TIME_GAP_S,
    min_joint_confidence: float = DEFAULT_MIN_JOINT_CONFIDENCE,
) -> dict[str, Any]:
    """Build a preview-only racket pose estimate artifact.

    Each contact requires enough 3D ball samples on both sides to estimate
    inbound/outbound velocities and a plausible wrist/forearm constraint near
    contact. Missing inputs skip that contact rather than fabricating a pose.
    """

    _validate_restitution_range(restitution_range)
    if velocity_window_s <= 0.0:
        raise ValueError("velocity_window_s must be positive")
    if max_sample_gap_s <= 0.0:
        raise ValueError("max_sample_gap_s must be positive")
    if min_velocity_vectors_per_side < 1:
        raise ValueError("min_velocity_vectors_per_side must be positive")
    if min_delta_v_mps <= 0.0:
        raise ValueError("min_delta_v_mps must be positive")
    if ball_position_noise_m < 0.0:
        raise ValueError("ball_position_noise_m must be non-negative")
    if wrist_to_paddle_center_m < 0.0:
        raise ValueError("wrist_to_paddle_center_m must be non-negative")
    if max_contact_to_wrist_m <= 0.0:
        raise ValueError("max_contact_to_wrist_m must be positive")
    if max_wrist_time_gap_s <= 0.0:
        raise ValueError("max_wrist_time_gap_s must be positive")

    contacts = _contact_events(contact_windows)
    ball_samples, ball_parse_warnings = _ball_samples(ball_track)
    wrist_index, wrist_warnings = _wrist_index(
        skeleton3d,
        min_joint_confidence=min_joint_confidence,
    )
    wrist_hints = _wrist_hints(wrist_peaks)

    estimates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    warning_set = {
        "physics_derived_preview_not_gate_verified",
        "never_write_canonical_racket_pose_json",
        *ball_parse_warnings,
        *wrist_warnings,
    }

    for contact_index, contact in enumerate(contacts):
        estimate, skip_reason, skip_details = _estimate_one_contact(
            contact_index=contact_index,
            contact=contact,
            ball_samples=ball_samples,
            wrist_index=wrist_index,
            wrist_hints=wrist_hints,
            restitution_range=restitution_range,
            velocity_window_s=velocity_window_s,
            max_sample_gap_s=max_sample_gap_s,
            min_velocity_vectors_per_side=min_velocity_vectors_per_side,
            min_delta_v_mps=min_delta_v_mps,
            ball_position_noise_m=ball_position_noise_m,
            wrist_to_paddle_center_m=wrist_to_paddle_center_m,
            max_contact_to_wrist_m=max_contact_to_wrist_m,
            max_wrist_time_gap_s=max_wrist_time_gap_s,
        )
        if estimate is None:
            skipped.append(
                {
                    "contact_index": contact_index,
                    "t": _round(contact["t"], 6),
                    "frame": contact.get("frame"),
                    "reason": skip_reason,
                    "details": skip_details,
                }
            )
            continue
        estimates.append(estimate)
        if estimate["ball_samples"]["approximate_fraction"] > 0.0:
            warning_set.add("court_plane_or_approximate_ball_source_uncertainty")

    blockers = _blockers(skipped, estimate_count=len(estimates), wrist_index=wrist_index, ball_samples=ball_samples)
    summary = _summary(
        contacts=contacts,
        estimates=estimates,
        skipped=skipped,
        ball_samples=ball_samples,
        wrist_index=wrist_index,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(clip_id),
        "status": "preview" if estimates else "blocked",
        "source": SOURCE,
        "physics_derived": True,
        "trusted_for_rkt_promotion": False,
        "never_canonical_racket_pose": True,
        "canonical_output_forbidden": "racket_pose.json",
        "contact_source_path": str(contact_source_path or ""),
        "ball_source_path": str(ball_source_path or ""),
        "skeleton_source_path": str(skeleton_source_path or ""),
        "wrist_peaks_source_path": str(wrist_peaks_source_path or ""),
        "parameters": {
            "restitution_range": [_round(restitution_range[0], 6), _round(restitution_range[1], 6)],
            "velocity_window_s": _round(velocity_window_s, 6),
            "max_sample_gap_s": _round(max_sample_gap_s, 6),
            "min_velocity_vectors_per_side": int(min_velocity_vectors_per_side),
            "min_delta_v_mps": _round(min_delta_v_mps, 6),
            "ball_position_noise_m": _round(ball_position_noise_m, 6),
            "wrist_to_paddle_center_m": _round(wrist_to_paddle_center_m, 6),
            "max_contact_to_wrist_m": _round(max_contact_to_wrist_m, 6),
            "max_wrist_time_gap_s": _round(max_wrist_time_gap_s, 6),
            "min_joint_confidence": _round(min_joint_confidence, 6),
        },
        "summary": summary,
        "blockers": blockers,
        "warnings": sorted(warning_set),
        "estimates": estimates,
        "skipped_contacts": skipped,
    }


def build_racket_physics_estimate_from_files(
    *,
    clip_id: str,
    contact_windows_path: str | Path,
    ball_track_path: str | Path,
    skeleton3d_path: str | Path | None = None,
    wrist_peaks_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    contact_windows = _read_json_object(contact_windows_path, "contact_windows")
    ball_track = _read_json_object(ball_track_path, "ball_track")
    skeleton3d = _read_json_object(skeleton3d_path, "skeleton3d") if skeleton3d_path is not None else None
    wrist_peaks = _read_json_object(wrist_peaks_path, "wrist_peaks") if wrist_peaks_path is not None else None
    return build_racket_physics_estimate(
        clip_id=clip_id,
        contact_windows=contact_windows,
        ball_track=ball_track,
        skeleton3d=skeleton3d,
        wrist_peaks=wrist_peaks,
        contact_source_path=contact_windows_path,
        ball_source_path=ball_track_path,
        skeleton_source_path=skeleton3d_path,
        wrist_peaks_source_path=wrist_peaks_path,
        **kwargs,
    )


def write_racket_physics_estimate(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_racket_physics_estimate_overlays(
    *,
    video_path: str | Path,
    court_calibration: CourtCalibration | Mapping[str, Any],
    estimate_artifact: Mapping[str, Any],
    output_dir: str | Path,
    max_overlays: int = 10,
) -> dict[str, Any]:
    """Render qualitative PNG overlays for human review.

    The overlays are review artifacts only. They show a paddle-face disc at the
    estimated pose and a short normal arrow projected into the source frame.
    """

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        return _overlay_blocked(video_path, output_dir, "missing_cv2", str(exc))

    if max_overlays <= 0:
        raise ValueError("max_overlays must be positive")

    parsed_calibration = (
        court_calibration
        if isinstance(court_calibration, CourtCalibration)
        else CourtCalibration.model_validate(court_calibration)
    )
    video = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return _overlay_blocked(video_path, output_dir, "cannot_open_video", f"cannot open video: {video}")

    rendered: list[dict[str, Any]] = []
    estimates = [
        estimate
        for estimate in estimate_artifact.get("estimates", [])
        if isinstance(estimate, Mapping) and estimate.get("frame") is not None
    ][:max_overlays]
    try:
        for estimate in estimates:
            frame_index = int(estimate["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                rendered.append({"frame": frame_index, "status": "skipped", "reason": "frame_read_failed"})
                continue
            draw_status = _draw_estimate_overlay(cv2, frame, parsed_calibration, estimate)
            if draw_status["status"] != "rendered":
                rendered.append({"frame": frame_index, **draw_status})
                continue
            output_path = out_dir / f"{estimate_artifact.get('clip_id', 'clip')}_contact_{int(estimate['contact_index']):02d}_frame_{frame_index:06d}.png"
            cv2.imwrite(str(output_path), frame)
            rendered.append(
                {
                    "frame": frame_index,
                    "contact_index": int(estimate["contact_index"]),
                    "status": "rendered",
                    "path": str(output_path),
                    "normal_angle_bound_deg": estimate.get("uncertainty", {}).get("normal_angle_bound_deg"),
                }
            )
    finally:
        cap.release()

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_pose_estimate_overlays",
        "status": "rendered" if any(item.get("status") == "rendered" for item in rendered) else "blocked",
        "qualitative_status": "review_only_not_gate_verified",
        "video_path": str(video),
        "output_dir": str(out_dir),
        "requested_overlay_count": len(estimates),
        "rendered_overlay_count": sum(1 for item in rendered if item.get("status") == "rendered"),
        "items": rendered,
        "warnings": ["physics_estimate_overlay_not_gate_verified"],
    }
    (out_dir / "racket_pose_estimate_overlay_index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _estimate_one_contact(
    *,
    contact_index: int,
    contact: Mapping[str, Any],
    ball_samples: Sequence[BallSample],
    wrist_index: Mapping[tuple[int | str, str], Sequence[WristSample]],
    wrist_hints: Sequence[Mapping[str, Any]],
    restitution_range: tuple[float, float],
    velocity_window_s: float,
    max_sample_gap_s: float,
    min_velocity_vectors_per_side: int,
    min_delta_v_mps: float,
    ball_position_noise_m: float,
    wrist_to_paddle_center_m: float,
    max_contact_to_wrist_m: float,
    max_wrist_time_gap_s: float,
) -> tuple[dict[str, Any] | None, str, Mapping[str, Any]]:
    t = float(contact["t"])
    frame = contact.get("frame")
    ball_motion = _ball_motion_around_contact(
        ball_samples,
        t=t,
        velocity_window_s=velocity_window_s,
        max_sample_gap_s=max_sample_gap_s,
        min_velocity_vectors_per_side=min_velocity_vectors_per_side,
    )
    if ball_motion is None:
        return None, "insufficient_ball_velocity_samples", {
            "ball_world_sample_count": len(ball_samples),
            "velocity_window_s": velocity_window_s,
        }

    v_in = ball_motion["v_in"].vector
    v_out = ball_motion["v_out"].vector
    delta_v = _sub(v_out, v_in)
    delta_norm = _norm(delta_v)
    if delta_norm < min_delta_v_mps:
        return None, "insufficient_ball_velocity_change", {"delta_v_mps": _round(delta_norm, 6)}
    normal = _normalize(delta_v)
    if _dot(normal, v_out) < 0.0:
        normal = _scale(normal, -1.0)
    contact_point = _interpolate_ball_position(ball_samples, t=t, max_gap_s=max_sample_gap_s)
    if contact_point is None:
        return None, "missing_ball_contact_position", {"t": _round(t, 6)}

    selected_wrist = _select_wrist(
        wrist_index,
        wrist_hints,
        t=t,
        contact_point=contact_point,
        player_id=contact.get("player_id"),
        wrist_to_paddle_center_m=wrist_to_paddle_center_m,
        max_contact_to_wrist_m=max_contact_to_wrist_m,
        max_wrist_time_gap_s=max_wrist_time_gap_s,
    )
    if selected_wrist is None:
        return None, "missing_plausible_wrist_constraint", {
            "candidate_wrist_track_count": len(wrist_index),
            "max_contact_to_wrist_m": _round(max_contact_to_wrist_m, 6),
            "max_wrist_time_gap_s": _round(max_wrist_time_gap_s, 6),
        }

    forearm_dir = _normalize(_sub(selected_wrist.wrist, selected_wrist.elbow))
    position = _add(selected_wrist.wrist, _scale(forearm_dir, wrist_to_paddle_center_m))
    wrist_distance = _distance(contact_point, selected_wrist.wrist)
    center_distance = _distance(contact_point, position)
    tangent = _orientation_tangent(normal, selected_wrist.wrist_velocity, forearm_dir)
    binormal = _normalize(_cross(normal, tangent))
    tangent = _normalize(_cross(binormal, normal))
    uncertainty = _normal_uncertainty(
        delta_v_norm=delta_norm,
        v_in=ball_motion["v_in"],
        v_out=ball_motion["v_out"],
        ball_position_noise_m=ball_position_noise_m,
        restitution_range=restitution_range,
        approximate_fraction=ball_motion["approximate_fraction"],
    )
    normal_dot_v_out = _dot(normal, v_out)
    swing_alignment = (
        _dot(_normalize(selected_wrist.wrist_velocity), normal)
        if selected_wrist.wrist_velocity is not None and _norm(selected_wrist.wrist_velocity) > 1e-9
        else None
    )
    return {
        "contact_index": int(contact_index),
        "t": _round(t, 6),
        "frame": int(frame) if isinstance(frame, int | float) else frame,
        "source": SOURCE,
        "physics_derived": True,
        "trusted_for_rkt_promotion": False,
        "ball_contact_world": _vec_json(contact_point),
        "velocity_in_mps": _vec_json(v_in),
        "velocity_out_mps": _vec_json(v_out),
        "delta_velocity_mps": _vec_json(delta_v),
        "delta_velocity_norm_mps": _round(delta_norm, 6),
        "face_normal_world": _vec_json(normal),
        "position_world": _vec_json(position),
        "orientation_basis": {
            "normal": _vec_json(normal),
            "tangent": _vec_json(tangent),
            "binormal": _vec_json(binormal),
        },
        "selected_wrist": {
            "player_id": selected_wrist.player_id,
            "side": selected_wrist.side,
            "wrist_world": _vec_json(selected_wrist.wrist),
            "elbow_world": _vec_json(selected_wrist.elbow),
            "forearm_direction_world": _vec_json(forearm_dir),
            "wrist_velocity_mps": _vec_json(selected_wrist.wrist_velocity) if selected_wrist.wrist_velocity is not None else None,
            "confidence": _round(selected_wrist.confidence, 6),
            "time_gap_s": _round(selected_wrist.time_gap_s, 6),
            "wrist_peak_hint": dict(selected_wrist.hint) if selected_wrist.hint is not None else None,
        },
        "reach": {
            "contact_to_wrist_m": _round(wrist_distance, 6),
            "contact_to_paddle_center_m": _round(center_distance, 6),
            "max_contact_to_wrist_m": _round(max_contact_to_wrist_m, 6),
            "plausible": wrist_distance <= max_contact_to_wrist_m,
        },
        "validation": {
            "outgoing_hemisphere": normal_dot_v_out >= -1e-9,
            "normal_dot_v_out": _round(normal_dot_v_out, 6),
            "normal_angle_to_delta_v_deg": _round(_angle_deg(normal, delta_v), 6),
            "wrist_swing_normal_alignment_cos": _round(swing_alignment, 6) if swing_alignment is not None else None,
        },
        "uncertainty": uncertainty,
        "ball_samples": {
            "pre_velocity_vector_count": len(ball_motion["v_in"].vectors),
            "post_velocity_vector_count": len(ball_motion["v_out"].vectors),
            "pre_median_dt_s": _round(ball_motion["v_in"].median_dt_s, 6),
            "post_median_dt_s": _round(ball_motion["v_out"].median_dt_s, 6),
            "approximate_fraction": _round(ball_motion["approximate_fraction"], 6),
        },
        "trust_band": {
            "status": "preview",
            "note": "physics-derived normal estimate; restitution/spin and ball-track quality bound the uncertainty",
        },
    }, "", {}


def _ball_motion_around_contact(
    samples: Sequence[BallSample],
    *,
    t: float,
    velocity_window_s: float,
    max_sample_gap_s: float,
    min_velocity_vectors_per_side: int,
) -> dict[str, Any] | None:
    window = [sample for sample in samples if t - velocity_window_s <= sample.t <= t + velocity_window_s]
    v_in, v_out = _finite_difference_velocity_around_contact(window, t=t, max_sample_gap_s=max_sample_gap_s)
    if v_in is None or v_out is None:
        return None
    if len(v_in.vectors) < min_velocity_vectors_per_side or len(v_out.vectors) < min_velocity_vectors_per_side:
        return None
    approximate_fraction = sum(1 for sample in window if sample.approximate) / max(1, len(window))
    return {"v_in": v_in, "v_out": v_out, "approximate_fraction": approximate_fraction}


def _finite_difference_velocity_around_contact(
    samples: Sequence[BallSample],
    *,
    t: float,
    max_sample_gap_s: float,
) -> tuple[VelocityEstimate | None, VelocityEstimate | None]:
    ordered = sorted(samples, key=lambda sample: sample.t)
    before_vectors: list[tuple[float, float, float]] = []
    before_dts: list[float] = []
    after_vectors: list[tuple[float, float, float]] = []
    after_dts: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        dt = current.t - previous.t
        if dt <= 1e-9 or dt > max_sample_gap_s:
            continue
        vector = _scale(_sub(current.world_xyz, previous.world_xyz), 1.0 / dt)
        midpoint = (previous.t + current.t) / 2.0
        if midpoint < t:
            before_vectors.append(vector)
            before_dts.append(dt)
        else:
            after_vectors.append(vector)
            after_dts.append(dt)
    return _velocity_estimate_from_vectors(before_vectors, before_dts), _velocity_estimate_from_vectors(after_vectors, after_dts)


def _velocity_estimate_from_vectors(
    vectors: Sequence[tuple[float, float, float]],
    dts: Sequence[float],
) -> VelocityEstimate | None:
    if not vectors or not dts:
        return None
    vector = (
        float(median([item[0] for item in vectors])),
        float(median([item[1] for item in vectors])),
        float(median([item[2] for item in vectors])),
    )
    scatter = _velocity_scatter(vectors, vector)
    return VelocityEstimate(vector=vector, vectors=tuple(vectors), median_dt_s=float(median(dts)), scatter_mps=scatter)


def _velocity_scatter(
    vectors: Sequence[tuple[float, float, float]],
    center: tuple[float, float, float],
) -> float:
    if len(vectors) <= 1:
        return 0.0
    distances = [_distance(vector, center) for vector in vectors]
    return float(median(distances))


def _normal_uncertainty(
    *,
    delta_v_norm: float,
    v_in: VelocityEstimate,
    v_out: VelocityEstimate,
    ball_position_noise_m: float,
    restitution_range: tuple[float, float],
    approximate_fraction: float,
) -> dict[str, Any]:
    dt_noise_in = ball_position_noise_m / max(v_in.median_dt_s, 1e-9)
    dt_noise_out = ball_position_noise_m / max(v_out.median_dt_s, 1e-9)
    delta_noise = math.sqrt((v_in.scatter_mps + dt_noise_in) ** 2 + (v_out.scatter_mps + dt_noise_out) ** 2)
    finite_diff_deg = math.degrees(math.atan2(delta_noise, max(delta_v_norm, 1e-9)))
    e_min, e_max = restitution_range
    e_mid = (e_min + e_max) / 2.0
    restitution_caveat_deg = math.degrees(math.atan2(max(0.05, e_max - e_min), 1.0 + e_mid))
    source_quality_caveat_deg = 20.0 * max(0.0, min(1.0, approximate_fraction))
    bound = min(90.0, finite_diff_deg + restitution_caveat_deg + source_quality_caveat_deg)
    return {
        "normal_angle_bound_deg": _round(bound, 6),
        "finite_difference_noise_deg": _round(finite_diff_deg, 6),
        "restitution_spin_caveat_deg": _round(restitution_caveat_deg, 6),
        "source_quality_caveat_deg": _round(source_quality_caveat_deg, 6),
        "delta_v_noise_mps": _round(delta_noise, 6),
        "ball_position_noise_m": _round(ball_position_noise_m, 6),
        "restitution_range": [_round(restitution_range[0], 6), _round(restitution_range[1], 6)],
    }


def _select_wrist(
    wrist_index: Mapping[tuple[int | str, str], Sequence[WristSample]],
    wrist_hints: Sequence[Mapping[str, Any]],
    *,
    t: float,
    contact_point: tuple[float, float, float],
    player_id: Any,
    wrist_to_paddle_center_m: float,
    max_contact_to_wrist_m: float,
    max_wrist_time_gap_s: float,
) -> InterpolatedWrist | None:
    player_filter = None if player_id is None else str(player_id)
    nearby_hints = _nearby_wrist_hints(wrist_hints, t=t, max_time_gap_s=max_wrist_time_gap_s)
    candidates: list[tuple[float, InterpolatedWrist]] = []
    for (candidate_player_id, side), samples in wrist_index.items():
        if player_filter is not None and str(candidate_player_id) != player_filter:
            continue
        interpolated = _interpolate_wrist(samples, t=t, max_time_gap_s=max_wrist_time_gap_s)
        if interpolated is None:
            continue
        hint = _matching_hint(nearby_hints, player_id=candidate_player_id, side=side)
        forearm = _sub(interpolated.wrist, interpolated.elbow)
        if _norm(forearm) <= 1e-9:
            continue
        center = _add(interpolated.wrist, _scale(_normalize(forearm), wrist_to_paddle_center_m))
        wrist_distance = _distance(contact_point, interpolated.wrist)
        if wrist_distance > max_contact_to_wrist_m:
            continue
        score = _distance(contact_point, center) + 0.2 * interpolated.time_gap_s
        if hint is not None:
            score -= 0.1 + 0.1 * float(hint.get("confidence", 0.0))
            interpolated = InterpolatedWrist(
                player_id=interpolated.player_id,
                side=interpolated.side,
                wrist=interpolated.wrist,
                elbow=interpolated.elbow,
                wrist_velocity=interpolated.wrist_velocity,
                confidence=interpolated.confidence,
                time_gap_s=interpolated.time_gap_s,
                hint=hint,
            )
        candidates.append((score, interpolated))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _interpolate_wrist(
    samples: Sequence[WristSample],
    *,
    t: float,
    max_time_gap_s: float,
) -> InterpolatedWrist | None:
    ordered = sorted(samples, key=lambda sample: sample.t)
    if not ordered:
        return None
    before = [sample for sample in ordered if sample.t <= t]
    after = [sample for sample in ordered if sample.t >= t]
    if before and after:
        left = before[-1]
        right = after[0]
        total_gap = right.t - left.t
        if total_gap <= max_time_gap_s:
            if total_gap <= 1e-9:
                wrist = left.wrist
                elbow = left.elbow
                velocity = None
                time_gap = 0.0
            else:
                alpha = (t - left.t) / total_gap
                wrist = _lerp(left.wrist, right.wrist, alpha)
                elbow = _lerp(left.elbow, right.elbow, alpha)
                velocity = _scale(_sub(right.wrist, left.wrist), 1.0 / total_gap)
                time_gap = max(abs(t - left.t), abs(right.t - t))
            return InterpolatedWrist(
                player_id=left.player_id,
                side=left.side,
                wrist=wrist,
                elbow=elbow,
                wrist_velocity=velocity,
                confidence=min(left.confidence, right.confidence),
                time_gap_s=time_gap,
            )
    nearest = min(ordered, key=lambda sample: abs(sample.t - t))
    time_gap = abs(nearest.t - t)
    if time_gap > max_time_gap_s:
        return None
    return InterpolatedWrist(
        player_id=nearest.player_id,
        side=nearest.side,
        wrist=nearest.wrist,
        elbow=nearest.elbow,
        wrist_velocity=None,
        confidence=nearest.confidence,
        time_gap_s=time_gap,
    )


def _wrist_index(
    skeleton3d: Mapping[str, Any] | None,
    *,
    min_joint_confidence: float,
) -> tuple[dict[tuple[int | str, str], list[WristSample]], list[str]]:
    if skeleton3d is None:
        return {}, ["missing_skeleton3d_wrist_constraint"]
    skeleton = semanticize_skeleton_payload(skeleton3d) or skeleton3d
    players = skeleton.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}, ["invalid_skeleton3d_players"]
    joint_names = skeleton.get("joint_names")
    mapping = _semantic_joint_indexes(joint_names)
    required = {"left_wrist", "right_wrist", "left_elbow", "right_elbow"}
    if not required <= set(mapping):
        return {}, ["missing_wrist_or_elbow_joint_mapping"]

    index: dict[tuple[int | str, str], list[WristSample]] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id")
        if player_id is None:
            continue
        frames = player.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            t = _finite(frame.get("t"))
            joints = frame.get("joints_world")
            if t is None or not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
                continue
            confs = frame.get("joint_conf")
            for side in ("left", "right"):
                wrist_idx = mapping[f"{side}_wrist"]
                elbow_idx = mapping[f"{side}_elbow"]
                wrist = _vec3(joints[wrist_idx] if wrist_idx < len(joints) else None)
                elbow = _vec3(joints[elbow_idx] if elbow_idx < len(joints) else None)
                if wrist is None or elbow is None:
                    continue
                conf = min(
                    _joint_confidence(confs, wrist_idx),
                    _joint_confidence(confs, elbow_idx),
                )
                if conf < min_joint_confidence:
                    continue
                index.setdefault((player_id, side), []).append(
                    WristSample(
                        t=t,
                        frame=int(frame["frame"]) if isinstance(frame.get("frame"), int | float) else None,
                        player_id=player_id,
                        side=side,
                        wrist=wrist,
                        elbow=elbow,
                        confidence=conf,
                    )
                )
    warnings = [] if index else ["no_usable_wrist_samples"]
    return index, warnings


def _semantic_joint_indexes(joint_names: Any) -> dict[str, int]:
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        return {}
    output: dict[str, int] = {}
    for index, name in enumerate(joint_names):
        normalized = _normalize_name(name)
        aliases = {
            "leftwrist": "left_wrist",
            "lwrist": "left_wrist",
            "lefthand": "left_wrist",
            "lhand": "left_wrist",
            "rightwrist": "right_wrist",
            "rwrist": "right_wrist",
            "righthand": "right_wrist",
            "rhand": "right_wrist",
            "leftelbow": "left_elbow",
            "lelbow": "left_elbow",
            "rightelbow": "right_elbow",
            "relbow": "right_elbow",
        }
        semantic = aliases.get(normalized)
        if semantic is not None:
            output[semantic] = index
    return output


def _normalize_name(value: Any) -> str:
    return "".join(part for part in str(value).lower().replace("-", "_").split("_") if part)


def _joint_confidence(confs: Any, index: int) -> float:
    if not isinstance(confs, Sequence) or isinstance(confs, (str, bytes)) or index >= len(confs):
        return 1.0
    value = _finite(confs[index])
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def _wrist_hints(wrist_peaks: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if wrist_peaks is None:
        return []
    peaks = wrist_peaks.get("peaks")
    if not isinstance(peaks, list):
        return []
    hints: list[Mapping[str, Any]] = []
    for peak in peaks:
        if not isinstance(peak, Mapping):
            continue
        t = _finite(peak.get("time_s", peak.get("t")))
        if t is None:
            continue
        hints.append(
            {
                "time_s": t,
                "player_id": peak.get("player_id"),
                "side": str(peak.get("wrist_side", "")).lower(),
                "confidence": _finite(peak.get("confidence")) or 0.0,
                "speed_mps": _finite(peak.get("speed_mps")),
            }
        )
    return hints


def _nearby_wrist_hints(
    wrist_hints: Sequence[Mapping[str, Any]],
    *,
    t: float,
    max_time_gap_s: float,
) -> list[Mapping[str, Any]]:
    return [
        hint
        for hint in wrist_hints
        if abs(float(hint["time_s"]) - t) <= max_time_gap_s
    ]


def _matching_hint(
    hints: Sequence[Mapping[str, Any]],
    *,
    player_id: int | str,
    side: str,
) -> Mapping[str, Any] | None:
    for hint in hints:
        hint_player = hint.get("player_id")
        if hint_player is not None and str(hint_player) != str(player_id):
            continue
        hint_side = hint.get("side")
        if hint_side and str(hint_side) != side:
            continue
        return hint
    return None


def _ball_samples(ball_track: Mapping[str, Any]) -> tuple[list[BallSample], list[str]]:
    frames_payload = ball_track.get("frames")
    source_kind = "ball_track.frames"
    warnings: list[str] = []
    if not isinstance(frames_payload, list) and isinstance(ball_track.get("ball"), Mapping):
        ball = ball_track["ball"]
        frames_payload = ball.get("frames")
        source_kind = "virtual_world.ball.frames"
        warnings.append("virtual_world_ball_track_preview_source")
    if not isinstance(frames_payload, list):
        return [], ["missing_ball_frames"]

    fps = _finite(ball_track.get("fps")) or 30.0
    samples: list[BallSample] = []
    for index, frame in enumerate(frames_payload):
        if not isinstance(frame, Mapping):
            continue
        t = _finite(frame.get("t"))
        if t is None:
            frame_number = _finite(frame.get("frame"))
            t = frame_number / fps if frame_number is not None and fps > 0.0 else None
        world_xyz = (
            _vec3(frame.get("world_xyz"))
            or _vec3(frame.get("position_world"))
            or _xyz_from_mapping(frame)
        )
        if t is None or world_xyz is None:
            continue
        frame_value = frame.get("frame")
        samples.append(
            BallSample(
                t=t,
                frame=int(frame_value) if isinstance(frame_value, int | float) else index,
                world_xyz=world_xyz,
                confidence=_finite(frame.get("conf", frame.get("confidence"))),
                approximate=bool(frame.get("approx", False)) or source_kind.startswith("virtual_world"),
            )
        )
    samples = sorted(samples, key=lambda sample: (sample.t, sample.frame if sample.frame is not None else -1))
    if not samples:
        warnings.append("no_world_ball_samples")
    return samples, warnings


def _xyz_from_mapping(frame: Mapping[str, Any]) -> tuple[float, float, float] | None:
    if not all(axis in frame for axis in ("x", "y", "z")):
        return None
    return _vec3([frame["x"], frame["y"], frame["z"]])


def _contact_events(contact_windows: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = contact_windows.get("events")
    if not isinstance(events, list):
        raise ValueError("contact_windows.events must be a list")
    contacts: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("type", "contact") != "contact":
            continue
        t = _finite(event.get("t"))
        if t is None:
            continue
        payload = dict(event)
        payload["t"] = t
        contacts.append(payload)
    return sorted(contacts, key=lambda item: (float(item["t"]), int(item.get("frame", 0) or 0)))


def _interpolate_ball_position(
    samples: Sequence[BallSample],
    *,
    t: float,
    max_gap_s: float,
) -> tuple[float, float, float] | None:
    ordered = sorted(samples, key=lambda sample: sample.t)
    before = [sample for sample in ordered if sample.t <= t]
    after = [sample for sample in ordered if sample.t >= t]
    if before and after:
        left = before[-1]
        right = after[0]
        if right.t - left.t <= max_gap_s:
            if right.t - left.t <= 1e-9:
                return left.world_xyz
            return _lerp(left.world_xyz, right.world_xyz, (t - left.t) / (right.t - left.t))
    if ordered:
        nearest = min(ordered, key=lambda sample: abs(sample.t - t))
        if abs(nearest.t - t) <= max_gap_s:
            return nearest.world_xyz
    return None


def _orientation_tangent(
    normal: tuple[float, float, float],
    wrist_velocity: tuple[float, float, float] | None,
    forearm_dir: tuple[float, float, float],
) -> tuple[float, float, float]:
    candidates = []
    if wrist_velocity is not None:
        candidates.append(wrist_velocity)
    candidates.append(forearm_dir)
    for candidate in candidates:
        projected = _sub(candidate, _scale(normal, _dot(candidate, normal)))
        if _norm(projected) > 1e-9:
            return _normalize(projected)
    fallback = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
    return _normalize(_cross(normal, fallback))


def _summary(
    *,
    contacts: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    ball_samples: Sequence[BallSample],
    wrist_index: Mapping[tuple[int | str, str], Sequence[WristSample]],
) -> dict[str, Any]:
    outgoing_count = sum(1 for estimate in estimates if estimate.get("validation", {}).get("outgoing_hemisphere") is True)
    reach_count = sum(1 for estimate in estimates if estimate.get("reach", {}).get("plausible") is True)
    estimate_count = len(estimates)
    approximate_count = sum(1 for sample in ball_samples if sample.approximate)
    return {
        "reviewed_contact_count": len(contacts),
        "estimate_count": estimate_count,
        "skipped_contact_count": len(skipped),
        "ball_world_sample_count": len(ball_samples),
        "ball_approximate_sample_count": approximate_count,
        "wrist_track_count": len(wrist_index),
        "outgoing_hemisphere_fraction": _fraction(outgoing_count, estimate_count),
        "plausible_reach_fraction": _fraction(reach_count, estimate_count),
        "temporal_smoothness": _temporal_smoothness(estimates),
        "skip_reasons": _reason_counts(skipped),
    }


def _temporal_smoothness(estimates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(estimates, key=lambda estimate: float(estimate["t"]))
    angles: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        prev_normal = _vec3(previous.get("face_normal_world"))
        current_normal = _vec3(current.get("face_normal_world"))
        if prev_normal is None or current_normal is None:
            continue
        angles.append(_angle_deg(prev_normal, current_normal))
    if not angles:
        return {"pair_count": 0, "max_adjacent_normal_angle_deg": None, "median_adjacent_normal_angle_deg": None}
    return {
        "pair_count": len(angles),
        "max_adjacent_normal_angle_deg": _round(max(angles), 6),
        "median_adjacent_normal_angle_deg": _round(float(median(angles)), 6),
    }


def _reason_counts(skipped: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _blockers(
    skipped: Sequence[Mapping[str, Any]],
    *,
    estimate_count: int,
    wrist_index: Mapping[tuple[int | str, str], Sequence[WristSample]],
    ball_samples: Sequence[BallSample],
) -> list[str]:
    blockers = sorted({str(item.get("reason")) for item in skipped if item.get("reason")})
    if not ball_samples:
        blockers.append("missing_usable_ball_world_track")
    if not wrist_index:
        blockers.append("missing_usable_wrist_constraint")
    if estimate_count == 0 and "no_estimable_contacts" not in blockers:
        blockers.append("no_estimable_contacts")
    return list(dict.fromkeys(blockers))


def _draw_estimate_overlay(
    cv2: Any,
    frame: Any,
    calibration: CourtCalibration,
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    center = _vec3(estimate.get("position_world"))
    normal = _vec3(estimate.get("face_normal_world"))
    basis = estimate.get("orientation_basis")
    tangent = _vec3(basis.get("tangent")) if isinstance(basis, Mapping) else None
    binormal = _vec3(basis.get("binormal")) if isinstance(basis, Mapping) else None
    if center is None or normal is None or tangent is None or binormal is None:
        return {"status": "skipped", "reason": "missing_pose_vectors"}
    radius_m = 0.11
    points = [
        _add(center, _add(_scale(tangent, radius_m * math.cos(theta)), _scale(binormal, radius_m * math.sin(theta))))
        for theta in [2.0 * math.pi * idx / 20.0 for idx in range(20)]
    ]
    try:
        projected = project_world_points(calibration.extrinsics, calibration.intrinsics, points)
        center_px = project_world_points(calibration.extrinsics, calibration.intrinsics, [center])[0]
        normal_px = project_world_points(calibration.extrinsics, calibration.intrinsics, [_add(center, _scale(normal, 0.35))])[0]
    except Exception as exc:
        return {"status": "skipped", "reason": "projection_failed", "details": str(exc)}
    poly = [_point(point) for point in projected]
    color = (80, 220, 255)
    line_type = getattr(cv2, "LINE_AA", 16)
    for start, end in zip(poly, [*poly[1:], poly[0]]):
        cv2.line(frame, start, end, color, 2, line_type)
    cv2.arrowedLine(frame, _point(center_px), _point(normal_px), (255, 120, 80), 2, line_type, tipLength=0.25)
    label = f"PHYS EST c{int(estimate['contact_index'])} +/-{float(estimate['uncertainty']['normal_angle_bound_deg']):.1f}deg"
    cv2.putText(frame, label, (_point(center_px)[0] + 6, _point(center_px)[1] - 6), getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0), 0.45, (0, 0, 0), 2, line_type)
    cv2.putText(frame, label, (_point(center_px)[0] + 6, _point(center_px)[1] - 6), getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0), 0.45, (255, 255, 255), 1, line_type)
    return {"status": "rendered"}


def _overlay_blocked(video_path: str | Path, output_dir: str | Path, reason: str, details: str) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_pose_estimate_overlays",
        "status": "blocked",
        "qualitative_status": "review_only_not_gate_verified",
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "requested_overlay_count": 0,
        "rendered_overlay_count": 0,
        "items": [],
        "blockers": [reason],
        "details": details,
    }
    (out_dir / "racket_pose_estimate_overlay_index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json_object(path: str | Path | None, name: str) -> dict[str, Any]:
    if path is None:
        raise ValueError(f"{name} path is required")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _validate_restitution_range(value: tuple[float, float]) -> None:
    if len(value) != 2:
        raise ValueError("restitution_range must have two values")
    low, high = float(value[0]), float(value[1])
    if not (0.0 < low <= high <= 1.2):
        raise ValueError("restitution_range must be ordered and physically plausible")


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / denominator, 6)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    x = _finite(value[0])
    y = _finite(value[1])
    z = _finite(value[2])
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(a)
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return _scale(a, 1.0 / norm)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return _norm(_sub(a, b))


def _lerp(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def _angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    norm_product = _norm(a) * _norm(b)
    if norm_product <= 1e-12:
        return 0.0
    cos_value = max(-1.0, min(1.0, _dot(a, b) / norm_product))
    return math.degrees(math.acos(cos_value))


def _vec_json(value: tuple[float, float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [_round(value[0], 9), _round(value[1], 9), _round(value[2], 9)]


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _point(value: Sequence[float]) -> tuple[int, int]:
    return int(round(float(value[0]))), int(round(float(value[1])))


__all__ = [
    "ARTIFACT_TYPE",
    "SOURCE",
    "build_racket_physics_estimate",
    "build_racket_physics_estimate_from_files",
    "render_racket_physics_estimate_overlays",
    "write_racket_physics_estimate",
]
