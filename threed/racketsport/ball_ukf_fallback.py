"""Default-off covariance-gated UKF/RTS candidate for short BALL arc gaps.

The candidate consumes only accepted ``status="fit"`` arc states as seeds.
It never turns a prediction into a measurement: every recovered sample is
``source=physics_interpolated``, ``band=physics_predicted``, and low confidence.
One-sided recovery is intentionally capped by
``DEFAULT_MAX_ONE_SIDED_HORIZON_FRAMES``; longer spans stay missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .ball_joint_anchor_search import CameraModel, project_world
from .virtual_world import ball_arc_segment_fail_closed_verdicts


ARTIFACT_TYPE = "racketsport_ball_ukf_fallback_candidate"
DEFAULT_MAX_ONE_SIDED_HORIZON_FRAMES = 12
RECOVERY_POLICY_V2_MAX_POSITION_COVARIANCE_M2 = 0.25
RECOVERY_POLICY_V2_MAX_STEP_SPEED_MPS = 35.0


@dataclass(frozen=True)
class UkfFallbackConfig:
    max_gap_frames: int = 12
    max_one_sided_horizon_frames: int = DEFAULT_MAX_ONE_SIDED_HORIZON_FRAMES
    gravity_mps2: float = 9.81
    drag_per_s: float = 0.0
    seed_position_std_m: float = 0.04
    seed_velocity_std_mps: float = 0.40
    process_accel_std_mps2: float = 2.0
    terminal_position_std_m: float = 0.08
    terminal_velocity_std_mps: float = 0.80
    max_position_covariance_m2: float = 0.25
    max_speed_mps: float = 45.0
    min_height_m: float = -0.05
    max_height_m: float = 12.0
    court_x_min_m: float = -7.048
    court_x_max_m: float = 7.048
    court_y_min_m: float = -10.7056
    court_y_max_m: float = 10.7056
    net_y_m: float = 0.0
    net_height_m: float = 0.864
    net_clearance_slack_m: float = 0.0

    def __post_init__(self) -> None:
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be positive")
        if self.max_one_sided_horizon_frames < 1:
            raise ValueError("max_one_sided_horizon_frames must be positive")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if self.drag_per_s < 0.0:
            raise ValueError("drag_per_s must be non-negative")
        if min(
            self.seed_position_std_m,
            self.seed_velocity_std_mps,
            self.process_accel_std_mps2,
            self.terminal_position_std_m,
            self.terminal_velocity_std_mps,
            self.max_position_covariance_m2,
            self.max_speed_mps,
        ) <= 0.0:
            raise ValueError("uncertainty and ceiling values must be positive")
        if self.min_height_m >= self.max_height_m:
            raise ValueError("height bounds must be ordered")
        if self.court_x_min_m >= self.court_x_max_m or self.court_y_min_m >= self.court_y_max_m:
            raise ValueError("court bounds must be ordered")


@dataclass(frozen=True)
class RecoveryPolicyV2Config:
    """Independent, default-off eligibility extensions for honest 3D recovery.

    The 0.25 m^2 position-variance ceiling (0.50 m standard deviation on any
    world axis) replaces v1's fixed 12-frame one-sided horizon. Two-sided
    bridges must solve and pass gates over the complete suppressed span.
    """

    enable_two_sided_bridge: bool = True
    enable_covariance_one_sided: bool = True
    enable_low_confidence_2d_updates: bool = False
    max_position_covariance_m2: float = RECOVERY_POLICY_V2_MAX_POSITION_COVARIANCE_M2
    max_step_speed_mps: float = RECOVERY_POLICY_V2_MAX_STEP_SPEED_MPS
    low_confidence_min: float = 0.05
    low_confidence_max: float = 0.50
    low_confidence_pixel_std_at_unit_conf: float = 12.0
    low_confidence_innovation_chi2_max: float = 9.21

    def __post_init__(self) -> None:
        if self.max_position_covariance_m2 <= 0.0:
            raise ValueError("max_position_covariance_m2 must be positive")
        if self.max_step_speed_mps <= 0.0 or self.max_step_speed_mps > 35.0:
            raise ValueError("max_step_speed_mps must be in (0, 35]")
        if not (0.0 < self.low_confidence_min <= self.low_confidence_max <= 1.0):
            raise ValueError("low-confidence bounds must satisfy 0 < min <= max <= 1")
        if self.low_confidence_pixel_std_at_unit_conf <= 0.0:
            raise ValueError("low_confidence_pixel_std_at_unit_conf must be positive")
        if self.low_confidence_innovation_chi2_max <= 0.0:
            raise ValueError("low_confidence_innovation_chi2_max must be positive")


def build_ukf_fallback(
    arc_solved: Mapping[str, Any],
    *,
    contact_proposals: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    net_plane: Mapping[str, Any] | None = None,
    config: UkfFallbackConfig | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a render-only candidate sidecar without mutating ``arc_solved``."""

    cfg = config or _config_from_artifact(arc_solved)
    frames = arc_solved.get("frames")
    segments_raw = arc_solved.get("segments")
    segments = [item for item in segments_raw if isinstance(item, Mapping)] if isinstance(segments_raw, list) else []
    segments.sort(key=lambda item: (_int(item.get("frame_start"), 10**12), _int(item.get("segment_id"), 10**12)))
    frame_list = frames if isinstance(frames, list) else []
    verdicts = ball_arc_segment_fail_closed_verdicts(segments)
    contact_frames = _contact_frames(contact_proposals, arc_solved)
    effective_net_y, effective_net_height = _net_geometry(net_plane, cfg)

    samples: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    attempted = 0
    rts_smoothed = 0
    one_sided = 0
    contact_refused = 0
    hard_gate_violations = 0
    trusted_non_fit_skipped = 0

    for index, segment in enumerate(segments):
        if _accepted_fit(segment, verdicts):
            continue
        segment_id = segment.get("segment_id")
        verdict = verdicts.get(segment_id) if isinstance(segment_id, int) and not isinstance(segment_id, bool) else None
        if verdict is not None and bool(verdict.get("trusted")):
            trusted_non_fit_skipped += 1
            continue
        start = _int(segment.get("frame_start"), -1)
        end = _int(segment.get("frame_end"), -1)
        if start < 0 or end < start:
            continue
        left = segments[index - 1] if index > 0 and _accepted_fit(segments[index - 1], verdicts) else None
        right = segments[index + 1] if index + 1 < len(segments) and _accepted_fit(segments[index + 1], verdicts) else None
        # Adjacent solver segments share their anchor frame. That accepted fit
        # endpoint is the seed, not a predicted sample or part of the horizon.
        gap_start = start + 1 if left is not None and _int(left.get("frame_end"), -2) == start else start
        gap_end = end - 1 if right is not None and _int(right.get("frame_start"), -2) == end else end
        gap_frames = list(range(gap_start, gap_end + 1))
        if not gap_frames:
            continue
        attempted += 1
        record = {
            "segment_id": segment.get("segment_id"),
            "frame_start": gap_start,
            "frame_end": gap_end,
            "frame_count": len(gap_frames),
        }
        if left is None and right is None:
            refused.append({**record, "reason": "no_adjacent_accepted_fit_seed"})
            continue
        if len(gap_frames) > cfg.max_gap_frames:
            refused.append({**record, "reason": "gap_above_short_gap_ceiling"})
            continue
        if (left is None or right is None) and len(gap_frames) > cfg.max_one_sided_horizon_frames:
            refused.append({**record, "reason": "one_sided_horizon_exceeded"})
            continue
        propagation_start = _int(left.get("frame_end"), gap_start) if left is not None else gap_start
        propagation_end = _int(right.get("frame_start"), gap_end) if right is not None else gap_end
        if any(propagation_start <= frame <= propagation_end for frame in contact_frames):
            contact_refused += 1
            refused.append({**record, "reason": "contact_proposal_inside_gap"})
            continue
        pts = _exact_pts(frame_list, gap_frames)
        if pts is None:
            refused.append({**record, "reason": "missing_or_non_monotonic_exact_pts"})
            continue

        try:
            if left is not None:
                gap_states = _forward_gap(left, right, pts, cfg)
                method = "ukf_forward_rts_backward" if right is not None else "ukf_forward_one_sided"
            else:
                gap_states = _backward_gap(right, pts, cfg)  # type: ignore[arg-type]
                method = "ukf_backward_one_sided"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            refused.append({**record, "reason": "ukf_numerical_failure"})
            continue

        boundary_before = None
        if left is not None:
            boundary_before = _segment_state_at(left, _float(left.get("t1"), pts[0]), cfg)
        boundary_after = None
        if right is not None:
            boundary_after = _segment_state_at(right, _float(right.get("t0"), pts[-1]), cfg)
        violation = _first_hard_gate_violation(
            gap_states,
            cfg,
            net_y_m=effective_net_y,
            net_height_m=effective_net_height,
            boundary_before=boundary_before,
            boundary_after=boundary_after,
        )
        if violation is not None:
            hard_gate_violations += 1
            refused.append({**record, "reason": violation})
            continue

        if right is not None and left is not None:
            rts_smoothed += 1
        else:
            one_sided += 1
        for offset, (frame_index, t, state, covariance) in enumerate(
            zip(gap_frames, pts, gap_states[0], gap_states[1], strict=True), start=1
        ):
            horizon_age = min(offset, len(gap_frames) - offset + 1) if left is not None and right is not None else (
                offset if left is not None else len(gap_frames) - offset + 1
            )
            position_covariance = covariance[:3, :3]
            samples.append(
                {
                    "frame_index": frame_index,
                    "t": float(t),
                    "segment_id": segment.get("segment_id"),
                    "world_xyz": [float(value) for value in state[:3]],
                    "velocity_mps": [float(value) for value in state[3:]],
                    "speed_mps": float(np.linalg.norm(state[3:])),
                    "source": "physics_interpolated",
                    "band": "physics_predicted",
                    "trust_band": {
                        "band": "low_confidence",
                        "reason": "covariance_gated_ukf_fallback_candidate",
                        "render_only": True,
                    },
                    "measured": False,
                    "render_only": True,
                    "not_for_detection_metrics": True,
                    "method": method,
                    "covariance_position_m2": position_covariance.tolist(),
                    "covariance_state": covariance.tolist(),
                    "position_covariance_max_m2": float(np.max(np.diag(position_covariance))),
                    "horizon_age_frames": int(horizon_age),
                }
            )

    samples.sort(key=lambda item: (item["frame_index"], item["t"]))
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(arc_solved.get("clip_id") or ""),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "candidate_flag_default": False,
        "render_only": True,
        "not_for_detection_metrics": True,
        "verified": False,
        "source_policy": "accepted fit states seed predictions; recovered samples are never measured",
        "measurement_covariance_policy": {
            "seed_position_std_m": cfg.seed_position_std_m,
            "seed_velocity_std_mps": cfg.seed_velocity_std_mps,
            "terminal_position_std_m": cfg.terminal_position_std_m,
            "terminal_velocity_std_mps": cfg.terminal_velocity_std_mps,
            "wasb_heatmap_footprint": "not_persisted",
            "future_source": (
                "persist connected-component extent from WASB raw candidate_frames into a dedicated "
                "ball-size observation sidecar before BallTrack schema validation"
            ),
        },
        "config": asdict(cfg),
        "samples": samples,
        "refused_gaps": refused,
        "summary": {
            "attempted_gap_count": attempted,
            "recovered_gap_count": len({sample["segment_id"] for sample in samples}),
            "recovered_sample_count": len(samples),
            "rts_smoothed_gap_count": rts_smoothed,
            "one_sided_gap_count": one_sided,
            "contact_refused_gap_count": contact_refused,
            "hard_gate_violation_count": hard_gate_violations,
            "trusted_non_fit_skipped_count": trusted_non_fit_skipped,
            "refused_gap_count": len(refused),
            "hard_gate_violations_emitted": 0,
        },
    }


def build_recovery_policy_v2(
    arc_solved: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    contact_proposals: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    net_plane: Mapping[str, Any] | None = None,
    config: RecoveryPolicyV2Config | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Recover fail-closed spans without changing observations or 2D metrics."""

    policy = config or RecoveryPolicyV2Config()
    base = _config_from_artifact(arc_solved)
    cfg = replace(base, max_position_covariance_m2=policy.max_position_covariance_m2)
    frames_raw = arc_solved.get("frames")
    frames = frames_raw if isinstance(frames_raw, list) else []
    segments_raw = arc_solved.get("segments")
    segments = [item for item in segments_raw if isinstance(item, Mapping)] if isinstance(segments_raw, list) else []
    segments.sort(key=lambda item: (_int(item.get("frame_start"), 10**12), _int(item.get("segment_id"), 10**12)))
    verdicts = ball_arc_segment_fail_closed_verdicts(segments)
    contact_frames = _contact_frames(contact_proposals, arc_solved)
    effective_net_y, effective_net_height = _net_geometry(net_plane, cfg)
    camera: CameraModel | None = None
    camera_refusal: str | None = None
    if policy.enable_low_confidence_2d_updates:
        if not isinstance(calibration, Mapping):
            camera_refusal = "low_confidence_2d_updates_require_calibration"
        else:
            try:
                camera = CameraModel.from_calibration(calibration)
            except (TypeError, ValueError):
                camera_refusal = "invalid_calibration_for_low_confidence_2d_updates"

    samples: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    attempted = recovered = bridge_recovered = one_sided_recovered = 0
    contact_refused = hard_gate_violations = low_confidence_updates = covariance_truncations = 0
    groups = _suppressed_segment_groups(segments, verdicts)
    for first_index, last_index in groups:
        span_segments = segments[first_index : last_index + 1]
        left = segments[first_index - 1] if first_index > 0 and _accepted_fit(segments[first_index - 1], verdicts) else None
        right = (
            segments[last_index + 1]
            if last_index + 1 < len(segments) and _accepted_fit(segments[last_index + 1], verdicts)
            else None
        )
        raw_start = _int(span_segments[0].get("frame_start"), -1)
        raw_end = _int(span_segments[-1].get("frame_end"), -1)
        if raw_start < 0 or raw_end < raw_start:
            continue
        gap_start = raw_start + 1 if left is not None and _int(left.get("frame_end"), -2) == raw_start else raw_start
        gap_end = raw_end - 1 if right is not None and _int(right.get("frame_start"), -2) == raw_end else raw_end
        gap_frames = list(range(gap_start, gap_end + 1))
        if not gap_frames:
            continue
        attempted += 1
        segment_ids = [item.get("segment_id") for item in span_segments]
        record = {
            "segment_ids": segment_ids,
            "frame_start": gap_start,
            "frame_end": gap_end,
            "frame_count": len(gap_frames),
        }
        if left is None and right is None:
            refused.append({**record, "reason": "no_accepted_fit_boundary"})
            continue
        two_sided = left is not None and right is not None
        if two_sided and not policy.enable_two_sided_bridge:
            refused.append({**record, "reason": "two_sided_bridge_policy_disabled"})
            continue
        if not two_sided and not policy.enable_covariance_one_sided:
            refused.append({**record, "reason": "covariance_one_sided_policy_disabled"})
            continue
        propagation_start = _int(left.get("frame_end"), gap_start) if left is not None else gap_start
        propagation_end = _int(right.get("frame_start"), gap_end) if right is not None else gap_end
        if any(propagation_start <= frame <= propagation_end for frame in contact_frames):
            contact_refused += 1
            refused.append({**record, "reason": "contact_proposal_inside_gap"})
            continue
        pts = _exact_pts(frames, gap_frames)
        if pts is None:
            refused.append({**record, "reason": "missing_or_non_monotonic_exact_pts"})
            continue
        try:
            if left is not None:
                states, covariances, update_frames = _forward_gap_v2(
                    left,
                    right,
                    pts,
                    gap_frames,
                    frames,
                    cfg,
                    policy,
                    camera if right is not None else None,
                )
                method = "ukf_v2_full_span_rts_bridge" if right is not None else "ukf_v2_covariance_forward"
            else:
                states, covariances = _backward_gap_v2(right, pts, frames, cfg)  # type: ignore[arg-type]
                update_frames = set()
                method = "ukf_v2_covariance_backward"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            refused.append({**record, "reason": "ukf_numerical_failure"})
            continue

        boundary_before = _accepted_boundary_state(left, frames, use_end=True, cfg=cfg)[0] if left is not None else None
        boundary_after = _accepted_boundary_state(right, frames, use_end=False, cfg=cfg)[0] if right is not None else None
        if two_sided:
            violation = _first_hard_gate_violation(
                (states, covariances),
                cfg,
                net_y_m=effective_net_y,
                net_height_m=effective_net_height,
                boundary_before=boundary_before,
                boundary_after=boundary_after,
                times=pts,
                max_step_speed_mps=policy.max_step_speed_mps,
            )
            if violation is not None:
                hard_gate_violations += 1
                refused.append({**record, "reason": violation})
                continue
            selected = slice(0, len(states))
        else:
            selected, stop_reason = _covariance_safe_one_sided_slice(
                states,
                covariances,
                pts,
                cfg,
                forward=left is not None,
                net_y_m=effective_net_y,
                net_height_m=effective_net_height,
                boundary_before=boundary_before,
                boundary_after=boundary_after,
                max_step_speed_mps=policy.max_step_speed_mps,
            )
            if selected.start == selected.stop:
                hard_gate_violations += 1
                refused.append({**record, "reason": stop_reason or "no_covariance_safe_prefix"})
                continue
            if (selected.stop - selected.start) < len(states):
                covariance_truncations += 1
                refused.append(
                    {
                        **record,
                        "reason": stop_reason or "one_sided_covariance_horizon_reached",
                        "recovered_frame_start": gap_frames[selected.start],
                        "recovered_frame_end": gap_frames[selected.stop - 1],
                    }
                )

        chosen_frames = gap_frames[selected]
        chosen_pts = pts[selected]
        chosen_states = states[selected]
        chosen_covariances = covariances[selected]
        overlay_violation = _overlay_boundary_step_violation(
            chosen_frames,
            chosen_pts,
            chosen_states,
            frames,
            left=left,
            right=right,
            max_step_speed_mps=policy.max_step_speed_mps,
        )
        if overlay_violation is not None:
            hard_gate_violations += 1
            refused.append({**record, "reason": overlay_violation})
            continue
        recovered += 1
        bridge_recovered += int(two_sided)
        one_sided_recovered += int(not two_sided)
        low_confidence_updates += sum(frame in update_frames for frame in chosen_frames)
        for offset, (frame_index, t, state, covariance) in enumerate(
            zip(chosen_frames, chosen_pts, chosen_states, chosen_covariances, strict=True), start=1
        ):
            horizon_age = min(offset, len(chosen_frames) - offset + 1) if two_sided else (
                offset if left is not None else len(chosen_frames) - offset + 1
            )
            original_segment_id = _segment_id_for_frame(span_segments, frame_index)
            measurement_support = frame_index in update_frames
            position_covariance = covariance[:3, :3]
            samples.append(
                {
                    "frame_index": frame_index,
                    "t": float(t),
                    "segment_id": original_segment_id,
                    "recovery_span_segment_ids": segment_ids,
                    "world_xyz": [float(value) for value in state[:3]],
                    "velocity_mps": [float(value) for value in state[3:]],
                    "speed_mps": float(np.linalg.norm(state[3:])),
                    "source": "physics_interpolated",
                    "band": "physics_predicted",
                    "trust_band": {
                        "band": "low_confidence",
                        "reason": "covariance_gated_recovery_policy_v2_candidate",
                        "render_only": True,
                    },
                    "measured": False,
                    "render_only": True,
                    "not_for_detection_metrics": True,
                    "method": method,
                    "low_confidence_2d_measurement_support": measurement_support,
                    "measurement_provenance": (
                        "inflated_covariance_2d_support_never_measured" if measurement_support else None
                    ),
                    "covariance_position_m2": position_covariance.tolist(),
                    "covariance_state": covariance.tolist(),
                    "position_covariance_max_m2": float(np.max(np.diag(position_covariance))),
                    "horizon_age_frames": int(horizon_age),
                }
            )

    samples.sort(key=lambda item: (item["frame_index"], item["t"]))
    return {
        "schema_version": 2,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(arc_solved.get("clip_id") or ""),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "candidate_flag": "ball.recovery_policy_v2",
        "candidate_flag_default": False,
        "render_only": True,
        "not_for_detection_metrics": True,
        "verified": False,
        "source_policy": "accepted fits bound physics predictions; no recovered sample is measured",
        "config": {"ukf": asdict(cfg), "recovery_policy_v2": asdict(policy)},
        "policy_status": {
            "two_sided_bridge": "enabled" if policy.enable_two_sided_bridge else "disabled",
            "covariance_one_sided": "enabled" if policy.enable_covariance_one_sided else "disabled",
            "low_confidence_2d_updates": (
                camera_refusal or ("enabled" if policy.enable_low_confidence_2d_updates else "disabled")
            ),
            "one_sided_position_covariance_ceiling_m2": policy.max_position_covariance_m2,
            "max_step_speed_mps": policy.max_step_speed_mps,
            "two_sided_requires_full_gap": True,
        },
        "samples": samples,
        "refused_gaps": refused,
        "summary": {
            "attempted_gap_count": attempted,
            "recovered_gap_count": recovered,
            "recovered_sample_count": len(samples),
            "two_sided_bridge_gap_count": bridge_recovered,
            "one_sided_gap_count": one_sided_recovered,
            "one_sided_covariance_truncation_count": covariance_truncations,
            "low_confidence_2d_update_count": low_confidence_updates,
            "contact_refused_gap_count": contact_refused,
            "hard_gate_violation_count": hard_gate_violations,
            "refused_gap_count": len(refused),
            "hard_gate_violations_emitted": 0,
            "measured_recovered_sample_count": 0,
        },
    }


def _suppressed_segment_groups(
    segments: Sequence[Mapping[str, Any]], verdicts: Mapping[int, Mapping[str, Any]]
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, segment in enumerate(segments):
        segment_id = segment.get("segment_id")
        verdict = verdicts.get(segment_id) if isinstance(segment_id, int) and not isinstance(segment_id, bool) else None
        suppressed = not _accepted_fit(segment, verdicts) and not (verdict is not None and bool(verdict.get("trusted")))
        if suppressed and start is None:
            start = index
        if not suppressed and start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(segments) - 1))
    return groups


def _segment_id_for_frame(segments: Sequence[Mapping[str, Any]], frame: int) -> Any:
    for segment in segments:
        if _int(segment.get("frame_start"), 10**12) <= frame <= _int(segment.get("frame_end"), -1):
            return segment.get("segment_id")
    return None


def _overlay_boundary_step_violation(
    gap_frames: Sequence[int],
    times: Sequence[float],
    states: Sequence[np.ndarray],
    artifact_frames: Sequence[Any],
    *,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    max_step_speed_mps: float,
) -> str | None:
    timed_positions = [(float(t), np.asarray(state[:3], dtype=float)) for t, state in zip(times, states, strict=True)]
    boundaries: list[tuple[float, np.ndarray]] = []
    if left is not None and gap_frames:
        left_index = _int(left.get("frame_end"), gap_frames[0] - 1)
        boundary = _artifact_frame_position(artifact_frames, left_index)
        if boundary is not None:
            boundaries.append(boundary)
    boundaries.extend(timed_positions)
    if right is not None and gap_frames:
        right_index = _int(right.get("frame_start"), gap_frames[-1] + 1)
        boundary = _artifact_frame_position(artifact_frames, right_index)
        if boundary is not None:
            boundaries.append(boundary)
    for (t0, p0), (t1, p1) in zip(boundaries, boundaries[1:], strict=False):
        if t1 <= t0:
            return "non_monotonic_boundary_pts"
        if float(np.linalg.norm(p1 - p0) / (t1 - t0)) > max_step_speed_mps:
            return "step_speed_above_35_mps"
    return None


def _artifact_frame_position(frames: Sequence[Any], index: int) -> tuple[float, np.ndarray] | None:
    if index < 0 or index >= len(frames) or not isinstance(frames[index], Mapping):
        return None
    frame = frames[index]
    xyz = frame.get("world_xyz")
    t = frame.get("t")
    if (
        not isinstance(xyz, Sequence)
        or isinstance(xyz, (str, bytes))
        or len(xyz) != 3
        or isinstance(t, bool)
        or not isinstance(t, (int, float))
    ):
        return None
    position = np.asarray([float(value) for value in xyz], dtype=float)
    if not np.all(np.isfinite(position)) or not math.isfinite(float(t)):
        return None
    return float(t), position


def _config_from_artifact(artifact: Mapping[str, Any]) -> UkfFallbackConfig:
    physics = artifact.get("physics_parameters")
    gravity = _float(physics.get("gravity_mps2"), 9.81) if isinstance(physics, Mapping) else 9.81
    drag = 0.0
    if isinstance(physics, Mapping):
        drag = _float(physics.get("drag_per_s"), _float(physics.get("drag_coefficient"), 0.0))
    return UkfFallbackConfig(gravity_mps2=gravity, drag_per_s=max(0.0, drag))


def _accepted_fit(segment: Mapping[str, Any], verdicts: Mapping[int, Mapping[str, Any]]) -> bool:
    segment_id = segment.get("segment_id")
    verdict = verdicts.get(segment_id) if isinstance(segment_id, int) and not isinstance(segment_id, bool) else None
    return str(segment.get("status") or "") == "fit" and (verdict is None or bool(verdict.get("trusted")))


def _forward_gap(
    left: Mapping[str, Any],
    right: Mapping[str, Any] | None,
    pts: Sequence[float],
    cfg: UkfFallbackConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    left_t = _float(left.get("t1"), pts[0] - 1.0 / 30.0)
    x = _segment_state_at(left, left_t, cfg)
    p = _seed_covariance(cfg)
    times = [left_t, *pts]
    xs = [x]
    ps = [p]
    predicted: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    cross_covariances: list[np.ndarray] = []
    for before, after in zip(times, times[1:], strict=False):
        x_next, p_next, cross = _ukf_predict(xs[-1], ps[-1], after - before, cfg)
        predicted.append(x_next)
        predicted_covariances.append(p_next)
        cross_covariances.append(cross)
        xs.append(x_next)
        ps.append(p_next)

    if right is None:
        return xs[1:], ps[1:]

    right_t = _float(right.get("t0"), pts[-1] + 1.0 / 30.0)
    terminal_pred, terminal_cov, terminal_cross = _ukf_predict(xs[-1], ps[-1], right_t - pts[-1], cfg)
    predicted.append(terminal_pred)
    predicted_covariances.append(terminal_cov)
    cross_covariances.append(terminal_cross)
    terminal_state = _segment_state_at(right, right_t, cfg)
    terminal_covariance = np.diag(
        [cfg.terminal_position_std_m**2] * 3 + [cfg.terminal_velocity_std_mps**2] * 3
    )
    terminal_x, terminal_p = _full_state_update(terminal_pred, terminal_cov, terminal_state, terminal_covariance)
    xs.append(terminal_x)
    ps.append(terminal_p)
    smooth_xs = list(xs)
    smooth_ps = list(ps)
    for k in range(len(xs) - 2, -1, -1):
        gain = cross_covariances[k] @ np.linalg.pinv(predicted_covariances[k])
        smooth_xs[k] = xs[k] + gain @ (smooth_xs[k + 1] - predicted[k])
        smooth_ps[k] = _symmetrize(ps[k] + gain @ (smooth_ps[k + 1] - predicted_covariances[k]) @ gain.T)
    return smooth_xs[1:-1], smooth_ps[1:-1]


def _forward_gap_v2(
    left: Mapping[str, Any],
    right: Mapping[str, Any] | None,
    pts: Sequence[float],
    gap_frames: Sequence[int],
    frames: Sequence[Any],
    cfg: UkfFallbackConfig,
    policy: RecoveryPolicyV2Config,
    camera: CameraModel | None,
) -> tuple[list[np.ndarray], list[np.ndarray], set[int]]:
    left_state, left_t = _accepted_boundary_state(left, frames, use_end=True, cfg=cfg)
    xs = [left_state]
    ps = [_seed_covariance(cfg)]
    predicted: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    cross_covariances: list[np.ndarray] = []
    update_frames: set[int] = set()
    before_t = left_t
    for frame_index, after_t in zip(gap_frames, pts, strict=True):
        x_next, p_next, cross = _ukf_predict(xs[-1], ps[-1], after_t - before_t, cfg)
        predicted.append(x_next)
        predicted_covariances.append(p_next)
        cross_covariances.append(cross)
        if camera is not None and policy.enable_low_confidence_2d_updates:
            observation = _low_confidence_observation(frames, frame_index, policy)
            if observation is not None:
                updated = _pixel_measurement_update(x_next, p_next, observation, camera, policy)
                if updated is not None:
                    x_next, p_next = updated
                    update_frames.add(frame_index)
        xs.append(x_next)
        ps.append(p_next)
        before_t = after_t

    if right is None:
        return xs[1:], ps[1:], update_frames

    terminal_state, right_t = _accepted_boundary_state(right, frames, use_end=False, cfg=cfg)
    terminal_pred, terminal_cov, terminal_cross = _ukf_predict(xs[-1], ps[-1], right_t - pts[-1], cfg)
    predicted.append(terminal_pred)
    predicted_covariances.append(terminal_cov)
    cross_covariances.append(terminal_cross)
    terminal_measurement_covariance = np.diag(
        [cfg.terminal_position_std_m**2] * 3 + [cfg.terminal_velocity_std_mps**2] * 3
    )
    terminal_x, terminal_p = _full_state_update(
        terminal_pred, terminal_cov, terminal_state, terminal_measurement_covariance
    )
    xs.append(terminal_x)
    ps.append(terminal_p)
    smooth_xs = list(xs)
    smooth_ps = list(ps)
    for k in range(len(xs) - 2, -1, -1):
        gain = cross_covariances[k] @ np.linalg.pinv(predicted_covariances[k])
        smooth_xs[k] = xs[k] + gain @ (smooth_xs[k + 1] - predicted[k])
        smooth_ps[k] = _symmetrize(
            ps[k] + gain @ (smooth_ps[k + 1] - predicted_covariances[k]) @ gain.T
        )
    return smooth_xs[1:-1], smooth_ps[1:-1], update_frames


def _backward_gap_v2(
    right: Mapping[str, Any],
    pts: Sequence[float],
    frames: Sequence[Any],
    cfg: UkfFallbackConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x, current_t = _accepted_boundary_state(right, frames, use_end=False, cfg=cfg)
    p = _seed_covariance(cfg)
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for t in reversed(pts):
        x, p, _ = _ukf_predict(x, p, t - current_t, cfg)
        states.append(x)
        covariances.append(p)
        current_t = t
    return list(reversed(states)), list(reversed(covariances))


def _accepted_boundary_state(
    segment: Mapping[str, Any],
    frames: Sequence[Any],
    *,
    use_end: bool,
    cfg: UkfFallbackConfig,
) -> tuple[np.ndarray, float]:
    frame_key = "frame_end" if use_end else "frame_start"
    time_key = "t1" if use_end else "t0"
    frame_index = _int(segment.get(frame_key), -1)
    boundary = _artifact_frame_position(frames, frame_index)
    t = boundary[0] if boundary is not None else _float(segment.get(time_key), 0.0)
    state = _segment_state_at(segment, t, cfg)
    if boundary is not None:
        state[:3] = boundary[1]
    return state, t


def _low_confidence_observation(
    frames: Sequence[Any], frame_index: int, policy: RecoveryPolicyV2Config
) -> tuple[np.ndarray, float] | None:
    if frame_index < 0 or frame_index >= len(frames) or not isinstance(frames[frame_index], Mapping):
        return None
    frame = frames[frame_index]
    confidence = frame.get("conf")
    xy = frame.get("xy")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not (policy.low_confidence_min <= float(confidence) <= policy.low_confidence_max)
        or not isinstance(xy, Sequence)
        or isinstance(xy, (str, bytes))
        or len(xy) < 2
    ):
        return None
    parsed = np.asarray([float(xy[0]), float(xy[1])], dtype=float)
    if not np.all(np.isfinite(parsed)):
        return None
    return parsed, float(confidence)


def _pixel_measurement_update(
    x: np.ndarray,
    p: np.ndarray,
    observation: tuple[np.ndarray, float],
    camera: CameraModel,
    policy: RecoveryPolicyV2Config,
) -> tuple[np.ndarray, np.ndarray] | None:
    xy, confidence = observation
    projected, depth = project_world(camera, x[:3])
    if depth <= 0.0 or not all(math.isfinite(value) for value in projected):
        return None
    predicted_xy = np.asarray(projected, dtype=float)
    h = np.zeros((2, 6), dtype=float)
    epsilon = 1e-4
    for axis in range(3):
        shifted = x[:3].copy()
        shifted[axis] += epsilon
        shifted_xy, shifted_depth = project_world(camera, shifted)
        if shifted_depth <= 0.0:
            return None
        h[:, axis] = (np.asarray(shifted_xy, dtype=float) - predicted_xy) / epsilon
    sigma_px = policy.low_confidence_pixel_std_at_unit_conf / max(confidence, policy.low_confidence_min)
    r = np.eye(2, dtype=float) * sigma_px * sigma_px
    innovation = xy - predicted_xy
    innovation_covariance = h @ p @ h.T + r
    mahalanobis = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
    if not math.isfinite(mahalanobis) or mahalanobis > policy.low_confidence_innovation_chi2_max:
        return None
    gain = p @ h.T @ np.linalg.pinv(innovation_covariance)
    updated = x + gain @ innovation
    identity = np.eye(6, dtype=float)
    covariance = (identity - gain @ h) @ p @ (identity - gain @ h).T + gain @ r @ gain.T
    return updated, _symmetrize(covariance)


def _backward_gap(
    right: Mapping[str, Any],
    pts: Sequence[float],
    cfg: UkfFallbackConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    right_t = _float(right.get("t0"), pts[-1] + 1.0 / 30.0)
    x = _segment_state_at(right, right_t, cfg)
    p = _seed_covariance(cfg)
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    current_t = right_t
    for t in reversed(pts):
        x, p, _ = _ukf_predict(x, p, t - current_t, cfg)
        states.append(x)
        covariances.append(p)
        current_t = t
    return list(reversed(states)), list(reversed(covariances))


def _ukf_predict(
    x: np.ndarray,
    p: np.ndarray,
    dt: float,
    cfg: UkfFallbackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma, weights_mean, weights_cov = _sigma_points(x, p)
    propagated = np.array([_dynamics(point, dt, cfg) for point in sigma])
    mean = np.sum(weights_mean[:, None] * propagated, axis=0)
    covariance = np.zeros((6, 6), dtype=float)
    cross = np.zeros((6, 6), dtype=float)
    for weight, before, after in zip(weights_cov, sigma, propagated, strict=True):
        covariance += weight * np.outer(after - mean, after - mean)
        cross += weight * np.outer(before - x, after - mean)
    q = _process_covariance(dt, cfg)
    return mean, _symmetrize(covariance + q), cross


def _sigma_points(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    alpha, beta, kappa = 0.3, 2.0, 0.0
    lam = alpha * alpha * (n + kappa) - n
    scaled = _symmetrize((n + lam) * p) + np.eye(n) * 1e-12
    root = np.linalg.cholesky(scaled)
    points = [x]
    for column in root.T:
        points.extend((x + column, x - column))
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1.0 - alpha * alpha + beta)
    return np.asarray(points), wm, wc


def _dynamics(state: np.ndarray, dt: float, cfg: UkfFallbackConfig) -> np.ndarray:
    position = state[:3]
    velocity = state[3:]
    if cfg.drag_per_s > 0.0:
        decay = math.exp(-cfg.drag_per_s * dt)
        basis = (1.0 - decay) / cfg.drag_per_s
        next_velocity = velocity * decay
        next_position = position + velocity * basis
    else:
        next_velocity = velocity.copy()
        next_position = position + velocity * dt
    next_position[2] -= 0.5 * cfg.gravity_mps2 * dt * dt
    next_velocity[2] -= cfg.gravity_mps2 * dt
    return np.concatenate((next_position, next_velocity))


def _process_covariance(dt: float, cfg: UkfFallbackConfig) -> np.ndarray:
    magnitude = abs(dt)
    q = cfg.process_accel_std_mps2**2
    block = np.array([[magnitude**4 / 4.0, magnitude**3 / 2.0], [magnitude**3 / 2.0, magnitude**2]]) * q
    out = np.zeros((6, 6), dtype=float)
    for axis in range(3):
        out[axis, axis] = block[0, 0]
        out[axis, axis + 3] = block[0, 1]
        out[axis + 3, axis] = block[1, 0]
        out[axis + 3, axis + 3] = block[1, 1]
    return out


def _full_state_update(
    x: np.ndarray, p: np.ndarray, measurement: np.ndarray, measurement_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = p + measurement_covariance
    gain = p @ np.linalg.pinv(innovation_covariance)
    updated = x + gain @ (measurement - x)
    covariance = (np.eye(6) - gain) @ p @ (np.eye(6) - gain).T + gain @ measurement_covariance @ gain.T
    return updated, _symmetrize(covariance)


def _segment_state_at(segment: Mapping[str, Any], t: float, cfg: UkfFallbackConfig) -> np.ndarray:
    p0 = _vec3(segment.get("initial_position_m"), "initial_position_m")
    v0 = _vec3(segment.get("initial_velocity_mps"), "initial_velocity_mps")
    t0 = _float(segment.get("t0"), t)
    return _dynamics(np.asarray([*p0, *v0], dtype=float), t - t0, cfg)


def _seed_covariance(cfg: UkfFallbackConfig) -> np.ndarray:
    return np.diag([cfg.seed_position_std_m**2] * 3 + [cfg.seed_velocity_std_mps**2] * 3)


def _first_hard_gate_violation(
    gap_states: tuple[Sequence[np.ndarray], Sequence[np.ndarray]],
    cfg: UkfFallbackConfig,
    *,
    net_y_m: float,
    net_height_m: float,
    boundary_before: np.ndarray | None = None,
    boundary_after: np.ndarray | None = None,
    times: Sequence[float] | None = None,
    max_step_speed_mps: float | None = None,
) -> str | None:
    states, covariances = gap_states
    for state, covariance in zip(states, covariances, strict=True):
        x, y, z = (float(value) for value in state[:3])
        if not (cfg.court_x_min_m <= x <= cfg.court_x_max_m and cfg.court_y_min_m <= y <= cfg.court_y_max_m):
            return "outside_court_volume"
        if z < cfg.min_height_m:
            return "height_below_floor"
        if z > cfg.max_height_m:
            return "height_above_ceiling"
        if float(np.linalg.norm(state[3:])) > cfg.max_speed_mps:
            return "speed_above_ceiling"
        if float(np.max(np.diag(covariance[:3, :3]))) > cfg.max_position_covariance_m2:
            return "covariance_above_ceiling"
    if max_step_speed_mps is not None and times is not None:
        timed_states: list[tuple[float, np.ndarray]] = list(zip(times, states, strict=True))
        if boundary_before is not None and times:
            first_dt = times[1] - times[0] if len(times) > 1 else 1.0 / 30.0
            timed_states.insert(0, (times[0] - first_dt, boundary_before))
        if boundary_after is not None and times:
            last_dt = times[-1] - times[-2] if len(times) > 1 else 1.0 / 30.0
            timed_states.append((times[-1] + last_dt, boundary_after))
        for (t0, state0), (t1, state1) in zip(timed_states, timed_states[1:], strict=False):
            dt = t1 - t0
            if dt <= 0.0:
                return "non_monotonic_boundary_pts"
            step_speed = float(np.linalg.norm(state1[:3] - state0[:3]) / dt)
            if step_speed > max_step_speed_mps:
                return "step_speed_above_35_mps"
    net_path = [
        *([boundary_before] if boundary_before is not None else []),
        *states,
        *([boundary_after] if boundary_after is not None else []),
    ]
    for before, after in zip(net_path, net_path[1:], strict=False):
        y0, y1 = float(before[1]), float(after[1])
        if (y0 - net_y_m) * (y1 - net_y_m) > 0.0 or y0 == y1:
            continue
        fraction = (net_y_m - y0) / (y1 - y0)
        z_crossing = float(before[2]) + fraction * (float(after[2]) - float(before[2]))
        if z_crossing < net_height_m - cfg.net_clearance_slack_m:
            return "net_clearance_below_slack"
    return None


def _covariance_safe_one_sided_slice(
    states: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    times: Sequence[float],
    cfg: UkfFallbackConfig,
    *,
    forward: bool,
    net_y_m: float,
    net_height_m: float,
    boundary_before: np.ndarray | None,
    boundary_after: np.ndarray | None,
    max_step_speed_mps: float,
) -> tuple[slice, str | None]:
    safe_count = 0
    stop_reason: str | None = None
    total = len(states)
    for count in range(1, total + 1):
        if forward:
            selected_states = states[:count]
            selected_covariances = covariances[:count]
            selected_times = times[:count]
            before = boundary_before
            after = None
        else:
            selected_states = states[total - count :]
            selected_covariances = covariances[total - count :]
            selected_times = times[total - count :]
            before = None
            after = boundary_after
        stop_reason = _first_hard_gate_violation(
            (selected_states, selected_covariances),
            cfg,
            net_y_m=net_y_m,
            net_height_m=net_height_m,
            boundary_before=before,
            boundary_after=after,
            times=selected_times,
            max_step_speed_mps=max_step_speed_mps,
        )
        if stop_reason is not None:
            break
        safe_count = count
    if forward:
        return slice(0, safe_count), stop_reason
    return slice(total - safe_count, total), stop_reason


def _exact_pts(frames: Sequence[Any], indices: Sequence[int]) -> list[float] | None:
    values: list[float] = []
    for index in indices:
        if index < 0 or index >= len(frames) or not isinstance(frames[index], Mapping):
            return None
        raw = frames[index].get("t")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            return None
        values.append(float(raw))
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        return None
    return values


def _contact_frames(
    explicit: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    artifact: Mapping[str, Any],
) -> set[int]:
    containers: list[Any] = [explicit]
    event_selection = artifact.get("event_selection")
    if isinstance(event_selection, Mapping):
        containers.extend((event_selection.get("selected"), event_selection.get("events")))
    anchors = artifact.get("anchors")
    containers.append(anchors)
    output: set[int] = set()
    artifact_frames = artifact.get("frames")
    frame_list = artifact_frames if isinstance(artifact_frames, list) else []
    for container in containers:
        if isinstance(container, Mapping):
            candidates = container.get("events") or container.get("selected") or []
        else:
            candidates = container
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or item.get("type") or "").lower()
            if kind not in {"contact", "hit", "racket_contact", "paddle_contact"}:
                continue
            frame = item.get("frame", item.get("frame_index"))
            if isinstance(frame, int) and not isinstance(frame, bool):
                output.add(frame)
            window = item.get("window")
            if not isinstance(window, Mapping):
                continue
            t0 = window.get("t0")
            t1 = window.get("t1")
            if (
                isinstance(t0, bool)
                or not isinstance(t0, (int, float))
                or isinstance(t1, bool)
                or not isinstance(t1, (int, float))
                or not math.isfinite(float(t0))
                or not math.isfinite(float(t1))
            ):
                continue
            lower, upper = sorted((float(t0), float(t1)))
            for frame_index, artifact_frame in enumerate(frame_list):
                if not isinstance(artifact_frame, Mapping):
                    continue
                pts = artifact_frame.get("t")
                if (
                    not isinstance(pts, bool)
                    and isinstance(pts, (int, float))
                    and math.isfinite(float(pts))
                    and lower <= float(pts) <= upper
                ):
                    output.add(frame_index)
    return output


def _net_geometry(net_plane: Mapping[str, Any] | None, cfg: UkfFallbackConfig) -> tuple[float, float]:
    if not isinstance(net_plane, Mapping):
        return cfg.net_y_m, cfg.net_height_m
    plane = net_plane.get("plane")
    point = plane.get("point") if isinstance(plane, Mapping) else None
    y = _float(point[1], cfg.net_y_m) if isinstance(point, Sequence) and len(point) >= 2 else cfg.net_y_m
    heights = net_plane.get("heights_m")
    if isinstance(heights, Mapping):
        height = _float(heights.get("center"), _float(heights.get("center_m"), cfg.net_height_m))
    else:
        height = cfg.net_height_m
    return y, height


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field} must be a 3-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must be finite")
    return result


def _float(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return float(default)
    return float(value)


def _int(value: Any, default: int) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_MAX_ONE_SIDED_HORIZON_FRAMES",
    "RECOVERY_POLICY_V2_MAX_POSITION_COVARIANCE_M2",
    "RECOVERY_POLICY_V2_MAX_STEP_SPEED_MPS",
    "RecoveryPolicyV2Config",
    "UkfFallbackConfig",
    "build_recovery_policy_v2",
    "build_ukf_fallback",
]
