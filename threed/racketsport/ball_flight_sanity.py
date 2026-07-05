"""Render-only flight sanity gate for solved BALL arcs.

This module does not synthesize or repair ball positions. It only inspects a
rendered arc artifact and returns frame-level demotion flags for airborne
segments whose rendered world trajectory is not plausibly parabolic.

Constants intentionally reuse the solver's physical sanity scale where that
scale applies:

- initial/world speed limits come from ``BallArcSolverConfig``.
- gravity comes from ``PhysicsParameters``.
- the per-frame speed-jump bound is a continuity bound, not a new absolute
  speed cap. It is set to 25% of the solver's max plausible speed by default,
  which is far above one frame of gravity at 30 fps but still catches rendered
  teleports inside an airborne segment.
- vertical finite differences use a 3-frame smoothing window. The 0.35 m/s
  sign tolerance is roughly one 30 fps frame of gravity, so a noisy apex plateau
  is not counted as multiple direction reversals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .ball_arc_solver import BallArcSolverConfig, PhysicsParameters


ARTIFACT_TYPE = "racketsport_ball_flight_sanity"
SOURCE = "ball_flight_sanity_render_gate_v1"
DEMOTED_BAND = "arc_weak"
MEASURED_BAND = "anchored_measured"
_BAND_COUNT_KEYS = {
    "anchored_measured": "anchored_measured_count",
    "arc_interpolated": "arc_interpolated_count",
    "arc_extrapolated": "arc_extrapolated_count",
    "arc_weak": "arc_weak_count",
    "hidden": "hidden_count",
}


@dataclass(frozen=True)
class FlightSanityConfig:
    """Thresholds for render-gating an already solved ball flight."""

    vertical_smoothing_window_frames: int = 3
    vertical_velocity_tolerance_mps: float = 0.35
    horizontal_reversal_threshold_deg: float = 120.0
    anchor_neighborhood_frames: int = 2
    max_frame_speed_jump_fraction_of_solver_cap: float = 0.25
    min_horizontal_step_m: float = 0.015
    min_segment_samples: int = 5

    def speed_jump_limit_mps(self, solver_config: BallArcSolverConfig | None = None) -> float:
        cfg = solver_config or BallArcSolverConfig()
        return float(cfg.max_plausible_speed_mps) * float(self.max_frame_speed_jump_fraction_of_solver_cap)

    def to_json(self, *, solver_config: BallArcSolverConfig, physics: PhysicsParameters) -> dict[str, Any]:
        return {
            "vertical_smoothing_window_frames": int(self.vertical_smoothing_window_frames),
            "vertical_velocity_tolerance_mps": float(self.vertical_velocity_tolerance_mps),
            "horizontal_reversal_threshold_deg": float(self.horizontal_reversal_threshold_deg),
            "anchor_neighborhood_frames": int(self.anchor_neighborhood_frames),
            "max_frame_speed_jump_fraction_of_solver_cap": float(self.max_frame_speed_jump_fraction_of_solver_cap),
            "max_frame_speed_jump_mps": round(self.speed_jump_limit_mps(solver_config), 6),
            "min_horizontal_step_m": float(self.min_horizontal_step_m),
            "min_segment_samples": int(self.min_segment_samples),
            "solver_max_plausible_speed_mps": float(solver_config.max_plausible_speed_mps),
            "solver_max_plausible_apex_m": float(solver_config.max_plausible_apex_m),
            "gravity_mps2": float(physics.gravity_mps2),
        }


@dataclass(frozen=True)
class _Sample:
    frame: int
    t: float
    world_xyz: tuple[float, float, float]


def evaluate_ball_flight_sanity(
    arc_solved: Mapping[str, Any],
    *,
    config: FlightSanityConfig | None = None,
    solver_config: BallArcSolverConfig | None = None,
    physics: PhysicsParameters | None = None,
) -> dict[str, Any]:
    """Evaluate rendered airborne segments and return per-frame demotion flags."""

    gate_config = config or FlightSanityConfig()
    solver_cfg = solver_config or BallArcSolverConfig()
    phys = physics or _physics_from_artifact(arc_solved)
    frames = list(arc_solved.get("frames") or [])
    fps = _fps(arc_solved, frames)
    anchors = _anchors(arc_solved)
    frame_reports: list[dict[str, Any]] = [
        {
            "frame": index,
            "t": _frame_t(frame, index, fps),
            "segment_id": None,
            "demote": False,
            "reasons": [],
        }
        for index, frame in enumerate(frames)
    ]
    segments: list[dict[str, Any]] = []
    demote_reasons_by_frame: dict[int, set[str]] = {}

    for segment_id, (start, end) in enumerate(zip(anchors, anchors[1:], strict=False)):
        samples = _samples_for_segment(frames, fps=fps, start_t=start["t"], end_t=end["t"])
        segment_report = _evaluate_segment(
            segment_id=segment_id,
            start=start,
            end=end,
            samples=samples,
            config=gate_config,
            solver_config=solver_cfg,
        )
        segments.append(segment_report)
        for sample in samples:
            frame_reports[sample.frame]["segment_id"] = segment_id
            if segment_report["verdict"] == "fail":
                reasons = demote_reasons_by_frame.setdefault(sample.frame, set())
                reasons.update(str(reason) for reason in segment_report["reasons"])

    for frame_index, reasons in demote_reasons_by_frame.items():
        if 0 <= frame_index < len(frame_reports):
            frame_reports[frame_index]["demote"] = True
            frame_reports[frame_index]["reasons"] = sorted(reasons)

    failed_segment_count = sum(1 for segment in segments if segment["verdict"] == "fail")
    passed_segment_count = sum(1 for segment in segments if segment["verdict"] == "pass")
    skipped_segment_count = sum(1 for segment in segments if segment["verdict"] == "not_evaluated")
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "source": SOURCE,
        "clip_id": str(arc_solved.get("clip_id") or ""),
        "config": gate_config.to_json(solver_config=solver_cfg, physics=phys),
        "policy": {
            "render_gate_only": True,
            "does_not_create_or_adjust_world_xyz": True,
            "demotion_only_removes_measured_status": True,
            "demoted_band": DEMOTED_BAND,
        },
        "summary": {
            "segment_count": len(segments),
            "passed_segment_count": passed_segment_count,
            "failed_segment_count": failed_segment_count,
            "skipped_segment_count": skipped_segment_count,
            "demoted_frame_count": len(demote_reasons_by_frame),
        },
        "segments": segments,
        "frames": frame_reports,
    }


def apply_flight_sanity_demotions(arc_solved: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of an arc artifact with failing-segment frames demoted."""

    demoted = _demoted_frames(report)
    payload = dict(arc_solved)
    frames: list[dict[str, Any]] = []
    for index, raw_frame in enumerate(list(arc_solved.get("frames") or [])):
        frame = dict(raw_frame) if isinstance(raw_frame, Mapping) else {}
        if index in demoted:
            frame["flight_sanity_demoted"] = True
            frame["flight_sanity_reasons"] = demoted[index]
            if _has_world_xyz(frame) and str(frame.get("band") or "") != "hidden":
                frame["band"] = DEMOTED_BAND
        frames.append(frame)
    payload["frames"] = frames
    payload["summary"] = _summary_with_demotions(arc_solved.get("summary"), frames, report)
    validation = dict(payload.get("validation") or {})
    validation["flight_sanity"] = dict(report.get("summary") or {})
    payload["validation"] = validation
    return payload


def apply_product_view_flight_sanity_demotions(product_view: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a 2D product view with the same demotion flags applied."""

    demoted = _demoted_frames(report)
    payload = dict(product_view)
    frames: list[dict[str, Any]] = []
    for index, raw_frame in enumerate(list(product_view.get("frames") or [])):
        frame = dict(raw_frame) if isinstance(raw_frame, Mapping) else {}
        if index in demoted:
            frame["flight_sanity_demoted"] = True
            frame["flight_sanity_reasons"] = demoted[index]
            frame["band"] = DEMOTED_BAND
            frame["approx"] = True
        frames.append(frame)
    payload["frames"] = frames
    payload["flight_sanity"] = dict(report.get("summary") or {})
    return payload


def _evaluate_segment(
    *,
    segment_id: int,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    samples: Sequence[_Sample],
    config: FlightSanityConfig,
    solver_config: BallArcSolverConfig,
) -> dict[str, Any]:
    if len(samples) < config.min_segment_samples:
        return {
            "segment_id": segment_id,
            "verdict": "not_evaluated",
            "reasons": ["insufficient_world_samples"],
            "start_anchor": _anchor_ref(start),
            "end_anchor": _anchor_ref(end),
            "frame_start": samples[0].frame if samples else None,
            "frame_end": samples[-1].frame if samples else None,
            "sample_count": len(samples),
        }

    vertical_changes = _vertical_sign_changes(samples, config)
    heading = _horizontal_heading_report(samples, config)
    speed = _speed_continuity_report(samples, config, solver_config)
    reasons: list[str] = []
    if vertical_changes["sign_change_count"] > 1:
        reasons.append("vertical_multi_apex")
    if heading["max_heading_change_deg"] > config.horizontal_reversal_threshold_deg:
        reasons.append("horizontal_direction_reversal")
    if speed["max_speed_jump_mps"] > config.speed_jump_limit_mps(solver_config):
        reasons.append("speed_jump")

    return {
        "segment_id": segment_id,
        "verdict": "fail" if reasons else "pass",
        "reasons": reasons,
        "start_anchor": _anchor_ref(start),
        "end_anchor": _anchor_ref(end),
        "frame_start": samples[0].frame,
        "frame_end": samples[-1].frame,
        "t_start": round(samples[0].t, 9),
        "t_end": round(samples[-1].t, 9),
        "sample_count": len(samples),
        "vertical": vertical_changes,
        "horizontal": heading,
        "speed_continuity": speed,
    }


def _vertical_sign_changes(samples: Sequence[_Sample], config: FlightSanityConfig) -> dict[str, Any]:
    smoothed_z = _moving_average([sample.world_xyz[2] for sample in samples], config.vertical_smoothing_window_frames)
    velocities: list[float] = []
    for left, right, z0, z1 in zip(samples[:-1], samples[1:], smoothed_z[:-1], smoothed_z[1:], strict=True):
        dt = right.t - left.t
        if dt <= 1e-9:
            continue
        velocities.append((z1 - z0) / dt)
    signs: list[int] = []
    for velocity in velocities:
        if abs(velocity) <= config.vertical_velocity_tolerance_mps:
            signs.append(0)
        else:
            signs.append(1 if velocity > 0.0 else -1)
    compact = [sign for sign in signs if sign != 0]
    sign_change_count = sum(1 for prev, cur in zip(compact, compact[1:], strict=False) if prev != cur)
    return {
        "sign_change_count": int(sign_change_count),
        "allowed_sign_change_count": 1,
        "velocity_tolerance_mps": float(config.vertical_velocity_tolerance_mps),
        "smoothing_window_frames": int(config.vertical_smoothing_window_frames),
        "sampled_velocity_count": len(velocities),
    }


def _horizontal_heading_report(samples: Sequence[_Sample], config: FlightSanityConfig) -> dict[str, Any]:
    velocities = _velocity_vectors(samples)
    max_angle = 0.0
    max_at_frame: int | None = None
    start = int(config.anchor_neighborhood_frames)
    end = max(start, len(velocities) - int(config.anchor_neighborhood_frames))
    previous_xy: tuple[float, float] | None = None
    for index in range(max(1, start), end):
        cur = velocities[index]
        cur_xy = (cur[0], cur[1])
        if _norm2(cur_xy) < config.min_horizontal_step_m:
            continue
        if previous_xy is not None:
            angle = _angle_deg(previous_xy, cur_xy)
            if angle > max_angle:
                max_angle = angle
                max_at_frame = samples[index].frame
        previous_xy = cur_xy
    return {
        "max_heading_change_deg": round(max_angle, 6),
        "threshold_deg": float(config.horizontal_reversal_threshold_deg),
        "max_heading_change_frame": max_at_frame,
        "anchor_neighborhood_frames": int(config.anchor_neighborhood_frames),
    }


def _speed_continuity_report(
    samples: Sequence[_Sample],
    config: FlightSanityConfig,
    solver_config: BallArcSolverConfig,
) -> dict[str, Any]:
    velocities = _velocity_vectors(samples)
    speeds = [_norm3(vector) for vector in velocities]
    max_jump = 0.0
    max_at_frame: int | None = None
    start = int(config.anchor_neighborhood_frames)
    end = max(start, len(speeds) - int(config.anchor_neighborhood_frames))
    for index in range(max(1, start), end):
        jump = abs(speeds[index] - speeds[index - 1])
        if jump > max_jump:
            max_jump = jump
            max_at_frame = samples[index].frame
    return {
        "max_speed_jump_mps": round(max_jump, 6),
        "limit_mps": round(config.speed_jump_limit_mps(solver_config), 6),
        "max_speed_jump_frame": max_at_frame,
        "anchor_neighborhood_frames": int(config.anchor_neighborhood_frames),
    }


def _velocity_vectors(samples: Sequence[_Sample]) -> list[tuple[float, float, float]]:
    velocities: list[tuple[float, float, float]] = []
    for left, right in zip(samples, samples[1:], strict=False):
        dt = right.t - left.t
        if dt <= 1e-9:
            velocities.append((0.0, 0.0, 0.0))
            continue
        velocities.append(
            (
                (right.world_xyz[0] - left.world_xyz[0]) / dt,
                (right.world_xyz[1] - left.world_xyz[1]) / dt,
                (right.world_xyz[2] - left.world_xyz[2]) / dt,
            )
        )
    return velocities


def _samples_for_segment(frames: Sequence[Any], *, fps: float, start_t: float, end_t: float) -> list[_Sample]:
    lo = min(start_t, end_t) - 1e-9
    hi = max(start_t, end_t) + 1e-9
    samples: list[_Sample] = []
    for index, raw_frame in enumerate(frames):
        frame = raw_frame if isinstance(raw_frame, Mapping) else {}
        t = _frame_t(frame, index, fps)
        if t < lo or t > hi:
            continue
        world = _world_xyz(frame)
        if world is None:
            continue
        samples.append(_Sample(frame=index, t=t, world_xyz=world))
    return samples


def _anchors(arc_solved: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in list(arc_solved.get("anchors") or []):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("kind") or "") not in {"bounce", "contact"}:
            continue
        t = _float_or_none(item.get("t"))
        frame = _int_or_none(item.get("frame"))
        world = _world_xyz(item)
        if t is None or frame is None or world is None:
            continue
        output.append(
            {
                "anchor_id": str(item.get("anchor_id") or f"anchor_{len(output):03d}"),
                "kind": str(item.get("kind") or ""),
                "status": str(item.get("status") or ""),
                "frame": frame,
                "t": t,
                "world_xyz": world,
            }
        )
    return sorted(output, key=lambda anchor: (anchor["t"], anchor["frame"], anchor["anchor_id"]))


def _anchor_ref(anchor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_id": str(anchor.get("anchor_id") or ""),
        "kind": str(anchor.get("kind") or ""),
        "status": str(anchor.get("status") or ""),
        "frame": int(anchor.get("frame") or 0),
        "t": round(float(anchor.get("t") or 0.0), 9),
    }


def _summary_with_demotions(raw_summary: Any, frames: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(raw_summary or {}) if isinstance(raw_summary, Mapping) else {}
    counts = {key: 0 for key in _BAND_COUNT_KEYS.values()}
    for frame in frames:
        band = str(frame.get("band") or "hidden")
        key = _BAND_COUNT_KEYS.get(band)
        if key is not None:
            counts[key] += 1
    summary.update(counts)
    report_summary = dict(report.get("summary") or {})
    summary["flight_sanity_demoted_frame_count"] = int(report_summary.get("demoted_frame_count") or 0)
    summary["flight_sanity_failed_segment_count"] = int(report_summary.get("failed_segment_count") or 0)
    return summary


def _demoted_frames(report: Mapping[str, Any]) -> dict[int, list[str]]:
    demoted: dict[int, list[str]] = {}
    for item in list(report.get("frames") or []):
        if not isinstance(item, Mapping) or item.get("demote") is not True:
            continue
        frame = _int_or_none(item.get("frame"))
        if frame is None:
            continue
        reasons = item.get("reasons")
        demoted[frame] = [str(reason) for reason in reasons] if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) else []
    return demoted


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    width = max(1, int(window))
    half = width // 2
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        output.append(sum(float(value) for value in values[start:end]) / max(end - start, 1))
    return output


def _physics_from_artifact(arc_solved: Mapping[str, Any]) -> PhysicsParameters:
    raw = arc_solved.get("physics_parameters")
    if not isinstance(raw, Mapping):
        return PhysicsParameters()
    ball_type = str(raw.get("ball_type") or "outdoor")
    return PhysicsParameters.for_ball_type(ball_type)


def _fps(arc_solved: Mapping[str, Any], frames: Sequence[Any]) -> float:
    fps = _float_or_none(arc_solved.get("fps"))
    if fps is not None and fps > 0.0:
        return fps
    times = [_float_or_none(frame.get("t")) for frame in frames if isinstance(frame, Mapping)]
    finite = [time for time in times if time is not None]
    if len(finite) >= 2:
        dt = finite[1] - finite[0]
        if dt > 1e-9:
            return 1.0 / dt
    return 30.0


def _frame_t(frame: Mapping[str, Any], index: int, fps: float) -> float:
    value = _float_or_none(frame.get("t"))
    if value is not None:
        return value
    return index / max(float(fps), 1e-9)


def _world_xyz(frame: Mapping[str, Any]) -> tuple[float, float, float] | None:
    raw = frame.get("world_xyz")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        return None
    try:
        xyz = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(value) for value in xyz) else None


def _has_world_xyz(frame: Mapping[str, Any]) -> bool:
    return _world_xyz(frame) is not None


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm2(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _angle_deg(left: tuple[float, float], right: tuple[float, float]) -> float:
    denom = _norm2(left) * _norm2(right)
    if denom <= 1e-12:
        return 0.0
    value = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denom))
    return math.degrees(math.acos(value))


__all__ = [
    "ARTIFACT_TYPE",
    "DEMOTED_BAND",
    "FlightSanityConfig",
    "apply_flight_sanity_demotions",
    "apply_product_view_flight_sanity_demotions",
    "evaluate_ball_flight_sanity",
]
