"""Render-only physics gap fill for sparse 3D ball world samples.

This module intentionally does not feed BALL detection metrics or gates. It
preserves confident world samples and marks every generated sample as
``source="physics_interpolated"`` with a low-confidence trust band.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ball_physics3d import BounceArcReconstruction
from .io_decode import frame_time_lookup, time_for_frame


LANE = "PHYS-BALLFILL"
ARTIFACT_TYPE = "racketsport_ball_track_physics_filled"


@dataclass(frozen=True)
class PhysicsFillConfig:
    gravity_mps2: float = 9.81
    drag_per_s: float = 0.0
    min_confidence: float = 0.70
    low_confidence: float = 0.50
    min_segment_samples: int = 4
    max_local_segment_samples: int = 8
    max_anchor_gap_frames: int = 12
    max_anchor_gap_s: float | None = None
    max_anchor_speed_mps: float = 45.0
    max_fit_rms_m: float = 0.35
    max_fit_max_residual_m: float = 0.75
    max_reprojection_error_px: float = 18.0
    max_extrapolate_frames: int = 2
    base_uncertainty_m: float = 0.05
    uncertainty_per_frame_m: float = 0.04
    max_xy_interpolate_gap_frames: int = 8
    xy_interpolate_base_uncertainty_px: float = 4.0
    xy_interpolate_per_frame_px: float = 2.0
    max_unreviewed_inflection_speed_px_s: float = 5000.0
    inflection_wrist_tolerance_frames: int = 3
    bounce_z_epsilon_m: float = 0.06
    floor_tolerance_m: float = 0.05
    restitution_bounds: tuple[float, float] = (0.3, 0.5)

    def __post_init__(self) -> None:
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")
        if self.drag_per_s < 0.0:
            raise ValueError("drag_per_s must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.low_confidence <= 1.0:
            raise ValueError("low_confidence must be in [0, 1]")
        if self.min_segment_samples < 3:
            raise ValueError("min_segment_samples must be at least 3")
        if self.max_local_segment_samples < self.min_segment_samples:
            raise ValueError("max_local_segment_samples must be >= min_segment_samples")
        if self.max_anchor_gap_frames < 1:
            raise ValueError("max_anchor_gap_frames must be positive")
        if self.max_anchor_gap_s is not None and self.max_anchor_gap_s <= 0.0:
            raise ValueError("max_anchor_gap_s must be positive when provided")
        if self.max_anchor_speed_mps <= 0.0:
            raise ValueError("max_anchor_speed_mps must be positive")
        if self.max_fit_rms_m < 0.0 or self.max_fit_max_residual_m < 0.0:
            raise ValueError("fit residual thresholds must be non-negative")
        if self.max_reprojection_error_px < 0.0:
            raise ValueError("max_reprojection_error_px must be non-negative")
        if self.max_extrapolate_frames < 0:
            raise ValueError("max_extrapolate_frames must be non-negative")
        if self.base_uncertainty_m < 0.0 or self.uncertainty_per_frame_m < 0.0:
            raise ValueError("uncertainty parameters must be non-negative")
        if self.max_xy_interpolate_gap_frames < 1:
            raise ValueError("max_xy_interpolate_gap_frames must be positive")
        if self.xy_interpolate_base_uncertainty_px < 0.0 or self.xy_interpolate_per_frame_px < 0.0:
            raise ValueError("xy interpolation uncertainty parameters must be non-negative")
        if self.max_unreviewed_inflection_speed_px_s <= 0.0:
            raise ValueError("max_unreviewed_inflection_speed_px_s must be positive")
        if self.inflection_wrist_tolerance_frames < 0:
            raise ValueError("inflection_wrist_tolerance_frames must be non-negative")
        low, high = self.restitution_bounds
        if low < 0.0 or high < low:
            raise ValueError("restitution_bounds must be ordered non-negative values")


@dataclass(frozen=True)
class WorldSample:
    frame_index: int
    t: float
    xyz: tuple[float, float, float]
    xy: tuple[float, float] | None
    conf: float
    visible: bool


@dataclass(frozen=True)
class BallisticFit:
    t0: float
    x0: float
    y0: float
    z0: float
    vx0: float
    vy0: float
    vz0: float
    gravity_mps2: float
    drag_per_s: float
    rms_residual_m: float
    max_residual_m: float

    @property
    def model_name(self) -> str:
        if self.drag_per_s > 0.0:
            return "fixed_gravity_linear_drag"
        return "fixed_gravity_no_drag"

    def predict(self, t: float) -> tuple[float, float, float]:
        dt = float(t) - self.t0
        basis = _drag_basis(dt, self.drag_per_s)
        if self.drag_per_s > 0.0:
            gravity_term = (self.gravity_mps2 / self.drag_per_s) * (basis - dt)
        else:
            gravity_term = -0.5 * self.gravity_mps2 * dt * dt
        return (
            self.x0 + self.vx0 * basis,
            self.y0 + self.vy0 * basis,
            self.z0 + self.vz0 * basis + gravity_term,
        )


@dataclass(frozen=True)
class PhysicsSegment:
    segment_id: int
    frame_start: int
    frame_end: int
    t_start: float
    t_end: float
    samples: tuple[WorldSample, ...]
    fit: BallisticFit

    @property
    def sample_frame_indices(self) -> tuple[int, ...]:
        return tuple(sample.frame_index for sample in self.samples)


@dataclass(frozen=True)
class SegmentFitResult:
    segments: tuple[PhysicsSegment, ...]
    bounce_boundaries: tuple[dict[str, Any], ...]
    confident_world_sample_count: int
    rejected_segment_count: int
    notes: tuple[str, ...] = ()


def fit_ballistic_segments(
    ball_payload: Mapping[str, Any],
    *,
    config: PhysicsFillConfig | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    ball_inflections: Mapping[str, Any] | None = None,
    wrist_velocity_peaks: Mapping[str, Any] | None = None,
) -> SegmentFitResult:
    """Fit fixed-gravity flight segments to confident world-coordinate samples."""

    cfg = config or PhysicsFillConfig()
    frames = _frame_list(ball_payload)
    fps = _payload_fps(ball_payload, frames)
    samples = _confident_world_samples(frames, cfg)
    forced_boundaries, forced_notes = _forced_boundary_records(
        reviewed_bounces=reviewed_bounces,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
        cfg=cfg,
        fps=fps,
    )
    if len(samples) < cfg.min_segment_samples:
        return SegmentFitResult(
            segments=(),
            bounce_boundaries=tuple(forced_boundaries),
            confident_world_sample_count=len(samples),
            rejected_segment_count=0,
            notes=(*forced_notes, f"need at least {cfg.min_segment_samples} confident world samples"),
        )

    segments: list[PhysicsSegment] = []
    bounces: list[dict[str, Any]] = []
    rejected_count = 0
    notes: list[str] = list(forced_notes)

    for run in _initial_sample_runs(samples, cfg, fps=fps):
        forced_runs, forced_run_bounces = _split_run_at_forced_boundaries(run, forced_boundaries)
        bounces.extend(forced_run_bounces)
        for forced_run in forced_runs:
            split_runs, run_bounces = _split_run_at_bounces(forced_run, cfg)
            bounces.extend(run_bounces)
            for split_run in split_runs:
                if len(split_run) < cfg.min_segment_samples:
                    continue
                fit = _fit_ballistic_model(split_run, cfg)
                if _fit_is_accepted(split_run, fit, cfg):
                    segments.append(_make_segment(segment_id=len(segments), samples=split_run, fit=fit))
                    continue

                local_segments = _fit_local_segments(split_run, cfg, first_segment_id=len(segments))
                if local_segments:
                    segments.extend(local_segments)
                    continue
                rejected_count += 1

    bounces = _merge_missing_forced_boundaries(bounces, forced_boundaries)

    return SegmentFitResult(
        segments=tuple(segments),
        bounce_boundaries=tuple(bounces),
        confident_world_sample_count=len(samples),
        rejected_segment_count=rejected_count,
        notes=tuple(notes),
    )


def fill_ball_track_physics(
    ball_payload: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    config: PhysicsFillConfig | None = None,
    evidence_path: str | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    ball_inflections: Mapping[str, Any] | None = None,
    wrist_velocity_peaks: Mapping[str, Any] | None = None,
    physics3d_reconstruction: BounceArcReconstruction | None = None,
    frame_times: Any = None,
) -> dict[str, Any]:
    """Return an additive render-only ball track with physics-filled samples."""

    cfg = config or PhysicsFillConfig()
    output = copy.deepcopy(dict(ball_payload))
    frames = output.get("frames")
    if not isinstance(frames, list):
        raise ValueError("ball payload must contain a frames list")

    original_frames = _frame_list(ball_payload)
    frame_time_map = frame_time_lookup(frame_times if frame_times is not None else ball_payload.get("frame_times"))
    payload_fps = _payload_fps(ball_payload, original_frames)
    fit_result = fit_ballistic_segments(
        ball_payload,
        config=cfg,
        reviewed_bounces=reviewed_bounces,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
    )
    filled_frame_count = 0
    filled_missing_world_count = 0
    filled_low_conf_count = 0
    physics3d_reconstructed_frame_count = 0
    lifted_2d_frame_count = 0
    xy_interpolated_frame_count = 0
    measured_xy_no_world_frame_count = 0
    short_gap_xy_interpolated_frame_count = 0
    extrapolated_frame_count = 0
    reprojection_rejected_frame_count = 0
    no_calibration_2d_rejected_frame_count = 0
    below_floor_rejected_frame_count = 0
    clamped_to_court_plane_frame_count = 0

    filled_indices: set[int] = set()
    rejection_reasons: dict[int, str] = {}

    physics3d_reconstructed_frame_count = _apply_physics3d_reconstruction(
        frames=frames,
        original_frames=original_frames,
        reconstruction=physics3d_reconstruction,
        filled_indices=filled_indices,
        evidence_path=evidence_path,
        cfg=cfg,
    )

    for segment in fit_result.segments:
        candidate_start = max(0, segment.frame_start - cfg.max_extrapolate_frames)
        candidate_end = min(len(frames) - 1, segment.frame_end + cfg.max_extrapolate_frames)
        for frame_index in range(candidate_start, candidate_end + 1):
            if frame_index in filled_indices:
                continue
            original = original_frames[frame_index]
            if _is_confident_world_frame(original, cfg):
                continue

            frame = frames[frame_index]
            if not isinstance(frame, dict):
                continue
            try:
                t = float(
                    original.get(
                        "t",
                        time_for_frame(frame_index, frame_times=frame_time_map, fps=payload_fps),
                    )
                )
            except (TypeError, ValueError):
                continue

            predicted = segment.fit.predict(t)
            if predicted[2] < -cfg.floor_tolerance_m:
                below_floor_rejected_frame_count += 1
                rejection_reasons[frame_index] = "predicted_below_court_plane"
                continue
            rendered_xyz = (
                predicted[0],
                predicted[1],
                max(0.0, predicted[2]),
            )
            clamped_to_court_plane = rendered_xyz[2] != predicted[2]
            if clamped_to_court_plane:
                clamped_to_court_plane_frame_count += 1

            reprojection_error_px: float | None = None
            original_xy = _xy_tuple(original)
            visible_2d = original.get("visible") is True and original_xy is not None
            if visible_2d:
                if calibration is None:
                    no_calibration_2d_rejected_frame_count += 1
                    rejection_reasons[frame_index] = "missing_calibration_for_2d_lift"
                    continue
                projected_xy = _project_world_to_image(calibration, rendered_xyz)
                if projected_xy is None:
                    no_calibration_2d_rejected_frame_count += 1
                    rejection_reasons[frame_index] = "unprojectable_calibration_for_2d_lift"
                    continue
                reprojection_error_px = _distance2(projected_xy, original_xy)
                if reprojection_error_px > cfg.max_reprojection_error_px:
                    reprojection_rejected_frame_count += 1
                    rejection_reasons[frame_index] = "reprojection_error_above_threshold"
                    continue

            if original.get("world_xyz") is None:
                filled_missing_world_count += 1
            else:
                filled_low_conf_count += 1
                frame["original_world_xyz"] = list(original["world_xyz"])
            if visible_2d:
                lifted_2d_frame_count += 1

            inside_segment = segment.frame_start <= frame_index <= segment.frame_end
            if not inside_segment:
                extrapolated_frame_count += 1
            uncertainty_m, gap_distance_frames = _uncertainty_for_frame(frame_index, segment, cfg)
            frame["world_xyz"] = [float(rendered_xyz[0]), float(rendered_xyz[1]), float(rendered_xyz[2])]
            frame["source"] = "physics_interpolated"
            frame["approx"] = True
            frame["trust_band"] = _physics_trust_band(evidence_path)
            frame["physics_fill"] = {
                "lane": LANE,
                "segment_id": segment.segment_id,
                "source_stage": LANE,
                "model": segment.fit.model_name,
                "render_only": True,
                "not_for_detection_metrics": True,
                "inside_segment": inside_segment,
                "extrapolated_frames": _extrapolated_frames(frame_index, segment),
                "gap_distance_frames": gap_distance_frames,
                "uncertainty_m": uncertainty_m,
                "raw_world_xyz": [float(predicted[0]), float(predicted[1]), float(predicted[2])],
                "clamped_to_court_plane": clamped_to_court_plane,
                "fit_rms_residual_m": segment.fit.rms_residual_m,
                "fit_max_residual_m": segment.fit.max_residual_m,
                "reprojection_error_px": reprojection_error_px,
            }
            frame["render_uncertainty_m"] = uncertainty_m
            filled_frame_count += 1
            filled_indices.add(frame_index)

    xy_counts = _apply_xy_interpolation(frames=frames, original_frames=original_frames, cfg=cfg)
    xy_interpolated_frame_count = xy_counts["xy_interpolated_frame_count"]
    measured_xy_no_world_frame_count = xy_counts["measured_xy_no_world_frame_count"]
    short_gap_xy_interpolated_frame_count = xy_counts["short_gap_xy_interpolated_frame_count"]

    output["physics_fill"] = {
        "artifact_type": ARTIFACT_TYPE,
        "lane": LANE,
        "render_only": True,
        "not_for_detection_metrics": True,
        "source_policy": (
            "confident world samples are preserved; generated samples are marked "
            "source=physics_interpolated and low_confidence"
        ),
        "config": _config_summary(cfg),
        "segments": [_segment_summary(segment) for segment in fit_result.segments],
        "bounce_boundaries": list(fit_result.bounce_boundaries),
        "coverage": {
            "input_frame_count": len(original_frames),
            "input_visible_count": sum(1 for frame in original_frames if frame.get("visible") is True),
            "input_world_xyz_count": sum(1 for frame in original_frames if _world_xyz_tuple(frame) is not None),
            "confident_world_sample_count": fit_result.confident_world_sample_count,
            "output_world_xyz_count": sum(1 for frame in frames if isinstance(frame, dict) and _world_xyz_tuple(frame) is not None),
            "filled_frame_count": filled_frame_count,
            "filled_missing_world_count": filled_missing_world_count,
            "filled_low_conf_count": filled_low_conf_count,
            "physics3d_reconstructed_frame_count": physics3d_reconstructed_frame_count,
            "lifted_2d_frame_count": lifted_2d_frame_count,
            "xy_interpolated_frame_count": xy_interpolated_frame_count,
            "measured_xy_no_world_frame_count": measured_xy_no_world_frame_count,
            "short_gap_xy_interpolated_frame_count": short_gap_xy_interpolated_frame_count,
            "extrapolated_frame_count": extrapolated_frame_count,
            "reprojection_rejected_frame_count": reprojection_rejected_frame_count,
            "no_calibration_2d_rejected_frame_count": no_calibration_2d_rejected_frame_count,
            "below_floor_rejected_frame_count": below_floor_rejected_frame_count,
            "clamped_to_court_plane_frame_count": clamped_to_court_plane_frame_count,
            "rejected_segment_count": fit_result.rejected_segment_count,
        },
        "rejected_frames": [
            {"frame_index": frame_index, "reason": reason}
            for frame_index, reason in sorted(rejection_reasons.items())
        ],
        "notes": list(fit_result.notes),
    }
    return output


def validate_physics_fill(
    ball_payload: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    config: PhysicsFillConfig | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    ball_inflections: Mapping[str, Any] | None = None,
    wrist_velocity_peaks: Mapping[str, Any] | None = None,
    seed: int = 0,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run deterministic leave-one-out validation on confident world samples."""

    cfg = config or PhysicsFillConfig()
    fit_result = fit_ballistic_segments(
        ball_payload,
        config=cfg,
        reviewed_bounces=reviewed_bounces,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
    )
    candidates: list[tuple[PhysicsSegment, WorldSample]] = []
    seen_frames: set[int] = set()
    for segment in fit_result.segments:
        for sample in segment.samples:
            if sample.frame_index in seen_frames:
                continue
            candidates.append((segment, sample))
            seen_frames.add(sample.frame_index)
    if max_samples is not None and max_samples < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, max_samples)
        candidates.sort(key=lambda item: item[1].frame_index)

    errors_3d: list[float] = []
    errors_2d: list[float] = []
    skipped: list[dict[str, Any]] = []

    for segment, held_out in candidates:
        retained = [sample for sample in segment.samples if sample.frame_index != held_out.frame_index]
        if len(retained) < cfg.min_segment_samples:
            skipped.append({"frame_index": held_out.frame_index, "reason": "insufficient_segment_samples_after_drop"})
            continue
        fit = _fit_ballistic_model(retained, cfg)
        predicted = fit.predict(held_out.t)
        errors_3d.append(_distance3(predicted, held_out.xyz))
        if calibration is not None and held_out.xy is not None:
            projected = _project_world_to_image(calibration, predicted)
            if projected is not None:
                errors_2d.append(_distance2(projected, held_out.xy))

    return {
        "lane": LANE,
        "render_only": True,
        "not_for_detection_metrics": True,
        "segments": [_segment_summary(segment) for segment in fit_result.segments],
        "bounce_boundaries": list(fit_result.bounce_boundaries),
        "leave_one_out": {
            "candidate_count": len(candidates),
            "sample_count": len(errors_3d),
            "skipped": skipped,
            "error_3d_m": _distribution(errors_3d),
            "reprojection_error_px": _distribution(errors_2d),
        },
    }


def _frame_list(ball_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = ball_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("ball payload must contain a frames list")
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _payload_fps(ball_payload: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> float:
    try:
        fps = float(ball_payload.get("fps"))
        if fps > 0.0:
            return fps
    except (TypeError, ValueError):
        pass
    times = []
    for frame in frames:
        try:
            times.append(float(frame["t"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(times) >= 2:
        deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
        if deltas:
            return 1.0 / (sum(deltas) / len(deltas))
    return 30.0


def _confident_world_samples(frames: Sequence[Mapping[str, Any]], cfg: PhysicsFillConfig) -> list[WorldSample]:
    samples: list[WorldSample] = []
    for index, frame in enumerate(frames):
        if not _is_confident_world_frame(frame, cfg):
            continue
        xyz = _world_xyz_tuple(frame)
        if xyz is None:
            continue
        try:
            t = float(frame["t"])
            conf = float(frame.get("conf", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        samples.append(
            WorldSample(
                frame_index=index,
                t=t,
                xyz=xyz,
                xy=_xy_tuple(frame),
                conf=conf,
                visible=frame.get("visible") is True,
            )
        )
    return sorted(samples, key=lambda sample: (sample.frame_index, sample.t))


def _is_confident_world_frame(frame: Mapping[str, Any], cfg: PhysicsFillConfig) -> bool:
    if frame.get("source") == "physics_interpolated":
        return False
    if frame.get("visible") is not True:
        return False
    try:
        conf = float(frame.get("conf", 0.0))
    except (TypeError, ValueError):
        return False
    return conf >= cfg.min_confidence and _world_xyz_tuple(frame) is not None


def _world_xyz_tuple(frame: Mapping[str, Any]) -> tuple[float, float, float] | None:
    value = frame.get("world_xyz")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        xyz = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in xyz):
        return None
    return xyz


def _xy_tuple(frame: Mapping[str, Any]) -> tuple[float, float] | None:
    value = frame.get("xy")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in xy):
        return None
    return xy


def _frame_index_from_mapping(item: Mapping[str, Any]) -> int | None:
    for key in ("frame_index", "frame"):
        value = item.get(key)
        try:
            frame_index = int(value)
        except (TypeError, ValueError):
            continue
        if frame_index >= 0:
            return frame_index
    return None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _has_wrist_peak_near(
    payload: Mapping[str, Any] | None,
    *,
    frame_index: int,
    t: float | None,
    fps: float,
    tolerance_frames: int,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    peaks = payload.get("peaks")
    if not isinstance(peaks, list):
        return False
    tolerance_s = tolerance_frames / max(fps, 1e-9)
    for item in peaks:
        if not isinstance(item, Mapping):
            continue
        peak_frame = _frame_index_from_mapping(item)
        if peak_frame is not None and abs(peak_frame - frame_index) <= tolerance_frames:
            return True
        peak_t = _optional_float(item.get("t"))
        if t is not None and peak_t is not None and abs(peak_t - t) <= tolerance_s:
            return True
    return False


def _initial_sample_runs(
    samples: Sequence[WorldSample],
    cfg: PhysicsFillConfig,
    *,
    fps: float,
) -> list[list[WorldSample]]:
    runs: list[list[WorldSample]] = []
    current: list[WorldSample] = []
    max_gap_s = cfg.max_anchor_gap_s or (cfg.max_anchor_gap_frames + 0.5) / max(fps, 1e-9)
    previous: WorldSample | None = None
    for sample in samples:
        if previous is not None:
            frame_gap = sample.frame_index - previous.frame_index
            time_gap = sample.t - previous.t
            speed = _distance3(sample.xyz, previous.xyz) / max(time_gap, 1e-9)
            if (
                frame_gap > cfg.max_anchor_gap_frames
                or time_gap > max_gap_s
                or speed > cfg.max_anchor_speed_mps
            ):
                if current:
                    runs.append(current)
                current = []
        current.append(sample)
        previous = sample
    if current:
        runs.append(current)
    return runs


def _split_run_at_bounces(
    run: Sequence[WorldSample],
    cfg: PhysicsFillConfig,
) -> tuple[list[list[WorldSample]], list[dict[str, Any]]]:
    if len(run) < 3:
        return [list(run)], []
    split_offsets: list[int] = []
    bounces: list[dict[str, Any]] = []
    low_bound, high_bound = cfg.restitution_bounds

    for offset in range(1, len(run) - 1):
        prev_sample = run[offset - 1]
        sample = run[offset]
        next_sample = run[offset + 1]
        before_dt = sample.t - prev_sample.t
        after_dt = next_sample.t - sample.t
        if before_dt <= 0.0 or after_dt <= 0.0:
            continue
        vz_before = (sample.xyz[2] - prev_sample.xyz[2]) / before_dt
        vz_after = (next_sample.xyz[2] - sample.xyz[2]) / after_dt
        local_minimum = sample.xyz[2] <= prev_sample.xyz[2] and sample.xyz[2] <= next_sample.xyz[2]
        sign_change = vz_before < 0.0 and vz_after > 0.0
        near_court = sample.xyz[2] <= cfg.bounce_z_epsilon_m
        if not (local_minimum and sign_change and near_court):
            continue
        restitution = vz_after / abs(vz_before) if abs(vz_before) > 1e-9 else None
        split_offsets.append(offset)
        bounces.append(
            {
                "frame_index": sample.frame_index,
                "t": sample.t,
                "world_xyz": list(sample.xyz),
                "incoming_vz_mps": vz_before,
                "outgoing_vz_mps": vz_after,
                "estimated_restitution": restitution,
                "restitution_bounds": [low_bound, high_bound],
                "within_restitution_bounds": (
                    restitution is not None and low_bound <= restitution <= high_bound
                ),
            }
        )

    if not split_offsets:
        return [list(run)], []

    split_runs: list[list[WorldSample]] = []
    start = 0
    for offset in split_offsets:
        split_runs.append(list(run[start : offset + 1]))
        start = offset
    split_runs.append(list(run[start:]))
    return split_runs, bounces


def _forced_boundary_records(
    *,
    reviewed_bounces: Mapping[str, Any] | None,
    ball_inflections: Mapping[str, Any] | None,
    wrist_velocity_peaks: Mapping[str, Any] | None,
    cfg: PhysicsFillConfig,
    fps: float,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    records.extend(_reviewed_bounce_records(reviewed_bounces))
    inflection_records, inflection_notes = _unreviewed_inflection_records(
        ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
        cfg=cfg,
        fps=fps,
    )
    records.extend(inflection_records)
    notes.extend(inflection_notes)

    deduped: dict[int, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (int(item["frame_index"]), str(item.get("source", "")))):
        frame_index = int(record["frame_index"])
        existing = deduped.get(frame_index)
        if existing is None or existing.get("source") != "human_reviewed":
            deduped[frame_index] = record
    return list(deduped.values()), tuple(notes)


def _reviewed_bounce_records(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    if payload.get("status") != "human_reviewed":
        return []
    bounces = payload.get("bounces")
    if not isinstance(bounces, list):
        return []
    records: list[dict[str, Any]] = []
    for item in bounces:
        if not isinstance(item, Mapping):
            continue
        frame_index = _frame_index_from_mapping(item)
        if frame_index is None:
            continue
        record: dict[str, Any] = {
            "frame_index": frame_index,
            "source": "human_reviewed",
            "forced_split": True,
        }
        try:
            record["t"] = float(item["t"])
        except (KeyError, TypeError, ValueError):
            pass
        review_id = item.get("review_id")
        if isinstance(review_id, str) and review_id:
            record["review_id"] = review_id
        records.append(record)
    return records


def _unreviewed_inflection_records(
    payload: Mapping[str, Any] | None,
    *,
    wrist_velocity_peaks: Mapping[str, Any] | None,
    cfg: PhysicsFillConfig,
    fps: float,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        return [], ()
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = payload.get("inflections")
    if not isinstance(candidates, list):
        return [], ()

    records: list[dict[str, Any]] = []
    notes: list[str] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        frame_index = _frame_index_from_mapping(item)
        if frame_index is None:
            continue
        speed = _optional_float(item.get("speed_before_px_s"))
        if speed is not None and speed > cfg.max_unreviewed_inflection_speed_px_s:
            notes.append(
                f"ignored unreviewed inflection frame {frame_index}: speed_before_px_s={speed:.3f} "
                f"exceeds sanity clamp {cfg.max_unreviewed_inflection_speed_px_s:.3f}"
            )
            continue
        t_value = _optional_float(item.get("t"))
        if not _has_wrist_peak_near(
            wrist_velocity_peaks,
            frame_index=frame_index,
            t=t_value,
            fps=fps,
            tolerance_frames=cfg.inflection_wrist_tolerance_frames,
        ):
            notes.append(f"ignored unreviewed inflection frame {frame_index}: missing wrist_velocity_peaks cross-check")
            continue
        records.append(
            {
                "frame_index": frame_index,
                "t": t_value,
                "source": "unreviewed_inflection_wrist_cross_checked",
                "forced_split": True,
                "speed_before_px_s": speed,
                "wrist_tolerance_frames": cfg.inflection_wrist_tolerance_frames,
            }
        )
    return records, tuple(notes)


def _split_run_at_forced_boundaries(
    run: Sequence[WorldSample],
    boundaries: Sequence[Mapping[str, Any]],
) -> tuple[list[list[WorldSample]], list[dict[str, Any]]]:
    if not run or not boundaries:
        return [list(run)], []

    chunks: list[list[WorldSample]] = [list(run)]
    emitted: list[dict[str, Any]] = []
    for boundary in sorted(boundaries, key=lambda item: int(item["frame_index"])):
        frame_index = int(boundary["frame_index"])
        next_chunks: list[list[WorldSample]] = []
        touched = False
        for chunk in chunks:
            if not chunk or frame_index < chunk[0].frame_index or frame_index > chunk[-1].frame_index:
                next_chunks.append(chunk)
                continue
            split = _split_chunk_at_frame_boundary(chunk, frame_index)
            next_chunks.extend(split)
            touched = True
        chunks = next_chunks
        if touched:
            emitted.append(dict(boundary))
    return chunks, emitted


def _merge_missing_forced_boundaries(
    emitted: Sequence[Mapping[str, Any]],
    forced_boundaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = [dict(item) for item in emitted]
    seen = {
        (int(item["frame_index"]), str(item.get("source", "")))
        for item in merged
        if "frame_index" in item
    }
    for boundary in forced_boundaries:
        if boundary.get("source") != "human_reviewed":
            continue
        key = (int(boundary["frame_index"]), str(boundary.get("source", "")))
        if key in seen:
            continue
        merged.append(dict(boundary))
        seen.add(key)
    return sorted(merged, key=lambda item: (int(item["frame_index"]), str(item.get("source", ""))))


def _split_chunk_at_frame_boundary(chunk: Sequence[WorldSample], frame_index: int) -> list[list[WorldSample]]:
    exact_offset = next((offset for offset, sample in enumerate(chunk) if sample.frame_index == frame_index), None)
    if exact_offset is not None:
        if exact_offset == 0 or exact_offset == len(chunk) - 1:
            return [list(chunk)]
        return [list(chunk[: exact_offset + 1]), list(chunk[exact_offset:])]
    insert_at = next((offset for offset, sample in enumerate(chunk) if sample.frame_index > frame_index), len(chunk))
    if insert_at <= 0 or insert_at >= len(chunk):
        return [list(chunk)]
    return [list(chunk[:insert_at]), list(chunk[insert_at:])]


def _apply_physics3d_reconstruction(
    *,
    frames: list[Any],
    original_frames: Sequence[Mapping[str, Any]],
    reconstruction: BounceArcReconstruction | None,
    filled_indices: set[int],
    evidence_path: str | None,
    cfg: PhysicsFillConfig,
) -> int:
    if reconstruction is None or reconstruction.status != "ran":
        return 0
    frame_indices = tuple(int(index) for index in reconstruction.frame_indices)
    if len(frame_indices) != len(reconstruction.samples):
        return 0
    count = 0
    uncertainty_m = _physics3d_uncertainty_m(reconstruction, cfg)
    for frame_index, sample in zip(frame_indices, reconstruction.samples, strict=True):
        if frame_index < 0 or frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        if not isinstance(frame, dict):
            continue
        original = original_frames[frame_index]
        predicted = [float(sample.x), float(sample.y), float(sample.z)]
        if predicted[2] < -cfg.floor_tolerance_m:
            continue
        rendered = [predicted[0], predicted[1], max(0.0, predicted[2])]
        original_world = _world_xyz_tuple(original)
        if original_world is not None and "original_world_xyz" not in frame:
            frame["original_world_xyz"] = list(original_world)
        frame["world_xyz"] = rendered
        frame["source"] = "physics3d_reconstructed"
        frame["approx"] = True
        frame["trust_band"] = _physics_trust_band(evidence_path)
        frame["physics_fill"] = {
            "lane": LANE,
            "source_stage": "ball_physics3d",
            "model": "calibrated_two_arc_bounce",
            "render_only": True,
            "not_for_detection_metrics": True,
            "uncertainty_m": uncertainty_m,
            "raw_world_xyz": predicted,
            "clamped_to_court_plane": rendered[2] != predicted[2],
            "reprojection_rmse_px": reconstruction.reprojection_rmse_px,
            "max_reprojection_error_px": reconstruction.max_reprojection_error_px,
            "candidate_count": reconstruction.candidate_count,
        }
        frame["render_uncertainty_m"] = uncertainty_m
        filled_indices.add(frame_index)
        count += 1
    return count


def _physics3d_uncertainty_m(reconstruction: BounceArcReconstruction, cfg: PhysicsFillConfig) -> float:
    reproj = reconstruction.reprojection_rmse_px
    reproj_component = 0.0 if reproj is None else min(0.20, max(0.0, float(reproj)) * 0.01)
    return math.sqrt(cfg.base_uncertainty_m * cfg.base_uncertainty_m + reproj_component * reproj_component)


def _apply_xy_interpolation(
    *,
    frames: list[Any],
    original_frames: Sequence[Mapping[str, Any]],
    cfg: PhysicsFillConfig,
) -> dict[str, int]:
    anchors: list[tuple[int, tuple[float, float]]] = [
        (index, xy)
        for index, original in enumerate(original_frames)
        if original.get("visible") is True
        for xy in [_xy_tuple(original)]
        if xy is not None
    ]
    if not anchors:
        return {
            "xy_interpolated_frame_count": 0,
            "measured_xy_no_world_frame_count": 0,
            "short_gap_xy_interpolated_frame_count": 0,
        }
    anchor_by_index = {index: xy for index, xy in anchors}
    measured_count = 0
    interp_count = 0

    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict) or "xy_interpolated" in frame:
            continue
        original = original_frames[frame_index]
        if _world_xyz_tuple(frame) is not None:
            continue
        original_xy = _xy_tuple(original)
        if original.get("visible") is True and original_xy is not None:
            frame["xy_interpolated"] = _xy_payload(
                xy=original_xy,
                source="measured_xy_no_world",
                uncertainty_px=cfg.xy_interpolate_base_uncertainty_px,
                gap_distance_frames=0,
            )
            measured_count += 1
            continue
        interpolated = _interpolate_xy_for_frame(frame_index, anchor_by_index, cfg)
        if interpolated is None:
            continue
        xy, gap_distance, total_gap = interpolated
        frame["xy_interpolated"] = _xy_payload(
            xy=xy,
            source="short_gap_linear_interpolation",
            uncertainty_px=cfg.xy_interpolate_base_uncertainty_px + cfg.xy_interpolate_per_frame_px * gap_distance,
            gap_distance_frames=gap_distance,
            total_gap_frames=total_gap,
        )
        interp_count += 1
    return {
        "xy_interpolated_frame_count": measured_count + interp_count,
        "measured_xy_no_world_frame_count": measured_count,
        "short_gap_xy_interpolated_frame_count": interp_count,
    }


def _interpolate_xy_for_frame(
    frame_index: int,
    anchor_by_index: Mapping[int, tuple[float, float]],
    cfg: PhysicsFillConfig,
) -> tuple[tuple[float, float], int, int] | None:
    prev_indices = [index for index in anchor_by_index if index < frame_index]
    next_indices = [index for index in anchor_by_index if index > frame_index]
    if not prev_indices or not next_indices:
        return None
    prev_index = max(prev_indices)
    next_index = min(next_indices)
    total_gap = next_index - prev_index
    if total_gap <= 0 or total_gap > cfg.max_xy_interpolate_gap_frames:
        return None
    alpha = (frame_index - prev_index) / total_gap
    prev_xy = anchor_by_index[prev_index]
    next_xy = anchor_by_index[next_index]
    xy = (
        prev_xy[0] + alpha * (next_xy[0] - prev_xy[0]),
        prev_xy[1] + alpha * (next_xy[1] - prev_xy[1]),
    )
    gap_distance = min(frame_index - prev_index, next_index - frame_index)
    return xy, gap_distance, total_gap


def _xy_payload(
    *,
    xy: Sequence[float],
    source: str,
    uncertainty_px: float,
    gap_distance_frames: int,
    total_gap_frames: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "xy": [float(xy[0]), float(xy[1])],
        "source": source,
        "render_only": True,
        "not_for_detection_metrics": True,
        "uncertainty_px": float(uncertainty_px),
        "gap_distance_frames": int(gap_distance_frames),
        "trust_band": _xy_trust_band(),
    }
    if total_gap_frames is not None:
        payload["total_gap_frames"] = int(total_gap_frames)
    return payload


def _xy_trust_band() -> dict[str, Any]:
    return {
        "stage": LANE,
        "gate_id": "xy_render_continuity_only",
        "gate_status": "not_a_ball_detection_gate",
        "badge": "low_confidence",
        "reason": "2D trail interpolation is additive render continuity only and never changes measured xy/visible.",
    }


def _fit_ballistic_model(samples: Sequence[WorldSample], cfg: PhysicsFillConfig) -> BallisticFit:
    if len(samples) < 2:
        raise ValueError("at least 2 samples are required for a ballistic fit")
    t0 = samples[0].t
    basis_values = [_drag_basis(sample.t - t0, cfg.drag_per_s) for sample in samples]
    rows = [[1.0, basis] for basis in basis_values]
    x0, vx0 = _least_squares_2(rows, [sample.xyz[0] for sample in samples])
    y0, vy0 = _least_squares_2(rows, [sample.xyz[1] for sample in samples])
    if cfg.drag_per_s > 0.0:
        z_values = [
            sample.xyz[2] - (cfg.gravity_mps2 / cfg.drag_per_s) * (basis - (sample.t - t0))
            for sample, basis in zip(samples, basis_values, strict=True)
        ]
    else:
        z_values = [
            sample.xyz[2] + 0.5 * cfg.gravity_mps2 * (sample.t - t0) * (sample.t - t0)
            for sample in samples
        ]
    z0, vz0 = _least_squares_2(rows, z_values)
    provisional = BallisticFit(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx0=vx0,
        vy0=vy0,
        vz0=vz0,
        gravity_mps2=cfg.gravity_mps2,
        drag_per_s=cfg.drag_per_s,
        rms_residual_m=0.0,
        max_residual_m=0.0,
    )
    residuals = [_distance3(provisional.predict(sample.t), sample.xyz) for sample in samples]
    rms = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    return BallisticFit(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx0=vx0,
        vy0=vy0,
        vz0=vz0,
        gravity_mps2=cfg.gravity_mps2,
        drag_per_s=cfg.drag_per_s,
        rms_residual_m=rms,
        max_residual_m=max(residuals),
    )


def _fit_is_accepted(samples: Sequence[WorldSample], fit: BallisticFit, cfg: PhysicsFillConfig) -> bool:
    if fit.rms_residual_m > cfg.max_fit_rms_m or fit.max_residual_m > cfg.max_fit_max_residual_m:
        return False
    return _min_predicted_z(samples, fit) >= -cfg.floor_tolerance_m


def _fit_local_segments(
    samples: Sequence[WorldSample],
    cfg: PhysicsFillConfig,
    *,
    first_segment_id: int,
) -> list[PhysicsSegment]:
    segments: list[PhysicsSegment] = []
    start = 0
    while start + cfg.min_segment_samples <= len(samples):
        max_end = min(len(samples), start + cfg.max_local_segment_samples)
        accepted: PhysicsSegment | None = None
        for end in range(max_end, start + cfg.min_segment_samples - 1, -1):
            window = list(samples[start:end])
            fit = _fit_ballistic_model(window, cfg)
            if _fit_is_accepted(window, fit, cfg):
                accepted = _make_segment(
                    segment_id=first_segment_id + len(segments),
                    samples=window,
                    fit=fit,
                )
                break
        if accepted is None:
            start += 1
            continue
        segments.append(accepted)
        start = max(start + 1, end - 1)
    return segments


def _make_segment(*, segment_id: int, samples: Sequence[WorldSample], fit: BallisticFit) -> PhysicsSegment:
    return PhysicsSegment(
        segment_id=segment_id,
        frame_start=min(sample.frame_index for sample in samples),
        frame_end=max(sample.frame_index for sample in samples),
        t_start=min(sample.t for sample in samples),
        t_end=max(sample.t for sample in samples),
        samples=tuple(samples),
        fit=fit,
    )


def _drag_basis(dt: float, drag_per_s: float) -> float:
    if drag_per_s <= 1e-9:
        return dt
    return (1.0 - math.exp(-drag_per_s * dt)) / drag_per_s


def _least_squares_2(rows: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, float]:
    if len(rows) != len(values) or not rows:
        raise ValueError("least squares requires paired rows and values")
    s00 = s01 = s11 = b0 = b1 = 0.0
    for row, value in zip(rows, values, strict=True):
        if len(row) != 2:
            raise ValueError("least squares rows must have width 2")
        a0 = float(row[0])
        a1 = float(row[1])
        s00 += a0 * a0
        s01 += a0 * a1
        s11 += a1 * a1
        b0 += a0 * float(value)
        b1 += a1 * float(value)
    det = s00 * s11 - s01 * s01
    if abs(det) < 1e-12:
        raise ValueError("cannot fit singular sample times")
    return ((b0 * s11 - b1 * s01) / det, (s00 * b1 - s01 * b0) / det)


def _min_predicted_z(samples: Sequence[WorldSample], fit: BallisticFit) -> float:
    frame_start = min(sample.frame_index for sample in samples)
    frame_end = max(sample.frame_index for sample in samples)
    t_by_frame = {sample.frame_index: sample.t for sample in samples}
    if frame_end <= frame_start:
        return fit.predict(samples[0].t)[2]
    sampled_z = []
    for frame_index in range(frame_start, frame_end + 1):
        if frame_index in t_by_frame:
            t = t_by_frame[frame_index]
        else:
            alpha = (frame_index - frame_start) / max(frame_end - frame_start, 1)
            t = samples[0].t + alpha * (samples[-1].t - samples[0].t)
        sampled_z.append(fit.predict(t)[2])
    return min(sampled_z)


def _uncertainty_for_frame(
    frame_index: int,
    segment: PhysicsSegment,
    cfg: PhysicsFillConfig,
) -> tuple[float, int]:
    anchor_distance = min(abs(frame_index - sample.frame_index) for sample in segment.samples)
    uncertainty = math.sqrt(
        segment.fit.rms_residual_m * segment.fit.rms_residual_m
        + (cfg.base_uncertainty_m + cfg.uncertainty_per_frame_m * anchor_distance) ** 2
    )
    return uncertainty, anchor_distance


def _extrapolated_frames(frame_index: int, segment: PhysicsSegment) -> int:
    if frame_index < segment.frame_start:
        return segment.frame_start - frame_index
    if frame_index > segment.frame_end:
        return frame_index - segment.frame_end
    return 0


def _project_world_to_image(
    calibration: Mapping[str, Any],
    world_xyz: Sequence[float],
) -> tuple[float, float] | None:
    try:
        intrinsics = calibration["intrinsics"]
        extrinsics = calibration["extrinsics"]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        rotation = extrinsics.get("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        translation = extrinsics.get("t", [0.0, 0.0, 0.0])
        wx, wy, wz = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
        camera_x = sum(float(rotation[0][col]) * (wx, wy, wz)[col] for col in range(3)) + float(translation[0])
        camera_y = sum(float(rotation[1][col]) * (wx, wy, wz)[col] for col in range(3)) + float(translation[1])
        camera_z = sum(float(rotation[2][col]) * (wx, wy, wz)[col] for col in range(3)) + float(translation[2])
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None
    if not all(math.isfinite(value) for value in (camera_x, camera_y, camera_z)) or abs(camera_z) < 1e-9:
        return None
    return (fx * camera_x / camera_z + cx, fy * camera_y / camera_z + cy)


def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _physics_trust_band(evidence_path: str | None) -> dict[str, Any]:
    return {
        "stage": LANE,
        "gate_id": "physics_render_continuity_only",
        "gate_status": "not_a_ball_detection_gate",
        "badge": "low_confidence",
        "reason": (
            "Physics interpolation is for world-rendering continuity only; it must "
            "not be used for BALL detection metrics, gates, training, or promotion."
        ),
        "evidence_path": evidence_path,
    }


def _config_summary(cfg: PhysicsFillConfig) -> dict[str, Any]:
    return {
        "gravity_mps2": cfg.gravity_mps2,
        "drag_per_s": cfg.drag_per_s,
        "min_confidence": cfg.min_confidence,
        "low_confidence": cfg.low_confidence,
        "min_segment_samples": cfg.min_segment_samples,
        "max_local_segment_samples": cfg.max_local_segment_samples,
        "max_anchor_gap_frames": cfg.max_anchor_gap_frames,
        "max_anchor_gap_s": cfg.max_anchor_gap_s,
        "max_anchor_speed_mps": cfg.max_anchor_speed_mps,
        "max_fit_rms_m": cfg.max_fit_rms_m,
        "max_fit_max_residual_m": cfg.max_fit_max_residual_m,
        "max_reprojection_error_px": cfg.max_reprojection_error_px,
        "max_extrapolate_frames": cfg.max_extrapolate_frames,
        "base_uncertainty_m": cfg.base_uncertainty_m,
        "uncertainty_per_frame_m": cfg.uncertainty_per_frame_m,
        "max_xy_interpolate_gap_frames": cfg.max_xy_interpolate_gap_frames,
        "xy_interpolate_base_uncertainty_px": cfg.xy_interpolate_base_uncertainty_px,
        "xy_interpolate_per_frame_px": cfg.xy_interpolate_per_frame_px,
        "max_unreviewed_inflection_speed_px_s": cfg.max_unreviewed_inflection_speed_px_s,
        "inflection_wrist_tolerance_frames": cfg.inflection_wrist_tolerance_frames,
        "bounce_z_epsilon_m": cfg.bounce_z_epsilon_m,
        "floor_tolerance_m": cfg.floor_tolerance_m,
        "restitution_bounds": list(cfg.restitution_bounds),
    }


def _segment_summary(segment: PhysicsSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "frame_start": segment.frame_start,
        "frame_end": segment.frame_end,
        "t_start": segment.t_start,
        "t_end": segment.t_end,
        "sample_frame_indices": list(segment.sample_frame_indices),
        "sample_count": len(segment.samples),
        "model": segment.fit.model_name,
        "fit_rms_residual_m": segment.fit.rms_residual_m,
        "fit_max_residual_m": segment.fit.max_residual_m,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


__all__ = [
    "ARTIFACT_TYPE",
    "LANE",
    "BallisticFit",
    "PhysicsFillConfig",
    "PhysicsSegment",
    "SegmentFitResult",
    "WorldSample",
    "fill_ball_track_physics",
    "fit_ballistic_segments",
    "validate_physics_fill",
]
