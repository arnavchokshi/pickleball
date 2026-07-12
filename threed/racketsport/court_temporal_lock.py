"""Stateful per-frame court lock for standalone CAL diagnostics.

The lock starts only from an externally supplied trusted calibration.  It
propagates that court-to-image homography with ``camera_motion.json``'s
current-image-to-reference transform, then applies covariance-weighted normal
observations from the hybrid paint-centerline refiner.  This module deliberately
does not integrate with the pipeline or any placement/ball/body consumer.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


COURT_LINES_M: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_baseline": ((-3.048, -6.7056), (3.048, -6.7056)),
    "far_baseline": ((-3.048, 6.7056), (3.048, 6.7056)),
    "left_sideline": ((-3.048, -6.7056), (-3.048, 6.7056)),
    "right_sideline": ((3.048, -6.7056), (3.048, 6.7056)),
    "near_nvz": ((-3.048, -2.1336), (3.048, -2.1336)),
    "far_nvz": ((-3.048, 2.1336), (3.048, 2.1336)),
    "near_centerline": ((0.0, -6.7056), (0.0, -2.1336)),
    "far_centerline": ((0.0, 2.1336), (0.0, 6.7056)),
}

PARAMETER_ORDER = (
    "h00",
    "h01",
    "h02",
    "h10",
    "h11",
    "h12",
    "h20",
    "h21",
)
PROVENANCE_KINDS = frozenset({"measured", "predicted", "missing", "reset"})
LOCK_STATES = frozenset({"locked", "coasting", "reacquiring", "absent"})


@dataclass(frozen=True)
class TemporalCourtLockConfig:
    fps: float = 30.0
    keyframe_interval_seconds: float = 0.75
    max_coast_frames: int = 30
    min_measurements: int = 6
    samples_per_line: int = 7
    innovation_gate_sigma: float = 4.0
    innovation_gate_px: float = 8.0
    innovation_spike_px: float = 3.0
    measurement_deadband_px: float = 0.25
    max_update_corner_px: float = 0.15
    max_unsupported_motion_px: float = 8.0
    initial_covariance: float = 1.0e-8
    process_noise: float = 2.0e-10
    missing_process_multiplier: float = 1.35
    low_inlier_process_multiplier: float = 3.0
    blur_process_multiplier: float = 2.0

    @property
    def keyframe_interval_frames(self) -> int:
        return max(1, int(round(self.fps * self.keyframe_interval_seconds)))


@dataclass(frozen=True)
class CourtLineObservation:
    line_id: str
    court_xy: tuple[float, float]
    image_xy: tuple[float, float]
    normal: tuple[float, float]
    variance_px2: float
    provenance: str = "band_refined"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trusted_calibration(calibration: Mapping[str, Any]) -> np.ndarray:
    """Return a normalized seed H, rejecting proposal/self-bootstrap inputs."""

    raw_h = calibration.get("homography")
    if raw_h is None:
        raise ValueError("trusted court calibration must contain homography")
    h = _normalized_h(raw_h)
    source = str(calibration.get("source", "")).lower()
    trust_text = " ".join(
        [
            source,
            str(calibration.get("metric_confidence", "")).lower(),
            " ".join(str(v).lower() for v in calibration.get("capture_quality", {}).get("reasons", [])),
        ]
    )
    rejected_markers = ("proposal", "auto_preview", "unreviewed", "self_bootstrap")
    if any(marker in trust_text for marker in rejected_markers):
        raise ValueError(f"calibration source is not a trusted static/reviewed seed: {source or 'unknown'}")
    trusted_markers = ("reviewed", "manual", "profile", "metric_15pt", "metric15", "static", "fixture")
    if not any(marker in trust_text for marker in trusted_markers):
        raise ValueError(
            "calibration must declare reviewed/manual/profile/static provenance; "
            f"got source={source or 'missing'}"
        )
    return h


class TemporalCourtLock:
    """Predict/update court geometry while keeping uncertainty and provenance explicit."""

    def __init__(
        self,
        calibration: Mapping[str, Any],
        *,
        config: TemporalCourtLockConfig | None = None,
        reference_frame_idx: int = 0,
    ) -> None:
        self.calibration = dict(calibration)
        self.config = config or TemporalCourtLockConfig()
        if self.config.fps <= 0.0:
            raise ValueError("fps must be positive")
        self.reference_h = validate_trusted_calibration(calibration)
        self.reference_frame_idx = int(reference_frame_idx)
        image_size = calibration.get("image_size", [1920, 1080])
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._scales = _parameter_scales(self.reference_h, self.image_size)
        self._h: np.ndarray | None = self.reference_h.copy()
        self._p = np.eye(8, dtype=np.float64) * self.config.initial_covariance
        self._previous_motion = np.eye(3, dtype=np.float64)
        self._generation = 0
        self._missing_frames = 0
        self._last_keyframe_idx = self.reference_frame_idx
        self._force_keyframe_reason: str | None = "startup"
        self._lock_state = "locked"

    def step(
        self,
        frame_idx: int,
        *,
        frame_bgr: np.ndarray | None = None,
        motion: Mapping[str, Any] | None = None,
        observations: Sequence[CourtLineObservation | Mapping[str, Any]] | None = None,
        hard_cut: bool = False,
        hard_cut_reason: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        frame_idx = int(frame_idx)
        if hard_cut or _motion_declares_hard_cut(motion):
            reason = hard_cut_reason or _motion_reason(motion) or "hard_scene_cut"
            return self._reset_after_cut(frame_idx, reason=reason, started=started)

        keyframe_reason = self._scheduled_keyframe_reason(frame_idx)
        motion_matrix, motion_source, motion_quality = _motion_matrix_and_quality(motion)
        predicted_h, q_multiplier = self._predict(motion_matrix, motion_source, motion_quality)
        if predicted_h is None:
            return self._absent_frame(
                frame_idx,
                provenance_kind="missing",
                reason="reference_generation_has_no_trusted_seed",
                motion_source=motion_source,
                keyframe_reason=keyframe_reason,
                started=started,
            )

        if observations is None and frame_bgr is not None:
            extracted, evidence_meta = extract_hybrid_paint_observations(
                frame_bgr,
                predicted_h,
                self.calibration,
                samples_per_line=self.config.samples_per_line,
            )
        else:
            extracted = [_coerce_observation(row) for row in (observations or [])]
            evidence_meta = {
                "provider": "supplied_observations" if observations is not None else "missing_frame",
                "expected_count": len(extracted),
                "refined_count": len(extracted),
                "fallback_count": 0,
            }

        accepted, rejected, pre_residuals = self._gate_observations(predicted_h, extracted)
        evidence_meta["accepted_count"] = len(accepted)
        evidence_meta["rejected_count"] = len(rejected)
        evidence_meta["coverage"] = (
            len(accepted) / float(evidence_meta["expected_count"])
            if evidence_meta["expected_count"]
            else 0.0
        )

        previous_h = self._h.copy() if self._h is not None else None
        unsupported_motion_px = (
            _max_corner_displacement(previous_h, predicted_h) if previous_h is not None else 0.0
        )
        reject_unsupported_motion = (
            motion_source == "camera_motion"
            and unsupported_motion_px > self.config.max_unsupported_motion_px
            and len(accepted) < self.config.min_measurements
        )

        if len(accepted) >= self.config.min_measurements:
            updated_h, update_meta = self._update(predicted_h, accepted)
            self._h = updated_h
            self._missing_frames = 0
            previous_state = self._lock_state
            self._lock_state = "locked"
            recovery = previous_state in {"coasting", "reacquiring", "absent"}
            if recovery:
                keyframe_reason = "evidence_recovery"
            innovation_median = _median(pre_residuals)
            if innovation_median is not None and innovation_median > self.config.innovation_spike_px:
                self._force_keyframe_reason = "innovation_spike"
            elif keyframe_reason is not None:
                self._force_keyframe_reason = None
                self._last_keyframe_idx = frame_idx
            provenance_kind = "reset" if recovery and keyframe_reason == "evidence_recovery" else "measured"
            provenance_reason = keyframe_reason or "robust_covariance_update"
        else:
            if reject_unsupported_motion and previous_h is not None:
                predicted_h = previous_h
                motion_source = "camera_motion_rejected_no_evidence"
                self._force_keyframe_reason = "unsupported_motion_spike"
            self._h = predicted_h
            self._missing_frames += 1
            self._p *= self.config.missing_process_multiplier
            update_meta = {"applied": False, "deadbanded_count": 0}
            if extracted and rejected:
                self._lock_state = "reacquiring"
                self._force_keyframe_reason = "innovation_rejected"
                provenance_kind = "predicted"
                provenance_reason = (
                    "unsupported_motion_without_paint_evidence"
                    if reject_unsupported_motion
                    else "innovation_gate_rejected"
                )
            elif self._missing_frames <= self.config.max_coast_frames:
                self._lock_state = "coasting"
                self._force_keyframe_reason = self._force_keyframe_reason or "evidence_missing"
                provenance_kind = "predicted"
                provenance_reason = "insufficient_paint_evidence"
            else:
                self._lock_state = "absent"
                provenance_kind = "missing"
                provenance_reason = "coast_limit_exceeded"

        tracked_residuals = _observation_residuals(self._h, accepted) if self._h is not None else []
        static_residuals = _observation_residuals(self.reference_h, accepted)
        evidence_meta.update(
            {
                "innovation_abs_px": _stats(pre_residuals),
                "tracked_residual_abs_px": _stats(tracked_residuals),
                "static_baseline_residual_abs_px": _stats(static_residuals),
                "line_ids": sorted({observation.line_id for observation in accepted}),
                "update": update_meta,
                "predicted_motion_corner_displacement_px": unsupported_motion_px,
                "motion_prediction_rejected": reject_unsupported_motion,
            }
        )
        if motion_source in {"camera_motion", "camera_motion_static_degenerate"}:
            self._previous_motion = motion_matrix
        frame = self._frame_payload(
            frame_idx,
            provenance_kind=provenance_kind,
            reason=provenance_reason,
            motion_source=motion_source,
            keyframe_reason=keyframe_reason,
            evidence=evidence_meta,
        )
        frame["runtime_ms"] = (time.perf_counter() - started) * 1000.0
        _validate_frame_payload(frame)
        return frame

    def _predict(
        self,
        motion_matrix: np.ndarray,
        motion_source: str,
        motion_quality: Mapping[str, float],
    ) -> tuple[np.ndarray | None, float]:
        if self._h is None:
            return None, self.config.missing_process_multiplier
        if motion_source in {"camera_motion", "camera_motion_static_degenerate"}:
            # M maps current pixels -> reference pixels.  Therefore the
            # previous-image -> current-image propagation is inv(M_t) @ M_prev.
            propagation = np.linalg.inv(motion_matrix) @ self._previous_motion
            predicted = _normalized_h(propagation @ self._h)
            multiplier = 1.0
        else:
            predicted = self._h.copy()
            multiplier = self.config.missing_process_multiplier
        inlier_ratio = float(motion_quality.get("inlier_ratio", 1.0))
        if inlier_ratio < 0.35:
            multiplier *= self.config.low_inlier_process_multiplier
        if float(motion_quality.get("blur_score", 0.0)) > 1.0:
            multiplier *= self.config.blur_process_multiplier
        if not bool(motion_quality.get("compensated", True)):
            multiplier *= self.config.missing_process_multiplier
        self._p += np.eye(8, dtype=np.float64) * self.config.process_noise * multiplier
        return predicted, multiplier

    def _gate_observations(
        self,
        h: np.ndarray,
        observations: Sequence[CourtLineObservation],
    ) -> tuple[list[CourtLineObservation], list[CourtLineObservation], list[float]]:
        accepted: list[CourtLineObservation] = []
        rejected: list[CourtLineObservation] = []
        residuals: list[float] = []
        z = _encode_h(h, self._scales)
        for observation in observations:
            residual, jacobian = _normal_residual_and_jacobian(z, self._scales, observation)
            innovation_var = float(jacobian @ self._p @ jacobian.T + observation.variance_px2)
            normalized = abs(residual) / math.sqrt(max(innovation_var, 1.0e-9))
            if normalized <= self.config.innovation_gate_sigma and abs(residual) <= self.config.innovation_gate_px:
                accepted.append(observation)
                residuals.append(abs(float(residual)))
            else:
                rejected.append(observation)
        return accepted, rejected, residuals

    def _update(
        self,
        h: np.ndarray,
        observations: Sequence[CourtLineObservation],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        z = _encode_h(h, self._scales)
        rows: list[np.ndarray] = []
        residuals: list[float] = []
        variances: list[float] = []
        deadbanded = 0
        for observation in observations:
            residual, jacobian = _normal_residual_and_jacobian(z, self._scales, observation)
            if abs(residual) <= self.config.measurement_deadband_px:
                residual = 0.0
                deadbanded += 1
            huber_weight = min(1.0, 2.0 / max(abs(residual), 2.0))
            rows.append(jacobian)
            residuals.append(float(residual))
            variances.append(float(observation.variance_px2) / max(huber_weight, 1.0e-3))
        j = np.vstack(rows)
        r = np.asarray(residuals, dtype=np.float64)
        measurement_covariance = np.diag(variances)
        innovation_covariance = j @ self._p @ j.T + measurement_covariance
        gain = self._p @ j.T @ np.linalg.pinv(innovation_covariance)
        delta = gain @ r
        # A normal measurement is always a partial update, never a state snap.
        delta *= 0.75
        raw_candidate = _decode_h(z + delta, self._scales)
        max_corner_update = _max_corner_displacement(h, raw_candidate)
        corner_limit_scale = min(
            1.0,
            self.config.max_update_corner_px / max(max_corner_update, 1.0e-12),
        )
        delta *= corner_limit_scale
        updated_z = z + delta
        identity = np.eye(8, dtype=np.float64)
        effective_gain = gain * (0.75 * corner_limit_scale)
        posterior = (identity - effective_gain @ j) @ self._p @ (identity - effective_gain @ j).T
        posterior += effective_gain @ measurement_covariance @ effective_gain.T
        self._p = 0.5 * (posterior + posterior.T)
        return _decode_h(updated_z, self._scales), {
            "applied": bool(np.any(np.abs(delta) > 0.0)),
            "observation_count": len(observations),
            "deadbanded_count": deadbanded,
            "normalized_parameter_delta_l2": float(np.linalg.norm(delta)),
            "gain_cap": 0.75,
            "raw_max_corner_update_px": max_corner_update,
            "applied_max_corner_update_px": max_corner_update * corner_limit_scale,
            "corner_limit_scale": corner_limit_scale,
        }

    def _scheduled_keyframe_reason(self, frame_idx: int) -> str | None:
        if self._force_keyframe_reason is not None:
            return self._force_keyframe_reason
        if frame_idx - self._last_keyframe_idx >= self.config.keyframe_interval_frames:
            return "fixed_cadence"
        return None

    def _reset_after_cut(self, frame_idx: int, *, reason: str, started: float) -> dict[str, Any]:
        self._generation += 1
        self._h = None
        self._p *= max(10.0, self.config.missing_process_multiplier)
        self._previous_motion = np.eye(3, dtype=np.float64)
        self._missing_frames = self.config.max_coast_frames + 1
        self._lock_state = "absent"
        self._last_keyframe_idx = frame_idx
        self._force_keyframe_reason = "new_generation_requires_trusted_seed"
        frame = self._frame_payload(
            frame_idx,
            provenance_kind="reset",
            reason=reason,
            motion_source="scene_cut",
            keyframe_reason="hard_scene_cut",
            evidence=_empty_evidence("not_run_after_scene_cut"),
        )
        frame["runtime_ms"] = (time.perf_counter() - started) * 1000.0
        _validate_frame_payload(frame)
        return frame

    def _absent_frame(
        self,
        frame_idx: int,
        *,
        provenance_kind: str,
        reason: str,
        motion_source: str,
        keyframe_reason: str | None,
        started: float,
    ) -> dict[str, Any]:
        self._p *= self.config.missing_process_multiplier
        self._lock_state = "absent"
        frame = self._frame_payload(
            frame_idx,
            provenance_kind=provenance_kind,
            reason=reason,
            motion_source=motion_source,
            keyframe_reason=keyframe_reason,
            evidence=_empty_evidence("reference_absent"),
        )
        frame["runtime_ms"] = (time.perf_counter() - started) * 1000.0
        _validate_frame_payload(frame)
        return frame

    def _frame_payload(
        self,
        frame_idx: int,
        *,
        provenance_kind: str,
        reason: str,
        motion_source: str,
        keyframe_reason: str | None,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "frame_idx": frame_idx,
            "H_court_to_image": self._h.tolist() if self._h is not None else None,
            "covariance": {
                "parameterization": "normalized_homography_8d",
                "parameter_order": list(PARAMETER_ORDER),
                "normalization_scales": self._scales.tolist(),
                "matrix": self._p.tolist(),
                "trace": float(np.trace(self._p)),
                "corner_sigma_px": _corner_sigma_px(self._h, self._p, self._scales),
            },
            "evidence": dict(evidence),
            "lock_state": self._lock_state,
            "motion_mode": motion_source,
            "semantic_line_identities": list(COURT_LINES_M),
            "identity_hypotheses": [
                {"id": "reviewed_seed", "rank": 1, "status": "selected", "source": "trusted_calibration"}
            ],
            "provenance": {
                "kind": provenance_kind,
                "reason": reason,
                "measured_or_predicted": provenance_kind,
                "motion_source": motion_source,
                "measurement_source": evidence.get("provider", "none"),
                "is_keyframe": keyframe_reason is not None,
                "keyframe_reason": keyframe_reason,
                "reference_keyframe_idx": self._last_keyframe_idx,
                "reference_generation": self._generation,
            },
        }


def extract_hybrid_paint_observations(
    image_bgr: np.ndarray,
    h_court_to_image: Sequence[Sequence[float]] | np.ndarray,
    seed_calibration: Mapping[str, Any],
    *,
    samples_per_line: int = 7,
) -> tuple[list[CourtLineObservation], dict[str, Any]]:
    """Sample predicted semantic lines through the read-only hybrid paint API."""

    from .court_line_bank import LegacyPaintEvidenceSample, refine_legacy_paint_samples

    h = _normalized_h(h_court_to_image)
    height, width = image_bgr.shape[:2]
    raw_samples: list[Any] = []
    sample_metadata: dict[str, tuple[str, tuple[float, float]]] = {}
    for line_id, (court_a, court_b) in COURT_LINES_M.items():
        image_a = _project(h, court_a)
        image_b = _project(h, court_b)
        direction = np.asarray(image_b) - np.asarray(image_a)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            continue
        normal = (-float(direction[1] / norm), float(direction[0] / norm))
        for sample_idx in range(samples_per_line):
            t = (sample_idx + 0.5) / float(samples_per_line)
            court_xy = (
                court_a[0] + (court_b[0] - court_a[0]) * t,
                court_a[1] + (court_b[1] - court_a[1]) * t,
            )
            image_xy = _project(h, court_xy)
            if not (2.0 <= image_xy[0] < width - 2.0 and 2.0 <= image_xy[1] < height - 2.0):
                continue
            source_id = f"{line_id}:{sample_idx}"
            raw_samples.append(LegacyPaintEvidenceSample(xy=image_xy, normal=normal, source_id=source_id))
            sample_metadata[source_id] = (line_id, court_xy)
    current_seed = dict(seed_calibration)
    current_seed["homography"] = h.tolist()
    refined = refine_legacy_paint_samples(image_bgr, raw_samples, seed_calibration=current_seed)
    observations: list[CourtLineObservation] = []
    fallback_count = 0
    for sample in refined:
        if sample.provenance != "band_refined" or sample.source_id not in sample_metadata:
            fallback_count += 1
            continue
        line_id, court_xy = sample_metadata[sample.source_id]
        observations.append(
            CourtLineObservation(
                line_id=line_id,
                court_xy=court_xy,
                image_xy=sample.xy,
                normal=sample.normal,
                variance_px2=max(float(sample.normal_variance_px2), 0.04),
                provenance=sample.provenance,
            )
        )
    return observations, {
        "provider": "hybrid_paint_refinement",
        "expected_count": len(raw_samples),
        "refined_count": len(observations),
        "fallback_count": fallback_count,
    }


def build_artifact(
    *,
    frames: Sequence[Mapping[str, Any]],
    video_path: str | Path,
    calibration_path: str | Path,
    camera_motion_path: str | Path | None,
    camera_motion_mode: str,
    config: TemporalCourtLockConfig,
    reference_frame_idx: int,
) -> dict[str, Any]:
    tracked = _collect_stat(frames, "tracked_residual_abs_px")
    static = _collect_stat(frames, "static_baseline_residual_abs_px")
    runtime = [float(frame.get("runtime_ms", 0.0)) for frame in frames]
    artifact = {
        "artifact_type": "racketsport_court_temporal_lock",
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_candidate_not_integrated",
        "coordinate_contract": {
            "homography_direction": "court_world_xy_to_current_image_pixels",
            "camera_motion_direction": "current_image_pixels_to_reference_image_pixels",
            "composition": "H_t = inv(M_t) @ H_reference",
            "distortion_space": "same_raw_image_pixel_space_as_seed_calibration_and_video_decode",
        },
        "reference": {
            "calibration_path": str(Path(calibration_path)),
            "calibration_sha256": file_sha256(calibration_path),
            "reference_frame_idx": int(reference_frame_idx),
            "initialization": "existing_trusted_reviewed_or_static_calibration_only",
        },
        "source": {
            "video_path": str(Path(video_path)),
            "video_sha256": file_sha256(video_path),
            "camera_motion_path": str(Path(camera_motion_path)) if camera_motion_path is not None else None,
            "camera_motion_sha256": file_sha256(camera_motion_path) if camera_motion_path is not None else None,
            "camera_motion_fallback": "identity_with_covariance_inflation" if camera_motion_path is None else None,
            "camera_motion_mode": camera_motion_mode,
        },
        "config": {
            key: value
            for key, value in config.__dict__.items()
        },
        "summary": {
            "frame_count": len(frames),
            "explicit_provenance_frame_count": sum(
                1 for frame in frames if frame.get("provenance", {}).get("kind") in PROVENANCE_KINDS
            ),
            "lock_state_counts": {
                state: sum(1 for frame in frames if frame.get("lock_state") == state)
                for state in sorted(LOCK_STATES)
            },
            "tracked_residual_abs_px": _stats(tracked),
            "static_baseline_residual_abs_px": _stats(static),
            "runtime_ms_per_frame": _stats(runtime),
            "best_stack_delta": "none",
            "verified": False,
        },
        "frames": [dict(frame) for frame in frames],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_type") != "racketsport_court_temporal_lock":
        raise ValueError("unexpected temporal-lock artifact_type")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("temporal-lock artifact requires at least one frame")
    expected = list(range(len(frames)))
    actual = [int(frame.get("frame_idx", -1)) for frame in frames]
    if actual != expected:
        raise ValueError(f"temporal-lock frame timeline must be contiguous from zero: {actual[:5]}")
    for frame in frames:
        _validate_frame_payload(frame)


def _validate_frame_payload(frame: Mapping[str, Any]) -> None:
    if frame.get("lock_state") not in LOCK_STATES:
        raise ValueError(f"invalid lock state: {frame.get('lock_state')}")
    provenance = frame.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("kind") not in PROVENANCE_KINDS:
        raise ValueError("every frame requires explicit temporal-lock provenance")
    if "reference_generation" not in provenance or not provenance.get("reason"):
        raise ValueError("every frame provenance requires generation and reason")
    h = frame.get("H_court_to_image")
    if h is not None:
        _normalized_h(h)


def _coerce_observation(row: CourtLineObservation | Mapping[str, Any]) -> CourtLineObservation:
    if isinstance(row, CourtLineObservation):
        return row
    return CourtLineObservation(
        line_id=str(row["line_id"]),
        court_xy=(float(row["court_xy"][0]), float(row["court_xy"][1])),
        image_xy=(float(row["image_xy"][0]), float(row["image_xy"][1])),
        normal=(float(row["normal"][0]), float(row["normal"][1])),
        variance_px2=max(float(row.get("variance_px2", 1.0)), 1.0e-6),
        provenance=str(row.get("provenance", "supplied")),
    )


def _motion_matrix_and_quality(
    motion: Mapping[str, Any] | None,
) -> tuple[np.ndarray, str, dict[str, float]]:
    if motion is None or motion.get("M") is None or not bool(motion.get("compensated", True)):
        quality = {
            "inlier_ratio": float(motion.get("inlier_ratio", 0.0)) if motion else 0.0,
            "blur_score": float(motion.get("blur_score", 0.0)) if motion else 0.0,
            "compensated": False,
        }
        return np.eye(3, dtype=np.float64), "identity_missing_motion", quality
    if bool(motion.get("static_degenerate")):
        matrix = np.eye(3, dtype=np.float64)
        source = "camera_motion_static_degenerate"
    else:
        matrix = _normalized_h(motion["M"])
        source = "camera_motion"
    quality = {
        "inlier_ratio": float(motion.get("inlier_ratio", 1.0)),
        "blur_score": float(motion.get("blur_score", 0.0)),
        "compensated": True,
    }
    return matrix, source, quality


def _motion_reason(motion: Mapping[str, Any] | None) -> str | None:
    if motion is None:
        return None
    reason = motion.get("reason") or motion.get("reset_reason")
    return str(reason) if reason else None


def _motion_declares_hard_cut(motion: Mapping[str, Any] | None) -> bool:
    if motion is None:
        return False
    if bool(motion.get("hard_cut")) or bool(motion.get("scene_cut")):
        return True
    reason = (_motion_reason(motion) or "").lower()
    return "scene_cut" in reason or "hard_cut" in reason


def _normal_residual_and_jacobian(
    z: np.ndarray,
    scales: np.ndarray,
    observation: CourtLineObservation,
) -> tuple[float, np.ndarray]:
    h = _decode_h(z, scales)
    predicted = np.asarray(_project(h, observation.court_xy), dtype=np.float64)
    observed = np.asarray(observation.image_xy, dtype=np.float64)
    normal = np.asarray(observation.normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    residual = float(normal @ (observed - predicted))
    jacobian = np.zeros(8, dtype=np.float64)
    epsilon = 1.0e-6
    for idx in range(8):
        perturbed = z.copy()
        perturbed[idx] += epsilon
        shifted = np.asarray(_project(_decode_h(perturbed, scales), observation.court_xy), dtype=np.float64)
        # residual = observed - prediction, so the measurement Jacobian used
        # with z_new = z + K*r is +d(prediction)/dz.
        jacobian[idx] = float(normal @ (shifted - predicted)) / epsilon
    return residual, jacobian


def _observation_residuals(
    h: np.ndarray | None,
    observations: Sequence[CourtLineObservation],
) -> list[float]:
    if h is None:
        return []
    residuals: list[float] = []
    for observation in observations:
        predicted = np.asarray(_project(h, observation.court_xy), dtype=np.float64)
        observed = np.asarray(observation.image_xy, dtype=np.float64)
        normal = np.asarray(observation.normal, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        residuals.append(abs(float(normal @ (observed - predicted))))
    return residuals


def _normalized_h(raw: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    h = np.asarray(raw, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(h))) < 1.0e-12 or abs(float(h[2, 2])) < 1.0e-12:
        raise ValueError("homography must be nonsingular and normalizable")
    return h / h[2, 2]


def _parameter_scales(h: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.asarray(
        [
            max(abs(float(h[0, 0])), 1.0),
            max(abs(float(h[0, 1])), 1.0),
            max(float(width), 1.0),
            max(abs(float(h[1, 0])), 1.0),
            max(abs(float(h[1, 1])), 1.0),
            max(float(height), 1.0),
            1.0 / max(float(width), 1.0),
            1.0 / max(float(height), 1.0),
        ],
        dtype=np.float64,
    )


def _encode_h(h: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(
        [h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]],
        dtype=np.float64,
    )
    return values / scales


def _decode_h(z: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64) * scales
    return _normalized_h(
        [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], 1.0],
        ]
    )


def _project(h: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    homogeneous = h @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if abs(float(homogeneous[2])) < 1.0e-12:
        raise ValueError("court point projects to infinity")
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


def _corner_sigma_px(h: np.ndarray | None, p: np.ndarray, scales: np.ndarray) -> float | None:
    if h is None:
        return None
    z = _encode_h(h, scales)
    sigmas: list[float] = []
    for point in ((-3.048, -6.7056), (3.048, -6.7056), (3.048, 6.7056), (-3.048, 6.7056)):
        base = np.asarray(_project(h, point), dtype=np.float64)
        jacobian = np.zeros((2, 8), dtype=np.float64)
        epsilon = 1.0e-6
        for idx in range(8):
            perturbed = z.copy()
            perturbed[idx] += epsilon
            shifted = np.asarray(_project(_decode_h(perturbed, scales), point), dtype=np.float64)
            jacobian[:, idx] = (shifted - base) / epsilon
        covariance = jacobian @ p @ jacobian.T
        sigmas.append(math.sqrt(max(float(np.trace(covariance)), 0.0)))
    return float(max(sigmas))


def _max_corner_displacement(before: np.ndarray, after: np.ndarray) -> float:
    corners = ((-3.048, -6.7056), (3.048, -6.7056), (3.048, 6.7056), (-3.048, 6.7056))
    return max(
        float(np.linalg.norm(np.asarray(_project(after, point)) - np.asarray(_project(before, point))))
        for point in corners
    )


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _empty_evidence(provider: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "expected_count": 0,
        "refined_count": 0,
        "fallback_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "coverage": 0.0,
        "innovation_abs_px": _stats([]),
        "tracked_residual_abs_px": _stats([]),
        "static_baseline_residual_abs_px": _stats([]),
        "line_ids": [],
        "update": {"applied": False, "deadbanded_count": 0},
    }


def _collect_stat(frames: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for frame in frames:
        stats = frame.get("evidence", {}).get(key, {})
        median = stats.get("median") if isinstance(stats, Mapping) else None
        if median is not None:
            values.append(float(median))
    return values


__all__ = [
    "COURT_LINES_M",
    "CourtLineObservation",
    "TemporalCourtLock",
    "TemporalCourtLockConfig",
    "build_artifact",
    "extract_hybrid_paint_observations",
    "file_sha256",
    "load_json",
    "validate_artifact",
    "validate_trusted_calibration",
]
