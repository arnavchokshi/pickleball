"""Deterministic static-camera evidence pooling and ``court_lock.json`` contract.

This is an additive foundation for the confidence-aware structured court
pipeline.  It does not read video, run a detector, or select authority.  Callers
provide immutable per-frame evidence; this module deterministically selects at
most eight frames, robustly pools observations that should share image pixels,
diagnoses violations of the fixed-camera assumption, and serializes one
versioned lock artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from threed.racketsport.court_camera_geometry import (
    DEFAULT_K1_BOUNDS,
    PinholeIntrinsics,
    project_planar_point_with_covariance,
    validate_bounded_k1,
)


COURT_LOCK_SCHEMA_VERSION = 1
COURT_LOCK_ARTIFACT_TYPE = "racketsport_court_lock"
COURT_LOCK_SELECTION_ALGORITHM = "quality_ranked_static_diversity_v1"
COURT_LOCK_POOLING_ALGORITHM = "weighted_median_mad_static_pixels_v1"

CourtLockSource = Literal[
    "owner_metric_15pt_reviewed",
    "multi_frame_point_and_line",
    "clearest_frame_point_and_line",
    "dense_line_only",
    "previous_static_lock",
    "camera_profile_prior",
]
MotionStatus = Literal["static", "moving", "ambiguous", "insufficient_evidence"]
AuthorityState = Literal["best_effort", "review_only", "authoritative"]

_LOCK_SOURCES = frozenset(
    {
        "owner_metric_15pt_reviewed",
        "multi_frame_point_and_line",
        "clearest_frame_point_and_line",
        "dense_line_only",
        "previous_static_lock",
        "camera_profile_prior",
    }
)
_MOTION_STATUSES = frozenset({"static", "moving", "ambiguous", "insufficient_evidence"})
_AUTHORITY_STATES = frozenset({"best_effort", "review_only", "authoritative"})
_OUTER_COURT_POINTS_M: tuple[tuple[float, float], ...] = (
    (-3.048, -6.7056),
    (3.048, -6.7056),
    (3.048, 6.7056),
    (-3.048, 6.7056),
)


@dataclass(frozen=True)
class StaticCourtObservation:
    semantic_name: str
    xy: tuple[float, float]
    confidence: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    kind: Literal["point", "line_intersection"] = "point"
    line_support: float = 0.0

    def __post_init__(self) -> None:
        if not self.semantic_name.strip():
            raise ValueError("semantic_name must be nonblank")
        _finite_vector(self.xy, size=2, name="xy")
        _unit_interval(self.confidence, name="confidence")
        _unit_interval(self.line_support, name="line_support")
        _covariance(self.covariance_px2, size=2, name="covariance_px2")
        if self.kind not in {"point", "line_intersection"}:
            raise ValueError("kind must be point or line_intersection")


@dataclass(frozen=True)
class StaticFrameEvidence:
    frame_index: int
    line_support: float
    surface_support: float
    visible_fraction: float
    sharpness: float
    occlusion_fraction: float
    observations: tuple[StaticCourtObservation, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or int(self.frame_index) < 0:
            raise ValueError("frame_index must be a non-negative integer")
        for name in (
            "line_support",
            "surface_support",
            "visible_fraction",
            "sharpness",
            "occlusion_fraction",
        ):
            _unit_interval(getattr(self, name), name=name)
        identities = [(observation.kind, observation.semantic_name) for observation in self.observations]
        if len(identities) != len(set(identities)):
            raise ValueError("a frame may contain at most one observation per kind and semantic_name")

    @property
    def quality_score(self) -> float:
        return float(
            0.35 * self.line_support
            + 0.20 * self.surface_support
            + 0.20 * self.visible_fraction
            + 0.15 * self.sharpness
            + 0.10 * (1.0 - self.occlusion_fraction)
        )


@dataclass(frozen=True)
class PooledStaticObservation:
    semantic_name: str
    kind: Literal["point", "line_intersection"]
    xy: tuple[float, float]
    confidence: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    line_support: float
    temporal_support: float
    contributing_frame_indices: tuple[int, ...]
    rejected_frame_indices: tuple[int, ...]

    def to_solver_observation(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic_name,
            "candidate_id": f"static_pool:{self.kind}:{self.semantic_name}",
            "xy": [float(value) for value in self.xy],
            "confidence": float(self.confidence),
            "covariance": [[float(value) for value in row] for row in self.covariance_px2],
            "line_support": float(self.line_support),
            "temporal_support": float(self.temporal_support),
            "source": "static_evidence_pool",
            "contributing_frame_indices": list(self.contributing_frame_indices),
        }


@dataclass(frozen=True)
class PooledStaticEvidence:
    candidate_frame_count: int
    selected_frame_indices: tuple[int, ...]
    observations: tuple[PooledStaticObservation, ...]
    selection_algorithm: str = COURT_LOCK_SELECTION_ALGORITHM
    pooling_algorithm: str = COURT_LOCK_POOLING_ALGORITHM

    def __post_init__(self) -> None:
        _validate_frame_indices(self.selected_frame_indices)
        if self.candidate_frame_count < len(self.selected_frame_indices):
            raise ValueError("candidate_frame_count cannot be smaller than selected frame count")

    def to_solver_observations(self) -> list[dict[str, Any]]:
        return [observation.to_solver_observation() for observation in self.observations]


@dataclass(frozen=True)
class FrameCourtTransform:
    frame_index: int
    homography_image_from_court: tuple[tuple[float, float, float], ...]
    confidence: float = 1.0
    transform_covariance: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or int(self.frame_index) < 0:
            raise ValueError("frame_index must be a non-negative integer")
        _unit_interval(self.confidence, name="confidence")
        _homography(self.homography_image_from_court)
        if self.transform_covariance is not None:
            _covariance(self.transform_covariance, size=8, name="transform_covariance")


@dataclass(frozen=True)
class StaticMotionConfig:
    min_frames: int = 3
    median_drift_threshold_px: float = 2.0
    p95_drift_threshold_px: float = 4.0
    net_displacement_threshold_px: float = 3.0
    monotonic_trend_threshold_px_per_frame: float = 0.05
    uncertainty_sigma_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.min_frames, bool) or int(self.min_frames) < 2:
            raise ValueError("min_frames must be at least two")
        for name in (
            "median_drift_threshold_px",
            "p95_drift_threshold_px",
            "net_displacement_threshold_px",
            "monotonic_trend_threshold_px_per_frame",
            "uncertainty_sigma_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")


@dataclass(frozen=True)
class StaticMotionDiagnostics:
    status: MotionStatus
    camera_motion_suspected: bool
    static_lock_usable: bool
    reference_frame_index: int | None
    frame_indices: tuple[int, ...]
    valid_frame_count: int
    invalid_frame_count: int
    median_corner_drift_px: float | None
    p95_corner_drift_px: float | None
    max_corner_drift_px: float | None
    net_center_displacement_px: float | None
    monotonic_trend_px_per_frame: float | None
    drift_uncertainty_px: float | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in _MOTION_STATUSES:
            raise ValueError(f"unsupported motion status: {self.status}")
        _validate_frame_indices(self.frame_indices, max_count=None)
        if self.valid_frame_count != len(self.frame_indices):
            raise ValueError("valid_frame_count must equal frame_indices length")
        if self.invalid_frame_count < 0:
            raise ValueError("invalid_frame_count must be non-negative")
        if self.static_lock_usable != (self.status == "static"):
            raise ValueError("static_lock_usable is true only for status=static")
        if self.camera_motion_suspected != (self.status in {"moving", "ambiguous"}):
            raise ValueError("camera_motion_suspected must match moving/ambiguous status")
        for name in (
            "median_corner_drift_px",
            "p95_corner_drift_px",
            "max_corner_drift_px",
            "net_center_displacement_px",
            "monotonic_trend_px_per_frame",
            "drift_uncertainty_px",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be non-negative and finite when present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "camera_motion_suspected": self.camera_motion_suspected,
            "static_lock_usable": self.static_lock_usable,
            "reference_frame_index": self.reference_frame_index,
            "frame_indices": list(self.frame_indices),
            "valid_frame_count": self.valid_frame_count,
            "invalid_frame_count": self.invalid_frame_count,
            "median_corner_drift_px": self.median_corner_drift_px,
            "p95_corner_drift_px": self.p95_corner_drift_px,
            "max_corner_drift_px": self.max_corner_drift_px,
            "net_center_displacement_px": self.net_center_displacement_px,
            "monotonic_trend_px_per_frame": self.monotonic_trend_px_per_frame,
            "drift_uncertainty_px": self.drift_uncertainty_px,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StaticMotionDiagnostics":
        return cls(
            status=str(payload["status"]),  # type: ignore[arg-type]
            camera_motion_suspected=bool(payload["camera_motion_suspected"]),
            static_lock_usable=bool(payload["static_lock_usable"]),
            reference_frame_index=(
                None if payload.get("reference_frame_index") is None else int(payload["reference_frame_index"])
            ),
            frame_indices=tuple(int(value) for value in payload.get("frame_indices", [])),
            valid_frame_count=int(payload["valid_frame_count"]),
            invalid_frame_count=int(payload["invalid_frame_count"]),
            median_corner_drift_px=_optional_float(payload.get("median_corner_drift_px")),
            p95_corner_drift_px=_optional_float(payload.get("p95_corner_drift_px")),
            max_corner_drift_px=_optional_float(payload.get("max_corner_drift_px")),
            net_center_displacement_px=_optional_float(payload.get("net_center_displacement_px")),
            monotonic_trend_px_per_frame=_optional_float(payload.get("monotonic_trend_px_per_frame")),
            drift_uncertainty_px=_optional_float(payload.get("drift_uncertainty_px")),
            reasons=tuple(str(value) for value in payload.get("reasons", [])),
        )


@dataclass(frozen=True)
class CourtLockCameraParameters:
    intrinsics: PinholeIntrinsics
    source: str
    rotation_world_to_camera: tuple[tuple[float, float, float], ...] | None = None
    translation_world_to_camera_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("camera parameter source must be nonblank")
        if (self.rotation_world_to_camera is None) != (self.translation_world_to_camera_m is None):
            raise ValueError("rotation and translation must either both be present or both be absent")
        if self.rotation_world_to_camera is not None:
            rotation = np.asarray(self.rotation_world_to_camera, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ValueError("rotation_world_to_camera must be a finite 3x3 matrix")
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-5, rtol=1.0e-5):
                raise ValueError("rotation_world_to_camera must be orthonormal")
            if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-5):
                raise ValueError("rotation_world_to_camera determinant must be one")
            assert self.translation_world_to_camera_m is not None
            _finite_vector(self.translation_world_to_camera_m, size=3, name="translation_world_to_camera_m")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "pinhole",
            "fx": float(self.intrinsics.fx),
            "fy": float(self.intrinsics.fy),
            "cx": float(self.intrinsics.cx),
            "cy": float(self.intrinsics.cy),
            "source": self.source,
            "rotation_world_to_camera": _nested_lists(self.rotation_world_to_camera),
            "translation_world_to_camera_m": (
                None
                if self.translation_world_to_camera_m is None
                else [float(value) for value in self.translation_world_to_camera_m]
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CourtLockCameraParameters":
        if payload.get("model") != "pinhole":
            raise ValueError("court lock camera model must be pinhole")
        rotation = payload.get("rotation_world_to_camera")
        translation = payload.get("translation_world_to_camera_m")
        return cls(
            intrinsics=PinholeIntrinsics(
                fx=float(payload["fx"]),
                fy=float(payload["fy"]),
                cx=float(payload["cx"]),
                cy=float(payload["cy"]),
            ),
            source=str(payload["source"]),
            rotation_world_to_camera=(None if rotation is None else _matrix_tuple(rotation, rows=3, cols=3)),
            translation_world_to_camera_m=(
                None if translation is None else tuple(float(value) for value in translation)  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True)
class CourtLockDistortion:
    k1: float
    k1_bounds: tuple[float, float] = DEFAULT_K1_BOUNDS
    k1_variance: float = 0.0
    source: str = "not_available_zero"
    optimized: bool = False

    def __post_init__(self) -> None:
        validate_bounded_k1(self.k1, bounds=self.k1_bounds)
        if not math.isfinite(float(self.k1_variance)) or float(self.k1_variance) < 0.0:
            raise ValueError("k1_variance must be non-negative and finite")
        if not self.source.strip():
            raise ValueError("distortion source must be nonblank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "bounded_radial_k1",
            "k1": float(self.k1),
            "k1_bounds": [float(value) for value in self.k1_bounds],
            "k1_variance": float(self.k1_variance),
            "source": self.source,
            "optimized": self.optimized,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CourtLockDistortion":
        if payload.get("model") != "bounded_radial_k1":
            raise ValueError("court lock distortion model must be bounded_radial_k1")
        bounds = tuple(float(value) for value in payload["k1_bounds"])
        if len(bounds) != 2:
            raise ValueError("k1_bounds must contain two values")
        return cls(
            k1=float(payload["k1"]),
            k1_bounds=(bounds[0], bounds[1]),
            k1_variance=float(payload["k1_variance"]),
            source=str(payload["source"]),
            optimized=bool(payload["optimized"]),
        )


@dataclass(frozen=True)
class CourtLockResidual:
    median_px: float
    p95_px: float

    def __post_init__(self) -> None:
        for name in ("median_px", "p95_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.p95_px < self.median_px:
            raise ValueError("p95_px cannot be smaller than median_px")

    def to_dict(self) -> dict[str, float]:
        return {"median_px": float(self.median_px), "p95_px": float(self.p95_px)}


@dataclass(frozen=True)
class CourtLockEvidenceSummary:
    candidate_frame_count: int
    selected_frame_indices: tuple[int, ...]
    pooled_semantic_count: int
    selection_algorithm: str = COURT_LOCK_SELECTION_ALGORITHM
    pooling_algorithm: str = COURT_LOCK_POOLING_ALGORITHM

    def __post_init__(self) -> None:
        _validate_frame_indices(self.selected_frame_indices)
        if self.candidate_frame_count < len(self.selected_frame_indices):
            raise ValueError("candidate_frame_count cannot be smaller than selected frame count")
        if self.pooled_semantic_count < 0:
            raise ValueError("pooled_semantic_count must be non-negative")
        if not self.selection_algorithm.strip() or not self.pooling_algorithm.strip():
            raise ValueError("evidence algorithms must be nonblank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_frame_count": self.candidate_frame_count,
            "selected_frame_indices": list(self.selected_frame_indices),
            "pooled_semantic_count": self.pooled_semantic_count,
            "selection_algorithm": self.selection_algorithm,
            "pooling_algorithm": self.pooling_algorithm,
        }


@dataclass(frozen=True)
class CourtLockArtifact:
    coordinate_space: Literal["pixels_raw_native", "pixels_undistorted_native"]
    homography_image_from_court: tuple[tuple[float, float, float], ...]
    camera_parameters: CourtLockCameraParameters
    distortion: CourtLockDistortion
    transform_covariance: tuple[tuple[float, ...], ...] | None
    source: CourtLockSource
    evidence: CourtLockEvidenceSummary
    static_motion: StaticMotionDiagnostics
    residual_px: CourtLockResidual
    score_components: Mapping[str, float] = field(default_factory=dict)
    scorer_version: str = "court_static_lock_score_v1"
    calibration_version: str = "court_lock_calibration_v1"
    checkpoint_sha256: str | None = None
    measurement_valid: bool = False
    authority_state: AuthorityState = "review_only"
    verified: bool = False
    schema_version: int = COURT_LOCK_SCHEMA_VERSION
    artifact_type: str = COURT_LOCK_ARTIFACT_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != COURT_LOCK_SCHEMA_VERSION:
            raise ValueError(f"court lock schema_version must be {COURT_LOCK_SCHEMA_VERSION}")
        if self.artifact_type != COURT_LOCK_ARTIFACT_TYPE:
            raise ValueError(f"artifact_type must be {COURT_LOCK_ARTIFACT_TYPE}")
        if self.coordinate_space not in {"pixels_raw_native", "pixels_undistorted_native"}:
            raise ValueError("unsupported coordinate_space")
        _homography(self.homography_image_from_court)
        if self.transform_covariance is not None:
            _covariance(self.transform_covariance, size=8, name="transform_covariance")
        if self.source not in _LOCK_SOURCES:
            raise ValueError(f"unsupported court lock source: {self.source}")
        if self.authority_state not in _AUTHORITY_STATES:
            raise ValueError(f"unsupported authority_state: {self.authority_state}")
        if not self.scorer_version.strip() or not self.calibration_version.strip():
            raise ValueError("scorer_version and calibration_version must be nonblank")
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("checkpoint_sha256 must be 64 lowercase hexadecimal characters")
        for key, value in self.score_components.items():
            if not str(key).strip() or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("score_components must map nonblank names to finite numbers")
        if self.measurement_valid and not self.static_motion.static_lock_usable:
            raise ValueError("measurement_valid requires a confirmed static camera")
        if self.verified and self.authority_state != "authoritative":
            raise ValueError("verified court locks must be authoritative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "coordinate_space": self.coordinate_space,
            "homography_image_from_court": _nested_lists(self.homography_image_from_court),
            "camera_parameters": self.camera_parameters.to_dict(),
            "distortion": self.distortion.to_dict(),
            "transform_covariance": _nested_lists(self.transform_covariance),
            "source": self.source,
            "evidence": self.evidence.to_dict(),
            "static_motion": self.static_motion.to_dict(),
            "residual_px": self.residual_px.to_dict(),
            "score_components": {
                str(key): float(value) for key, value in sorted(self.score_components.items())
            },
            "scorer_version": self.scorer_version,
            "calibration_version": self.calibration_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "measurement_valid": self.measurement_valid,
            "authority_state": self.authority_state,
            "verified": self.verified,
        }

    def to_json(self) -> str:
        """Return canonical, byte-stable JSON with one trailing newline."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CourtLockArtifact":
        unknown = sorted(set(payload) - _COURT_LOCK_KEYS)
        missing = sorted(_COURT_LOCK_REQUIRED_KEYS - set(payload))
        if unknown:
            raise ValueError(f"unknown court lock fields: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing court lock fields: {', '.join(missing)}")
        evidence_payload = payload["evidence"]
        residual_payload = payload["residual_px"]
        if not isinstance(evidence_payload, Mapping) or not isinstance(residual_payload, Mapping):
            raise ValueError("evidence and residual_px must be objects")
        transform_covariance = payload.get("transform_covariance")
        return cls(
            schema_version=int(payload["schema_version"]),
            artifact_type=str(payload["artifact_type"]),
            coordinate_space=str(payload["coordinate_space"]),  # type: ignore[arg-type]
            homography_image_from_court=_matrix_tuple(payload["homography_image_from_court"], rows=3, cols=3),
            camera_parameters=CourtLockCameraParameters.from_dict(payload["camera_parameters"]),
            distortion=CourtLockDistortion.from_dict(payload["distortion"]),
            transform_covariance=(
                None
                if transform_covariance is None
                else _matrix_tuple(transform_covariance, rows=8, cols=8)
            ),
            source=str(payload["source"]),  # type: ignore[arg-type]
            evidence=CourtLockEvidenceSummary(
                candidate_frame_count=int(evidence_payload["candidate_frame_count"]),
                selected_frame_indices=tuple(int(value) for value in evidence_payload["selected_frame_indices"]),
                pooled_semantic_count=int(evidence_payload["pooled_semantic_count"]),
                selection_algorithm=str(evidence_payload["selection_algorithm"]),
                pooling_algorithm=str(evidence_payload["pooling_algorithm"]),
            ),
            static_motion=StaticMotionDiagnostics.from_dict(payload["static_motion"]),
            residual_px=CourtLockResidual(
                median_px=float(residual_payload["median_px"]),
                p95_px=float(residual_payload["p95_px"]),
            ),
            score_components={str(key): float(value) for key, value in payload["score_components"].items()},
            scorer_version=str(payload["scorer_version"]),
            calibration_version=str(payload["calibration_version"]),
            checkpoint_sha256=(
                None if payload.get("checkpoint_sha256") is None else str(payload["checkpoint_sha256"])
            ),
            measurement_valid=bool(payload["measurement_valid"]),
            authority_state=str(payload["authority_state"]),  # type: ignore[arg-type]
            verified=bool(payload["verified"]),
        )


_COURT_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "coordinate_space",
        "homography_image_from_court",
        "camera_parameters",
        "distortion",
        "transform_covariance",
        "source",
        "evidence",
        "static_motion",
        "residual_px",
        "score_components",
        "scorer_version",
        "calibration_version",
        "checkpoint_sha256",
        "measurement_valid",
        "authority_state",
        "verified",
    }
)
_COURT_LOCK_REQUIRED_KEYS = _COURT_LOCK_KEYS


def select_static_frame_evidence(
    frames: Sequence[StaticFrameEvidence],
    *,
    max_frames: int = 8,
    min_frame_separation: int = 0,
    min_quality: float = 0.0,
) -> tuple[StaticFrameEvidence, ...]:
    """Select at most eight clear frames, independent of input ordering."""

    if isinstance(max_frames, bool) or not 1 <= int(max_frames) <= 8:
        raise ValueError("max_frames must be an integer in [1, 8]")
    if isinstance(min_frame_separation, bool) or int(min_frame_separation) < 0:
        raise ValueError("min_frame_separation must be a non-negative integer")
    _unit_interval(min_quality, name="min_quality")
    unique: dict[int, StaticFrameEvidence] = {}
    for frame in frames:
        previous = unique.get(frame.frame_index)
        if previous is None or _frame_rank_key(frame) < _frame_rank_key(previous):
            unique[frame.frame_index] = frame
    ranked = sorted(unique.values(), key=_frame_rank_key)
    eligible = [frame for frame in ranked if frame.quality_score >= float(min_quality)]
    if not eligible and ranked:
        eligible = ranked[:1]
    selected: list[StaticFrameEvidence] = []
    for frame in eligible:
        if any(abs(frame.frame_index - chosen.frame_index) < int(min_frame_separation) for chosen in selected):
            continue
        selected.append(frame)
        if len(selected) >= int(max_frames):
            break
    return tuple(sorted(selected, key=lambda frame: frame.frame_index))


def pool_static_frame_evidence(
    frames: Sequence[StaticFrameEvidence],
    *,
    max_frames: int = 8,
    min_frame_separation: int = 0,
    min_quality: float = 0.0,
    outlier_floor_px: float = 3.0,
) -> PooledStaticEvidence:
    """Robustly pool observations that share pixels under a fixed camera."""

    if not math.isfinite(float(outlier_floor_px)) or float(outlier_floor_px) <= 0.0:
        raise ValueError("outlier_floor_px must be positive and finite")
    selected = select_static_frame_evidence(
        frames,
        max_frames=max_frames,
        min_frame_separation=min_frame_separation,
        min_quality=min_quality,
    )
    grouped: dict[tuple[str, str], list[tuple[StaticFrameEvidence, StaticCourtObservation]]] = {}
    for frame in selected:
        for observation in frame.observations:
            grouped.setdefault((observation.kind, observation.semantic_name), []).append((frame, observation))
    pooled: list[PooledStaticObservation] = []
    for kind, semantic_name in sorted(grouped):
        rows = grouped[(kind, semantic_name)]
        coordinates = np.asarray([observation.xy for _, observation in rows], dtype=np.float64)
        weights = np.asarray(
            [
                max(1.0e-9, frame.quality_score)
                * max(1.0e-9, observation.confidence)
                / max(1.0, float(np.trace(np.asarray(observation.covariance_px2, dtype=np.float64))))
                for frame, observation in rows
            ],
            dtype=np.float64,
        )
        initial = np.asarray(
            [_weighted_median(coordinates[:, axis], weights) for axis in range(2)], dtype=np.float64
        )
        distances = np.linalg.norm(coordinates - initial[None, :], axis=1)
        distance_median = float(np.median(distances))
        distance_mad = float(np.median(np.abs(distances - distance_median)))
        threshold = max(float(outlier_floor_px), distance_median + 3.0 * 1.4826 * distance_mad)
        inlier_mask = distances <= threshold + 1.0e-12
        if not np.any(inlier_mask):
            inlier_mask[int(np.argmax(weights))] = True
        inlier_coordinates = coordinates[inlier_mask]
        inlier_weights = weights[inlier_mask]
        center = np.asarray(
            [_weighted_median(inlier_coordinates[:, axis], inlier_weights) for axis in range(2)],
            dtype=np.float64,
        )
        normalized_weights = inlier_weights / float(np.sum(inlier_weights))
        input_covariances = [
            np.asarray(rows[index][1].covariance_px2, dtype=np.float64)
            for index, keep in enumerate(inlier_mask)
            if keep
        ]
        covariance = np.zeros((2, 2), dtype=np.float64)
        for weight, coordinate, input_covariance in zip(
            normalized_weights, inlier_coordinates, input_covariances
        ):
            delta = coordinate - center
            covariance += float(weight) ** 2 * input_covariance
            covariance += float(weight) * np.outer(delta, delta)
        covariance = 0.5 * (covariance + covariance.T) + np.eye(2, dtype=np.float64) * 1.0e-9
        inlier_rows = [row for row, keep in zip(rows, inlier_mask) if keep]
        rejected_rows = [row for row, keep in zip(rows, inlier_mask) if not keep]
        temporal_support = len(inlier_rows) / float(max(1, len(selected)))
        evidence_probabilities = [
            min(0.999, frame.quality_score * observation.confidence)
            for frame, observation in inlier_rows
        ]
        fused_probability = 1.0 - math.prod(1.0 - probability for probability in evidence_probabilities)
        confidence = min(0.999, fused_probability * (0.5 + 0.5 * temporal_support))
        line_support = float(
            np.average(
                [observation.line_support for _, observation in inlier_rows],
                weights=inlier_weights,
            )
        )
        pooled.append(
            PooledStaticObservation(
                semantic_name=semantic_name,
                kind=kind,  # type: ignore[arg-type]
                xy=(float(center[0]), float(center[1])),
                confidence=float(confidence),
                covariance_px2=(
                    (float(covariance[0, 0]), float(covariance[0, 1])),
                    (float(covariance[1, 0]), float(covariance[1, 1])),
                ),
                line_support=line_support,
                temporal_support=float(temporal_support),
                contributing_frame_indices=tuple(sorted(frame.frame_index for frame, _ in inlier_rows)),
                rejected_frame_indices=tuple(sorted(frame.frame_index for frame, _ in rejected_rows)),
            )
        )
    return PooledStaticEvidence(
        candidate_frame_count=len({frame.frame_index for frame in frames}),
        selected_frame_indices=tuple(frame.frame_index for frame in selected),
        observations=tuple(pooled),
    )


def diagnose_static_camera(
    transforms: Sequence[FrameCourtTransform],
    *,
    config: StaticMotionConfig | None = None,
) -> StaticMotionDiagnostics:
    """Classify static/moving/ambiguous from court-transform drift and uncertainty."""

    settings = config or StaticMotionConfig()
    unique: dict[int, FrameCourtTransform] = {}
    invalid_count = 0
    for transform in transforms:
        previous = unique.get(transform.frame_index)
        if previous is None or transform.confidence > previous.confidence:
            unique[transform.frame_index] = transform
    valid: list[tuple[FrameCourtTransform, np.ndarray, list[np.ndarray] | None]] = []
    for transform in sorted(unique.values(), key=lambda item: item.frame_index):
        try:
            homography = _homography(transform.homography_image_from_court)
            corners = _project_points(homography, _OUTER_COURT_POINTS_M)
            point_covariances = None
            if transform.transform_covariance is not None:
                point_covariances = [
                    project_planar_point_with_covariance(
                        homography,
                        point,
                        transform.transform_covariance,
                    )[1]
                    for point in _OUTER_COURT_POINTS_M
                ]
        except ValueError:
            invalid_count += 1
            continue
        valid.append((transform, corners, point_covariances))

    frame_indices = tuple(item[0].frame_index for item in valid)
    if len(valid) < settings.min_frames:
        return StaticMotionDiagnostics(
            status="insufficient_evidence",
            camera_motion_suspected=False,
            static_lock_usable=False,
            reference_frame_index=(None if not valid else valid[0][0].frame_index),
            frame_indices=frame_indices,
            valid_frame_count=len(valid),
            invalid_frame_count=invalid_count,
            median_corner_drift_px=None,
            p95_corner_drift_px=None,
            max_corner_drift_px=None,
            net_center_displacement_px=None,
            monotonic_trend_px_per_frame=None,
            drift_uncertainty_px=None,
            reasons=("fewer_than_minimum_valid_transforms",),
        )

    reference_transform, reference_corners, reference_covariances = valid[0]
    drift_values: list[float] = []
    uncertainty_values: list[float] = []
    centers: list[np.ndarray] = []
    center_covariances: list[np.ndarray | None] = []
    for transform, corners, point_covariances in valid:
        centers.append(np.mean(corners, axis=0))
        if transform.transform_covariance is None:
            center_covariances.append(None)
        else:
            center_covariances.append(
                project_planar_point_with_covariance(
                    transform.homography_image_from_court,
                    (0.0, 0.0),
                    transform.transform_covariance,
                )[1]
            )
        if transform.frame_index == reference_transform.frame_index:
            continue
        corner_drifts = np.linalg.norm(corners - reference_corners, axis=1)
        drift_values.extend(float(value) for value in corner_drifts)
        if point_covariances is not None or reference_covariances is not None:
            for index in range(len(_OUTER_COURT_POINTS_M)):
                combined = np.zeros((2, 2), dtype=np.float64)
                if reference_covariances is not None:
                    combined += reference_covariances[index]
                if point_covariances is not None:
                    combined += point_covariances[index]
                uncertainty_values.append(math.sqrt(max(0.0, float(np.trace(combined)))))

    drifts = np.asarray(drift_values, dtype=np.float64)
    median_drift = float(np.median(drifts))
    p95_drift = float(np.percentile(drifts, 95))
    max_drift = float(np.max(drifts))
    uncertainty = float(np.percentile(uncertainty_values, 95)) if uncertainty_values else 0.0
    net_displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    net_uncertainty = 0.0
    if center_covariances[0] is not None or center_covariances[-1] is not None:
        combined_center_covariance = np.zeros((2, 2), dtype=np.float64)
        if center_covariances[0] is not None:
            combined_center_covariance += center_covariances[0]
        if center_covariances[-1] is not None:
            combined_center_covariance += center_covariances[-1]
        net_uncertainty = math.sqrt(max(0.0, float(np.trace(combined_center_covariance))))
    trend = _theil_sen_vector_trend(frame_indices, centers)
    sigma = settings.uncertainty_sigma_multiplier
    moving = (
        median_drift - sigma * uncertainty > settings.median_drift_threshold_px
        or (
            p95_drift - sigma * uncertainty > settings.p95_drift_threshold_px
            and net_displacement - sigma * net_uncertainty > settings.net_displacement_threshold_px
        )
        or (
            net_displacement - sigma * net_uncertainty > settings.net_displacement_threshold_px
            and trend > settings.monotonic_trend_threshold_px_per_frame
        )
    )
    confidently_static = (
        median_drift + sigma * uncertainty <= settings.median_drift_threshold_px
        and p95_drift + sigma * uncertainty <= settings.p95_drift_threshold_px
        and net_displacement + sigma * net_uncertainty <= settings.net_displacement_threshold_px
    )
    if moving:
        status: MotionStatus = "moving"
        reasons = ("court_projection_drift_exceeds_static_threshold",)
    elif confidently_static:
        status = "static"
        reasons = ()
    else:
        status = "ambiguous"
        reasons = ("static_threshold_overlaps_transform_uncertainty",)
    return StaticMotionDiagnostics(
        status=status,
        camera_motion_suspected=status in {"moving", "ambiguous"},
        static_lock_usable=status == "static",
        reference_frame_index=reference_transform.frame_index,
        frame_indices=frame_indices,
        valid_frame_count=len(valid),
        invalid_frame_count=invalid_count,
        median_corner_drift_px=median_drift,
        p95_corner_drift_px=p95_drift,
        max_corner_drift_px=max_drift,
        net_center_displacement_px=net_displacement,
        monotonic_trend_px_per_frame=trend,
        drift_uncertainty_px=max(uncertainty, net_uncertainty),
        reasons=reasons,
    )


def write_court_lock(lock: CourtLockArtifact, path: str | Path) -> None:
    """Atomically write canonical ``court_lock.json`` bytes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(lock.to_json(), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_court_lock(path: str | Path) -> CourtLockArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("court_lock.json must contain a JSON object")
    return CourtLockArtifact.from_dict(payload)


def _frame_rank_key(frame: StaticFrameEvidence) -> tuple[Any, ...]:
    observation_identity = tuple(
        sorted(
            (
                observation.kind,
                observation.semantic_name,
                observation.xy,
                observation.confidence,
            )
            for observation in frame.observations
        )
    )
    return (-frame.quality_score, -len(frame.observations), frame.frame_index, observation_identity)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.lexsort((np.arange(len(values)), values))
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _project_points(homography: np.ndarray, court_points: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(court_points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) <= 1.0e-12):
        raise ValueError("court transform projects a diagnostic point to infinity")
    pixels = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("court transform projects non-finite diagnostic points")
    return pixels


def _theil_sen_vector_trend(frame_indices: Sequence[int], centers: Sequence[np.ndarray]) -> float:
    slopes_x: list[float] = []
    slopes_y: list[float] = []
    for left in range(len(frame_indices)):
        for right in range(left + 1, len(frame_indices)):
            delta_frame = int(frame_indices[right]) - int(frame_indices[left])
            if delta_frame <= 0:
                continue
            delta = (centers[right] - centers[left]) / float(delta_frame)
            slopes_x.append(float(delta[0]))
            slopes_y.append(float(delta[1]))
    if not slopes_x:
        return 0.0
    return float(math.hypot(float(np.median(slopes_x)), float(np.median(slopes_y))))


def _validate_frame_indices(indices: Sequence[int], *, max_count: int | None = 8) -> None:
    normalized = tuple(int(value) for value in indices)
    if any(isinstance(value, bool) or int(value) < 0 for value in indices):
        raise ValueError("frame indices must be non-negative integers")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("frame indices must be sorted and unique")
    if max_count is not None and len(normalized) > max_count:
        raise ValueError(f"at most {max_count} frame indices are allowed")


def _unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _finite_vector(values: Sequence[float], *, size: int, name: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def _covariance(values: Sequence[Sequence[float]], *, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    if not np.allclose(matrix, matrix.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(matrix).min(initial=0.0) < -1.0e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _homography(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography_image_from_court must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise ValueError("homography_image_from_court must be nonsingular")
    if abs(float(matrix[2, 2])) <= 1.0e-12:
        raise ValueError("homography h22 must be nonzero")
    return matrix / float(matrix[2, 2])


def _matrix_tuple(values: Any, *, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (rows, cols) or not np.isfinite(matrix).all():
        raise ValueError(f"matrix must be finite with shape {rows}x{cols}")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _nested_lists(values: Sequence[Sequence[float]] | None) -> list[list[float]] | None:
    if values is None:
        return None
    return [[float(value) for value in row] for row in values]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "COURT_LOCK_ARTIFACT_TYPE",
    "COURT_LOCK_POOLING_ALGORITHM",
    "COURT_LOCK_SCHEMA_VERSION",
    "COURT_LOCK_SELECTION_ALGORITHM",
    "CourtLockArtifact",
    "CourtLockCameraParameters",
    "CourtLockDistortion",
    "CourtLockEvidenceSummary",
    "CourtLockResidual",
    "FrameCourtTransform",
    "PooledStaticEvidence",
    "PooledStaticObservation",
    "StaticCourtObservation",
    "StaticFrameEvidence",
    "StaticMotionConfig",
    "StaticMotionDiagnostics",
    "diagnose_static_camera",
    "pool_static_frame_evidence",
    "read_court_lock",
    "select_static_frame_evidence",
    "write_court_lock",
]
