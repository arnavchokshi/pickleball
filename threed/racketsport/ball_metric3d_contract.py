"""A-3 metric-3D observation contract: triangulated GT + solver-observation logs.

Measurement-only schema module for the ball-3D program (Phase A of
``runs/ball3d_lifting_plan_20260723/PLAN.md``, §5.A + v2 reframe). It defines
the two versioned, validated record shapes the metric-3D lane exchanges:

1. **Triangulated ground-truth observations** (``GroundTruthObservationSet``):
   per-frame world position WITH its evidence — per-axis sigma, contributing
   camera set, triangulation residual, and quality flags. Never a bare xyz;
   evaluation must always be able to see how trustworthy each GT sample is.
2. **Solver-observation logs** (``SolverObservationLog``): a frozen record of
   exactly what the production solver saw per frame — pixel observation,
   world ray (only when the calibration is sha-verified against the solve's
   recorded input; fail closed otherwise), candidate-set summary, anchor
   events available at that frame, and the fail-closed solver verdict.

Validation is fail-closed: missing per-axis sigmas, non-monotonic timestamps,
unknown quality flags / statuses / verdicts, and unknown schema versions are
rejected with ``ContractValidationError``. ``VERIFIED=0`` stays binding —
nothing in this module measures accuracy or promotes anything; it only fixes
the shapes that the frozen A-4 judge and future lifter work consume.

Serialization is deterministic: sorted keys, floats rounded to
``FLOAT_DECIMALS``, no timestamps/hostnames/absolute paths in the payload
(source artifacts are recorded as root-relative path + sha256).

Scope note vs PLAN v2 A-3: per-axis sigma + camera set + residual + quality
flags are REQUIRED; the full world covariance, triangulation angle, and
review state are optional evidence fields (validated when present, never
fabricated). Per-camera uv observations/covariances/residuals, per-camera
visibility/occlusion, and the raw-vs-smoothed dual GT reference are schema-v2
extensions to add when the multi-view rig (A-1) starts producing them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
GT_ARTIFACT_TYPE = "racketsport_ball_metric3d_gt_observations"
SOLVER_LOG_ARTIFACT_TYPE = "racketsport_ball_solver_observation_log"
FLOAT_DECIMALS = 6

# Metric world frame fixed by PLAN §5.A A3 and matching the production court
# calibration artifacts: x width, y along the baseline direction (net plane
# y=0), z up, origin under the net center, court surface z=0, meters.
WORLD_FRAME = "court_netcenter_z_up_m"

# GT quality flags (PLAN v2 A-3: evidence + quality, not one XYZ). Unknown
# flags are rejected so a typo can never silently create a new quality tier.
KNOWN_QUALITY_FLAGS = frozenset(
    {
        "gold",
        "reviewed",
        "weak_triangulation_geometry",
        "partially_occluded",
        "interpolated_review",
        "low_confidence",
    }
)

# Observed / weakly-observed / ambiguous / missing / inferred distinction
# (PLAN §5.A A3): preserved verbatim, never collapsed before the 3D model.
OBSERVATION_STATUSES = frozenset(
    {"observed", "weakly_observed", "ambiguous", "missing", "inferred"}
)

# Fail-closed solver verdict vocabulary (mirrors the characterization
# harness's accepted-frame definition; "hidden" = no usable world position).
SOLVER_VERDICTS = frozenset({"accepted", "rejected_fail_closed", "hidden"})

RAY_STATUS_COMPUTED = "computed"
RAY_STATUSES = frozenset(
    {
        RAY_STATUS_COMPUTED,
        "no_pixel",
        "missing_calibration",
        "calibration_not_sha_verified",
    }
)

_TIMESTAMP_EPS_S = 0.0
_UNIT_NORM_TOLERANCE = 1e-4


class ContractValidationError(ValueError):
    """Raised when a payload violates the A-3 observation contract."""


# ---------------------------------------------------------------------------
# Triangulated ground-truth observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthObservation:
    """One triangulated GT ball sample with its evidence, never a bare xyz.

    ``sigma_xyz_m`` (per-axis, required) is the minimum uncertainty evidence.
    The optional fields carry the fuller PLAN v2 A-3 evidence when the GT rig
    produces it: the full 3x3 world covariance, the triangulation angle, and
    the human-review state. Optional means "absent when not measured" —
    never fabricated, and validated whenever present.
    """

    timestamp_s: float
    xyz_world_m: tuple[float, float, float]
    sigma_xyz_m: tuple[float, float, float]
    cameras_used: tuple[str, ...]
    triangulation_residual_px: float
    quality_flags: tuple[str, ...] = ()
    covariance_world_m2: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None
    triangulation_angle_deg: float | None = None
    reviewed: bool | None = None

    def validate(self, *, path: str = "observation") -> None:
        _require_finite(self.timestamp_s, path=f"{path}.timestamp_s")
        _require_vec3(self.xyz_world_m, path=f"{path}.xyz_world_m")
        _require_vec3(self.sigma_xyz_m, path=f"{path}.sigma_xyz_m")
        for axis, sigma in zip("xyz", self.sigma_xyz_m):
            if not (float(sigma) > 0.0):
                raise ContractValidationError(
                    f"{path}.sigma_xyz_m.{axis}: per-axis sigma must be > 0, got {sigma!r}"
                )
        if not self.cameras_used:
            raise ContractValidationError(f"{path}.cameras_used: must not be empty")
        for index, camera in enumerate(self.cameras_used):
            if not isinstance(camera, str) or not camera:
                raise ContractValidationError(
                    f"{path}.cameras_used[{index}]: expected non-empty string, got {camera!r}"
                )
        _require_finite(
            self.triangulation_residual_px, path=f"{path}.triangulation_residual_px"
        )
        if float(self.triangulation_residual_px) < 0.0:
            raise ContractValidationError(
                f"{path}.triangulation_residual_px: must be >= 0, "
                f"got {self.triangulation_residual_px!r}"
            )
        unknown = sorted(set(self.quality_flags) - KNOWN_QUALITY_FLAGS)
        if unknown:
            raise ContractValidationError(
                f"{path}.quality_flags: unknown flags {unknown}; "
                f"known: {sorted(KNOWN_QUALITY_FLAGS)}"
            )
        if self.covariance_world_m2 is not None:
            _require_matrix3(self.covariance_world_m2, path=f"{path}.covariance_world_m2")
            for axis_index, axis in enumerate("xyz"):
                if float(self.covariance_world_m2[axis_index][axis_index]) < 0.0:
                    raise ContractValidationError(
                        f"{path}.covariance_world_m2: negative {axis}{axis} variance"
                    )
        if self.triangulation_angle_deg is not None:
            _require_finite(
                self.triangulation_angle_deg, path=f"{path}.triangulation_angle_deg"
            )
        if self.reviewed is not None and not isinstance(self.reviewed, bool):
            raise ContractValidationError(f"{path}.reviewed: expected bool or null")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "xyz_world_m": [float(v) for v in self.xyz_world_m],
            "sigma_xyz_m": [float(v) for v in self.sigma_xyz_m],
            "cameras_used": list(self.cameras_used),
            "triangulation_residual_px": float(self.triangulation_residual_px),
            "quality_flags": sorted(self.quality_flags),
            "covariance_world_m2": (
                [[float(v) for v in row] for row in self.covariance_world_m2]
                if self.covariance_world_m2 is not None
                else None
            ),
            "triangulation_angle_deg": _float_or_none_json(self.triangulation_angle_deg),
            "reviewed": self.reviewed,
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "observation") -> "GroundTruthObservation":
        record = _require_mapping(payload, path=path)
        raw_covariance = record.get("covariance_world_m2")
        covariance = None
        if raw_covariance is not None:
            _require_matrix3(raw_covariance, path=f"{path}.covariance_world_m2")
            covariance = tuple(
                (float(row[0]), float(row[1]), float(row[2])) for row in raw_covariance
            )
        raw_reviewed = record.get("reviewed")
        if raw_reviewed is not None and not isinstance(raw_reviewed, bool):
            raise ContractValidationError(f"{path}.reviewed: expected bool or null")
        observation = cls(
            timestamp_s=_read_float(record, "timestamp_s", path=path),
            xyz_world_m=_read_vec3(record, "xyz_world_m", path=path),
            sigma_xyz_m=_read_vec3(record, "sigma_xyz_m", path=path),
            cameras_used=tuple(_read_str_list(record, "cameras_used", path=path)),
            triangulation_residual_px=_read_float(
                record, "triangulation_residual_px", path=path
            ),
            quality_flags=tuple(
                _read_str_list(record, "quality_flags", path=path, default=[])
            ),
            covariance_world_m2=covariance,
            triangulation_angle_deg=_read_optional_float(
                record, "triangulation_angle_deg", path=path
            ),
            reviewed=raw_reviewed,
        )
        observation.validate(path=path)
        return observation


@dataclass(frozen=True)
class GroundTruthObservationSet:
    """Versioned per-clip set of triangulated GT observations."""

    clip: str
    observations: tuple[GroundTruthObservation, ...]
    schema_version: int = SCHEMA_VERSION
    world_frame: str = WORLD_FRAME

    def validate(self) -> None:
        _require_schema_version(self.schema_version, path="ground_truth.schema_version")
        if not isinstance(self.clip, str) or not self.clip:
            raise ContractValidationError("ground_truth.clip: expected non-empty string")
        if self.world_frame != WORLD_FRAME:
            raise ContractValidationError(
                f"ground_truth.world_frame: expected {WORLD_FRAME!r}, got {self.world_frame!r}"
            )
        for index, observation in enumerate(self.observations):
            observation.validate(path=f"ground_truth.observations[{index}]")
        _require_strictly_increasing(
            [obs.timestamp_s for obs in self.observations],
            path="ground_truth.observations",
        )

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return _round_floats(
            {
                "schema_version": self.schema_version,
                "artifact_type": GT_ARTIFACT_TYPE,
                "world_frame": self.world_frame,
                "clip": self.clip,
                "observations": [obs.to_json_dict() for obs in self.observations],
            }
        )

    @classmethod
    def from_json_dict(cls, payload: Any) -> "GroundTruthObservationSet":
        record = _require_mapping(payload, path="ground_truth")
        _require_artifact_type(record, GT_ARTIFACT_TYPE, path="ground_truth")
        raw_observations = record.get("observations")
        if not isinstance(raw_observations, list):
            raise ContractValidationError("ground_truth.observations: expected a list")
        result = cls(
            clip=_read_str(record, "clip", path="ground_truth"),
            observations=tuple(
                GroundTruthObservation.from_json_dict(
                    item, path=f"ground_truth.observations[{index}]"
                )
                for index, item in enumerate(raw_observations)
            ),
            schema_version=_read_int(record, "schema_version", path="ground_truth"),
            world_frame=_read_str(record, "world_frame", path="ground_truth"),
        )
        result.validate()
        return result


# ---------------------------------------------------------------------------
# Solver-observation log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldRay:
    """Camera-origin world ray for one pixel observation (unit direction)."""

    origin_m: tuple[float, float, float]
    direction: tuple[float, float, float]

    def validate(self, *, path: str = "ray") -> None:
        _require_vec3(self.origin_m, path=f"{path}.origin_m")
        _require_vec3(self.direction, path=f"{path}.direction")
        norm = math.sqrt(sum(float(v) * float(v) for v in self.direction))
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ContractValidationError(
                f"{path}.direction: expected unit vector, got norm {norm!r}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "origin_m": [float(v) for v in self.origin_m],
            "direction": [float(v) for v in self.direction],
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "ray") -> "WorldRay":
        record = _require_mapping(payload, path=path)
        ray = cls(
            origin_m=_read_vec3(record, "origin_m", path=path),
            direction=_read_vec3(record, "direction", path=path),
        )
        ray.validate(path=path)
        return ray


@dataclass(frozen=True)
class CandidateSetSummary:
    """Per-frame summary of the solver's candidate evidence.

    ``candidate_count`` is ``None`` when the source artifact did not persist a
    per-frame candidate set (the current arc-solved artifacts do not); the
    distinction between "no candidates recorded" and "zero candidates" is
    preserved rather than collapsed to 0.
    """

    candidate_count: int | None = None
    selected_residual_px: float | None = None
    inlier_sighting: bool | None = None
    outlier_pruned: bool | None = None
    rescued: bool | None = None

    def validate(self, *, path: str = "candidate_summary") -> None:
        if self.candidate_count is not None:
            if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
                raise ContractValidationError(f"{path}.candidate_count: expected int or null")
            if self.candidate_count < 0:
                raise ContractValidationError(f"{path}.candidate_count: must be >= 0")
        if self.selected_residual_px is not None:
            _require_finite(self.selected_residual_px, path=f"{path}.selected_residual_px")
            if float(self.selected_residual_px) < 0.0:
                raise ContractValidationError(f"{path}.selected_residual_px: must be >= 0")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "selected_residual_px": _float_or_none_json(self.selected_residual_px),
            "inlier_sighting": self.inlier_sighting,
            "outlier_pruned": self.outlier_pruned,
            "rescued": self.rescued,
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "candidate_summary") -> "CandidateSetSummary":
        record = _require_mapping(payload, path=path)
        summary = cls(
            candidate_count=_read_optional_int(record, "candidate_count", path=path),
            selected_residual_px=_read_optional_float(record, "selected_residual_px", path=path),
            inlier_sighting=_read_optional_bool(record, "inlier_sighting", path=path),
            outlier_pruned=_read_optional_bool(record, "outlier_pruned", path=path),
            rescued=_read_optional_bool(record, "rescued", path=path),
        )
        summary.validate(path=path)
        return summary


@dataclass(frozen=True)
class AnchorEvent:
    """One anchor event available to the solver at a frame."""

    anchor_id: str
    kind: str
    status: str
    source: str

    def validate(self, *, path: str = "anchor_event") -> None:
        for name, value in (
            ("anchor_id", self.anchor_id),
            ("kind", self.kind),
            ("status", self.status),
            ("source", self.source),
        ):
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"{path}.{name}: expected non-empty string")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "kind": self.kind,
            "status": self.status,
            "source": self.source,
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "anchor_event") -> "AnchorEvent":
        record = _require_mapping(payload, path=path)
        event = cls(
            anchor_id=_read_str(record, "anchor_id", path=path),
            kind=_read_str(record, "kind", path=path),
            status=_read_str(record, "status", path=path),
            source=_read_str(record, "source", path=path),
        )
        event.validate(path=path)
        return event


@dataclass(frozen=True)
class SolverFrameObservation:
    """What the solver saw + concluded at one frame (evidence preserved)."""

    frame_index: int
    timestamp_s: float
    observation_status: str
    pixel_xy: tuple[float, float] | None
    pixel_confidence: float | None
    ray: WorldRay | None
    ray_status: str
    candidate_summary: CandidateSetSummary
    anchor_events: tuple[AnchorEvent, ...]
    solver_verdict: str
    segment_id: int | None = None
    band: str | None = None

    def validate(self, *, path: str = "frame") -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ContractValidationError(f"{path}.frame_index: expected int")
        if self.frame_index < 0:
            raise ContractValidationError(f"{path}.frame_index: must be >= 0")
        _require_finite(self.timestamp_s, path=f"{path}.timestamp_s")
        if self.observation_status not in OBSERVATION_STATUSES:
            raise ContractValidationError(
                f"{path}.observation_status: unknown status {self.observation_status!r}; "
                f"known: {sorted(OBSERVATION_STATUSES)}"
            )
        if self.pixel_xy is not None:
            _require_vec2(self.pixel_xy, path=f"{path}.pixel_xy")
        elif self.observation_status == "observed":
            raise ContractValidationError(
                f"{path}.pixel_xy: required when observation_status is 'observed'"
            )
        if self.pixel_confidence is not None:
            _require_finite(self.pixel_confidence, path=f"{path}.pixel_confidence")
        if self.ray_status not in RAY_STATUSES:
            raise ContractValidationError(
                f"{path}.ray_status: unknown status {self.ray_status!r}; "
                f"known: {sorted(RAY_STATUSES)}"
            )
        if self.ray_status == RAY_STATUS_COMPUTED:
            if self.ray is None:
                raise ContractValidationError(
                    f"{path}.ray: required when ray_status is '{RAY_STATUS_COMPUTED}'"
                )
            self.ray.validate(path=f"{path}.ray")
        elif self.ray is not None:
            raise ContractValidationError(
                f"{path}.ray: must be null when ray_status is {self.ray_status!r}"
            )
        self.candidate_summary.validate(path=f"{path}.candidate_summary")
        for index, event in enumerate(self.anchor_events):
            event.validate(path=f"{path}.anchor_events[{index}]")
        if self.solver_verdict not in SOLVER_VERDICTS:
            raise ContractValidationError(
                f"{path}.solver_verdict: unknown verdict {self.solver_verdict!r}; "
                f"known: {sorted(SOLVER_VERDICTS)}"
            )
        if self.segment_id is not None and (
            isinstance(self.segment_id, bool) or not isinstance(self.segment_id, int)
        ):
            raise ContractValidationError(f"{path}.segment_id: expected int or null")
        if self.band is not None and not isinstance(self.band, str):
            raise ContractValidationError(f"{path}.band: expected string or null")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": float(self.timestamp_s),
            "observation_status": self.observation_status,
            "pixel_xy": [float(v) for v in self.pixel_xy] if self.pixel_xy is not None else None,
            "pixel_confidence": _float_or_none_json(self.pixel_confidence),
            "ray": self.ray.to_json_dict() if self.ray is not None else None,
            "ray_status": self.ray_status,
            "candidate_summary": self.candidate_summary.to_json_dict(),
            "anchor_events": [event.to_json_dict() for event in self.anchor_events],
            "solver_verdict": self.solver_verdict,
            "segment_id": self.segment_id,
            "band": self.band,
        }

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "frame") -> "SolverFrameObservation":
        record = _require_mapping(payload, path=path)
        raw_pixel = record.get("pixel_xy")
        raw_ray = record.get("ray")
        raw_events = record.get("anchor_events")
        if not isinstance(raw_events, list):
            raise ContractValidationError(f"{path}.anchor_events: expected a list")
        frame = cls(
            frame_index=_read_int(record, "frame_index", path=path),
            timestamp_s=_read_float(record, "timestamp_s", path=path),
            observation_status=_read_str(record, "observation_status", path=path),
            pixel_xy=None if raw_pixel is None else _coerce_vec2(raw_pixel, path=f"{path}.pixel_xy"),
            pixel_confidence=_read_optional_float(record, "pixel_confidence", path=path),
            ray=None if raw_ray is None else WorldRay.from_json_dict(raw_ray, path=f"{path}.ray"),
            ray_status=_read_str(record, "ray_status", path=path),
            candidate_summary=CandidateSetSummary.from_json_dict(
                record.get("candidate_summary"), path=f"{path}.candidate_summary"
            ),
            anchor_events=tuple(
                AnchorEvent.from_json_dict(item, path=f"{path}.anchor_events[{index}]")
                for index, item in enumerate(raw_events)
            ),
            solver_verdict=_read_str(record, "solver_verdict", path=path),
            segment_id=_read_optional_int(record, "segment_id", path=path),
            band=_read_optional_str(record, "band", path=path),
        )
        frame.validate(path=path)
        return frame


@dataclass(frozen=True)
class SourceArtifact:
    """Provenance record for one input the log was built from."""

    kind: str
    path: str
    sha256: str

    def validate(self, *, path: str = "input") -> None:
        for name, value in (("kind", self.kind), ("path", self.path), ("sha256", self.sha256)):
            if not isinstance(value, str) or not value:
                raise ContractValidationError(f"{path}.{name}: expected non-empty string")

    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_json_dict(cls, payload: Any, *, path: str = "input") -> "SourceArtifact":
        record = _require_mapping(payload, path=path)
        artifact = cls(
            kind=_read_str(record, "kind", path=path),
            path=_read_str(record, "path", path=path),
            sha256=_read_str(record, "sha256", path=path),
        )
        artifact.validate(path=path)
        return artifact


@dataclass(frozen=True)
class SolverObservationLog:
    """Versioned per-clip solver-observation log (A-5 dataset artifact)."""

    clip: str
    frames: tuple[SolverFrameObservation, ...]
    inputs: tuple[SourceArtifact, ...] = ()
    calibration_sha_verified: bool | None = None
    # Clip identity recorded inside the source solved artifact (its
    # ``clip_id``); ``None`` when the artifact carries none. Distinct from
    # ``clip``, the caller-chosen log name.
    source_clip_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    world_frame: str = WORLD_FRAME

    def validate(self) -> None:
        _require_schema_version(self.schema_version, path="solver_log.schema_version")
        if not isinstance(self.clip, str) or not self.clip:
            raise ContractValidationError("solver_log.clip: expected non-empty string")
        if self.source_clip_id is not None and (
            not isinstance(self.source_clip_id, str) or not self.source_clip_id
        ):
            raise ContractValidationError(
                "solver_log.source_clip_id: expected non-empty string or null"
            )
        if self.world_frame != WORLD_FRAME:
            raise ContractValidationError(
                f"solver_log.world_frame: expected {WORLD_FRAME!r}, got {self.world_frame!r}"
            )
        for index, frame in enumerate(self.frames):
            frame.validate(path=f"solver_log.frames[{index}]")
        _require_strictly_increasing(
            [frame.timestamp_s for frame in self.frames], path="solver_log.frames"
        )
        _require_strictly_increasing(
            [float(frame.frame_index) for frame in self.frames],
            path="solver_log.frames.frame_index",
        )
        for index, artifact in enumerate(self.inputs):
            artifact.validate(path=f"solver_log.inputs[{index}]")
        if self.calibration_sha_verified is not None and not isinstance(
            self.calibration_sha_verified, bool
        ):
            raise ContractValidationError(
                "solver_log.calibration_sha_verified: expected bool or null"
            )

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return _round_floats(
            {
                "schema_version": self.schema_version,
                "artifact_type": SOLVER_LOG_ARTIFACT_TYPE,
                "world_frame": self.world_frame,
                "clip": self.clip,
                "source_clip_id": self.source_clip_id,
                "calibration_sha_verified": self.calibration_sha_verified,
                "inputs": [artifact.to_json_dict() for artifact in self.inputs],
                "frames": [frame.to_json_dict() for frame in self.frames],
            }
        )

    @classmethod
    def from_json_dict(cls, payload: Any) -> "SolverObservationLog":
        record = _require_mapping(payload, path="solver_log")
        _require_artifact_type(record, SOLVER_LOG_ARTIFACT_TYPE, path="solver_log")
        raw_frames = record.get("frames")
        if not isinstance(raw_frames, list):
            raise ContractValidationError("solver_log.frames: expected a list")
        raw_inputs = record.get("inputs")
        if not isinstance(raw_inputs, list):
            raise ContractValidationError("solver_log.inputs: expected a list")
        raw_verified = record.get("calibration_sha_verified")
        if raw_verified is not None and not isinstance(raw_verified, bool):
            raise ContractValidationError(
                "solver_log.calibration_sha_verified: expected bool or null"
            )
        log = cls(
            clip=_read_str(record, "clip", path="solver_log"),
            frames=tuple(
                SolverFrameObservation.from_json_dict(
                    item, path=f"solver_log.frames[{index}]"
                )
                for index, item in enumerate(raw_frames)
            ),
            inputs=tuple(
                SourceArtifact.from_json_dict(item, path=f"solver_log.inputs[{index}]")
                for index, item in enumerate(raw_inputs)
            ),
            calibration_sha_verified=raw_verified,
            source_clip_id=_read_optional_str(record, "source_clip_id", path="solver_log"),
            schema_version=_read_int(record, "schema_version", path="solver_log"),
            world_frame=_read_str(record, "world_frame", path="solver_log"),
        )
        log.validate()
        return log


# ---------------------------------------------------------------------------
# Deterministic JSON (de)serialization helpers
# ---------------------------------------------------------------------------


def dumps_contract_json(payload: Mapping[str, Any]) -> str:
    """Deterministic bytes: sorted keys, fixed separators, trailing newline."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_ground_truth(path: str | Path, ground_truth: GroundTruthObservationSet) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_contract_json(ground_truth.to_json_dict()), encoding="utf-8")
    return target


def read_ground_truth(path: str | Path) -> GroundTruthObservationSet:
    return GroundTruthObservationSet.from_json_dict(_read_json(Path(path)))


def write_solver_observation_log(path: str | Path, log: SolverObservationLog) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_contract_json(log.to_json_dict()), encoding="utf-8")
    return target


def read_solver_observation_log(path: str | Path) -> SolverObservationLog:
    return SolverObservationLog.from_json_dict(_read_json(Path(path)))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"{path.name}: invalid JSON ({exc})") from exc


# ---------------------------------------------------------------------------
# Field readers / validators (fail closed on shape drift)
# ---------------------------------------------------------------------------


def _require_mapping(payload: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"{path}: expected a JSON object")
    return payload


def _require_artifact_type(record: Mapping[str, Any], expected: str, *, path: str) -> None:
    actual = record.get("artifact_type")
    if actual != expected:
        raise ContractValidationError(
            f"{path}.artifact_type: expected {expected!r}, got {actual!r}"
        )


def _require_schema_version(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path}: expected int schema_version")
    if value != SCHEMA_VERSION:
        raise ContractValidationError(
            f"{path}: unsupported schema_version {value!r} (supported: {SCHEMA_VERSION})"
        )


def _require_strictly_increasing(values: Sequence[float], *, path: str) -> None:
    for index in range(1, len(values)):
        if not (float(values[index]) - float(values[index - 1]) > _TIMESTAMP_EPS_S):
            raise ContractValidationError(
                f"{path}: non-monotonic order at index {index} "
                f"({values[index - 1]!r} -> {values[index]!r})"
            )


def _require_finite(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{path}: expected a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ContractValidationError(f"{path}: expected a finite number, got {value!r}")


def _require_vec3(value: Any, *, path: str) -> None:
    _require_vec(value, 3, path=path)


def _require_vec2(value: Any, *, path: str) -> None:
    _require_vec(value, 2, path=path)


def _require_matrix3(value: Any, *, path: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ContractValidationError(f"{path}: expected a 3x3 matrix")
    for row_index, row in enumerate(value):
        _require_vec3(row, path=f"{path}[{row_index}]")


def _require_vec(value: Any, length: int, *, path: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ContractValidationError(f"{path}: expected a length-{length} vector")
    for index, item in enumerate(value):
        _require_finite(item, path=f"{path}[{index}]")


def _read_float(record: Mapping[str, Any], key: str, *, path: str) -> float:
    if key not in record:
        raise ContractValidationError(f"{path}.{key}: missing required field")
    value = record[key]
    _require_finite(value, path=f"{path}.{key}")
    return float(value)


def _read_optional_float(record: Mapping[str, Any], key: str, *, path: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    _require_finite(value, path=f"{path}.{key}")
    return float(value)


def _read_int(record: Mapping[str, Any], key: str, *, path: str) -> int:
    if key not in record:
        raise ContractValidationError(f"{path}.{key}: missing required field")
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path}.{key}: expected int, got {value!r}")
    return value


def _read_optional_int(record: Mapping[str, Any], key: str, *, path: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path}.{key}: expected int or null, got {value!r}")
    return value


def _read_optional_bool(record: Mapping[str, Any], key: str, *, path: str) -> bool | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ContractValidationError(f"{path}.{key}: expected bool or null, got {value!r}")
    return value


def _read_str(record: Mapping[str, Any], key: str, *, path: str) -> str:
    if key not in record:
        raise ContractValidationError(f"{path}.{key}: missing required field")
    value = record[key]
    if not isinstance(value, str):
        raise ContractValidationError(f"{path}.{key}: expected string, got {value!r}")
    return value


def _read_optional_str(record: Mapping[str, Any], key: str, *, path: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractValidationError(f"{path}.{key}: expected string or null, got {value!r}")
    return value


def _read_str_list(
    record: Mapping[str, Any],
    key: str,
    *,
    path: str,
    default: list[str] | None = None,
) -> list[str]:
    if key not in record:
        if default is not None:
            return list(default)
        raise ContractValidationError(f"{path}.{key}: missing required field")
    value = record[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractValidationError(f"{path}.{key}: expected a list of strings")
    return list(value)


def _read_vec3(record: Mapping[str, Any], key: str, *, path: str) -> tuple[float, float, float]:
    if key not in record or record[key] is None:
        raise ContractValidationError(f"{path}.{key}: missing required field")
    value = record[key]
    _require_vec3(value, path=f"{path}.{key}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _coerce_vec2(value: Any, *, path: str) -> tuple[float, float]:
    _require_vec2(value, path=path)
    return (float(value[0]), float(value[1]))


def _float_or_none_json(value: float | None) -> float | None:
    return None if value is None else float(value)


def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value
