"""Default-off classical court-line hardening for static cameras.

The module is deliberately additive.  It never changes the legacy
``detect_court_keypoints_from_image`` path, never overwrites per-frame
observations, and never promotes an automatic solve.  A preliminary court
geometry supplies a regulation-template ROI; raw frame evidence is assigned
inside that ROI, pooled with deterministic robust medians, and offered to the
existing guarded point/line optimizer as a preview-only candidate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from heapq import heappop, heappush
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .court_calibration import (
    homography_from_planar_points,
    project_planar_points,
    reprojection_error,
)
from .court_line_evidence import (
    aggregate_court_line_evidence,
    required_court_line_ids,
    required_court_net_ids,
    select_best_line_observation,
)
from .court_line_keypoints import (
    DetectedCourtLineCandidate,
    detect_court_line_candidates_from_image,
)
from .court_keypoint_net import PICKLEBALL_KEYPOINTS
from .court_templates import get_court_template
from .schemas import (
    CourtLineEvidence,
    CourtLineObservation,
    PICKLEBALL_COURT_KEYPOINT_NAMES,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_static_court_line_hardening"
RAW_EVIDENCE_ARTIFACT_TYPE = "racketsport_raw_frame_court_line_evidence"
POOLED_EVIDENCE_ARTIFACT_TYPE = "racketsport_pooled_static_court_line_evidence"
DETECTOR_IMPLEMENTATION = "seed_guided_paired_gradient_edges_v1"
POOL_IMPLEMENTATION = "static_sample_median_geometry_filter_v1"
REFINEMENT_IMPLEMENTATION = "existing_guarded_point_line_optimizer_adapter_v1"
STATIC_CONSISTENCY_IMPLEMENTATION = (
    "cross_frame_semantic_offset_mad_changepoint_raw_coverage_v4"
)
PROVEN_POOL_SAMPLE_COUNT = 96
PROVEN_POOL_PROVIDER = "seed_guided_paired_edges"
PAINT_WIDTH_M = 0.0508
FLOOR_LINE_IDS: tuple[str, ...] = (
    "far_baseline",
    "far_nvz",
    "near_nvz",
    "near_baseline",
    "left_sideline",
    "near_centerline",
    "far_centerline",
    "right_sideline",
)
CROSS_LINE_IDS = frozenset({"far_baseline", "far_nvz", "near_nvz", "near_baseline"})
LONGITUDINAL_LINE_IDS = frozenset({"left_sideline", "near_centerline", "far_centerline", "right_sideline"})

Point2 = tuple[float, float]
Segment2 = tuple[Point2, Point2]


@dataclass(frozen=True)
class CourtLineHardeningConfig:
    """Pre-registered controls for the bounded, preview-only candidate."""

    enabled: bool = False
    provider: str = "seed_guided_paired_edges"
    preprocessing: str = "raw"
    white_threshold: int = 200
    roi_polygon_px: tuple[Point2, ...] | None = None
    roi_margin_px: float = 18.0
    max_angle_delta_deg: float = 8.0
    max_normal_distance_px: float = 18.0
    min_overlap_fraction: float = 0.10
    min_candidate_margin: float = 0.06
    min_joint_intersections: int = 6
    max_joint_homography_p90_px: float = 3.0
    max_joint_assignment_hypotheses: int = 64
    min_assigned_lines: int = 4
    min_contributing_frames: int = 3
    min_holdout_contributing_frames: int = 2
    max_static_line_mad_px: float = 3.0
    pool_outlier_mad_scale: float = 3.5
    frame_holdout_stride: int = 4
    profile_sample_count: int = 41
    profile_step_px: float = 0.25
    profile_scan_radius_px: float = 24.0
    profile_roi_radius_px: float = 18.0
    min_band_contrast: float = 7.0
    min_edge_strength: float = 4.0
    min_edge_symmetry: float = 0.25
    paint_width_low_factor: float = 0.42
    paint_width_high_factor: float = 2.0
    min_pooled_line_samples: int = 10
    min_heldout_line_samples: int = 6
    max_line_geometry_p90_px: float = 1.75
    heldout_p90_tolerance_px: float = 0.25
    heldout_line_family_p90_tolerance_px: float = 1.0
    line_weight: float = 0.60
    point_weight: float = 0.40

    def validate(self) -> None:
        if self.provider not in {
            "legacy_hough",
            "classical_paired_edges",
            "hybrid_paired_hough",
            "seed_guided_paired_edges",
        }:
            raise ValueError(f"unsupported court-line hardening provider: {self.provider}")
        if self.preprocessing != "raw":
            raise ValueError(
                "shadow preprocessing is unavailable until a measured failing "
                "stratum artifact and source-image hash are bound to the evidence"
            )
        if not 0 <= int(self.white_threshold) <= 255:
            raise ValueError("white_threshold must be in [0, 255]")
        if self.roi_polygon_px is not None and len(self.roi_polygon_px) < 3:
            raise ValueError("roi_polygon_px must contain at least three points")
        for name, value in (
            ("roi_margin_px", self.roi_margin_px),
            ("max_angle_delta_deg", self.max_angle_delta_deg),
            ("max_normal_distance_px", self.max_normal_distance_px),
            ("min_candidate_margin", self.min_candidate_margin),
            (
                "max_joint_homography_p90_px",
                self.max_joint_homography_p90_px,
            ),
            ("max_static_line_mad_px", self.max_static_line_mad_px),
            ("pool_outlier_mad_scale", self.pool_outlier_mad_scale),
            ("profile_step_px", self.profile_step_px),
            ("profile_scan_radius_px", self.profile_scan_radius_px),
            ("profile_roi_radius_px", self.profile_roi_radius_px),
            ("min_band_contrast", self.min_band_contrast),
            ("min_edge_strength", self.min_edge_strength),
            ("min_edge_symmetry", self.min_edge_symmetry),
            ("paint_width_low_factor", self.paint_width_low_factor),
            ("paint_width_high_factor", self.paint_width_high_factor),
            ("max_line_geometry_p90_px", self.max_line_geometry_p90_px),
            ("heldout_p90_tolerance_px", self.heldout_p90_tolerance_px),
            (
                "heldout_line_family_p90_tolerance_px",
                self.heldout_line_family_p90_tolerance_px,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be in [0, 1]")
        if self.min_assigned_lines < 4:
            raise ValueError("min_assigned_lines must be at least four")
        if self.min_joint_intersections < 6:
            raise ValueError("min_joint_intersections must be at least six")
        if self.max_joint_assignment_hypotheses < 8:
            raise ValueError(
                "max_joint_assignment_hypotheses must be at least eight"
            )
        if self.min_contributing_frames < 3:
            raise ValueError("min_contributing_frames must be at least three")
        if self.min_holdout_contributing_frames < 2:
            raise ValueError("min_holdout_contributing_frames must be at least two")
        if self.frame_holdout_stride < 3:
            raise ValueError("frame_holdout_stride must be at least three")
        if self.profile_sample_count < 9:
            raise ValueError("profile_sample_count must be at least nine")
        if self.profile_step_px > self.profile_scan_radius_px:
            raise ValueError(
                "profile_step_px cannot exceed the profile scan radius"
            )
        if self.min_pooled_line_samples < 5:
            raise ValueError("min_pooled_line_samples must be at least five")
        if self.min_heldout_line_samples < 3:
            raise ValueError("min_heldout_line_samples must be at least three")
        if self.paint_width_low_factor >= self.paint_width_high_factor:
            raise ValueError("paint width low factor must be below the high factor")
        if self.min_edge_symmetry > 1.0:
            raise ValueError("min_edge_symmetry must be at most one")
        if self.line_weight < 0.0 or self.point_weight < 0.0:
            raise ValueError("line and point weights must be non-negative")
        if not math.isclose(self.line_weight + self.point_weight, 1.0, abs_tol=1e-12):
            raise ValueError("line and point weights must sum to one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "provider": self.provider,
            "preprocessing": self.preprocessing,
            "white_threshold": int(self.white_threshold),
            "roi_polygon_px": (
                [[float(value) for value in point] for point in self.roi_polygon_px]
                if self.roi_polygon_px is not None
                else None
            ),
            "roi_margin_px": float(self.roi_margin_px),
            "max_angle_delta_deg": float(self.max_angle_delta_deg),
            "max_normal_distance_px": float(self.max_normal_distance_px),
            "min_overlap_fraction": float(self.min_overlap_fraction),
            "min_candidate_margin": float(self.min_candidate_margin),
            "min_joint_intersections": int(self.min_joint_intersections),
            "max_joint_homography_p90_px": float(
                self.max_joint_homography_p90_px
            ),
            "max_joint_assignment_hypotheses": int(
                self.max_joint_assignment_hypotheses
            ),
            "min_assigned_lines": int(self.min_assigned_lines),
            "min_contributing_frames": int(self.min_contributing_frames),
            "min_holdout_contributing_frames": int(self.min_holdout_contributing_frames),
            "max_static_line_mad_px": float(self.max_static_line_mad_px),
            "pool_outlier_mad_scale": float(self.pool_outlier_mad_scale),
            "frame_holdout_stride": int(self.frame_holdout_stride),
            "profile_sample_count": int(self.profile_sample_count),
            "profile_step_px": float(self.profile_step_px),
            "profile_scan_radius_px": float(self.profile_scan_radius_px),
            "profile_roi_radius_px": float(self.profile_roi_radius_px),
            "min_band_contrast": float(self.min_band_contrast),
            "min_edge_strength": float(self.min_edge_strength),
            "min_edge_symmetry": float(self.min_edge_symmetry),
            "paint_width_low_factor": float(self.paint_width_low_factor),
            "paint_width_high_factor": float(self.paint_width_high_factor),
            "min_pooled_line_samples": int(self.min_pooled_line_samples),
            "min_heldout_line_samples": int(self.min_heldout_line_samples),
            "max_line_geometry_p90_px": float(self.max_line_geometry_p90_px),
            "heldout_p90_tolerance_px": float(self.heldout_p90_tolerance_px),
            "heldout_line_family_p90_tolerance_px": float(
                self.heldout_line_family_p90_tolerance_px
            ),
            "line_weight": float(self.line_weight),
            "point_weight": float(self.point_weight),
            "detector_implementation": DETECTOR_IMPLEMENTATION,
            "pool_implementation": POOL_IMPLEMENTATION,
            "refinement_implementation": REFINEMENT_IMPLEMENTATION,
        }

    def evidence_config_dict(self) -> dict[str, Any]:
        """Detection/pooling identity, intentionally independent of fit weights."""

        payload = self.as_dict()
        for field in (
            "heldout_p90_tolerance_px",
            "heldout_line_family_p90_tolerance_px",
            "line_weight",
            "point_weight",
            "refinement_implementation",
        ):
            payload.pop(field)
        return payload

    def refinement_config_dict(self) -> dict[str, Any]:
        from .court_proposal_optimizer import RefinementConfig

        return {
            "line_weight": float(self.line_weight),
            "point_weight": float(self.point_weight),
            "implementation": REFINEMENT_IMPLEMENTATION,
            "fallback_optimizer_config": asdict(
                RefinementConfig(
                    line_weight=self.line_weight,
                    point_weight=self.point_weight,
                    heldout_p90_tolerance_px=self.heldout_p90_tolerance_px,
                    heldout_max_line_family_p90_regression_px=(
                        self.heldout_line_family_p90_tolerance_px
                    ),
                )
            ),
            "heldout_p90_tolerance_px": float(self.heldout_p90_tolerance_px),
            "heldout_line_family_p90_tolerance_px": float(
                self.heldout_line_family_p90_tolerance_px
            ),
        }


@dataclass(frozen=True)
class AssignedCourtLine:
    line_id: str
    candidate_id: str
    segment: Segment2
    expected_segment: Segment2
    score: float
    normal_distance_px: float
    angle_delta_deg: float
    overlap_fraction: float
    support_length_px: float
    selection_margin: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "candidate_id": self.candidate_id,
            "segment": _segment_payload(self.segment),
            "expected_segment": _segment_payload(self.expected_segment),
            "score": float(self.score),
            "normal_distance_px": float(self.normal_distance_px),
            "angle_delta_deg": float(self.angle_delta_deg),
            "overlap_fraction": float(self.overlap_fraction),
            "support_length_px": float(self.support_length_px),
            "selection_margin": float(self.selection_margin) if self.selection_margin is not None else None,
        }


@dataclass(frozen=True)
class SeedGuidedPaintSample:
    """One immutable paired-edge paint observation from one decoded frame."""

    line_id: str
    sample_index: int
    t: float
    world_xyz_m: tuple[float, float, float]
    seed_xy: Point2
    normal_xy: Point2
    observed_xy: Point2
    signed_offset_px: float
    expected_width_px: float
    band_width_px: float
    contrast: float
    edge_strength: float
    edge_symmetry: float
    selection_rank: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "sample_index": int(self.sample_index),
            "t": float(self.t),
            "world_xyz_m": [float(value) for value in self.world_xyz_m],
            "seed_xy": [float(value) for value in self.seed_xy],
            "normal_xy": [float(value) for value in self.normal_xy],
            "observed_xy": [float(value) for value in self.observed_xy],
            "signed_offset_px": float(self.signed_offset_px),
            "expected_width_px": float(self.expected_width_px),
            "band_width_px": float(self.band_width_px),
            "contrast": float(self.contrast),
            "edge_strength": float(self.edge_strength),
            "edge_symmetry": float(self.edge_symmetry),
            "selection_rank": float(self.selection_rank),
        }


@dataclass(frozen=True)
class FrameCourtLineEvidence:
    frame_index: int
    frame_sha256: str
    image_size: tuple[int, int]
    coordinate_space: str
    distortion_state: str
    provider: str
    raw_candidates: tuple[DetectedCourtLineCandidate, ...]
    assignments: tuple[AssignedCourtLine, ...]
    status: str
    rejection_reasons: tuple[str, ...]
    roi_polygon_px: tuple[Point2, ...]
    roi_source: str
    template_samples: tuple[SeedGuidedPaintSample, ...] = ()
    detector_metadata: Mapping[str, Any] | None = None
    seed_calibration_sha256: str | None = None
    template_projection_sha256: str | None = None
    assignment_model: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "frame_sha256": self.frame_sha256,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "coordinate_space": self.coordinate_space,
            "distortion_state": self.distortion_state,
            "provider": self.provider,
            "raw_candidates": [candidate.as_dict() for candidate in self.raw_candidates],
            "assignments": [assignment.as_dict() for assignment in self.assignments],
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
            "roi_polygon_px": [[float(value) for value in point] for point in self.roi_polygon_px],
            "roi_source": self.roi_source,
            "seed_calibration_sha256": self.seed_calibration_sha256,
            "template_projection_sha256": self.template_projection_sha256,
            "assignment_model": (
                _thaw_json_value(self.assignment_model)
                if self.assignment_model is not None
                else None
            ),
            "template_samples": [sample.as_dict() for sample in self.template_samples],
            "detector_metadata": (
                _thaw_json_value(self.detector_metadata)
                if self.detector_metadata is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PooledPaintSample:
    """Robust temporal median for one regulation-line sample location."""

    line_id: str
    sample_index: int
    t: float
    world_xyz_m: tuple[float, float, float]
    seed_xy: Point2
    normal_xy: Point2
    observed_xy: Point2
    signed_offset_px: float
    contributing_frame_indexes: tuple[int, ...]
    contributing_frame_hashes: tuple[str, ...]
    rejected_frame_indexes: tuple[int, ...]
    temporal_mad_px: float
    median_band_width_px: float
    median_contrast: float
    median_edge_strength: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "sample_index": int(self.sample_index),
            "t": float(self.t),
            "world_xyz_m": [float(value) for value in self.world_xyz_m],
            "seed_xy": [float(value) for value in self.seed_xy],
            "normal_xy": [float(value) for value in self.normal_xy],
            "observed_xy": [float(value) for value in self.observed_xy],
            "signed_offset_px": float(self.signed_offset_px),
            "contributing_frame_indexes": [
                int(value) for value in self.contributing_frame_indexes
            ],
            "contributing_frame_hashes": list(self.contributing_frame_hashes),
            "rejected_frame_indexes": [
                int(value) for value in self.rejected_frame_indexes
            ],
            "temporal_mad_px": float(self.temporal_mad_px),
            "median_band_width_px": float(self.median_band_width_px),
            "median_contrast": float(self.median_contrast),
            "median_edge_strength": float(self.median_edge_strength),
        }


@dataclass(frozen=True)
class PooledSemanticLine:
    line_id: str
    segment: Segment2
    expected_segment: Segment2
    contributing_frame_indexes: tuple[int, ...]
    heldout_frame_indexes: tuple[int, ...]
    rejected_frame_indexes: tuple[int, ...]
    heldout_segments: tuple[Segment2, ...]
    dispersion_mad_px: float
    optimize_samples: tuple[PooledPaintSample, ...] = ()
    heldout_samples: tuple[PooledPaintSample, ...] = ()
    geometry_fit_p90_px: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "source": "pooled_static",
            "segment": _segment_payload(self.segment),
            "expected_segment": _segment_payload(self.expected_segment),
            "contributing_frame_indexes": [int(value) for value in self.contributing_frame_indexes],
            "heldout_frame_indexes": [int(value) for value in self.heldout_frame_indexes],
            "rejected_frame_indexes": [int(value) for value in self.rejected_frame_indexes],
            "heldout_segments": [_segment_payload(segment) for segment in self.heldout_segments],
            "dispersion_mad_px": float(self.dispersion_mad_px),
            "optimize_samples": [sample.as_dict() for sample in self.optimize_samples],
            "heldout_samples": [sample.as_dict() for sample in self.heldout_samples],
            "geometry_fit_p90_px": (
                float(self.geometry_fit_p90_px)
                if self.geometry_fit_p90_px is not None
                else None
            ),
        }


@dataclass(frozen=True)
class StaticConsistencyResult:
    """All-or-nothing guard for the fixed-camera pooling assumption."""

    status: str
    dispersion_bound_px: float
    max_observed_mad_px: float | None
    max_observed_temporal_span_px: float | None
    per_measurement_mad_px: tuple[tuple[str, float], ...]
    per_measurement_temporal_span_px: tuple[tuple[str, float], ...]
    violating_measurements: tuple[str, ...]
    violating_assignment_dropout_spans: tuple[
        tuple[int, int, int, int, int],
        ...,
    ]
    violating_raw_template_dropout_spans: tuple[
        tuple[int, int, int, int, int],
        ...,
    ]
    violating_boundary_degraded_frames: tuple[
        tuple[int, int, int],
        ...,
    ]
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "implementation": STATIC_CONSISTENCY_IMPLEMENTATION,
            "dispersion_bound_px": float(self.dispersion_bound_px),
            "max_observed_mad_px": (
                float(self.max_observed_mad_px)
                if self.max_observed_mad_px is not None
                else None
            ),
            "max_observed_temporal_span_px": (
                float(self.max_observed_temporal_span_px)
                if self.max_observed_temporal_span_px is not None
                else None
            ),
            "per_measurement_mad_px": {
                measurement: float(value)
                for measurement, value in self.per_measurement_mad_px
            },
            "per_measurement_temporal_span_px": {
                measurement: float(value)
                for measurement, value in self.per_measurement_temporal_span_px
            },
            "violating_measurements": list(self.violating_measurements),
            "violating_assignment_dropout_spans": [
                {
                    "start_sample_position": int(start_position),
                    "end_sample_position": int(end_position),
                    "sample_count": int(sample_count),
                    "start_frame_index": int(start_frame_index),
                    "end_frame_index": int(end_frame_index),
                }
                for (
                    start_position,
                    end_position,
                    sample_count,
                    start_frame_index,
                    end_frame_index,
                ) in self.violating_assignment_dropout_spans
            ],
            "violating_raw_template_dropout_spans": [
                {
                    "start_sample_position": int(start_position),
                    "end_sample_position": int(end_position),
                    "sample_count": int(sample_count),
                    "start_frame_index": int(start_frame_index),
                    "end_frame_index": int(end_frame_index),
                }
                for (
                    start_position,
                    end_position,
                    sample_count,
                    start_frame_index,
                    end_frame_index,
                ) in self.violating_raw_template_dropout_spans
            ],
            "violating_boundary_degraded_frames": [
                {
                    "sample_position": int(sample_position),
                    "frame_index": int(frame_index),
                    "usable_line_count": int(usable_line_count),
                }
                for (
                    sample_position,
                    frame_index,
                    usable_line_count,
                ) in self.violating_boundary_degraded_frames
            ],
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class PooledCourtLineEvidence:
    image_size: tuple[int, int]
    coordinate_space: str
    distortion_state: str
    config_sha256: str
    source_frame_hashes: tuple[tuple[int, str], ...]
    lines: tuple[PooledSemanticLine, ...]
    missing_line_ids: tuple[str, ...]
    status: str
    rejection_reasons: tuple[str, ...]
    source_frame_evidence_hashes: tuple[tuple[int, str], ...] = ()
    seed_calibration_sha256: str | None = None
    template_projection_sha256: str | None = None
    static_consistency: StaticConsistencyResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": POOLED_EVIDENCE_ARTIFACT_TYPE,
            "status": self.status,
            "authority": "preview",
            "verified": False,
            "not_cal3_verified": True,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "coordinate_space": self.coordinate_space,
            "distortion_state": self.distortion_state,
            "config_sha256": self.config_sha256,
            "seed_calibration_sha256": self.seed_calibration_sha256,
            "template_projection_sha256": self.template_projection_sha256,
            "source_frame_hashes": [
                {"frame_index": int(index), "sha256": digest}
                for index, digest in self.source_frame_hashes
            ],
            "source_frame_evidence_hashes": [
                {"frame_index": int(index), "sha256": digest}
                for index, digest in self.source_frame_evidence_hashes
            ],
            "provenance": {
                "source": "pooled_static",
                "detector_implementation": DETECTOR_IMPLEMENTATION,
                "pool_implementation": POOL_IMPLEMENTATION,
                "raw_evidence_policy": "referenced_by_hash_never_overwritten",
            },
            "static_consistency": (
                self.static_consistency.as_dict()
                if self.static_consistency is not None
                else None
            ),
            "lines": [line.as_dict() for line in self.lines],
            "missing_line_ids": list(self.missing_line_ids),
            "rejection_reasons": list(self.rejection_reasons),
            "raw_evidence_policy": "referenced_by_hash_never_overwritten",
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


@dataclass(frozen=True)
class CourtLineHardeningResult:
    config: CourtLineHardeningConfig
    seed_calibration_sha256: str
    raw_frame_evidence: tuple[FrameCourtLineEvidence, ...]
    pooled_evidence: PooledCourtLineEvidence
    refinement: Mapping[str, Any]
    candidate_calibration: Mapping[str, Any]

    def raw_evidence_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": RAW_EVIDENCE_ARTIFACT_TYPE,
            "authority": "preview",
            "verified": False,
            "not_cal3_verified": True,
            "seed_calibration_sha256": self.seed_calibration_sha256,
            "evidence_config_sha256": _sha256_payload(
                self.config.evidence_config_dict()
            ),
            "frames": [frame.as_dict() for frame in self.raw_frame_evidence],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "status": "candidate" if bool(self.refinement.get("accepted")) else "abstained",
            "authority": "preview",
            "verified": False,
            "not_cal3_verified": True,
            "config": self.config.as_dict(),
            "config_sha256": _sha256_payload(self.config.as_dict()),
            "seed_calibration_sha256": self.seed_calibration_sha256,
            "raw_frame_evidence_sha256": _sha256_payload(self.raw_evidence_artifact()),
            "pooled_evidence": self.pooled_evidence.as_dict(),
            "refinement": dict(self.refinement),
            "candidate_calibration": dict(self.candidate_calibration),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


@dataclass(frozen=True)
class PooledCourtLineReadiness:
    """Additive effective evidence; baseline observations are never replaced."""

    effective_evidence: CourtLineEvidence
    added_line_observations: tuple[CourtLineObservation, ...]
    status: str
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": "pooled_static",
            "added_line_observations": [
                observation.model_dump(mode="json")
                for observation in self.added_line_observations
            ],
            "effective_aggregate": self.effective_evidence.aggregate.model_dump(
                mode="json"
            ),
            "effective_evidence": self.effective_evidence.model_dump(
                mode="json"
            ),
            "rejection_reasons": list(self.rejection_reasons),
            "acceptance_policy": (
                "threed.racketsport.court_line_evidence defaults; no overrides"
            ),
        }


def canonical_json_bytes(payload: Any) -> bytes:
    """Strict deterministic JSON used for evidence and rebuild checks."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze JSON-like metadata stored inside frozen artifacts."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _longitudinal_family_count(line_ids: set[str]) -> int:
    """Count distinct world-x families; centerline halves are one family."""

    families = set()
    if "left_sideline" in line_ids:
        families.add("left")
    if line_ids.intersection({"near_centerline", "far_centerline"}):
        families.add("center")
    if "right_sideline" in line_ids:
        families.add("right")
    return len(families)


def maybe_apply_court_line_hardening(
    indexed_frames: Sequence[tuple[int, Any]],
    seed_calibration: Mapping[str, Any],
    *,
    config: CourtLineHardeningConfig | None = None,
) -> Mapping[str, Any] | CourtLineHardeningResult:
    """Return the exact seed when off, or a preview wrapper when explicitly on.

    The enabled result is deliberately not a drop-in calibration mapping.  A
    future integration owner must inspect its preview authority and promotion
    state instead of silently replacing a verified seed with candidate bytes.
    """

    if config is None or not config.enabled:
        return seed_calibration
    return run_court_line_hardening(indexed_frames, seed_calibration, config=config)


def proven_court_line_pool_config() -> CourtLineHardeningConfig:
    """Return the exact, untuned configuration exercised by farline_diag."""

    return CourtLineHardeningConfig(
        enabled=True,
        provider=PROVEN_POOL_PROVIDER,
    )


def run_proven_court_line_pool_from_video(
    video_path: str | Path,
    seed_calibration: Mapping[str, Any],
    *,
    cv2_module: Any | None = None,
) -> CourtLineHardeningResult:
    """Decode the diagnostic's exact 96-frame sample and run its exact pool."""

    from .court_auto_evidence import _sample_frame_indexes

    cv2 = cv2_module
    if cv2 is None:
        try:
            import cv2 as imported_cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError("court-line evidence pooling requires opencv-python") from exc
        cv2 = imported_cv2

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video for court-line pooling: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError(
                f"cannot determine frame count for court-line pooling: {path}"
            )
        indexes = _sample_frame_indexes(frame_count, PROVEN_POOL_SAMPLE_COUNT)
        indexed_frames: list[tuple[int, Any]] = []
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"cannot decode court-line pooling frame {frame_index} from {path}"
                )
            indexed_frames.append((int(frame_index), frame))
    finally:
        capture.release()

    return run_court_line_hardening(
        indexed_frames,
        seed_calibration,
        config=proven_court_line_pool_config(),
    )


def run_court_line_hardening(
    indexed_frames: Sequence[tuple[int, Any]],
    seed_calibration: Mapping[str, Any],
    *,
    config: CourtLineHardeningConfig,
) -> CourtLineHardeningResult:
    """Build immutable raw evidence, a separate pool, and a guarded candidate."""

    config.validate()
    if not config.enabled:
        raise ValueError("run_court_line_hardening requires enabled=True")
    if not indexed_frames:
        raise ValueError("at least one indexed frame is required")
    ordered = sorted(indexed_frames, key=lambda item: int(item[0]))
    indexes = [int(index) for index, _image in ordered]
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate frame indexes are not allowed")

    expected = projected_floor_semantic_lines(seed_calibration)
    seed_calibration_sha256 = _sha256_payload(seed_calibration)
    template_projection_sha256 = _sha256_payload(expected)
    image_size = _calibration_image_size(seed_calibration)
    roi, roi_source = _resolve_roi(config, seed_calibration, expected)
    distortion_state = _distortion_state(seed_calibration)
    raw_evidence: list[FrameCourtLineEvidence] = []
    for frame_index, image in ordered:
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            raise ValueError(f"frame {frame_index} is not an image array")
        frame_size = (int(image.shape[1]), int(image.shape[0]))
        if frame_size != image_size:
            raise ValueError(
                f"frame {frame_index} size {frame_size} does not match calibration image size {image_size}"
            )
        if config.provider == "seed_guided_paired_edges":
            evidence = _detect_seed_guided_frame_evidence(
                image,
                seed_calibration,
                expected_lines=expected,
                image_size=image_size,
                frame_index=int(frame_index),
                frame_sha256=_frame_sha256(image),
                config=config,
                roi_polygon_px=roi,
                roi_source=roi_source,
                distortion_state=distortion_state,
                seed_calibration_sha256=seed_calibration_sha256,
                template_projection_sha256=template_projection_sha256,
            )
        else:
            detected = detect_court_line_candidates_from_image(
                image,
                white_threshold=config.white_threshold,
                provider=config.provider,
                seed_calibration=seed_calibration,
                preprocessing=config.preprocessing,
            )
            frozen_before = canonical_json_bytes(detected.as_dict())
            evidence = select_regulation_template_lines(
                detected.candidates,
                expected_lines=expected,
                image_size=image_size,
                frame_index=int(frame_index),
                frame_sha256=_frame_sha256(image),
                config=config,
                roi_polygon_px=roi,
                roi_source=roi_source,
                coordinate_space="pixels_raw_native",
                distortion_state=distortion_state,
                seed_calibration_sha256=seed_calibration_sha256,
                template_projection_sha256=template_projection_sha256,
                seed_calibration=seed_calibration,
            )
            evidence = replace(
                evidence,
                detector_metadata=_freeze_json_value(
                    {
                        "implementation": "court_line_keypoints_additive_candidates_v1",
                        "white_threshold": int(config.white_threshold),
                        "raw_segment_count": int(detected.raw_segment_count),
                        "merged_line_count": int(detected.merged_line_count),
                    }
                ),
            )
            if canonical_json_bytes(detected.as_dict()) != frozen_before:
                raise RuntimeError("raw frame line evidence was mutated during assignment")
        raw_evidence.append(evidence)

    raw_frozen_before = canonical_json_bytes([frame.as_dict() for frame in raw_evidence])
    pooled = pool_static_semantic_lines(raw_evidence, config=config)
    if canonical_json_bytes([frame.as_dict() for frame in raw_evidence]) != raw_frozen_before:
        raise RuntimeError("raw frame line evidence was mutated during pooling")
    refinement, candidate = refine_pooled_homography(
        seed_calibration,
        pooled,
        config=config,
    )
    return CourtLineHardeningResult(
        config=config,
        seed_calibration_sha256=seed_calibration_sha256,
        raw_frame_evidence=tuple(raw_evidence),
        pooled_evidence=pooled,
        refinement=refinement,
        candidate_calibration=candidate,
    )


def load_raw_frame_court_line_evidence_artifact(
    payload: Mapping[str, Any],
    *,
    seed_calibration: Mapping[str, Any],
    config: CourtLineHardeningConfig | None = None,
) -> tuple[FrameCourtLineEvidence, ...]:
    """Rehydrate immutable frame evidence for deterministic fixture replay."""

    resolved_config = config or proven_court_line_pool_config()
    resolved_config.validate()
    if payload.get("artifact_type") != RAW_EVIDENCE_ARTIFACT_TYPE:
        raise ValueError("unexpected raw court-line evidence artifact_type")
    expected_seed_hash = _sha256_payload(seed_calibration)
    if payload.get("seed_calibration_sha256") != expected_seed_hash:
        raise ValueError("raw frame evidence seed calibration hash mismatch")
    expected_config_hash = _sha256_payload(
        resolved_config.evidence_config_dict()
    )
    if payload.get("evidence_config_sha256") != expected_config_hash:
        raise ValueError("raw frame evidence config hash mismatch")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, Sequence) or isinstance(
        raw_frames, (str, bytes, bytearray)
    ):
        raise ValueError("raw frame evidence frames must be a sequence")

    frames: list[FrameCourtLineEvidence] = []
    for frame_position, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise ValueError(f"frames[{frame_position}] must be an object")
        raw_candidates_payload = raw_frame.get("raw_candidates")
        assignments_payload = raw_frame.get("assignments")
        samples_payload = raw_frame.get("template_samples")
        if not isinstance(raw_candidates_payload, Sequence) or isinstance(
            raw_candidates_payload, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"frames[{frame_position}].raw_candidates must be a sequence"
            )
        if not isinstance(assignments_payload, Sequence) or isinstance(
            assignments_payload, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"frames[{frame_position}].assignments must be a sequence"
            )
        if not isinstance(samples_payload, Sequence) or isinstance(
            samples_payload, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"frames[{frame_position}].template_samples must be a sequence"
            )

        raw_candidates: list[DetectedCourtLineCandidate] = []
        for candidate_position, raw_candidate in enumerate(
            raw_candidates_payload
        ):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(
                    f"frames[{frame_position}].raw_candidates"
                    f"[{candidate_position}] must be an object"
                )
            raw_candidates.append(
                DetectedCourtLineCandidate(
                    candidate_id=str(raw_candidate["candidate_id"]),
                    endpoints=_coerce_segment(
                        raw_candidate["endpoints"],
                        f"frames[{frame_position}].raw_candidates"
                        f"[{candidate_position}].endpoints",
                    ),
                    support_length_px=float(
                        raw_candidate["support_length_px"]
                    ),
                    source_segment_count=int(
                        raw_candidate["source_segment_count"]
                    ),
                    angle_deg=float(raw_candidate["angle_deg"]),
                    provider=str(raw_candidate["provider"]),
                    preprocessing=str(
                        raw_candidate.get("preprocessing", "raw")
                    ),
                    source_candidate_ids=tuple(
                        str(value)
                        for value in (
                            raw_candidate.get("source_candidate_ids") or []
                        )
                    ),
                )
            )

        assignments: list[AssignedCourtLine] = []
        for assignment_position, raw_assignment in enumerate(
            assignments_payload
        ):
            if not isinstance(raw_assignment, Mapping):
                raise ValueError(
                    f"frames[{frame_position}].assignments"
                    f"[{assignment_position}] must be an object"
                )
            assignments.append(
                AssignedCourtLine(
                    line_id=str(raw_assignment["line_id"]),
                    candidate_id=str(raw_assignment["candidate_id"]),
                    segment=_coerce_segment(
                        raw_assignment["segment"],
                        f"frames[{frame_position}].assignments"
                        f"[{assignment_position}].segment",
                    ),
                    expected_segment=_coerce_segment(
                        raw_assignment["expected_segment"],
                        f"frames[{frame_position}].assignments"
                        f"[{assignment_position}].expected_segment",
                    ),
                    score=float(raw_assignment["score"]),
                    normal_distance_px=float(
                        raw_assignment["normal_distance_px"]
                    ),
                    angle_delta_deg=float(
                        raw_assignment["angle_delta_deg"]
                    ),
                    overlap_fraction=float(
                        raw_assignment["overlap_fraction"]
                    ),
                    support_length_px=float(
                        raw_assignment["support_length_px"]
                    ),
                    selection_margin=(
                        float(raw_assignment["selection_margin"])
                        if raw_assignment.get("selection_margin") is not None
                        else None
                    ),
                )
            )

        samples: list[SeedGuidedPaintSample] = []
        for sample_position, raw_sample in enumerate(samples_payload):
            if not isinstance(raw_sample, Mapping):
                raise ValueError(
                    f"frames[{frame_position}].template_samples"
                    f"[{sample_position}] must be an object"
                )
            samples.append(
                SeedGuidedPaintSample(
                    line_id=str(raw_sample["line_id"]),
                    sample_index=int(raw_sample["sample_index"]),
                    t=float(raw_sample["t"]),
                    world_xyz_m=_coerce_point3(
                        raw_sample["world_xyz_m"],
                        f"frames[{frame_position}].template_samples"
                        f"[{sample_position}].world_xyz_m",
                    ),
                    seed_xy=_coerce_point2(
                        raw_sample["seed_xy"],
                        f"frames[{frame_position}].template_samples"
                        f"[{sample_position}].seed_xy",
                    ),
                    normal_xy=_coerce_point2(
                        raw_sample["normal_xy"],
                        f"frames[{frame_position}].template_samples"
                        f"[{sample_position}].normal_xy",
                    ),
                    observed_xy=_coerce_point2(
                        raw_sample["observed_xy"],
                        f"frames[{frame_position}].template_samples"
                        f"[{sample_position}].observed_xy",
                    ),
                    signed_offset_px=float(
                        raw_sample["signed_offset_px"]
                    ),
                    expected_width_px=float(
                        raw_sample["expected_width_px"]
                    ),
                    band_width_px=float(raw_sample["band_width_px"]),
                    contrast=float(raw_sample["contrast"]),
                    edge_strength=float(raw_sample["edge_strength"]),
                    edge_symmetry=float(raw_sample["edge_symmetry"]),
                    selection_rank=float(raw_sample["selection_rank"]),
                )
            )

        image_size_payload = raw_frame.get("image_size")
        if (
            not isinstance(image_size_payload, Sequence)
            or isinstance(image_size_payload, (str, bytes, bytearray))
            or len(image_size_payload) != 2
        ):
            raise ValueError(
                f"frames[{frame_position}].image_size must contain two values"
            )
        roi_payload = raw_frame.get("roi_polygon_px")
        if not isinstance(roi_payload, Sequence) or isinstance(
            roi_payload, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"frames[{frame_position}].roi_polygon_px must be a sequence"
            )
        frames.append(
            FrameCourtLineEvidence(
                frame_index=int(raw_frame["frame_index"]),
                frame_sha256=str(raw_frame["frame_sha256"]),
                image_size=(
                    int(image_size_payload[0]),
                    int(image_size_payload[1]),
                ),
                coordinate_space=str(raw_frame["coordinate_space"]),
                distortion_state=str(raw_frame["distortion_state"]),
                provider=str(raw_frame["provider"]),
                raw_candidates=tuple(raw_candidates),
                assignments=tuple(assignments),
                status=str(raw_frame["status"]),
                rejection_reasons=tuple(
                    str(value)
                    for value in (raw_frame.get("rejection_reasons") or [])
                ),
                roi_polygon_px=tuple(
                    _coerce_point2(
                        point,
                        f"frames[{frame_position}].roi_polygon_px",
                    )
                    for point in roi_payload
                ),
                roi_source=str(raw_frame["roi_source"]),
                template_samples=tuple(samples),
                detector_metadata=(
                    _freeze_json_value(raw_frame["detector_metadata"])
                    if raw_frame.get("detector_metadata") is not None
                    else None
                ),
                seed_calibration_sha256=(
                    str(raw_frame["seed_calibration_sha256"])
                    if raw_frame.get("seed_calibration_sha256") is not None
                    else None
                ),
                template_projection_sha256=(
                    str(raw_frame["template_projection_sha256"])
                    if raw_frame.get("template_projection_sha256") is not None
                    else None
                ),
                assignment_model=(
                    _freeze_json_value(raw_frame["assignment_model"])
                    if raw_frame.get("assignment_model") is not None
                    else None
                ),
            )
        )
    return tuple(frames)


def load_pooled_court_line_evidence_artifact(
    payload: Mapping[str, Any],
) -> PooledCourtLineEvidence:
    """Rehydrate the serialized pool before recomputing readiness."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected pooled court-line evidence schema_version")
    if payload.get("artifact_type") != POOLED_EVIDENCE_ARTIFACT_TYPE:
        raise ValueError("unexpected pooled court-line evidence artifact_type")
    image_size_payload = payload.get("image_size")
    if (
        not isinstance(image_size_payload, Sequence)
        or isinstance(image_size_payload, (str, bytes, bytearray))
        or len(image_size_payload) != 2
    ):
        raise ValueError("pooled evidence image_size must contain two values")

    def _frame_hash_rows(field: str) -> tuple[tuple[int, str], ...]:
        rows = payload.get(field)
        if not isinstance(rows, Sequence) or isinstance(
            rows, (str, bytes, bytearray)
        ):
            raise ValueError(f"pooled evidence {field} must be a sequence")
        parsed: list[tuple[int, str]] = []
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"pooled evidence {field}[{position}] must be an object"
                )
            parsed.append((int(row["frame_index"]), str(row["sha256"])))
        return tuple(parsed)

    def _pooled_sample(
        raw: Mapping[str, Any],
        field: str,
    ) -> PooledPaintSample:
        return PooledPaintSample(
            line_id=str(raw["line_id"]),
            sample_index=int(raw["sample_index"]),
            t=float(raw["t"]),
            world_xyz_m=_coerce_point3(raw["world_xyz_m"], f"{field}.world_xyz_m"),
            seed_xy=_coerce_point2(raw["seed_xy"], f"{field}.seed_xy"),
            normal_xy=_coerce_point2(raw["normal_xy"], f"{field}.normal_xy"),
            observed_xy=_coerce_point2(
                raw["observed_xy"],
                f"{field}.observed_xy",
            ),
            signed_offset_px=float(raw["signed_offset_px"]),
            contributing_frame_indexes=tuple(
                int(value)
                for value in raw.get("contributing_frame_indexes", [])
            ),
            contributing_frame_hashes=tuple(
                str(value)
                for value in raw.get("contributing_frame_hashes", [])
            ),
            rejected_frame_indexes=tuple(
                int(value)
                for value in raw.get("rejected_frame_indexes", [])
            ),
            temporal_mad_px=float(raw["temporal_mad_px"]),
            median_band_width_px=float(raw["median_band_width_px"]),
            median_contrast=float(raw["median_contrast"]),
            median_edge_strength=float(raw["median_edge_strength"]),
        )

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, Sequence) or isinstance(
        raw_lines, (str, bytes, bytearray)
    ):
        raise ValueError("pooled evidence lines must be a sequence")
    lines: list[PooledSemanticLine] = []
    for line_position, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, Mapping):
            raise ValueError(
                f"pooled evidence lines[{line_position}] must be an object"
            )
        if raw_line.get("source") != "pooled_static":
            raise ValueError(
                f"pooled evidence lines[{line_position}] source must be pooled_static"
            )
        optimize_payload = raw_line.get("optimize_samples") or []
        heldout_payload = raw_line.get("heldout_samples") or []
        heldout_segments_payload = raw_line.get("heldout_segments") or []
        for field_name, field_payload in (
            ("optimize_samples", optimize_payload),
            ("heldout_samples", heldout_payload),
            ("heldout_segments", heldout_segments_payload),
        ):
            if not isinstance(field_payload, Sequence) or isinstance(
                field_payload, (str, bytes, bytearray)
            ):
                raise ValueError(
                    f"pooled evidence lines[{line_position}].{field_name} "
                    "must be a sequence"
                )
        optimize_samples = tuple(
            _pooled_sample(
                raw_sample,
                f"lines[{line_position}].optimize_samples[{sample_position}]",
            )
            for sample_position, raw_sample in enumerate(optimize_payload)
            if isinstance(raw_sample, Mapping)
        )
        heldout_samples = tuple(
            _pooled_sample(
                raw_sample,
                f"lines[{line_position}].heldout_samples[{sample_position}]",
            )
            for sample_position, raw_sample in enumerate(heldout_payload)
            if isinstance(raw_sample, Mapping)
        )
        if len(optimize_samples) != len(optimize_payload) or len(
            heldout_samples
        ) != len(heldout_payload):
            raise ValueError(
                f"pooled evidence lines[{line_position}] samples must be objects"
            )
        geometry_fit_p90 = raw_line.get("geometry_fit_p90_px")
        lines.append(
            PooledSemanticLine(
                line_id=str(raw_line["line_id"]),
                segment=_coerce_segment(
                    raw_line["segment"],
                    f"lines[{line_position}].segment",
                ),
                expected_segment=_coerce_segment(
                    raw_line["expected_segment"],
                    f"lines[{line_position}].expected_segment",
                ),
                contributing_frame_indexes=tuple(
                    int(value)
                    for value in raw_line.get(
                        "contributing_frame_indexes", []
                    )
                ),
                heldout_frame_indexes=tuple(
                    int(value)
                    for value in raw_line.get("heldout_frame_indexes", [])
                ),
                rejected_frame_indexes=tuple(
                    int(value)
                    for value in raw_line.get("rejected_frame_indexes", [])
                ),
                heldout_segments=tuple(
                    _coerce_segment(
                        segment,
                        f"lines[{line_position}].heldout_segments",
                    )
                    for segment in heldout_segments_payload
                ),
                dispersion_mad_px=float(raw_line["dispersion_mad_px"]),
                optimize_samples=optimize_samples,
                heldout_samples=heldout_samples,
                geometry_fit_p90_px=(
                    float(geometry_fit_p90)
                    if geometry_fit_p90 is not None
                    else None
                ),
            )
        )

    raw_static = payload.get("static_consistency")
    if not isinstance(raw_static, Mapping):
        raise ValueError("pooled evidence static_consistency must be an object")
    if raw_static.get("implementation") != STATIC_CONSISTENCY_IMPLEMENTATION:
        raise ValueError(
            "pooled evidence static consistency implementation mismatch"
        )
    raw_mads = raw_static.get("per_measurement_mad_px")
    raw_spans = raw_static.get("per_measurement_temporal_span_px")
    raw_dropouts = raw_static.get("violating_assignment_dropout_spans")
    raw_template_dropouts = raw_static.get(
        "violating_raw_template_dropout_spans"
    )
    raw_degraded = raw_static.get("violating_boundary_degraded_frames")
    if not isinstance(raw_mads, Mapping) or not isinstance(raw_spans, Mapping):
        raise ValueError("pooled evidence static measurements must be objects")
    if not isinstance(raw_dropouts, Sequence) or isinstance(
        raw_dropouts, (str, bytes, bytearray)
    ):
        raise ValueError(
            "pooled evidence static dropout spans must be a sequence"
        )
    if not isinstance(raw_template_dropouts, Sequence) or isinstance(
        raw_template_dropouts,
        (str, bytes, bytearray),
    ):
        raise ValueError(
            "pooled evidence static raw-template dropout spans must be a "
            "sequence"
        )
    if not isinstance(raw_degraded, Sequence) or isinstance(
        raw_degraded, (str, bytes, bytearray)
    ):
        raise ValueError(
            "pooled evidence static degraded frames must be a sequence"
        )
    dropout_spans: list[tuple[int, int, int, int, int]] = []
    for position, raw_span in enumerate(raw_dropouts):
        if not isinstance(raw_span, Mapping):
            raise ValueError(
                "pooled evidence static dropout span "
                f"{position} must be an object"
            )
        dropout_spans.append(
            (
                int(raw_span["start_sample_position"]),
                int(raw_span["end_sample_position"]),
                int(raw_span["sample_count"]),
                int(raw_span["start_frame_index"]),
                int(raw_span["end_frame_index"]),
            )
        )
    raw_template_dropout_spans: list[
        tuple[int, int, int, int, int]
    ] = []
    for position, raw_span in enumerate(raw_template_dropouts):
        if not isinstance(raw_span, Mapping):
            raise ValueError(
                "pooled evidence static raw-template dropout span "
                f"{position} must be an object"
            )
        raw_template_dropout_spans.append(
            (
                int(raw_span["start_sample_position"]),
                int(raw_span["end_sample_position"]),
                int(raw_span["sample_count"]),
                int(raw_span["start_frame_index"]),
                int(raw_span["end_frame_index"]),
            )
        )
    degraded_frames: list[tuple[int, int, int]] = []
    for position, raw_frame in enumerate(raw_degraded):
        if not isinstance(raw_frame, Mapping):
            raise ValueError(
                "pooled evidence static degraded frame "
                f"{position} must be an object"
            )
        degraded_frames.append(
            (
                int(raw_frame["sample_position"]),
                int(raw_frame["frame_index"]),
                int(raw_frame["usable_line_count"]),
            )
        )
    max_mad = raw_static.get("max_observed_mad_px")
    max_span = raw_static.get("max_observed_temporal_span_px")
    static_consistency = StaticConsistencyResult(
        status=str(raw_static["status"]),
        dispersion_bound_px=float(raw_static["dispersion_bound_px"]),
        max_observed_mad_px=(
            float(max_mad) if max_mad is not None else None
        ),
        max_observed_temporal_span_px=(
            float(max_span) if max_span is not None else None
        ),
        per_measurement_mad_px=tuple(
            sorted((str(key), float(value)) for key, value in raw_mads.items())
        ),
        per_measurement_temporal_span_px=tuple(
            sorted((str(key), float(value)) for key, value in raw_spans.items())
        ),
        violating_measurements=tuple(
            str(value)
            for value in raw_static.get("violating_measurements", [])
        ),
        violating_assignment_dropout_spans=tuple(dropout_spans),
        violating_raw_template_dropout_spans=tuple(
            raw_template_dropout_spans
        ),
        violating_boundary_degraded_frames=tuple(degraded_frames),
        rejection_reasons=tuple(
            str(value)
            for value in raw_static.get("rejection_reasons", [])
        ),
    )
    return PooledCourtLineEvidence(
        image_size=(
            int(image_size_payload[0]),
            int(image_size_payload[1]),
        ),
        coordinate_space=str(payload["coordinate_space"]),
        distortion_state=str(payload["distortion_state"]),
        config_sha256=str(payload["config_sha256"]),
        source_frame_hashes=_frame_hash_rows("source_frame_hashes"),
        lines=tuple(lines),
        missing_line_ids=tuple(
            str(value) for value in payload.get("missing_line_ids", [])
        ),
        status=str(payload["status"]),
        rejection_reasons=tuple(
            str(value) for value in payload.get("rejection_reasons", [])
        ),
        source_frame_evidence_hashes=_frame_hash_rows(
            "source_frame_evidence_hashes"
        ),
        seed_calibration_sha256=(
            str(payload["seed_calibration_sha256"])
            if payload.get("seed_calibration_sha256") is not None
            else None
        ),
        template_projection_sha256=(
            str(payload["template_projection_sha256"])
            if payload.get("template_projection_sha256") is not None
            else None
        ),
        static_consistency=static_consistency,
    )


def combine_pooled_static_court_line_evidence(
    baseline_evidence: CourtLineEvidence | Mapping[str, Any],
    pooled_evidence: PooledCourtLineEvidence,
    *,
    seed_calibration: Mapping[str, Any],
    config: CourtLineHardeningConfig | None = None,
) -> PooledCourtLineReadiness:
    """Add only missing, selector-accepted pooled lines at the frozen bar."""

    resolved_config = config or proven_court_line_pool_config()
    resolved_config.validate()
    baseline = (
        baseline_evidence.model_copy(deep=True)
        if isinstance(baseline_evidence, CourtLineEvidence)
        else CourtLineEvidence.model_validate(baseline_evidence)
    )
    expected_config_hash = _sha256_payload(
        resolved_config.evidence_config_dict()
    )
    if pooled_evidence.config_sha256 != expected_config_hash:
        raise ValueError("pooled evidence config hash mismatch")
    expected_seed_hash = _sha256_payload(seed_calibration)
    if pooled_evidence.seed_calibration_sha256 != expected_seed_hash:
        raise ValueError("pooled evidence seed calibration hash mismatch")
    expected_template_hash = _sha256_payload(
        projected_floor_semantic_lines(seed_calibration)
    )
    if pooled_evidence.template_projection_sha256 != expected_template_hash:
        raise ValueError("pooled evidence template projection hash mismatch")

    if (
        pooled_evidence.status != "accepted"
        or pooled_evidence.static_consistency is None
        or pooled_evidence.static_consistency.status != "accepted"
    ):
        reasons = tuple(pooled_evidence.rejection_reasons)
        if pooled_evidence.static_consistency is None:
            reasons = (*reasons, "static_consistency_missing")
        return PooledCourtLineReadiness(
            effective_evidence=baseline,
            added_line_observations=(),
            status="abstained",
            rejection_reasons=tuple(sorted(set(reasons))),
        )

    required_lines = required_court_line_ids(baseline.sport)
    required_nets = required_court_net_ids(baseline.sport)
    already_accepted = set(baseline.aggregate.accepted_line_ids)
    pooled_by_id = {line.line_id: line for line in pooled_evidence.lines}
    additions: list[CourtLineObservation] = []
    rejection_reasons: list[str] = []
    for line_id in required_lines:
        if line_id in already_accepted:
            continue
        pooled_line = pooled_by_id.get(line_id)
        if pooled_line is None:
            rejection_reasons.append(f"pooled_required_line_missing:{line_id}")
            continue
        support_indexes = sorted(
            set(pooled_line.contributing_frame_indexes)
            | set(pooled_line.heldout_frame_indexes)
        )
        observation = select_best_line_observation(
            line_id=line_id,
            expected_segment=pooled_line.expected_segment,
            candidate_segments=[pooled_line.segment],
            frame_indexes=support_indexes,
            source="pooled_static",
        )
        if observation is None:
            rejection_reasons.append(
                f"pooled_line_failed_existing_acceptance_bar:{line_id}"
            )
            continue
        additions.append(observation)

    effective = aggregate_court_line_evidence(
        sport=baseline.sport,
        line_observations=[*baseline.line_observations, *additions],
        keypoint_observations=baseline.keypoint_observations,
        net_observations=baseline.net_observations,
        required_line_ids=required_lines,
        required_net_ids=required_nets,
    )
    effective.source = f"{baseline.source}_plus_pooled_static"
    if not effective.aggregate.auto_calibration_ready:
        rejection_reasons.extend(effective.aggregate.reasons)
    return PooledCourtLineReadiness(
        effective_evidence=effective,
        added_line_observations=tuple(additions),
        status=(
            "accepted"
            if effective.aggregate.auto_calibration_ready
            else "not_ready"
        ),
        rejection_reasons=tuple(sorted(set(rejection_reasons))),
    )


def projected_floor_semantic_lines(calibration: Mapping[str, Any]) -> dict[str, Segment2]:
    """Project the exact regulation floor-line model into raw native pixels."""

    template = get_court_template("pickleball")
    result: dict[str, Segment2] = {}
    for line_id in FLOOR_LINE_IDS:
        world_segment = template.line_segments_m[line_id]
        projected = _project_raw(calibration, world_segment)
        result[line_id] = (
            (float(projected[0][0]), float(projected[0][1])),
            (float(projected[1][0]), float(projected[1][1])),
        )
    return result


def _detect_seed_guided_frame_evidence(
    image: Any,
    calibration: Mapping[str, Any],
    *,
    expected_lines: Mapping[str, Segment2],
    image_size: tuple[int, int],
    frame_index: int,
    frame_sha256: str,
    config: CourtLineHardeningConfig,
    roi_polygon_px: tuple[Point2, ...],
    roi_source: str,
    distortion_state: str,
    seed_calibration_sha256: str,
    template_projection_sha256: str,
) -> FrameCourtLineEvidence:
    """Scan only regulation-template ROIs for finite-width paint bands."""

    import cv2

    working = image
    gray = (
        cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        if len(working.shape) == 3
        else np.asarray(working)
    )
    gray = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), 0.65)
    height, width = gray.shape[:2]
    offsets = np.arange(
        -config.profile_scan_radius_px,
        config.profile_scan_radius_px + config.profile_step_px * 0.5,
        config.profile_step_px,
        dtype=np.float64,
    )
    template = get_court_template("pickleball")
    samples: list[SeedGuidedPaintSample] = []
    candidates: list[DetectedCourtLineCandidate] = []
    per_line_counts: dict[str, int] = {}
    insufficient: list[str] = []
    for line_id in FLOOR_LINE_IDS:
        geometry = _seed_line_geometry(
            calibration,
            template.line_segments_m[line_id],
            sample_count=config.profile_sample_count,
        )
        bases = geometry["pixels"]
        normals = geometry["normals"]
        map_x = (
            bases[:, 0:1] + normals[:, 0:1] * offsets[None, :]
        ).astype(np.float32)
        map_y = (
            bases[:, 1:2] + normals[:, 1:2] * offsets[None, :]
        ).astype(np.float32)
        profiles = cv2.remap(
            gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        line_samples: list[SeedGuidedPaintSample] = []
        for sample_index, (base, normal, expected_width, profile) in enumerate(
            zip(
                bases,
                normals,
                geometry["widths"],
                profiles,
                strict=True,
            )
        ):
            if not (
                2.0 <= float(base[0]) < width - 2.0
                and 2.0 <= float(base[1]) < height - 2.0
            ):
                continue
            detected = _detect_paired_edge_profile(
                profile,
                offsets,
                expected_width=float(expected_width),
                config=config,
            )
            if detected is None:
                continue
            observed = base + detected["signed_offset_px"] * normal
            observed_xy = (float(observed[0]), float(observed[1]))
            if not _point_in_or_near_polygon(
                observed_xy,
                roi_polygon_px,
                margin=config.roi_margin_px,
            ):
                continue
            world = geometry["world"][sample_index]
            line_samples.append(
                SeedGuidedPaintSample(
                    line_id=line_id,
                    sample_index=sample_index,
                    t=float(geometry["t"][sample_index]),
                    world_xyz_m=(
                        float(world[0]),
                        float(world[1]),
                        float(world[2]),
                    ),
                    seed_xy=(float(base[0]), float(base[1])),
                    normal_xy=(float(normal[0]), float(normal[1])),
                    observed_xy=observed_xy,
                    signed_offset_px=float(detected["signed_offset_px"]),
                    expected_width_px=float(expected_width),
                    band_width_px=float(detected["band_width_px"]),
                    contrast=float(detected["contrast"]),
                    edge_strength=float(detected["edge_strength"]),
                    edge_symmetry=float(detected["edge_symmetry"]),
                    selection_rank=float(detected["selection_rank"]),
                )
            )
        line_samples.sort(key=lambda item: item.sample_index)
        per_line_counts[line_id] = len(line_samples)
        samples.extend(line_samples)
        if len(line_samples) < 5:
            insufficient.append(
                f"{line_id}:paired_edge_samples_below_frame_minimum:{len(line_samples)}<5"
            )
            continue
        candidates.append(
            _candidate_from_seed_guided_samples(
                line_id,
                line_samples,
                preprocessing=config.preprocessing,
            )
        )

    evidence = select_regulation_template_lines(
        candidates,
        expected_lines=expected_lines,
        image_size=image_size,
        frame_index=frame_index,
        frame_sha256=frame_sha256,
        config=config,
        roi_polygon_px=roi_polygon_px,
        roi_source=roi_source,
        coordinate_space="pixels_raw_native",
        distortion_state=distortion_state,
        seed_calibration_sha256=seed_calibration_sha256,
        template_projection_sha256=template_projection_sha256,
        seed_calibration=calibration,
    )
    return replace(
        evidence,
        rejection_reasons=tuple(
            sorted(set((*evidence.rejection_reasons, *insufficient)))
        ),
        # Raw detections are immutable observations.  Assignment and pooling
        # select from them later; rejected-line samples remain in this record.
        template_samples=tuple(samples),
        detector_metadata=_freeze_json_value(
            {
                "implementation": DETECTOR_IMPLEMENTATION,
                "preprocessing": config.preprocessing,
                "profile_sample_count_per_line": int(config.profile_sample_count),
                "profile_step_px": float(config.profile_step_px),
                "profile_scan_radius_px": float(config.profile_scan_radius_px),
                "profile_roi_radius_px": float(config.profile_roi_radius_px),
                "min_band_contrast": float(config.min_band_contrast),
                "min_edge_strength": float(config.min_edge_strength),
                "min_edge_symmetry": float(config.min_edge_symmetry),
                "paint_width_low_factor": float(config.paint_width_low_factor),
                "paint_width_high_factor": float(config.paint_width_high_factor),
                "per_line_detected_sample_count": dict(sorted(per_line_counts.items())),
                "raw_sample_count": len(samples),
                "candidate_line_count": len(candidates),
            }
        ),
    )


def _seed_line_geometry(
    calibration: Mapping[str, Any],
    endpoints: Sequence[Sequence[float]],
    *,
    sample_count: int,
) -> dict[str, np.ndarray]:
    first = np.asarray(endpoints[0], dtype=np.float64)
    second = np.asarray(endpoints[1], dtype=np.float64)
    t = np.linspace(0.04, 0.96, sample_count, dtype=np.float64)
    world = first[None, :] + t[:, None] * (second - first)[None, :]
    pixels = _project_raw(calibration, world)

    tangent_world = second - first
    tangent_world /= max(float(np.linalg.norm(tangent_world[:2])), 1e-12)
    epsilon_m = 0.01
    before = _project_raw(calibration, world - epsilon_m * tangent_world)
    after = _project_raw(calibration, world + epsilon_m * tangent_world)
    tangents = after - before
    tangents /= np.maximum(
        np.linalg.norm(tangents, axis=1, keepdims=True),
        1e-12,
    )
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    direction_xy = (second - first)[:2]
    direction_xy /= max(float(np.linalg.norm(direction_xy)), 1e-12)
    world_normal = np.asarray(
        [-direction_xy[1], direction_xy[0], 0.0],
        dtype=np.float64,
    )
    edge_a = _project_raw(
        calibration,
        world - (PAINT_WIDTH_M * 0.5) * world_normal,
    )
    edge_b = _project_raw(
        calibration,
        world + (PAINT_WIDTH_M * 0.5) * world_normal,
    )
    projected_normal = edge_b - edge_a
    flip = np.sum(normals * projected_normal, axis=1) < 0.0
    normals[flip] *= -1.0
    widths = np.linalg.norm(projected_normal, axis=1)
    return {
        "world": world,
        "t": t,
        "pixels": pixels,
        "normals": normals,
        "widths": widths,
    }


def _detect_paired_edge_profile(
    raw_profile: Any,
    offsets: np.ndarray,
    *,
    expected_width: float,
    config: CourtLineHardeningConfig,
) -> dict[str, float] | None:
    profile = np.asarray(raw_profile, dtype=np.float64)
    finite = np.isfinite(profile)
    if not finite.all():
        center_index = int(np.argmin(np.abs(offsets)))
        if not bool(finite[center_index]):
            return None
        first = center_index
        last = center_index
        while first > 0 and bool(finite[first - 1]):
            first -= 1
        while last + 1 < len(finite) and bool(finite[last + 1]):
            last += 1
        profile = profile[first : last + 1]
        offsets = offsets[first : last + 1]
    if len(profile) < 9:
        return None
    maximum_width = min(
        20.0,
        max(3.0, expected_width * config.paint_width_high_factor),
    )
    minimum_two_sided_context = max(
        4.0,
        0.5 * maximum_width + 3.0,
    )
    if (
        float(offsets[0]) > -minimum_two_sided_context
        or float(offsets[-1]) < minimum_two_sided_context
    ):
        return None
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    smooth = np.convolve(profile, kernel, mode="same")
    step = float(offsets[1] - offsets[0])
    gradient = np.gradient(smooth, step)
    positive = _local_peak_indexes(gradient, minimum=config.min_edge_strength)
    negative = _local_peak_indexes(-gradient, minimum=config.min_edge_strength)
    minimum_width = max(1.0, expected_width * config.paint_width_low_factor)
    rows: list[dict[str, float]] = []
    for left_index in positive:
        left_offset, left_strength = _parabolic_peak(
            offsets,
            gradient,
            left_index,
        )
        for right_index in negative:
            if right_index <= left_index:
                continue
            right_offset, right_strength = _parabolic_peak(
                offsets,
                -gradient,
                right_index,
            )
            band_width = right_offset - left_offset
            if not minimum_width <= band_width <= maximum_width:
                continue
            if (
                left_offset - 3.0 < float(offsets[0]) + 2.0 * step
                or right_offset + 3.0
                > float(offsets[-1]) - 2.0 * step
            ):
                continue
            center = 0.5 * (left_offset + right_offset)
            if abs(center) > config.profile_roi_radius_px:
                continue
            interior = profile[
                (offsets >= left_offset + 0.35)
                & (offsets <= right_offset - 0.35)
            ]
            outside = profile[
                (
                    (offsets >= left_offset - 3.0)
                    & (offsets <= left_offset - 0.5)
                )
                | (
                    (offsets >= right_offset + 0.5)
                    & (offsets <= right_offset + 3.0)
                )
            ]
            if interior.size < 2 or outside.size < 2:
                continue
            contrast = float(np.median(interior) - np.median(outside))
            if contrast < config.min_band_contrast:
                continue
            edge_strength = min(left_strength, right_strength)
            symmetry = edge_strength / max(left_strength, right_strength, 1e-9)
            if symmetry < config.min_edge_symmetry:
                continue
            width_log_error = abs(
                math.log(
                    max(band_width, 1e-6) / max(expected_width, 1e-6)
                )
            )
            rank = (
                abs(center)
                + 1.75 * width_log_error
                - 0.012 * edge_strength
                - 0.006 * contrast
            )
            rows.append(
                {
                    "signed_offset_px": float(center),
                    "band_width_px": float(band_width),
                    "contrast": contrast,
                    "edge_strength": float(edge_strength),
                    "edge_symmetry": float(symmetry),
                    "selection_rank": float(rank),
                }
            )
    if not rows:
        return None
    return min(
        rows,
        key=lambda item: (
            item["selection_rank"],
            abs(item["signed_offset_px"]),
        ),
    )


def _local_peak_indexes(values: np.ndarray, *, minimum: float) -> list[int]:
    if len(values) < 3:
        return []
    indexes = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= minimum)
    )
    return [int(index + 1) for index in indexes]


def _parabolic_peak(
    offsets: np.ndarray,
    values: np.ndarray,
    index: int,
) -> tuple[float, float]:
    left, center, right = [
        float(value) for value in values[index - 1 : index + 2]
    ]
    denominator = left - 2.0 * center + right
    delta = (
        0.0
        if abs(denominator) <= 1e-12
        else float(
            np.clip(
                0.5 * (left - right) / denominator,
                -1.0,
                1.0,
            )
        )
    )
    peak = center - 0.25 * (left - right) * delta
    return (
        float(offsets[index] + delta * (offsets[1] - offsets[0])),
        float(peak),
    )


def _candidate_from_seed_guided_samples(
    line_id: str,
    samples: Sequence[SeedGuidedPaintSample],
    *,
    preprocessing: str,
) -> DetectedCourtLineCandidate:
    points = np.asarray([sample.observed_xy for sample in samples], dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, -1]
    expected_direction = (
        np.asarray(samples[-1].seed_xy) - np.asarray(samples[0].seed_xy)
    )
    if float(direction @ expected_direction) < 0.0:
        direction = -direction
    positions = centered @ direction
    first = center + float(np.min(positions)) * direction
    second = center + float(np.max(positions)) * direction
    segment: Segment2 = (
        (float(first[0]), float(first[1])),
        (float(second[0]), float(second[1])),
    )
    candidate_id = f"seed_guided_paired_edges:{line_id}"
    return DetectedCourtLineCandidate(
        candidate_id=candidate_id,
        endpoints=segment,
        support_length_px=_segment_length(segment),
        source_segment_count=len(samples),
        angle_deg=_segment_angle_deg(segment),
        provider="seed_guided_paired_edges",
        preprocessing=preprocessing,
        source_candidate_ids=(candidate_id,),
    )


def select_regulation_template_lines(
    candidates: Sequence[DetectedCourtLineCandidate | Mapping[str, Any]],
    *,
    expected_lines: Mapping[str, Sequence[Sequence[float]]],
    image_size: tuple[int, int],
    frame_index: int,
    frame_sha256: str,
    config: CourtLineHardeningConfig,
    roi_polygon_px: Sequence[Sequence[float]],
    roi_source: str,
    coordinate_space: str = "pixels_raw_native",
    distortion_state: str = "unknown",
    seed_calibration_sha256: str | None = None,
    template_projection_sha256: str | None = None,
    seed_calibration: Mapping[str, Any] | None = None,
) -> FrameCourtLineEvidence:
    """Assign candidates to exact template lines and abstain on ambiguity."""

    config.validate()
    normalized = tuple(_coerce_candidate(candidate, index) for index, candidate in enumerate(candidates))
    expected = {
        line_id: _coerce_segment(expected_lines[line_id], f"expected_lines.{line_id}")
        for line_id in FLOOR_LINE_IDS
        if line_id in expected_lines
    }
    roi = tuple((float(point[0]), float(point[1])) for point in roi_polygon_px)
    if len(roi) < 3:
        raise ValueError("a declared or seed-derived ROI polygon is required")
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: dict[str, dict[str, int]] = {}
    for line_id, expected_segment in sorted(expected.items()):
        rows: list[dict[str, Any]] = []
        counts = {
            "outside_roi": 0,
            "angle": 0,
            "distance": 0,
            "overlap": 0,
            "template_identity": 0,
        }
        for candidate in normalized:
            if (
                candidate.provider == "seed_guided_paired_edges"
                and candidate.candidate_id.rsplit(":", 1)[-1] != line_id
            ):
                counts["template_identity"] += 1
                continue
            roi_fraction = _segment_roi_support_fraction(
                candidate.endpoints,
                roi,
                margin=config.roi_margin_px,
            )
            if roi_fraction < max(0.25, config.min_overlap_fraction):
                counts["outside_roi"] += 1
                continue
            angle_delta = _axial_angle_delta(candidate.endpoints, expected_segment)
            if angle_delta > config.max_angle_delta_deg:
                counts["angle"] += 1
                continue
            distance = _symmetric_line_distance(candidate.endpoints, expected_segment)
            if distance > config.max_normal_distance_px:
                counts["distance"] += 1
                continue
            overlap = _segment_overlap_fraction(candidate.endpoints, expected_segment)
            if overlap < config.min_overlap_fraction:
                counts["overlap"] += 1
                continue
            support_ratio = min(
                1.0,
                float(candidate.support_length_px) / max(1.0, _segment_length(expected_segment)),
            )
            score = (
                0.50 * distance / config.max_normal_distance_px
                + 0.20 * angle_delta / config.max_angle_delta_deg
                + 0.25 * (1.0 - overlap)
                + 0.05 * (1.0 - support_ratio)
            )
            rows.append(
                {
                    "candidate": candidate,
                    "score": float(score),
                    "distance": float(distance),
                    "angle_delta": float(angle_delta),
                    "overlap": float(overlap),
                    "roi_fraction": float(roi_fraction),
                }
            )
        rows.sort(key=lambda row: (row["score"], row["candidate"].candidate_id))
        pair_rows[line_id] = rows
        rejection_counts[line_id] = counts

    assignments: list[AssignedCourtLine] = []
    reasons: list[str] = []
    candidate_ids = [candidate.candidate_id for candidate in normalized]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("court-line candidate IDs must be unique")
    line_order = sorted(expected)
    candidate_order = sorted(normalized, key=lambda candidate: candidate.candidate_id)
    candidate_column = {
        candidate.candidate_id: index
        for index, candidate in enumerate(candidate_order)
    }
    pair_lookup = {
        (line_id, row["candidate"].candidate_id): row
        for line_id, rows in pair_rows.items()
        for row in rows
    }
    costs: np.ndarray | None = None
    initial_selected_columns: tuple[int, ...] = ()
    ambiguous: dict[str, float] = {}
    if line_order:
        from scipy.optimize import linear_sum_assignment

        real_column_count = len(candidate_order)
        invalid_cost = 1_000_000.0
        missing_cost = 10.0
        costs = np.full(
            (len(line_order), real_column_count + len(line_order)),
            invalid_cost,
            dtype=np.float64,
        )
        for line_index, line_id in enumerate(line_order):
            for row in pair_rows[line_id]:
                column = candidate_column[row["candidate"].candidate_id]
                costs[line_index, column] = (
                    float(row["score"]) + column * 1e-12
                )
            costs[line_index, real_column_count + line_index] = missing_cost
        selected_rows, selected_columns = linear_sum_assignment(costs)
        selected_by_line = {
            line_order[int(row_index)]: int(column_index)
            for row_index, column_index in zip(
                selected_rows,
                selected_columns,
                strict=True,
            )
        }
        initial_selected_columns = tuple(
            selected_by_line[line_id] for line_id in line_order
        )
        best_total = float(costs[selected_rows, selected_columns].sum())
        selection_margins: dict[str, float] = {}
        for line_index, line_id in enumerate(line_order):
            column = selected_by_line[line_id]
            if column >= real_column_count:
                continue
            alternate = costs.copy()
            alternate[line_index, column] = invalid_cost
            alternate_rows, alternate_columns = linear_sum_assignment(alternate)
            alternate_total = float(
                alternate[alternate_rows, alternate_columns].sum()
            )
            selection_margins[line_id] = alternate_total - best_total
        ambiguous = {
            line_id: margin
            for line_id, margin in selection_margins.items()
            if margin < config.min_candidate_margin
        }
        if ambiguous:
            # Do not discard a recoverable court here.  The bounded k-best
            # search below lets the joint regulation model resolve pairwise
            # ambiguity and still abstains if two coherent courts remain.
            pass
    else:
        selected_by_line = {}
        real_column_count = 0
        selection_margins = {}

    for line_id in line_order:
        column = selected_by_line.get(line_id, real_column_count)
        if column >= real_column_count:
            counts = rejection_counts[line_id]
            detail = ",".join(f"{name}={counts[name]}" for name in sorted(counts))
            reasons.append(f"{line_id}:no_consistent_candidate:{detail}")
            continue
        candidate = candidate_order[column]
        best = pair_lookup[(line_id, candidate.candidate_id)]
        margin = selection_margins.get(line_id)
        assignments.append(
            AssignedCourtLine(
                line_id=line_id,
                candidate_id=candidate.candidate_id,
                segment=candidate.endpoints,
                expected_segment=expected[line_id],
                score=float(best["score"]),
                normal_distance_px=float(best["distance"]),
                angle_delta_deg=float(best["angle_delta"]),
                overlap_fraction=float(best["overlap"]),
                support_length_px=float(candidate.support_length_px),
                selection_margin=margin,
            )
        )

    assignment_model = _score_joint_regulation_assignment(
        assignments,
        coordinate_space=coordinate_space,
        distortion_state=distortion_state,
        seed_calibration=seed_calibration,
    )
    joint_rejection = _joint_assignment_rejection_reason(
        assignments,
        assignment_model,
        config=config,
    )
    if (
        (joint_rejection is not None or ambiguous)
        and assignments
        and costs is not None
    ):
        recovered, recovered_model, recovery_reason = (
            _recover_joint_consistent_assignment(
                costs,
                initial_selected_columns=initial_selected_columns,
                line_order=line_order,
                candidate_order=candidate_order,
                pair_lookup=pair_lookup,
                expected=expected,
                coordinate_space=coordinate_space,
                distortion_state=distortion_state,
                seed_calibration=seed_calibration,
                config=config,
                initial_model=assignment_model,
            )
        )
        if recovered is not None:
            assignments = recovered
            assignment_model = recovered_model
            joint_rejection = None
            ambiguous = {}
        else:
            assignment_model = recovered_model
            if ambiguous:
                detail = ",".join(
                    f"{line_id}={margin:.6f}"
                    for line_id, margin in sorted(ambiguous.items())
                )
                reasons.append(
                    f"global_assignment_ambiguous:{detail}"
                )
            if recovery_reason is not None:
                reasons.append(recovery_reason)
            assignments = []
    else:
        assignment_model = {
            **assignment_model,
            "assignment_search": {
                "strategy": "pairwise_hungarian_then_joint_gate",
                "hypotheses_evaluated": 1 if assignments else 0,
                "hypothesis_budget": int(
                    config.max_joint_assignment_hypotheses
                ),
                "coherent_alternative_selected": False,
            },
        }
    if joint_rejection is not None and assignments:
        reasons.append(joint_rejection)
        assignments = []

    assignments.sort(key=lambda item: item.line_id)
    assigned_ids = {assignment.line_id for assignment in assignments}
    cross_count = len(assigned_ids & CROSS_LINE_IDS)
    longitudinal_count = _longitudinal_family_count(assigned_ids)
    if len(assignments) < config.min_assigned_lines:
        reasons.append(f"assigned_line_count_below_minimum:{len(assignments)}<{config.min_assigned_lines}")
    if cross_count < 2:
        reasons.append(f"cross_line_family_incomplete:{cross_count}<2")
    if longitudinal_count < 2:
        reasons.append(f"longitudinal_line_family_incomplete:{longitudinal_count}<2")
    if not assigned_ids.intersection({"left_sideline", "right_sideline"}):
        reasons.append("longitudinal_boundary_line_missing")
    status = "accepted" if not any(
        reason.startswith(
            (
                "assigned_line_count_",
                "cross_line_family_",
                "longitudinal_line_family_",
                "longitudinal_boundary_line_",
            )
        )
        for reason in reasons
    ) else "abstained"
    return FrameCourtLineEvidence(
        frame_index=int(frame_index),
        frame_sha256=str(frame_sha256),
        image_size=(int(image_size[0]), int(image_size[1])),
        coordinate_space=coordinate_space,
        distortion_state=distortion_state,
        provider=config.provider,
        raw_candidates=normalized,
        assignments=tuple(assignments),
        status=status,
        rejection_reasons=tuple(sorted(set(reasons))),
        roi_polygon_px=roi,
        roi_source=roi_source,
        seed_calibration_sha256=seed_calibration_sha256,
        template_projection_sha256=template_projection_sha256,
        assignment_model=_freeze_json_value(assignment_model),
    )


def _joint_assignment_rejection_reason(
    assignments: Sequence[AssignedCourtLine],
    assignment_model: Mapping[str, Any],
    *,
    config: CourtLineHardeningConfig,
) -> str | None:
    if not assignments:
        return None
    if (
        int(assignment_model.get("intersection_count") or 0)
        < config.min_joint_intersections
    ):
        return (
            "joint_regulation_intersections_below_minimum:"
            f"{assignment_model.get('intersection_count', 0)}"
            f"<{config.min_joint_intersections}"
        )
    if assignment_model.get("status") != "accepted":
        return (
            "joint_regulation_homography_unavailable:"
            f"{assignment_model.get('reason', 'unknown')}"
        )
    joint_p90 = assignment_model.get("joint_p90_px")
    if (
        not isinstance(joint_p90, (int, float))
        or not math.isfinite(float(joint_p90))
    ):
        return "joint_regulation_homography_unavailable:nonfinite_p90"
    if float(joint_p90) > config.max_joint_homography_p90_px:
        return (
            "joint_regulation_homography_p90_exceeds_max:"
            f"{float(joint_p90):.6f}"
            f">{config.max_joint_homography_p90_px:.6f}"
        )
    return None


def _assignment_family_reasons(
    assignments: Sequence[AssignedCourtLine],
    *,
    config: CourtLineHardeningConfig,
) -> list[str]:
    assigned_ids = {assignment.line_id for assignment in assignments}
    cross_count = len(assigned_ids & CROSS_LINE_IDS)
    longitudinal_count = _longitudinal_family_count(assigned_ids)
    reasons: list[str] = []
    if len(assignments) < config.min_assigned_lines:
        reasons.append(
            "assigned_line_count_below_minimum:"
            f"{len(assignments)}<{config.min_assigned_lines}"
        )
    if cross_count < 2:
        reasons.append(f"cross_line_family_incomplete:{cross_count}<2")
    if longitudinal_count < 2:
        reasons.append(
            f"longitudinal_line_family_incomplete:{longitudinal_count}<2"
        )
    if not assigned_ids.intersection(
        {"left_sideline", "right_sideline"}
    ):
        reasons.append("longitudinal_boundary_line_missing")
    return reasons


def _assignments_from_columns(
    selected_columns: Sequence[int],
    *,
    line_order: Sequence[str],
    candidate_order: Sequence[DetectedCourtLineCandidate],
    pair_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    expected: Mapping[str, Segment2],
) -> list[AssignedCourtLine]:
    real_column_count = len(candidate_order)
    assignments: list[AssignedCourtLine] = []
    for line_index, line_id in enumerate(line_order):
        column = int(selected_columns[line_index])
        if column >= real_column_count:
            continue
        candidate = candidate_order[column]
        row = pair_lookup.get((line_id, candidate.candidate_id))
        if row is None:
            continue
        assignments.append(
            AssignedCourtLine(
                line_id=line_id,
                candidate_id=candidate.candidate_id,
                segment=candidate.endpoints,
                expected_segment=expected[line_id],
                score=float(row["score"]),
                normal_distance_px=float(row["distance"]),
                angle_delta_deg=float(row["angle_delta"]),
                overlap_fraction=float(row["overlap"]),
                support_length_px=float(candidate.support_length_px),
                selection_margin=None,
            )
        )
    assignments.sort(key=lambda item: item.line_id)
    return assignments


def _recover_joint_consistent_assignment(
    costs: np.ndarray,
    *,
    initial_selected_columns: Sequence[int],
    line_order: Sequence[str],
    candidate_order: Sequence[DetectedCourtLineCandidate],
    pair_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    expected: Mapping[str, Segment2],
    coordinate_space: str,
    distortion_state: str,
    seed_calibration: Mapping[str, Any] | None,
    config: CourtLineHardeningConfig,
    initial_model: Mapping[str, Any],
) -> tuple[
    list[AssignedCourtLine] | None,
    dict[str, Any],
    str | None,
]:
    """Search bounded k-best Hungarian hypotheses for one coherent court.

    Pairwise seed proximity proposes hypotheses; regulation-wide geometry is
    the acceptance gate.  The bounded search is deterministic and fails
    closed if its budget cannot establish the configured ambiguity margin.
    """

    from scipy.optimize import linear_sum_assignment

    real_column_count = len(candidate_order)
    invalid_cost = 1_000_000.0
    queue: list[
        tuple[
            float,
            tuple[int, ...],
            tuple[tuple[int, int], ...],
            frozenset[tuple[int, int]],
        ]
    ] = []
    seen_states: set[frozenset[tuple[int, int]]] = set()

    def enqueue(banned: frozenset[tuple[int, int]]) -> None:
        if banned in seen_states:
            return
        seen_states.add(banned)
        candidate_costs = costs.copy()
        for row_index, column_index in banned:
            candidate_costs[row_index, column_index] = invalid_cost
        rows, columns = linear_sum_assignment(candidate_costs)
        selected = [real_column_count + index for index in range(len(line_order))]
        for row_index, column_index in zip(rows, columns, strict=True):
            selected[int(row_index)] = int(column_index)
        selected_tuple = tuple(selected)
        total = float(candidate_costs[rows, columns].sum())
        heappush(
            queue,
            (
                total,
                selected_tuple,
                tuple(sorted(banned)),
                banned,
            ),
        )

    enqueue(frozenset())

    evaluated = 0
    coherent: tuple[
        float,
        tuple[int, ...],
        list[AssignedCourtLine],
        dict[str, Any],
    ] | None = None
    coherent_solutions: set[tuple[int, ...]] = set()
    while queue and evaluated < config.max_joint_assignment_hypotheses:
        total, selected_columns, _ordered_bans, banned = heappop(queue)
        evaluated += 1
        assignments = _assignments_from_columns(
            selected_columns,
            line_order=line_order,
            candidate_order=candidate_order,
            pair_lookup=pair_lookup,
            expected=expected,
        )
        family_reasons = _assignment_family_reasons(
            assignments,
            config=config,
        )
        model = _score_joint_regulation_assignment(
            assignments,
            coordinate_space=coordinate_space,
            distortion_state=distortion_state,
            seed_calibration=seed_calibration,
        )
        joint_rejection = _joint_assignment_rejection_reason(
            assignments,
            model,
            config=config,
        )
        if (
            not family_reasons
            and joint_rejection is None
            and selected_columns not in coherent_solutions
        ):
            coherent_solutions.add(selected_columns)
            if coherent is None:
                coherent = (
                    total,
                    selected_columns,
                    assignments,
                    model,
                )
            elif total - coherent[0] < config.min_candidate_margin:
                failed_model = {
                    **dict(initial_model),
                    "assignment_search": {
                        "strategy": (
                            "bounded_k_best_hungarian_with_joint_gate"
                        ),
                        "status": "ambiguous",
                        "hypotheses_evaluated": evaluated,
                        "hypothesis_budget": int(
                            config.max_joint_assignment_hypotheses
                        ),
                        "best_coherent_pair_cost": float(coherent[0]),
                        "alternate_coherent_pair_cost": float(total),
                        "coherent_pair_cost_margin": float(
                            total - coherent[0]
                        ),
                        "coherent_alternative_selected": False,
                    },
                }
                return (
                    None,
                    failed_model,
                    "joint_consistent_assignment_ambiguous:"
                    f"{total - coherent[0]:.6f}"
                    f"<{config.min_candidate_margin:.6f}",
                )

        for row_index, column_index in enumerate(selected_columns):
            edge = (row_index, int(column_index))
            if column_index >= real_column_count or edge in banned:
                continue
            enqueue(frozenset((*banned, edge)))

        if (
            coherent is not None
            and (
                not queue
                or queue[0][0]
                >= coherent[0] + config.min_candidate_margin
            )
        ):
            selected_model = {
                **coherent[3],
                "assignment_search": {
                    "strategy": (
                        "bounded_k_best_hungarian_with_joint_gate"
                    ),
                    "status": "accepted",
                    "hypotheses_evaluated": evaluated,
                    "hypothesis_budget": int(
                        config.max_joint_assignment_hypotheses
                    ),
                    "best_coherent_pair_cost": float(coherent[0]),
                    "coherent_margin_lower_bound": (
                        None
                        if not queue
                        else float(queue[0][0] - coherent[0])
                    ),
                    "coherent_alternative_selected": (
                        coherent[1]
                        != tuple(initial_selected_columns)
                    ),
                    "initial_pairwise_assignment_selected": (
                        coherent[1]
                        == tuple(initial_selected_columns)
                    ),
                    "selected_candidate_ids": [
                        assignment.candidate_id
                        for assignment in coherent[2]
                    ],
                },
            }
            return coherent[2], selected_model, None

    if coherent is not None and (
        not queue
        or queue[0][0] >= coherent[0] + config.min_candidate_margin
    ):
        selected_model = {
            **coherent[3],
            "assignment_search": {
                "strategy": "bounded_k_best_hungarian_with_joint_gate",
                "status": "accepted",
                "hypotheses_evaluated": evaluated,
                "hypothesis_budget": int(
                    config.max_joint_assignment_hypotheses
                ),
                "best_coherent_pair_cost": float(coherent[0]),
                "coherent_margin_lower_bound": (
                    None
                    if not queue
                    else float(queue[0][0] - coherent[0])
                ),
                "coherent_alternative_selected": (
                    coherent[1] != tuple(initial_selected_columns)
                ),
                "initial_pairwise_assignment_selected": (
                    coherent[1] == tuple(initial_selected_columns)
                ),
                "selected_candidate_ids": [
                    assignment.candidate_id for assignment in coherent[2]
                ],
            },
        }
        return coherent[2], selected_model, None

    failed_model = {
        **dict(initial_model),
        "assignment_search": {
            "strategy": "bounded_k_best_hungarian_with_joint_gate",
            "status": (
                "budget_exhausted"
                if queue
                else "no_coherent_alternative"
            ),
            "hypotheses_evaluated": evaluated,
            "hypothesis_budget": int(
                config.max_joint_assignment_hypotheses
            ),
            "coherent_alternative_selected": False,
        },
    }
    reason = (
        "joint_assignment_search_budget_exhausted_before_margin"
        if queue
        else "joint_assignment_no_coherent_alternative"
    )
    return None, failed_model, reason


def _score_joint_regulation_assignment(
    assignments: Sequence[AssignedCourtLine],
    *,
    coordinate_space: str,
    distortion_state: str,
    seed_calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fit one court homography to all assigned line intersections.

    A raw distorted court is not a projective straight-line grid.  Candidate
    endpoints are therefore mapped to native-pixel undistorted coordinates
    before this joint consistency check.  The configured tolerance remains in
    pixel-like native camera units rather than being weakened for distortion.
    """

    if not assignments:
        return {
            "status": "not_evaluated",
            "reason": "no_assignments",
            "intersection_count": 0,
            "intersection_p90_px": None,
            "line_p90_px": None,
            "joint_p90_px": None,
            "homography_image_from_court": None,
            "evaluation_coordinate_space": None,
        }
    if coordinate_space != "pixels_raw_native":
        return {
            "status": "abstained",
            "reason": f"unsupported_joint_coordinate_space:{coordinate_space}",
            "intersection_count": 0,
            "intersection_p90_px": None,
            "line_p90_px": None,
            "joint_p90_px": None,
            "homography_image_from_court": None,
            "evaluation_coordinate_space": None,
        }
    try:
        assignment_segments, evaluation_coordinate_space = (
            _joint_assignment_segments(
                assignments,
                distortion_state=distortion_state,
                seed_calibration=seed_calibration,
            )
        )
    except ValueError as exc:
        return {
            "status": "abstained",
            "reason": str(exc),
            "intersection_count": 0,
            "intersection_p90_px": None,
            "line_p90_px": None,
            "joint_p90_px": None,
            "homography_image_from_court": None,
            "evaluation_coordinate_space": None,
        }
    template = get_court_template("pickleball")
    world_points: list[Point2] = []
    image_points: list[Point2] = []
    line_ids = {assignment.line_id for assignment in assignments}
    assignment_by_id = {
        assignment.line_id: assignment for assignment in assignments
    }
    for cross_id in sorted(line_ids & CROSS_LINE_IDS):
        for longitudinal_id in sorted(line_ids & LONGITUDINAL_LINE_IDS):
            world_intersection = _finite_segment_intersection(
                template.line_segments_m[cross_id],
                template.line_segments_m[longitudinal_id],
            )
            if world_intersection is None:
                continue
            try:
                image_intersection = _infinite_line_intersection(
                    assignment_segments[cross_id],
                    assignment_segments[longitudinal_id],
                )
            except ValueError:
                return {
                    "status": "abstained",
                    "reason": (
                        "assigned_cross_and_longitudinal_lines_near_parallel:"
                        f"{cross_id},{longitudinal_id}"
                    ),
                    "intersection_count": len(world_points),
                    "intersection_p90_px": None,
                    "line_p90_px": None,
                    "joint_p90_px": None,
                    "homography_image_from_court": None,
                    "evaluation_coordinate_space": evaluation_coordinate_space,
                }
            world_points.append(world_intersection)
            image_points.append(image_intersection)
    if len(world_points) < 4:
        return {
            "status": "insufficient",
            "reason": "fewer_than_four_finite_regulation_intersections",
            "intersection_count": len(world_points),
            "intersection_p90_px": None,
            "line_p90_px": None,
            "joint_p90_px": None,
            "homography_image_from_court": None,
            "evaluation_coordinate_space": evaluation_coordinate_space,
        }
    try:
        homography = homography_from_planar_points(world_points, image_points)
        projected_intersections = np.asarray(
            project_planar_points(homography, world_points),
            dtype=np.float64,
        )
    except (ValueError, ZeroDivisionError) as exc:
        return {
            "status": "abstained",
            "reason": f"joint_homography_fit_failed:{type(exc).__name__}",
            "intersection_count": len(world_points),
            "intersection_p90_px": None,
            "line_p90_px": None,
            "joint_p90_px": None,
            "homography_image_from_court": None,
            "evaluation_coordinate_space": evaluation_coordinate_space,
        }
    intersection_residuals = np.linalg.norm(
        projected_intersections - np.asarray(image_points, dtype=np.float64),
        axis=1,
    )
    line_residuals = []
    for assignment in assignments:
        projected_segment_raw = project_planar_points(
            homography,
            template.line_segments_m[assignment.line_id],
        )
        projected_segment: Segment2 = (
            (
                float(projected_segment_raw[0][0]),
                float(projected_segment_raw[0][1]),
            ),
            (
                float(projected_segment_raw[1][0]),
                float(projected_segment_raw[1][1]),
            ),
        )
        line_residuals.append(
            _symmetric_line_distance(
                assignment_segments[assignment.line_id],
                projected_segment,
            )
        )
    intersection_p90 = float(np.percentile(intersection_residuals, 90))
    line_p90 = float(np.percentile(np.asarray(line_residuals), 90))
    return {
        "status": "accepted",
        "reason": None,
        "intersection_count": len(world_points),
        "intersection_p90_px": intersection_p90,
        "line_p90_px": line_p90,
        "joint_p90_px": max(intersection_p90, line_p90),
        "homography_image_from_court": homography,
        "evaluation_coordinate_space": evaluation_coordinate_space,
    }


def _joint_assignment_segments(
    assignments: Sequence[AssignedCourtLine],
    *,
    distortion_state: str,
    seed_calibration: Mapping[str, Any] | None,
) -> tuple[dict[str, Segment2], str]:
    segments = {
        assignment.line_id: assignment.segment for assignment in assignments
    }
    if distortion_state == "raw_pinhole":
        return segments, "pixels_raw_native_pinhole"
    if distortion_state != "raw_distorted_with_declared_model":
        raise ValueError(
            f"joint_distortion_state_not_declared:{distortion_state}"
        )
    if seed_calibration is None:
        raise ValueError(
            "joint_distorted_geometry_requires_seed_calibration"
        )
    intrinsics = seed_calibration.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("joint_distorted_geometry_intrinsics_missing")
    try:
        import cv2

        camera = np.asarray(
            [
                [
                    float(intrinsics["fx"]),
                    0.0,
                    float(intrinsics["cx"]),
                ],
                [
                    0.0,
                    float(intrinsics["fy"]),
                    float(intrinsics["cy"]),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(
            intrinsics.get("dist", []),
            dtype=np.float64,
        ).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "joint_distorted_geometry_intrinsics_invalid"
        ) from exc
    if (
        camera.shape != (3, 3)
        or not np.all(np.isfinite(camera))
        or distortion.size < 4
        or not np.all(np.isfinite(distortion))
    ):
        raise ValueError("joint_distorted_geometry_intrinsics_invalid")
    result: dict[str, Segment2] = {}
    for assignment in assignments:
        raw = np.asarray(assignment.segment, dtype=np.float64).reshape(
            -1, 1, 2
        )
        undistorted = cv2.undistortPoints(
            raw,
            camera,
            distortion,
            P=camera,
        ).reshape(-1, 2)
        if not np.all(np.isfinite(undistorted)):
            raise ValueError(
                "joint_distorted_geometry_undistortion_nonfinite"
            )
        result[assignment.line_id] = (
            (
                float(undistorted[0][0]),
                float(undistorted[0][1]),
            ),
            (
                float(undistorted[1][0]),
                float(undistorted[1][1]),
            ),
        )
    return result, "pixels_undistorted_native"


def _finite_segment_intersection(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> Point2 | None:
    p = np.asarray(first[0][:2], dtype=np.float64)
    r = np.asarray(first[1][:2], dtype=np.float64) - p
    q = np.asarray(second[0][:2], dtype=np.float64)
    s = np.asarray(second[1][:2], dtype=np.float64) - q
    determinant = float(r[0] * s[1] - r[1] * s[0])
    if abs(determinant) <= 1e-12:
        return None
    q_minus_p = q - p
    t = float(
        (q_minus_p[0] * s[1] - q_minus_p[1] * s[0])
        / determinant
    )
    u = float(
        (q_minus_p[0] * r[1] - q_minus_p[1] * r[0])
        / determinant
    )
    if not (-1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9):
        return None
    point = p + t * r
    return float(point[0]), float(point[1])


def _infinite_line_intersection(first: Segment2, second: Segment2) -> Point2:
    a1, b1, c1 = _line_coefficients(first)
    a2, b2, c2 = _line_coefficients(second)
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= 1e-9:
        raise ValueError("court lines are parallel")
    return (
        float((b1 * c2 - b2 * c1) / determinant),
        float((a2 * c1 - a1 * c2) / determinant),
    )


def evaluate_static_consistency(
    frame_evidence: Sequence[FrameCourtLineEvidence],
    *,
    config: CourtLineHardeningConfig,
) -> StaticConsistencyResult:
    """Measure coherent semantic-line drift before admitting any pooled line.

    The detector can produce isolated bad paint samples on a genuinely static
    clip.  Camera motion, by contrast, shifts the robust center of many samples
    on the same semantic line together or moves the court outside every seed
    ROI.  Center every sample over time, reduce each frame/line to robust
    whole/start/end measurements, and apply the diagnostic's same dispersion
    bound to both scaled MAD and robust temporal-window span.  A substantial
    run of frames with no semantic assignments is also a typed abstention.
    """

    config.validate()
    if not frame_evidence:
        raise ValueError("at least one frame evidence record is required")
    ordered = sorted(frame_evidence, key=lambda item: item.frame_index)
    frame_position = {
        frame.frame_index: position
        for position, frame in enumerate(ordered)
    }
    measurements: dict[str, list[tuple[int, float]]] = {}
    usable_line_count_by_position: dict[int, int] = {}
    raw_template_count_by_position: dict[int, int] = {}

    if config.provider == "seed_guided_paired_edges":
        by_sample: dict[
            tuple[str, int],
            list[tuple[int, float]],
        ] = {}
        assigned_by_frame = {
            frame.frame_index: {assignment.line_id for assignment in frame.assignments}
            for frame in ordered
        }
        for frame in ordered:
            assigned = assigned_by_frame[frame.frame_index]
            for sample in frame.template_samples:
                if sample.line_id not in assigned:
                    continue
                by_sample.setdefault(
                    (sample.line_id, sample.sample_index),
                    [],
                ).append((frame.frame_index, float(sample.signed_offset_px)))

        sample_centers = {
            key: float(np.median([value for _index, value in rows]))
            for key, rows in by_sample.items()
            if len(rows) >= config.min_contributing_frames
        }
        for frame in ordered:
            position = frame_position[frame.frame_index]
            raw_deltas = [
                float(sample.signed_offset_px)
                - sample_centers[(sample.line_id, sample.sample_index)]
                for sample in frame.template_samples
                if (sample.line_id, sample.sample_index) in sample_centers
            ]
            raw_template_count_by_position[position] = len(raw_deltas)
            if not raw_deltas:
                continue
            measurements.setdefault("camera_raw:signed", []).append(
                (position, float(np.median(raw_deltas)))
            )
            measurements.setdefault("camera_raw:magnitude", []).append(
                (
                    position,
                    float(
                        np.median(np.abs(np.asarray(raw_deltas)))
                    ),
                )
            )
        per_frame_line: dict[
            tuple[int, str],
            list[tuple[int, float]],
        ] = {}
        for frame in ordered:
            assigned = assigned_by_frame[frame.frame_index]
            for sample in frame.template_samples:
                key = (sample.line_id, sample.sample_index)
                if sample.line_id not in assigned or key not in sample_centers:
                    continue
                per_frame_line.setdefault(
                    (frame.frame_index, sample.line_id),
                    [],
                ).append(
                    (
                        sample.sample_index,
                        float(sample.signed_offset_px) - sample_centers[key],
                    )
                )

        per_frame_whole_measurements: dict[int, list[float]] = {}
        for (_frame_index, line_id), rows in sorted(per_frame_line.items()):
            ordered_rows = sorted(rows)
            if len(ordered_rows) >= config.min_heldout_line_samples:
                whole_measurement = float(
                    np.median(
                        [value for _sample_index, value in ordered_rows]
                    )
                )
                measurements.setdefault(f"{line_id}:all", []).append(
                    (
                        frame_position[_frame_index],
                        whole_measurement,
                    )
                )
                per_frame_whole_measurements.setdefault(
                    _frame_index,
                    [],
                ).append(whole_measurement)
            if len(ordered_rows) >= 2 * config.min_heldout_line_samples:
                third = max(
                    config.min_holdout_contributing_frames,
                    len(ordered_rows) // 3,
                )
                measurements.setdefault(f"{line_id}:start", []).append(
                    (
                        frame_position[_frame_index],
                        float(
                            np.median(
                                [
                                    value
                                    for _sample_index, value in ordered_rows[:third]
                                ]
                            )
                        ),
                    )
                )
                measurements.setdefault(f"{line_id}:end", []).append(
                    (
                        frame_position[_frame_index],
                        float(
                            np.median(
                                [
                                    value
                                    for _sample_index, value in ordered_rows[-third:]
                                ]
                            )
                        ),
                    )
                )
        for frame_index, values in sorted(
            per_frame_whole_measurements.items()
        ):
            position = frame_position[frame_index]
            usable_line_count_by_position[position] = len(values)
            measurements.setdefault(
                "camera_available:magnitude",
                [],
            ).append(
                (
                    position,
                    float(np.median(np.abs(np.asarray(values)))),
                )
            )
            if len(values) < config.min_assigned_lines:
                continue
            measurements.setdefault(
                "camera_consensus:signed",
                [],
            ).append((position, float(np.median(values))))
            measurements.setdefault(
                "camera_consensus:magnitude",
                [],
            ).append(
                (
                    position,
                    float(np.median(np.abs(np.asarray(values)))),
                )
            )
    else:
        by_line: dict[str, list[tuple[int, float]]] = {}
        for position, frame in enumerate(ordered):
            for assignment in frame.assignments:
                by_line.setdefault(assignment.line_id, []).append(
                    (
                        position,
                        _signed_segment_offset(
                            assignment.segment,
                            assignment.expected_segment,
                        ),
                    )
                )
        measurements = {
            f"{line_id}:all": values
            for line_id, values in by_line.items()
        }

    measured_mads: list[tuple[str, float]] = []
    measured_temporal_spans: list[tuple[str, float]] = []
    temporal_window_size = max(
        config.min_contributing_frames,
        config.min_heldout_line_samples,
    )
    assignment_dropout_limit = temporal_window_size
    boundary_assignment_dropout_limit = 1
    for measurement, rows in sorted(measurements.items()):
        ordered_rows = sorted(rows)
        values = [value for _position, value in ordered_rows]
        if len(values) < config.min_contributing_frames:
            continue
        array = np.asarray(values, dtype=np.float64)
        center = float(np.median(array))
        scaled_mad = float(1.4826 * np.median(np.abs(array - center)))
        measured_mads.append((measurement, scaled_mad))
        if measurement.startswith("camera_"):
            measured_temporal_spans.append(
                (
                    measurement,
                    float(max(values) - min(values)),
                )
            )
        elif len(values) >= 2 * temporal_window_size:
            window_medians = [
                float(np.median(values[start : start + temporal_window_size]))
                for start in range(
                    0,
                    len(values) - temporal_window_size + 1,
                )
            ]
            measured_temporal_spans.append(
                (
                    measurement,
                    float(max(window_medians) - min(window_medians)),
                )
            )

    missing_assignment_spans: list[tuple[int, int, int, int, int]] = []
    span_start: int | None = None
    for position, frame in enumerate(ordered):
        has_usable_evidence = (
            usable_line_count_by_position.get(position, 0) > 0
            if config.provider == "seed_guided_paired_edges"
            else bool(frame.assignments)
        )
        if not has_usable_evidence and span_start is None:
            span_start = position
        if has_usable_evidence and span_start is not None:
            span_end = position - 1
            span_count = span_end - span_start + 1
            boundary_span = span_start == 0
            if (
                span_count > assignment_dropout_limit
                or (
                    boundary_span
                    and span_count >= boundary_assignment_dropout_limit
                )
            ):
                missing_assignment_spans.append(
                    (
                        span_start,
                        span_end,
                        span_count,
                        ordered[span_start].frame_index,
                        ordered[span_end].frame_index,
                    )
                )
            span_start = None
    if span_start is not None:
        span_end = len(ordered) - 1
        span_count = span_end - span_start + 1
        if (
            span_count > assignment_dropout_limit
            or span_count >= boundary_assignment_dropout_limit
        ):
            missing_assignment_spans.append(
                (
                    span_start,
                    span_end,
                    span_count,
                    ordered[span_start].frame_index,
                    ordered[span_end].frame_index,
                )
            )

    missing_raw_template_spans: list[
        tuple[int, int, int, int, int]
    ] = []
    if config.provider == "seed_guided_paired_edges":
        span_start = None
        for position in range(len(ordered)):
            has_raw_template_evidence = (
                raw_template_count_by_position.get(position, 0) > 0
            )
            if not has_raw_template_evidence and span_start is None:
                span_start = position
            if has_raw_template_evidence and span_start is not None:
                span_end = position - 1
                missing_raw_template_spans.append(
                    (
                        span_start,
                        span_end,
                        span_end - span_start + 1,
                        ordered[span_start].frame_index,
                        ordered[span_end].frame_index,
                    )
                )
                span_start = None
        if span_start is not None:
            span_end = len(ordered) - 1
            missing_raw_template_spans.append(
                (
                    span_start,
                    span_end,
                    span_end - span_start + 1,
                    ordered[span_start].frame_index,
                    ordered[span_end].frame_index,
                )
            )

    boundary_degraded_frames: list[tuple[int, int, int]] = []
    if config.provider == "seed_guided_paired_edges":
        for position in sorted({0, len(ordered) - 1}):
            usable_line_count = usable_line_count_by_position.get(
                position,
                0,
            )
            if 0 < usable_line_count < config.min_assigned_lines:
                boundary_degraded_frames.append(
                    (
                        position,
                        ordered[position].frame_index,
                        usable_line_count,
                    )
                )

    if not measured_mads:
        return StaticConsistencyResult(
            status="abstained",
            dispersion_bound_px=float(config.max_static_line_mad_px),
            max_observed_mad_px=None,
            max_observed_temporal_span_px=None,
            per_measurement_mad_px=(),
            per_measurement_temporal_span_px=(),
            violating_measurements=(),
            violating_assignment_dropout_spans=tuple(
                missing_assignment_spans
            ),
            violating_raw_template_dropout_spans=tuple(
                missing_raw_template_spans
            ),
            violating_boundary_degraded_frames=tuple(
                boundary_degraded_frames
            ),
            rejection_reasons=("static_consistency_insufficient_measurements",),
        )

    violating_mad = {
        measurement
        for measurement, value in measured_mads
        if value > config.max_static_line_mad_px
    }
    violating_temporal = {
        measurement
        for measurement, value in measured_temporal_spans
        if value > config.max_static_line_mad_px
    }
    violating = tuple(sorted(violating_mad | violating_temporal))
    max_observed_mad = max(value for _measurement, value in measured_mads)
    max_observed_temporal_span = (
        max(value for _measurement, value in measured_temporal_spans)
        if measured_temporal_spans
        else None
    )
    reasons: list[str] = []
    if violating_mad:
        details = ",".join(
            f"{measurement}={value:.6f}"
            for measurement, value in measured_mads
            if measurement in violating_mad
        )
        reasons.append(
            "static_consistency_drift_exceeds_bound:"
            f"{details}>{config.max_static_line_mad_px:.6f}"
        )
    if violating_temporal:
        details = ",".join(
            f"{measurement}={value:.6f}"
            for measurement, value in measured_temporal_spans
            if measurement in violating_temporal
        )
        reasons.append(
            "static_consistency_temporal_shift_exceeds_bound:"
            f"{details}>{config.max_static_line_mad_px:.6f}"
        )
    if missing_assignment_spans:
        details = ",".join(
            (
                f"samples[{start_position}:{end_position}]"
                f"/frames[{start_frame_index}:{end_frame_index}]"
                f"={sample_count}"
            )
            for (
                start_position,
                end_position,
                sample_count,
                start_frame_index,
                end_frame_index,
            ) in missing_assignment_spans
        )
        reasons.append(
            "static_consistency_assignment_dropout_span:"
            f"{details};boundary_min="
            f"{boundary_assignment_dropout_limit},"
            f"internal_max={assignment_dropout_limit}"
        )
    if missing_raw_template_spans:
        details = ",".join(
            (
                f"samples[{start_position}:{end_position}]"
                f"/frames[{start_frame_index}:{end_frame_index}]"
                f"={sample_count}"
            )
            for (
                start_position,
                end_position,
                sample_count,
                start_frame_index,
                end_frame_index,
            ) in missing_raw_template_spans
        )
        reasons.append(
            "static_consistency_raw_template_dropout_span:"
            f"{details};max=0"
        )
    if boundary_degraded_frames:
        details = ",".join(
            (
                f"sample[{sample_position}]"
                f"/frame[{frame_index}]={assigned_line_count}"
            )
            for (
                sample_position,
                frame_index,
                assigned_line_count,
            ) in boundary_degraded_frames
        )
        reasons.append(
            "static_consistency_boundary_assignment_below_consensus:"
            f"{details}<{config.min_assigned_lines}"
        )
    return StaticConsistencyResult(
        status=(
            "abstained"
            if (
                violating
                or missing_assignment_spans
                or missing_raw_template_spans
                or boundary_degraded_frames
            )
            else "accepted"
        ),
        dispersion_bound_px=float(config.max_static_line_mad_px),
        max_observed_mad_px=float(max_observed_mad),
        max_observed_temporal_span_px=(
            float(max_observed_temporal_span)
            if max_observed_temporal_span is not None
            else None
        ),
        per_measurement_mad_px=tuple(measured_mads),
        per_measurement_temporal_span_px=tuple(measured_temporal_spans),
        violating_measurements=violating,
        violating_assignment_dropout_spans=tuple(missing_assignment_spans),
        violating_raw_template_dropout_spans=tuple(
            missing_raw_template_spans
        ),
        violating_boundary_degraded_frames=tuple(
            boundary_degraded_frames
        ),
        rejection_reasons=tuple(reasons),
    )


def pool_static_semantic_lines(
    frame_evidence: Sequence[FrameCourtLineEvidence],
    *,
    config: CourtLineHardeningConfig,
) -> PooledCourtLineEvidence:
    """Robust-median pool line parameters while retaining frame provenance."""

    config.validate()
    if not frame_evidence:
        raise ValueError("at least one frame evidence record is required")
    ordered = sorted(frame_evidence, key=lambda item: item.frame_index)
    indexes = [item.frame_index for item in ordered]
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate frame evidence indexes are not allowed")
    frame_hashes = [item.frame_sha256 for item in ordered]
    if len(set(frame_hashes)) != len(frame_hashes):
        raise ValueError("duplicate decoded frame hashes are not independent evidence")
    image_sizes = {item.image_size for item in ordered}
    coordinate_spaces = {item.coordinate_space for item in ordered}
    distortion_states = {item.distortion_state for item in ordered}
    providers = {item.provider for item in ordered}
    seed_hashes = {
        item.seed_calibration_sha256
        for item in ordered
        if item.seed_calibration_sha256 is not None
    }
    template_hashes = {
        item.template_projection_sha256
        for item in ordered
        if item.template_projection_sha256 is not None
    }
    if len(image_sizes) != 1:
        raise ValueError("cannot pool court lines across different image sizes")
    if len(coordinate_spaces) != 1:
        raise ValueError("cannot pool court lines across coordinate spaces")
    if len(distortion_states) != 1:
        raise ValueError("cannot pool court lines across distortion states")
    if providers != {config.provider}:
        raise ValueError(
            f"frame providers {sorted(providers)} do not match config provider {config.provider}"
        )
    if seed_hashes and (
        len(seed_hashes) != 1
        or any(item.seed_calibration_sha256 is None for item in ordered)
    ):
        raise ValueError("frame evidence carries inconsistent seed calibration hashes")
    if template_hashes and (
        len(template_hashes) != 1
        or any(item.template_projection_sha256 is None for item in ordered)
    ):
        raise ValueError("frame evidence carries inconsistent template projection hashes")
    if config.provider == "seed_guided_paired_edges" and (
        len(seed_hashes) != 1 or len(template_hashes) != 1
    ):
        raise ValueError(
            "seed-guided frame evidence must bind one seed and template projection"
        )
    for frame in ordered:
        line_ids = [assignment.line_id for assignment in frame.assignments]
        if len(set(line_ids)) != len(line_ids):
            raise ValueError(
                f"frame {frame.frame_index} contains duplicate semantic assignments"
            )
        sample_keys = [
            (sample.line_id, sample.sample_index) for sample in frame.template_samples
        ]
        if len(set(sample_keys)) != len(sample_keys):
            raise ValueError(
                f"frame {frame.frame_index} contains duplicate template samples"
            )

    static_consistency = evaluate_static_consistency(ordered, config=config)
    if config.provider == "seed_guided_paired_edges":
        pooled_lines, missing, reasons = _pool_seed_guided_lines(
            ordered,
            config=config,
        )
    else:
        if any(frame.template_samples for frame in ordered):
            raise ValueError(
                "segment-only providers cannot carry seed-guided template samples"
            )
        pooled_lines, missing, reasons = _pool_segment_lines(
            ordered,
            config=config,
        )
    line_ids = {line.line_id for line in pooled_lines}
    if len(pooled_lines) < config.min_assigned_lines:
        reasons.append(f"pooled_line_count_below_minimum:{len(pooled_lines)}<{config.min_assigned_lines}")
    if len(line_ids & CROSS_LINE_IDS) < 2:
        reasons.append("pooled_cross_line_family_incomplete")
    if _longitudinal_family_count(line_ids) < 2:
        reasons.append("pooled_longitudinal_line_family_incomplete")
    if not line_ids.intersection({"left_sideline", "right_sideline"}):
        reasons.append("pooled_longitudinal_boundary_line_missing")
    fatal_prefixes = (
        "pooled_line_count_",
        "pooled_cross_line_",
        "pooled_longitudinal_line_",
    )
    reasons.extend(static_consistency.rejection_reasons)
    status = (
        "accepted"
        if (
            static_consistency.status == "accepted"
            and not any(reason.startswith(fatal_prefixes) for reason in reasons)
        )
        else "abstained"
    )
    return PooledCourtLineEvidence(
        image_size=next(iter(image_sizes)),
        coordinate_space=next(iter(coordinate_spaces)),
        distortion_state=next(iter(distortion_states)),
        config_sha256=_sha256_payload(config.evidence_config_dict()),
        source_frame_hashes=tuple((frame.frame_index, frame.frame_sha256) for frame in ordered),
        lines=tuple(pooled_lines),
        missing_line_ids=tuple(sorted(missing)),
        status=status,
        rejection_reasons=tuple(sorted(set(reasons))),
        source_frame_evidence_hashes=tuple(
            (frame.frame_index, _sha256_payload(frame.as_dict()))
            for frame in ordered
        ),
        seed_calibration_sha256=(
            next(iter(seed_hashes)) if seed_hashes else None
        ),
        template_projection_sha256=(
            next(iter(template_hashes)) if template_hashes else None
        ),
        static_consistency=static_consistency,
    )


def _pool_segment_lines(
    ordered: Sequence[FrameCourtLineEvidence],
    *,
    config: CourtLineHardeningConfig,
) -> tuple[list[PooledSemanticLine], list[str], list[str]]:
    """Compatibility pool for additive global segment providers."""

    holdout_indexes = {
        frame.frame_index
        for position, frame in enumerate(ordered, start=1)
        if position % config.frame_holdout_stride == 0
    }
    by_line: dict[str, list[tuple[int, AssignedCourtLine]]] = {}
    for frame in ordered:
        for assignment in frame.assignments:
            by_line.setdefault(assignment.line_id, []).append(
                (frame.frame_index, assignment)
            )
    pooled_lines: list[PooledSemanticLine] = []
    missing: list[str] = []
    reasons: list[str] = []
    for line_id in FLOOR_LINE_IDS:
        rows = sorted(by_line.get(line_id, []), key=lambda item: item[0])
        optimize_rows = [
            row for row in rows if row[0] not in holdout_indexes
        ]
        heldout_rows = [row for row in rows if row[0] in holdout_indexes]
        if len(optimize_rows) < config.min_contributing_frames:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:contributors_below_minimum:"
                f"{len(optimize_rows)}<{config.min_contributing_frames}"
            )
            continue
        if len(heldout_rows) < config.min_holdout_contributing_frames:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:heldout_contributors_below_minimum:"
                f"{len(heldout_rows)}<{config.min_holdout_contributing_frames}"
            )
            continue
        expected = optimize_rows[0][1].expected_segment
        if any(
            _segment_endpoint_delta(row[1].expected_segment, expected) > 1e-6
            for row in rows[1:]
        ):
            raise ValueError(
                f"expected template projection changed across static frames for {line_id}"
            )
        signed_offsets = np.asarray(
            [
                _signed_segment_offset(assignment.segment, expected)
                for _index, assignment in optimize_rows
            ],
            dtype=np.float64,
        )
        center = float(np.median(signed_offsets))
        scaled_mad = float(
            1.4826 * np.median(np.abs(signed_offsets - center))
        )
        if scaled_mad > config.max_static_line_mad_px:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:static_signed_mad_exceeds_max:"
                f"{scaled_mad:.6f}>{config.max_static_line_mad_px:.6f}"
            )
            continue
        threshold = max(
            1.0,
            config.pool_outlier_mad_scale * max(0.15, scaled_mad),
        )
        kept = [
            row
            for row, offset in zip(optimize_rows, signed_offsets, strict=True)
            if abs(float(offset) - center) <= threshold + 1e-12
        ]
        rejected = [
            row[0]
            for row, offset in zip(optimize_rows, signed_offsets, strict=True)
            if abs(float(offset) - center) > threshold + 1e-12
        ]
        if len(kept) < config.min_contributing_frames:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:inliers_below_minimum:"
                f"{len(kept)}<{config.min_contributing_frames}"
            )
            continue
        pooled_segment = _median_projected_segment(
            [assignment.segment for _index, assignment in kept],
            expected,
        )
        final_deviations = np.asarray(
            [
                _symmetric_line_distance(assignment.segment, pooled_segment)
                for _index, assignment in kept
            ],
            dtype=np.float64,
        )
        final_p90 = float(np.percentile(final_deviations, 90))
        if final_p90 > config.max_static_line_mad_px:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:static_dispersion_p90_exceeds_max:"
                f"{final_p90:.6f}>{config.max_static_line_mad_px:.6f}"
            )
            continue
        pooled_lines.append(
            PooledSemanticLine(
                line_id=line_id,
                segment=pooled_segment,
                expected_segment=expected,
                contributing_frame_indexes=tuple(
                    index for index, _assignment in kept
                ),
                heldout_frame_indexes=tuple(
                    index for index, _assignment in heldout_rows
                ),
                rejected_frame_indexes=tuple(sorted(rejected)),
                heldout_segments=tuple(
                    assignment.segment for _index, assignment in heldout_rows
                ),
                dispersion_mad_px=scaled_mad,
                geometry_fit_p90_px=final_p90,
            )
        )
    pooled_lines.sort(key=lambda item: item.line_id)
    return pooled_lines, missing, reasons


def _pool_seed_guided_lines(
    ordered: Sequence[FrameCourtLineEvidence],
    *,
    config: CourtLineHardeningConfig,
) -> tuple[list[PooledSemanticLine], list[str], list[str]]:
    """Pool template-local samples with a deterministic frame holdout."""

    holdout_indexes = {
        frame.frame_index
        for position, frame in enumerate(ordered, start=1)
        if position % config.frame_holdout_stride == 0
    }
    optimize_frames = [
        frame for frame in ordered if frame.frame_index not in holdout_indexes
    ]
    heldout_frames = [
        frame for frame in ordered if frame.frame_index in holdout_indexes
    ]
    if len(optimize_frames) < config.min_contributing_frames:
        return (
            [],
            list(FLOOR_LINE_IDS),
            [
                "optimize_frame_count_below_minimum:"
                f"{len(optimize_frames)}<{config.min_contributing_frames}"
            ],
        )
    if len(heldout_frames) < config.min_holdout_contributing_frames:
        return (
            [],
            list(FLOOR_LINE_IDS),
            [
                "heldout_frame_count_below_minimum:"
                f"{len(heldout_frames)}<{config.min_holdout_contributing_frames}"
            ],
        )
    pooled_lines: list[PooledSemanticLine] = []
    missing: list[str] = []
    reasons: list[str] = []
    for line_id in FLOOR_LINE_IDS:
        optimize, optimize_reason = _pool_seed_guided_sample_set(
            optimize_frames,
            line_id=line_id,
            min_frame_support=config.min_contributing_frames,
            min_line_samples=config.min_pooled_line_samples,
            max_geometry_p90=config.max_line_geometry_p90_px,
            config=config,
        )
        heldout, heldout_reason = _pool_seed_guided_sample_set(
            heldout_frames,
            line_id=line_id,
            min_frame_support=config.min_holdout_contributing_frames,
            min_line_samples=config.min_heldout_line_samples,
            max_geometry_p90=config.max_line_geometry_p90_px + 0.25,
            config=config,
        )
        if optimize is None or heldout is None:
            missing.append(line_id)
            reasons.append(
                f"{line_id}:optimize={optimize_reason};heldout={heldout_reason}"
            )
            continue
        expected = next(
            assignment.expected_segment
            for frame in ordered
            for assignment in frame.assignments
            if assignment.line_id == line_id
        )
        segment = _segment_from_pooled_samples(optimize["samples"], expected)
        heldout_segment = _segment_from_pooled_samples(
            heldout["samples"],
            expected,
        )
        optimize_indexes = tuple(
            sorted(
                {
                    index
                    for sample in optimize["samples"]
                    for index in sample.contributing_frame_indexes
                }
            )
        )
        heldout_line_indexes = tuple(
            sorted(
                {
                    index
                    for sample in heldout["samples"]
                    for index in sample.contributing_frame_indexes
                }
            )
        )
        rejected_indexes = tuple(
            sorted(
                {
                    index
                    for payload in (optimize, heldout)
                    for sample in payload["samples"]
                    for index in sample.rejected_frame_indexes
                }
            )
        )
        temporal_mads = [
            sample.temporal_mad_px
            for sample in (*optimize["samples"], *heldout["samples"])
        ]
        pooled_lines.append(
            PooledSemanticLine(
                line_id=line_id,
                segment=segment,
                expected_segment=expected,
                contributing_frame_indexes=optimize_indexes,
                heldout_frame_indexes=heldout_line_indexes,
                rejected_frame_indexes=rejected_indexes,
                heldout_segments=(heldout_segment,),
                dispersion_mad_px=float(np.median(temporal_mads)),
                optimize_samples=tuple(optimize["samples"]),
                heldout_samples=tuple(heldout["samples"]),
                geometry_fit_p90_px=float(optimize["geometry_fit_p90_px"]),
            )
        )
    pooled_lines.sort(key=lambda item: item.line_id)
    return pooled_lines, missing, reasons


def _pool_seed_guided_sample_set(
    frames: Sequence[FrameCourtLineEvidence],
    *,
    line_id: str,
    min_frame_support: int,
    min_line_samples: int,
    max_geometry_p90: float,
    config: CourtLineHardeningConfig,
) -> tuple[dict[str, Any] | None, str]:
    for frame in frames:
        for assignment in frame.assignments:
            if assignment.line_id != line_id:
                continue
            if (
                frame.provider == "seed_guided_paired_edges"
                and assignment.candidate_id.rsplit(":", 1)[-1] != line_id
            ):
                raise ValueError(
                    "seed-guided assignment candidate identity does not match "
                    f"its template sample family for frame {frame.frame_index}: "
                    f"{assignment.candidate_id}!={line_id}"
                )
    assigned = {
        frame.frame_index
        for frame in frames
        if any(assignment.line_id == line_id for assignment in frame.assignments)
    }
    by_sample: dict[
        int,
        list[tuple[FrameCourtLineEvidence, SeedGuidedPaintSample]],
    ] = {}
    for frame in frames:
        if frame.frame_index not in assigned:
            continue
        for sample in frame.template_samples:
            if sample.line_id == line_id:
                by_sample.setdefault(sample.sample_index, []).append(
                    (frame, sample)
                )
    pooled: list[PooledPaintSample] = []
    for sample_index, rows in sorted(by_sample.items()):
        if len(rows) < min_frame_support:
            continue
        reference_sample = rows[0][1]
        for _frame, sample in rows[1:]:
            if (
                not math.isclose(sample.t, reference_sample.t, abs_tol=1e-12)
                or np.max(
                    np.abs(
                        np.asarray(sample.world_xyz_m)
                        - np.asarray(reference_sample.world_xyz_m)
                    )
                )
                > 1e-12
                or np.max(
                    np.abs(
                        np.asarray(sample.seed_xy)
                        - np.asarray(reference_sample.seed_xy)
                    )
                )
                > 1e-9
                or np.max(
                    np.abs(
                        np.asarray(sample.normal_xy)
                        - np.asarray(reference_sample.normal_xy)
                    )
                )
                > 1e-9
            ):
                raise ValueError(
                    f"static sample basis changed across frames for "
                    f"{line_id}[{sample_index}]"
                )
        offsets = np.asarray(
            [sample.signed_offset_px for _frame, sample in rows],
            dtype=np.float64,
        )
        center = float(np.median(offsets))
        scaled_mad = float(
            1.4826 * np.median(np.abs(offsets - center))
        )
        if scaled_mad > config.max_static_line_mad_px:
            continue
        threshold = max(
            1.0,
            config.pool_outlier_mad_scale * max(0.15, scaled_mad),
        )
        kept = [
            row
            for row, offset in zip(rows, offsets, strict=True)
            if abs(float(offset) - center) <= threshold + 1e-12
        ]
        rejected = [
            frame.frame_index
            for (frame, _sample), offset in zip(rows, offsets, strict=True)
            if abs(float(offset) - center) > threshold + 1e-12
        ]
        if len(kept) < min_frame_support:
            continue
        kept_offsets = np.asarray(
            [sample.signed_offset_px for _frame, sample in kept],
            dtype=np.float64,
        )
        median_offset = float(np.median(kept_offsets))
        reference = kept[0][1]
        observed = np.asarray(reference.seed_xy) + median_offset * np.asarray(
            reference.normal_xy
        )
        pooled.append(
            PooledPaintSample(
                line_id=line_id,
                sample_index=sample_index,
                t=reference.t,
                world_xyz_m=reference.world_xyz_m,
                seed_xy=reference.seed_xy,
                normal_xy=reference.normal_xy,
                observed_xy=(float(observed[0]), float(observed[1])),
                signed_offset_px=median_offset,
                contributing_frame_indexes=tuple(
                    frame.frame_index for frame, _sample in kept
                ),
                contributing_frame_hashes=tuple(
                    frame.frame_sha256 for frame, _sample in kept
                ),
                rejected_frame_indexes=tuple(sorted(rejected)),
                temporal_mad_px=float(
                    1.4826
                    * np.median(np.abs(kept_offsets - median_offset))
                ),
                median_band_width_px=float(
                    np.median(
                        [sample.band_width_px for _frame, sample in kept]
                    )
                ),
                median_contrast=float(
                    np.median([sample.contrast for _frame, sample in kept])
                ),
                median_edge_strength=float(
                    np.median(
                        [sample.edge_strength for _frame, sample in kept]
                    )
                ),
            )
        )
    if len(pooled) < min_line_samples:
        return (
            None,
            f"pooled_samples_below_minimum:{len(pooled)}<{min_line_samples}",
        )
    filtered, geometry_p90 = _geometry_filter_pooled_samples(pooled)
    if len(filtered) < min_line_samples:
        return (
            None,
            f"geometry_inliers_below_minimum:{len(filtered)}<{min_line_samples}",
        )
    if geometry_p90 > max_geometry_p90:
        return (
            None,
            f"geometry_fit_p90_exceeds_max:{geometry_p90:.6f}>{max_geometry_p90:.6f}",
        )
    return (
        {
            "samples": tuple(filtered),
            "geometry_fit_p90_px": float(geometry_p90),
        },
        "accepted",
    )


def _geometry_filter_pooled_samples(
    samples: Sequence[PooledPaintSample],
) -> tuple[list[PooledPaintSample], float]:
    t = np.asarray([sample.t for sample in samples], dtype=np.float64)
    offsets = np.asarray(
        [sample.signed_offset_px for sample in samples],
        dtype=np.float64,
    )
    keep = np.ones(len(samples), dtype=bool)
    coefficients = np.polyfit(t, offsets, min(2, len(samples) - 1))
    for _iteration in range(5):
        coefficients = np.polyfit(
            t[keep],
            offsets[keep],
            min(2, int(keep.sum()) - 1),
        )
        residuals = offsets - np.polyval(coefficients, t)
        center = float(np.median(residuals[keep]))
        scaled_mad = float(
            1.4826 * np.median(np.abs(residuals[keep] - center))
        )
        limit = max(
            1.25,
            3.5 * max(0.20, scaled_mad),
        )
        next_keep = np.abs(residuals - center) <= limit
        if int(next_keep.sum()) < 5 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    coefficients = np.polyfit(
        t[keep],
        offsets[keep],
        min(2, int(keep.sum()) - 1),
    )
    residuals = offsets - np.polyval(coefficients, t)
    filtered = [
        sample
        for sample, accepted in zip(samples, keep, strict=True)
        if bool(accepted)
    ]
    p90 = (
        float(np.percentile(np.abs(residuals[keep]), 90))
        if filtered
        else math.inf
    )
    return filtered, p90


def _segment_from_pooled_samples(
    samples: Sequence[PooledPaintSample],
    expected: Segment2,
) -> Segment2:
    points = np.asarray([sample.observed_xy for sample in samples], dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, -1]
    expected_direction = np.asarray(expected[1]) - np.asarray(expected[0])
    if float(direction @ expected_direction) < 0.0:
        direction = -direction
    positions = centered @ direction
    first = center + float(np.min(positions)) * direction
    second = center + float(np.max(positions)) * direction
    return (
        (float(first[0]), float(first[1])),
        (float(second[0]), float(second[1])),
    )


def _signed_segment_offset(segment: Segment2, expected: Segment2) -> float:
    a, b, c = _line_coefficients(expected)
    midpoint = (
        0.5 * (segment[0][0] + segment[1][0]),
        0.5 * (segment[0][1] + segment[1][1]),
    )
    return float(a * midpoint[0] + b * midpoint[1] + c)


def refine_pooled_homography(
    seed_calibration: Mapping[str, Any],
    pooled: PooledCourtLineEvidence,
    *,
    config: CourtLineHardeningConfig,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Run a guarded fit with explicit line-over-point weights."""

    config.validate()
    seed_copy = deepcopy(dict(seed_calibration))
    seed_sha = _sha256_payload(seed_calibration)
    expected_config_sha = _sha256_payload(config.evidence_config_dict())
    if pooled.config_sha256 != expected_config_sha:
        raise ValueError(
            "pooled evidence config hash does not match refinement config"
        )
    if pooled.seed_calibration_sha256 != seed_sha:
        raise ValueError(
            "pooled evidence seed hash does not match refinement seed calibration"
        )
    expected_projection_sha = _sha256_payload(
        projected_floor_semantic_lines(seed_calibration)
    )
    if pooled.template_projection_sha256 != expected_projection_sha:
        raise ValueError(
            "pooled evidence template projection hash does not match refinement seed"
        )
    if pooled.image_size != _calibration_image_size(seed_calibration):
        raise ValueError("pooled evidence image size does not match seed calibration")
    if pooled.distortion_state != _distortion_state(seed_calibration):
        raise ValueError(
            "pooled evidence distortion state does not match seed calibration"
        )
    if pooled.coordinate_space != "pixels_raw_native":
        raise ValueError("court-line hardening requires raw native pooled pixels")
    if pooled.status != "accepted":
        return (
            {
                "accepted": False,
                "selection": "seed",
                "selection_reason": "pooled_evidence_abstained",
                "reject_reasons": ["pooled_evidence_abstained", *pooled.rejection_reasons],
                "objective": {
                    "line_weight": config.line_weight,
                    "point_weight": config.point_weight,
                },
                "pose_update": None,
            },
            seed_copy,
        )
    from .court_proposal_optimizer import RefinementConfig, refine_homography_with_lines

    semantic_lines: dict[str, dict[str, Any]] = {}
    for line in pooled.lines:
        semantic_lines[line.line_id] = {
            "optimize": [_segment_payload(line.segment)],
            "heldout": [_segment_payload(segment) for segment in line.heldout_segments],
            "confidence": max(0.05, 1.0 / (1.0 + line.dispersion_mad_px)),
        }
    point_prior_provenance = _point_prior_provenance(seed_calibration)
    if config.point_weight > 0.0 and not point_prior_provenance["fit_eligible"]:
        return (
            {
                "accepted": False,
                "selection": "seed",
                "selection_reason": "automatic_point_prior_provenance_required",
                "reject_reasons": [
                    "automatic_point_prior_provenance_required",
                    *point_prior_provenance["reasons"],
                ],
                "objective": {
                    "line_weight": config.line_weight,
                    "point_weight": config.point_weight,
                },
                "point_prior_provenance": point_prior_provenance,
                "pose_update": None,
            },
            seed_copy,
        )
    seed_point_priors = (
        _point_priors_from_calibration(seed_calibration)
        if config.point_weight > 0.0
        else {}
    )
    # The existing optimizer also uses point priors for initialization and
    # stability guards outside the weighted residual.  A declared line-only
    # arm must therefore remove them entirely, not merely set weight to zero.
    point_priors = (
        seed_point_priors if config.point_weight > 0.0 else {}
    )
    refinement_config = RefinementConfig(
        line_weight=config.line_weight,
        point_weight=config.point_weight,
        heldout_p90_tolerance_px=config.heldout_p90_tolerance_px,
        heldout_max_line_family_p90_regression_px=(
            config.heldout_line_family_p90_tolerance_px
        ),
    )
    raw_refinement = refine_homography_with_lines(
        seed_calibration["homography"],
        semantic_lines,
        None,
        point_priors,
        calibration=seed_calibration,
        coordinate_space=pooled.coordinate_space,
        config=refinement_config,
    )
    refinement = dict(raw_refinement)
    refinement["acceptance_policy"] = (
        "pre_registered_full_optimizer_step_only"
    )
    refinement["fit_arm_role"] = (
        "diagnostic_line_only_no_point_stability"
        if config.point_weight == 0.0
        else "preview_candidate"
    )
    refinement["promotion_eligible"] = False
    selected_alpha_raw = (
        refinement.get("telemetry", {}).get("selected_line_search_alpha")
        if isinstance(refinement.get("telemetry"), Mapping)
        else None
    )
    selected_alpha = (
        float(selected_alpha_raw)
        if isinstance(selected_alpha_raw, (int, float))
        and math.isfinite(float(selected_alpha_raw))
        else None
    )
    adapter_reject_reason = None
    if refinement.get("accepted") and config.point_weight == 0.0:
        adapter_reject_reason = (
            "diagnostic_line_only_arm_not_acceptance_eligible"
        )
    elif refinement.get("accepted") and selected_alpha != 1.0:
        adapter_reject_reason = (
            "optimizer_selected_non_preregistered_damping"
        )
    if adapter_reject_reason is not None:
        refinement["rejected_optimizer_diagnostics"] = deepcopy(
            raw_refinement
        )
        refinement["accepted"] = False
        refinement["selection"] = "seed"
        refinement["selection_reason"] = adapter_reject_reason
        refinement["reject_reasons"] = [adapter_reject_reason]
        if selected_alpha != 1.0:
            refinement["reject_reasons"].append(
                f"selected_line_search_alpha:{selected_alpha}"
            )
        refinement["homography_image_from_court"] = deepcopy(
            seed_calibration["homography"]
        )
        refinement["scores_after"] = deepcopy(
            raw_refinement.get("scores_before")
        )
        refinement["covariance_inflation_required"] = True
        refinement["robust_initialization"] = {
            "status": "not_applied",
            "reason": adapter_reject_reason,
        }
        raw_telemetry = raw_refinement.get("telemetry")
        audit_fields = (
            "semantic_line_family_count",
            "hybrid_line_segment_count",
            "provided_floor_point_count",
            "excluded_net_top_point_count",
            "distortion_present",
            "input_coordinate_space",
            "line_observation_count",
            "optimizer_point_count",
            "geometry_synthesized_point_count",
            "hybrid_intersection_point_count",
            "hybrid_band_refined_sample_count",
            "net_top_point_count_in_planar_fit",
            "coordinate_space",
            "homography_output_space",
            "opencv_version",
        )
        refinement["telemetry"] = {
            **(
                {
                    field: deepcopy(raw_telemetry[field])
                    for field in audit_fields
                    if isinstance(raw_telemetry, Mapping)
                    and field in raw_telemetry
                }
            ),
            "adapter_output": "seed",
            "selected_line_search_alpha": 0.0,
            "rejected_optimizer_selected_line_search_alpha": (
                selected_alpha
            ),
        }
        refinement["pose_update"] = None
    refinement["input_provenance"] = {
        "pooled_evidence_sha256": hashlib.sha256(pooled.canonical_bytes()).hexdigest(),
        "point_priors_sha256": _sha256_payload(point_priors),
        "point_prior_provenance": point_prior_provenance,
        "refinement_config_sha256": _sha256_payload(config.refinement_config_dict()),
    }
    candidate = deepcopy(dict(seed_calibration))
    if refinement.get("accepted"):
        candidate["homography"] = deepcopy(refinement["homography_image_from_court"])
        pose_update = refinement.get("pose_update")
        if isinstance(pose_update, Mapping):
            extrinsics = dict(candidate.get("extrinsics") or {})
            extrinsics["R"] = deepcopy(pose_update["R"])
            extrinsics["t"] = deepcopy(pose_update["t"])
            rotation = np.asarray(extrinsics["R"], dtype=np.float64)
            translation = np.asarray(extrinsics["t"], dtype=np.float64)
            camera_center = -(rotation.T @ translation)
            extrinsics["camera_height_m"] = abs(float(camera_center[2]))
            candidate["extrinsics"] = extrinsics
        invalidated_fields = [
            field
            for field in (
                "capture_quality",
                "coordinate_contract",
                *(
                    (
                        "floor_reprojection_error_px",
                        "floor_reprojection_keypoint_names",
                        "per_floor_keypoint_residual_px",
                        "per_keypoint_residual_px",
                        "reprojection_error_px",
                    )
                    if not isinstance(pose_update, Mapping)
                    else ()
                ),
                "gsd_model",
                "metric_confidence",
                "provenance",
                "solved_over_frames",
                "trust_band",
            )
            if field in candidate
        ]
        for field in invalidated_fields:
            candidate.pop(field, None)
        if not isinstance(pose_update, Mapping):
            if "extrinsics" in candidate:
                candidate.pop("extrinsics")
                invalidated_fields.append("extrinsics")
            candidate["geometry_scope"] = "planar_homography_only"
        candidate["source"] = "court_line_hardening_preview"
        candidate["authority"] = "preview"
        candidate["artifact_type"] = (
            "racketsport_court_line_homography_preview"
        )
        candidate["trust_band"] = "preview"
        candidate["verified"] = False
        candidate["upstream_seed_source"] = seed_calibration.get("source")
        candidate["invalidated_seed_fields"] = sorted(invalidated_fields)
        candidate["provenance"] = {
            "implementation": REFINEMENT_IMPLEMENTATION,
            "upstream_seed_sha256": seed_sha,
            "pooled_evidence_sha256": refinement["input_provenance"][
                "pooled_evidence_sha256"
            ],
            "point_priors_sha256": refinement["input_provenance"][
                "point_priors_sha256"
            ],
            "refinement_config_sha256": refinement["input_provenance"][
                "refinement_config_sha256"
            ],
        }
        refinement["invalidated_seed_fields"] = sorted(invalidated_fields)
        _update_candidate_reprojection(
            candidate,
            floor_only=not isinstance(pose_update, Mapping),
        )
    if _sha256_payload(seed_calibration) != seed_sha:
        raise RuntimeError("seed calibration was mutated during line refinement")
    return refinement, candidate


def _resolve_roi(
    config: CourtLineHardeningConfig,
    calibration: Mapping[str, Any],
    expected: Mapping[str, Segment2],
) -> tuple[tuple[Point2, ...], str]:
    if config.roi_polygon_px is not None:
        return tuple(config.roi_polygon_px), "declared_config"
    corners_world = get_court_template("pickleball").corners_m
    projected = _project_raw(calibration, corners_world)
    # Template corner order is near-left, near-right, far-right, far-left.
    return tuple((float(point[0]), float(point[1])) for point in projected), "seed_regulation_projection"


def _calibration_image_size(calibration: Mapping[str, Any]) -> tuple[int, int]:
    raw = calibration.get("image_size")
    if isinstance(raw, Sequence) and len(raw) == 2:
        width, height = int(round(float(raw[0]))), int(round(float(raw[1])))
        if width > 0 and height > 0:
            return width, height
    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("calibration image_size or intrinsics are required")
    width = int(round(float(intrinsics["cx"]) * 2.0))
    height = int(round(float(intrinsics["cy"]) * 2.0))
    if width <= 0 or height <= 0:
        raise ValueError("calibration image size is invalid")
    return width, height


def _distortion_state(calibration: Mapping[str, Any]) -> str:
    intrinsics = calibration.get("intrinsics")
    raw = intrinsics.get("dist", []) if isinstance(intrinsics, Mapping) else []
    nonzero = any(not math.isclose(float(value), 0.0, abs_tol=1e-12) for value in raw)
    return "raw_distorted_with_declared_model" if nonzero else "raw_pinhole"


def _project_raw(
    calibration: Mapping[str, Any],
    world_points: Sequence[Sequence[float]],
) -> np.ndarray:
    intrinsics = calibration.get("intrinsics")
    extrinsics = calibration.get("extrinsics")
    if (
        _distortion_state(calibration) == "raw_distorted_with_declared_model"
        and isinstance(intrinsics, Mapping)
        and isinstance(extrinsics, Mapping)
    ):
        import cv2

        camera = np.asarray(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(extrinsics["R"], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(rotation)
        translation = np.asarray(extrinsics["t"], dtype=np.float64).reshape(3, 1)
        distortion = np.asarray(intrinsics.get("dist", []), dtype=np.float64)
        world = np.asarray(
            [
                [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
                for point in world_points
            ],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(world, rvec, translation, camera, distortion)
        return projected.reshape(-1, 2)
    return np.asarray(
        project_planar_points(calibration["homography"], world_points),
        dtype=np.float64,
    )


def _point_priors_from_calibration(calibration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    image_points = calibration.get("image_pts")
    if not isinstance(image_points, Sequence):
        return {}
    if len(image_points) != len(PICKLEBALL_COURT_KEYPOINT_NAMES):
        return {}
    return {
        name: {
            "xy": [float(image_points[index][0]), float(image_points[index][1])],
            "confidence": 1.0,
        }
        for index, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
    }


def _point_prior_provenance(
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject scorer-only/manual correspondences from the candidate fit."""

    source = str(calibration.get("source") or "").strip()
    declared = calibration.get("point_evidence_provenance")
    lowered = source.lower()
    disallowed_tokens = (
        "reviewed",
        "manual",
        "oracle",
        "ground_truth",
        "ground-truth",
        "metric_15pt",
        "label",
    )
    reasons: list[str] = []
    if not source:
        reasons.append("point_prior_source_missing")
    matched = sorted(token for token in disallowed_tokens if token in lowered)
    if matched:
        reasons.append(
            "point_prior_source_is_scorer_or_manual:" + ",".join(matched)
        )
    declared_source: str | None = None
    declared_hash: str | None = None
    declared_correspondence_hash: str | None = None
    declared_authority: str | None = None
    if not isinstance(declared, Mapping):
        reasons.append("automatic_point_evidence_provenance_missing")
    else:
        declared_source = str(declared.get("source") or "").strip() or None
        declared_hash = str(declared.get("artifact_sha256") or "").lower() or None
        declared_correspondence_hash = (
            str(declared.get("correspondences_sha256") or "").lower() or None
        )
        declared_authority = (
            str(declared.get("authority") or "").strip().lower() or None
        )
        if declared_authority != "automatic":
            reasons.append(
                "point_evidence_authority_not_automatic:"
                f"{declared_authority or 'missing'}"
            )
        if declared_source is None:
            reasons.append("point_evidence_source_missing")
        elif any(token in declared_source.lower() for token in disallowed_tokens):
            reasons.append("point_evidence_source_is_scorer_or_manual")
        elif source and declared_source != source:
            reasons.append("point_evidence_source_does_not_match_seed_source")
        if (
            declared_hash is None
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdef" for character in declared_hash)
        ):
            reasons.append("point_evidence_artifact_sha256_invalid")
        if (
            declared_correspondence_hash is None
            or len(declared_correspondence_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in declared_correspondence_hash
            )
        ):
            reasons.append("point_evidence_correspondences_sha256_invalid")

    expected_count = len(PICKLEBALL_COURT_KEYPOINT_NAMES)
    model_by_name = {
        point.name: point.world_xyz_m for point in PICKLEBALL_KEYPOINTS
    }
    if set(model_by_name) != set(PICKLEBALL_COURT_KEYPOINT_NAMES):
        reasons.append("internal_keypoint_schema_model_mismatch")
    image_points = calibration.get("image_pts")
    world_points = calibration.get("world_pts")
    valid_image_count = 0
    if not _is_point_sequence(image_points):
        reasons.append("point_prior_image_points_not_a_sequence")
    elif len(image_points) != expected_count:
        reasons.append(
            "point_prior_correspondence_count_invalid:"
            f"{len(image_points)}!={expected_count}"
        )
    else:
        for index, (name, point) in enumerate(
            zip(
                PICKLEBALL_COURT_KEYPOINT_NAMES,
                image_points,
                strict=True,
            )
        ):
            if not _is_point_sequence(point) or len(point) != 2:
                reasons.append(
                    f"point_prior_image_shape_invalid:{index}:{name}"
                )
                continue
            try:
                values = tuple(float(value) for value in point)
            except (TypeError, ValueError):
                reasons.append(
                    f"point_prior_image_value_invalid:{index}:{name}"
                )
                continue
            if not all(math.isfinite(value) for value in values):
                reasons.append(
                    f"point_prior_image_nonfinite:{index}:{name}"
                )
                continue
            valid_image_count += 1

    if not _is_point_sequence(world_points):
        reasons.append("point_prior_world_points_not_a_sequence")
    elif len(world_points) != expected_count:
        reasons.append(
            "point_prior_world_correspondence_count_invalid:"
            f"{len(world_points)}!={expected_count}"
        )
    else:
        for index, (name, point) in enumerate(
            zip(
                PICKLEBALL_COURT_KEYPOINT_NAMES,
                world_points,
                strict=True,
            )
        ):
            if not _is_point_sequence(point) or len(point) != 3:
                reasons.append(
                    f"point_prior_world_shape_invalid:{index}:{name}"
                )
                continue
            try:
                values = tuple(float(value) for value in point)
            except (TypeError, ValueError):
                reasons.append(
                    f"point_prior_world_value_invalid:{index}:{name}"
                )
                continue
            if not all(math.isfinite(value) for value in values):
                reasons.append(
                    f"point_prior_world_nonfinite:{index}:{name}"
                )
                continue
            expected_world = model_by_name.get(name)
            if expected_world is None or any(
                not math.isclose(
                    actual,
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for actual, expected in zip(
                    values,
                    expected_world or (),
                    strict=True,
                )
            ):
                reasons.append(
                    "point_prior_world_order_or_value_mismatch:"
                    f"{index}:{name}"
                )

    try:
        actual_correspondence_hash = _sha256_payload(
            {
                "image_pts": image_points,
                "world_pts": world_points,
            }
        )
    except (TypeError, ValueError):
        actual_correspondence_hash = None
        reasons.append("point_prior_correspondences_not_strict_json")
    if (
        actual_correspondence_hash is None
        or declared_correspondence_hash != actual_correspondence_hash
    ):
        reasons.append("point_evidence_correspondence_hash_mismatch")
    return {
        "source": source or None,
        "declared_source": declared_source,
        "declared_authority": declared_authority,
        "artifact_sha256": declared_hash,
        "correspondences_sha256": declared_correspondence_hash,
        "actual_correspondences_sha256": actual_correspondence_hash,
        "fit_eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "point_count": valid_image_count,
        "policy": "automatic_seed_correspondences_only_no_reviewed_labels",
    }


def _is_point_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _update_candidate_reprojection(
    candidate: dict[str, Any],
    *,
    floor_only: bool,
) -> None:
    image_points = candidate.get("image_pts")
    world_points = candidate.get("world_pts")
    if not isinstance(image_points, Sequence) or not isinstance(world_points, Sequence):
        return
    selected_image_points = image_points
    selected_world_points = world_points
    selected_names = list(PICKLEBALL_COURT_KEYPOINT_NAMES)
    if floor_only:
        model_by_name = {
            point.name: point.world_xyz_m for point in PICKLEBALL_KEYPOINTS
        }
        floor_indexes = [
            index
            for index, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
            if name in model_by_name
            and math.isclose(
                float(model_by_name[name][2]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        selected_image_points = [
            image_points[index] for index in floor_indexes
        ]
        selected_world_points = [
            world_points[index] for index in floor_indexes
        ]
        selected_names = [
            PICKLEBALL_COURT_KEYPOINT_NAMES[index]
            for index in floor_indexes
        ]
    projected = _project_raw(candidate, selected_world_points)
    error = reprojection_error(
        selected_image_points,
        projected.tolist(),
    )
    residuals = [
        math.hypot(
            float(observed[0]) - float(predicted[0]),
            float(observed[1]) - float(predicted[1]),
        )
        for observed, predicted in zip(
            selected_image_points,
            projected,
            strict=True,
        )
    ]
    normalized_residuals = [
        0.0 if value < 1e-9 else float(value)
        for value in residuals
    ]
    if floor_only:
        candidate["floor_reprojection_error_px"] = error.model_dump(
            mode="json"
        )
        candidate["floor_reprojection_keypoint_names"] = selected_names
        candidate["per_floor_keypoint_residual_px"] = (
            normalized_residuals
        )
        return
    candidate["reprojection_error_px"] = error.model_dump(mode="json")
    candidate["per_keypoint_residual_px"] = normalized_residuals


def _coerce_candidate(
    raw: DetectedCourtLineCandidate | Mapping[str, Any],
    fallback_index: int,
) -> DetectedCourtLineCandidate:
    if isinstance(raw, DetectedCourtLineCandidate):
        return raw
    endpoints_raw = raw.get("endpoints")
    if endpoints_raw is None and "p1" in raw and "p2" in raw:
        endpoints_raw = [raw["p1"], raw["p2"]]
    endpoints = _coerce_segment(endpoints_raw, f"candidates[{fallback_index}].endpoints")
    support = float(raw.get("support_length_px", _segment_length(endpoints)))
    if not math.isfinite(support) or support <= 0.0:
        raise ValueError(f"candidates[{fallback_index}] support must be positive")
    angle = float(raw.get("angle_deg", _segment_angle_deg(endpoints)))
    return DetectedCourtLineCandidate(
        candidate_id=str(raw.get("candidate_id", f"candidate:{fallback_index:04d}")),
        endpoints=endpoints,
        support_length_px=support,
        source_segment_count=int(raw.get("source_segment_count", 1)),
        angle_deg=angle,
        provider=str(raw.get("provider", "external")),
        preprocessing=str(raw.get("preprocessing", "raw")),
        source_candidate_ids=tuple(
            str(value) for value in raw.get("source_candidate_ids", ())
        ),
    )


def _coerce_point2(raw: Any, field: str) -> Point2:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != 2
    ):
        raise ValueError(f"{field} must be an xy pair")
    point = (float(raw[0]), float(raw[1]))
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{field} must be finite")
    return point


def _coerce_point3(raw: Any, field: str) -> tuple[float, float, float]:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != 3
    ):
        raise ValueError(f"{field} must be an xyz triple")
    point = (float(raw[0]), float(raw[1]), float(raw[2]))
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{field} must be finite")
    return point


def _coerce_segment(raw: Any, field: str) -> Segment2:
    if not isinstance(raw, Sequence) or len(raw) != 2:
        raise ValueError(f"{field} must contain two points")
    points: list[Point2] = []
    for index, point in enumerate(raw):
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"{field}[{index}] must be an xy pair")
        xy = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in xy):
            raise ValueError(f"{field}[{index}] must be finite")
        points.append(xy)
    segment = (points[0], points[1])
    if _segment_length(segment) <= 1e-9:
        raise ValueError(f"{field} must have non-zero length")
    return segment


def _segment_payload(segment: Segment2) -> list[list[float]]:
    return [[float(value) for value in point] for point in segment]


def _line_coefficients(segment: Segment2) -> tuple[float, float, float]:
    (x1, y1), (x2, y2) = segment
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        raise ValueError("line segment must have non-zero length")
    a, b = -dy / length, dx / length
    c = -(a * x1 + b * y1)
    if a < -1e-12 or (abs(a) <= 1e-12 and b < 0.0):
        a, b, c = -a, -b, -c
    return a, b, c


def _project_point_to_line(point: Point2, segment: Segment2) -> Point2:
    a, b, c = _line_coefficients(segment)
    distance = a * point[0] + b * point[1] + c
    return point[0] - distance * a, point[1] - distance * b


def _median_projected_segment(segments: Sequence[Segment2], expected: Segment2) -> Segment2:
    projected_first = [_project_point_to_line(expected[0], segment) for segment in segments]
    projected_second = [_project_point_to_line(expected[1], segment) for segment in segments]
    first = np.median(np.asarray(projected_first, dtype=np.float64), axis=0)
    second = np.median(np.asarray(projected_second, dtype=np.float64), axis=0)
    return (
        (float(first[0]), float(first[1])),
        (float(second[0]), float(second[1])),
    )


def _symmetric_line_distance(first: Segment2, second: Segment2) -> float:
    first_line = _line_coefficients(first)
    second_line = _line_coefficients(second)
    distances = [
        abs(first_line[0] * point[0] + first_line[1] * point[1] + first_line[2])
        for point in second
    ]
    distances.extend(
        abs(second_line[0] * point[0] + second_line[1] * point[1] + second_line[2])
        for point in first
    )
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _axial_angle_delta(first: Segment2, second: Segment2) -> float:
    first_angle = _segment_angle_deg(first)
    second_angle = _segment_angle_deg(second)
    delta = abs(first_angle - second_angle) % 180.0
    return min(delta, 180.0 - delta)


def _segment_angle_deg(segment: Segment2) -> float:
    return math.degrees(
        math.atan2(
            segment[1][1] - segment[0][1],
            segment[1][0] - segment[0][0],
        )
    )


def _segment_overlap_fraction(candidate: Segment2, expected: Segment2) -> float:
    x1, y1 = expected[0]
    dx = expected[1][0] - x1
    dy = expected[1][1] - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return 0.0
    values = [
        ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_sq
        for point in candidate
    ]
    low = max(0.0, min(values))
    high = min(1.0, max(values))
    return max(0.0, high - low)


def _point_in_or_near_polygon(
    point: Point2,
    polygon: Sequence[Point2],
    *,
    margin: float,
) -> bool:
    if _point_in_polygon(point, polygon):
        return True
    return min(_point_segment_distance(point, (polygon[index], polygon[(index + 1) % len(polygon)]))
               for index in range(len(polygon))) <= margin


def _segment_roi_support_fraction(
    segment: Segment2,
    polygon: Sequence[Point2],
    *,
    margin: float,
    sample_count: int = 33,
) -> float:
    """Approximate how much of a candidate's own extent lies in the court ROI."""

    first = np.asarray(segment[0], dtype=np.float64)
    second = np.asarray(segment[1], dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    supported = sum(
        _point_in_or_near_polygon(
            (float(point[0]), float(point[1])),
            polygon,
            margin=margin,
        )
        for point in (
            first[None, :] + fractions[:, None] * (second - first)[None, :]
        )
    )
    return supported / float(sample_count)


def _point_in_polygon(point: Point2, polygon: Sequence[Point2]) -> bool:
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            # ``intersects`` guarantees a non-zero vertical delta.  Preserve
            # its sign: clamping a descending edge to a positive epsilon
            # mirrors its crossing and makes the answer depend on winding.
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(point: Point2, segment: Segment2) -> float:
    p = np.asarray(point, dtype=np.float64)
    first = np.asarray(segment[0], dtype=np.float64)
    second = np.asarray(segment[1], dtype=np.float64)
    delta = second - first
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        return float(np.linalg.norm(p - first))
    t = max(0.0, min(1.0, float((p - first) @ delta) / denominator))
    return float(np.linalg.norm(p - (first + t * delta)))


def _segment_endpoint_delta(first: Segment2, second: Segment2) -> float:
    direct = max(math.dist(first[0], second[0]), math.dist(first[1], second[1]))
    reversed_delta = max(math.dist(first[0], second[1]), math.dist(first[1], second[0]))
    return min(direct, reversed_delta)


def _segment_length(segment: Segment2) -> float:
    return math.dist(segment[0], segment[1])


def _frame_sha256(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)).cast("B"))
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
