"""Fail-closed racket-sport pipeline orchestration."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from scripts.racketsport.track import build_tracks

from .ball_stage_runner import BallStageRunner
from .best_stack import body_detector_fov_defaults
from .body_compute import (
    TIER2_BODY_JOINTS_REPRESENTATION,
    build_body_compute_execution,
    build_empty_body_compute_execution,
    body_frame_batches_from_execution,
    write_body_compute_execution,
)
from .body_full_clip_gate import build_body_full_clip_gate
from .body_grounding_quality import build_body_grounding_quality, write_body_grounding_quality
from .body_joint_quality import build_body_joint_quality
from .body_array_native import (
    body_mesh_export_parts_from_smpl_motion_view,
    body_mesh_payload_from_parts,
    build_body_array_native_artifacts_from_fast_sam,
)
from .body_mesh_index import build_body_mesh_index_from_arrays, build_body_mesh_index_from_payload
from .body_mesh_readiness import build_body_mesh_readiness
from .body_postchain import BodyPostChainConfig, RAW_GROUNDED_JOINTS_ARTIFACT
from .ball_inflections import build_ball_inflections_from_ball_track
from .court_auto_evidence import build_auto_court_line_evidence_from_frame, build_auto_court_line_evidence_from_video
from .court_calibration import (
    CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
    calibration_from_manual_taps,
    calibration_image_size,
    metric_calibration_from_sidecar_and_keypoints,
)
from .court_line_evidence import aggregate_court_line_evidence, required_court_line_ids, required_court_net_ids
from .court_templates import Sport
from .court_zones import build_court_zones
from .detection_scaling import scale_detection_payload_bboxes
from .event_fusion import fuse_contact_windows_from_cue_payloads
from .frame_rating import build_frame_compute_plan, build_frame_compute_plan_from_files, write_frame_compute_plan
from .contact_splice import splice_contact_skeleton_with_body_mesh
from .hmr_deep import (
    DEFAULT_BODY_MANIFEST_PATH,
    DEFAULT_FAST_SAM_REPO,
    REQUIRED_FAST_SAM_MODEL_IDS,
    SAM3D_FOOT_KEYPOINT_INDICES,
    FastSam3DBodyRuntime,
    FastSam3DBodySubprocessRuntime,
    PlayerCropRequest,
    fast_sam_required_model_ids,
    normalize_fast_sam_body_output,
    verify_fast_sam_manifest_assets,
)
from .model_manifest import verify_model_checkpoint
from .net_plane import build_net_plane
from .pipeline_contracts import PIPELINE_STAGE_CONTRACTS, PipelineContractError, PipelineStageContract, build_readiness_report
from . import mesh_export as _mesh_export
from .mesh_export import build_body_mesh_export
from .pose_temporal import apply_sam3d_wrist_bone_lock
from .racket_stage_runner import RacketStageRunner
from .sam3d_body_input_prep import (
    ACCURACY_OPT_SOURCE,
    load_mask_prompt_lookup,
    normalize_body_input_size,
    normalize_crop_padding_scale,
    normalize_soft_background_alpha,
    padded_bbox_xyxy,
    request_prep_artifact,
    static_camera_intrinsics_k,
    write_soft_background_image,
)
from .schemas import CaptureSidecar, CourtCalibration, StrictArtifact, Tracks, validate_artifact_file
from .virtual_world import build_virtual_world_state_from_files, write_virtual_world
from .wrist_velocity_peaks import build_wrist_velocity_peaks_from_file
from .worldhmr import assemble_body_monolith_payloads, build_body_artifacts_from_fast_sam, compute_body_skeleton_and_metrics


YOLO26M_MODEL_ID = "yolo26m"
DEFAULT_MODEL_MANIFEST = Path("models/MANIFEST.json")
DEFAULT_BODY_MODEL_MANIFEST = DEFAULT_BODY_MANIFEST_PATH
DEFAULT_BOTSORT_REID_CONFIG = Path("configs/racketsport/botsort_reid.yaml")
DEFAULT_BODY_DETECTOR_NAME, DEFAULT_BODY_FOV_NAME = body_detector_fov_defaults()
DEFAULT_BODY_MAX_ROOT_SPEED_MPS = 8.0
DEFAULT_BODY_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M = 0.75
R3_GROUNDING_ANCHOR_SOURCE = "placement_track_world_xy"
DEFAULT_GROUNDING_ANCHOR_SOURCE = "track_world_xy"
FAST_SAM_PYTHON_ENV = "FAST_SAM_PYTHON"
FAST_SAM_REQUIRE_SUBPROCESS_ENV = "FAST_SAM_REQUIRE_SUBPROCESS"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
BODY_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")
SAM3D_UPSTREAM_ENV_WHITELIST = frozenset(
    {
        "USE_COMPILE_BACKBONE",
        "DECODER_COMPILE",
        "INTERM_COMPILE",
        "INTERM_SLIM",
        "COMPILE_MODE",
        "MHR_NO_CORRECTIVES",
    }
)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

ARTIFACT_SCHEMA_BY_FILENAME: dict[str, str] = {
    "court_calibration.json": "court_calibration",
    "court_line_evidence.json": "court_line_evidence",
    "court_zones.json": "court_zones",
    "net_plane.json": "net_plane",
    "court_keypoints.json": "court_keypoints",
    "tracks.json": "tracks",
    "player_ground.json": "player_ground",
    "calls.json": "court_calls",
    "drift_log.json": "drift_log",
    "smpl_motion.json": "smpl_motion",
    "skeleton3d.json": "skeleton3d",
    "body_compute_execution.json": "body_compute_execution",
    "body_serialization_timing.json": "body_serialization_timing",
    "body_stage_phase_timing.json": "body_stage_phase_timing",
    "body_mesh_readiness.json": "body_mesh_readiness",
    "ball_track.json": "ball_track",
    "contact_windows.json": "contact_windows",
    "racket_pose.json": "racket_pose",
    "racket_pose_readiness.json": "racket_pose_readiness",
    "racket_promotion_audit.json": "racket_promotion_audit",
    "virtual_world.json": "virtual_world",
    "racket_sport_metrics.json": "racket_sport_metrics",
    "habit_report.json": "habit_report",
    "coach_report.json": "coach_report",
    "physics_refinement.json": "physics_refinement",
    "drill_report.json": "drill_report",
    "replay_scene.json": "replay_scene",
    "pipeline_run.json": "pipeline_run",
}

PIPELINE_STATUS_PASS = "pass"
PIPELINE_STATUS_FAIL = "fail"
PIPELINE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class StageContext:
    clip: str
    inputs_dir: Path
    run_dir: Path
    sport: Sport
    device: str | None = None
    max_frames: int | None = None
    expected_players: int = 4
    tracking_video: Path | None = None
    ball_source_path: Path | None = None


@dataclass(frozen=True)
class StageRun:
    stage: str
    status: str
    real_model: bool
    source_mode: str
    produced_artifacts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "status": self.status,
            "real_model": self.real_model,
            "source_mode": self.source_mode,
            "produced_artifacts": list(self.produced_artifacts),
            "notes": list(self.notes),
            "metrics": self.metrics,
        }
        if self.wall_seconds is not None:
            payload["wall_seconds"] = self.wall_seconds
        return payload


class StageRunner(Protocol):
    stage: str
    real_model: bool
    source_mode: str

    def run(self, context: StageContext) -> StageRun:
        """Run one stage and write its artifacts."""


class ManualCalibrationRunner:
    stage = "calibration"
    real_model = False
    source_mode = "manual_sidecar"

    def run(self, context: StageContext) -> StageRun:
        sidecar_path = context.inputs_dir / "capture_sidecar.json"
        net_plane = build_net_plane(context.sport)
        if not sidecar_path.is_file():
            line_evidence, evidence_notes = _unseeded_calibration_line_evidence(context)
            artifacts = {
                "court_zones.json": build_court_zones(context.sport),
                "net_plane.json": net_plane,
                "court_line_evidence.json": line_evidence,
            }
            for filename, artifact in artifacts.items():
                _write_json_artifact(context.run_dir / filename, artifact)
            note = "; ".join(evidence_notes)
            raise FileNotFoundError(
                f"missing calibration sidecar: {sidecar_path}; {note}; no trusted no-tap calibration seed"
            )

        court_keypoints_path = context.inputs_dir / "court_keypoints.json"
        if court_keypoints_path.is_file():
            calibration = metric_calibration_from_sidecar_and_keypoints(
                sidecar_path,
                court_keypoints_path,
                sport=context.sport,
            )
            source_mode = "arkit_plane_keypoints"
            calibration_note = "no-tap ARKit floor-plane calibration from court_keypoints.json"
            # No-tap ARKit-only calibration is intentionally NOT treated as a trusted
            # source for the automatic evidence gate below (Task #45 S1): the product's
            # current v1 path is owner-tap-driven, and this fully-automatic no-tap path
            # is the future-facing capability the evidence gate itself is designed for --
            # leave it fail-closed exactly as before rather than assuming trust here too.
            trusted = False
        else:
            calibration = calibration_from_manual_taps(sidecar_path, sport=context.sport)
            capture_quality = calibration.capture_quality
            quality_reasons = {str(reason) for reason in (capture_quality.reasons or [])}
            auto_seeded = bool(
                {
                    "process_video_auto_court_corners_preview",
                    "manual_taps_seeded_from_unverified_detector",
                }
                & quality_reasons
            )
            source_mode = "auto_preview_sidecar" if auto_seeded else self.source_mode
            calibration_note = (
                "auto-predicted 4-corner calibration seed; unverified preview, not human-reviewed"
                if auto_seeded
                else "manual 4-corner calibration seed; requires human-reviewed corners for product verification"
            )
            # Product intake must be able to produce a preview replay from an automatic
            # court seed. "trusted" here only controls whether weak automatic line
            # evidence blocks the calibration stage; the calibration itself remains
            # low-confidence via capture_quality/trust-band metadata and is not a
            # metric-15pt or training-ready source.
            trusted = True
        line_evidence, evidence_notes = _calibration_line_evidence(context, calibration=calibration, net_plane=net_plane)
        artifacts = {
            "court_calibration.json": calibration,
            "court_zones.json": build_court_zones(context.sport),
            "net_plane.json": net_plane,
            "court_line_evidence.json": line_evidence,
        }
        for filename, artifact in artifacts.items():
            _write_json_artifact(context.run_dir / filename, artifact)
        advisory_note = _raise_if_video_evidence_not_ready(context, line_evidence, trusted=trusted)

        notes = (calibration_note, *evidence_notes)
        metrics: dict[str, Any] = {
            "reprojection_median_px": calibration.reprojection_error_px.median,
            "reprojection_p95_px": calibration.reprojection_error_px.p95,
        }
        if advisory_note:
            notes = (*notes, advisory_note)
            metrics["calibration_confidence"] = _calibration_confidence_proxy(calibration)

        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=source_mode,
            produced_artifacts=tuple(artifacts),
            notes=notes,
            metrics=metrics,
        )


class ExternalCalibrationRunner:
    """Consumes an already-solved ``court_calibration.json``-shaped artifact instead of
    re-deriving one from ``capture_sidecar.json`` manual taps / ARKit keypoints.

    This is the seam Task #33 (CAL-MIGRATION) wires ``process_video.py --court-calibration``
    into: e.g. the metric-15pt reviewed calibration from
    ``threed.racketsport.court_calibration_metric15.metric_calibration_from_reviewed_keypoints_15pt``,
    which fits real (fx, fy, k1, k2) intrinsics + a `solvePnP` pose from 15 reviewed court
    keypoints instead of the guessed-intrinsics 4-corner PnP `ManualCalibrationRunner` falls
    back to. Only calibrations whose ``intrinsics.source`` is in ``trusted_intrinsics_sources``
    are accepted -- this runner exists specifically so a reviewed, real-focal-length
    calibration can bypass the guessed-intrinsics path, not so any arbitrary
    externally-produced calibration can skip validation.

    court_zones.json/net_plane.json are pure sport-template artifacts (independent of how
    the calibration was solved) and are always regenerated here. court_line_evidence.json
    still runs the same real automatic video/frame line+net detection
    (``_calibration_line_evidence``) and the same fail-closed-if-not-ready gate
    (``_raise_if_video_evidence_not_ready``) that ``ManualCalibrationRunner`` uses, so a
    video-backed run with unreadable court lines still hard-fails here exactly as it would
    on the manual-taps path -- consuming an external calibration does not weaken that gate.
    """

    stage = "calibration"
    real_model = False
    source_mode = "external_metric_calibration"

    #: intrinsics.source values accepted as already-reviewed/real (never a guessed-focal
    #: source like "estimated_from_declared_court_corners") -- see court_calibration_metric15.py.
    TRUSTED_INTRINSICS_SOURCES = frozenset({"metric_15pt_reviewed"})

    def __init__(self, *, source_path: str | Path, trusted_intrinsics_sources: frozenset[str] | None = None) -> None:
        self.source_path = Path(source_path)
        self.trusted_intrinsics_sources = trusted_intrinsics_sources or self.TRUSTED_INTRINSICS_SOURCES

    def run(self, context: StageContext) -> StageRun:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"--court-calibration artifact not found: {self.source_path}")

        calibration = validate_artifact_file("court_calibration", self.source_path)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError(f"{self.source_path} did not validate as CourtCalibration")
        if calibration.sport != context.sport:
            raise ValueError(
                f"{self.source_path}: calibration.sport={calibration.sport!r} does not match "
                f"the requested sport={context.sport!r}"
            )
        source = calibration.intrinsics.source
        if source not in self.trusted_intrinsics_sources:
            raise ValueError(
                f"{self.source_path}: intrinsics.source={source!r} is not a trusted external calibration "
                f"source (expected one of {sorted(self.trusted_intrinsics_sources)}); refusing to consume an "
                "unreviewed/guessed calibration through --court-calibration"
            )

        net_plane = build_net_plane(context.sport)
        artifacts: dict[str, Any] = {
            "court_calibration.json": calibration,
            "court_zones.json": build_court_zones(context.sport),
            "net_plane.json": net_plane,
        }
        for filename, artifact in artifacts.items():
            _write_json_artifact(context.run_dir / filename, artifact)

        line_evidence, evidence_notes = _calibration_line_evidence(context, calibration=calibration, net_plane=net_plane)
        _write_json_artifact(context.run_dir / "court_line_evidence.json", line_evidence)
        artifacts["court_line_evidence.json"] = line_evidence
        # Task #45 S1: `source` has already been checked above to be in
        # `self.trusted_intrinsics_sources` (metric_15pt_reviewed by default) -- any
        # calibration that reaches this point is, by construction, trusted. So a
        # not-ready automatic evidence result downgrades to an advisory note instead of
        # hard-failing the stage; an untrusted source never gets here at all (it already
        # raised above), so that path's fail-closed behavior is completely unchanged.
        advisory_note = _raise_if_video_evidence_not_ready(context, line_evidence, trusted=True)

        dist = [float(value) for value in (calibration.intrinsics.dist or [])]
        notes = [
            f"consumed externally-provided court_calibration.json from {self.source_path} "
            f"(intrinsics.source={source!r}); PnP/homography re-derivation from manual taps skipped",
            *evidence_notes,
        ]
        if any(abs(value) > 1e-9 for value in dist):
            notes.append(
                f"intrinsics.dist is nonzero ({dist}): homography-based footpoint consumers must undistort "
                "pixels before applying calibration.homography or they will carry a systematic, "
                "distortion-shaped error -- see threed/racketsport/court_calibration_metric15.py's migration "
                "notes and runs/cal_metric_15pt_20260702T041729Z/ for the measured before/after."
            )
        metrics: dict[str, Any] = {
            "reprojection_median_px": calibration.reprojection_error_px.median,
            "reprojection_p95_px": calibration.reprojection_error_px.p95,
        }
        if advisory_note:
            notes.append(advisory_note)
            metrics["calibration_confidence"] = _calibration_confidence_proxy(calibration)

        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=tuple(artifacts),
            notes=tuple(notes),
            metrics={
                **metrics,
                "intrinsics_source": source,
                "intrinsics_dist_nonzero": any(abs(value) > 1e-9 for value in dist),
            },
        )


class PrecomputedTrackingRunner:
    stage = "tracking"
    real_model = False
    source_mode = "precomputed_detections"

    def __init__(self, *, max_players: int = 4, court_margin_m: float = 0.0, id_strategy: str = "auto") -> None:
        self.max_players = max_players
        self.court_margin_m = court_margin_m
        self.id_strategy = id_strategy

    def run(self, context: StageContext) -> StageRun:
        detections_path = context.inputs_dir / "detections.json"
        calibration_path = context.run_dir / "court_calibration.json"
        if not detections_path.is_file():
            raise FileNotFoundError(f"missing detections: {detections_path}")
        calibration = validate_artifact_file("court_calibration", calibration_path)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        detections_payload = _read_json(detections_path)
        tracks, counts = build_tracks(
            detections_payload,
            calibration,
            max_step_m=2.0,
            max_players=self.max_players,
            court_margin_m=self.court_margin_m,
            id_strategy=self.id_strategy,  # type: ignore[arg-type]
        )
        if counts["accepted"] <= 0 or not tracks.players:
            raise ValueError(f"tracking failed: no accepted on-court person tracks; counts={counts}")

        _write_json_artifact(context.run_dir / "tracks.json", tracks)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("tracks.json",),
            notes=(
                "uses precomputed detections; not a GPU model invocation",
            ),
            metrics=counts,
        )


class PrecomputedTracksRunner:
    stage = "tracking"
    real_model = False
    source_mode = "precomputed_tracks"

    def run(self, context: StageContext) -> StageRun:
        tracks_path = context.inputs_dir / "tracks.json"
        if not tracks_path.is_file():
            raise FileNotFoundError(f"missing tracks: {tracks_path}")
        tracks = validate_artifact_file("tracks", tracks_path)
        if not isinstance(tracks, Tracks):
            raise ValueError("precomputed tracks.json did not validate as Tracks")
        _write_json_artifact(context.run_dir / "tracks.json", tracks)
        track_frame_count = sum(len(player.frames) for player in tracks.players)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("tracks.json",),
            notes=(
                "uses precomputed tracks.json; not a detector or tracker invocation",
            ),
            metrics={
                "output_players": len(tracks.players),
                "track_frame_count": track_frame_count,
                "tracks_fps": float(tracks.fps),
            },
        )


class RealYOLO26BoTSORTReIDTrackingRunner:
    stage = "tracking"
    real_model = True
    source_mode = "yolo26m_botsort_reid"

    def __init__(
        self,
        *,
        manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
        tracker_config_path: str | Path = DEFAULT_BOTSORT_REID_CONFIG,
        video_path: str | Path | None = None,
        imgsz: int = 1536,
        conf: float = 0.05,
        iou: float = 0.6,
        max_step_m: float = 2.0,
        max_players: int = 4,
        court_margin_m: float = 0.0,
        id_strategy: str = "auto",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.tracker_config_path = Path(tracker_config_path)
        self.video_path = Path(video_path) if video_path is not None else None
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_step_m = max_step_m
        self.max_players = max_players
        self.court_margin_m = court_margin_m
        self.id_strategy = id_strategy

    def run(self, context: StageContext) -> StageRun:
        calibration = validate_artifact_file("court_calibration", context.run_dir / "court_calibration.json")
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        tracker_config = _verified_botsort_reid_config(self.tracker_config_path)
        video_path = _tracking_video_path(context, explicit=self.video_path)
        fps = _tracking_fps(context.inputs_dir, video_path=video_path)
        checkpoint_entry = verify_model_checkpoint(self.manifest_path, YOLO26M_MODEL_ID)
        checkpoint = Path(str(checkpoint_entry.local_path))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for real YOLO26m BoT-SORT-ReID tracking") from exc

        model = YOLO(str(checkpoint))
        results = model.track(
            source=str(video_path),
            tracker=str(tracker_config),
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=context.device,
            stream=True,
            persist=False,
            verbose=False,
        )
        try:
            raw_detections_payload, raw_counts = _detections_payload_from_tracked_results(
                results,
                fps=fps,
                max_frames=context.max_frames,
            )
        finally:
            close = getattr(results, "close", None)
            if callable(close):
                close()
        scale_x, scale_y, scale_counts = _detection_bbox_scale(
            calibration,
            video_path,
            source_size=_payload_source_size(raw_detections_payload),
        )
        detections_payload = scale_detection_payload_bboxes(raw_detections_payload, scale_x=scale_x, scale_y=scale_y)
        tracks, counts = build_tracks(
            detections_payload,
            calibration,
            max_step_m=self.max_step_m,
            max_players=self.max_players,
            court_margin_m=self.court_margin_m,
            id_strategy=self.id_strategy,  # type: ignore[arg-type]
        )
        if counts["accepted"] <= 0 or not tracks.players:
            raise ValueError(f"real tracking failed: no accepted on-court tracked person boxes; counts={counts | raw_counts}")

        tracking_counts = {**counts, **raw_counts, **scale_counts}
        metrics_payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_person_tracker_candidate",
            "clip": context.clip,
            "source_mode": self.source_mode,
            "model": str(checkpoint),
            "tracker_config": str(tracker_config),
            "source_video": str(video_path),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_frames": context.max_frames,
            "max_players": self.max_players,
            "court_margin_m": self.court_margin_m,
            "id_strategy": self.id_strategy,
            "counts": tracking_counts,
        }
        _write_json_artifact(context.run_dir / "raw_tracked_detections.json", raw_detections_payload)
        _write_json_artifact(context.run_dir / "tracked_detections.json", detections_payload)
        _write_json_artifact(context.run_dir / "metrics.json", metrics_payload)
        _write_json_artifact(context.run_dir / "tracks.json", tracks)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("raw_tracked_detections.json", "tracked_detections.json", "metrics.json", "tracks.json"),
            notes=(
                "invoked manifest yolo26m checkpoint through Ultralytics model.track with BoT-SORT ReID enabled",
                "uses Ultralytics ReID model=auto appearance encoder; not a precomputed detections adapter",
                "exported raw/source-pixel and calibration-scaled detection pools for raw-pool global association",
            ),
            metrics={
                **tracking_counts,
                "checkpoint_sha256_verified": checkpoint.name,
                "tracker_config": str(tracker_config),
                "source_video": str(video_path),
                "imgsz": self.imgsz,
                "conf": self.conf,
                "iou": self.iou,
            },
        )


class PoseStageRunner:
    stage = "pose"
    real_model = False
    source_mode = "removed_legacy_pose_stage"

    def __init__(
        self,
        *,
        manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
        runtime: Any | None = None,
        motionbert_runtime: Any | None = None,
        model_id: str = "removed_legacy_pose_stage",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self._runtime = runtime
        self._motionbert_runtime = motionbert_runtime
        self.model_id = model_id

    def run(self, context: StageContext) -> StageRun:
        del context
        return StageRun(
            stage=self.stage,
            status=PIPELINE_STATUS_FAIL,
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=(),
            notes=(
                "legacy pose skeleton stage was removed from the production SAM-3D skeleton path; "
                "run BODY to produce sam3d_body_joints skeleton3d.json",
            ),
            metrics={
                "legacy_pose_stage_removed": True,
                "replacement_skeleton_source": "sam3d_body_joints",
            },
        )


class BodyStageRunner:
    stage = "body"
    real_model = True
    source_mode = "fast_sam_3d_body"

    def __init__(
        self,
        *,
        manifest_path: str | Path = DEFAULT_BODY_MODEL_MANIFEST,
        fast_sam_repo: str | Path = DEFAULT_FAST_SAM_REPO,
        runtime: Any | None = None,
        detector_name: str = DEFAULT_BODY_DETECTOR_NAME,
        fov_name: str = DEFAULT_BODY_FOV_NAME,
        smoothing_alpha: float = 1.0,
        max_root_speed_mps: float | None = DEFAULT_BODY_MAX_ROOT_SPEED_MPS,
        max_track_anchor_smoothing_residual_m: float | None = DEFAULT_BODY_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M,
        tier2_body_joints_all_tracked: bool = False,
        mesh_vertex_serialization_policy: Literal["all", "tier1_only"] = "all",
        write_body_monoliths: bool = False,
        sam3d_body_input_size_px: int | None = None,
        sam3d_crop_bucket_sizes: tuple[int, ...] = (),
        sam3d_crop_padding_scale: float = 1.0,
        sam3d_mask_prompt_mode: Literal["off", "manifest"] = "manifest",
        sam3d_mask_prompt_artifact: str = "sam3d_body_mask_prompts.json",
        sam3d_soft_background_alpha: float = 1.0,
        sam3d_torch_compile: bool = False,
        sam3d_compile_warmup_buckets: tuple[int, ...] = (),
        sam3d_compile_warmup_passes: int = 2,
        sam3d_steady_state_empty_cache: bool = True,
        sam3d_inner_bucket_sync: bool = True,
        sam3d_upstream_env: Mapping[str, Any] | None = None,
        sam3d_tier2_output_lite: bool = False,
        sam3d_wrist_bone_lock: bool = True,
        body_temporal_smoothing: bool = True,
        body_foot_lock: bool = True,
        body_foot_pin: bool = True,
        body_contact_splice: bool = True,
        body_world_joint_visual_smoothing: bool = True,
        experimental_body_array_native: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.fast_sam_repo = Path(fast_sam_repo)
        self._runtime = runtime
        self.detector_name = detector_name
        self.fov_name = fov_name
        self.smoothing_alpha = smoothing_alpha
        self.max_root_speed_mps = max_root_speed_mps
        self.max_track_anchor_smoothing_residual_m = max_track_anchor_smoothing_residual_m
        self.tier2_body_joints_all_tracked = bool(tier2_body_joints_all_tracked)
        self.mesh_vertex_serialization_policy = mesh_vertex_serialization_policy
        self.write_body_monoliths = bool(write_body_monoliths)
        self.sam3d_body_input_size_px = normalize_body_input_size(sam3d_body_input_size_px)
        self.sam3d_crop_bucket_sizes = tuple(int(value) for value in sam3d_crop_bucket_sizes)
        self.sam3d_crop_padding_scale = normalize_crop_padding_scale(sam3d_crop_padding_scale)
        self.sam3d_mask_prompt_mode = sam3d_mask_prompt_mode
        self.sam3d_mask_prompt_artifact = str(sam3d_mask_prompt_artifact)
        self.sam3d_soft_background_alpha = normalize_soft_background_alpha(sam3d_soft_background_alpha)
        self.sam3d_torch_compile = bool(sam3d_torch_compile)
        self.sam3d_compile_warmup_buckets = tuple(int(value) for value in sam3d_compile_warmup_buckets)
        if int(sam3d_compile_warmup_passes) <= 0:
            raise ValueError("sam3d_compile_warmup_passes must be positive")
        self.sam3d_compile_warmup_passes = int(sam3d_compile_warmup_passes)
        self.sam3d_steady_state_empty_cache = bool(sam3d_steady_state_empty_cache)
        self.sam3d_inner_bucket_sync = bool(sam3d_inner_bucket_sync)
        self.sam3d_upstream_env = _normalize_sam3d_upstream_env(sam3d_upstream_env or {})
        self.sam3d_tier2_output_lite = bool(sam3d_tier2_output_lite)
        self.sam3d_wrist_bone_lock = bool(sam3d_wrist_bone_lock)
        self.body_postchain = BodyPostChainConfig(
            temporal_smoothing=bool(body_temporal_smoothing),
            foot_lock=bool(body_foot_lock),
            foot_pin=bool(body_foot_pin),
            contact_splice=bool(body_contact_splice),
            wrist_lock=bool(sam3d_wrist_bone_lock),
            world_joint_visual_smoothing=bool(body_world_joint_visual_smoothing),
        )
        self.experimental_body_array_native = bool(experimental_body_array_native)

    def _sam3d_tier2_config(self, body_execution: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_sam3d_tier2_config",
            "source": "sam3d_tier2_impl_20260703T0xZ",
            "body_stage": {
                "skeleton_source": "sam3d_body_joints",
                "tier2_body_joints_all_tracked": self.tier2_body_joints_all_tracked,
                "tier1_mesh_policy": "ball_aware_100",
                "legacy_pose_path": "removed_from_production_skeleton_path",
            },
            "serialization": {
                "mesh_vertex_serialization_policy": self.mesh_vertex_serialization_policy,
                "tier2_mesh_vertices_serialized": self.mesh_vertex_serialization_policy != "tier1_only",
                "write_body_monoliths": self.write_body_monoliths,
            },
            "optimization": {
                "sam3d_body_input_size_px": self.sam3d_body_input_size_px,
                "crop_bucket_sizes": list(self.sam3d_crop_bucket_sizes),
                "crop_padding_scale": self.sam3d_crop_padding_scale,
                "mask_prompt_mode": self.sam3d_mask_prompt_mode,
                "mask_prompt_artifact": self.sam3d_mask_prompt_artifact,
                "soft_background_alpha": self.sam3d_soft_background_alpha,
                "torch_compile": self.sam3d_torch_compile,
                "compile_warmup_buckets": list(self.sam3d_compile_warmup_buckets),
                "compile_warmup_passes": self.sam3d_compile_warmup_passes,
                "steady_state_empty_cache": self.sam3d_steady_state_empty_cache,
                "inner_bucket_sync": self.sam3d_inner_bucket_sync,
                "upstream_env": dict(self.sam3d_upstream_env),
                "tier2_output_lite": self.sam3d_tier2_output_lite,
                "sam3d_wrist_bone_lock": self.sam3d_wrist_bone_lock,
                "experimental_body_array_native": self.experimental_body_array_native,
                "body_joint_builder": (
                    "array_native_shared_worldhmr_compute"
                    if self.experimental_body_array_native
                    else "legacy_worldhmr_build_body_artifacts_from_fast_sam"
                ),
                "static_clip_intrinsics": True,
                "compile_warmup_static_intrinsics": bool(
                    self.sam3d_torch_compile and self.sam3d_compile_warmup_buckets
                ),
                "compile_stall_regression_target_s": 1.0,
                "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
            },
            "accuracy_opt": {
                "source": ACCURACY_OPT_SOURCE,
                "mask_prompt_fallback": "box_only_when_mask_absent",
                "camera_intrinsics_policy": "static_per_clip_from_court_calibration",
                "crop_resolution_sweep_sizes_px": [384, 448, 512],
            },
            "scheduled_summary": {
                "scheduled_player_frame_count": body_execution.get("summary", {}).get("scheduled_player_frame_count", 0),
                "tier1_mesh_player_frame_count": body_execution.get("summary", {}).get("tier1_mesh_player_frame_count", 0),
                "tier2_body_joint_player_frame_count": body_execution.get("summary", {}).get(
                    "tier2_body_joint_player_frame_count", 0
                ),
            },
            "validation": {
                "protected_eval_labels_used": False,
                "gpu_required_for_timing": True,
                "local_run_is_config_only": True,
                "target_ms_per_person_batched": 55.0,
            },
        }
        if not self.body_postchain.is_default:
            payload["optimization"]["body_postchain"] = self.body_postchain.to_artifact_dict()
        return payload

    def run(self, context: StageContext) -> StageRun:
        body_wall_start = time.perf_counter()
        phase_timings: dict[str, Any] = {}
        phase_boundaries: dict[str, str] = {
            "model_load_s": "true FastSAM model_setup_load from run_sam3dbody_batch.py when subprocess timing is available; otherwise BodyStageRunner-visible model construction",
            "orchestrator_model_setup_s": "manifest asset verification plus BodyStageRunner-visible subprocess runtime construction",
            "input_prep_s": "BodyStageRunner-visible frame, bbox, mask, soft-background, and request-payload preparation",
            "subprocess_outer_call_s": "wall around runtime.process_frame_batches including subprocess launch/result handoff",
            "inference_s": "steady SAM3D bucket inference from run_sam3dbody_batch.py when available; otherwise runtime process_frame/process_frame_batches outer call",
            "runner_preprocessing_s": "run_sam3dbody_batch.py request/crop/bucket/tensor preparation",
            "runner_postprocessing_s": "run_sam3dbody_batch.py output record conversion outside model inference",
            "runner_result_serialization_handoff_s": "run_sam3dbody_batch.py stream chunk/monolithic output serialization",
            "runner_other_s": "run_sam3dbody_batch.py timing summary remainder after runner-local attribution",
            "subprocess_wrapper_handoff_s": "BodyStageRunner outer batch wall not covered by the runner timing sidecar",
            "keypoints_2d_s": "sam3d_keypoints_2d sidecar derivation and write",
            "mesh_smpl_payload_assembly_s": "smpl_motion/body_mesh Python payload construction; zero in default slim array-native mode",
            "smpl_motion_payload_assembly_s": "build_body_artifacts_from_fast_sam/monolith assembly or empty BODY payload construction before serialization; zero in default slim array-native mode",
            "array_native_gate_feed_s": "default slim mode BODY gate/readiness/splice/skeleton/index payload views built from shared worldhmr joint compute without monolithic smpl/body payloads",
            "mesh_export_payload_assembly_s": "build_body_mesh_export Python payload construction before serialization",
            "contact_splice_s": "contact skeleton splice, optional wrist lock, skeleton/contact artifact writes",
            "gates_s": "BODY quality, full-clip gate, grounding quality, and mesh-readiness artifact builds/writes",
            "serialization_s": "sum from body_serialization_timing.json for compact smpl_motion/body_mesh writes",
            "index_build_s": "body_mesh_index built from the in-memory body_mesh payload after body_mesh.json write",
            "artifact_io_s": "small BODY config/plan artifact writes outside compact monolith serialization",
        }
        not_instrumentable: dict[str, str] = {}
        timing_sources: dict[str, str] = {}
        tracks = validate_artifact_file("tracks", context.run_dir / "tracks.json")
        calibration = validate_artifact_file("court_calibration", context.run_dir / "court_calibration.json")
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks.json did not validate as Tracks")
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")
        placement_payload = _read_optional_json(context.run_dir / "placement.json")
        foot_contact_phases_payload = _read_optional_json(context.run_dir / "foot_contact_phases.json")
        stance_index = _body_stance_index_from_placement(
            placement_payload,
            foot_contact_phases=foot_contact_phases_payload,
            fps=float(tracks.fps),
        )
        grounding_anchor_source = (
            R3_GROUNDING_ANCHOR_SOURCE
            if stance_index and getattr(tracks, "placement_provenance", None)
            else DEFAULT_GROUNDING_ANCHOR_SOURCE
        )
        body_plan_path = context.run_dir / "frame_compute_plan.json"
        required_model_ids = fast_sam_required_model_ids(detector_name=self.detector_name, fov_name=self.fov_name)
        try:
            assets = verify_fast_sam_manifest_assets(self.manifest_path, required_model_ids=required_model_ids)
        except Exception as exc:
            body_execution = build_empty_body_compute_execution(
                tracks,
                mode="adaptive_frame_compute_plan" if body_plan_path.is_file() else "lane_b_requires_frame_compute_plan",
                source_plan=str(body_plan_path) if body_plan_path.is_file() else None,
            )
            if "sha256 mismatch" in str(exc):
                body_execution["fail_closed_reason"] = "manifest_asset_preflight_sha256_mismatch"
            write_body_compute_execution(context.run_dir / "body_compute_execution.json", body_execution)
            raise

        artifact_io_start = time.perf_counter()
        lane_b_plan = _ensure_body_frame_plan_from_sam3d(context, tracks)
        body_execution = build_body_compute_execution(
            tracks,
            frame_plan_path=body_plan_path,
            max_frames=context.max_frames,
            include_tier2_body_joints=self.tier2_body_joints_all_tracked,
        )
        write_body_compute_execution(context.run_dir / "body_compute_execution.json", body_execution)
        tier2_config = self._sam3d_tier2_config(body_execution)
        _write_json_artifact(context.run_dir / "sam3d_tier2_config.json", tier2_config)
        phase_timings["artifact_io_s"] = phase_timings.get("artifact_io_s", 0.0) + max(
            0.0,
            time.perf_counter() - artifact_io_start,
        )
        input_prep_start = time.perf_counter()
        frame_batches = body_frame_batches_from_execution(tracks, body_execution)
        if not frame_batches:
            raise ValueError("adaptive BODY schedule contains no SAM3D body-mode frames")
        target_representation_by_request = _target_representation_lookup(body_execution)
        request_metadata_by_request = _body_request_metadata_lookup(body_execution)
        mask_lookup = load_mask_prompt_lookup(
            context.run_dir,
            artifact_name=self.sam3d_mask_prompt_artifact,
            mode=self.sam3d_mask_prompt_mode,
        )

        model_load_start = time.perf_counter()
        fast_sam_runtime_unavailable_note = ""
        fast_sam_subprocess_degrade_note = ""
        fast_sam_subprocess_degrade_summary: dict[str, Any] = {"status": "not_degraded"}
        runtime = self._runtime
        fast_sam_python = os.environ.get(FAST_SAM_PYTHON_ENV, "").strip()
        fast_sam_runtime_mode = "injected" if runtime is not None else ("subprocess" if fast_sam_python else "in_process")
        try:
            if runtime is None:
                try:
                    if fast_sam_python:
                        runtime = FastSam3DBodySubprocessRuntime(
                            python_executable=fast_sam_python,
                            fast_sam_repo=self.fast_sam_repo,
                            checkpoint_dir=assets["fast_sam_3d_body_dinov3"].path.parent,
                            detector_name=self.detector_name,
                            detector_model=str(assets["yolo26m"].path) if self.detector_name and "yolo26m" in assets else "",
                            fov_name=self.fov_name,
                            body_input_size_px=self.sam3d_body_input_size_px,
                            work_dir=context.run_dir / "fast_sam_subprocess",
                        )
                        not_instrumentable["model_load_s"] = (
                            "subprocess mode performs real FastSAM model_setup_load inside "
                            "scripts/racketsport/run_sam3dbody_batch.py; BodyStageRunner can only time "
                            "manifest verification and subprocess-runtime construction without touching forbidden files"
                        )
                    else:
                        fast_sam_subprocess_degrade_note = (
                            f"{FAST_SAM_PYTHON_ENV} is unset; BODY is degrading from the batched "
                            "FastSam3DBodySubprocessRuntime path to the in-process per-frame FastSam3DBodyRuntime path"
                        )
                        fast_sam_subprocess_degrade_summary = {
                            "status": "degraded_to_in_process",
                            "reason": fast_sam_subprocess_degrade_note,
                            "strict_env": FAST_SAM_REQUIRE_SUBPROCESS_ENV,
                            "strict": _env_flag_enabled(FAST_SAM_REQUIRE_SUBPROCESS_ENV),
                        }
                        if fast_sam_subprocess_degrade_summary["strict"]:
                            raise RuntimeError(
                                f"{fast_sam_subprocess_degrade_note}; set {FAST_SAM_PYTHON_ENV} or unset "
                                f"{FAST_SAM_REQUIRE_SUBPROCESS_ENV}"
                            )
                        warnings.warn(fast_sam_subprocess_degrade_note, RuntimeWarning, stacklevel=2)
                        runtime = FastSam3DBodyRuntime(
                            assets=assets,
                            fast_sam_repo=self.fast_sam_repo,
                            detector_name=self.detector_name,
                            fov_name=self.fov_name,
                            body_input_size_px=self.sam3d_body_input_size_px,
                        )
                except RuntimeError as exc:
                    if fast_sam_subprocess_degrade_summary.get("strict"):
                        raise
                    fast_sam_runtime_mode = "unavailable"
                    fast_sam_runtime_unavailable_note = (
                        "Fast SAM-3D-Body runtime unavailable; SAM-3D samples will be absent for this run: "
                        f"{exc}"
                    )
        finally:
            phase_timings["model_load_s"] = max(0.0, time.perf_counter() - model_load_start)
        if isinstance(runtime, FastSam3DBodySubprocessRuntime):
            runtime = _BinaryHandoffSubprocessRuntime(runtime)
        samples: list[dict[str, Any]] = []
        sam3d_missing_output_count = 0
        mesh_requests: list[dict[str, Any]] = []
        prep_records: list[dict[str, Any]] = []
        static_intrinsics: list[list[float]] | None = None
        static_intrinsics_image_size: tuple[int, int] | None = None
        for frame_idx, frame_requests in frame_batches:
            image_path = _find_body_frame_image(context, frame_idx)
            track_bboxes = [list(track_frame.bbox) for _, track_frame in frame_requests]
            image_size = _image_size_from_body_frame_or_calibration(
                image_path,
                calibration=calibration,
                bboxes=track_bboxes,
            )
            if static_intrinsics_image_size is None:
                static_intrinsics_image_size = image_size
                static_intrinsics = static_camera_intrinsics_k(calibration, image_size_px=image_size)
            elif image_size != static_intrinsics_image_size:
                raise ValueError(
                    "BODY frame image size changed across SAM3D runtime requests: "
                    f"first size={static_intrinsics_image_size}, "
                    f"current size={image_size} at {image_path}"
                )
            bbox_scale_x, bbox_scale_y = _bbox_scale_from_calibration_to_image(calibration, image_size)
            scaled_frame_requests = [
                (
                    player_id,
                    track_frame,
                    _scale_bbox(list(track_frame.bbox), scale_x=bbox_scale_x, scale_y=bbox_scale_y),
                )
                for player_id, track_frame in frame_requests
            ]
            for player_id, track_frame, bbox in scaled_frame_requests:
                request_key = (frame_idx, int(player_id))
                request_metadata = request_metadata_by_request.get(request_key, {})
                prepared_bbox = padded_bbox_xyxy(
                    bbox,
                    image_size_px=image_size,
                    padding_scale=self.sam3d_crop_padding_scale,
                )
                mask_path = mask_lookup.path_for(frame_idx=frame_idx, player_id=int(player_id))
                runtime_image_path = image_path
                soft_background_applied = False
                soft_background_status = "disabled"
                if mask_path is not None and self.sam3d_soft_background_alpha < 1.0:
                    soft_path = (
                        context.run_dir
                        / "sam3d_soft_background"
                        / f"frame_{frame_idx:06d}_player_{int(player_id)}.png"
                    )
                    runtime_image_path, soft_background_applied, soft_background_status = write_soft_background_image(
                        image_path=image_path,
                        mask_path=mask_path,
                        out_path=soft_path,
                        background_alpha=self.sam3d_soft_background_alpha,
                    )
                runtime_request = {
                    "request_id": f"{frame_idx}:{int(player_id)}",
                    "frame_idx": frame_idx,
                    "player_id": int(player_id),
                    "image_path": runtime_image_path,
                    "bboxes": [prepared_bbox],
                    "mask_paths": [mask_path] if mask_path is not None else [],
                    "camera_intrinsics": static_intrinsics,
                    "camera_intrinsics_source": "court_calibration.json",
                    "sam3d_body_input_size_px": self.sam3d_body_input_size_px,
                    "crop_padding_scale": self.sam3d_crop_padding_scale,
                    "soft_background_alpha": self.sam3d_soft_background_alpha,
                    "target_representation": target_representation_by_request.get(request_key, "world_mesh"),
                    "reasons": list(request_metadata.get("reasons", [])),
                }
                prep_records.append(
                    {
                        "request_id": runtime_request["request_id"],
                        "frame_idx": frame_idx,
                        "player_id": int(player_id),
                        "image_path": str(image_path),
                        "runtime_image_path": str(runtime_image_path),
                        "image_size_px": [int(image_size[0]), int(image_size[1])],
                        "original_bbox_xyxy": [float(value) for value in bbox],
                        "prepared_bbox_xyxy": prepared_bbox,
                        "mask_prompt_path": str(mask_path) if mask_path is not None else "",
                        "mask_prompt_mode": self.sam3d_mask_prompt_mode,
                        "target_representation": runtime_request["target_representation"],
                        "reasons": runtime_request["reasons"],
                        "soft_background_applied": soft_background_applied,
                        "soft_background_status": soft_background_status,
                    }
                )
                mesh_requests.append(
                    {
                        "frame_idx": frame_idx,
                        "player_id": player_id,
                        "track_frame": track_frame,
                        "bbox": prepared_bbox,
                        "image_path": runtime_image_path,
                        "image_size": image_size,
                        "target_representation": target_representation_by_request.get(
                            request_key,
                            "world_mesh",
                        ),
                        "runtime_request": runtime_request,
                    }
                )

        if static_intrinsics is None or static_intrinsics_image_size is None:
            raise ValueError("adaptive BODY schedule produced no SAM3D runtime requests")
        _write_json_artifact(
            context.run_dir / "sam3d_body_input_prep.json",
            request_prep_artifact(
                camera_intrinsics_k=static_intrinsics,
                camera_intrinsics_image_size_px=static_intrinsics_image_size,
                mask_mode=self.sam3d_mask_prompt_mode,
                mask_manifest_path=mask_lookup.manifest_path,
                records=prep_records,
                body_input_size_px=self.sam3d_body_input_size_px,
                crop_padding_scale=self.sam3d_crop_padding_scale,
                soft_background_alpha=self.sam3d_soft_background_alpha,
            ),
        )
        phase_timings["input_prep_s"] = max(0.0, time.perf_counter() - input_prep_start)

        batched_outputs: list[list[dict[str, Any]]] | None = None
        batch_runner = getattr(runtime, "process_frame_batches", None) if runtime is not None else None
        if callable(batch_runner) and mesh_requests:
            if self.sam3d_torch_compile and self.sam3d_compile_warmup_buckets:
                not_instrumentable["compile_warmup_s"] = (
                    "compile warmup is executed inside the process_frame_batches implementation/subprocess; "
                    "BodyStageRunner has no clean timer boundary for warmup separate from the outer batch call"
                )
            timing_sidecars_before = _sam3d_batch_timing_sidecars(runtime)
            batch_start = time.perf_counter()
            batched_outputs = _call_sam3d_batch_runner(
                batch_runner,
                [request["runtime_request"] for request in mesh_requests],
                clip_intrinsics=_sam3d_static_clip_intrinsics_payload(
                    calibration=calibration,
                    camera_intrinsics_k=static_intrinsics,
                ),
                sam3d_body_input_size_px=self.sam3d_body_input_size_px,
                crop_bucket_sizes=self.sam3d_crop_bucket_sizes,
                torch_compile=self.sam3d_torch_compile,
                compile_warmup_buckets=self.sam3d_compile_warmup_buckets,
                compile_warmup_passes=self.sam3d_compile_warmup_passes,
                steady_state_empty_cache=self.sam3d_steady_state_empty_cache,
                inner_bucket_sync=self.sam3d_inner_bucket_sync,
                upstream_env=self.sam3d_upstream_env,
                tier2_output_lite=self.sam3d_tier2_output_lite,
            )
            subprocess_outer_call_s = max(0.0, time.perf_counter() - batch_start)
            subprocess_timing = _read_new_sam3d_batch_timing(runtime, before=timing_sidecars_before)
            _merge_sam3d_batch_timing(
                phase_timings,
                not_instrumentable=not_instrumentable,
                timing_sources=timing_sources,
                subprocess_timing=subprocess_timing,
                subprocess_outer_call_s=subprocess_outer_call_s,
            )
            if len(batched_outputs) != len(mesh_requests):
                raise RuntimeError(
                    f"FastSAM-3D-Body batch returned {len(batched_outputs)} outputs for {len(mesh_requests)} requests"
                )

        for request_index, mesh_request in enumerate(mesh_requests):
            frame_idx = int(mesh_request["frame_idx"])
            player_id = int(mesh_request["player_id"])
            track_frame = mesh_request["track_frame"]
            bbox = [float(value) for value in mesh_request["bbox"]]
            image_path = Path(mesh_request["image_path"])
            image_size = mesh_request["image_size"]
            raw_outputs = (
                batched_outputs[request_index]
                if batched_outputs is not None
                else []
            )
            if batched_outputs is None and runtime is not None:
                frame_inference_start = time.perf_counter()
                raw_outputs = runtime.process_frame(
                    image_path,
                    bboxes_xyxy=[bbox],
                    mask_paths=mesh_request["runtime_request"].get("mask_paths", []),
                    camera_intrinsics=mesh_request["runtime_request"].get("camera_intrinsics"),
                )
                phase_timings["inference_s"] = phase_timings.get("inference_s", 0.0) + max(
                    0.0, time.perf_counter() - frame_inference_start
                )
            if not raw_outputs:
                sam3d_missing_output_count += 1
                continue
            raw_output = _match_body_outputs(raw_outputs, [bbox])[0]
            target_representation = str(mesh_request["target_representation"])
            if (
                self.sam3d_tier2_output_lite
                and target_representation == "world_mesh"
                and not _body_record_has_dense_mesh(raw_output)
            ):
                raise ValueError("tier-1 world_mesh SAM3D output is missing pred_vertices while tier2_output_lite is enabled")
            player_request = PlayerCropRequest(
                frame_idx=frame_idx,
                player_id=player_id,
                bbox_xyxy=bbox,
                image_size_px=image_size,
                track_confidence=track_frame.conf,
            )
            sample = normalize_fast_sam_body_output(raw_output, request=player_request)
            sample["t"] = track_frame.t
            sample["track_world_xy"] = list(track_frame.world_xy)
            sample["target_representation"] = target_representation
            if (
                self.mesh_vertex_serialization_policy == "tier1_only"
                and sample["target_representation"] == TIER2_BODY_JOINTS_REPRESENTATION
            ):
                sample["vertices_camera"] = []
                sample["mesh_faces"] = []
            samples.append(sample)

        keypoints_start = time.perf_counter()
        sam3d_keypoints_sidecar = _sam3d_keypoints_sidecar_from_samples(samples)
        if sam3d_keypoints_sidecar["players"]:
            _write_json_artifact(context.run_dir / "sam3d_keypoints_2d.json", sam3d_keypoints_sidecar)
        phase_timings["keypoints_2d_s"] = max(0.0, time.perf_counter() - keypoints_start)

        body_mesh_metadata: dict[str, Any] | None = None
        body_mesh_players: list[dict[str, Any]] | None = None
        body_mesh_summary: dict[str, Any] | None = None
        raw_grounded_joints: dict[str, Any] | None = None
        body_joint_builder = "legacy_worldhmr_build_body_artifacts_from_fast_sam"
        smpl_payload_start = time.perf_counter()
        if samples and self.experimental_body_array_native:
            body_joint_builder = "array_native_shared_worldhmr_compute"
            array_native_start = time.perf_counter()
            array_native = build_body_array_native_artifacts_from_fast_sam(
                samples,
                calibration=calibration,
                fps=tracks.fps,
                clip=context.clip,
                body_compute_execution=body_execution,
                smoothing_alpha=self.smoothing_alpha,
                max_root_speed_mps=self.max_root_speed_mps,
                max_track_anchor_smoothing_residual_m=self.max_track_anchor_smoothing_residual_m,
                sam3d_wrist_bone_lock=self.sam3d_wrist_bone_lock,
                stance_index=stance_index,
                grounding_anchor_source=grounding_anchor_source,
                body_postchain=self.body_postchain,
            )
            if self.write_body_monoliths:
                smpl_motion, skeleton3d, grounding_metrics = assemble_body_monolith_payloads(
                    array_native.smpl_motion_view,
                    array_native.skeleton3d,
                    array_native.grounding_metrics,
                )
                phase_timings["smpl_motion_payload_assembly_s"] = max(0.0, time.perf_counter() - smpl_payload_start)
            else:
                smpl_motion = array_native.smpl_motion_view
                skeleton3d = array_native.skeleton3d
                grounding_metrics = array_native.grounding_metrics
                phase_timings["smpl_motion_payload_assembly_s"] = 0.0
            body_mesh_metadata = array_native.body_mesh_metadata
            body_mesh_players = array_native.body_mesh_players
            body_mesh_summary = array_native.body_mesh_summary
            raw_grounded_joints = array_native.raw_grounded_joints
            phase_timings["array_native_gate_feed_s"] = max(0.0, time.perf_counter() - array_native_start)
        elif samples:
            if self.body_postchain.is_default:
                smpl_motion, skeleton3d, grounding_metrics = build_body_artifacts_from_fast_sam(
                    samples,
                    calibration=calibration,
                    fps=tracks.fps,
                    smoothing_alpha=self.smoothing_alpha,
                    max_root_speed_mps=self.max_root_speed_mps,
                    max_track_anchor_smoothing_residual_m=self.max_track_anchor_smoothing_residual_m,
                    sam3d_wrist_bone_lock=self.sam3d_wrist_bone_lock,
                    stance_index=stance_index,
                    grounding_anchor_source=grounding_anchor_source,
                )
            else:
                computed = compute_body_skeleton_and_metrics(
                    samples,
                    calibration=calibration,
                    fps=tracks.fps,
                    smoothing_alpha=self.smoothing_alpha,
                    max_root_speed_mps=self.max_root_speed_mps,
                    max_track_anchor_smoothing_residual_m=self.max_track_anchor_smoothing_residual_m,
                    sam3d_wrist_bone_lock=self.sam3d_wrist_bone_lock,
                    stance_index=stance_index,
                    grounding_anchor_source=grounding_anchor_source,
                    body_postchain=self.body_postchain,
                )
                smpl_motion, skeleton3d, grounding_metrics = assemble_body_monolith_payloads(
                    computed.smpl_motion_view,
                    computed.skeleton3d,
                    computed.metrics,
                )
                raw_grounded_joints = computed.raw_grounded_joints
            phase_timings["smpl_motion_payload_assembly_s"] = max(0.0, time.perf_counter() - smpl_payload_start)
        else:
            smpl_motion = _empty_smpl_motion(fps=tracks.fps)
            skeleton3d = _empty_body_preview_skeleton(fps=tracks.fps)
            grounding_metrics = {
                "body_samples": 0,
                "players": 0,
                "frames": 0,
                "world_frame": "court_Z0",
                "grounding": "no_fast_sam_body_samples",
            }
            phase_timings["smpl_motion_payload_assembly_s"] = max(0.0, time.perf_counter() - smpl_payload_start)
        grounding_metrics = {**grounding_metrics, "calibration_confidence": _calibration_confidence_proxy(calibration)}
        raw_grounded_joints_written = False
        if raw_grounded_joints is not None:
            _write_json_artifact(context.run_dir / RAW_GROUNDED_JOINTS_ARTIFACT, raw_grounded_joints)
            raw_grounded_joints_written = True
        monolith_skip_reason = "not built (speed default; rerun with --fetch-body-monoliths to produce them)"
        serialization_timings = []
        if self.write_body_monoliths:
            serialization_timings.append(_write_compact_json_artifact(context.run_dir / "smpl_motion.json", smpl_motion))
        else:
            serialization_timings.append(
                _skipped_compact_json_artifact(context.run_dir / "smpl_motion.json", reason=monolith_skip_reason)
            )
        gates_start = time.perf_counter()
        body_grounding_quality = build_body_grounding_quality(
            clip=context.clip,
            grounding_metrics=grounding_metrics,
        )
        write_body_grounding_quality(context.run_dir / "body_grounding_quality.json", body_grounding_quality)
        phase_timings["gates_s"] = phase_timings.get("gates_s", 0.0) + max(0.0, time.perf_counter() - gates_start)
        smpl_motion_payload = smpl_motion.model_dump(mode="json") if hasattr(smpl_motion, "model_dump") else smpl_motion
        skeleton3d_path = context.run_dir / "skeleton3d.json"
        existing_skeleton3d = _read_optional_json(skeleton3d_path)
        preserved_existing_skeleton = (
            _is_real_sam3d_skeleton3d(existing_skeleton3d)
            and not self.tier2_body_joints_all_tracked
        )
        if preserved_existing_skeleton:
            skeleton3d_payload = existing_skeleton3d
        else:
            _write_json_artifact(skeleton3d_path, skeleton3d)
            skeleton3d_payload = skeleton3d.model_dump(mode="json") if hasattr(skeleton3d, "model_dump") else skeleton3d
        mesh_export_start = time.perf_counter()
        body_mesh: dict[str, Any] | None = None
        if self.write_body_monoliths:
            body_mesh = build_body_mesh_export(
                smpl_motion_payload,
                clip=context.clip,
                body_compute_execution=body_execution,
            )
            body_mesh_summary = dict(body_mesh["summary"])
        else:
            if body_mesh_metadata is None or body_mesh_players is None or body_mesh_summary is None:
                body_mesh_metadata, body_mesh_players, body_mesh_summary = body_mesh_export_parts_from_smpl_motion_view(
                    smpl_motion_payload,
                    clip=context.clip,
                    body_compute_execution=body_execution,
                )
        phase_timings["mesh_export_payload_assembly_s"] = max(0.0, time.perf_counter() - mesh_export_start)
        if self.write_body_monoliths or body_joint_builder != "array_native_shared_worldhmr_compute":
            phase_timings["mesh_smpl_payload_assembly_s"] = (
                phase_timings.get("smpl_motion_payload_assembly_s", 0.0)
                + phase_timings.get("mesh_export_payload_assembly_s", 0.0)
            )
        else:
            phase_timings["mesh_smpl_payload_assembly_s"] = 0.0
        if self.write_body_monoliths:
            assert body_mesh is not None
            serialization_timings.append(_write_compact_json_artifact(context.run_dir / "body_mesh.json", body_mesh))
        else:
            serialization_timings.append(
                _skipped_compact_json_artifact(context.run_dir / "body_mesh.json", reason=monolith_skip_reason)
            )
        index_build_start = time.perf_counter()
        if self.write_body_monoliths:
            assert body_mesh is not None
            body_mesh_index_result = build_body_mesh_index_from_payload(
                body_mesh,
                out_dir=context.run_dir / "body_mesh_index",
            )
            body_mesh_for_splice = body_mesh
        else:
            assert body_mesh_metadata is not None and body_mesh_players is not None
            body_mesh_index_result = build_body_mesh_index_from_arrays(
                metadata=body_mesh_metadata,
                players=body_mesh_players,
                out_dir=context.run_dir / "body_mesh_index",
            )
            body_mesh_for_splice = body_mesh_payload_from_parts(
                body_mesh_metadata,
                body_mesh_players,
                summary=body_mesh_summary,
            )
        phase_timings["index_build_s"] = max(0.0, time.perf_counter() - index_build_start)
        _write_body_serialization_timing(context.run_dir, serialization_timings)
        phase_timings["serialization_s"] = sum(float(item["serialization_seconds"]) for item in serialization_timings)
        contact_splice_start = time.perf_counter()
        if self.body_postchain.contact_splice:
            skeleton3d_payload, contact_splice = splice_contact_skeleton_with_body_mesh(
                skeleton3d_payload,
                body_mesh=body_mesh_for_splice,
                body_compute_execution=body_execution,
            )
        else:
            contact_splice = _bypassed_contact_splice_artifact(body_execution)
        if self.sam3d_wrist_bone_lock:
            skeleton3d_payload = apply_sam3d_wrist_bone_lock(skeleton3d_payload)
        _write_json_artifact(skeleton3d_path, skeleton3d_payload)
        _write_json_artifact(context.run_dir / "contact_splice.json", contact_splice)
        phase_timings["contact_splice_s"] = max(0.0, time.perf_counter() - contact_splice_start)
        gates_start = time.perf_counter()
        body_joint_quality = build_body_joint_quality(
            clip=context.clip,
            smpl_motion=smpl_motion_payload,
            skeleton3d=skeleton3d_payload,
            body_compute_execution=body_execution,
            smpl_motion_path=str(context.run_dir / "smpl_motion.json"),
            skeleton3d_path=str(context.run_dir / "skeleton3d.json"),
            body_compute_execution_path=str(context.run_dir / "body_compute_execution.json"),
        )
        full_clip_gate = build_body_full_clip_gate(
            clip=context.clip,
            tracks=tracks.model_dump(mode="json"),
            body_compute_execution=body_execution,
            body_joint_quality=body_joint_quality,
            contact_splice=contact_splice,
            runtime_timing={"body_wall_seconds": max(0.0, time.perf_counter() - body_wall_start)},
            tracks_path=str(context.run_dir / "tracks.json"),
            body_compute_execution_path=str(context.run_dir / "body_compute_execution.json"),
            body_joint_quality_path=str(context.run_dir / "body_joint_quality.json"),
            contact_splice_path=str(context.run_dir / "contact_splice.json"),
            runtime_timing_path="body_stage_wall_clock",
        )
        _write_json_artifact(context.run_dir / "body_full_clip_gate.json", full_clip_gate)
        body_joint_quality = _body_joint_quality_after_full_clip_gate(body_joint_quality, full_clip_gate)
        _write_json_artifact(context.run_dir / "body_joint_quality.json", body_joint_quality)
        body_mesh_readiness = build_body_mesh_readiness(
            clip=context.clip,
            smpl_motion=smpl_motion_payload,
            skeleton3d=skeleton3d_payload,
            frame_compute_plan=_read_optional_json(context.run_dir / "frame_compute_plan.json"),
            body_compute_execution=body_execution,
            body_full_clip_gate=full_clip_gate,
            smpl_motion_path=str(context.run_dir / "smpl_motion.json"),
            skeleton3d_path=str(context.run_dir / "skeleton3d.json"),
            frame_compute_plan_path=str(context.run_dir / "frame_compute_plan.json"),
            body_compute_execution_path=str(context.run_dir / "body_compute_execution.json"),
            body_full_clip_gate_path=str(context.run_dir / "body_full_clip_gate.json"),
        )
        if not self.write_body_monoliths:
            body_mesh_readiness["monoliths"] = {
                "status": "not_built",
                "note": monolith_skip_reason,
                "smpl_motion_path": "",
                "body_mesh_path": "",
            }
            body_mesh_readiness["warnings"] = _dedupe_strings(
                [*body_mesh_readiness.get("warnings", []), "body_monoliths_not_built_speed_default"]
            )
        _write_json_artifact(context.run_dir / "body_mesh_readiness.json", body_mesh_readiness)
        phase_timings["gates_s"] = phase_timings.get("gates_s", 0.0) + max(0.0, time.perf_counter() - gates_start)
        body_stage_wall_seconds = max(0.0, time.perf_counter() - body_wall_start)
        _write_body_stage_phase_timing(
            context.run_dir,
            stage_wall_seconds=body_stage_wall_seconds,
            phase_timings=phase_timings,
            person_frame_count=len(mesh_requests),
            phase_boundaries=phase_boundaries,
            not_instrumentable=not_instrumentable,
            timing_sources=timing_sources,
            postchain_bypasses=self.body_postchain.bypass_summary(),
        )
        produced_artifacts = [
            "body_compute_execution.json",
            "sam3d_tier2_config.json",
            "sam3d_body_input_prep.json",
            "body_mesh_index/body_mesh_index.json",
            "body_mesh_index/body_mesh_faces.json",
            "body_serialization_timing.json",
            "body_stage_phase_timing.json",
            "contact_splice.json",
            "skeleton3d.json",
            "body_mesh_readiness.json",
            "body_joint_quality.json",
            "body_full_clip_gate.json",
            "body_grounding_quality.json",
        ]
        if self.write_body_monoliths:
            produced_artifacts[3:3] = ["smpl_motion.json", "body_mesh.json"]
        if raw_grounded_joints_written:
            produced_artifacts.append(RAW_GROUNDED_JOINTS_ARTIFACT)
        notes = [
            "Fast SAM-3D-Body runtime output converted to court/world coordinates with court_calibration.json",
            (
                "BODY skeleton/joint computation used legacy worldhmr.build_body_artifacts_from_fast_sam; "
                "array-native shared compute was explicitly disabled"
                if body_joint_builder != "array_native_shared_worldhmr_compute"
                else "BODY skeleton/joint computation used shared worldhmr.compute_body_skeleton_and_metrics; monolith assembly skipped unless requested"
            ),
            "BODY frame execution follows frame_compute_plan.json when present; selected human-review frames emit preview-badged ghost mesh, while unselected preview-only frames still skip mesh",
            "preserved existing skeleton3d.json"
            if preserved_existing_skeleton
            else "wrote SAM3D body-mode skeleton3d.json as the offline skeleton source",
            "spliced scheduled hitter mesh joints into existing skeleton3d.json at contact frames"
            if preserved_existing_skeleton
            else (
                "kept SAM3D body-mode skeleton3d.json as the skeleton source while body_mesh.json carries tier-1 mesh frames"
                if self.write_body_monoliths
                else "kept SAM3D body-mode skeleton3d.json as the skeleton source while body_mesh_index/ carries tier-1 mesh frames"
            ),
            f"SAM-3D returned no output for {sam3d_missing_output_count} scheduled request(s); no legacy pose fallback was used",
            "BODY artifacts are real runner outputs; BODY accuracy gate still requires labeled world-MPJPE evaluation",
            (
                "surfaced calibration_confidence (raw reprojection-error proxy, non-blocking) in "
                "body_grounding_quality.json; this is a coarse proxy, not a validated BODY-quality signal -- "
                "see runs/cal_body_projection_bias_20260702T014121Z (95.3% of observed dy-bias attributed to "
                "BODY's own joint-layout asymmetry, not calibration error) before gating on it"
            ),
        ]
        if fast_sam_runtime_unavailable_note:
            notes.append(fast_sam_runtime_unavailable_note)
        if fast_sam_subprocess_degrade_note:
            notes.append(fast_sam_subprocess_degrade_note)
        binary_handoff_note = str(getattr(runtime, "binary_handoff_note", "") or "")
        if binary_handoff_note:
            notes.append(binary_handoff_note)
        if not self.write_body_monoliths:
            notes.append("BODY monoliths smpl_motion.json/body_mesh.json were not built (speed default; rerun with --fetch-body-monoliths to produce them)")
        if not self.body_postchain.is_default:
            notes.append(
                "BODY post-chain bypassed stages: "
                + ", ".join(self.body_postchain.bypassed_stages())
                + (
                    f"; raw grounded joints sidecar: {RAW_GROUNDED_JOINTS_ARTIFACT}"
                    if raw_grounded_joints_written
                    else ""
                )
            )
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=tuple(produced_artifacts),
            notes=tuple(notes),
            wall_seconds=body_stage_wall_seconds,
            metrics={
                **grounding_metrics,
                "body_compute_mode": body_execution["mode"],
                "sam3d_tier2_body_joints_all_tracked": self.tier2_body_joints_all_tracked,
                "sam3d_mesh_vertex_serialization_policy": self.mesh_vertex_serialization_policy,
                "sam3d_body_input_size_px": self.sam3d_body_input_size_px,
                "sam3d_crop_bucket_sizes": list(self.sam3d_crop_bucket_sizes),
                "sam3d_crop_padding_scale": self.sam3d_crop_padding_scale,
                "sam3d_mask_prompt_mode": self.sam3d_mask_prompt_mode,
                "sam3d_soft_background_alpha": self.sam3d_soft_background_alpha,
                "sam3d_mask_prompt_available_count": sum(1 for record in prep_records if record.get("mask_prompt_path")),
                "sam3d_mask_prompt_missing_count": sum(1 for record in prep_records if not record.get("mask_prompt_path")),
                "sam3d_camera_intrinsics_static_per_clip": True,
                "sam3d_torch_compile": self.sam3d_torch_compile,
                "sam3d_compile_warmup_buckets": list(self.sam3d_compile_warmup_buckets),
                "sam3d_compile_warmup_passes": self.sam3d_compile_warmup_passes,
                "sam3d_steady_state_empty_cache": self.sam3d_steady_state_empty_cache,
                "sam3d_inner_bucket_sync": self.sam3d_inner_bucket_sync,
                "sam3d_upstream_env": dict(self.sam3d_upstream_env),
                "sam3d_tier2_output_lite": self.sam3d_tier2_output_lite,
                "sam3d_wrist_bone_lock": self.sam3d_wrist_bone_lock,
                "experimental_body_array_native": self.experimental_body_array_native,
                "body_joint_builder": body_joint_builder,
                "sam3d_wrist_bone_lock_status": (
                    skeleton3d_payload.get("provenance", {})
                    .get("sam3d_wrist_bone_lock", {})
                    .get("status", "disabled" if not self.sam3d_wrist_bone_lock else "absent")
                    if isinstance(skeleton3d_payload, Mapping)
                    else "absent"
                ),
                "sam3d_wrist_bone_lock_locked_frame_count": (
                    skeleton3d_payload.get("provenance", {})
                    .get("sam3d_wrist_bone_lock", {})
                    .get("locked_frame_count", 0)
                    if isinstance(skeleton3d_payload, Mapping)
                    else 0
                ),
                "sam3d_core_body_speed_clamp_engagement_by_player": (
                    skeleton3d_payload.get("provenance", {})
                    .get("temporal_refine", {})
                    .get("physical_plausibility", {})
                    .get("core_body_speed_clamp_engagement_by_player", {})
                    if isinstance(skeleton3d_payload, Mapping)
                    else {}
                ),
                "sam3d_tier2_body_joint_player_frame_count": body_execution["summary"].get(
                    "tier2_body_joint_player_frame_count", 0
                ),
                "sam3d_tier1_mesh_player_frame_count": body_execution["summary"].get(
                    "tier1_mesh_player_frame_count", 0
                ),
                "lane_b_frame_plan_source": lane_b_plan["source"],
                "lane_b_contact_event_count": lane_b_plan["contact_event_count"],
                "lane_b_frame_plan_generated_artifacts": lane_b_plan["generated_artifacts"],
                "scheduled_body_frames": body_execution["summary"]["scheduled_frame_count"],
                "scheduled_body_player_frames": body_execution["summary"]["scheduled_player_frame_count"],
                "body_mesh_frame_count": body_mesh_summary["mesh_frame_count"],
                "body_mesh_index_window_count": body_mesh_index_result["summary"].get("window_count", 0),
                "body_mesh_index_build_s": phase_timings.get("index_build_s", 0.0),
                "body_monoliths_written": self.write_body_monoliths,
                "sam3d_binary_handoff_status": str(getattr(runtime, "binary_handoff_status", "")),
                "sam3d_missing_output_count": sam3d_missing_output_count,
                "fast_sam_runtime_mode": fast_sam_runtime_mode,
                "fast_sam_python": fast_sam_python,
                "fast_sam_subprocess_degraded": bool(fast_sam_subprocess_degrade_note),
                "fast_sam_subprocess_degrade": fast_sam_subprocess_degrade_summary,
                "fast_sam_runtime_unavailable": bool(fast_sam_runtime_unavailable_note),
                "fast_sam_runtime_unavailable_note": fast_sam_runtime_unavailable_note,
                "contact_splice_spliced_contact_count": contact_splice["summary"]["spliced_contact_count"],
                "contact_splice_mesh_unavailable_count": contact_splice["summary"]["mesh_unavailable_count"],
                "contact_splice_fallback_spliced_count": contact_splice["summary"].get("fallback_spliced_count", 0),
                "contact_splice_overridden_joint_count": contact_splice["summary"]["overridden_joint_count"],
                "verified_model_ids": list(required_model_ids),
                "detector_name": self.detector_name,
                "detector_model_id": "yolo26m" if self.detector_name else "",
                "detector_model_path": str(assets["yolo26m"].path) if "yolo26m" in assets else "",
                "fov_name": self.fov_name,
                "fov_model_id": "moge_2_vitl_normal" if self.fov_name else "",
                "fov_model_path": str(assets["moge_2_vitl_normal"].path) if "moge_2_vitl_normal" in assets else "",
                "body_model_path": str(assets["fast_sam_3d_body_dinov3"].path),
                "max_root_speed_mps": self.max_root_speed_mps,
                "max_track_anchor_smoothing_residual_m": self.max_track_anchor_smoothing_residual_m,
            },
        )


def _bypassed_contact_splice_artifact(body_compute_execution: Mapping[str, Any]) -> dict[str, Any]:
    summary = body_compute_execution.get("summary", {}) if isinstance(body_compute_execution, Mapping) else {}
    scheduled = int(summary.get("scheduled_player_frame_count", 0) or 0)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_splice",
        "source": "body_postchain_contact_splice_bypass",
        "summary": {
            "status": "bypassed",
            "reason": "body_postchain_contact_splice_disabled",
            "scheduled_player_frame_count": scheduled,
            "scheduled_contact_count": 0,
            "spliced_contact_count": 0,
            "overridden_joint_count": 0,
            "mesh_unavailable_count": 0,
            "fallback_spliced_count": 0,
            "strict_mode_loud": True,
        },
        "events": [],
    }


def _calibration_confidence_proxy(calibration: CourtCalibration) -> dict[str, Any]:
    """Surface calibration reprojection error alongside BODY output (non-blocking).

    This deliberately does *not* block or reject BODY mesh output: the
    ``cal_body_projection_bias_20260702T014121Z`` diagnostic found 95.3% of the
    observed Burlington world-joint dy-bias is attributable to BODY's own
    joint-layout asymmetry, not calibration error, and a flat-calibration clip
    (Wolverine) showed a comparable PnP-vs-track mismatch -- so a naive
    "reject BODY when reprojection_error_px is high" rule would misattribute
    BODY's own systematic bias to calibration on some clips while
    under-flagging genuine calibration problems on others. This value is a
    coarse proxy for visibility/debugging only.
    """

    median_px = float(calibration.reprojection_error_px.median)
    p95_px = float(calibration.reprojection_error_px.p95)
    return {
        "reprojection_median_px": median_px,
        "reprojection_p95_px": p95_px,
        "reprojection_gate_median_px": CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
        "below_reprojection_gate": median_px <= CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
        "note": (
            "coarse proxy only, not a validated BODY-quality signal -- see "
            "runs/cal_body_projection_bias_20260702T014121Z/section_b_joint_asymmetry_explanation.json "
            "(95.3% of observed Burlington dy-bias is BODY's own joint-layout asymmetry, not calibration "
            "error; Wolverine's flat calibration shows a comparable mismatch). Do not gate/reject BODY "
            "output on this value alone."
        ),
    }


def _empty_smpl_motion(*, fps: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": "sam3dbody_world_joints",
        "fps": float(fps),
        "world_frame": "court_Z0",
        "players": [],
    }


def _empty_body_preview_skeleton(*, fps: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": float(fps),
        "world_frame": "court_Z0",
        "source_model": "sam3dbody_world_joints",
        "joint_names": [],
        "preview_only": True,
        "players": [],
        "provenance": {"lane": "B", "body_samples": 0},
    }


def _body_mesh_export_parts_from_smpl_motion(
    smpl_motion: Mapping[str, Any],
    *,
    clip: str,
    body_compute_execution: Mapping[str, Any] | None,
    faces_ref: str = _mesh_export.DEFAULT_FACES_REF,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    scheduled = _mesh_export._scheduled_targets(body_compute_execution)
    windows = _mesh_export._scheduled_windows(body_compute_execution)
    joint_names = _mesh_export._joint_names(smpl_motion)
    mesh_faces = _mesh_export._mesh_faces(smpl_motion)
    players_payload: list[dict[str, Any]] = []
    contact_window_indexes: set[int] = set()
    mesh_frame_count = 0
    for player in smpl_motion.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        betas = _mesh_export._float_list(player.get("betas", []))
        frames_payload: list[dict[str, Any]] = []
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * float(smpl_motion.get("fps", 30.0)))))
            scheduled_record = scheduled.get((frame_idx, player_id))
            if scheduled and scheduled_record is None:
                continue
            vertices = frame.get("mesh_vertices_world", [])
            if not isinstance(vertices, list) or not vertices:
                continue
            source_window_index = scheduled_record.get("source_window_index") if scheduled_record else None
            if source_window_index is not None:
                contact_window_indexes.add(int(source_window_index))
            frame_payload = {
                "frame_idx": frame_idx,
                "t": float(frame.get("t", frame_idx / float(smpl_motion.get("fps", 30.0)))),
                "source_window_index": source_window_index,
                "blend_weight": _mesh_export._blend_weight_for_frame(frame_idx, scheduled_record),
                "mesh_vertices_world": frame.get("mesh_vertices_world", []),
                "smplx_params": {
                    "global_orient": _mesh_export._float_list(frame.get("global_orient", [])),
                    "body_pose": _mesh_export._float_list(frame.get("body_pose", [])),
                    "left_hand_pose": _mesh_export._float_list(frame.get("left_hand_pose", [])),
                    "right_hand_pose": _mesh_export._float_list(frame.get("right_hand_pose", [])),
                    "betas": betas,
                    "transl_world": _mesh_export._float_list(frame.get("transl_world", [])),
                },
                "reasons": list(scheduled_record.get("reasons", [])) if scheduled_record else [],
            }
            joints_world = _mesh_export._vector3_list(frame.get("joints_world", []))
            if joints_world:
                frame_payload["joints_world"] = joints_world
            joint_conf = _mesh_export._float_list(frame.get("joint_conf", []))
            if joint_conf:
                frame_payload["joint_conf"] = joint_conf
            frames_payload.append(frame_payload)
        if frames_payload:
            mesh_frame_count += len(frames_payload)
            players_payload.append({"id": player_id, "frames": frames_payload})
    metadata = {
        "clip": clip,
        "model": str(smpl_motion.get("model", "")),
        "fps": float(smpl_motion.get("fps", 0.0)),
        "world_frame": str(smpl_motion.get("world_frame", "")),
        "faces_ref": faces_ref,
        "mesh_faces": mesh_faces,
        "joint_names": joint_names,
        "windows": windows,
    }
    summary = {
        "mesh_frame_count": mesh_frame_count,
        "player_count": len(players_payload),
        "contact_window_count": len(contact_window_indexes) if scheduled else 0,
    }
    return metadata, players_payload, summary


def _body_mesh_payload_from_parts(
    metadata: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": metadata["clip"],
        "model": metadata["model"],
        "fps": metadata["fps"],
        "world_frame": metadata["world_frame"],
        "faces_ref": metadata["faces_ref"],
        "mesh_faces": metadata.get("mesh_faces", []),
        "joint_names": metadata.get("joint_names", []),
        "windows": metadata.get("windows", []),
        "players": [dict(player) for player in players],
        "summary": dict(summary),
    }


def _skeleton_player_frame_count(payload: Mapping[str, Any] | None) -> int:
    if not isinstance(payload, Mapping):
        return 0
    players = payload.get("players")
    if not isinstance(players, list):
        return 0
    return sum(
        len(player.get("frames", []))
        for player in players
        if isinstance(player, Mapping) and isinstance(player.get("frames"), list)
    )


def _sam3d_static_clip_intrinsics_payload(
    *,
    calibration: CourtCalibration,
    camera_intrinsics_k: Sequence[Sequence[float]] | None,
) -> dict[str, Any] | None:
    if camera_intrinsics_k is None:
        return None
    intrinsics = calibration.intrinsics
    matrix = [[float(value) for value in row] for row in camera_intrinsics_k]
    payload = {
        "fx": float(matrix[0][0]),
        "fy": float(matrix[1][1]),
        "cx": float(matrix[0][2]),
        "cy": float(matrix[1][2]),
        "dist": [float(value) for value in (intrinsics.dist or [])],
        "source": str(intrinsics.source),
        "matrix": matrix,
        "static_per_clip": True,
    }
    _assert_sam3d_static_clip_intrinsics_payload_consistent(payload)
    return payload


def _assert_sam3d_static_clip_intrinsics_payload_consistent(payload: Mapping[str, Any]) -> None:
    matrix = payload.get("matrix")
    if (
        not isinstance(matrix, Sequence)
        or len(matrix) != 3
        or any(not isinstance(row, Sequence) or len(row) != 3 for row in matrix)
    ):
        raise ValueError("clip_intrinsics matrix must be 3x3")
    checks = {
        "fx": float(matrix[0][0]),
        "fy": float(matrix[1][1]),
        "cx": float(matrix[0][2]),
        "cy": float(matrix[1][2]),
    }
    for name, matrix_value in checks.items():
        payload_value = float(payload[name])
        if abs(payload_value - matrix_value) > 1e-6:
            raise ValueError(
                f"clip_intrinsics payload is inconsistent: {name}={payload_value} "
                f"but matrix value is {matrix_value}"
            )


def _call_sam3d_batch_runner(
    batch_runner: Any,
    requests: list[dict[str, Any]],
    *,
    clip_intrinsics: Mapping[str, Any] | None,
    sam3d_body_input_size_px: int | None,
    crop_bucket_sizes: Sequence[int],
    torch_compile: bool,
    compile_warmup_buckets: Sequence[int],
    compile_warmup_passes: int,
    steady_state_empty_cache: bool,
    inner_bucket_sync: bool,
    upstream_env: Mapping[str, Any],
    tier2_output_lite: bool,
) -> list[list[dict[str, Any]]]:
    kwargs = {
        "clip_intrinsics": dict(clip_intrinsics) if clip_intrinsics is not None else None,
        "sam3d_body_input_size_px": sam3d_body_input_size_px,
        "crop_bucket_sizes": tuple(int(value) for value in crop_bucket_sizes),
        "torch_compile": bool(torch_compile),
        "compile_warmup_buckets": tuple(int(value) for value in compile_warmup_buckets),
        "compile_warmup_passes": int(compile_warmup_passes),
        "steady_state_empty_cache": bool(steady_state_empty_cache),
        "inner_bucket_sync": bool(inner_bucket_sync),
        "upstream_env": dict(upstream_env),
        "tier2_output_lite": bool(tier2_output_lite),
    }
    try:
        signature = inspect.signature(batch_runner)
    except (TypeError, ValueError):
        return batch_runner(requests)
    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    accepted = {key: value for key, value in kwargs.items() if accepts_kwargs or key in parameters}
    if accepted:
        return batch_runner(requests, **accepted)
    return batch_runner(requests)


class _BinaryHandoffSubprocessRuntime:
    def __init__(self, runtime: FastSam3DBodySubprocessRuntime) -> None:
        self._runtime = runtime
        self.work_dir = runtime.work_dir
        self.binary_handoff_note = ""
        self.binary_handoff_status = "not_started"

    def process_frame_batches(
        self,
        requests: list[Any],
        *,
        clip_intrinsics: Mapping[str, Any] | None = None,
        sam3d_body_input_size_px: int | None = None,
        crop_bucket_sizes: Sequence[int] = (),
        torch_compile: bool = False,
        compile_warmup_buckets: Sequence[int] = (),
        compile_warmup_passes: int = 2,
        steady_state_empty_cache: bool = True,
        inner_bucket_sync: bool = True,
        upstream_env: Mapping[str, Any] | None = None,
        tier2_output_lite: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if not requests:
            return []
        normalized_requests = [_normalize_sam3d_subprocess_request(request) for request in requests]
        request_ids = [str(request.get("request_id") or index) for index, request in enumerate(normalized_requests)]
        body_input_size_px = normalize_body_input_size(sam3d_body_input_size_px or self._runtime.body_input_size_px)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.work_dir / f"batch_requests-{uuid.uuid4().hex}.json"
        out_path = self.work_dir / f"batch_outputs-{uuid.uuid4().hex}.json"
        request_payload = {
            "schema_version": 1,
            "clip_intrinsics": dict(clip_intrinsics) if clip_intrinsics is not None else None,
            "optimization": {
                "sam3d_body_input_size_px": body_input_size_px,
                "crop_bucket_sizes": [int(value) for value in crop_bucket_sizes],
                "torch_compile": bool(torch_compile),
                "compile_warmup_buckets": [int(value) for value in compile_warmup_buckets],
                "compile_warmup_passes": int(compile_warmup_passes),
                "steady_state_empty_cache": bool(steady_state_empty_cache),
                "inner_bucket_sync": bool(inner_bucket_sync),
                "upstream_env": dict(upstream_env or {}),
                "tier2_output_lite": bool(tier2_output_lite),
                "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
            },
            "requests": [
                {
                    "request_id": request_ids[index],
                    "image": str(request["image_path"]),
                    "bboxes": [[float(value) for value in bbox] for bbox in request["bboxes"]],
                    "mask_paths": [str(path) for path in request.get("mask_paths", []) if path],
                    "camera_intrinsics": request.get("camera_intrinsics"),
                    "sam3d_body_input_size_px": body_input_size_px,
                    "target_representation": request.get("target_representation", "world_mesh"),
                }
                for index, request in enumerate(normalized_requests)
            ],
        }
        request_path.write_text(json.dumps(request_payload, separators=(",", ":")) + "\n", encoding="utf-8")
        chunk_format = "pickle"
        command = [
            str(self._runtime.python_executable),
            str(Path(__file__).resolve().parents[2] / "scripts/racketsport/run_sam3dbody_batch.py"),
            "--requests",
            str(request_path),
            "--out",
            str(out_path),
            "--fast-sam-repo",
            str(self._runtime.fast_sam_repo),
            "--checkpoint-dir",
            str(self._runtime.checkpoint_dir),
            "--detector-model",
            self._runtime.detector_model,
            "--detector-name",
            self._runtime.detector_name,
            "--fov-name",
            self._runtime.fov_name,
            "--chunk-format",
            # pickle, NOT binary: measured live on the A100 2026-07-05 (Wolverine,
            # 1177 person-frames) the .npy-sidecar transport regressed the whole
            # BODY dispatch 1057->1301s (handoff 376->490s, wrapper 55->304s,
            # preprocessing 13->137s, steady inference polluted 15.3->151ms/person
            # by in-loop array saves). The pickle chunk path is the proven-fast
            # transport; binary remains available for explicit experiments.
            chunk_format,
            "--no-monolithic-output",
        ]
        if body_input_size_px is not None:
            command.extend(["--body-input-size", str(body_input_size_px)])
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            # A missing/unspawnable FAST_SAM python must degrade to the
            # per-frame runtime path, exactly like a nonzero exit does --
            # subprocess.run raises instead of returning in that case.
            return self._fallback(
                requests,
                reason=f"batch runner spawn failed ({type(exc).__name__}: {exc})",
                clip_intrinsics=clip_intrinsics,
                sam3d_body_input_size_px=sam3d_body_input_size_px,
                crop_bucket_sizes=crop_bucket_sizes,
                torch_compile=torch_compile,
                compile_warmup_buckets=compile_warmup_buckets,
                steady_state_empty_cache=steady_state_empty_cache,
                inner_bucket_sync=inner_bucket_sync,
                upstream_env=upstream_env,
                tier2_output_lite=tier2_output_lite,
            )
        if completed.returncode != 0:
            return self._fallback(
                requests,
                reason=_binary_handoff_failure_reason(completed),
                clip_intrinsics=clip_intrinsics,
                sam3d_body_input_size_px=sam3d_body_input_size_px,
                crop_bucket_sizes=crop_bucket_sizes,
                torch_compile=torch_compile,
                compile_warmup_buckets=compile_warmup_buckets,
                steady_state_empty_cache=steady_state_empty_cache,
                inner_bucket_sync=inner_bucket_sync,
                upstream_env=upstream_env,
                tier2_output_lite=tier2_output_lite,
            )
        index_path = out_path.with_name(f"{out_path.name}.chunks") / "index.json"
        try:
            from scripts.racketsport.run_sam3dbody_batch import load_sam3dbody_binary_outputs_from_chunk_index

            outputs = load_sam3dbody_binary_outputs_from_chunk_index(
                index_path,
                request_ids=request_ids,
                mmap_mode="r",
            )
        except Exception as exc:  # noqa: BLE001 - compatibility fallback reports the exact mismatch.
            return self._fallback(
                requests,
                reason=f"binary sidecar load failed ({type(exc).__name__}: {exc})",
                clip_intrinsics=clip_intrinsics,
                sam3d_body_input_size_px=sam3d_body_input_size_px,
                crop_bucket_sizes=crop_bucket_sizes,
                torch_compile=torch_compile,
                compile_warmup_buckets=compile_warmup_buckets,
                steady_state_empty_cache=steady_state_empty_cache,
                inner_bucket_sync=inner_bucket_sync,
                upstream_env=upstream_env,
                tier2_output_lite=tier2_output_lite,
            )
        self.binary_handoff_status = f"{chunk_format}_chunks_v1"
        self.binary_handoff_note = f"SAM3D subprocess returned BODY records through {chunk_format} chunks (contract v1)"
        return outputs

    def _fallback(self, requests: list[Any], *, reason: str, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.binary_handoff_status = "legacy_fallback"
        self.binary_handoff_note = (
            "SAM3D binary sidecar handoff was unavailable; fell back to legacy subprocess result transport "
            f"for compatibility ({reason})"
        )
        batch_runner = getattr(self._runtime, "process_frame_batches", None)
        if batch_runner is not None:
            return batch_runner(requests, **kwargs)
        # Base runtime contract (process_frame only): degrade per-frame, same
        # guarded convention the stage loop itself uses for unbatched runtimes.
        outputs: list[list[dict[str, Any]]] = []
        for request in requests:
            normalized = _normalize_sam3d_subprocess_request(request)
            outputs.append(
                self._runtime.process_frame(
                    normalized["image_path"],
                    bboxes_xyxy=normalized["bboxes"],
                )
            )
        return outputs


def _normalize_sam3d_subprocess_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("SAM3D subprocess batch request must be a mapping")
    image_path = request.get("image_path", request.get("image"))
    bboxes = request.get("bboxes")
    if bboxes is None and request.get("bbox") is not None:
        bboxes = [request["bbox"]]
    if image_path is None or bboxes is None:
        raise ValueError("SAM3D subprocess batch request requires image_path/image and bboxes/bbox")
    return {
        "request_id": str(request.get("request_id", "")),
        "image_path": Path(image_path),
        "bboxes": [[float(value) for value in bbox] for bbox in bboxes],
        "mask_paths": [Path(path) for path in request.get("mask_paths", []) if path],
        "camera_intrinsics": request.get("camera_intrinsics"),
        "target_representation": str(request.get("target_representation", "world_mesh")),
    }


def _binary_handoff_failure_reason(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or f"exit={completed.returncode}"
    if "unrecognized arguments" in detail and ("--chunk-format" in detail or "--no-monolithic-output" in detail):
        return "runner does not support binary sidecar flags"
    return detail


def _sam3d_batch_timing_sidecars(runtime: Any) -> set[Path]:
    work_dir = getattr(runtime, "work_dir", None)
    if work_dir is None:
        return set()
    root = Path(work_dir)
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.glob("batch_outputs-*.json.timing.json") if path.is_file()}


def _read_new_sam3d_batch_timing(runtime: Any, *, before: set[Path]) -> tuple[dict[str, Any], Path] | None:
    after = _sam3d_batch_timing_sidecars(runtime)
    candidates = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    path = candidates[-1]
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return None
    if payload.get("artifact_type") != "racketsport_sam3dbody_batch_timing":
        return None
    return dict(payload), path


def _timing_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_sam3d_batch_timing(
    phase_timings: dict[str, Any],
    *,
    not_instrumentable: dict[str, str],
    timing_sources: dict[str, str],
    subprocess_timing: tuple[dict[str, Any], Path] | None,
    subprocess_outer_call_s: float,
) -> None:
    phase_timings["subprocess_outer_call_s"] = phase_timings.get("subprocess_outer_call_s", 0.0) + float(
        subprocess_outer_call_s
    )
    if subprocess_timing is None:
        phase_timings["inference_s"] = phase_timings.get("inference_s", 0.0) + float(subprocess_outer_call_s)
        return

    payload, path = subprocess_timing
    timing_sources["sam3d_batch_timing"] = str(path)
    local_model_setup_s = phase_timings.get("model_load_s")
    if local_model_setup_s is not None:
        phase_timings["orchestrator_model_setup_s"] = phase_timings.get("orchestrator_model_setup_s", 0.0) + float(
            local_model_setup_s
        )
    for source_key, target_key in (
        ("model_setup_load_s", "model_load_s"),
        ("compile_warmup_s", "compile_warmup_s"),
        ("steady_inference_s", "inference_s"),
        ("ms_per_person_steady", "ms_per_person_steady"),
        ("request_parse_s", "runner_request_parse_s"),
        ("crop_bucket_tensor_prep_s", "runner_preprocessing_s"),
        ("postprocessing_s", "runner_postprocessing_s"),
        ("result_serialization_handoff_s", "runner_result_serialization_handoff_s"),
        ("other_s", "runner_other_s"),
    ):
        value = _timing_float(payload, source_key)
        if value is not None:
            phase_timings[target_key] = value
    total_s = _timing_float(payload, "total_s")
    if total_s is not None:
        phase_timings["subprocess_wrapper_handoff_s"] = max(0.0, float(subprocess_outer_call_s) - total_s)
    per_bucket = payload.get("per_bucket")
    if isinstance(per_bucket, list):
        phase_timings["per_bucket_timing"] = [dict(item) for item in per_bucket if isinstance(item, Mapping)]
    if phase_timings.get("model_load_s") is not None:
        not_instrumentable.pop("model_load_s", None)
    if phase_timings.get("compile_warmup_s") is not None:
        not_instrumentable.pop("compile_warmup_s", None)


def _normalize_sam3d_upstream_env(raw_env: Mapping[str, Any]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key)
        if key not in SAM3D_UPSTREAM_ENV_WHITELIST:
            raise ValueError(f"unsupported SAM3D upstream env key {key!r}")
        if isinstance(raw_value, bool):
            value = "1" if raw_value else "0"
        elif isinstance(raw_value, int):
            value = str(raw_value)
        elif isinstance(raw_value, str) and raw_value:
            value = raw_value
        else:
            raise ValueError(f"SAM3D upstream env {key} must be a non-empty string, integer, or boolean")
        parsed[key] = value
    return parsed


def _body_record_has_dense_mesh(raw_output: Mapping[str, Any]) -> bool:
    for key in ("pred_vertices", "vertices", "mesh_vertices_xyz"):
        if key not in raw_output or raw_output[key] is None:
            continue
        value = raw_output[key]
        try:
            return len(value) > 0
        except TypeError:
            return True
    return False


def _target_representation_lookup(body_execution: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    for frame in body_execution.get("scheduled_frames", []):
        if not isinstance(frame, Mapping):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        target_representation = str(frame.get("target_representation", "world_mesh"))
        for player_id in frame.get("target_player_ids", []):
            lookup[(frame_idx, int(player_id))] = target_representation
    return lookup


def _body_request_metadata_lookup(body_execution: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in body_execution.get("scheduled_frames", []):
        if not isinstance(frame, Mapping):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        frame_reasons = [str(reason) for reason in frame.get("reasons", [])]
        player_targets = {
            int(target.get("player_id")): target
            for target in frame.get("player_targets", [])
            if isinstance(target, Mapping) and target.get("player_id") is not None
        }
        for player_id in frame.get("target_player_ids", []):
            player_id_int = int(player_id)
            target = player_targets.get(player_id_int, {})
            reasons = target.get("reasons", frame_reasons) if isinstance(target, Mapping) else frame_reasons
            lookup[(frame_idx, player_id_int)] = {
                "reasons": [str(reason) for reason in reasons],
                "recommended_tier": str(frame.get("recommended_tier", "")),
                "target_representation": str(frame.get("target_representation", "")),
            }
    return lookup


def _ensure_body_frame_plan_from_sam3d(context: StageContext, tracks: Tracks) -> dict[str, Any]:
    frame_plan_path = context.run_dir / "frame_compute_plan.json"
    if frame_plan_path.is_file():
        return {
            "source": "existing_frame_compute_plan",
            "contact_event_count": _contact_event_count(_read_optional_json(context.run_dir / "contact_windows.json")),
            "generated_artifacts": [],
            "notes": [],
        }

    generated_artifacts: list[str] = []
    notes: list[str] = []
    skeleton_path = _existing_file(context.run_dir / "skeleton3d.json")
    if skeleton_path is None:
        return {
            "source": "missing_sam3d_skeleton3d",
            "contact_event_count": 0,
            "generated_artifacts": generated_artifacts,
            "notes": ["frame_compute_plan.json not derived because skeleton3d.json is missing"],
        }
    skeleton_payload = _read_json(skeleton_path)
    if not _is_real_sam3d_skeleton3d(skeleton_payload):
        return {
            "source": "invalid_sam3d_skeleton3d",
            "contact_event_count": 0,
            "generated_artifacts": generated_artifacts,
            "notes": ["frame_compute_plan.json not derived because skeleton3d.json is not a real SAM-3D skeleton"],
        }

    wrist_path = _first_existing_artifact(context, "wrist_velocity_peaks.json")
    wrist_payload = _read_json(wrist_path) if wrist_path is not None else None
    if not _is_current_sam3d_wrist_velocity_peaks(wrist_payload, skeleton_path=skeleton_path):
        wrist_payload = build_wrist_velocity_peaks_from_file(skeleton_path, require_lane_a=False)
        wrist_path = context.run_dir / "wrist_velocity_peaks.json"
        _write_json(wrist_path, wrist_payload)
        generated_artifacts.append("wrist_velocity_peaks.json")
        notes.append("derived wrist_velocity_peaks.json from SAM-3D skeleton3d.json")
    elif wrist_path.parent != context.run_dir:
        wrist_path = context.run_dir / "wrist_velocity_peaks.json"
        _write_json(wrist_path, wrist_payload)
        generated_artifacts.append("wrist_velocity_peaks.json")
        notes.append("copied SAM-3D wrist_velocity_peaks.json into run output")

    ball_inflections_path = _first_existing_artifact(context, "ball_inflections.json")
    if ball_inflections_path is None:
        ball_track_path = _first_existing_artifact(context, "ball_track.json") or _existing_file(context.ball_source_path)
        if ball_track_path is None:
            return {
                "source": "missing_ball_inflections",
                "contact_event_count": 0,
                "generated_artifacts": generated_artifacts,
                "notes": [*notes, "frame_compute_plan.json not derived because ball_inflections.json is missing"],
            }
        ball_payload = _read_json(ball_track_path)
        if not isinstance(ball_payload, Mapping):
            return {
                "source": "invalid_ball_track",
                "contact_event_count": 0,
                "generated_artifacts": generated_artifacts,
                "notes": [*notes, "frame_compute_plan.json not derived because ball_track.json is not an object"],
            }
        ball_inflections = build_ball_inflections_from_ball_track(ball_payload)
        ball_inflections_path = context.run_dir / "ball_inflections.json"
        _write_json(ball_inflections_path, ball_inflections)
        generated_artifacts.append("ball_inflections.json")
        notes.append("derived ball_inflections.json from ball_track.json for BODY contact scheduling")

    contact_windows_path = _first_existing_artifact(context, "contact_windows.json")
    if contact_windows_path is None:
        # Prefer audio_onsets_v2.json (pop-tuned detector) over the older
        # audio_onsets.json when both exist; fall back to [] (no audio cues)
        # when neither is present -- this preserves fail-closed behavior for
        # clips without audio. Audio only ever refines contact timing here
        # (require_audio=False, fuse_contact_windows_from_cue_payloads already
        # treats it as one of several cues); it never overrides or gates on
        # visual (wrist/ball) evidence, and does not relax any downstream gate
        # (e.g. ball_bounce_gate.py's M4 stays at its own fail-closed threshold).
        audio_onsets_path = _first_existing_artifact(context, "audio_onsets_v2.json") or _first_existing_artifact(
            context, "audio_onsets.json"
        )
        audio_onsets_payload = _read_json(audio_onsets_path) if audio_onsets_path is not None else []
        contact_windows = fuse_contact_windows_from_cue_payloads(
            fps=tracks.fps,
            audio_onsets_payload=audio_onsets_payload,
            wrist_velocity_peaks_payload=_read_json(wrist_path),
            ball_inflections_payload=_read_json(ball_inflections_path),
            require_audio=False,
        )
        contact_windows_path = context.run_dir / "contact_windows.json"
        _write_json(contact_windows_path, contact_windows)
        generated_artifacts.append("contact_windows.json")
        if audio_onsets_path is not None:
            notes.append(
                f"fused contact_windows.json from SAM-3D wrists, ball inflections, and audio cues ({audio_onsets_path.name}); "
                "audio only refines timing, never overrides visual evidence"
            )
        else:
            notes.append("fused contact_windows.json from SAM-3D wrists and ball inflections (no audio_onsets artifact found)")
    else:
        contact_windows = _read_json(contact_windows_path)
        if contact_windows_path.parent != context.run_dir:
            contact_windows_path = context.run_dir / "contact_windows.json"
            _write_json(contact_windows_path, contact_windows)
            generated_artifacts.append("contact_windows.json")
            notes.append("copied contact_windows.json into run output for BODY scheduling")

    ball_track_path = _first_existing_artifact(context, "ball_track.json") or _existing_file(context.ball_source_path)
    ball_track = None
    if ball_track_path is not None:
        maybe_ball_track = _read_json(ball_track_path)
        if isinstance(maybe_ball_track, Mapping):
            ball_track = maybe_ball_track
    frame_plan = build_frame_compute_plan(
        tracks,
        ball_track=ball_track,
        contact_windows=contact_windows,
        expected_players=context.expected_players,
    )
    write_frame_compute_plan(frame_plan_path, frame_plan)
    generated_artifacts.append("frame_compute_plan.json")
    return {
        "source": "sam3d_wrist_velocity_peaks",
        "contact_event_count": _contact_event_count(contact_windows),
        "generated_artifacts": generated_artifacts,
        "notes": notes,
    }


def _first_existing_artifact(context: StageContext, filename: str) -> Path | None:
    for root in (context.run_dir, context.inputs_dir):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _contact_event_count(payload: Any | None) -> int:
    if not isinstance(payload, Mapping):
        return 0
    events = payload.get("events")
    return len(events) if isinstance(events, list) else 0


def _is_real_sam3d_skeleton3d(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    provenance = payload.get("provenance")
    joint_names = payload.get("joint_names")
    model_candidates = {str(payload.get("source_model", ""))}
    if isinstance(provenance, Mapping):
        for key in ("source", "model_family", "skeleton_source"):
            value = provenance.get(key)
            if value is not None:
                model_candidates.add(str(value))
    return (
        payload.get("artifact_type") == "racketsport_skeleton3d"
        and payload.get("preview_only") is False
        and isinstance(joint_names, list)
        and len(joint_names) == 70
        and bool({"sam3d_body_joints", "sam3dbody_world_joints"}.intersection(model_candidates))
    )


def _is_current_sam3d_wrist_velocity_peaks(payload: Any, *, skeleton_path: Path) -> bool:
    if not isinstance(payload, Mapping):
        return False
    source_provenance = payload.get("source_provenance")
    return (
        payload.get("artifact_type") == "racketsport_wrist_velocity_peaks"
        and payload.get("source") == "sam3d_body_skeleton3d_world_joints"
        and isinstance(source_provenance, Mapping)
        and source_provenance.get("preview_only") is False
        and source_provenance.get("joint_count") == 70
        and _same_artifact_path(payload.get("source_path"), skeleton_path)
    )


def _same_artifact_path(value: Any, path: Path) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).resolve() == path.resolve()
    except OSError:
        return Path(str(value)) == path


def _body_joint_quality_after_full_clip_gate(
    quality: Mapping[str, Any],
    full_clip_gate: Mapping[str, Any],
) -> dict[str, Any]:
    if full_clip_gate.get("passed") is not True:
        return dict(quality)
    updated = dict(quality)
    quality_blockers = _string_items(updated.get("quality_blockers"))
    promotion_blockers = [
        blocker
        for blocker in _string_items(updated.get("promotion_blockers"))
        if blocker != "missing_full_clip_body_gate"
    ]
    updated["promotion_blockers"] = promotion_blockers
    updated["blockers"] = _dedupe_strings([*quality_blockers, *promotion_blockers])
    return updated


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _default_runners(
    *,
    tracking_mode: Literal["real", "precomputed", "precomputed_tracks"],
    tracking_video: str | Path | None,
    manifest_path: str | Path,
    tracker_config_path: str | Path,
    max_players: int,
    court_margin_m: float,
    id_strategy: str,
    ball_source_path: str | Path | None,
) -> dict[str, StageRunner]:
    if tracking_mode == "precomputed":
        tracking_runner: StageRunner = PrecomputedTrackingRunner(
            max_players=max_players,
            court_margin_m=court_margin_m,
            id_strategy=id_strategy,
        )
    elif tracking_mode == "precomputed_tracks":
        tracking_runner = PrecomputedTracksRunner()
    elif tracking_mode == "real":
        tracking_runner = RealYOLO26BoTSORTReIDTrackingRunner(
            manifest_path=manifest_path,
            tracker_config_path=tracker_config_path,
            video_path=tracking_video,
            max_players=max_players,
            court_margin_m=court_margin_m,
            id_strategy=id_strategy,
        )
    else:
        raise ValueError(f"unknown tracking_mode: {tracking_mode}")
    return {
        "calibration": ManualCalibrationRunner(),
        "tracking": tracking_runner,
        "pose": PoseStageRunner(manifest_path=manifest_path),
        "body": BodyStageRunner(manifest_path=manifest_path),
        "ball_events": BallStageRunner(source_path=ball_source_path),
        "racket": RacketStageRunner(),
    }


def run_pipeline(
    *,
    clip: str,
    inputs_dir: str | Path,
    run_dir: str | Path,
    stage: str = "e2e",
    sport: Sport = "pickleball",
    runners: dict[str, StageRunner] | None = None,
    device: str | None = None,
    max_frames: int | None = None,
    tracking_mode: Literal["real", "precomputed", "precomputed_tracks"] = "real",
    tracking_video: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
    tracker_config_path: str | Path = DEFAULT_BOTSORT_REID_CONFIG,
    max_players: int = 4,
    court_margin_m: float = 0.0,
    id_strategy: str = "auto",
    ball_source_path: str | Path | None = None,
    reuse_existing_stage_artifacts: bool = False,
) -> dict[str, Any]:
    """Run the pipeline through ``stage`` and stop rather than fabricate artifacts.

    ``reuse_existing_stage_artifacts`` (Task #45 S2, default ``False``): by default this
    function re-derives every dependency stage in ``stage``'s closure on every call --
    e.g. calling ``run_pipeline(stage="tracking", ...)`` always re-runs "calibration"
    first, even if a prior ``run_pipeline(stage="calibration", ...)`` call against the
    same ``run_dir`` already wrote a valid ``court_calibration.json``. That is
    deliberately left as this function's default behavior (every existing caller/test
    relies on it, and it is the correct behavior for a fresh/standalone run), but it is
    wasteful and can be actively wrong for a caller like
    ``scripts/racketsport/process_video.py`` that invokes this function once per stage
    against the *same* clip directory over the course of one pipeline run: a dependency
    stage's automatic evidence check can behave differently the second time it runs (see
    ``_raise_if_video_evidence_not_ready``), or the dependency runner registered for this
    call may not even be able to re-derive it at all (``ExternalCalibrationRunner``
    deliberately never writes a ``capture_sidecar.json``, so a plain
    ``ManualCalibrationRunner`` re-derivation attempt inside a later stage's dependency
    walk fails with "missing calibration sidecar" even though a good calibration already
    exists on disk). When a caller opts in with ``reuse_existing_stage_artifacts=True``,
    any dependency contract (every contract in the walk *except* the one actually
    requested via ``stage``) whose required artifacts are already present and schema-valid
    on disk is treated as authoritative and its runner is not invoked again. This does
    NOT change re-derivation behavior for the requested ``stage`` itself (a caller that
    asks to run "tracking" always gets a real tracking attempt), and it does not change
    anything for callers that leave the default ``False`` -- documented here rather than
    changed globally, since other callers/tests intentionally exercise full re-derivation.
    """

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    context = StageContext(
        clip=clip,
        inputs_dir=Path(inputs_dir),
        run_dir=run_path,
        sport=sport,
        device=device,
        max_frames=max_frames,
        expected_players=max_players,
        tracking_video=Path(tracking_video) if tracking_video is not None else None,
        ball_source_path=Path(ball_source_path) if ball_source_path is not None else None,
    )
    registry = _default_runners(
        tracking_mode=tracking_mode,
        tracking_video=tracking_video,
        manifest_path=manifest_path,
        tracker_config_path=tracker_config_path,
        max_players=max_players,
        court_margin_m=court_margin_m,
        id_strategy=id_strategy,
        ball_source_path=ball_source_path,
    )
    if runners:
        registry.update(runners)

    stage_runs: list[dict[str, Any]] = []
    summary_status = PIPELINE_STATUS_PASS

    for contract in _ordered_contracts_for(stage):
        stage_wall_start = time.perf_counter()
        runner = registry.get(contract.stage)
        if runner is None:
            stage_runs.append(
                _with_stage_wall_seconds(
                    _blocked_stage_run(contract, f"no runner registered for stage: {contract.stage}"),
                    stage_wall_start,
                ).as_dict()
            )
            summary_status = PIPELINE_STATUS_BLOCKED
            break

        if (
            reuse_existing_stage_artifacts
            and contract.stage != stage
            and _contract_artifacts_already_valid(contract, run_path)
        ):
            stage_runs.append(_with_stage_wall_seconds(_reused_stage_run(contract, runner, run_path), stage_wall_start).as_dict())
            continue

        try:
            result = runner.run(context)
            _validate_contract_artifacts(contract, run_path)
        except Exception as exc:
            stage_runs.append(
                _with_stage_wall_seconds(
                    StageRun(
                        stage=contract.stage,
                        status=PIPELINE_STATUS_FAIL,
                        real_model=getattr(runner, "real_model", False),
                        source_mode=getattr(runner, "source_mode", "unknown"),
                        notes=(f"{contract.stage} failed: {exc}",),
                    ),
                    stage_wall_start,
                ).as_dict()
            )
            summary_status = PIPELINE_STATUS_FAIL
            break
        result = _with_stage_wall_seconds(result, stage_wall_start)
        stage_runs.append(result.as_dict())
        if result.status == PIPELINE_STATUS_FAIL:
            summary_status = PIPELINE_STATUS_FAIL
            break
        if result.status == PIPELINE_STATUS_BLOCKED:
            summary_status = PIPELINE_STATUS_BLOCKED
            break

    review_artifacts = _write_best_effort_review_artifacts(
        context,
        expected_players=max_players,
        protected_artifacts=_successful_stage_artifacts(stage_runs),
    )
    readiness = build_readiness_report(run_path, stage=stage)
    if summary_status == PIPELINE_STATUS_PASS and readiness["status"] != "ready":
        summary_status = PIPELINE_STATUS_BLOCKED
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_pipeline_run",
        "clip": clip,
        "requested_stage": stage,
        "status": summary_status,
        "run_dir": str(run_path),
        "inputs_dir": str(context.inputs_dir),
        "stages": stage_runs,
        "review_artifacts": review_artifacts,
        "readiness": readiness,
    }
    _write_json(run_path / "pipeline_run.json", summary)
    return summary


def _ordered_contracts_for(stage: str) -> list[PipelineStageContract]:
    contracts_by_stage = {contract.stage: contract for contract in PIPELINE_STAGE_CONTRACTS}
    if stage not in contracts_by_stage:
        valid = ", ".join(contract.stage for contract in PIPELINE_STAGE_CONTRACTS)
        raise PipelineContractError(f"unknown pipeline stage: {stage}; expected one of: {valid}")

    needed: set[str] = set()

    def visit(current: str) -> None:
        if current in needed:
            return
        contract = contracts_by_stage[current]
        for dependency in contract.depends_on:
            visit(dependency)
        needed.add(current)

    visit(stage)
    return [contract for contract in PIPELINE_STAGE_CONTRACTS if contract.stage in needed]


#: Artifacts a contract in ``pipeline_contracts.PIPELINE_STAGE_CONTRACTS`` lists as
#: "required" for readiness bookkeeping/other tooling, but that no stage on the
#: ``scripts/racketsport/process_video.py`` product path actually reads back as an
#: *input* (Task #45 S4; verified by grepping every read of these filenames across
#: ``scripts/racketsport/process_video.py`` and every module it imports artifacts
#: through -- ``court_zones.json`` is read only by the offline
#: ``threed.racketsport.court_keypoint_eval`` eval script, and ``net_plane.json`` only by
#: review/visualization tooling (``threed.racketsport.calibration_overlay``,
#: ``threed.racketsport.review_action_manifest``), neither of which sits on the
#: process_video.py stage path). They stay pure, cheap functions of ``sport``
#: (``build_court_zones``/``build_net_plane``) so every calibration runner keeps writing
#: them for those other consumers -- this only stops a missing/invalid copy from
#: hard-failing the *stage itself* here. This intentionally does NOT touch
#: ``pipeline_contracts.PIPELINE_STAGE_CONTRACTS`` (out of this lane's scope and used by
#: other stages/lanes); it only relaxes this function's own post-run validation.
_SOFT_REQUIRED_ARTIFACTS: dict[str, frozenset[str]] = {
    "calibration": frozenset({"court_zones.json", "net_plane.json"}),
}

# Contract artifacts a stage may legitimately omit by configuration, validated
# only when present: BODY writes smpl_motion.json only under
# write_body_monoliths=True (fetch_body_monoliths); the slim speed default
# ships skeleton3d.json + body_mesh_index/ instead, so absence is legal while
# a present-but-invalid file must still fail.
_PRESENT_ONLY_REQUIRED_ARTIFACTS: dict[str, frozenset[str]] = {
    "body": frozenset({"smpl_motion.json"}),
}


def _validate_contract_artifacts(contract: PipelineStageContract, run_dir: Path) -> None:
    soft = _SOFT_REQUIRED_ARTIFACTS.get(contract.stage, frozenset())
    present_only = _PRESENT_ONLY_REQUIRED_ARTIFACTS.get(contract.stage, frozenset())
    for artifact in contract.required_artifacts:
        if artifact in soft:
            continue
        if artifact in present_only and not (run_dir / artifact).is_file():
            continue
        schema_name = ARTIFACT_SCHEMA_BY_FILENAME.get(artifact)
        if schema_name is None:
            raise ValueError(f"no schema mapping for required artifact: {artifact}")
        validate_artifact_file(schema_name, run_dir / artifact)


def _contract_artifacts_already_valid(contract: PipelineStageContract, run_dir: Path) -> bool:
    """Task #45 S2: has ``contract`` already produced valid artifacts on ``run_dir``?

    Used only when a caller opts into ``run_pipeline(reuse_existing_stage_artifacts=True)``
    -- see that parameter's docstring. Reuses the same (S4-relaxed) validation
    ``_validate_contract_artifacts`` performs right after a runner actually runs, so
    "already valid" means exactly what "just ran successfully" would have meant.
    """

    if not contract.required_artifacts:
        return False
    try:
        _validate_contract_artifacts(contract, run_dir)
    except Exception:
        return False
    return True


def _reused_stage_run(contract: PipelineStageContract, runner: StageRunner, run_dir: Path) -> StageRun:
    present = tuple(name for name in contract.required_artifacts if (run_dir / name).is_file())
    return StageRun(
        stage=contract.stage,
        status="ran",
        real_model=getattr(runner, "real_model", False),
        source_mode="reused_existing_run_artifacts",
        produced_artifacts=present,
        notes=(
            f"{contract.stage} artifacts already valid on disk for this run and "
            "reuse_existing_stage_artifacts=True was requested by the caller; treating this "
            "completed stage's artifacts as authoritative instead of re-deriving/re-validating "
            "it again (Task #45 S2) -- re-derivation was skipped, the registered "
            f"{type(runner).__name__} runner for {contract.stage!r} was not invoked",
        ),
    )


def _with_stage_wall_seconds(stage_run: StageRun, started: float) -> StageRun:
    if stage_run.wall_seconds is not None:
        return stage_run
    return replace(stage_run, wall_seconds=max(0.0, time.perf_counter() - started))


def _blocked_stage_run(contract: PipelineStageContract, note: str) -> StageRun:
    return StageRun(
        stage=contract.stage,
        status=PIPELINE_STATUS_BLOCKED,
        real_model=False,
        source_mode="unregistered",
        notes=(note,),
    )


def _successful_stage_artifacts(stage_runs: Sequence[dict[str, Any]]) -> set[str]:
    protected: set[str] = set()
    for stage_run in stage_runs:
        if stage_run.get("status") in {PIPELINE_STATUS_FAIL, PIPELINE_STATUS_BLOCKED}:
            continue
        produced = stage_run.get("produced_artifacts", [])
        if isinstance(produced, list):
            protected.update(str(artifact) for artifact in produced)
    return protected


def _write_best_effort_review_artifacts(
    context: StageContext,
    *,
    expected_players: int,
    protected_artifacts: set[str] | None = None,
) -> dict[str, Any]:
    produced_artifacts: list[str] = []
    reused_artifacts: list[str] = []
    notes: list[str] = []
    protected = protected_artifacts or set()

    tracks_path = context.run_dir / "tracks.json"
    court_path = context.run_dir / "court_calibration.json"
    ball_path = _existing_file(context.run_dir / "ball_track.json") or _existing_file(context.ball_source_path)
    ball_physics_path = _existing_file(context.run_dir / "ball_track_physics_filled.json")
    contact_windows_path = _existing_file(context.run_dir / "contact_windows.json")
    racket_path = _existing_file(context.run_dir / "racket_pose.json")
    racket_estimate_path = _existing_file(context.run_dir / "racket_pose_estimate.json")
    physics_footlock_path = _existing_file(context.run_dir / "physics_footlock.json")
    smpl_path = _existing_file(context.run_dir / "smpl_motion.json")
    skeleton_path = _existing_file(context.run_dir / "skeleton3d.json")

    if tracks_path.is_file():
        frame_plan_path = context.run_dir / "frame_compute_plan.json"
        protect_body_plan = bool(
            protected.intersection(
                {
                    "body_compute_execution.json",
                    "body_mesh_readiness.json",
                    "smpl_motion.json",
                    "skeleton3d.json",
                }
            )
        )
        if protect_body_plan and frame_plan_path.is_file():
            reused_artifacts.append("frame_compute_plan.json")
        else:
            try:
                frame_plan = build_frame_compute_plan_from_files(
                    tracks_path=tracks_path,
                    ball_track_path=ball_path,
                    contact_windows_path=contact_windows_path,
                    expected_players=expected_players,
                )
                write_frame_compute_plan(frame_plan_path, frame_plan)
                produced_artifacts.append("frame_compute_plan.json")
            except Exception as exc:
                notes.append(f"frame_compute_plan.json not written: {exc}")

        body_execution_path = context.run_dir / "body_compute_execution.json"
        body_execution_payload = _read_optional_json(body_execution_path) if body_execution_path.is_file() else None
        reuse_existing_body_execution = (
            "body_compute_execution.json" in protected
            or (
                isinstance(body_execution_payload, Mapping)
                and body_execution_payload.get("fail_closed_reason") == "manifest_asset_preflight_sha256_mismatch"
            )
        )
        if body_execution_path.is_file() and reuse_existing_body_execution:
            reused_artifacts.append("body_compute_execution.json")
        else:
            try:
                tracks = validate_artifact_file("tracks", tracks_path)
                if not isinstance(tracks, Tracks):
                    raise ValueError("tracks artifact did not parse as Tracks")
                body_execution = build_body_compute_execution(
                    tracks,
                    frame_plan_path=context.run_dir / "frame_compute_plan.json",
                    max_frames=context.max_frames,
                )
                write_body_compute_execution(body_execution_path, body_execution)
                produced_artifacts.append("body_compute_execution.json")
            except Exception as exc:
                notes.append(f"body_compute_execution.json not written: {exc}")

    if court_path.is_file():
        try:
            virtual_world = build_virtual_world_state_from_files(
                court_calibration_path=court_path,
                tracks_path=tracks_path if tracks_path.is_file() else None,
                smpl_motion_path=smpl_path,
                skeleton3d_path=skeleton_path,
                ball_track_path=ball_path,
                racket_pose_path=racket_path,
                physics_footlock_path=physics_footlock_path,
                ball_track_physics_filled_path=ball_physics_path,
                racket_pose_estimate_path=racket_estimate_path,
            )
            write_virtual_world(context.run_dir / "virtual_world.json", virtual_world)
            produced_artifacts.append("virtual_world.json")
        except Exception as exc:
            notes.append(f"virtual_world.json not written: {exc}")

    return {
        "produced_artifacts": produced_artifacts,
        "reused_artifacts": reused_artifacts,
        "notes": notes,
    }


def _existing_file(path: Path | None) -> Path | None:
    return path if path is not None and path.is_file() else None


def _verified_botsort_reid_config(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing BoT-SORT ReID tracker config: {path}")
    text = path.read_text(encoding="utf-8")
    required = {
        "tracker_type: botsort": "BoT-SORT tracker_type",
        "with_reid: True": "with_reid enabled",
    }
    for needle, label in required.items():
        if needle not in text:
            raise ValueError(f"tracker config {path} does not declare {label}")
    return path


def _tracking_video_path(context: StageContext, *, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    elif context.tracking_video is not None:
        candidates = [context.tracking_video]
    else:
        preferred = [
            context.inputs_dir / "source.mp4",
            context.inputs_dir / "clip.mp4",
            context.inputs_dir / "video.mp4",
            context.inputs_dir / "input.mp4",
        ]
        discovered = sorted(path for path in context.inputs_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
        candidates = preferred + discovered

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "real tracking requires a source video file; expected --tracking-video or a video in inputs_dir, "
        "and will not fall back to detections.json"
    )


def _tracking_fps(inputs_dir: Path, *, video_path: Path) -> float:
    sidecar_path = inputs_dir / "capture_sidecar.json"
    if sidecar_path.is_file():
        sidecar = validate_artifact_file("capture_sidecar", sidecar_path)
        if isinstance(sidecar, CaptureSidecar):
            return float(sidecar.fps)
        raise ValueError("capture_sidecar.json did not validate as CaptureSidecar")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"missing capture_sidecar.json and cv2 is unavailable to read FPS from {video_path}") from exc

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"could not determine FPS for tracking video: {video_path}")
    return fps


def _calibration_line_evidence(context: StageContext, *, calibration: CourtCalibration, net_plane: Any) -> tuple[Any, tuple[str, ...]]:
    video_path = _calibration_video_path(context)
    if video_path is not None:
        try:
            evidence = build_auto_court_line_evidence_from_video(
                video_path,
                calibration,
                net_plane=net_plane,
                sample_count=7,
            )
        except Exception as exc:
            evidence = _fail_closed_court_line_evidence(
                context,
                source="auto_video_evidence_failed",
                reason=f"video_evidence_failed:{type(exc).__name__}",
            )
            return evidence, (
                f"court_line_evidence.json is fail-closed because automatic video evidence failed for {video_path}: {exc}",
            )
        return evidence, (f"court_line_evidence.json generated automatically from video {video_path}",)

    frame_path = _calibration_frame_path(context)
    if frame_path is None:
        evidence = _fail_closed_court_line_evidence(
            context,
            source="manual_sidecar_no_auto_frame",
            reason="missing_auto_frame_or_video",
        )
        evidence.source = "manual_sidecar_no_auto_frame"
        return evidence, (
            "court_line_evidence.json is fail-closed because no calibration frame was available for automatic line/net detection",
        )

    evidence = build_auto_court_line_evidence_from_frame(
        frame_path,
        calibration,
        net_plane=net_plane,
        frame_index=0,
    )
    return evidence, (f"court_line_evidence.json generated automatically from {frame_path}",)


def _unseeded_calibration_line_evidence(context: StageContext) -> tuple[Any, tuple[str, ...]]:
    video_path = _calibration_video_path(context)
    if video_path is not None:
        evidence = _fail_closed_court_line_evidence(
            context,
            source="auto_video_no_calibration_seed",
            reason="missing_calibration_seed",
        )
        return evidence, (
            f"video court evidence attempt recorded from {video_path}, but calibration is fail-closed without a trusted sidecar or no-tap solve",
        )

    frame_path = _calibration_frame_path(context)
    if frame_path is not None:
        evidence = _fail_closed_court_line_evidence(
            context,
            source="auto_frame_no_calibration_seed",
            reason="missing_calibration_seed",
        )
        return evidence, (
            f"frame court evidence attempt recorded from {frame_path}, but calibration is fail-closed without a trusted sidecar or no-tap solve",
        )

    evidence = _fail_closed_court_line_evidence(
        context,
        source="no_auto_court_source",
        reason="missing_auto_frame_or_video",
    )
    evidence.aggregate.reasons.append("missing_calibration_seed")
    return evidence, (
        "court calibration is fail-closed because no video/frame and no trusted calibration sidecar were available",
    )


def _raise_if_video_evidence_not_ready(context: StageContext, evidence: Any, *, trusted: bool = False) -> str | None:
    """Fail-closed gate on the automatic court-line/net evidence detector.

    Task #45 S1: this automatic no-tap evidence detector is a future-facing signal --
    today it cannot reliably see every court's net/lines from a single video (see
    runs/v1_coldstart_20260702T061658Z/summary.json's ``missing_top_net`` finding,
    reproduced across all 4 eval clips regardless of calibration source). When
    ``trusted`` is True the caller has already established this run's
    ``court_calibration.json`` came from a real, owner-provided/reviewed source
    (``ExternalCalibrationRunner`` with an intrinsics.source in
    ``TRUSTED_INTRINSICS_SOURCES``, or ``ManualCalibrationRunner``'s human-tapped-corners
    branch) -- for those sources a not-ready automatic evidence result is returned as an
    advisory note instead of raised, so the one calibration input the current v1
    owner-tap product path has does not get blocked by a gate designed for a different,
    not-yet-built no-tap product surface. Untrusted/no-tap calibration (this function
    called with the default ``trusted=False``) keeps the exact fail-closed behavior this
    function always had -- see ``ManualCalibrationRunner``'s no-tap ARKit-keypoints
    branch, which deliberately opts out of ``trusted`` too.
    """

    if _calibration_video_path(context) is None:
        return None
    aggregate = getattr(evidence, "aggregate", None)
    if aggregate is None or getattr(aggregate, "auto_calibration_ready", False):
        return None
    reasons = ", ".join(getattr(aggregate, "reasons", []) or ["unknown"])
    message = f"automatic court evidence not ready for video-backed run: {reasons}"
    if trusted:
        return (
            f"ADVISORY (not blocking -- trusted calibration source): {message}; the automatic "
            "no-tap court-evidence detector is not yet a blocking gate for a trusted, "
            "owner-provided calibration -- see this stage's calibration_confidence metric and "
            "downstream trust bands for the honest confidence signal instead"
        )
    raise ValueError(message)


def _fail_closed_court_line_evidence(context: StageContext, *, source: str, reason: str) -> Any:
    evidence = aggregate_court_line_evidence(
        sport=context.sport,
        line_observations=[],
        net_observations=[],
        required_line_ids=required_court_line_ids(context.sport),
        required_net_ids=required_court_net_ids(context.sport),
    )
    evidence.source = source
    if reason not in evidence.aggregate.reasons:
        evidence.aggregate.reasons.append(reason)
    return evidence


def _calibration_video_path(context: StageContext) -> Path | None:
    candidates: list[Path] = []
    if context.tracking_video is not None:
        candidates.append(context.tracking_video)
    candidates.extend(
        [
            context.inputs_dir / "source.mp4",
            context.inputs_dir / "clip.mp4",
            context.inputs_dir / "video.mp4",
            context.inputs_dir / "input.mp4",
        ]
    )
    if context.inputs_dir.is_dir():
        candidates.extend(sorted(path for path in context.inputs_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return candidate
    return None


def _calibration_frame_path(context: StageContext) -> Path | None:
    names = (
        "calibration_frame.jpg",
        "calibration_frame.png",
        "court_frame.jpg",
        "court_frame.png",
        "frame_000000.jpg",
        "frame_000000.png",
        "frame_000001.jpg",
        "frame_000001.png",
    )
    roots = (
        context.inputs_dir,
        context.inputs_dir / "frames",
        context.inputs_dir / "calibration_frames",
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _detection_bbox_scale(
    calibration: CourtCalibration,
    video_path: Path,
    *,
    source_size: tuple[float, float] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    if source_size is None:
        source_size = _video_source_size(video_path)
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("tracking source video dimensions are unavailable; refusing identity bbox scaling")
    target_width, target_height = calibration_image_size(calibration, fallback_target=(source_width, source_height))

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return (
        scale_x,
        scale_y,
        {
            "bbox_scale_x": round(scale_x, 6),
            "bbox_scale_y": round(scale_y, 6),
            "source_width": int(source_width),
            "source_height": int(source_height),
            "calibration_width": int(target_width),
            "calibration_height": int(target_height),
            "bbox_scale_status": "scaled" if scale_x != 1.0 or scale_y != 1.0 else "identity",
        },
    )


def _payload_source_size(payload: dict[str, Any]) -> tuple[float, float] | None:
    width = payload.get("source_width")
    height = payload.get("source_height")
    if isinstance(width, int | float) and isinstance(height, int | float):
        return float(width), float(height)
    return None


def _video_source_size(video_path: Path) -> tuple[float, float]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("tracking source video dimensions are unavailable because cv2 is not installed") from exc

    cap = cv2.VideoCapture(str(video_path))
    try:
        source_width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if source_width <= 0 or source_height <= 0:
        raise ValueError("tracking source video dimensions are unavailable; refusing identity bbox scaling")
    return source_width, source_height


def _detections_payload_from_tracked_results(results: Any, *, fps: float, max_frames: int | None) -> tuple[dict[str, Any], dict[str, int]]:
    frames: list[dict[str, Any]] = []
    counts = {
        "tracker_frames": 0,
        "tracker_boxes": 0,
        "tracked_person_boxes": 0,
        "untracked_person_boxes": 0,
        "tracker_non_person": 0,
    }
    source_size: tuple[float, float] | None = None
    for frame_index, result in enumerate(results):
        if max_frames is not None and frame_index >= max_frames:
            break
        counts["tracker_frames"] += 1
        result_size = _result_source_size(result)
        if result_size is not None:
            if source_size is None:
                source_size = result_size
            elif result_size != source_size:
                raise ValueError(f"tracking result source size changed across frames: {source_size} then {result_size}")
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", []) or []
        for box in boxes:
            counts["tracker_boxes"] += 1
            cls = int(_box_scalar(getattr(box, "cls", 0)))
            if cls != 0:
                counts["tracker_non_person"] += 1
                continue
            track_id_raw = getattr(box, "id", None)
            if track_id_raw is None:
                counts["untracked_person_boxes"] += 1
                continue
            detections.append(
                {
                    "bbox": _box_xyxy(box),
                    "conf": float(_box_scalar(getattr(box, "conf", 1.0))),
                    "class": "person",
                    "track_id": int(_box_scalar(track_id_raw)),
                }
            )
            counts["tracked_person_boxes"] += 1
        frames.append({"frame": frame_index, "detections": detections})

    if not frames:
        raise ValueError("real tracking produced no frames")
    payload: dict[str, Any] = {"fps": fps, "frames": frames}
    if source_size is not None:
        source_width, source_height = source_size
        payload["source_width"] = source_width
        payload["source_height"] = source_height
        counts["tracker_source_width"] = int(source_width)
        counts["tracker_source_height"] = int(source_height)
    return payload, counts


def _result_source_size(result: Any) -> tuple[float, float] | None:
    orig_shape = getattr(result, "orig_shape", None)
    if isinstance(orig_shape, list | tuple) and len(orig_shape) >= 2:
        height = float(orig_shape[0])
        width = float(orig_shape[1])
        if width > 0 and height > 0:
            return width, height
    orig_img = getattr(result, "orig_img", None)
    shape = getattr(orig_img, "shape", None)
    if isinstance(shape, list | tuple) and len(shape) >= 2:
        height = float(shape[0])
        width = float(shape[1])
        if width > 0 and height > 0:
            return width, height
    return None


def _box_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "cpu"):
        return _box_scalar(value.cpu())
    if isinstance(value, list | tuple):
        if not value:
            raise ValueError("empty scalar tensor/list")
        return _box_scalar(value[0])
    return float(value)


def _box_xyxy(box: Any) -> list[float]:
    value = getattr(box, "xyxy")
    if isinstance(value, list | tuple) and len(value) == 4 and not hasattr(value[0], "cpu"):
        return [float(item) for item in value]
    first = value[0] if isinstance(value, list | tuple) or hasattr(value, "__getitem__") else value
    if hasattr(first, "cpu"):
        first = first.cpu()
    if hasattr(first, "tolist"):
        first = first.tolist()
    if not isinstance(first, list | tuple) or len(first) != 4:
        raise ValueError("tracked YOLO box did not expose four xyxy values")
    return [float(item) for item in first]


def _find_body_frame_image(context: StageContext, frame_idx: int) -> Path:
    names = [f"frame_{frame_idx:06d}{suffix}" for suffix in BODY_FRAME_SUFFIXES]
    roots: list[Path] = []
    env_root = os.environ.get("RACKETSPORT_BODY_FRAMES")
    if env_root:
        roots.append(Path(env_root))
        roots.append(Path(env_root) / context.clip)
    roots.extend(
        [
            context.inputs_dir / "body_frames",
            context.inputs_dir / "frames",
            context.inputs_dir,
            context.run_dir / "body_frames",
            context.run_dir / "frames",
            context.run_dir,
        ]
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate

    recursive_matches: list[Path] = []
    for suffix in BODY_FRAME_SUFFIXES:
        recursive_matches.extend(context.inputs_dir.rglob(f"frame_{frame_idx:06d}{suffix}"))
    if recursive_matches:
        return sorted(recursive_matches, key=_body_frame_priority)[0]

    raise FileNotFoundError(
        f"missing BODY frame image for frame {frame_idx}; expected body_frames/frame_{frame_idx:06d}.jpg "
        f"under {context.inputs_dir} or RACKETSPORT_BODY_FRAMES"
    )


def _body_frame_priority(path: Path) -> tuple[int, str]:
    parts = {part.lower() for part in path.parts}
    body_rank = 0 if any("body" in part for part in parts) else 1
    return (body_rank, str(path))


def _image_size_from_calibration_and_bboxes(calibration: CourtCalibration, bboxes: list[list[float]]) -> tuple[int, int]:
    calibration_width, calibration_height = calibration_image_size(calibration)
    max_x = max([calibration_width, *[bbox[2] + 1.0 for bbox in bboxes]])
    max_y = max([calibration_height, *[bbox[3] + 1.0 for bbox in bboxes]])
    return (max(1, int(round(max_x))), max(1, int(round(max_y))))


def _image_size_from_body_frame_or_calibration(
    image_path: Path,
    *,
    calibration: CourtCalibration,
    bboxes: list[list[float]],
) -> tuple[int, int]:
    del calibration, bboxes
    frame_size = _read_image_size(image_path)
    if frame_size is not None:
        return frame_size
    raise ValueError(
        f"unable to read BODY frame image size for {image_path}; "
        "refusing to derive SAM3D static intrinsics from calibration/bbox fallback"
    )


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _bbox_scale_from_calibration_to_image(
    calibration: CourtCalibration,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    calibration_width, calibration_height = calibration_image_size(
        calibration,
        fallback_target=(float(image_size[0]), float(image_size[1])),
    )
    if calibration_width <= 0.0 or calibration_height <= 0.0:
        return 1.0, 1.0
    return float(image_size[0]) / calibration_width, float(image_size[1]) / calibration_height


def _scale_bbox(bbox: list[float], *, scale_x: float, scale_y: float) -> list[float]:
    return [
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    ]


def _sam3d_keypoints_sidecar_from_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        keypoints = sample.get("pred_foot_keypoints_2d") or []
        if not keypoints:
            continue
        frame_idx = int(sample["frame_idx"])
        player_id = int(sample["player_id"])
        by_player[player_id].append(
            {
                "frame_idx": frame_idx,
                "t": float(sample.get("t", 0.0)),
                "keypoints": [
                    {
                        "name": str(item["name"]),
                        "index": int(item["index"]),
                        "xy_px": [float(item["xy_px"][0]), float(item["xy_px"][1])],
                        "conf": float(item.get("conf", sample.get("confidence", 1.0))),
                    }
                    for item in keypoints
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_keypoints_2d",
        "source": "orchestrator_body_stage",
        "foot_keypoint_indices": dict(SAM3D_FOOT_KEYPOINT_INDICES),
        "players": [
            {"id": player_id, "frames": sorted(frames, key=lambda frame: frame["frame_idx"])}
            for player_id, frames in sorted(by_player.items())
        ],
    }


def _match_body_outputs(raw_outputs: list[Any], requested_bboxes: list[list[float]]) -> list[Any]:
    if len(raw_outputs) == len(requested_bboxes) and any(_raw_output_bbox(output) is None for output in raw_outputs):
        return raw_outputs
    remaining = list(raw_outputs)
    matched: list[Any] = []
    for requested in requested_bboxes:
        best_index = 0
        best_iou = -1.0
        for index, output in enumerate(remaining):
            bbox = _raw_output_bbox(output)
            if bbox is None:
                continue
            iou = _bbox_iou(requested, bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        matched.append(remaining.pop(best_index))
    return matched


def _raw_output_bbox(output: Any) -> list[float] | None:
    mapping = output if isinstance(output, dict) else getattr(output, "__dict__", {})
    if not isinstance(mapping, dict):
        return None
    raw = _first_present_mapping_value(mapping, ("bbox", "box", "xyxy", "pred_box"))
    raw = _to_python_container(raw)
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return None


def _first_present_mapping_value(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _to_python_container(value: Any) -> Any:
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                item = method()
            except Exception:
                return value
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return value


def _body_stance_index_from_placement(
    placement_payload: Any,
    *,
    foot_contact_phases: Any,
    fps: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not isinstance(placement_payload, Mapping):
        return {}
    stance_frames = _stance_frames_from_contact_phases(foot_contact_phases)
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for player in placement_payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        try:
            player_id = int(player.get("id"))
        except (TypeError, ValueError):
            continue
        frames = [frame for frame in player.get("frames", []) or [] if isinstance(frame, Mapping)]
        frames.sort(key=lambda frame: int(frame.get("frame_idx", 0)))
        xy_by_frame: dict[int, list[float]] = {}
        t_by_frame: dict[int, float] = {}
        for frame in frames:
            try:
                frame_idx = int(frame.get("frame_idx"))
                xy = frame.get("smoothed_world_xy")
                if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) < 2:
                    continue
                xy_by_frame[frame_idx] = [float(xy[0]), float(xy[1])]
                t_by_frame[frame_idx] = float(frame.get("t", frame_idx / fps))
            except (TypeError, ValueError):
                continue
        sorted_indices = sorted(xy_by_frame)
        for pos, frame_idx in enumerate(sorted_indices):
            frame = next(item for item in frames if int(item.get("frame_idx", -1)) == frame_idx)
            velocity = _derived_xy_velocity(
                frame_idx,
                pos=pos,
                sorted_indices=sorted_indices,
                xy_by_frame=xy_by_frame,
                t_by_frame=t_by_frame,
                fps=fps,
            )
            index[(player_id, frame_idx)] = {
                "stance": bool(frame.get("stance", False)) or frame_idx in stance_frames.get(player_id, set()),
                "velocity": velocity,
                "covariance_m2": frame.get("covariance_m2"),
                "source": "placement.json",
            }
    return index


def _stance_frames_from_contact_phases(payload: Any) -> dict[int, set[int]]:
    if not isinstance(payload, Mapping):
        return {}
    phases = payload.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        return {}
    out: dict[int, set[int]] = {}
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        try:
            player_id = int(phase.get("player_id"))
        except (TypeError, ValueError):
            continue
        frame_indices = phase.get("frame_indices", [])
        if not isinstance(frame_indices, Sequence) or isinstance(frame_indices, (str, bytes)):
            continue
        for frame_idx in frame_indices:
            try:
                out.setdefault(player_id, set()).add(int(frame_idx))
            except (TypeError, ValueError):
                continue
    return out


def _derived_xy_velocity(
    frame_idx: int,
    *,
    pos: int,
    sorted_indices: Sequence[int],
    xy_by_frame: Mapping[int, Sequence[float]],
    t_by_frame: Mapping[int, float],
    fps: float,
) -> list[float]:
    if len(sorted_indices) <= 1:
        return [0.0, 0.0]
    if pos == 0:
        other_idx = sorted_indices[1]
    elif pos == len(sorted_indices) - 1:
        other_idx = sorted_indices[pos - 1]
    else:
        before_idx = sorted_indices[pos - 1]
        after_idx = sorted_indices[pos + 1]
        before_t = t_by_frame.get(before_idx, before_idx / fps)
        after_t = t_by_frame.get(after_idx, after_idx / fps)
        dt = after_t - before_t
        if dt > 0.0:
            before_xy = xy_by_frame[before_idx]
            after_xy = xy_by_frame[after_idx]
            return [
                (float(after_xy[0]) - float(before_xy[0])) / dt,
                (float(after_xy[1]) - float(before_xy[1])) / dt,
            ]
        other_idx = before_idx
    current_xy = xy_by_frame[frame_idx]
    other_xy = xy_by_frame[other_idx]
    current_t = t_by_frame.get(frame_idx, frame_idx / fps)
    other_t = t_by_frame.get(other_idx, other_idx / fps)
    dt = other_t - current_t
    if dt == 0.0:
        return [0.0, 0.0]
    return [
        (float(other_xy[0]) - float(current_xy[0])) / dt,
        (float(other_xy[1]) - float(current_xy[1])) / dt,
    ]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _write_json_artifact(path: Path, artifact: StrictArtifact | Any) -> None:
    if hasattr(artifact, "model_dump"):
        payload = artifact.model_dump(mode="json")
    else:
        payload = artifact
    _write_json(path, payload)


def _json_artifact_payload(artifact: StrictArtifact | Any) -> Any:
    return artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else artifact


def _write_compact_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "serialization_seconds": max(0.0, time.perf_counter() - started),
    }


def _write_compact_json_artifact(path: Path, artifact: StrictArtifact | Any) -> dict[str, Any]:
    return _write_compact_json(path, _json_artifact_payload(artifact))


def _skipped_compact_json_artifact(path: Path, *, reason: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": 0,
        "serialization_seconds": 0.0,
        "skipped": True,
        "reason": reason,
    }


def _write_body_serialization_timing(run_dir: Path, timings: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_serialization_timing",
        "artifacts": [
            {
                "artifact": Path(item["path"]).name,
                "path": item["path"],
                "bytes": int(item["bytes"]),
                "serialization_seconds": float(item["serialization_seconds"]),
                "skipped": bool(item.get("skipped", False)),
                "reason": str(item["reason"]) if item.get("reason") else None,
            }
            for item in timings
        ],
        "summary": {
            "artifact_count": len(timings),
            "total_bytes": sum(int(item["bytes"]) for item in timings),
            "total_serialization_seconds": sum(float(item["serialization_seconds"]) for item in timings),
            "json_format": "compact_no_indent_no_sort_keys_newline_terminated",
            "written_count": sum(1 for item in timings if not item.get("skipped", False)),
            "skipped_count": sum(1 for item in timings if item.get("skipped", False)),
        },
    }
    _write_json(run_dir / "body_serialization_timing.json", payload)


def _write_body_stage_phase_timing(
    run_dir: Path,
    *,
    stage_wall_seconds: float,
    phase_timings: Mapping[str, Any],
    person_frame_count: int,
    phase_boundaries: Mapping[str, str],
    not_instrumentable: Mapping[str, str],
    timing_sources: Mapping[str, str],
    postchain_bypasses: Mapping[str, Any] | None = None,
) -> None:
    timed_keys = (
        "orchestrator_model_setup_s",
        "model_load_s",
        "compile_warmup_s",
        "inference_s",
        "input_prep_s",
        "runner_request_parse_s",
        "runner_preprocessing_s",
        "runner_postprocessing_s",
        "runner_result_serialization_handoff_s",
        "runner_other_s",
        "subprocess_wrapper_handoff_s",
        "mesh_smpl_payload_assembly_s",
        "array_native_gate_feed_s",
        "keypoints_2d_s",
        "contact_splice_s",
        "gates_s",
        "serialization_s",
        "index_build_s",
        "artifact_io_s",
    )
    attributed = sum(float(phase_timings.get(key, 0.0)) for key in timed_keys if phase_timings.get(key) is not None)
    other_s = max(0.0, float(stage_wall_seconds) - attributed)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_stage_phase_timing",
        "stage_wall_seconds": float(stage_wall_seconds),
        "model_load_s": float(phase_timings.get("model_load_s", 0.0)),
        "orchestrator_model_setup_s": phase_timings.get("orchestrator_model_setup_s"),
        "compile_warmup_s": phase_timings.get("compile_warmup_s"),
        "inference_s": float(phase_timings.get("inference_s", 0.0)),
        "subprocess_outer_call_s": phase_timings.get("subprocess_outer_call_s"),
        "person_frame_count": int(person_frame_count),
        "ms_per_person_steady": phase_timings.get("ms_per_person_steady"),
        "input_prep_s": phase_timings.get("input_prep_s"),
        "runner_request_parse_s": phase_timings.get("runner_request_parse_s"),
        "runner_preprocessing_s": phase_timings.get("runner_preprocessing_s"),
        "runner_postprocessing_s": phase_timings.get("runner_postprocessing_s"),
        "runner_result_serialization_handoff_s": phase_timings.get("runner_result_serialization_handoff_s"),
        "runner_other_s": phase_timings.get("runner_other_s"),
        "subprocess_wrapper_handoff_s": phase_timings.get("subprocess_wrapper_handoff_s"),
        "mesh_smpl_payload_assembly_s": phase_timings.get("mesh_smpl_payload_assembly_s"),
        "smpl_motion_payload_assembly_s": phase_timings.get("smpl_motion_payload_assembly_s"),
        "array_native_gate_feed_s": phase_timings.get("array_native_gate_feed_s"),
        "mesh_export_payload_assembly_s": phase_timings.get("mesh_export_payload_assembly_s"),
        "keypoints_2d_s": float(phase_timings.get("keypoints_2d_s", 0.0)),
        "contact_splice_s": float(phase_timings.get("contact_splice_s", 0.0)),
        "gates_s": float(phase_timings.get("gates_s", 0.0)),
        "serialization_s": float(phase_timings.get("serialization_s", 0.0)),
        "index_build_s": phase_timings.get("index_build_s"),
        "artifact_io_s": phase_timings.get("artifact_io_s"),
        "attributed_s": float(attributed),
        "other_s": float(other_s),
        "per_bucket_timing": list(phase_timings.get("per_bucket_timing", [])),
        "timing_sources": dict(timing_sources),
        "phase_boundaries": dict(phase_boundaries),
        "not_instrumentable": dict(not_instrumentable),
        "notes": [
            "Phase timings are speed instrumentation only; subprocess_outer_call_s is retained for comparison and is not double-counted in attributed_s when runner timing is available.",
            "VERIFIED=0 unchanged; this artifact is speed instrumentation only.",
        ],
    }
    if postchain_bypasses:
        payload["postchain_bypasses"] = dict(postchain_bypasses)
    _write_json(run_dir / "body_stage_phase_timing.json", payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fail-closed racket-sport pipeline spine.")
    parser.add_argument("--clip", required=True, help="Clip identifier for the run summary.")
    parser.add_argument("--inputs", type=Path, required=True, help="Directory containing stage inputs for this clip.")
    parser.add_argument("--out", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--stage", default="e2e", help="Target pipeline stage from pipeline_contracts.py.")
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    parser.add_argument("--device", default=None, help="Optional GPU device hint for future GPU runners.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tracking-mode", choices=["real", "precomputed", "precomputed_tracks"], default="real")
    parser.add_argument("--tracking-video", type=Path, default=None, help="Source video for real tracking.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MODEL_MANIFEST, help="Model manifest with yolo26m checksum.")
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_BOTSORT_REID_CONFIG)
    parser.add_argument(
        "--ball-source",
        type=Path,
        default=None,
        help="Optional no-click BALL source artifact, normally ball_track_fusion_temporal_vball100_localtraj.json.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        choices=(2, 4),
        default=4,
        help="Maximum on-court player identities to keep: 2 for singles, 4 for doubles.",
    )
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=0.0,
        help="Runoff margin around the regulation court footprint for accepting player footpoints.",
    )
    parser.add_argument(
        "--id-strategy",
        choices=("auto", "raw_track", "role_lock"),
        default="auto",
        help=(
            "auto role-locks prototype player-label detections without tracker IDs and otherwise keeps raw tracker IDs; "
            "raw_track keeps tracker IDs; role_lock assigns stable logical near/far left/right player IDs per frame."
        ),
    )
    args = parser.parse_args(argv)

    summary = run_pipeline(
        clip=args.clip,
        inputs_dir=args.inputs,
        run_dir=args.out,
        stage=args.stage,
        sport=args.sport,
        device=args.device,
        max_frames=args.max_frames,
        tracking_mode=args.tracking_mode,
        tracking_video=args.tracking_video,
        manifest_path=args.manifest,
        tracker_config_path=args.tracker_config,
        max_players=args.max_players,
        court_margin_m=args.court_margin_m,
        id_strategy=args.id_strategy,
        ball_source_path=args.ball_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == PIPELINE_STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
