#!/usr/bin/env python3
"""One-command pickleball pipeline: a video file -> a scrubber-ready bundle.

    scripts/racketsport/process_video.py --video X.mp4 --court-corners corners.json

This is the "send in a video, the entire pipeline runs seamlessly together"
entrypoint (MASTER_PLAN.md's product goal). It chains, in order:

  1. ingest       -- validate the video, build capture_sidecar.json from
                      --court-corners (declared image_size required) or an
                      already-built --capture-sidecar (ARKit, when present).
  2. calibration   -- court_calibration.json + court_zones/net_plane/
                      court_line_evidence via the existing fail-closed spine
                      (threed.racketsport.orchestrator, unmodified logic), OR --
                      when --court-calibration is given (or a
                      court_calibration_metric15pt.json is auto-discovered next to
                      the clip's labels/) -- consumed directly via
                      threed.racketsport.orchestrator.ExternalCalibrationRunner,
                      which validates schema + intrinsics.source and skips PnP
                      re-derivation (see that class's docstring for the
                      intrinsics.dist migration note).
  3. tracking      -- BoT-SORT-ReID "loose pool" (YOLO26m + BoT-SORT via the
                      existing spine), optionally refined by raw-pool global
                      association (threed.racketsport.raw_pool_person_authority,
                      the champion recipe cited in MASTER_PLAN.md), or reused
                      from an already-computed --tracks artifact.
  4. placement     -- placement.json plus an in-place tracks.json world_xy rewrite
                      from bbox, native-2D foot keypoints, covariance fusion, and
                      offline smoothing. The original tracks artifact is backed up
                      as tracks_prewrite_backup.json.
  5. frames        -- body_frames/frame_NNNNNN.jpg JPEGs the body stage reads
                      (threed.racketsport.orchestrator's
                      BODY-runtime frame lookup); scheduled from tracks.json
                      (every tracked-player frame for SAM-3D body joints) and, when a
                      frame_compute_plan.json already exists on this clip_dir,
                      unioned with its deep_mesh_windows too (see
                      threed.racketsport.process_video_body_frames for the
                      cap/fallback rules). Reused by remote BODY dispatch's
                      rsync-up and by local body.
  6. ball          -- ball_track.json via WASB-tennis zero-shot (or reused
                      from --ball-track), then 2D bounce detection and
                      geometry-corrected manual-court in/out (both imported
                      unmodified from threed.racketsport.ball_bounce_2d /
                      ball_manual_court_inout -- never rewritten here).
  7. events        -- ball_inflections / audio onsets and any existing SAM-3D
                      wrist_velocity_peaks fused into contact_windows.json, plus
                      frame_compute_plan.json. If SAM-3D joints are not available
                      before BODY, wrist cues are explicitly blocked instead of
                      replaced by a legacy pose source.
  8. body          -- Fast SAM-3D-Body body-mode joints for all safe tracked
                      person-frames; mesh vertices serialized only for the
                      tier-1 ball_aware_100 windows. BODY is
                      dispatched to the remote A100 by default (see
                      remote_body_dispatch.py) since most hosts running this
                      script have no local GPU; --no-gpu (or a failed/busy
                      dispatch) degrades to skeleton-only, never a crash.
  9. placement_refine -- optional post-BODY placement rewrite from compact SAM-3D
                      foot keypoint sidecar and/or 3D contact phase anchors.
 10. grounding     -- render-honest rigid root-level BODY grounding refinement
                      when foot_contact_phases.json + calibration + tracks exist;
                      not accuracy/gate evidence, skipped untouched on zero
                      contact phases.
 11. paddle_pose   -- render-only fused wrist+palm+grip paddle estimate
                      (`racket_pose_estimate.json`) when SAM-3D wrist/palm
                      evidence exists; fail-closed with a loud summary block
                      otherwise. This is estimated preview evidence only, never
                      an RKT promotion.
 12. world         -- virtual_world.json + trust_bands.json (every entity
                      badged from real upstream gate/artifact state via
                      threed.racketsport.trust_band, never invented).
 13. confidence    -- confidence_gated_world.json via the Wave-B additive
                      confidence gate (default on; --no-confidence-gate keeps
                      raw virtual_world.json as the viewer world).
 14. manifest      -- replay_viewer_manifest.json, the same bundle shape
                      web/replay already loads.
 15. verify        -- optional (--verify-viewer) headless load check of the
                      web viewer against the freshly built manifest.

Resilience: every stage checks for an already-valid artifact first and skips
real work when one exists (--force to redo), a failed/unavailable stage
degrades to a trust-banded gap instead of raising past its own boundary
(hard failures are reserved for calibration, the one stage nothing else can
substitute for), and PIPELINE_SUMMARY.json always gets written with
per-stage wall-clock, status, and trust badges -- even on a partial run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Literal, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport import orchestrator  # noqa: E402
from threed.racketsport.ball_physics_fill import PhysicsFillConfig, fill_ball_track_physics  # noqa: E402
from threed.racketsport.ball_physics3d import reconstruct_bounce_arcs_from_image_track  # noqa: E402
from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain  # noqa: E402
from threed.racketsport.best_stack import load_best_stack_manifest  # noqa: E402
from threed.racketsport.ball_inflections import build_ball_inflections_from_ball_track  # noqa: E402
from threed.racketsport.court_calibration import calibration_image_size  # noqa: E402
from threed.racketsport.court_corner_review import SIDECAR_CORNER_ORDER  # noqa: E402
from threed.racketsport.confidence_gate import (  # noqa: E402
    ConfidenceGateConfig,
    apply_confidence_gate_to_world,
    summarize_bands,
)
from threed.racketsport.body_grounding_refine import GroundingRefineConfig, refine_body_grounding  # noqa: E402
from threed.racketsport.event_fusion import fuse_contact_windows_from_cue_payloads  # noqa: E402
from threed.racketsport.frame_rating import (  # noqa: E402
    DEFAULT_BALL_PROXIMITY_M,
    DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR,
    MESH_COVERAGE_MODES,
    build_frame_compute_plan,
    write_frame_compute_plan,
)
from threed.racketsport.io_decode import time_for_frame, write_frame_time_table  # noqa: E402
from threed.racketsport.camera_motion import (  # noqa: E402
    CAMERA_MOTION_AUTO_THRESHOLD,
    CAMERA_MOTION_PROBE_MAX_CORNERS,
    CAMERA_MOTION_PROBE_PROCESSING_SCALE,
    CameraMotionParams,
    estimate_camera_motion,
    estimate_camera_motion_probe,
    validate_camera_motion_payload,
    write_camera_motion_json,
)
from threed.racketsport.person_reid_diagnostics import resolve_reid_device  # noqa: E402
from threed.racketsport.placement import PlacementConfig, rewrite_tracks_with_placement  # noqa: E402
from threed.racketsport.paddle_pose_fused import (  # noqa: E402
    ARTIFACT_TYPE as PADDLE_POSE_ARTIFACT_TYPE,
    SOURCE as PADDLE_POSE_SOURCE,
    build_paddle_pose_fused_from_file,
    write_paddle_pose_fused,
)
from threed.racketsport.process_video_body_frames import materialize_process_video_frames  # noqa: E402
from threed.racketsport.rally_gating import build_rally_spans_artifact, in_rally_span  # noqa: E402
from threed.racketsport.raw_pool_person_authority import (  # noqa: E402
    RawPoolAuthorityConfig,
    run_raw_pool_authority_candidate,
)
from threed.racketsport.replay_export import (  # noqa: E402
    build_replay_review_export_from_virtual_world,
    validate_replay_export_manifest,
    write_replay_scene,
)
from threed.racketsport.replay_viewer_manifest import (  # noqa: E402
    build_replay_viewer_manifest,
    write_replay_viewer_manifest,
)
from threed.racketsport.schemas import validate_artifact_file  # noqa: E402
from threed.racketsport.trust_band import (  # noqa: E402
    build_trust_band,
    derive_ball_trust_band,
    derive_court_trust_band,
    derive_track_trust_band,
)
from threed.racketsport.virtual_world import build_virtual_world_state, write_virtual_world  # noqa: E402
from threed.racketsport.wrist_velocity_peaks import (  # noqa: E402
    build_blocked_wrist_velocity_peaks,
    build_wrist_velocity_peaks_from_file,
)

from scripts.racketsport.remote_body_dispatch import (  # noqa: E402
    RemoteBodyDispatchError,
    RemoteConfig,
    dispatch_body_stage,
)


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DEFAULT_RUN_ROOT = ROOT / "runs"
BEST_STACK_MANIFEST = load_best_stack_manifest()
DEFAULT_REID_MODEL = BEST_STACK_MANIFEST.path_value("tracking.reid_model", must_exist=False)
DEFAULT_WASB_CHECKPOINT = BEST_STACK_MANIFEST.path_value("ball.wasb_checkpoint")
DEFAULT_WASB_REPO = BEST_STACK_MANIFEST.path_value("ball.wasb_repo")
DEFAULT_CONFIDENCE_CALIBRATION_CURVES = BEST_STACK_MANIFEST.path_value("confidence.calibration_curves")
DEFAULT_MESH_COVERAGE_MODE = BEST_STACK_MANIFEST.string_value("mesh.coverage_mode")
DEFAULT_TARGET_MESH_FRAME_BUDGET = BEST_STACK_MANIFEST.value("mesh.target_frame_budget")
DEFAULT_MESH_BYTE_BUDGET_MIB = BEST_STACK_MANIFEST.number_value("mesh.byte_budget_mib")
DEFAULT_PADDLE_FUSED_ESTIMATOR = BEST_STACK_MANIFEST.value("paddle.fused_estimator")
PADDLE_POSE_ARTIFACT_NAME = "racket_pose_estimate.json"
GROUNDING_REFINE_POLICY_NOTE = "render-honest estimated grounding, not gate evidence"
REPLAY_POINT_MAX_SPAN_SECONDS = 1.5
AUTO_COURT_PREVIEW_TRACKING_MARGIN_M = 1000.0
AUTO_COURT_PREVIEW_DEMO_MESH_MAX_FRAMES = 12
COURT_CORRECTION_TASK_NAME = "court_correction_task.json"
COURT_DETECTOR_V2_PROPOSAL_NAME = "court_detector_v2_proposals.json"
COURT_PROPOSALS_NAME = "court_proposals.json"
COURT_CORRECTION_BLOCKED_DOWNSTREAM = [
    "tracking_court_filter",
    "body_world",
    "ball_world",
    "virtual_world_metric",
]
UNVERIFIED_COURT_REASON_TOKENS = (
    "unverified",
    "estimated_intrinsics",
    "estimated_from_declared_court_corners",
    "process_video_manual_court_corners",
    "process_video_auto_court_corners_preview",
    "manual_taps_seeded_from_unverified_detector",
)

StageStatus = Literal["ran", "reused", "skipped", "degraded", "blocked", "failed"]


@dataclass
class StageOutcome:
    stage: str
    status: StageStatus
    wall_seconds: float
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    trust_badge: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "wall_seconds": round(self.wall_seconds, 3),
            "notes": self.notes,
            "artifacts": self.artifacts,
            "trust_badge": self.trust_badge,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class VideoTiming:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass
class RawPoolGlobalAssociationProfile:
    name: str
    note: str
    court_margin_m: float = 2.0
    min_conf: float | None = None
    appearance_weight: float | None = None
    max_gap_fill_frames: int | None = None
    max_merge_cost: float | None = None
    cardinality_backfill: bool | None = None


# The default raw-pool association profile is intentionally explicit and now
# manifest-owned. Per-clip evaluation overrides below remain eval-only.
DEFAULT_GLOBAL_ASSOCIATION_PROFILE = BEST_STACK_MANIFEST.string_value("tracking.global_association_profile")
RAW_POOL_GLOBAL_ASSOCIATION_PROFILES: dict[str, RawPoolGlobalAssociationProfile] = {
    "wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill": RawPoolGlobalAssociationProfile(
        name="wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill",
        note=(
            "Wolverine internal-val TRK-12 cfg151 profile: min_conf=0.3, "
            "court_margin_m=1.0, appearance_weight=0.5, cardinality_backfill enabled, "
            "48-frame gap fill, and max_merge_cost=2.0."
        ),
        court_margin_m=1.0,
        min_conf=0.3,
        appearance_weight=0.5,
        max_gap_fill_frames=48,
        max_merge_cost=2.0,
        cardinality_backfill=True,
    ),
    "outdoor_preregistered_unshopped_base": RawPoolGlobalAssociationProfile(
        name="outdoor_preregistered_unshopped_base",
        note=(
            "Outdoor strict-holdout TRK-11 profile: unshopped raw-pool BASE config "
            "with court_margin_m=3.0, cardinality_backfill enabled, 24-frame gap fill, "
            "and max_merge_cost=3.0."
        ),
        court_margin_m=3.0,
        min_conf=0.0,
        appearance_weight=1.0,
        max_gap_fill_frames=24,
        max_merge_cost=3.0,
        cardinality_backfill=True,
    ),
    DEFAULT_GLOBAL_ASSOCIATION_PROFILE: RawPoolGlobalAssociationProfile(
        name=DEFAULT_GLOBAL_ASSOCIATION_PROFILE,
        note="Wolverine internal-val tuned TRK-10 profile: iter2 with court_margin_m=2.0.",
        court_margin_m=2.0,
    ),
    "burlington_internal_val_trk10_iter5_minconf05_appw2_margin2": RawPoolGlobalAssociationProfile(
        name="burlington_internal_val_trk10_iter5_minconf05_appw2_margin2",
        note="Burlington internal-val tuned TRK-10 profile: iter5-style min_conf=0.5, appearance_weight=2.0, court_margin_m=2.0.",
        court_margin_m=2.0,
        min_conf=0.5,
        appearance_weight=2.0,
    ),
}


@dataclass
class PipelineOptions:
    video: Path
    clip: str
    run_dir: Path
    sport: str = "pickleball"
    max_players: int = 4
    max_frames: int | None = None
    force: bool = False
    device: str | None = None

    court_corners: Path | None = None
    capture_sidecar: Path | None = None
    court_keypoints: Path | None = None
    court_calibration: Path | None = None
    allow_auto_court_corners_preview: bool = False
    court_proposals_preview: bool = False

    tracks_reuse: Path | None = None
    global_association: bool = True
    global_association_profile: str | None = None
    reid_model: Path = DEFAULT_REID_MODEL

    ball_track_reuse: Path | None = None
    ball_candidates_reuse: tuple[Path, ...] = ()
    emit_ball_candidates: bool = True
    ball_track_auto_discovery: bool = False
    skip_ball: bool = False
    no_ball_arc: bool = False
    wasb_checkpoint: Path = DEFAULT_WASB_CHECKPOINT
    wasb_repo: Path | None = None

    skip_audio: bool = False
    rally_gating: bool = False
    rally_gating_pad_seconds: float = 0.5

    placement_keypoints_2d: Path | None = None
    camera_motion_path: Path | None = None
    enable_camera_motion: bool = False
    skip_camera_motion: bool = False
    camera_motion_auto_threshold: float = CAMERA_MOTION_AUTO_THRESHOLD
    camera_motion_estimator: str = "hardened"
    camera_motion_flow_backend: str = "lk"
    camera_motion_person_masks: bool = True
    placement_undistort: bool = True

    mesh_coverage_mode: str = DEFAULT_MESH_COVERAGE_MODE
    target_mesh_frame_budget: int | None = DEFAULT_TARGET_MESH_FRAME_BUDGET
    mesh_byte_budget_mib: float | None = DEFAULT_MESH_BYTE_BUDGET_MIB
    ball_proximity_m: float = DEFAULT_BALL_PROXIMITY_M
    high_confidence_swing_floor: float = DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR
    events_selected: Path | None = None
    ball_track_arc_solved: Path | None = None

    no_gpu: bool = False
    body_remote: bool = True
    body_schedule: str = "serial"
    remote_config: RemoteConfig = field(default_factory=RemoteConfig)

    grounding_refine: bool = True
    paddle_pose: bool = bool(DEFAULT_PADDLE_FUSED_ESTIMATOR.get("enabled", True))

    confidence_gate: bool = True
    confidence_calibration_curves: Path | None = None
    scene_points: bool = True

    manifest_path: Path = orchestrator.DEFAULT_MODEL_MANIFEST
    tracker_config_path: Path = orchestrator.DEFAULT_BOTSORT_REID_CONFIG

    verify_viewer: bool = False
    vite_allow_root: Path = ROOT

    @property
    def clip_dir(self) -> Path:
        return self.run_dir / self.clip


class ProcessVideoPipeline:
    """Runs every stage of `process_video.py` against one clip directory."""

    def __init__(self, options: PipelineOptions) -> None:
        self.options = options
        self.clip_dir = options.clip_dir
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.trust_bands: dict[str, dict[str, Any] | None] = {}
        self.stage_outcomes: list[StageOutcome] = []
        self._parallel_body_block: dict[str, Any] | None = None
        self._camera_motion_auto: dict[str, Any] = self._camera_motion_auto_decision(
            score=None,
            threshold=self.options.camera_motion_auto_threshold,
            enabled=False,
            forced="auto",
            probe_wall_seconds=0.0,
            sampled_frame_count=0,
        )
        self._camera_motion_auto_decision_made = False
        if options.force:
            self._clear_force_regenerated_artifacts()

    def _clear_force_regenerated_artifacts(self) -> None:
        """Remove generated helper outputs that must not leak into --force runs."""

        for name in (
            "skeleton3d.json",
            "ball_track.json",
            "ball_candidates.json",
            "ball_bounce_candidates.json",
            "ball_track_arc_solved.json",
            "ball_flight_sanity.json",
            "ball_chain_manifest.json",
            "ball_track_physics_filled.json",
            "ball_inflections.json",
            "wrist_velocity_peaks.json",
            "audio_onsets_v2.json",
            "audio_onsets.json",
            "contact_windows.json",
            "frame_compute_plan.json",
            "placement.json",
            "sam3d_keypoints_2d.json",
            "smpl_motion.json",
            "body_compute_execution.json",
            "body_mesh.json",
            "body_mesh_readiness.json",
            "body_full_clip_gate.json",
            "body_grounding_quality.json",
            "body_grounding_refinement.json",
            "skeleton3d_pre_grounding_refine.json",
            "smpl_motion_pre_grounding_refine.json",
            PADDLE_POSE_ARTIFACT_NAME,
            "pipeline_run.json",
            "virtual_world.json",
            "confidence_gated_world.json",
            "confidence_gate_summary.json",
            "trust_bands.json",
            "replay_scene.json",
            "replay_viewer_manifest.json",
        ):
            path = self.clip_dir / name
            if self.options.ball_track_reuse is not None and path.resolve() == self.options.ball_track_reuse.resolve():
                continue
            if any(path.resolve() == candidate_path.resolve() for candidate_path in self.options.ball_candidates_reuse):
                continue
            if path.is_file():
                path.unlink()

    @staticmethod
    def _camera_motion_auto_decision(
        *,
        score: float | None,
        threshold: float,
        enabled: bool,
        forced: str,
        probe_wall_seconds: float,
        sampled_frame_count: int,
    ) -> dict[str, Any]:
        return {
            "score": None if score is None else round(float(score), 6),
            "threshold": round(float(threshold), 6),
            "enabled": bool(enabled),
            "forced": str(forced),
            "probe_wall_seconds": round(float(probe_wall_seconds), 6),
            "sampled_frame_count": int(sampled_frame_count),
        }

    def _set_camera_motion_auto(self, decision: dict[str, Any]) -> dict[str, Any]:
        self._camera_motion_auto = decision
        self._camera_motion_auto_decision_made = True
        return decision

    # ------------------------------------------------------------------
    # top-level driver
    # ------------------------------------------------------------------

    # Optional BODY-adjacent artifacts a --body-schedule=overlap run may not
    # have produced yet at the moment BODY is dispatched, because they are
    # computed by ball/ball_arc/events/ball_fill running concurrently on the
    # main thread. Today's serial order already fires BODY dispatch without
    # waiting for ball-aware triggers, so this is not a practical regression
    # -- but overlap must record it, never hide it (HONESTY CONTRACTS b).
    _OPTIONAL_BODY_OVERLAP_INPUTS: tuple[str, ...] = (
        "ball_track.json",
        "ball_track_arc_solved.json",
        "ball_inflections.json",
        "wrist_velocity_peaks.json",
        "contact_windows.json",
        "frame_compute_plan.json",
    )

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        if self.options.body_schedule == "overlap":
            self._run_overlap()
        else:
            self._run_serial()
        summary = self._write_summary(wall_seconds=time.monotonic() - started)
        return summary

    def _build_prefix_stage_fns(self) -> list[tuple[str, Callable[[], StageOutcome]]]:
        stage_fns: list[tuple[str, Callable[[], StageOutcome]]] = [
            ("ingest", self._stage_ingest),
            ("calibration", self._stage_calibration),
            ("tracking", self._stage_tracking),
            ("camera_motion", self._stage_camera_motion),
            ("placement", self._stage_placement),
        ]
        if self.options.rally_gating:
            stage_fns.append(("rally_gating", self._stage_rally_gating))
        return stage_fns

    def _middle_stage_fns(self) -> list[tuple[str, Callable[[], StageOutcome]]]:
        return [
            ("ball", self._stage_ball),
            ("ball_arc", self._stage_ball_arc),
            ("events", self._stage_events),
            ("ball_fill", self._stage_ball_fill),
        ]

    def _build_suffix_stage_fns(self) -> list[tuple[str, Callable[[], StageOutcome]]]:
        stage_fns: list[tuple[str, Callable[[], StageOutcome]]] = [
            ("placement_refine", self._stage_placement_refine),
            ("grounding_refine", self._stage_grounding_refine),
            ("paddle_pose", self._stage_paddle_pose),
            ("world", self._stage_world),
            ("confidence_gate", self._stage_confidence_gate),
            ("manifest", self._stage_manifest),
        ]
        if self.options.verify_viewer:
            stage_fns.append(("verify", self._stage_verify))
        return stage_fns

    def _run_stage_list(self, stage_fns: list[tuple[str, Callable[[], StageOutcome]]]) -> bool:
        """Run stage functions in order, appending outcomes. Returns True iff
        a hard failure was hit (nothing downstream can substitute for it, so
        the caller must stop instead of cascading confusing "degraded" noise
        through every later stage that also needs this input)."""

        for name, fn in stage_fns:
            outcome = self._run_stage_safely(name, fn)
            self.stage_outcomes.append(outcome)
            if outcome.status == "failed":
                return True
        return False

    def _run_serial(self) -> None:
        """--body-schedule=serial (the default): derive ball/events first so
        cold-run frame scheduling can use contact-dense ball_aware windows."""

        stage_fns = (
            self._build_prefix_stage_fns()
            + self._middle_stage_fns()
            + [("frames", self._stage_frames)]
            + [("body", self._stage_body)]
            + self._build_suffix_stage_fns()
        )
        self._run_stage_list(stage_fns)

    def _run_overlap(self) -> None:
        """--body-schedule=overlap: once frames/tracks/calibration BODY
        inputs are ready, dispatch BODY on a background thread running the
        exact same stage code path, and run ball/ball_arc/events/ball_fill on
        the main thread while BODY is in flight, joining before any
        BODY-dependent stage (placement_refine/grounding_refine/world)."""

        if self._run_stage_list(self._build_prefix_stage_fns() + [("frames", self._stage_frames)]):
            return

        reuse_outcome = self._body_stage_reuse_skip()
        if reuse_outcome is not None:
            # Reuse semantics unchanged: valid BODY artifacts already exist,
            # so this run takes the plain serial path -- no thread spun up.
            remaining = self._middle_stage_fns() + [("body", lambda: reuse_outcome)] + self._build_suffix_stage_fns()
            self._run_stage_list(remaining)
            body_final = next((o for o in self.stage_outcomes if o.stage == "body"), None)
            self._parallel_body_block = {
                "enabled": False,
                "body_started_after": "frames",
                "overlapped_stages": [],
                "body_wall_s": round(body_final.wall_seconds, 3) if body_final else 0.0,
                "join_wait_s": 0.0,
                "overlap_saved_s_estimate": 0.0,
                "body_inputs_missing_due_to_overlap": [],
                "input_mutation_guard": {"checked_inputs": [], "mutated_inputs": [], "tripped": False},
            }
            return

        if self._run_overlap_body(self._middle_stage_fns()):
            return
        self._run_stage_list(self._build_suffix_stage_fns())

    def _body_dispatch_input_paths(self) -> dict[str, Path]:
        """The exact BODY dispatch inputs the overlap input-mutation guard
        watches (HONESTY CONTRACTS c): tracks.json, body_frames/,
        court_calibration.json, and whichever camera_motion.json this run
        would actually dispatch with (explicit --camera-motion, else the
        canonical clip-dir file, present or not)."""

        camera_motion_path = self.options.camera_motion_path
        if camera_motion_path is None:
            camera_motion_path = self.clip_dir / "camera_motion.json"
        return {
            "tracks.json": self.clip_dir / "tracks.json",
            "body_frames/": self.clip_dir / "body_frames",
            "court_calibration.json": self.clip_dir / "court_calibration.json",
            "camera_motion.json": camera_motion_path,
        }

    def _snapshot_body_dispatch_input_hashes(self) -> dict[str, str | None]:
        return {name: _content_hash(path) for name, path in self._body_dispatch_input_paths().items()}

    def _run_overlap_body(self, middle_stage_fns: list[tuple[str, Callable[[], StageOutcome]]]) -> bool:
        """Dispatch BODY on a background thread while ball/ball_arc/events/
        ball_fill run on the main thread; hard-join before returning. Returns
        True iff the pipeline must stop (BODY or an overlapped local stage
        hard failed) -- the caller must not run placement_refine/
        grounding_refine/world/confidence_gate/manifest in that case."""

        before_hashes = self._snapshot_body_dispatch_input_hashes()
        missing_optional_inputs = [
            name for name in self._OPTIONAL_BODY_OVERLAP_INPUTS if not (self.clip_dir / name).is_file()
        ]

        thread_result: dict[str, Any] = {}

        def _run_body_thread() -> None:
            body_thread_started = time.monotonic()
            try:
                outcome = self._run_stage_safely("body", self._stage_body)
            except Exception as exc:  # noqa: BLE001 - the BODY thread must never crash the process
                trace = traceback.format_exc(limit=6)
                outcome = StageOutcome(
                    stage="body",
                    status="failed",
                    wall_seconds=time.monotonic() - body_thread_started,
                    notes=[f"BODY overlap thread raised {type(exc).__name__}: {exc}", trace],
                )
            thread_result["outcome"] = outcome

        body_thread = threading.Thread(target=_run_body_thread, name="process-video-body-overlap")
        body_thread.start()

        local_failed = self._run_stage_list(middle_stage_fns)
        overlapped_stage_names = [name for name, _ in middle_stage_fns]
        overlapped_names_set = set(overlapped_stage_names)
        overlapped_wall_sum = sum(
            outcome.wall_seconds for outcome in self.stage_outcomes if outcome.stage in overlapped_names_set
        )

        join_started = time.monotonic()
        body_thread.join()
        join_wait_s = time.monotonic() - join_started

        body_outcome = thread_result.get("outcome")
        if body_outcome is None:
            body_outcome = StageOutcome(
                stage="body",
                status="failed",
                wall_seconds=0.0,
                notes=["BODY overlap thread ended without producing a stage outcome"],
            )
        if missing_optional_inputs:
            body_outcome.notes.append(
                "overlap readiness note: BODY dispatch started before these optional same-run inputs existed "
                "(they are computed concurrently by ball/ball_arc/events/ball_fill in overlap mode): "
                f"{', '.join(missing_optional_inputs)}"
            )

        after_hashes = self._snapshot_body_dispatch_input_hashes()
        mutated_inputs = sorted(name for name in before_hashes if before_hashes[name] != after_hashes.get(name))
        guard_tripped = bool(mutated_inputs)
        if guard_tripped:
            body_outcome = StageOutcome(
                stage="body",
                status="failed",
                wall_seconds=body_outcome.wall_seconds,
                notes=[
                    *body_outcome.notes,
                    "OVERLAP INPUT-MUTATION GUARD TRIPPED: an overlapped local stage changed BODY dispatch "
                    f"input(s) {', '.join(mutated_inputs)} while the BODY thread was in flight; fail-closed "
                    "per the parallel_body honesty contract",
                ],
                artifacts=body_outcome.artifacts,
                metrics=body_outcome.metrics,
            )
            self.trust_bands.pop("body", None)

        self.stage_outcomes.append(body_outcome)

        self._parallel_body_block = {
            "enabled": True,
            "body_started_after": "frames",
            "overlapped_stages": overlapped_stage_names,
            "body_wall_s": round(body_outcome.wall_seconds, 3),
            "join_wait_s": round(join_wait_s, 3),
            "overlap_saved_s_estimate": round(min(body_outcome.wall_seconds, overlapped_wall_sum), 3),
            "body_inputs_missing_due_to_overlap": missing_optional_inputs,
            "input_mutation_guard": {
                "checked_inputs": sorted(before_hashes),
                "mutated_inputs": mutated_inputs,
                "tripped": guard_tripped,
            },
        }

        return local_failed or body_outcome.status == "failed"

    def _run_stage_safely(self, name: str, fn) -> StageOutcome:
        started = time.monotonic()
        try:
            outcome = fn()
        except _HardStageFailure as exc:
            return StageOutcome(stage=name, status="failed", wall_seconds=time.monotonic() - started, notes=[str(exc)])
        except Exception as exc:  # noqa: BLE001 - a stage bug must not crash the whole run
            trace = traceback.format_exc(limit=6)
            return StageOutcome(
                stage=name,
                status="degraded",
                wall_seconds=time.monotonic() - started,
                notes=[f"unexpected {type(exc).__name__}: {exc}", trace],
            )
        outcome.wall_seconds = time.monotonic() - started
        return outcome

    # ------------------------------------------------------------------
    # stage 1: ingest
    # ------------------------------------------------------------------

    def _stage_ingest(self) -> StageOutcome:
        video = self.options.video
        if not video.is_file():
            raise _HardStageFailure(f"video not found: {video}")
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            raise _HardStageFailure(f"unsupported video suffix {video.suffix!r}; expected one of {sorted(VIDEO_SUFFIXES)}")

        target = self.clip_dir / f"source{video.suffix.lower()}"
        if not target.exists():
            try:
                target.symlink_to(video.resolve())
            except OSError:
                import shutil

                shutil.copy2(video, target)

        width, height, fps = _video_probe(video)
        frame_times_path = self.clip_dir / "frame_times.json"
        frame_times = write_frame_time_table(target, frame_times_path)
        notes = [
            f"ingested {video} as {target.name} ({width}x{height} @ {fps:.3f}fps)",
            "wrote frame_times.json from ffprobe PTS"
            if frame_times.get("provenance") == "ffprobe_pts"
            else "wrote frame_times.json with constant_fps_assumed fallback",
        ]
        return StageOutcome(stage="ingest", status="ran", wall_seconds=0.0, notes=notes, artifacts=[target.name, "frame_times.json"])

    # ------------------------------------------------------------------
    # stage 2: calibration
    # ------------------------------------------------------------------

    def _stage_calibration(self) -> StageOutcome:
        target = self.clip_dir / "court_calibration.json"
        if target.is_file() and not self.options.force and _valid_artifact("court_calibration", target):
            payload = _read_json(target)
            self._set_court_trust_band(payload, target)
            return StageOutcome(
                stage="calibration",
                status="skipped",
                wall_seconds=0.0,
                notes=["reusing existing valid court_calibration.json"],
                artifacts=["court_calibration.json"],
                trust_badge=self.trust_bands["court"]["badge"] if self.trust_bands.get("court") else None,
            )

        opts = self.options
        external_calibration_path = self._resolved_court_calibration_path()
        if external_calibration_path is not None:
            return self._run_external_calibration(target, external_calibration_path)

        if opts.capture_sidecar is not None:
            sidecar_target = self.clip_dir / "capture_sidecar.json"
            _copy_json(opts.capture_sidecar, sidecar_target)
            source_note = f"copied ARKit/explicit capture_sidecar.json from {opts.capture_sidecar}"
            if opts.court_keypoints is not None:
                _copy_json(opts.court_keypoints, self.clip_dir / "court_keypoints.json")
            elif opts.allow_auto_court_corners_preview and not _capture_sidecar_has_manual_taps(sidecar_target):
                preview_corners = _auto_court_corners_preview_from_video(
                    opts.video,
                    self.clip_dir / "auto_court_corners_preview.json",
                )
                _write_json(sidecar_target, _capture_sidecar_with_preview_corners(sidecar_target, preview_corners))
                source_note = (
                    f"{source_note}; seeded missing manual_court_taps from auto-court preview corners "
                    f"{preview_corners} (unverified, demo fallback)"
                )
        elif opts.court_corners is not None:
            _, _, fps = _video_probe(opts.video)
            sidecar_payload = _capture_sidecar_from_court_corners(opts.court_corners, fps=fps)
            _write_json(self.clip_dir / "capture_sidecar.json", sidecar_payload)
            source_note = f"built capture_sidecar.json from --court-corners {opts.court_corners} (manual 4-corner taps)"
        elif opts.allow_auto_court_corners_preview:
            _, _, fps = _video_probe(opts.video)
            preview_corners = _auto_court_corners_preview_from_video(
                opts.video,
                self.clip_dir / "auto_court_corners_preview.json",
            )
            sidecar_payload = _capture_sidecar_from_court_corners(preview_corners, fps=fps)
            sidecar_payload["capture_quality"] = _preview_capture_quality(
                sidecar_payload.get("capture_quality"),
                extra_reasons=["process_video_auto_court_corners_preview"],
            )
            _write_json(self.clip_dir / "capture_sidecar.json", sidecar_payload)
            source_note = (
                f"built capture_sidecar.json from auto-court preview corners {preview_corners} "
                "(unverified, demo fallback)"
            )
        elif opts.court_proposals_preview:
            proposal_path = self.clip_dir / COURT_PROPOSALS_NAME
            try:
                _court_proposals_preview_from_video(
                    opts.video,
                    clip=opts.clip,
                    out_path=proposal_path,
                    max_frames=opts.max_frames or 5,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as fail-closed calibration status
                raise _HardStageFailure(f"court proposals preview failed: {type(exc).__name__}: {exc}") from exc
            proposal = _read_json(proposal_path)
            correction = _build_court_proposals_correction_task(proposal_path=proposal_path, proposal=proposal)
            _write_json(self.clip_dir / COURT_CORRECTION_TASK_NAME, correction)
            raise _HardStageFailure(
                "court proposals preview wrote review-only court_proposals.json but is not trusted calibration"
            )
        else:
            raise _HardStageFailure(
                "calibration requires --court-corners (declared image_size) or --capture-sidecar "
                "(ARKit); this is the one hard-fail stage -- nothing downstream can substitute for court geometry"
            )

        result = orchestrator.run_pipeline(
            clip=self.options.clip,
            inputs_dir=self.clip_dir,
            run_dir=self.clip_dir,
            stage="calibration",
            sport=self.options.sport,  # type: ignore[arg-type]
        )
        if not _spine_stage_succeeded(result, stage="calibration"):
            raise _HardStageFailure(f"calibration spine failed: {_spine_failure_detail(result)}")

        payload = _read_json(target)
        self._set_court_trust_band(payload, target)
        return StageOutcome(
            stage="calibration",
            status="ran",
            wall_seconds=0.0,
            notes=[source_note, "ran calibration through threed.racketsport.orchestrator (real 4-corner PnP solve)"],
            artifacts=["court_calibration.json", "court_zones.json", "net_plane.json", "court_line_evidence.json"],
            trust_badge=self.trust_bands["court"]["badge"] if self.trust_bands.get("court") else None,
            metrics={
                "reprojection_median_px": payload.get("reprojection_error_px", {}).get("median"),
                "reprojection_p95_px": payload.get("reprojection_error_px", {}).get("p95"),
            },
        )

    def _set_court_trust_band(self, calibration_payload: Mapping[str, Any], path: Path) -> None:
        self.trust_bands["court"] = derive_court_trust_band(calibration_payload, evidence_path=str(path))

    def _resolved_court_calibration_path(self) -> Path | None:
        """Resolve the --court-calibration source, if this run should consume an
        already-solved calibration artifact instead of re-deriving one.

        Precedence: an explicit ``--court-calibration`` always wins. Otherwise, only when
        the caller gave *neither* ``--capture-sidecar`` nor ``--court-corners`` do we fall
        back to auto-discovering ``<video_dir>/labels/court_calibration_metric15pt.json``
        (the eval_clips/ball/<clip>/labels/ convention) -- an explicit tap/ARKit choice is
        never silently overridden by auto-discovery.
        """

        opts = self.options
        if opts.court_calibration is not None:
            return opts.court_calibration
        if opts.capture_sidecar is not None or opts.court_corners is not None:
            return None
        return _auto_discover_court_calibration(opts.video)

    def _run_external_calibration(self, target: Path, source_path: Path) -> StageOutcome:
        opts = self.options
        if not source_path.is_file():
            raise _HardStageFailure(f"--court-calibration {source_path} not found")

        try:
            result = orchestrator.run_pipeline(
                clip=opts.clip,
                inputs_dir=self.clip_dir,
                run_dir=self.clip_dir,
                stage="calibration",
                sport=opts.sport,  # type: ignore[arg-type]
                runners={"calibration": orchestrator.ExternalCalibrationRunner(source_path=source_path)},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a hard failure below, not a crash
            raise _HardStageFailure(
                f"external calibration consumption failed for --court-calibration {source_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not _spine_stage_succeeded(result, stage="calibration"):
            raise _HardStageFailure(
                f"calibration spine failed while consuming --court-calibration {source_path}: "
                f"{_spine_failure_detail(result)}"
            )

        payload = _read_json(target)
        self._set_court_trust_band(payload, target)

        auto_discovered = opts.court_calibration is None
        intrinsics = payload.get("intrinsics", {}) or {}
        notes = [
            f"{'auto-discovered' if auto_discovered else 'consumed'} externally-provided court calibration "
            f"from {source_path} (intrinsics.source={intrinsics.get('source')!r}); schema + intrinsics.source "
            "validated (threed.racketsport.orchestrator.ExternalCalibrationRunner), PnP re-derivation from "
            "--court-corners/--capture-sidecar skipped",
        ]
        dist = [float(v) for v in (intrinsics.get("dist") or [])]
        dist_nonzero = any(abs(v) > 1e-9 for v in dist)
        if dist_nonzero:
            notes.append(
                f"intrinsics.dist is nonzero ({dist}): the tracking stage's court-plane footpoint projection "
                "(threed.racketsport.person_fast.build_tracks) and the world stage's ball/paddle grounding "
                "(threed.racketsport.virtual_world) both apply calibration.homography directly to raw pixels "
                "without undistorting first -- a known, documented gap from the metric-15pt migration notes "
                "(see threed/racketsport/court_calibration_metric15.py and "
                "runs/cal_metric_15pt_20260702T041729Z/); fixing those call sites is out of this migration's "
                "scope. BODY-mesh grounding/overlay consumers are affected too and are owned by another lane."
            )
        return StageOutcome(
            stage="calibration",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=["court_calibration.json", "court_zones.json", "net_plane.json", "court_line_evidence.json"],
            trust_badge=self.trust_bands["court"]["badge"] if self.trust_bands.get("court") else None,
            metrics={
                "reprojection_median_px": payload.get("reprojection_error_px", {}).get("median"),
                "reprojection_p95_px": payload.get("reprojection_error_px", {}).get("p95"),
                "intrinsics_source": intrinsics.get("source"),
                "intrinsics_dist_nonzero": dist_nonzero,
            },
        )

    # ------------------------------------------------------------------
    # stage 3: tracking
    # ------------------------------------------------------------------

    def _stage_tracking(self) -> StageOutcome:
        target = self.clip_dir / "tracks.json"
        opts = self.options

        detector_v2_gate = self._detector_v2_correction_gate_before_tracking()
        if detector_v2_gate is not None:
            return detector_v2_gate

        court_gate = self._court_correction_gate_before_tracking()
        if court_gate is not None:
            return court_gate

        if target.is_file() and not opts.force and _valid_artifact("tracks", target):
            self.trust_bands["track"] = derive_track_trust_band(idf1=None, evidence_path=str(target))
            return StageOutcome(
                stage="tracking",
                status="skipped",
                wall_seconds=0.0,
                notes=["reusing existing valid tracks.json"],
                artifacts=["tracks.json"],
                trust_badge=self.trust_bands["track"]["badge"],
            )

        if opts.tracks_reuse is not None:
            _copy_json(opts.tracks_reuse, target)
            if not _valid_artifact("tracks", target):
                raise _HardStageFailure(f"--tracks {opts.tracks_reuse} did not validate as a tracks.json artifact")
            self.trust_bands["track"] = derive_track_trust_band(idf1=None, evidence_path=str(opts.tracks_reuse))
            return StageOutcome(
                stage="tracking",
                status="reused",
                wall_seconds=0.0,
                notes=[f"reused already-computed champion tracks.json from {opts.tracks_reuse} (resume/reuse path)"],
                artifacts=["tracks.json"],
                trust_badge=self.trust_bands["track"]["badge"],
            )

        if opts.no_gpu:
            return StageOutcome(
                stage="tracking",
                status="blocked",
                wall_seconds=0.0,
                notes=["--no-gpu set and no --tracks reuse artifact given; skipping live BoT-SORT tracking"],
            )

        # Step 1: BoT-SORT-ReID "loose pool" via the existing real spine.
        manifest_path, manifest_notes = self._runtime_manifest_for_local_host()
        court_margin_m, court_margin_notes = self._tracking_court_margin()
        try:
            result = orchestrator.run_pipeline(
                clip=opts.clip,
                inputs_dir=self.clip_dir,
                run_dir=self.clip_dir,
                stage="tracking",
                sport=opts.sport,  # type: ignore[arg-type]
                device=opts.device,
                max_frames=opts.max_frames,
                tracking_mode="real",
                tracking_video=self.clip_dir / f"source{opts.video.suffix.lower()}",
                manifest_path=manifest_path,
                tracker_config_path=opts.tracker_config_path,
                max_players=opts.max_players,
                court_margin_m=court_margin_m,
                # Task #45 S2: this clip_dir's calibration stage already ran (via
                # _stage_calibration, earlier in this same process_video run) -- treat its
                # artifacts as authoritative instead of having run_pipeline's internal
                # tracking-depends-on-calibration walk re-derive them from scratch here.
                reuse_existing_stage_artifacts=True,
            )
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(
                stage="tracking",
                status="degraded",
                wall_seconds=0.0,
                notes=[f"live BoT-SORT tracking unavailable ({type(exc).__name__}: {exc}); no tracks produced"],
            )
        if not _spine_stage_succeeded(result, stage="tracking"):
            return StageOutcome(
                stage="tracking",
                status="degraded",
                wall_seconds=0.0,
                notes=[f"live BoT-SORT tracking failed: {_spine_failure_detail(result)}"],
            )

        notes = [
            "ran BoT-SORT-ReID loose-pool tracking (YOLO26m + BoT-SORT via threed.racketsport.orchestrator)",
            *court_margin_notes,
            *manifest_notes,
        ]

        # Step 2: raw-pool global association refinement (the champion recipe).
        if opts.global_association:
            refined = self._attempt_global_association()
            notes.extend(refined)

        artifacts = ["tracks.json"]
        for raw_pool_name in ("raw_tracked_detections.json", "tracked_detections.json", "metrics.json"):
            if (self.clip_dir / raw_pool_name).is_file():
                artifacts.append(raw_pool_name)

        self.trust_bands["track"] = derive_track_trust_band(idf1=None, evidence_path=str(target))
        return StageOutcome(
            stage="tracking",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=artifacts,
            trust_badge=self.trust_bands["track"]["badge"],
        )

    def _tracking_court_margin(self) -> tuple[float, list[str]]:
        if not (self.clip_dir / "auto_court_corners_preview.json").is_file():
            return 0.0, []
        return AUTO_COURT_PREVIEW_TRACKING_MARGIN_M, [
            f"auto-court preview calibration detected; using {AUTO_COURT_PREVIEW_TRACKING_MARGIN_M:.1f}m "
            "tracking court margin so unverified preview uploads do not collapse to an empty replay"
        ]

    def _court_correction_gate_before_tracking(self) -> StageOutcome | None:
        calibration_path = self.clip_dir / "court_calibration.json"
        evidence_path = self.clip_dir / "court_line_evidence.json"
        calibration = _read_optional_json(calibration_path)
        if not isinstance(calibration, Mapping):
            return None
        evidence = _read_optional_json(evidence_path)
        if not _court_calibration_needs_correction(calibration, evidence):
            return None

        correction = _build_court_correction_task(
            calibration_path=calibration_path,
            calibration=calibration,
            evidence_path=evidence_path,
            evidence=evidence,
        )
        _write_json(self.clip_dir / COURT_CORRECTION_TASK_NAME, correction)
        return StageOutcome(
            stage="tracking",
            status="blocked",
            wall_seconds=0.0,
            notes=[
                "court_calibration_unverified_or_evidence_not_ready: wrote "
                f"{COURT_CORRECTION_TASK_NAME} and blocked court-dependent tracking"
            ],
            artifacts=[COURT_CORRECTION_TASK_NAME],
            metrics={
                "court_status": "needs_user_correction",
                "blocked_downstream": COURT_CORRECTION_BLOCKED_DOWNSTREAM,
                "reason": "court_calibration_unverified_or_evidence_not_ready",
            },
        )

    def _detector_v2_correction_gate_before_tracking(self) -> StageOutcome | None:
        proposal_path = self.clip_dir / COURT_DETECTOR_V2_PROPOSAL_NAME
        if not proposal_path.is_file():
            return None
        proposal = _read_optional_json(proposal_path)
        if not isinstance(proposal, Mapping):
            return None
        if _court_detector_v2_promoted(proposal):
            return None

        correction = _build_detector_v2_correction_task(proposal_path=proposal_path, proposal=proposal)
        _write_json(self.clip_dir / COURT_CORRECTION_TASK_NAME, correction)
        return StageOutcome(
            stage="tracking",
            status="blocked",
            wall_seconds=0.0,
            notes=[
                "court_detector_v2_not_promoted: wrote "
                f"{COURT_CORRECTION_TASK_NAME} and blocked court-dependent tracking"
            ],
            artifacts=[COURT_CORRECTION_TASK_NAME],
            metrics={
                "court_status": "needs_user_correction",
                "blocked_downstream": COURT_CORRECTION_BLOCKED_DOWNSTREAM,
                "reason": "court_detector_v2_not_promoted",
            },
        )

    def _attempt_global_association(self) -> list[str]:
        opts = self.options
        if not _has_raw_pool_artifact(self.clip_dir):
            return [
                "raw-pool global association skipped: no tracked_detections.json or "
                "raw_tracked_detections.json exported by tracking; kept loose-pool tracks.json"
            ]
        if not opts.reid_model.is_file():
            return [f"raw-pool global association skipped: --reid-model {opts.reid_model} not found; kept loose-pool tracks.json"]

        out_dir = self.clip_dir / "global_association"
        resolved_reid_device = resolve_reid_device(opts.device)
        reid_batch_size = 64
        requested_profile = opts.global_association_profile or DEFAULT_GLOBAL_ASSOCIATION_PROFILE
        authority_config, profile_name = _raw_pool_authority_config_for_profile(
            requested_profile,
            expected_players=opts.max_players,
            reid_device=resolved_reid_device,
            reid_batch_size=reid_batch_size,
        )
        try:
            report = run_raw_pool_authority_candidate(
                clip_id=opts.clip,
                candidate="botsort_loose_pool_raw",
                video_path=self.clip_dir / f"source{opts.video.suffix.lower()}",
                raw_pool_dir=self.clip_dir,
                calibration_path=self.clip_dir / "court_calibration.json",
                out_dir=out_dir,
                reid_model_path=opts.reid_model,
                embedding_export_path=None,
                ground_truth_path=None,
                expected_players=opts.max_players,
                config=authority_config,
            )
        except Exception as exc:  # noqa: BLE001
            return [f"raw-pool global association failed ({type(exc).__name__}: {exc}); kept loose-pool tracks.json"]

        refined_tracks = Path(report["tracks_path"])
        if refined_tracks.is_file() and _valid_artifact("tracks", refined_tracks):
            _copy_json(refined_tracks, self.clip_dir / "tracks.json")
            return [
                "refined tracks.json with raw-pool global association "
                f"(profile={profile_name}, osnet ReID + motion, device={resolved_reid_device}, "
                f"batch={reid_batch_size}, out_dir={out_dir})"
            ]
        return ["raw-pool global association ran but produced no valid tracks.json; kept loose-pool tracks.json"]

    def _stage_camera_motion(self) -> StageOutcome:
        opts = self.options
        threshold = float(opts.camera_motion_auto_threshold)
        params = CameraMotionParams.legacy() if opts.camera_motion_estimator == "legacy" else CameraMotionParams()
        params = replace(
            params,
            flow_backend=opts.camera_motion_flow_backend,
            use_person_masks=opts.camera_motion_person_masks,
        )
        if opts.camera_motion_path is not None:
            self._set_camera_motion_auto(
                self._camera_motion_auto_decision(
                    score=None,
                    threshold=threshold,
                    enabled=True,
                    forced="explicit",
                    probe_wall_seconds=0.0,
                    sampled_frame_count=0,
                )
            )
            return StageOutcome(
                stage="camera_motion",
                status="skipped",
                wall_seconds=0.0,
                notes=[f"camera_motion explicit artifact supplied: {opts.camera_motion_path}"],
                artifacts=[],
                trust_badge="preview",
                metrics={"camera_motion_auto": self._camera_motion_auto},
            )
        if opts.skip_camera_motion:
            self._set_camera_motion_auto(
                self._camera_motion_auto_decision(
                    score=None,
                    threshold=threshold,
                    enabled=False,
                    forced="off",
                    probe_wall_seconds=0.0,
                    sampled_frame_count=0,
                )
            )
            return StageOutcome(
                stage="camera_motion",
                status="skipped",
                wall_seconds=0.0,
                trust_badge="preview",
                notes=["camera_motion force-disabled by --disable-camera-motion"],
                metrics={"camera_motion_auto": self._camera_motion_auto},
            )

        target = self.clip_dir / "camera_motion.json"
        source_video = self.clip_dir / f"source{opts.video.suffix.lower()}"
        calibration_path = self.clip_dir / "court_calibration.json"
        tracks_path = self.clip_dir / "tracks.json"
        missing = [path.name for path in (source_video, calibration_path, tracks_path) if not path.is_file()]
        if missing:
            self._set_camera_motion_auto(
                self._camera_motion_auto_decision(
                    score=None,
                    threshold=threshold,
                    enabled=False,
                    forced="on" if opts.enable_camera_motion else "auto",
                    probe_wall_seconds=0.0,
                    sampled_frame_count=0,
                )
            )
            return StageOutcome(
                stage="camera_motion",
                status="skipped",
                wall_seconds=0.0,
                notes=[f"camera_motion requires source video, court_calibration.json, and tracks.json; missing={missing}"],
                trust_badge="preview",
                metrics={"camera_motion_auto": self._camera_motion_auto},
            )

        if opts.enable_camera_motion:
            self._set_camera_motion_auto(
                self._camera_motion_auto_decision(
                    score=None,
                    threshold=threshold,
                    enabled=True,
                    forced="on",
                    probe_wall_seconds=0.0,
                    sampled_frame_count=0,
                )
            )
        else:
            probe_params = replace(
                params,
                processing_scale=min(float(params.processing_scale), CAMERA_MOTION_PROBE_PROCESSING_SCALE),
                max_corners=min(int(params.max_corners), CAMERA_MOTION_PROBE_MAX_CORNERS),
                temporal_smoothing=False,
            )
            try:
                probe = estimate_camera_motion_probe(
                    source_video,
                    calibration_path,
                    tracks_path=tracks_path,
                    params=probe_params,
                    threshold=threshold,
                )
            except Exception as exc:  # noqa: BLE001
                self._set_camera_motion_auto(
                    self._camera_motion_auto_decision(
                        score=None,
                        threshold=threshold,
                        enabled=False,
                        forced="auto",
                        probe_wall_seconds=0.0,
                        sampled_frame_count=0,
                    )
                )
                return StageOutcome(
                    stage="camera_motion",
                    status="degraded",
                    wall_seconds=0.0,
                    notes=[f"camera_motion AUTO probe failed ({type(exc).__name__}: {exc}); placement will proceed without camera compensation"],
                    trust_badge="preview",
                    metrics={"camera_motion_auto": self._camera_motion_auto},
                )
            camera_motion_auto = self._camera_motion_auto_decision(
                score=float(probe.get("motion_score", 0.0) or 0.0),
                threshold=float(probe.get("threshold", threshold) or threshold),
                enabled=bool(probe.get("enabled")),
                forced=str(probe.get("forced") or "auto"),
                probe_wall_seconds=float(probe.get("wall_seconds", 0.0) or 0.0),
                sampled_frame_count=int(probe.get("sampled_frame_count", 0) or 0),
            )
            for key in (
                "decode_orientation_mismatch",
                "decode_orientation_consequential_mismatch",
                "decode_orientation_untrusted",
                "decode_orientation_mismatch_reason",
            ):
                if key in probe:
                    camera_motion_auto[key] = probe[key]
            self._set_camera_motion_auto(camera_motion_auto)
            if not self._camera_motion_auto["enabled"]:
                return StageOutcome(
                    stage="camera_motion",
                    status="skipped",
                    wall_seconds=self._camera_motion_auto["probe_wall_seconds"],
                    notes=[
                        "camera_motion AUTO probe classified clip as static; placement will use the default no-camera-motion path",
                        f"motion_score={self._camera_motion_auto['score']} threshold={self._camera_motion_auto['threshold']}",
                    ],
                    trust_badge="preview",
                    metrics={"camera_motion_auto": self._camera_motion_auto},
                )

        if target.is_file() and not opts.force:
            try:
                validate_camera_motion_payload(_read_json(target))
            except Exception as exc:  # noqa: BLE001
                return StageOutcome(
                    stage="camera_motion",
                    status="degraded",
                    wall_seconds=0.0,
                    notes=[f"existing camera_motion.json failed validation ({type(exc).__name__}: {exc}); placement will proceed without rewriting it"],
                    trust_badge="preview",
                    metrics={"camera_motion_auto": self._camera_motion_auto},
                )
            return StageOutcome(
                stage="camera_motion",
                status="reused",
                wall_seconds=0.0,
                notes=["reusing existing valid camera_motion.json"],
                artifacts=["camera_motion.json"],
                trust_badge="preview",
                metrics={"camera_motion_auto": self._camera_motion_auto},
            )

        started = time.monotonic()
        try:
            payload = estimate_camera_motion(source_video, calibration_path, tracks_path=tracks_path, params=params)
            write_camera_motion_json(payload, target)
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(
                stage="camera_motion",
                status="degraded",
                wall_seconds=time.monotonic() - started,
                notes=[f"camera_motion failed ({type(exc).__name__}: {exc}); placement will proceed without camera compensation"],
                trust_badge="preview",
                metrics={"camera_motion_auto": self._camera_motion_auto},
            )
        summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
        frame_count = int(summary.get("n_frames", 0) or 0)
        wall_seconds = time.monotonic() - started
        runtime_ms_per_frame = round((wall_seconds * 1000.0) / frame_count, 3) if frame_count else 0.0
        return StageOutcome(
            stage="camera_motion",
            status="ran",
            wall_seconds=wall_seconds,
            notes=[
                f"estimated preview-only camera_motion.json estimator={params.estimator_mode} flow_backend={params.flow_backend} person_masks={params.use_person_masks}",
                f"verified={payload.get('verified')} not_gate_verified={payload.get('not_gate_verified')}",
            ],
            artifacts=["camera_motion.json"],
            trust_badge="preview",
            metrics={
                "n_frames": frame_count,
                "n_compensated": int(summary.get("n_compensated", 0) or 0),
                "runtime_ms_per_frame": runtime_ms_per_frame,
                "camera_motion_auto": self._camera_motion_auto,
            },
        )

    def _stage_placement(self) -> StageOutcome:
        return self._run_placement_stage(refine_from_sam3d=False)

    def _stage_placement_refine(self) -> StageOutcome:
        return StageOutcome(
            stage="placement_refine",
            status="skipped",
            wall_seconds=0.0,
            notes=[
                "same-pass post-BODY placement_refine is disabled by R3; SAM3D foot pixels may only feed a "
                "second pass before a fresh BODY run, never an in-place tracks.json rewrite before world build"
            ],
            metrics={"same_pass_track_rewrite_disabled": True},
        )

    def _run_placement_stage(self, *, refine_from_sam3d: bool) -> StageOutcome:
        stage_name = "placement_refine" if refine_from_sam3d else "placement"
        tracks_path = self.clip_dir / "tracks.json"
        calibration_path = self.clip_dir / "court_calibration.json"
        if not tracks_path.is_file():
            return StageOutcome(stage=stage_name, status="blocked", wall_seconds=0.0, notes=["requires tracks.json"])
        if not calibration_path.is_file():
            return StageOutcome(stage=stage_name, status="blocked", wall_seconds=0.0, notes=["requires court_calibration.json"])

        native2d_path = self._placement_native2d_path()
        camera_motion_path, camera_motion_source = self._placement_camera_motion_path()
        sam3d_path = self.clip_dir / "sam3d_keypoints_2d.json" if refine_from_sam3d else None
        if sam3d_path is not None and not sam3d_path.is_file():
            sam3d_path = None
        stance_phases_path = self._placement_stance_phases_path() if refine_from_sam3d else None
        placement_path = self.clip_dir / "placement.json"
        try:
            result = rewrite_tracks_with_placement(
                tracks_path=tracks_path,
                calibration_path=calibration_path,
                placement_path=placement_path,
                native2d_keypoints_path=native2d_path,
                sam3d_keypoints_path=sam3d_path,
                stance_phases_path=stance_phases_path,
                foot_contact_phases_out_path=self.clip_dir / "foot_contact_phases.json",
                camera_motion_path=camera_motion_path,
                refine_from_sam3d=refine_from_sam3d,
                config=PlacementConfig(undistort=self.options.placement_undistort),
            )
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(
                stage=stage_name,
                status="degraded",
                wall_seconds=0.0,
                notes=[f"{stage_name} failed ({type(exc).__name__}: {exc}); kept current tracks.json"],
            )

        source_notes = [f"{name}={count}" for name, count in sorted(result.source_counts.items()) if count]
        placement_summary = getattr(result, "summary", {}) if isinstance(getattr(result, "summary", {}), dict) else {}
        notes = [
            "rewrote tracks.json world_xy via foot-keypoint placement fusion",
            f"coverage_unchanged={result.coverage_unchanged}",
            f"source_counts({', '.join(source_notes) if source_notes else 'none'})",
        ]
        notes.extend(_placement_camera_motion_notes(camera_motion_path, camera_motion_source, placement_summary))
        notes.extend(_placement_honesty_notes(placement_summary))
        if native2d_path is not None:
            notes.append(f"native2d_keypoints={native2d_path}")
        if sam3d_path is not None:
            notes.append(f"sam3d_keypoints={sam3d_path}")
        if stance_phases_path is not None:
            notes.append(f"stance_phases={stance_phases_path}")
        artifacts = ["placement.json", "tracks.json", "tracks_prewrite_backup.json"]
        if (self.clip_dir / "foot_contact_phases.json").is_file():
            artifacts.append("foot_contact_phases.json")
            notes.append("foot_contact_phases=foot_contact_phases.json")
        return StageOutcome(
            stage=stage_name,
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=artifacts,
            metrics={
                "coverage_unchanged": result.coverage_unchanged,
                "source_counts": result.source_counts,
                "court_bounds_violations": getattr(result, "court_bounds_violations", 0),
                "camera_motion_frames_used": int(placement_summary.get("camera_motion_frames_used", 0) or 0),
                "camera_motion_frames_uncompensated": int(placement_summary.get("camera_motion_frames_uncompensated", 0) or 0),
            },
        )

    def _placement_native2d_path(self) -> Path | None:
        explicit = self.options.placement_keypoints_2d
        if explicit is not None:
            return explicit if explicit.is_file() else None
        candidates = sorted(ROOT.glob(f"runs/native2d_{self.options.clip}*/keypoints_2d.json"))
        if candidates:
            return candidates[-1]
        slug = self.options.clip.split("/")[-1]
        candidates = sorted(ROOT.glob(f"runs/native2d_{slug}*/keypoints_2d.json"))
        if candidates:
            return candidates[-1]
        prefix = slug.split("_", 1)[0]
        candidates = sorted(ROOT.glob(f"runs/native2d_{prefix}*/keypoints_2d.json"))
        return candidates[-1] if candidates else None

    def _placement_stance_phases_path(self) -> Path | None:
        for name in ("foot_contact_phases.json", "foot_pin_audit.json"):
            path = self.clip_dir / name
            if path.is_file():
                return path
        return None

    def _placement_camera_motion_path(self) -> tuple[Path | None, str]:
        explicit = self.options.camera_motion_path
        if explicit is not None:
            return explicit, "explicit"
        if self._camera_motion_auto_decision_made and not bool(self._camera_motion_auto.get("enabled")):
            return None, "auto_disabled"
        candidate = self.clip_dir / "camera_motion.json"
        if candidate.is_file():
            return candidate, "auto_discovered"
        return None, "absent"

    def _runtime_manifest_for_local_host(self) -> tuple[Path, list[str]]:
        """Return a per-run manifest with local checkpoint paths when available."""

        manifest_path = self.options.manifest_path
        try:
            payload = _read_json(manifest_path)
        except Exception:  # noqa: BLE001 - let the real runner report manifest errors
            return manifest_path, []
        models = payload.get("models")
        if not isinstance(models, list):
            return manifest_path, []

        local_overrides = {
            "yolo26m": ROOT / "models" / "checkpoints" / "yolo26m.pt",
        }
        notes: list[str] = []
        changed = False
        runtime_payload = copy.deepcopy(payload)
        for entry in runtime_payload.get("models", []):
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id", ""))
            local_candidate = local_overrides.get(model_id)
            if local_candidate is None or not local_candidate.is_file():
                continue
            declared = Path(str(entry.get("local_path", ""))).expanduser()
            if declared.is_file():
                continue
            entry["local_path"] = str(local_candidate)
            changed = True
            notes.append(
                f"runtime manifest override: {model_id} declared checkpoint {declared} was missing on this host; "
                f"using repo-local {local_candidate}"
            )

        if not changed:
            return manifest_path, []
        out = self.clip_dir / "runtime_model_manifest.json"
        _write_json(out, runtime_payload)
        return out, notes

    # ------------------------------------------------------------------
    # stage 3b: opt-in rally gating
    # ------------------------------------------------------------------

    def _stage_rally_gating(self) -> StageOutcome:
        tracks_path = self.clip_dir / "tracks.json"
        ball_track_path = self.clip_dir / "ball_track.json"
        notes: list[str] = [
            "rally gating active: deriving loose rally spans from already-computed cheap signals before downstream heavy stages",
            "trust note: runtime optimization only, not ground truth; signals are OR-fused and biased toward over-inclusion",
        ]
        artifacts: list[str] = []

        if not ball_track_path.is_file() and not self.options.skip_ball:
            reusable = self._resolved_ball_track_reuse_path()
            if reusable is not None:
                source_path, source_kind = reusable
                _copy_json(source_path, ball_track_path)
                if _valid_artifact("ball_track", ball_track_path):
                    notes.append(f"rally gating preloaded {source_kind} ball_track.json from {source_path}")
                    guard_notes, _, _ = self._guard_reused_ball_track_timing(
                        ball_track_path,
                        source_path=source_path,
                        source_kind=source_kind,
                    )
                    notes.extend(guard_notes)
                else:
                    ball_track_path.unlink(missing_ok=True)
                    notes.append(f"rally gating ignored {source_kind} ball track from {source_path}: artifact did not validate")
        elif ball_track_path.is_file() and _valid_artifact("ball_track", ball_track_path):
            guard_notes, _, _ = self._guard_reused_ball_track_timing(
                ball_track_path,
                source_path=ball_track_path,
                source_kind="existing valid",
            )
            notes.extend(guard_notes)

        tracks_payload = _read_optional_json(tracks_path)
        ball_payload = _read_optional_json(ball_track_path)
        audio_onsets_path = self.clip_dir / "audio_onsets_v2.json"
        audio_payload = _read_optional_json(audio_onsets_path) if audio_onsets_path.is_file() else None
        if tracks_payload is None and ball_payload is None and audio_payload is None:
            return StageOutcome(
                stage="rally_gating",
                status="blocked",
                wall_seconds=0.0,
                notes=[*notes, "no tracks.json, ball_track.json, or audio_onsets_v2.json available; no spans derived"],
            )

        duration_s = _clip_duration_seconds(
            self.options.video,
            tracks_payload=tracks_payload,
            ball_track_payload=ball_payload,
            audio_onsets_payload=audio_payload,
        )
        rally_payload = build_rally_spans_artifact(
            clip_id=self.options.clip,
            duration_s=duration_s,
            ball_track=ball_payload,
            tracks=tracks_payload,
            audio_onsets=audio_payload,
            ball_track_path=str(ball_track_path) if ball_payload is not None else None,
            tracks_path=str(tracks_path) if tracks_payload is not None else None,
            audio_onsets_path=str(audio_onsets_path) if audio_payload is not None else None,
            pad_seconds=self.options.rally_gating_pad_seconds,
        )
        rally_path = self.clip_dir / "rally_spans.json"
        _write_json(rally_path, rally_payload)
        artifacts.append("rally_spans.json")
        spans = list(rally_payload.get("spans", []))
        if not spans:
            return StageOutcome(
                stage="rally_gating",
                status="degraded",
                wall_seconds=0.0,
                notes=[*notes, "no rally spans derived; downstream stages keep full available artifacts"],
                artifacts=artifacts,
                metrics={"span_count": 0, "dead_time_fraction": rally_payload.get("dead_time_fraction")},
            )

        metrics: dict[str, Any] = {
            "span_count": len(spans),
            "dead_time_fraction": rally_payload.get("dead_time_fraction"),
        }
        if tracks_payload is not None:
            pre_tracks_path = self.clip_dir / "tracks_pre_rally_gating.json"
            if self.options.force or not pre_tracks_path.is_file():
                _write_json(pre_tracks_path, tracks_payload)
            filtered_tracks, before_count, after_count = _filter_tracks_payload_to_rally_spans(tracks_payload, spans)
            _write_json(tracks_path, filtered_tracks)
            if not _valid_artifact("tracks", tracks_path):
                _copy_json(pre_tracks_path, tracks_path)
                notes.append("filtered tracks.json did not validate; restored tracks_pre_rally_gating.json")
            else:
                skipped = before_count - after_count
                artifacts.append("tracks_pre_rally_gating.json")
                metrics.update({"track_frames_before": before_count, "track_frames_after": after_count, "track_frames_skipped": skipped})
                notes.append(f"filtered tracks.json to rally spans: kept {after_count}/{before_count} tracked player-frame(s), skipped {skipped} dead-time frame(s)")

        if ball_payload is not None:
            pre_ball_path = self.clip_dir / "ball_track_pre_rally_gating.json"
            if self.options.force or not pre_ball_path.is_file():
                _write_json(pre_ball_path, ball_payload)
            filtered_ball, before_count, after_count = _filter_ball_payload_to_rally_spans(ball_payload, spans)
            _write_json(ball_track_path, filtered_ball)
            if not _valid_artifact("ball_track", ball_track_path):
                _copy_json(pre_ball_path, ball_track_path)
                notes.append("filtered ball_track.json did not validate; restored ball_track_pre_rally_gating.json")
            else:
                skipped = before_count - after_count
                artifacts.append("ball_track_pre_rally_gating.json")
                metrics.update({"ball_frames_before": before_count, "ball_frames_after": after_count, "ball_frames_skipped": skipped})
                notes.append(f"filtered ball_track.json to rally spans: kept {after_count}/{before_count} ball frame(s), skipped {skipped} dead-time frame(s)")

        return StageOutcome(
            stage="rally_gating",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=artifacts,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # stage 4: frames (body_frames/ JPEGs for body)
    # ------------------------------------------------------------------

    def _stage_frames(self) -> StageOutcome:
        """Materialize body_frames/ -- the JPEG directory the body stage
        (Fast SAM-3D-Body joints and tier-1 mesh, local or remote) reads via
        threed.racketsport.orchestrator._find_body_frame_image. Task #46: this
        directory previously never got populated by process_video.py at all, so
        BODY degraded to skeleton-only/court-only on every run regardless of
        GPU/checkpoint availability. See threed.racketsport.process_video_body_frames
        for the scheduling rules (tracks.json-bounded default, tier-rule union
        when frame_compute_plan.json already exists, hard cap).
        """

        opts = self.options
        tracks_path = self.clip_dir / "tracks.json"
        out_dir = self.clip_dir / "body_frames"

        if not tracks_path.is_file():
            return StageOutcome(
                stage="frames",
                status="blocked",
                wall_seconds=0.0,
                notes=["requires tracks.json (tracking stage did not produce one); BODY will degrade to court/skeleton-only"],
            )

        existing = sorted(out_dir.glob("frame_*.jpg")) if out_dir.is_dir() else []
        if existing and not opts.force:
            return StageOutcome(
                stage="frames",
                status="skipped",
                wall_seconds=0.0,
                notes=[f"reusing {len(existing)} already-extracted body_frames/ JPEG(s)"],
                artifacts=["body_frames/"],
                metrics={"frame_count": len(existing)},
            )

        if opts.no_gpu:
            return StageOutcome(
                stage="frames",
                status="skipped",
                wall_seconds=0.0,
                notes=["--no-gpu set: BODY is skipped by design this run, so body_frames/ extraction is not needed"],
            )

        try:
            result = materialize_process_video_frames(
                video_path=self.clip_dir / f"source{opts.video.suffix.lower()}",
                tracks_path=tracks_path,
                out_dir=out_dir,
                frame_compute_plan_path=self.clip_dir / "frame_compute_plan.json",
                max_frames=opts.max_frames,
            )
        except Exception as exc:  # noqa: BLE001 - frame extraction must never crash the pipeline
            return StageOutcome(
                stage="frames",
                status="degraded",
                wall_seconds=0.0,
                notes=[
                    f"body_frames/ extraction unavailable ({type(exc).__name__}: {exc}); BODY will degrade "
                    "to court/skeleton-only for this run (SAM-3D needs per-frame JPEGs to run the real model)"
                ],
            )

        total_mb = result["total_bytes"] / (1024 * 1024)
        notes = [
            *result["notes"],
            f"extracted {result['frame_count']} scheduled JPEG(s) into body_frames/ ({total_mb:.2f} MB) via ffmpeg "
            "(threed.racketsport.body_frame_materialization, the same extraction body_video_smoke.py already exercises)",
        ]
        return StageOutcome(
            stage="frames",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=["body_frames/"],
            metrics={
                "frame_count": result["frame_count"],
                "total_bytes": result["total_bytes"],
                "total_mb": round(total_mb, 3),
                "capped": result["schedule"]["capped"],
                "schedule_source": result["schedule"]["source"],
            },
        )

    # ------------------------------------------------------------------
    # stage 5: pose
    # ------------------------------------------------------------------

    def _stage_pose(self) -> StageOutcome:
        return StageOutcome(
            stage="pose",
            status="skipped",
            wall_seconds=0.0,
            notes=[
                "production pose stage removed by SAM-3D-only program; SAM-3D BODY is the offline skeleton source"
            ],
        )

    # ------------------------------------------------------------------
    # stage 6: ball
    # ------------------------------------------------------------------

    def _stage_ball(self) -> StageOutcome:
        opts = self.options
        target = self.clip_dir / "ball_track.json"
        notes: list[str] = []

        if opts.skip_ball:
            return StageOutcome(stage="ball", status="skipped", wall_seconds=0.0, notes=["--skip-ball set"])

        status: StageStatus = "ran"
        source_kind = "fresh runtime"
        timing_metrics: dict[str, Any] = {}
        warnings: list[str] = []
        if target.is_file() and not opts.force and _valid_artifact("ball_track", target):
            notes.append("reusing existing valid ball_track.json")
            guard_notes, timing_metrics, warnings = self._guard_reused_ball_track_timing(
                target,
                source_path=target,
                source_kind="existing valid",
            )
            notes.extend(guard_notes)
            status = "skipped"
        elif (reusable := self._resolved_ball_track_reuse_path()) is not None:
            source_path, source_kind = reusable
            _copy_json(source_path, target)
            if not _valid_artifact("ball_track", target):
                raise _HardStageFailure(f"{source_kind} ball track {source_path} did not validate as ball_track.json")
            guard_notes, timing_metrics, warnings = self._guard_reused_ball_track_timing(
                target,
                source_path=source_path,
                source_kind=source_kind,
            )
            notes.append(f"reused {source_kind} ball_track.json from {source_path}")
            notes.extend(guard_notes)
            status = "reused"
            if source_kind.startswith("auto-discovered"):
                notes.append(
                    "auto-discovered ball track is an explicitly allowed low-confidence preview source, "
                    "not a verified fresh BALL run"
                )
        else:
            wasb_ok = self._run_wasb_zero_shot(target)
            if not wasb_ok:
                return StageOutcome(
                    stage="ball",
                    status="blocked",
                    wall_seconds=0.0,
                    notes=[
                        "no current BALL runtime/source is available: provide explicit --ball-track, install/configure "
                        "the WASB repo+checkpoint for a fresh runtime run, or pass --allow-auto-ball-track to opt "
                        "into low-confidence precomputed preview reuse; no ball data this run"
                    ],
                )
            notes.append("ran WASB-tennis zero-shot ball tracking (best-available per MASTER_PLAN.md, no fine-tuning)")

        width, height, _ = _video_probe(opts.video)
        court_corners = self._resolved_court_corners_path()
        if court_corners is not None:
            notes.extend(self._run_ball_bounce_and_inout(target, court_corners, width, height))
        else:
            notes.append("no court_corners.json available for bounce/in-out geometry; ball_track.json has trajectory only")

        payload = _read_json(target)
        self.trust_bands["ball"] = derive_ball_trust_band(source=payload.get("source"), evidence_path=str(target))
        artifacts = ["ball_track.json"]
        if (self.clip_dir / "ball_candidates.json").is_file():
            artifacts.append("ball_candidates.json")
            notes.append("wrote top-K ball candidate sidecar for the default arc chain")

        return StageOutcome(
            stage="ball",
            status=status,
            wall_seconds=0.0,
            notes=notes,
            artifacts=artifacts,
            trust_badge=self.trust_bands["ball"]["badge"],
            metrics={
                "frame_count": len(payload.get("frames", [])),
                "bounce_count": len(payload.get("bounces", [])),
                "source_kind": source_kind,
                "verified_full_ball_run": status == "ran",
                **timing_metrics,
                **({"warnings": warnings} if warnings else {}),
            },
        )

    def _guard_reused_ball_track_timing(
        self,
        ball_track_path: Path,
        *,
        source_path: Path,
        source_kind: str,
    ) -> tuple[list[str], dict[str, Any], list[str]]:
        payload = _read_json(ball_track_path)
        frames = payload.get("frames", []) if isinstance(payload, Mapping) else []
        if not isinstance(frames, list):
            raise _HardStageFailure(f"reused ball_track frames must be a list for {source_kind}: {source_path}")

        timing = _video_timing_probe(self.options.video)
        frame_times_path = self.clip_dir / "frame_times.json"
        frame_times_payload = _read_optional_json(frame_times_path) if frame_times_path.is_file() else None
        source_frame_count = len(frames)
        if source_frame_count != timing.frame_count:
            raise _HardStageFailure(
                "reused ball_track frame count mismatch: "
                f"{source_frame_count} ball frames from {source_kind} {source_path} "
                f"vs {timing.frame_count} video frames in {self.options.video}; refusing silently misaligned BALL timing"
            )

        source_fps = _maybe_positive_float(payload.get("fps"))
        source_dt = _median_frame_dt(frames)
        expected_dt = 1.0 / timing.fps
        if frame_times_payload is not None:
            expected_samples = [
                {"t": time_for_frame(index, frame_times=frame_times_payload, fps=timing.fps)}
                for index in range(source_frame_count)
            ]
            expected_dt = _median_frame_dt(expected_samples) or expected_dt
        before_coverage = _ball_timeline_coverage_fraction(
            frames,
            fps=source_fps,
            median_dt=source_dt,
            video_duration_s=timing.duration_s,
        )
        metrics: dict[str, Any] = {
            "ball_video_frame_count": timing.frame_count,
            "ball_video_fps": timing.fps,
            "ball_reuse_frame_count": source_frame_count,
            "ball_source_fps": source_fps,
            "ball_source_median_dt": source_dt,
            "ball_timeline_coverage_before": before_coverage,
        }
        notes: list[str] = []
        warnings: list[str] = []
        dt_mismatch = source_dt is not None and abs(source_dt - expected_dt) > max(0.002, expected_dt * 0.02)
        fps_mismatch = source_fps is not None and abs(source_fps - timing.fps) > max(0.01, timing.fps * 0.01)
        if not (dt_mismatch or (fps_mismatch and before_coverage is not None and before_coverage < 0.95)):
            metrics["ball_timeline_coverage_after"] = before_coverage
            return notes, metrics, warnings

        for index, frame in enumerate(frames):
            if isinstance(frame, dict):
                frame["t"] = round(time_for_frame(index, frame_times=frame_times_payload, fps=timing.fps), 6)
        payload["fps"] = float(timing.fps)
        _write_json(ball_track_path, payload)
        if not _valid_artifact("ball_track", ball_track_path):
            raise _HardStageFailure(f"rescaled reused ball_track failed schema validation: {ball_track_path}")

        normalized = _read_json(ball_track_path)
        normalized_frames = normalized.get("frames", []) if isinstance(normalized, Mapping) else []
        after_coverage = _ball_timeline_coverage_fraction(
            normalized_frames if isinstance(normalized_frames, list) else [],
            fps=timing.fps,
            median_dt=expected_dt,
            video_duration_s=timing.duration_s,
        )
        warning = (
            "WARNING: rescaled reused ball_track timestamps to source video fps because frame count matched "
            f"but timing cadence did not ({source_kind}: source_fps={source_fps}, source_dt={source_dt}, "
            f"video_fps={timing.fps})"
        )
        provenance = {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_track_timing_provenance",
            "status": "rescaled_reused_ball_track_timestamps",
            "warning": warning,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "ball_track_path": str(ball_track_path),
            "video_path": str(self.options.video),
            "source_fps": source_fps,
            "video_fps": timing.fps,
            "source_frame_count": source_frame_count,
            "video_frame_count": timing.frame_count,
            "source_median_dt": source_dt,
            "expected_dt": expected_dt,
            "frame_times_path": str(frame_times_path) if frame_times_payload is not None else None,
            "source_last_t": _last_frame_time(frames),
            "normalized_last_t": _last_frame_time(normalized_frames if isinstance(normalized_frames, list) else []),
            "coverage_before": before_coverage,
            "coverage_after": after_coverage,
            "normalization": "t=frame_times[index]" if frame_times_payload is not None else "t=index/video_fps",
            "not_ball_gate_evidence": True,
        }
        _write_json(ball_track_path.with_name("ball_track_timing_provenance.json"), provenance)
        metrics.update(
            {
                "ball_timing_normalized": True,
                "ball_timeline_coverage_after": after_coverage,
                "ball_timing_provenance": str(ball_track_path.with_name("ball_track_timing_provenance.json")),
            }
        )
        notes.append(warning)
        warnings.append(warning)
        return notes, metrics, warnings

    def _resolved_ball_track_reuse_path(self) -> tuple[Path, str] | None:
        if self.options.ball_track_reuse is not None:
            return self.options.ball_track_reuse, "explicit --ball-track"
        if not self.options.ball_track_auto_discovery:
            return None
        discovered = _auto_discover_ball_track(self.options.clip)
        if discovered is None:
            return None
        return discovered, "auto-discovered precomputed"

    def _resolved_court_corners_path(self) -> Path | None:
        if self.options.court_corners is not None and self.options.court_corners.is_file():
            return self.options.court_corners
        return None

    def _run_wasb_zero_shot(self, target: Path) -> bool:
        opts = self.options
        wasb_repo = opts.wasb_repo or (DEFAULT_WASB_REPO if DEFAULT_WASB_REPO.is_dir() else None)
        if wasb_repo is None or not opts.wasb_checkpoint.is_file():
            return False
        try:
            from threed.racketsport.wasb_adapter import run_wasb_or_convert
        except ImportError:
            return False

        _, _, fps = _video_probe(opts.video)
        try:
            run_wasb_or_convert(
                out=target,
                fps=fps,
                frame_times=_existing_optional_path(self.clip_dir / "frame_times.json"),
                metadata_out=self.clip_dir / "wasb_run.json",
                predictions_csv=None,
                video=self.clip_dir / f"source{opts.video.suffix.lower()}",
                checkpoint=opts.wasb_checkpoint,
                wasb_repo=wasb_repo,
                prediction_csv_out=None,
                batch_size=8,
                visible_threshold=0.5,
                video_range=None,
                max_frames=opts.max_frames,
                device="cpu" if opts.no_gpu else (opts.device or "cuda"),
                emit_candidates=opts.emit_ball_candidates,
                candidate_top_k=5,
            )
        except Exception:  # noqa: BLE001
            return False
        return target.is_file() and _valid_artifact("ball_track", target)

    def _resolved_ball_candidate_paths(self) -> list[Path]:
        if self.options.ball_candidates_reuse:
            return list(self.options.ball_candidates_reuse)
        local_sidecar = self.clip_dir / "ball_candidates.json"
        return [local_sidecar] if local_sidecar.is_file() else []

    def _run_ball_bounce_and_inout(self, ball_track_path: Path, court_corners_path: Path, width: int, height: int) -> list[str]:
        notes: list[str] = []
        try:
            from threed.racketsport.ball_bounce_2d import write_2d_bounce_ball_track

            write_2d_bounce_ball_track(
                ball_track_path=ball_track_path,
                court_corners_path=court_corners_path,
                out=ball_track_path,
                detector_out=self.clip_dir / "ball_bounce_2d_detector.json",
                target_image_size=(width, height),
                sport=self.options.sport,  # type: ignore[arg-type]
                command="process_video.py ball stage (2D bounce detection)",
            )
            notes.append("detected 2D bounces from image-velocity inflections (threed.racketsport.ball_bounce_2d)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"2D bounce detection skipped ({type(exc).__name__}: {exc})")
            return notes

        try:
            from threed.racketsport.ball_manual_court_inout import write_manual_court_inout_ball_track

            write_manual_court_inout_ball_track(
                ball_track_path=ball_track_path,
                court_corners_path=court_corners_path,
                out=ball_track_path,
                target_image_size=(width, height),
                summary_out=self.clip_dir / "ball_inout_summary.json",
                sport=self.options.sport,  # type: ignore[arg-type]
                uncertainty_m=None,
            )
            notes.append(
                "applied geometry-corrected manual-court in/out with honest gray zones "
                "(threed.racketsport.ball_manual_court_inout / ball_inout_uncertainty)"
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"manual-court in/out skipped ({type(exc).__name__}: {exc})")
        return notes

    # ------------------------------------------------------------------
    # stage 6b: ball_arc (auto bounces + arc solver + render sanity)
    # ------------------------------------------------------------------

    def _stage_ball_arc(self) -> StageOutcome:
        if self.options.no_ball_arc:
            return StageOutcome(stage="ball_arc", status="skipped", wall_seconds=0.0, notes=["--no-ball-arc set"])

        ball_track_path = self.clip_dir / "ball_track.json"
        calibration_path = self.clip_dir / "court_calibration.json"
        if not ball_track_path.is_file():
            return StageOutcome(
                stage="ball_arc",
                status="skipped",
                wall_seconds=0.0,
                notes=["requires ball_track.json; no default 3D ball arc artifact produced"],
            )
        if not calibration_path.is_file():
            return StageOutcome(
                stage="ball_arc",
                status="skipped",
                wall_seconds=0.0,
                notes=["requires court_calibration.json; no default 3D ball arc artifact produced"],
            )

        artifact_paths = [
            self.clip_dir / "ball_bounce_candidates.json",
            self.clip_dir / "ball_track_arc_solved.json",
            self.clip_dir / "ball_arc_render.json",
            self.clip_dir / "ball_flight_sanity.json",
            self.clip_dir / "ball_chain_manifest.json",
        ]
        if all(path.is_file() for path in artifact_paths) and not self.options.force:
            arc_payload = _read_optional_json(self.clip_dir / "ball_track_arc_solved.json") or {}
            return StageOutcome(
                stage="ball_arc",
                status="skipped",
                wall_seconds=0.0,
                notes=["reusing existing ball_arc artifacts"],
                artifacts=[path.name for path in artifact_paths],
                metrics={"solver_status": str(arc_payload.get("status") or "unknown")},
            )

        started = time.monotonic()
        try:
            ball_candidate_paths = self._resolved_ball_candidate_paths()
            result = run_default_ball_arc_chain(
                clip=self.options.clip,
                ball_track_path=ball_track_path,
                court_calibration_path=calibration_path,
                out_dir=self.clip_dir,
                ball_candidate_paths=ball_candidate_paths,
                contact_windows_path=_existing_optional_path(self.clip_dir / "contact_windows.json"),
                skeleton3d_path=_existing_optional_path(self.clip_dir / "skeleton3d.json"),
                net_plane_path=_existing_optional_path(self.clip_dir / "net_plane.json"),
                rally_spans_path=_existing_optional_path(self.clip_dir / "rally_spans.json"),
                frame_times_path=_existing_optional_path(self.clip_dir / "frame_times.json"),
            )
        except Exception as exc:  # noqa: BLE001 - default arc is fail-closed
            return StageOutcome(
                stage="ball_arc",
                status="degraded",
                wall_seconds=time.monotonic() - started,
                notes=[f"ball_arc default chain failed fail-closed ({type(exc).__name__}: {exc}); world will omit arc-solved 3D ball"],
            )

        summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
        status = str(result.get("status") or "unknown")
        notes = [
            "ran default ball_arc chain: auto-bounce candidates -> single-primary-track arc solver -> flight-sanity demotion",
        ]
        if summary.get("chain_config_degraded") == "no_candidate_sidecars":
            notes.append("ball_arc chain ran in degraded config mode: no candidate sidecars were available")
        else:
            notes.append("ball_arc chain consumed candidate sidecars under the frozen row-22 default config")
        if status != "ran":
            notes.append(f"arc solver self-killed with status={status}; virtual_world will ignore this artifact")
        return StageOutcome(
            stage="ball_arc",
            status="ran",
            wall_seconds=time.monotonic() - started,
            notes=notes,
            artifacts=["ball_bounce_candidates.json", "ball_track_arc_solved.json", "ball_arc_render.json", "ball_flight_sanity.json", "ball_chain_manifest.json"],
            metrics={
                "solver_status": status,
                "auto_bounce_candidate_count": summary.get("auto_bounce_candidate_count"),
                "coverage_world_xyz_count": summary.get("coverage_world_xyz_count"),
                "segment_count": summary.get("segment_count"),
                "ball_arc_render_sample_count": summary.get("ball_arc_render_sample_count"),
                "ball_arc_render_bridge_sample_count": summary.get("ball_arc_render_bridge_sample_count"),
                "flight_sanity_demoted_frame_count": summary.get("flight_sanity_demoted_frame_count"),
                "flight_sanity_failed_segment_count": summary.get("flight_sanity_failed_segment_count"),
                "chain_config_degraded": summary.get("chain_config_degraded"),
            },
        )

    # ------------------------------------------------------------------
    # stage 7: events (contact windows, frame_compute_plan)
    # ------------------------------------------------------------------

    def _stage_events(self) -> StageOutcome:
        skeleton_path = self.clip_dir / "skeleton3d.json"
        tracks_path = self.clip_dir / "tracks.json"
        ball_track_path = self.clip_dir / "ball_track.json"
        contact_windows_path = self.clip_dir / "contact_windows.json"
        notes: list[str] = []

        if contact_windows_path.is_file() and not self.options.force and _valid_artifact("contact_windows", contact_windows_path):
            artifacts = ["contact_windows.json"]
            if tracks_path.is_file() and not (self.clip_dir / "frame_compute_plan.json").is_file():
                contact_windows = _read_json(contact_windows_path)
                tracks = validate_artifact_file("tracks", tracks_path)
                ball_aware_events, ball_track_arc_solved, ba_notes = self._resolve_ball_aware_inputs()
                plan = build_frame_compute_plan(
                    tracks,
                    ball_track=_read_json(ball_track_path) if ball_track_path.is_file() else None,
                    contact_windows=contact_windows,
                    expected_players=self.options.max_players,
                    ball_aware_events=ball_aware_events,
                    ball_track_arc_solved=ball_track_arc_solved,
                    **self._mesh_coverage_kwargs(),
                )
                plan, demo_mesh_count = self._ensure_auto_court_preview_demo_mesh(plan)
                write_frame_compute_plan(self.clip_dir / "frame_compute_plan.json", plan)
                artifacts.append("frame_compute_plan.json")
                events = contact_windows.get("events", []) if isinstance(contact_windows, Mapping) else []
                notes = [
                    "reused existing valid contact_windows.json",
                    "regenerated missing frame_compute_plan.json from existing contact windows before BODY scheduling",
                    *ba_notes,
                ]
                if demo_mesh_count:
                    notes.append(
                        f"added auto-court preview demo mesh window covering {demo_mesh_count} tracked frame(s)"
                    )
                return StageOutcome(
                    stage="events",
                    status="ran",
                    wall_seconds=0.0,
                    notes=notes,
                    artifacts=artifacts,
                    metrics={
                        "contact_event_count": len(events),
                        "contact_cues": _contact_cue_summary(contact_windows),
                        **self._mesh_coverage_plan_metrics(plan),
                    },
                )
            return StageOutcome(
                stage="events",
                status="skipped",
                wall_seconds=0.0,
                notes=["reusing existing valid contact_windows.json"],
                artifacts=artifacts,
            )

        fps = _read_json(tracks_path).get("fps", 30.0) if tracks_path.is_file() else 30.0

        if _is_real_sam3d_skeleton(skeleton_path):
            wrist_payload = build_wrist_velocity_peaks_from_file(skeleton_path, require_lane_a=False)
            notes.append("derived wrist_velocity_peaks.json from SAM-3D skeleton3d.json")
        else:
            wrist_payload = build_blocked_wrist_velocity_peaks(
                source_path=skeleton_path,
                blocker="missing_sam3d_skeleton3d",
            )
            notes.append(
                "SAM-3D skeleton unavailable before BODY; wrist_velocity_peaks.json is blocked, not pose-derived"
            )
        _write_json(self.clip_dir / "wrist_velocity_peaks.json", wrist_payload)

        ball_inflections_path = self.clip_dir / "ball_inflections.json"
        if ball_track_path.is_file():
            ball_inflections = build_ball_inflections_from_ball_track(
                _read_json(ball_track_path),
                frame_times=_existing_optional_path(self.clip_dir / "frame_times.json"),
            )
            _write_json(ball_inflections_path, ball_inflections)
            notes.append("derived ball_inflections.json from ball_track.json")
        else:
            ball_inflections = {"schema_version": 1, "artifact_type": "racketsport_ball_inflections", "fps": fps, "candidates": []}
            _write_json(ball_inflections_path, ball_inflections)
            notes.append("no ball_track.json available; ball_inflections.json has zero candidates (wrist cues only)")

        audio_onsets_payload: Any = []
        if not self.options.skip_audio:
            audio_onsets_payload, audio_note = self._attempt_audio_onsets(fps)
            notes.append(audio_note)

        contact_windows = fuse_contact_windows_from_cue_payloads(
            fps=fps,
            frame_times=_existing_optional_path(self.clip_dir / "frame_times.json"),
            audio_onsets_payload=audio_onsets_payload,
            wrist_velocity_peaks_payload=wrist_payload,
            ball_inflections_payload=ball_inflections,
            require_audio=False,
            allow_wrist_only_contact_hints=True,
        )
        _write_json(contact_windows_path, contact_windows)
        cue_summary = _contact_cue_summary(
            contact_windows,
            wrist_payload=wrist_payload,
            ball_inflections_payload=ball_inflections,
            audio_onsets_payload=audio_onsets_payload,
        )
        notes.append(
            "fused contact_windows.json; cues used: "
            f"wrist={cue_summary['wrist']}, ball_inflections={cue_summary['ball_inflections']}, "
            f"audio={cue_summary['audio']}, reviewed_contacts={cue_summary['reviewed_contacts']}, "
            "missing="
            f"{','.join(cue_summary['missing']) if cue_summary['missing'] else 'none'}"
        )

        artifacts = ["wrist_velocity_peaks.json", "ball_inflections.json", "contact_windows.json"]
        plan: Mapping[str, Any] | None = None
        if tracks_path.is_file():
            tracks = validate_artifact_file("tracks", tracks_path)
            ball_aware_events, ball_track_arc_solved, ba_notes = self._resolve_ball_aware_inputs()
            plan = build_frame_compute_plan(
                tracks,
                ball_track=_read_json(ball_track_path) if ball_track_path.is_file() else None,
                contact_windows=contact_windows,
                expected_players=self.options.max_players,
                ball_aware_events=ball_aware_events,
                ball_track_arc_solved=ball_track_arc_solved,
                **self._mesh_coverage_kwargs(),
            )
            plan, demo_mesh_count = self._ensure_auto_court_preview_demo_mesh(plan)
            write_frame_compute_plan(self.clip_dir / "frame_compute_plan.json", plan)
            artifacts.append("frame_compute_plan.json")
            deep_frames = plan.get("summary", {}).get("world_mesh_frame_count") or plan.get("summary", {}).get("deep_mesh_frame_count")
            notes.append(
                "wrote frame_compute_plan.json (deep_mesh_windows authoritative: MESH only at contact windows"
                f"{f', {deep_frames} frames' if deep_frames is not None else ''}, JOINTS everywhere else)"
            )
            notes.extend(ba_notes)
            if demo_mesh_count:
                notes.append(
                    f"added auto-court preview demo mesh window covering {demo_mesh_count} tracked frame(s)"
                )
        else:
            notes.append("frame_compute_plan.json not derived (no tracks.json)")

        events = contact_windows.get("events", []) if isinstance(contact_windows, Mapping) else []
        return StageOutcome(
            stage="events",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=artifacts,
            metrics={
                "contact_event_count": len(events),
                "contact_cues": cue_summary,
                **self._mesh_coverage_plan_metrics(plan),
            },
        )

    def _resolve_ball_aware_inputs(self) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, list[str]]:
        """Resolve events_selected.json / ball_track_arc_solved.json for
        mesh_coverage_mode="ball_aware" tier-1 scheduling.

        Explicit --events-selected/--ball-track-arc-solved win; otherwise
        auto-discovered as events_selected.json/ball_track_arc_solved.json
        in the clip dir (the ball-arc-solver's own convention, e.g.
        scripts/racketsport/solve_ball_arcs.py --out <clip_dir>). Both are
        the solver's advisory (render_only, not_for_detection_metrics)
        outputs -- read as plain JSON here, never schema-validated (see
        threed.racketsport.frame_rating docstring). A no-op, returning
        (None, None, []), unless mesh_coverage_mode == "ball_aware".
        """
        if self.options.mesh_coverage_mode != "ball_aware":
            return None, None, []

        notes: list[str] = []
        events_selected_path = self.options.events_selected or (self.clip_dir / "events_selected.json")
        ball_track_arc_solved_path = self.options.ball_track_arc_solved or (self.clip_dir / "ball_track_arc_solved.json")

        ball_aware_events = None
        if events_selected_path.is_file():
            ball_aware_events = _read_json(events_selected_path)
            notes.append(f"ball_aware mesh scheduling: loaded events_selected.json from {events_selected_path}")
        else:
            notes.append(
                "ball_aware mesh scheduling: no events_selected.json found "
                f"(looked at {events_selected_path}); ball_aware_contact trigger unavailable this run"
            )

        ball_track_arc_solved = None
        if ball_track_arc_solved_path.is_file():
            ball_track_arc_solved = _read_json(ball_track_arc_solved_path)
            notes.append(
                f"ball_aware mesh scheduling: loaded ball_track_arc_solved.json from {ball_track_arc_solved_path}"
            )
        else:
            notes.append(
                "ball_aware mesh scheduling: no ball_track_arc_solved.json found "
                f"(looked at {ball_track_arc_solved_path}); ball_proximity trigger unavailable this run"
            )

        return ball_aware_events, ball_track_arc_solved, notes

    def _mesh_coverage_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "mesh_coverage_mode": self.options.mesh_coverage_mode,
            "target_mesh_frame_budget": self.options.target_mesh_frame_budget,
            "ball_proximity_m": self.options.ball_proximity_m,
            "high_confidence_swing_floor": self.options.high_confidence_swing_floor,
        }
        if self.options.mesh_byte_budget_mib is not None:
            kwargs["mesh_byte_budget_mib"] = self.options.mesh_byte_budget_mib
        return kwargs

    @staticmethod
    def _mesh_coverage_plan_metrics(plan: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(plan, Mapping):
            return {}
        policy = plan.get("mesh_coverage_policy")
        if not isinstance(policy, Mapping):
            return {}
        metrics: dict[str, Any] = {"mesh_coverage_mode": policy.get("mode")}
        trigger_counts = policy.get("ball_aware_trigger_source_counts")
        if trigger_counts is not None:
            metrics["ball_aware_trigger_source_counts"] = trigger_counts
        if policy.get("mesh_budget_policy") == "byte_budget":
            metrics["mesh_budget_policy"] = "byte_budget"
            metrics["mesh_byte_budget_mib"] = policy.get("mesh_byte_budget_mib")
            metrics["selected_estimated_mesh_bytes"] = policy.get("selected_estimated_mesh_bytes")
        return metrics

    def _ensure_auto_court_preview_demo_mesh(self, plan: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        if not (self.clip_dir / "auto_court_corners_preview.json").is_file():
            return dict(plan), 0
        payload = copy.deepcopy(dict(plan))
        if payload.get("deep_mesh_windows"):
            return payload, 0
        frames = payload.get("frames")
        if not isinstance(frames, list):
            return payload, 0
        candidates = [
            frame
            for frame in frames
            if isinstance(frame, dict) and _frame_active_player_ids(frame)
        ]
        selected = _evenly_sample_frames(candidates, max_count=AUTO_COURT_PREVIEW_DEMO_MESH_MAX_FRAMES)
        if not selected:
            return payload, 0

        selected_indexes = {int(frame["frame_idx"]) for frame in selected}
        for frame in frames:
            if not isinstance(frame, dict) or int(frame.get("frame_idx", -1)) not in selected_indexes:
                continue
            active_ids = _frame_active_player_ids(frame)
            reasons = [str(reason) for reason in frame.get("reasons", [])]
            if "auto_court_preview_demo_mesh" not in reasons:
                reasons.append("auto_court_preview_demo_mesh")
            frame["reasons"] = reasons
            frame["score"] = max(float(frame.get("score", 0.0)), 0.55)
            frame["recommended_tier"] = "deep_mesh"
            frame["target_representation"] = "world_mesh"
            frame["player_targets"] = [
                {
                    "player_id": player_id,
                    "score": frame["score"],
                    "target_representation": "world_mesh",
                    "reasons": reasons,
                }
                for player_id in active_ids
            ]

        windows = _deep_mesh_windows_from_plan_frames(frames, fps=float(payload.get("fps", 30.0)))
        payload["deep_mesh_windows"] = windows
        payload["summary"] = _frame_compute_plan_summary(frames, deep_mesh_windows=windows)
        return payload, sum(int(window["frame_count"]) for window in windows)

    # ------------------------------------------------------------------
    # stage 7b: render-only ball fill (ball_track_physics_filled.json)
    # ------------------------------------------------------------------

    def _stage_ball_fill(self) -> StageOutcome:
        ball_track_path = self.clip_dir / "ball_track.json"
        target = self.clip_dir / "ball_track_physics_filled.json"
        if not ball_track_path.is_file():
            return StageOutcome(
                stage="ball_fill",
                status="blocked",
                wall_seconds=0.0,
                notes=["requires ball_track.json; no render-only ball physics fill produced"],
            )
        if target.is_file() and not self.options.force:
            return StageOutcome(
                stage="ball_fill",
                status="skipped",
                wall_seconds=0.0,
                notes=["reusing existing ball_track_physics_filled.json"],
                artifacts=["ball_track_physics_filled.json"],
            )

        ball_payload = _read_json(ball_track_path)
        calibration = _read_optional_json(self.clip_dir / "court_calibration.json")
        reviewed_bounces = _read_optional_json(self.clip_dir / "reviewed_ball_bounces.json")
        ball_inflections = _read_optional_json(self.clip_dir / "ball_inflections.json")
        wrist_velocity_peaks = _read_optional_json(self.clip_dir / "wrist_velocity_peaks.json")
        config = PhysicsFillConfig()
        notes = [
            "built render-only ball_track_physics_filled.json; output is not for BALL detection metrics, gates, training, or promotion",
        ]
        physics3d_reconstruction = None
        physics3d_summary: dict[str, Any] | None = None
        if calibration is not None:
            physics3d_reconstruction = reconstruct_bounce_arcs_from_image_track(
                ball_payload,
                calibration,
                max_reprojection_rmse_px=12.0,
                max_fit_samples=13,
            )
            physics3d_summary = physics3d_reconstruction.summary()
            if physics3d_reconstruction.status == "ran":
                notes.append(
                    "applied ball_physics3d calibrated two-arc z reconstruction "
                    f"(rmse_px={physics3d_reconstruction.reprojection_rmse_px})"
                )
            else:
                notes.append(f"ball_physics3d z reconstruction not applied: status={physics3d_reconstruction.status}")
                physics3d_reconstruction = None
        else:
            notes.append("ball_physics3d z reconstruction skipped: court_calibration.json missing")

        filled = fill_ball_track_physics(
            ball_payload,
            calibration=calibration,
            config=config,
            evidence_path=str(ball_track_path),
            reviewed_bounces=reviewed_bounces,
            ball_inflections=ball_inflections,
            wrist_velocity_peaks=wrist_velocity_peaks,
            physics3d_reconstruction=physics3d_reconstruction,
            frame_times=_existing_optional_path(self.clip_dir / "frame_times.json"),
        )
        filled["physics_fill"]["physics3d_reconstruction"] = physics3d_summary
        filled["physics_fill"]["reviewed_bounces_input"] = str(self.clip_dir / "reviewed_ball_bounces.json") if reviewed_bounces is not None else None
        filled["physics_fill"]["ball_inflections_input"] = str(self.clip_dir / "ball_inflections.json") if ball_inflections is not None else None
        filled["physics_fill"]["wrist_velocity_peaks_input"] = str(self.clip_dir / "wrist_velocity_peaks.json") if wrist_velocity_peaks is not None else None
        _write_json(target, filled)

        coverage = filled.get("physics_fill", {}).get("coverage", {})
        return StageOutcome(
            stage="ball_fill",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=["ball_track_physics_filled.json"],
            metrics={
                "input_frame_count": coverage.get("input_frame_count"),
                "output_world_xyz_count": coverage.get("output_world_xyz_count"),
                "filled_frame_count": coverage.get("filled_frame_count"),
                "physics3d_reconstructed_frame_count": coverage.get("physics3d_reconstructed_frame_count"),
                "xy_interpolated_frame_count": coverage.get("xy_interpolated_frame_count"),
                "reviewed_bounce_boundary_count": sum(
                    1
                    for boundary in filled.get("physics_fill", {}).get("bounce_boundaries", [])
                    if isinstance(boundary, Mapping) and boundary.get("source") == "human_reviewed"
                ),
            },
        )

    def _attempt_audio_onsets(self, fps: float) -> tuple[Any, str]:
        target = self.clip_dir / "audio_onsets_v2.json"
        try:
            from threed.racketsport.audio_onsets_v2 import build_audio_onsets_v2_from_video, write_audio_onsets_v2

            payload = build_audio_onsets_v2_from_video(
                self.clip_dir / f"source{self.options.video.suffix.lower()}",
                clip=self.options.clip,
                frame_rate=fps,
            )
            write_audio_onsets_v2(target, payload)
            return payload, f"extracted audio_onsets_v2.json ({len(payload.get('onsets', []))} onsets; refines contact timing only)"
        except Exception as exc:  # noqa: BLE001
            return [], f"no usable audio track / audio onset extraction unavailable ({type(exc).__name__}: {exc}); contact windows use wrist+ball cues only"

    # ------------------------------------------------------------------
    # stage 8: body
    # ------------------------------------------------------------------

    def _body_stage_reuse_skip(self) -> StageOutcome | None:
        """The no-force-reuse checks `_stage_body` short-circuits on, factored
        out so the overlap scheduler can decide -- before spending a thread --
        whether this run takes the plain serial path (spec design 1: "Reuse
        semantics unchanged: if BODY artifacts are valid for no-force reuse,
        take the serial path (no thread)"). Returns None when BODY must
        actually run (dispatch or local)."""

        opts = self.options
        target = self.clip_dir / "smpl_motion.json"

        if target.is_file() and not opts.force and _valid_artifact("smpl_motion", target):
            return StageOutcome(stage="body", status="skipped", wall_seconds=0.0, notes=["reusing existing valid smpl_motion.json"], artifacts=["smpl_motion.json"])

        skeleton_path = self.clip_dir / "skeleton3d.json"
        body_gate_path = self.clip_dir / "body_full_clip_gate.json"
        if (
            not opts.force
            and not target.is_file()
            and _is_real_sam3d_skeleton(skeleton_path)
            and body_gate_path.is_file()
        ):
            return StageOutcome(
                stage="body",
                status="skipped",
                wall_seconds=0.0,
                notes=[
                    "reusing completed BODY evidence without smpl_motion.json: "
                    f"skeleton3d.json + body_full_clip_gate.json ({skeleton_path} + {body_gate_path}; "
                    "remote BODY monoliths may be not fetched by speed default)"
                ],
                artifacts=["skeleton3d.json", "body_full_clip_gate.json"],
            )
        return None

    def _stage_body(self) -> StageOutcome:
        reuse_outcome = self._body_stage_reuse_skip()
        if reuse_outcome is not None:
            return reuse_outcome

        opts = self.options
        if opts.no_gpu:
            return StageOutcome(
                stage="body",
                status="degraded",
                wall_seconds=0.0,
                notes=[
                    "--no-gpu requested: SAM-3D BODY skipped by design; no new offline skeleton3d.json or "
                    "smpl_motion.json was produced in this run."
                ],
            )

        if not (self.clip_dir / "tracks.json").is_file():
            return StageOutcome(
                stage="body",
                status="blocked",
                wall_seconds=0.0,
                notes=["requires tracks.json (tracking stage did not produce one); degrading to TRK/court-only bundle"],
            )

        if opts.body_remote:
            return self._dispatch_body_remote()

        return self._run_body_local()

    def _run_body_local(self) -> StageOutcome:
        opts = self.options
        manifest_path, manifest_notes = self._runtime_manifest_for_local_host()
        if opts.remote_config.fast_sam_python:
            os.environ.setdefault("FAST_SAM_PYTHON", opts.remote_config.fast_sam_python)
        try:
            result = orchestrator.run_pipeline(
                clip=opts.clip,
                inputs_dir=self.clip_dir,
                run_dir=self.clip_dir,
                stage="body",
                sport=opts.sport,  # type: ignore[arg-type]
                device=opts.device,
                max_frames=opts.max_frames,
                tracking_mode="precomputed_tracks",
                manifest_path=manifest_path,
                max_players=opts.max_players,
                runners={
                    "body": orchestrator.BodyStageRunner(
                        manifest_path=manifest_path,
                        fast_sam_repo=opts.remote_config.fast_sam_root,
                        detector_name=opts.remote_config.body_detector_name,
                        fov_name=opts.remote_config.body_fov_name,
                        tier2_body_joints_all_tracked=True,
                        mesh_vertex_serialization_policy="tier1_only"
                        if opts.remote_config.sam3d_skip_tier2_mesh_vertices
                        else "all",
                        sam3d_body_input_size_px=opts.remote_config.sam3d_body_input_size_px,
                        sam3d_crop_bucket_sizes=opts.remote_config.sam3d_crop_bucket_sizes,
                        sam3d_torch_compile=opts.remote_config.sam3d_torch_compile,
                        sam3d_compile_warmup_buckets=opts.remote_config.sam3d_compile_warmup_buckets,
                        sam3d_wrist_bone_lock=opts.remote_config.sam3d_wrist_bone_lock,
                        body_temporal_smoothing=opts.remote_config.body_temporal_smoothing,
                        body_foot_lock=opts.remote_config.body_foot_lock,
                        body_foot_pin=opts.remote_config.body_foot_pin,
                        body_contact_splice=opts.remote_config.body_contact_splice,
                        body_world_joint_visual_smoothing=opts.remote_config.body_world_joint_visual_smoothing,
                    )
                },
                # Calibration/tracking already ran earlier in this same process_video
                # run -- reuse those artifacts instead of re-deriving them.
                reuse_existing_stage_artifacts=True,
            )
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(stage="body", status="degraded", wall_seconds=0.0, notes=[f"local BODY stage unavailable ({type(exc).__name__}: {exc}); degraded to skeleton-only"])
        if not _spine_stage_succeeded(result, stage="body"):
            return StageOutcome(stage="body", status="degraded", wall_seconds=0.0, notes=[f"local BODY stage failed: {_spine_failure_detail(result)}; degraded to skeleton-only"])

        badge = build_trust_band(
            stage="BODY",
            gate_id="full_clip_body_gate",
            gate_status="structural_only",
            badge="preview",
            reason="Fast SAM-3D-Body ran locally with SAM-3D body-mode joints as the offline skeleton source; world-MPJPE accuracy gate not yet measured for this run",
            evidence_path=str(self.clip_dir / "body_full_clip_gate.json"),
        )
        self.trust_bands["body"] = badge
        return StageOutcome(
            stage="body",
            status="ran",
            wall_seconds=0.0,
            notes=[
                "ran BODY (Fast SAM-3D-Body mesh at contact windows, joints elsewhere) locally "
                "with detector_name='' and fov_name='' to match the VM-proven no-MoGe configuration "
                f"(fast_sam_repo={opts.remote_config.fast_sam_root}); SAM3D body-mode joints are the "
                "offline skeleton source for all safe tracked person-frames, with tier-2 mesh vertices "
                "omitted from serialization",
                *manifest_notes,
            ],
            artifacts=[
                "smpl_motion.json",
                "skeleton3d.json",
                "body_mesh.json",
                "body_compute_execution.json",
                "sam3d_tier2_config.json",
            ],
            trust_badge=badge["badge"],
        )

    def _ensure_remote_calibration_seed(self) -> str:
        """Make sure a ``capture_sidecar.json`` exists in clip_dir before remote
        BODY dispatch (Task #46).

        The remote A100 runs the *committed* orchestrator, whose dependency walk
        always re-derives calibration via ``ManualCalibrationRunner`` -- which
        hard-requires ``capture_sidecar.json`` (it has no ExternalCalibrationRunner
        or artifact-reuse support). The --court-corners/--capture-sidecar variants
        already leave one in clip_dir, but the --court-calibration (metric15)
        variant deliberately never writes one, so remote dispatch used to fail at
        remote-side calibration for exactly that variant. Here we derive the four
        corner taps from the trusted external calibration's own image_pts/world_pts
        (the 15 reviewed points include the 4 court corners, in the video's native
        pixel space -- crucially at the calibration's own declared image_size, so
        the remote's bbox scaling stays 1:1 with the local tracks' pixel space) and
        write a clearly-labeled seed sidecar for the remote's own dependency walk.
        The local world bundle keeps the trusted metric15 calibration untouched.
        """

        sidecar_path = self.clip_dir / "capture_sidecar.json"
        if sidecar_path.is_file():
            return "remote calibration seed: existing capture_sidecar.json will sync up"

        calibration_path = self.clip_dir / "court_calibration.json"
        if not calibration_path.is_file():
            return (
                "remote calibration seed unavailable (no capture_sidecar.json and no court_calibration.json "
                "to derive corners from); the remote dependency walk will fail at its calibration stage"
            )
        try:
            _, _, fps = _video_probe(self.options.video)
            sidecar = _remote_seed_capture_sidecar_from_calibration(_read_json(calibration_path), fps=fps)
            _write_json(sidecar_path, sidecar)
        except Exception as exc:  # noqa: BLE001 - dispatch proceeds; remote failure note stays honest
            return (
                f"remote calibration seed derivation failed ({type(exc).__name__}: {exc}); the remote "
                "dependency walk will fail at its calibration stage unless a capture_sidecar.json exists"
            )
        return (
            "remote calibration seed: derived capture_sidecar.json corner taps from the external "
            "court_calibration.json's own image_pts/world_pts (remote's committed orchestrator re-derives "
            "calibration from manual taps; the local world keeps the trusted external calibration)"
        )

    def _dispatch_body_remote(self) -> StageOutcome:
        opts = self.options
        seed_note = self._ensure_remote_calibration_seed()
        try:
            result = dispatch_body_stage(
                clip=opts.clip,
                clip_dir=self.clip_dir,
                video_path=self.clip_dir / f"source{opts.video.suffix.lower()}",
                # The "frames" stage materializes body_frames/ into this exact
                # directory; pass it explicitly so the remote A100 sees the
                # same SAM-3D body inputs as the local wrapper.
                body_frames_dir=self.clip_dir / "body_frames",
                # B12: thread the pipeline's RESOLVED --camera-motion path
                # (already .expanduser().resolve()'d in build_options_from_args)
                # into remote dispatch so it is not left to placement alone.
                # This is intentionally opts.camera_motion_path *raw* (not
                # `_placement_camera_motion_path()`'s auto-discovery result):
                # remote_body_dispatch._rsync_up already auto-syncs the
                # canonical clip_dir/camera_motion.json via BODY_INPUT_ARTIFACTS
                # whenever camera_motion_path is None, so the clip-dir-only
                # case (no explicit flag) must stay None here to keep that
                # existing sync path unchanged -- passing the resolved-equal
                # path through as "explicit" would make _rsync_up's own
                # explicit-vs-canonical dedupe skip syncing it entirely.
                camera_motion_path=opts.camera_motion_path,
                config=opts.remote_config,
                max_frames=opts.max_frames,
                max_players=opts.max_players,
            )
        except RemoteBodyDispatchError as exc:
            return StageOutcome(
                stage="body",
                status="degraded",
                wall_seconds=0.0,
                notes=[
                    seed_note,
                    f"remote BODY dispatch to {opts.remote_config.host} did not complete: {exc}",
                    "remote SAM-3D BODY did not complete; no fallback pose skeleton was generated",
                ],
            )

        smpl_synced = (self.clip_dir / "smpl_motion.json").is_file()
        skeleton_synced = _is_real_sam3d_skeleton(self.clip_dir / "skeleton3d.json")
        if not smpl_synced and not skeleton_synced:
            return StageOutcome(
                stage="body",
                status="degraded",
                wall_seconds=0.0,
                notes=["remote BODY dispatch reported success but neither smpl_motion.json nor skeleton3d.json synced back; degraded to skeleton-only", *result.notes],
            )

        # A remote run can legitimately come back skeleton-level only: SAM-3D
        # body-mode joints synced but no smpl_motion.json or mesh vertices did.
        # That is retained as low-confidence BODY output and never papered over
        # with a fabricated mesh or legacy pose fallback.
        skeleton_level_only = skeleton_synced and not smpl_synced
        if skeleton_level_only:
            reason = (
                f"remote A100 BODY run ({result.remote_run_dir}) came back skeleton-level only without "
                "smpl_motion.json; this is treated as low-confidence SAM-3D BODY output, not a mesh claim."
            )
        else:
            reason = (
                f"Fast SAM-3D-Body ran on the remote A100 ({result.remote_run_dir}) with body-mode joints "
                "as the offline skeleton source for all safe tracked person-frames and mesh vertices only "
                "for tier-1 ball-aware windows; world-MPJPE accuracy gate not yet measured for this run."
            )
        badge = build_trust_band(
            stage="BODY",
            gate_id="full_clip_body_gate",
            gate_status="structural_only",
            badge="low_confidence" if skeleton_level_only else "preview",
            reason=reason,
            evidence_path=str(self.clip_dir / "body_full_clip_gate.json"),
        )
        self.trust_bands["body"] = badge
        notes = [seed_note, *result.notes, f"synced back: {', '.join(result.synced_outputs)}"]
        if skeleton_level_only:
            notes.append(
                "remote BODY result is skeleton-level only with no smpl_motion.json; retained as a "
                "low-confidence SAM-3D BODY skeleton and not claimed as mesh output"
            )
        return StageOutcome(
            stage="body",
            status="ran",
            wall_seconds=result.wall_seconds,
            notes=notes,
            artifacts=list(result.synced_outputs),
            trust_badge=badge["badge"],
        )

    # ------------------------------------------------------------------
    # stage 8b: render-honest BODY grounding refinement
    # ------------------------------------------------------------------

    def _stage_grounding_refine(self) -> StageOutcome:
        if not self.options.grounding_refine:
            return StageOutcome(
                stage="grounding_refine",
                status="skipped",
                wall_seconds=0.0,
                notes=["--no-grounding-refine set: keeping BODY/skeleton artifacts untouched"],
                metrics={"policy_note": GROUNDING_REFINE_POLICY_NOTE},
            )

        required = {
            "foot_contact_phases.json": self.clip_dir / "foot_contact_phases.json",
            "court_calibration.json": self.clip_dir / "court_calibration.json",
            "tracks.json": self.clip_dir / "tracks.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            return StageOutcome(
                stage="grounding_refine",
                status="skipped",
                wall_seconds=0.0,
                notes=[
                    "grounding_refine skipped: requires foot_contact_phases.json + court_calibration.json + tracks.json",
                    f"missing: {', '.join(missing)}",
                    GROUNDING_REFINE_POLICY_NOTE,
                ],
                metrics={"status": "skipped_missing_inputs", "missing_inputs": missing, "policy_note": GROUNDING_REFINE_POLICY_NOTE},
            )

        phases = _read_json(required["foot_contact_phases.json"])
        phase_count = _foot_contact_phase_count(phases)
        if phase_count == 0:
            report = _grounding_refine_stage_report(
                status="skipped_no_contact_phases",
                phase_count=0,
                reports={},
                summary={"phase_count": 0, "correction_magnitude_m": {"count": 0, "max": 0.0, "mean": 0.0, "rms": 0.0, "warn_count": 0}},
            )
            _write_json(self.clip_dir / "body_grounding_refinement.json", report)
            return StageOutcome(
                stage="grounding_refine",
                status="skipped",
                wall_seconds=0.0,
                notes=[
                    "grounding_refine skipped_no_contact_phases: zero contact phases detected, so BODY/skeleton artifacts passed through untouched",
                    GROUNDING_REFINE_POLICY_NOTE,
                ],
                artifacts=["body_grounding_refinement.json"],
                metrics={
                    "status": "skipped_no_contact_phases",
                    "phase_count": 0,
                    "correction_magnitude_m": report["summary"]["correction_magnitude_m"],
                    "policy_note": GROUNDING_REFINE_POLICY_NOTE,
                },
            )

        targets = [
            ("smpl_motion", self.clip_dir / "smpl_motion.json", self.clip_dir / "smpl_motion_pre_grounding_refine.json"),
            ("skeleton3d", self.clip_dir / "skeleton3d.json", self.clip_dir / "skeleton3d_pre_grounding_refine.json"),
        ]
        present_targets = [(artifact_type, path, pre_path) for artifact_type, path, pre_path in targets if path.is_file()]
        if not present_targets:
            return StageOutcome(
                stage="grounding_refine",
                status="skipped",
                wall_seconds=0.0,
                notes=["grounding_refine skipped: no smpl_motion.json or skeleton3d.json available to refine", GROUNDING_REFINE_POLICY_NOTE],
                metrics={"status": "skipped_missing_body_artifacts", "phase_count": phase_count, "policy_note": GROUNDING_REFINE_POLICY_NOTE},
            )

        tracks = _read_json(required["tracks.json"])
        r3_grounded = _has_r3_grounding_provenance(self.clip_dir)
        config = GroundingRefineConfig(xy_translation_enabled=not r3_grounded)
        reports: dict[str, Any] = {}
        transl_world_backfilled_frames: dict[str, int] = {}
        originals: dict[Path, Any] = {}
        artifacts = ["body_grounding_refinement.json"]
        for artifact_type, path, pre_path in present_targets:
            original = _read_json(path)
            originals[path] = original
            if self.options.force or not pre_path.is_file():
                _write_json(pre_path, original)
            payload_for_refine = copy.deepcopy(original)
            backfilled = _populate_missing_transl_world_from_tracks(payload_for_refine, tracks)
            if backfilled:
                transl_world_backfilled_frames[path.name] = backfilled
            refined, report = refine_body_grounding(
                payload_for_refine,
                foot_contact_phases=phases,
                tracks=tracks,
                config=config,
            )
            reports[path.name] = report
            artifacts.append(pre_path.name)
            artifacts.append(path.name)
            if not report.get("summary", {}).get("kill_recommended"):
                _write_json(path, refined)

        summary = _aggregate_grounding_refine_reports(reports, phase_count=phase_count)
        summary["transl_world_backfilled_frames"] = dict(transl_world_backfilled_frames)
        summary["xy_translation_enabled"] = bool(config.xy_translation_enabled)
        summary["grounding_anchor_source"] = "placement_track_world_xy" if r3_grounded else "legacy_or_unknown"
        killed = bool(summary.get("kill_recommended"))
        if killed:
            for path, original in originals.items():
                _write_json(path, original)
            status: StageStatus = "degraded"
            notes = [
                "grounding_refine sanity gate failed: refiner reported worsened residuals, so original BODY/skeleton artifacts were restored",
                GROUNDING_REFINE_POLICY_NOTE,
            ]
        else:
            status = "ran"
            notes = [
                f"ran grounding_refine on {', '.join(report_name for report_name in reports)} after BODY/skeleton sync and before world assembly",
                GROUNDING_REFINE_POLICY_NOTE,
            ]
            if r3_grounded:
                notes.append("R3 placement grounding provenance present; grounding_refine ran in z-only mode with XY translation disabled")

        stage_report = _grounding_refine_stage_report(
            status="sanity_gate_failed" if killed else "ran",
            phase_count=phase_count,
            reports=reports,
            summary=summary,
        )
        _write_json(self.clip_dir / "body_grounding_refinement.json", stage_report)

        return StageOutcome(
            stage="grounding_refine",
            status=status,
            wall_seconds=0.0,
            notes=notes,
            artifacts=sorted(set(artifacts), key=artifacts.index),
            metrics={
                "status": stage_report["status"],
                "phase_count": phase_count,
                "correction_magnitude_m": summary.get("correction_magnitude_m", {}),
                "warn_count": summary.get("correction_magnitude_m", {}).get("warn_count", 0),
                "transl_world_backfilled_frames": transl_world_backfilled_frames,
                "xy_translation_enabled": bool(config.xy_translation_enabled),
                "grounding_anchor_source": summary["grounding_anchor_source"],
                "kill_recommended": killed,
                "policy_note": GROUNDING_REFINE_POLICY_NOTE,
            },
        )

    # ------------------------------------------------------------------
    # stage 9: paddle pose (racket_pose_estimate.json)
    # ------------------------------------------------------------------

    def _stage_paddle_pose(self) -> StageOutcome:
        out = self.clip_dir / PADDLE_POSE_ARTIFACT_NAME
        if not self.options.paddle_pose:
            return StageOutcome(
                stage="paddle_pose",
                status="skipped",
                wall_seconds=0.0,
                notes=["--no-paddle-pose set: fused paddle estimator disabled before world assembly"],
                metrics=_paddle_pose_blocked_stage_metrics("no_paddle_pose_flag", status="skipped"),
            )

        if out.is_file() and not self.options.force:
            try:
                payload = _read_optional_json(out)
            except Exception:
                payload = None
            if _is_valid_paddle_pose_payload(payload):
                self._record_paddle_pose_trust_band(payload)
                return StageOutcome(
                    stage="paddle_pose",
                    status="reused",
                    wall_seconds=0.0,
                    notes=[f"reused existing {PADDLE_POSE_ARTIFACT_NAME} before world assembly"],
                    artifacts=[PADDLE_POSE_ARTIFACT_NAME],
                    trust_badge="low_confidence",
                    metrics={"paddle_pose": _paddle_pose_stage_metrics(payload)},
                )
            out.unlink(missing_ok=True)

        skeleton_path = self.clip_dir / "skeleton3d.json"
        if not skeleton_path.is_file():
            return StageOutcome(
                stage="paddle_pose",
                status="blocked",
                wall_seconds=0.0,
                notes=[
                    "paddle_pose fail-closed: missing skeleton3d.json from BODY/SAM-3D, so no racket_pose_estimate.json was emitted",
                ],
                metrics=_paddle_pose_blocked_stage_metrics("missing_sam3d_skeleton3d"),
            )

        calibration_path = self.clip_dir / "court_calibration.json"
        detector_boxes_path = _first_existing_path(
            self.clip_dir,
            (
                "paddle_detector_boxes.json",
                "racket_detector_boxes.json",
                "detector_boxes.json",
                "paddle_boxes.json",
            ),
        )
        membership_path = _first_existing_path(
            self.clip_dir,
            (
                "court_membership.json",
                "player_membership.json",
                "membership.json",
            ),
        )
        try:
            detector_boxes = _read_optional_json(detector_boxes_path) if detector_boxes_path is not None else None
            calibration = _read_optional_json(calibration_path) if calibration_path.is_file() else None
            membership = _read_optional_json(membership_path) if membership_path is not None else None
            use_detector_boxes = isinstance(detector_boxes, Mapping) and isinstance(calibration, Mapping)
            payload = build_paddle_pose_fused_from_file(
                skeleton_path,
                clip_id=self.options.clip,
                ball_track=_read_optional_json(self.clip_dir / "ball_track.json"),
                contact_windows=_read_optional_json(self.clip_dir / "contact_windows.json"),
                physics_estimate=_read_optional_json(self.clip_dir / "racket_physics_estimate.json"),
                detector_boxes=detector_boxes if isinstance(detector_boxes, Mapping) else None,
                calibration=calibration if isinstance(calibration, Mapping) else None,
                membership=membership if isinstance(membership, Mapping) else None,
                use_reflection=True,
                use_detector_boxes=use_detector_boxes,
                use_detector_box_handedness=use_detector_boxes,
            )
        except Exception as exc:  # noqa: BLE001 - stage is fail-closed by contract.
            out.unlink(missing_ok=True)
            return StageOutcome(
                stage="paddle_pose",
                status="blocked",
                wall_seconds=0.0,
                notes=[
                    "paddle_pose fail-closed: fused estimator raised while reading current run evidence; no artifact was emitted",
                    f"{type(exc).__name__}: {exc}",
                ],
                metrics=_paddle_pose_blocked_stage_metrics("fused_estimator_exception"),
            )

        metrics = _paddle_pose_stage_metrics(payload)
        if payload.get("status") != "preview" or metrics["coverage"]["estimate_frame_count"] <= 0:
            out.unlink(missing_ok=True)
            reason = _paddle_pose_blocked_reason(payload)
            return StageOutcome(
                stage="paddle_pose",
                status="blocked",
                wall_seconds=0.0,
                notes=[
                    f"paddle_pose fail-closed: {reason}; no racket_pose_estimate.json was emitted",
                ],
                metrics={"paddle_pose": {**metrics, "status": "blocked", "reason": reason}},
            )

        write_paddle_pose_fused(out, payload)
        self._record_paddle_pose_trust_band(payload)
        notes = [
            "built render-only fused wrist+palm+grip paddle estimate before world assembly",
            "RKT remains unverified: racket_pose_estimate.json is estimated preview evidence, not a promotion artifact",
        ]
        if detector_boxes_path is None:
            notes.append("no detector-box sidecar found; estimator used SAM-3D wrist/palm/grip evidence only")
        else:
            notes.append(f"detector-box evidence sidecar: {detector_boxes_path.name}")
        return StageOutcome(
            stage="paddle_pose",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=[PADDLE_POSE_ARTIFACT_NAME],
            trust_badge="low_confidence",
            metrics={"paddle_pose": metrics},
        )

    def _record_paddle_pose_trust_band(self, payload: Mapping[str, Any]) -> None:
        trust_band = payload.get("trust_band")
        if isinstance(trust_band, Mapping):
            self.trust_bands["racket_pose_estimate"] = dict(trust_band)

    # ------------------------------------------------------------------
    # stage 10: world (virtual_world.json + trust_bands.json)
    # ------------------------------------------------------------------

    def _stage_world(self) -> StageOutcome:
        court_path = self.clip_dir / "court_calibration.json"
        if not court_path.is_file():
            return StageOutcome(stage="world", status="blocked", wall_seconds=0.0, notes=["requires court_calibration.json"])

        tracks = _read_optional_json(self.clip_dir / "tracks.json")
        smpl_motion = _read_optional_json(self.clip_dir / "smpl_motion.json")
        skeleton3d = _read_optional_json(self.clip_dir / "skeleton3d.json")
        ball_track = _read_optional_json(self.clip_dir / "ball_track.json")
        physics_footlock_path = self.clip_dir / "physics_footlock.json"
        ball_physics_path = self.clip_dir / "ball_track_physics_filled.json"
        ball_arc_solved_path = self.clip_dir / "ball_track_arc_solved.json"
        racket_estimate_path = self.clip_dir / "racket_pose_estimate.json"
        racket_pose_path = self.clip_dir / "racket_pose.json"

        payload = build_virtual_world_state(
            court_calibration=_read_json(court_path),
            tracks=tracks,
            smpl_motion=smpl_motion,
            skeleton3d=skeleton3d,
            ball_track=ball_track,
            racket_pose=_read_optional_json(racket_pose_path),
            trust_bands=self.trust_bands,
            physics_footlock=_read_optional_json(physics_footlock_path),
            ball_track_physics_filled=_read_optional_json(ball_physics_path),
            ball_track_arc_solved=_read_optional_json(ball_arc_solved_path),
            racket_pose_estimate=_read_optional_json(racket_estimate_path) if self.options.paddle_pose else None,
            placement_calibration_path=court_path,
            artifact_paths={
                "physics_footlock": physics_footlock_path if physics_footlock_path.is_file() else None,
                "ball_track_physics_filled": ball_physics_path if ball_physics_path.is_file() else None,
                "ball_track_arc_solved": ball_arc_solved_path if ball_arc_solved_path.is_file() else None,
                "racket_pose_estimate": racket_estimate_path if self.options.paddle_pose and racket_estimate_path.is_file() else None,
            },
        )
        out = self.clip_dir / "virtual_world.json"
        write_virtual_world(out, payload)
        _write_json(self.clip_dir / "trust_bands.json", self.trust_bands)

        return StageOutcome(
            stage="world",
            status="ran",
            wall_seconds=0.0,
            notes=["assembled virtual_world.json with per-entity trust bands (never invented; read from real upstream state)"],
            artifacts=["virtual_world.json", "trust_bands.json"],
            metrics=dict(payload.get("summary", {})),
        )

    # ------------------------------------------------------------------
    # stage 11: confidence gate (confidence_gated_world.json)
    # ------------------------------------------------------------------

    def _stage_confidence_gate(self) -> StageOutcome:
        if not self.options.confidence_gate:
            return StageOutcome(
                stage="confidence_gate",
                status="skipped",
                wall_seconds=0.0,
                notes=["--no-confidence-gate set: keeping raw virtual_world.json as the viewer world"],
            )

        world_path = self.clip_dir / "virtual_world.json"
        if not world_path.is_file():
            return StageOutcome(
                stage="confidence_gate",
                status="blocked",
                wall_seconds=0.0,
                notes=["requires virtual_world.json"],
            )

        curves_path = self._resolved_confidence_calibration_curves_path()
        if self.options.confidence_calibration_curves is not None and curves_path is None:
            return StageOutcome(
                stage="confidence_gate",
                status="degraded",
                wall_seconds=0.0,
                notes=[f"--confidence-calibration-curves {self.options.confidence_calibration_curves} not found; raw world left as manifest fallback"],
            )
        calibration_curves = _read_optional_json(curves_path) if curves_path is not None else None
        gated = apply_confidence_gate_to_world(
            _read_json(world_path),
            ball_track_physics_filled=_read_optional_json(self.clip_dir / "ball_track_physics_filled.json"),
            physics_footlock=_read_optional_json(self.clip_dir / "physics_footlock.json"),
            racket_pose_estimate=_read_optional_json(self.clip_dir / "racket_pose_estimate.json") if self.options.paddle_pose else None,
            contact_windows=_read_optional_json(self.clip_dir / "contact_windows.json"),
            calibration_curves=calibration_curves,
            config=ConfidenceGateConfig(),
        )
        counts = summarize_bands(gated)
        out_world = self.clip_dir / "confidence_gated_world.json"
        out_summary = self.clip_dir / "confidence_gate_summary.json"
        _write_json(out_world, gated)
        summary = {
            "schema_version": 1,
            "out": str(out_world),
            "run_dir": str(self.clip_dir),
            "calibration_curves": str(curves_path) if curves_path is not None else None,
            "counts_by_entity_band": counts,
            "policy": {
                "additive_only": True,
                "protected_eval_labels_used": False,
                "outdoor_indoor_labels_read": False,
            },
        }
        _write_json(out_summary, summary)

        expected_phys = ["ball_track_physics_filled.json", "physics_footlock.json"]
        if self.options.paddle_pose:
            expected_phys.append(PADDLE_POSE_ARTIFACT_NAME)
        missing_phys = [name for name in expected_phys if not (self.clip_dir / name).is_file()]
        notes = [
            "applied Wave-B confidence gate additively; raw virtual_world.json remains on disk",
            "no Outdoor/Indoor labels read; gate consumes run-dir artifacts only",
        ]
        if curves_path is not None:
            notes.append(f"calibration curves: {curves_path}")
        else:
            notes.append("no calibration_curves.json found; physics predictions fall back to conservative low bands")
        if missing_phys:
            notes.append(f"absent PHYS artifact(s) passed through with provenance where possible: {', '.join(missing_phys)}")

        return StageOutcome(
            stage="confidence_gate",
            status="ran",
            wall_seconds=0.0,
            notes=notes,
            artifacts=["confidence_gated_world.json", "confidence_gate_summary.json"],
            metrics={
                "counts_by_entity_band": counts,
                "calibration_curves": str(curves_path) if curves_path is not None else None,
                "protected_eval_labels_used": False,
                "outdoor_indoor_labels_read": False,
            },
        )

    def _resolved_confidence_calibration_curves_path(self) -> Path | None:
        if self.options.confidence_calibration_curves is not None:
            return self.options.confidence_calibration_curves if self.options.confidence_calibration_curves.is_file() else None
        candidates = [
            self.clip_dir / "calibration_curves.json",
            self.options.run_dir / "calibration_curves.json",
            self.options.run_dir.parent / "calibration_curves.json",
            DEFAULT_CONFIDENCE_CALIBRATION_CURVES,
        ]
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    # ------------------------------------------------------------------
    # stage 11: manifest (replay_viewer_manifest.json)
    # ------------------------------------------------------------------

    def _stage_manifest(self) -> StageOutcome:
        vw_path = self._manifest_world_path()
        if not vw_path.is_file():
            return StageOutcome(stage="manifest", status="blocked", wall_seconds=0.0, notes=["requires virtual_world.json"])

        contact_windows_path = self.clip_dir / "contact_windows.json"
        ball_inflections_path = self.clip_dir / "ball_inflections.json"
        ball_arc_solved_path = self.clip_dir / "ball_track_arc_solved.json"
        ball_arc_render_path = self.clip_dir / "ball_arc_render.json"
        ball_bounce_candidates_path = self.clip_dir / "ball_bounce_candidates.json"
        ball_flight_sanity_path = self.clip_dir / "ball_flight_sanity.json"
        reviewed_bounces_path = self.clip_dir / "reviewed_ball_bounces.json"
        coaching_card_facts_path = self.clip_dir / "coaching_card_facts.json"
        rally_spans_path = self.clip_dir / "rally_spans.json"
        contact_metrics = self._ensure_contact_window_trust_notes(contact_windows_path) if contact_windows_path.is_file() else {}
        replay_scene_path, replay_metrics, replay_notes = self._build_replay_scene_for_manifest(
            world_path=vw_path,
            contact_windows_path=contact_windows_path if contact_windows_path.is_file() else None,
            rally_spans_path=rally_spans_path if rally_spans_path.is_file() else None,
        )
        body_mesh_path = self.clip_dir / "body_mesh.json" if (self.clip_dir / "body_mesh.json").is_file() else None
        root_body_mesh_index_path = self.clip_dir / "body_mesh_index.json"
        nested_body_mesh_index_path = self.clip_dir / "body_mesh_index" / "body_mesh_index.json"
        body_mesh_index_path = (
            root_body_mesh_index_path
            if root_body_mesh_index_path.is_file()
            else nested_body_mesh_index_path
            if nested_body_mesh_index_path.is_file()
            else None
        )
        mesh_status = (
            "windowed_index"
            if body_mesh_index_path is not None
            else "monolithic_unverified"
            if body_mesh_path is not None
            else "skeleton_only"
        )
        mesh_notes: list[str] = []
        if body_mesh_index_path is not None and body_mesh_path is None:
            mesh_notes.append("body_mesh.json not fetched (speed default); using body_mesh_index/ for review-only mesh wiring")
        manifest = build_replay_viewer_manifest(
            clip=self.options.clip,
            video_path=self.clip_dir / f"source{self.options.video.suffix.lower()}",
            virtual_world_path=vw_path,
            player_labels_path=None,
            replay_scene_path=replay_scene_path,
            body_mesh_path=body_mesh_path,
            body_mesh_index_path=body_mesh_index_path,
            physics_refinement_path=None,
            contact_windows_path=contact_windows_path if contact_windows_path.is_file() else None,
            ball_inflections_path=ball_inflections_path if ball_inflections_path.is_file() else None,
            ball_arc_solved_path=ball_arc_solved_path if ball_arc_solved_path.is_file() else None,
            ball_arc_render_path=ball_arc_render_path if ball_arc_render_path.is_file() else None,
            ball_bounce_candidates_path=ball_bounce_candidates_path if ball_bounce_candidates_path.is_file() else None,
            ball_flight_sanity_path=ball_flight_sanity_path if ball_flight_sanity_path.is_file() else None,
            reviewed_bounces_path=reviewed_bounces_path if reviewed_bounces_path.is_file() else None,
            coaching_card_facts_path=coaching_card_facts_path if coaching_card_facts_path.is_file() else None,
            rally_spans_path=rally_spans_path if rally_spans_path.is_file() else None,
            annotation_sources=[],
            vite_allow_root=self.options.vite_allow_root,
            mesh_status=mesh_status,
        )
        out = self.clip_dir / "replay_viewer_manifest.json"
        write_replay_viewer_manifest(out, manifest)
        return StageOutcome(
            stage="manifest",
            status="ran",
            wall_seconds=0.0,
            notes=[
                f"built replay_viewer_manifest.json with {'confidence_gated_world.json' if vw_path.name == 'confidence_gated_world.json' else 'virtual_world.json'} as world URL",
                f"mesh_status={mesh_status}",
                *mesh_notes,
                *replay_notes,
            ],
            artifacts=["replay_viewer_manifest.json", *(["replay_scene.json", "replay_review/"] if replay_scene_path is not None else [])],
            metrics={**contact_metrics, **replay_metrics},
        )

    def _manifest_world_path(self) -> Path:
        gated_path = self.clip_dir / "confidence_gated_world.json"
        if self.options.confidence_gate and gated_path.is_file():
            return gated_path
        return self.clip_dir / "virtual_world.json"

    def _ensure_contact_window_trust_notes(self, path: Path) -> dict[str, Any]:
        payload = _read_json(path)
        events = payload.get("events", []) if isinstance(payload, Mapping) else []
        if not isinstance(events, list):
            return {"contact_event_count": 0, "contact_event_trust_notes": {}}
        changed = False
        note_counts: Counter[str] = Counter()
        for event in events:
            if not isinstance(event, dict):
                continue
            note = event.get("trust_band_note")
            if not isinstance(note, str) or not note.strip():
                note = _contact_event_trust_note(event)
                event["trust_band_note"] = note
                changed = True
            note_counts[str(note)] += 1
        if changed:
            _write_json(path, payload)
        return {"contact_event_count": sum(note_counts.values()), "contact_event_trust_notes": dict(sorted(note_counts.items()))}

    def _build_replay_scene_for_manifest(
        self,
        *,
        world_path: Path,
        contact_windows_path: Path | None,
        rally_spans_path: Path | None,
    ) -> tuple[Path | None, dict[str, Any], list[str]]:
        if not self.options.scene_points:
            self._remove_replay_scene_outputs()
            return (
                None,
                {
                    "replay_point_count": 0,
                    "replay_point_source": "disabled",
                    "replay_point_skipped_broad_span_count": 0,
                },
                ["no replay_scene.json written: --no-scene-points set"],
            )

        contact_windows = _read_optional_json(contact_windows_path) if contact_windows_path is not None else None
        rally_spans = _read_optional_json(rally_spans_path) if rally_spans_path is not None else None
        spans, span_source, span_metrics = _replay_point_spans(contact_windows, rally_spans)
        if not spans:
            self._remove_replay_scene_outputs()
            return (
                None,
                {"replay_point_count": 0, "replay_point_source": "none", **span_metrics},
                ["no replay_scene.json written: no tight contact-window/rally point spans available"],
            )

        scene = build_replay_review_export_from_virtual_world(
            _read_json(world_path),
            export_root=self.clip_dir / "replay_review",
            scene_root=self.clip_dir,
            point_id=1,
        )
        payload = scene.model_dump(mode="json")
        if not payload["points"]:
            return None, {"replay_point_count": 0, "replay_point_source": "none"}, ["no replay_scene.json written: review GLB export produced zero point refs"]
        point_ref = payload["points"][0]["glb_url"]
        point_size_mb = payload["points"][0]["size_mb"]
        point_entries: list[dict[str, Any]] = []
        source_point_glb = self.clip_dir / point_ref
        for index, (t0, t1) in enumerate(spans, start=1):
            if index == 1:
                glb_ref = point_ref
            else:
                glb_ref = (Path(point_ref).parent / f"point_{index:03d}_review.glb").as_posix()
                shutil.copyfile(source_point_glb, self.clip_dir / glb_ref)
            point_entries.append({"id": index, "t0": t0, "t1": t1, "glb_url": glb_ref, "size_mb": point_size_mb})
        payload["points"] = [
            *point_entries
        ]
        validated = validate_replay_export_manifest(self.clip_dir, payload)
        scene_path = self.clip_dir / "replay_scene.json"
        write_replay_scene(scene_path, validated)
        return (
            scene_path,
            {"replay_point_count": len(payload["points"]), "replay_point_source": span_source, **span_metrics},
            [
                f"wrote replay_scene.json with {len(payload['points'])} tight review-static replay point(s) from {span_source}",
                "replay GLBs are review-static load artifacts, not production animated replay evidence",
            ],
        )

    def _remove_replay_scene_outputs(self) -> None:
        (self.clip_dir / "replay_scene.json").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # stage 10: verify (optional headless web-viewer check)
    # ------------------------------------------------------------------

    def _stage_verify(self) -> StageOutcome:
        manifest_path = self.clip_dir / "replay_viewer_manifest.json"
        if not manifest_path.is_file():
            return StageOutcome(stage="verify", status="blocked", wall_seconds=0.0, notes=["requires replay_viewer_manifest.json"])

        from scripts.racketsport.verify_process_video_viewer import verify_viewer_loads

        try:
            result = verify_viewer_loads(manifest_path, out_dir=self.clip_dir / "screenshots")
        except Exception as exc:  # noqa: BLE001
            return StageOutcome(stage="verify", status="degraded", wall_seconds=0.0, notes=[f"headless viewer check unavailable ({type(exc).__name__}: {exc})"])

        status: StageStatus = "ran" if result.get("ok") else "degraded"
        return StageOutcome(
            stage="verify",
            status=status,
            wall_seconds=0.0,
            notes=result.get("notes", []),
            artifacts=result.get("screenshots", []),
            metrics={k: v for k, v in result.items() if k not in {"notes", "screenshots", "ok"}},
        )

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def _write_summary(self, *, wall_seconds: float) -> dict[str, Any]:
        hard_failed = any(outcome.status == "failed" for outcome in self.stage_outcomes)
        any_gap = any(outcome.status in {"degraded", "blocked"} for outcome in self.stage_outcomes)
        if hard_failed:
            status = "failed"
        elif any_gap:
            status = "partial"
        else:
            status = "complete"

        summary = {
            "schema_version": 1,
            "artifact_type": "racketsport_process_video_pipeline_summary",
            "clip": self.options.clip,
            "video": str(self.options.video),
            "run_dir": str(self.options.run_dir),
            "clip_dir": str(self.clip_dir),
            "status": status,
            "wall_seconds": round(wall_seconds, 3),
            "stages": [outcome.as_dict() for outcome in self.stage_outcomes],
            "trust_bands": self.trust_bands,
            "camera_motion_auto": self._camera_motion_auto,
            "best_stack": {
                "manifest_revision": BEST_STACK_MANIFEST.revision,
                "resolved": resolved_best_stack_config_from_options(self.options),
                "overrides": best_stack_overrides_from_options(self.options),
            },
        }
        if self._parallel_body_block is not None:
            summary["parallel_body"] = self._parallel_body_block
        _write_json(self.options.run_dir / "PIPELINE_SUMMARY.json", summary)
        _write_json(self.clip_dir / "PIPELINE_SUMMARY.json", summary)
        return summary


class _HardStageFailure(RuntimeError):
    """Raised for the small set of stages nothing downstream can substitute for."""


# ----------------------------------------------------------------------
# module-level helpers
# ----------------------------------------------------------------------


def _content_hash(path: Path) -> str | None:
    """A stable content hash for the --body-schedule=overlap input-mutation
    guard: None when the path does not exist, a file hash for a single
    artifact, or a hash over every (relative path, content) pair for a
    directory (e.g. body_frames/). This only detects same-run mutation while
    BODY runs on a background thread -- never a promotion/accuracy signal."""

    if path.is_file():
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(child.read_bytes())
        return digest.hexdigest()
    return None


def _foot_contact_phase_count(payload: Mapping[str, Any]) -> int:
    phases = payload.get("phases")
    if isinstance(phases, list):
        return len(phases)
    count = payload.get("phase_count")
    if isinstance(count, bool):
        return 0
    try:
        return max(0, int(count))
    except (TypeError, ValueError):
        return 0


def _populate_missing_transl_world_from_tracks(payload: dict[str, Any], tracks: Mapping[str, Any]) -> int:
    fps = _maybe_positive_float(payload.get("fps")) or _maybe_positive_float(tracks.get("fps")) or 30.0
    track_index: dict[tuple[int, int], list[float]] = {}
    time_index: dict[tuple[int, float], list[float]] = {}
    for player in tracks.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = _maybe_int(player.get("id"))
        if player_id is None:
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            world_xy = frame.get("world_xy")
            t = _maybe_float(frame.get("t"))
            if not isinstance(world_xy, Sequence) or len(world_xy) < 2 or t is None:
                continue
            transl = [float(world_xy[0]), float(world_xy[1]), 0.0]
            track_index[(player_id, int(round(t * fps)))] = transl
            time_index[(player_id, round(t, 6))] = transl

    backfilled = 0
    for player in payload.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        player_id = _maybe_int(player.get("id"))
        if player_id is None:
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            if _vec3_or_none(frame.get("transl_world")) is not None:
                continue
            frame_idx = _maybe_int(frame.get("frame_idx"))
            t = _maybe_float(frame.get("t"))
            transl = track_index.get((player_id, frame_idx)) if frame_idx is not None else None
            if transl is None and t is not None:
                transl = time_index.get((player_id, round(t, 6)))
            if transl is None:
                continue
            frame["transl_world"] = list(transl)
            backfilled += 1
    return backfilled


def _has_r3_grounding_provenance(clip_dir: Path) -> bool:
    quality = _read_optional_json(clip_dir / "body_grounding_quality.json")
    if _payload_has_r3_grounding_provenance(quality):
        return True
    for name in ("skeleton3d.json", "smpl_motion.json"):
        payload = _read_optional_json(clip_dir / name)
        if _payload_has_r3_grounding_provenance(payload):
            return True
    return False


def _payload_has_r3_grounding_provenance(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    grounding_metrics = payload.get("grounding_metrics")
    if isinstance(grounding_metrics, Mapping) and grounding_metrics.get("grounding_anchor_source") == "placement_track_world_xy":
        return True
    for player in payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            confidence = frame.get("confidence_provenance")
            if isinstance(confidence, Mapping) and confidence.get("grounding_anchor_source") == "placement_track_world_xy":
                return True
    return False


def _aggregate_grounding_refine_reports(reports: Mapping[str, Any], *, phase_count: int) -> dict[str, Any]:
    correction_count = 0
    correction_max = 0.0
    correction_mean_total = 0.0
    correction_rms_square_total = 0.0
    warn_count = 0
    kill_recommended = False
    residual_family_worse: dict[str, bool] = {}
    for report in reports.values():
        if not isinstance(report, Mapping):
            continue
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            continue
        correction = summary.get("correction_magnitude_m")
        if isinstance(correction, Mapping):
            count = _nonnegative_int(correction.get("count"))
            correction_count += count
            correction_max = max(correction_max, float(correction.get("max") or 0.0))
            correction_mean_total += float(correction.get("mean") or 0.0) * count
            rms = float(correction.get("rms") or 0.0)
            correction_rms_square_total += (rms * rms) * count
            warn_count += _nonnegative_int(correction.get("warn_count"))
        kill_recommended = kill_recommended or bool(summary.get("kill_recommended"))
        worse = summary.get("residual_family_worse")
        if isinstance(worse, Mapping):
            for family, value in worse.items():
                residual_family_worse[str(family)] = residual_family_worse.get(str(family), False) or bool(value)
    if correction_count:
        correction_mean = correction_mean_total / correction_count
        correction_rms = (correction_rms_square_total / correction_count) ** 0.5
    else:
        correction_mean = 0.0
        correction_rms = 0.0
    return {
        "phase_count": phase_count,
        "target_count": len(reports),
        "correction_magnitude_m": {
            "count": correction_count,
            "max": correction_max,
            "mean": correction_mean,
            "rms": correction_rms,
            "warn_count": warn_count,
        },
        "residual_family_worse": residual_family_worse,
        "kill_recommended": kill_recommended,
    }


def _grounding_refine_stage_report(
    *,
    status: str,
    phase_count: int,
    reports: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_grounding_refine_stage",
        "status": status,
        "phase_count": phase_count,
        "policy": {
            "accuracy_claim": GROUNDING_REFINE_POLICY_NOTE,
            "render_honest_only": True,
            "not_gate_evidence": True,
            "protected_eval_labels_used": False,
            "outdoor_indoor_labels_read": False,
        },
        "summary": dict(summary),
        "reports": dict(reports),
    }


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_positive_float(value: Any) -> float | None:
    number = _maybe_float(value)
    if number is None or number <= 0.0:
        return None
    return number


def _vec3_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _video_probe(video: Path) -> tuple[int, int, float]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(f"opencv-python is required to probe {video}") from exc
    cap = cv2.VideoCapture(str(video))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"could not probe video dimensions/fps for {video}")
    return width, height, fps


def _video_timing_probe(video: Path) -> VideoTiming:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(f"opencv-python is required to probe {video}") from exc
    cap = cv2.VideoCapture(str(video))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"could not probe video timing for {video}")
    return VideoTiming(width=width, height=height, fps=fps, frame_count=frame_count, duration_s=frame_count / fps)


def _median_frame_dt(frames: Sequence[Any]) -> float | None:
    times: list[float] = []
    for frame in frames:
        if isinstance(frame, Mapping):
            time_s = _maybe_float(frame.get("t"))
            if time_s is not None:
                times.append(time_s)
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return None
    return float(median(deltas))


def _last_frame_time(frames: Sequence[Any]) -> float | None:
    for frame in reversed(frames):
        if isinstance(frame, Mapping):
            time_s = _maybe_float(frame.get("t"))
            if time_s is not None:
                return time_s
    return None


def _ball_timeline_coverage_fraction(
    frames: Sequence[Any],
    *,
    fps: float | None,
    median_dt: float | None,
    video_duration_s: float,
) -> float | None:
    if video_duration_s <= 0:
        return None
    last_t = _last_frame_time(frames)
    if last_t is None:
        return None
    sample_dt = median_dt or ((1.0 / fps) if fps and fps > 0 else None)
    if sample_dt is None:
        return None
    return min(1.0, max(0.0, (last_t + sample_dt) / video_duration_s))


def _placement_camera_motion_notes(path: Path | None, source: str, summary: Mapping[str, Any]) -> list[str]:
    if path is None:
        return [f"camera_motion=not_used source={source} frames_used=0 frames_uncompensated=0"]
    frames_used = int(summary.get("camera_motion_frames_used", 0) or 0)
    frames_uncompensated = int(summary.get("camera_motion_frames_uncompensated", 0) or 0)
    return [
        "camera_motion=used "
        f"source={source} path={path} frames_used={frames_used} frames_uncompensated={frames_uncompensated} "
        "preview_advisory=true"
    ]


def _placement_honesty_notes(summary: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    side_summary = summary.get("side_quadrant_consistency")
    if isinstance(side_summary, Mapping):
        players = side_summary.get("players")
        if isinstance(players, Mapping):
            for player_id, item in sorted(players.items(), key=lambda pair: str(pair[0])):
                if not isinstance(item, Mapping):
                    continue
                original_side = str(item.get("side_label_original", ""))
                recomputed_side = str(item.get("side_recomputed", ""))
                original_role = str(item.get("role_original", ""))
                recomputed_role = str(item.get("role_recomputed", ""))
                side_changed = original_side and recomputed_side and original_side != recomputed_side
                role_changed = original_role and recomputed_role and original_role != recomputed_role
                if side_changed or role_changed:
                    notes.append(
                        f"side_recompute(player={player_id}: side {original_side}->{recomputed_side}, "
                        f"role {original_role}->{recomputed_role})"
                    )
    guard_counts: dict[str, int] = {}
    for key in ("boundary_guards", "smoothing_guards"):
        group = summary.get(key)
        if isinstance(group, Mapping) and isinstance(group.get("totals"), Mapping):
            for name, value in group["totals"].items():
                guard_counts[str(name)] = int(value)
    sidecar = summary.get("sidecar_identity")
    if isinstance(sidecar, Mapping):
        for source_name in ("native2d", "sam3d"):
            source_item = sidecar.get(source_name)
            totals = source_item.get("totals") if isinstance(source_item, Mapping) else None
            if isinstance(totals, Mapping):
                guard_counts[f"{source_name}_reassigned"] = int(totals.get("reassigned_obs", 0) or 0)
                guard_counts[f"{source_name}_dropped"] = int(totals.get("dropped_obs", 0) or 0)
    if guard_counts:
        notes.append("placement_guards(" + ", ".join(f"{key}={value}" for key, value in sorted(guard_counts.items())) + ")")
    return notes


def _auto_discover_court_calibration(video: Path) -> Path | None:
    """Look for the eval_clips/ball/<clip>/labels/court_calibration_metric15pt.json
    convention next to the video's own directory, e.g. for
    ``eval_clips/ball/<clip>/source.mp4`` this checks
    ``eval_clips/ball/<clip>/labels/court_calibration_metric15pt.json``. Only used when
    --court-calibration/--capture-sidecar/--court-corners were all omitted."""

    candidate = video.parent / "labels" / "court_calibration_metric15pt.json"
    return candidate if candidate.is_file() else None


def _auto_discover_ball_track(clip: str, *, run_root: Path | None = None) -> Path | None:
    """Find an already-computed ball_track.json for ``clip`` under runs/.

    This is intentionally conservative: every candidate must validate against
    the strict ``ball_track`` schema before it can be used, so summary/run
    metadata files with ball_track-like names are ignored.
    """

    root = run_root or DEFAULT_RUN_ROOT
    preferred = [
        root / "ball_goal_m8_wasb_fullsuite_20260701T183000Z" / clip / "wasb_tennis_tracknet_smoke" / "ball_track.json",
        root
        / "ball_goal_m1_wasb_finetune_20260701T210431Z"
        / "step0_zero_shot_sweep"
        / "scored"
        / "tennis"
        / clip
        / "wasb_tennis_zeroshot_thr_0_500"
        / "ball_track.json",
        root / "eval0" / "prototype_gate_h100_v2" / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100_localtraj.json",
        root / "eval0" / "prototype_gate_h100_v2" / clip / "tracknet_smoke_0000_0010" / "ball_track_fusion_temporal_vball100.json",
        root / "eval0" / "prototype_gate_h100_v2" / clip / "tracknet_smoke_0000_0010" / "ball_track_0000_0010.json",
        root / "eval0" / "prototype_gate_h100_v2" / clip / "ball_track.json",
    ]
    seen: set[Path] = set()
    for candidate in preferred:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid_artifact("ball_track", candidate):
            return candidate

    patterns = [
        f"ball_wasb*/**/{clip}/**/ball_track.json",
        f"ball_goal*wasb*/**/{clip}/**/ball_track.json",
        f"eval0/prototype_gate_h100_v2/{clip}/**/ball_track*.json",
    ]
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if _valid_artifact("ball_track", candidate):
                return candidate
    return None


def _clip_duration_seconds(
    video: Path,
    *,
    tracks_payload: Mapping[str, Any] | None = None,
    ball_track_payload: Mapping[str, Any] | None = None,
    audio_onsets_payload: Mapping[str, Any] | None = None,
) -> float:
    width, height, fps = _video_probe(video)
    del width, height
    duration = 1.0 / fps
    try:
        import cv2  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(str(video))
        try:
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        finally:
            cap.release()
        if frame_count > 0:
            duration = max(duration, frame_count / fps)
    except Exception:  # noqa: BLE001
        pass

    observed_times: list[float] = []
    if tracks_payload is not None:
        for player in tracks_payload.get("players", []) or []:
            if isinstance(player, Mapping):
                observed_times.extend(float(frame["t"]) for frame in player.get("frames", []) or [])
    if ball_track_payload is not None:
        observed_times.extend(float(frame["t"]) for frame in ball_track_payload.get("frames", []) or [])
    if audio_onsets_payload is not None:
        for onset in audio_onsets_payload.get("onsets", []) or []:
            if isinstance(onset, Mapping):
                time_s = onset.get("time_s", onset.get("raw_time_s"))
                if time_s is not None:
                    observed_times.append(float(time_s))
            else:
                observed_times.append(float(onset))
    if observed_times:
        duration = max(duration, max(observed_times) + (1.0 / fps))
    return max(duration, 1.0 / fps)


def _contact_cue_summary(
    contact_windows: Mapping[str, Any],
    *,
    wrist_payload: Mapping[str, Any] | None = None,
    ball_inflections_payload: Mapping[str, Any] | None = None,
    audio_onsets_payload: Any = None,
) -> dict[str, Any]:
    events = [event for event in contact_windows.get("events", []) if isinstance(event, Mapping)]
    event_sources = {
        str(source)
        for event in events
        for source in ((event.get("sources") or {}).keys() if isinstance(event.get("sources"), Mapping) else ())
    }
    wrist_count = _cue_count(wrist_payload, "peaks", "peak_count")
    ball_count = _cue_count(ball_inflections_payload, "candidates", "candidate_count")
    audio_count = _cue_count(audio_onsets_payload, "onsets", "onset_count")
    reviewed_count = sum(1 for event in events if "human_review" in ((event.get("sources") or {}) if isinstance(event.get("sources"), Mapping) else {}))

    missing: list[str] = []
    if wrist_count == 0 and "wrist_vel" not in event_sources:
        missing.append("wrist")
    if ball_count == 0 and "ball_inflection" not in event_sources:
        missing.append("ball_inflections")
    if audio_count == 0 and "audio" not in event_sources:
        missing.append("audio")
    if reviewed_count == 0:
        missing.append("reviewed_contacts")

    return {
        "wrist": "used" if "wrist_vel" in event_sources else ("available" if wrist_count > 0 else "missing"),
        "ball_inflections": "used" if "ball_inflection" in event_sources else ("available" if ball_count > 0 else "missing"),
        "audio": "used" if "audio" in event_sources else ("available" if audio_count > 0 else "missing"),
        "reviewed_contacts": "used" if reviewed_count else "missing",
        "missing": missing,
        "event_source_counts": dict(sorted((source, sum(1 for event in events if source in (event.get("sources") or {}))) for source in event_sources)),
        "event_count": len(events),
        "wrist_candidate_count": wrist_count,
        "ball_inflection_candidate_count": ball_count,
        "audio_onset_count": audio_count,
    }


def _contact_event_trust_note(event: Mapping[str, Any]) -> str:
    sources = event.get("sources") if isinstance(event.get("sources"), Mapping) else {}
    human = _source_score(sources, "human_review")
    if human is not None and human >= 0.5:
        return "human-reviewed contact, replay-only"
    audio = (_source_score(sources, "audio") or 0.0) > 0.0
    wrist = (_source_score(sources, "wrist_vel") or 0.0) > 0.0
    ball = (_source_score(sources, "ball_inflection") or 0.0) > 0.0
    if wrist and ball and audio:
        return "audio+wrist+ball cues, unverified"
    if wrist and ball:
        return "wrist+ball cues, unverified"
    if wrist:
        return "wrist-cue-only, unverified"
    if ball:
        return "ball-cue-only, unverified"
    if audio:
        return "audio-cue-only, unverified"
    return f"{event.get('type', 'event')}-cue unverified"


def _source_score(sources: Mapping[str, Any], key: str) -> float | None:
    value = sources.get(key)
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _replay_point_spans(
    contact_windows: Mapping[str, Any] | None,
    rally_spans: Mapping[str, Any] | None,
) -> tuple[list[tuple[float, float]], str, dict[str, Any]]:
    skipped_broad = 0
    rally_tight: list[tuple[float, float]] = []
    if isinstance(rally_spans, Mapping) and isinstance(rally_spans.get("spans"), list):
        rally_all = _normalized_replay_spans(
            (_safe_float(span.get("t0")), _safe_float(span.get("t1")))
            for span in rally_spans["spans"]
            if isinstance(span, Mapping)
        )
        rally_tight = _tight_replay_spans(rally_all)
        skipped_broad += len(rally_all) - len(rally_tight)

    events = contact_windows.get("events") if isinstance(contact_windows, Mapping) else None
    event_spans: list[tuple[float | None, float | None]] = []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            window = event.get("window") if isinstance(event.get("window"), Mapping) else {}
            t0 = _safe_float(window.get("t0"))
            t1 = _safe_float(window.get("t1"))
            if t0 is None or t1 is None:
                event_t = _safe_float(event.get("t"))
                if event_t is None:
                    continue
                t0 = event_t
                t1 = event_t + 0.001
            event_spans.append((t0, t1))
    event_all = _normalized_replay_spans(event_spans)
    event_tight = _tight_replay_spans(event_all)
    skipped_broad += len(event_all) - len(event_tight)
    metrics = {"replay_point_skipped_broad_span_count": skipped_broad}
    if event_tight:
        return event_tight, "contact_windows", metrics
    if rally_tight:
        return rally_tight, "rally_spans", metrics
    return [], "none", metrics


def _normalized_replay_spans(spans: Sequence[tuple[float | None, float | None]]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    previous_t1: float | None = None
    for raw_t0, raw_t1 in sorted(spans, key=lambda item: (float("inf") if item[0] is None else item[0])):
        if raw_t0 is None or raw_t1 is None:
            continue
        t0 = max(0.0, float(raw_t0))
        t1 = max(0.0, float(raw_t1))
        if previous_t1 is not None and t0 < previous_t1:
            t0 = previous_t1
        if t1 <= t0:
            continue
        normalized.append((t0, t1))
        previous_t1 = t1
    return normalized


def _tight_replay_spans(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(t0, t1) for t0, t1 in spans if (t1 - t0) < REPLAY_POINT_MAX_SPAN_SECONDS]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _cue_count(payload: Any, item_key: str, summary_key: str) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, Mapping):
        return 0
    items = payload.get(item_key)
    if isinstance(items, list):
        return len(items)
    summary = payload.get("summary")
    if isinstance(summary, Mapping) and summary.get(summary_key) is not None:
        try:
            return max(0, int(summary[summary_key]))
        except (TypeError, ValueError):
            return 0
    return 0


def _filter_tracks_payload_to_rally_spans(payload: Mapping[str, Any], spans: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], int, int]:
    filtered = copy.deepcopy(dict(payload))
    before = 0
    after = 0
    for player in filtered.get("players", []) or []:
        frames = list(player.get("frames", []) or [])
        before += len(frames)
        kept = [frame for frame in frames if in_rally_span(float(frame["t"]), spans)]
        player["frames"] = kept
        after += len(kept)
    filtered["rally_spans"] = [{"t0": float(span["t0"]), "t1": float(span["t1"])} for span in spans]
    return filtered, before, after


def _filter_ball_payload_to_rally_spans(payload: Mapping[str, Any], spans: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], int, int]:
    filtered = copy.deepcopy(dict(payload))
    frames = list(filtered.get("frames", []) or [])
    kept = [frame for frame in frames if in_rally_span(float(frame["t"]), spans)]
    filtered["frames"] = kept
    return filtered, len(frames), len(kept)


def _read_declared_court_corners(path: Path) -> tuple[dict[str, list[float]], tuple[int, int]]:
    item = _read_declared_court_corners_item(path)

    raw_corners = item["court_corners"]
    missing = [key for key in SIDECAR_CORNER_ORDER if key not in raw_corners]
    if missing:
        raise ValueError(f"{path}: missing court corner(s): {', '.join(missing)}")

    image_size = item.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(
            f"{path}: missing declared image_size [width, height] on the court_corners item -- required since the "
            "2026-07-02 pixel-space audit (state the resolution the corners were tapped against; a mismatch here "
            "silently misplaces the homography, see threed/racketsport/ball_manual_court_inout.py's module docstring)"
        )
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: image_size must be positive, got {image_size}")

    corners = {key: [float(raw_corners[key][0]), float(raw_corners[key][1])] for key in SIDECAR_CORNER_ORDER}
    return corners, (width, height)


def _read_declared_court_corners_item(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = (payload.get("annotation") or {}).get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: no annotation.items[] with court_corners found")
    item = next((it for it in items if isinstance(it, dict) and isinstance(it.get("court_corners"), dict)), None)
    if item is None:
        raise ValueError(f"{path}: no item with a court_corners object found")
    return item


def _declared_court_corners_are_auto_preview(item: Mapping[str, Any]) -> bool:
    source = str(item.get("source") or "")
    status = str(item.get("status") or "")
    review_status = str(item.get("review_status") or "")
    return (
        bool(item.get("not_cal3_verified"))
        or status == "auto_preview_unverified"
        or review_status == "auto_predicted_unreviewed"
        or "auto" in source
        or "detector" in source
    )


def _capture_sidecar_has_manual_taps(path: Path) -> bool:
    payload = _read_json(path)
    taps = payload.get("manual_court_taps")
    return isinstance(taps, list) and len(taps) >= 4


def _capture_sidecar_with_preview_corners(sidecar_path: Path, court_corners_path: Path) -> dict[str, Any]:
    payload = _read_json(sidecar_path)
    corners, (width, height) = _read_declared_court_corners(court_corners_path)
    payload["manual_court_taps"] = [corners[key] for key in SIDECAR_CORNER_ORDER]
    payload["resolution"] = [int(width), int(height)]
    payload["capture_quality"] = _preview_capture_quality(
        payload.get("capture_quality"),
        extra_reasons=["process_video_auto_court_corners_preview", "manual_taps_seeded_from_unverified_detector"],
    )
    return payload


def _preview_capture_quality(existing: Any, *, extra_reasons: Sequence[str]) -> dict[str, Any]:
    reasons: list[str] = []
    if isinstance(existing, Mapping):
        reasons.extend(str(reason) for reason in (existing.get("reasons") or []))
    reasons.extend(extra_reasons)
    return {
        "grade": "poor",
        "reasons": list(dict.fromkeys(reasons)),
    }


def _auto_court_corners_preview_from_video(video_path: Path, out_path: Path) -> Path:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("auto-court preview requires opencv-python") from exc

    from threed.racketsport.court_line_keypoints import detect_court_keypoints_from_image

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video for auto-court preview: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"could not read video metadata for auto-court preview: {video_path}")

        best: tuple[float, int, dict[str, Any], dict[str, Any]] | None = None
        errors: list[str] = []
        for frame_index in _auto_court_preview_frame_indexes(frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                errors.append(f"frame {frame_index}: unreadable")
                continue
            try:
                detected = detect_court_keypoints_from_image(frame, cv2_module=cv2)
            except Exception as exc:  # noqa: BLE001 - try several frames before failing the hard stage
                errors.append(f"frame {frame_index}: {exc}")
                continue
            corners = _court_corners_from_keypoints(detected.keypoints)
            score = float(detected.confidence)
            metadata = {
                "detector": "auto_white_line_preview",
                "detector_confidence": score,
                "raw_segment_count": int(detected.raw_segment_count),
                "merged_line_count": int(detected.merged_line_count),
            }
            if best is None or score > best[0]:
                best = (score, frame_index, corners, metadata)
    finally:
        capture.release()

    if best is None:
        detail = "; ".join(errors[:5]) if errors else "no candidate frames sampled"
        raise ValueError(f"auto-court preview failed to detect usable court corners: {detail}")

    _, frame_index, corners, metadata = best
    payload = {
        "annotation": {
            "items": [
                {
                    "court_corners": corners,
                    "frame": f"frame_{frame_index:06d}.jpg",
                    "image_size": [int(width), int(height)],
                    "source": "auto_white_line_preview",
                    "status": "auto_preview_unverified",
                    "not_cal3_verified": True,
                    **metadata,
                }
            ]
        }
    }
    _write_json(out_path, payload)
    return out_path


def _court_proposals_preview_from_video(
    video_path: Path,
    *,
    clip: str,
    out_path: Path,
    max_frames: int,
) -> Path:
    from scripts.racketsport.build_court_proposals import build_court_proposal_report
    from threed.racketsport.court_proposals import write_court_proposal_report

    report = build_court_proposal_report(video=str(video_path), clip=clip, max_frames=max_frames)
    write_court_proposal_report(out_path, report)
    return out_path


def _auto_court_preview_frame_indexes(frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    candidates = [0, frame_count // 4, frame_count // 2, (frame_count * 3) // 4, frame_count - 1]
    return list(dict.fromkeys(max(0, min(frame_count - 1, int(index))) for index in candidates))


def _court_corners_from_keypoints(keypoints: Mapping[str, Mapping[str, Any]]) -> dict[str, list[float]]:
    name_map = {
        "near_left": "near_left_corner",
        "near_right": "near_right_corner",
        "far_right": "far_right_corner",
        "far_left": "far_left_corner",
    }
    return {corner: _keypoint_xy(keypoints[keypoint], keypoint) for corner, keypoint in name_map.items()}


def _keypoint_xy(keypoint: Mapping[str, Any], name: str) -> list[float]:
    xy = keypoint.get("xy")
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError(f"auto-court preview keypoint {name} is missing xy")
    return [float(xy[0]), float(xy[1])]


def _capture_sidecar_from_court_corners(court_corners_path: Path, *, fps: float) -> dict[str, Any]:
    item = _read_declared_court_corners_item(court_corners_path)
    corners, (width, height) = _read_declared_court_corners(court_corners_path)
    manual_taps = [corners[key] for key in SIDECAR_CORNER_ORDER]
    focal = float(max(width, height) * 1.2)
    auto_preview = _declared_court_corners_are_auto_preview(item)
    capture_quality = (
        {
            "grade": "poor",
            "reasons": [
                "process_video_auto_court_corners_preview",
                "manual_taps_seeded_from_unverified_detector",
                "estimated_intrinsics",
                "corrected_unverified",
            ],
        }
        if auto_preview
        else {
            "grade": "warn",
            "reasons": ["process_video_manual_court_corners", "estimated_intrinsics", "corrected_unverified"],
        }
    )
    return {
        "schema_version": 1,
        "device_tier": "fallback",
        "device_model": f"process_video_cli:{court_corners_path.name}",
        "fps": int(round(fps)),
        "format": "hevc",
        "resolution": [int(width), int(height)],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {
            "fx": focal,
            "fy": focal,
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "dist": [],
            "source": "estimated_from_declared_court_corners",
        },
        "arkit_camera_pose": None,
        "court_plane": None,
        "manual_court_taps": manual_taps,
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": capture_quality,
    }


def _remote_seed_capture_sidecar_from_calibration(payload: Mapping[str, Any], *, fps: float) -> dict[str, Any]:
    """Derive a manual-taps ``capture_sidecar.json`` from an externally-solved
    ``court_calibration.json`` (Task #46 remote BODY dispatch seed).

    The external metric calibration's ``image_pts``/``world_pts`` include the four
    court corners (world x/y extremes) in the calibration's own declared pixel
    space -- exactly the four taps ``ManualCalibrationRunner`` needs on the remote
    A100, whose committed orchestrator cannot consume the external calibration
    artifact directly. Corner naming follows the human tap convention: the
    baseline lower in the image (larger mean image y) is "near", and within each
    baseline "left" is the smaller image x -- verified against Wolverine's real
    human-reviewed corner taps, which this derivation reproduces to within ~3px
    (after resolution scaling). Intrinsics come from the calibration itself, not
    the coarse ``max(w,h)*1.2`` estimate used for tap-only runs.
    """

    image_pts = payload.get("image_pts") or []
    world_pts = payload.get("world_pts") or []
    image_size = payload.get("image_size") or []
    if len(image_pts) != len(world_pts) or len(image_pts) < 4:
        raise ValueError("calibration artifact lacks matched image_pts/world_pts for corner derivation")
    if not (isinstance(image_size, (list, tuple)) and len(image_size) == 2):
        raise ValueError("calibration artifact lacks a declared image_size for corner derivation")
    width, height = int(image_size[0]), int(image_size[1])

    xs = [float(pt[0]) for pt in world_pts]
    ys = [float(pt[1]) for pt in world_pts]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    def _image_point_at(world_x: float, world_y: float) -> list[float]:
        for image_pt, world_pt in zip(image_pts, world_pts):
            if abs(float(world_pt[0]) - world_x) < 1e-6 and abs(float(world_pt[1]) - world_y) < 1e-6:
                return [float(image_pt[0]), float(image_pt[1])]
        raise ValueError(f"calibration has no image point for world corner ({world_x}, {world_y})")

    baseline_pairs = {
        world_y: sorted(
            (_image_point_at(min_x, world_y), _image_point_at(max_x, world_y)),
            key=lambda pt: pt[0],  # image-left first
        )
        for world_y in (min_y, max_y)
    }
    # "near" = the baseline that sits lower in the image (larger mean image y).
    near_y = max(baseline_pairs, key=lambda world_y: sum(pt[1] for pt in baseline_pairs[world_y]) / 2.0)
    far_y = min_y if near_y == max_y else max_y
    near_left, near_right = baseline_pairs[near_y]
    far_left, far_right = baseline_pairs[far_y]
    corners = {"near_left": near_left, "near_right": near_right, "far_right": far_right, "far_left": far_left}
    manual_taps = [corners[key] for key in SIDECAR_CORNER_ORDER]

    intrinsics = payload.get("intrinsics") or {}
    return {
        "schema_version": 1,
        "device_tier": "fallback",
        "device_model": "process_video_remote_body_seed:external_calibration_corners",
        "fps": int(round(fps)),
        "format": "hevc",
        "resolution": [width, height],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {
            "fx": float(intrinsics.get("fx", max(width, height) * 1.2)),
            "fy": float(intrinsics.get("fy", max(width, height) * 1.2)),
            "cx": float(intrinsics.get("cx", width / 2.0)),
            "cy": float(intrinsics.get("cy", height / 2.0)),
            "dist": [float(value) for value in (intrinsics.get("dist") or [])],
            "source": "estimated_from_declared_court_corners",
        },
        "arkit_camera_pose": None,
        "court_plane": None,
        "manual_court_taps": manual_taps,
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {
            "grade": "warn",
            "reasons": [
                "process_video_remote_body_seed",
                "derived_from_external_metric_calibration",
                "remote_dependency_only_not_local_world_calibration",
                "corrected_unverified",
            ],
        },
    }


def _court_calibration_needs_correction(calibration: Mapping[str, Any], evidence: Any) -> bool:
    return _calibration_is_unverified_or_estimated(calibration) and not _court_line_evidence_ready(evidence)


def _court_detector_v2_promoted(proposal: Mapping[str, Any]) -> bool:
    return (
        proposal.get("artifact_type") == "racketsport_court_detector_v2_proposals"
        and proposal.get("promoted") is True
        and proposal.get("verified") is True
        and proposal.get("not_cal3_verified") is False
        and proposal.get("promotion_status") == "promoted"
    )


def _calibration_is_unverified_or_estimated(calibration: Mapping[str, Any]) -> bool:
    quality = calibration.get("capture_quality")
    grade = ""
    reasons: list[str] = []
    if isinstance(quality, Mapping):
        grade = str(quality.get("grade") or "").lower()
        reasons = [str(reason).lower() for reason in (quality.get("reasons") or [])]
    intrinsics = calibration.get("intrinsics")
    intrinsics_source = ""
    if isinstance(intrinsics, Mapping):
        intrinsics_source = str(intrinsics.get("source") or "").lower()
    if grade == "poor":
        return True
    haystack = [intrinsics_source, *reasons]
    return any(any(token in value for token in UNVERIFIED_COURT_REASON_TOKENS) for value in haystack)


def _court_line_evidence_ready(evidence: Any) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    aggregate = evidence.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return False
    return aggregate.get("auto_calibration_ready") is True


def _build_court_correction_task(
    *,
    calibration_path: Path,
    calibration: Mapping[str, Any],
    evidence_path: Path,
    evidence: Any,
) -> dict[str, Any]:
    quality = calibration.get("capture_quality") if isinstance(calibration.get("capture_quality"), Mapping) else {}
    intrinsics = calibration.get("intrinsics") if isinstance(calibration.get("intrinsics"), Mapping) else {}
    evidence_aggregate = evidence.get("aggregate") if isinstance(evidence, Mapping) and isinstance(evidence.get("aggregate"), Mapping) else {}
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_correction_task",
        "court_status": "needs_user_correction",
        "blocked_downstream": COURT_CORRECTION_BLOCKED_DOWNSTREAM,
        "reason": "court_calibration_unverified_or_evidence_not_ready",
        "calibration": {
            "path": str(calibration_path),
            "capture_quality_grade": quality.get("grade"),
            "capture_quality_reasons": list(quality.get("reasons") or []),
            "intrinsics_source": intrinsics.get("source"),
        },
        "court_line_evidence": {
            "path": str(evidence_path),
            "auto_calibration_ready": evidence_aggregate.get("auto_calibration_ready"),
            "missing_required_line_ids": list(evidence_aggregate.get("missing_required_line_ids") or []),
            "missing_required_net_ids": list(evidence_aggregate.get("missing_required_net_ids") or []),
            "reasons": list(evidence_aggregate.get("reasons") or []),
        },
    }


def _build_detector_v2_correction_task(*, proposal_path: Path, proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_correction_task",
        "court_status": "needs_user_correction",
        "blocked_downstream": COURT_CORRECTION_BLOCKED_DOWNSTREAM,
        "reason": "court_detector_v2_not_promoted",
        "detector_v2": {
            "path": str(proposal_path),
            "promoted": proposal.get("promoted"),
            "verified": proposal.get("verified"),
            "not_cal3_verified": proposal.get("not_cal3_verified"),
            "promotion_status": proposal.get("promotion_status"),
            "promotion_blockers": list(proposal.get("promotion_blockers") or []),
            "needs_user_input": list(proposal.get("needs_user_input") or []),
            "selected_hypothesis_id": proposal.get("selected_hypothesis_id"),
        },
    }


def _build_court_proposals_correction_task(*, proposal_path: Path, proposal: Mapping[str, Any]) -> dict[str, Any]:
    ranking = proposal.get("ranking") if isinstance(proposal.get("ranking"), Mapping) else {}
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_correction_task",
        "court_status": "needs_user_correction",
        "blocked_downstream": COURT_CORRECTION_BLOCKED_DOWNSTREAM,
        "reason": "court_proposals_preview_not_trusted_calibration",
        "court_proposals": {
            "path": str(proposal_path),
            "status": proposal.get("status"),
            "verified": proposal.get("verified"),
            "not_cal3_verified": proposal.get("not_cal3_verified"),
            "selected_proposal_id": ranking.get("selected_proposal_id"),
            "abstain": ranking.get("abstain"),
            "abstain_reasons": list(ranking.get("abstain_reasons") or []),
            "proposal_count": len(proposal.get("proposals") or []),
        },
    }


def _valid_artifact(schema: str, path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        validate_artifact_file(schema, path)
    except Exception:  # noqa: BLE001
        return False
    return True


def _is_real_sam3d_skeleton(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return False
    provenance = payload.get("provenance")
    joint_names = payload.get("joint_names")
    model_candidates = {
        str(payload.get("source_model", "")),
    }
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


def _has_raw_pool_artifact(pool_dir: Path) -> bool:
    return (pool_dir / "tracked_detections.json").is_file() or (pool_dir / "raw_tracked_detections.json").is_file()


def _default_raw_pool_authority_profile_for_clip(clip: str, *, raw_pool_dir: Path | None = None) -> str:
    _ = (clip, raw_pool_dir)
    return DEFAULT_GLOBAL_ASSOCIATION_PROFILE


def _raw_pool_has_identity_bbox_scale(raw_pool_dir: Path | None) -> bool:
    if raw_pool_dir is None:
        return False
    metrics = _read_optional_json(Path(raw_pool_dir) / "metrics.json")
    if not isinstance(metrics, Mapping):
        return False
    counts = metrics.get("counts")
    if not isinstance(counts, Mapping):
        return False
    if counts.get("bbox_scale_status") == "identity":
        return True
    return (
        _float_or_none(counts.get("bbox_scale_x")) == 1.0
        and _float_or_none(counts.get("bbox_scale_y")) == 1.0
        and _float_or_none(counts.get("source_width")) == _float_or_none(counts.get("calibration_width"))
        and _float_or_none(counts.get("source_height")) == _float_or_none(counts.get("calibration_height"))
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_pool_authority_config_for_profile(
    profile_name: str,
    *,
    expected_players: int,
    reid_device: str | None,
    reid_batch_size: int,
) -> tuple[RawPoolAuthorityConfig, str]:
    try:
        profile = RAW_POOL_GLOBAL_ASSOCIATION_PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown raw-pool global-association profile {profile_name!r}") from exc
    config_kwargs: dict[str, Any] = {
        "expected_players": expected_players,
        "reid_backend": "osnet",
        "reid_model_name": "osnet_x1_0",
        "reid_device": reid_device,
        "reid_batch_size": reid_batch_size,
        "court_margin_m": profile.court_margin_m,
    }
    if profile.min_conf is not None:
        config_kwargs["min_conf"] = profile.min_conf
    if profile.appearance_weight is not None:
        config_kwargs["appearance_weight"] = profile.appearance_weight
    if profile.max_gap_fill_frames is not None:
        config_kwargs["max_gap_fill_frames"] = profile.max_gap_fill_frames
    if profile.max_merge_cost is not None:
        config_kwargs["max_merge_cost"] = profile.max_merge_cost
    if profile.cardinality_backfill is not None:
        config_kwargs["cardinality_backfill"] = profile.cardinality_backfill
    return RawPoolAuthorityConfig(**config_kwargs), profile_name


def _spine_stage_succeeded(result: Mapping[str, Any], *, stage: str) -> bool:
    """Did ``stage`` itself succeed within a ``run_pipeline(stage=stage, ...)`` call?

    Task #45 S1 (continued): run_pipeline's own top-level ``status`` can come back
    "blocked" even when ``stage``'s own runner completed successfully -- e.g. an
    upstream calibration stage's real automatic court-line/net evidence is
    advisory-not-ready for a trusted calibration source (S1), so
    ManualCalibrationRunner/ExternalCalibrationRunner return normally, but
    ``pipeline_contracts.build_readiness_report`` (a separate, unrelated readiness
    signal about whether *every* stage in the requested closure is ready, not just
    whether ``stage`` itself ran) still reports "not_ready" for that inherited
    upstream reason, and run_pipeline downgrades its own summary status to "blocked"
    because of it. Every process_video.py stage call
    (calibration/tracking/pose/body-local) used to treat any non-"pass" result the
    same as a real failure, which re-introduced S1's hard-fail/false-degrade bug one
    layer up for every stage downstream of calibration, even after orchestrator.py's
    own gate was fixed and the stage's real work (e.g. a full BoT-SORT tracking pass)
    had actually succeeded and written valid artifacts. Only a literal failure of
    ``stage`` itself (its own StageRun entry reports "fail", or run_pipeline never even
    reached it) should be treated as a real failure here.
    """

    if result.get("status") == orchestrator.PIPELINE_STATUS_PASS:
        return True
    stages = result.get("stages")
    if not isinstance(stages, list) or not stages:
        return False
    matching = [item for item in stages if isinstance(item, Mapping) and item.get("stage") == stage]
    if not matching:
        return False
    return matching[-1].get("status") not in {"fail", "blocked"}


def _spine_failure_detail(summary: Mapping[str, Any]) -> str:
    stages = summary.get("stages")
    if not isinstance(stages, list) or not stages:
        return "no stage details available"
    last = stages[-1]
    if not isinstance(last, Mapping):
        return "last stage details malformed"
    notes = last.get("notes")
    if isinstance(notes, list) and notes:
        return "; ".join(str(note) for note in notes)
    return f"last stage {last.get('stage', 'unknown')} status {last.get('status', 'unknown')}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> Any | None:
    return _read_json(path) if path.is_file() else None


def _first_existing_path(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _is_valid_paddle_pose_payload(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("artifact_type") == PADDLE_POSE_ARTIFACT_TYPE
        and payload.get("source") == PADDLE_POSE_SOURCE
    )


def _paddle_pose_zero_coverage() -> dict[str, int]:
    return {
        "estimate_frame_count": 0,
        "input_player_count": 0,
        "hidden_frame_count": 0,
    }


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _paddle_pose_coverage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, Mapping):
        return _paddle_pose_zero_coverage()
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return _paddle_pose_zero_coverage()
    return {
        "estimate_frame_count": _nonnegative_int(summary.get("estimate_frame_count")),
        "input_player_count": _nonnegative_int(summary.get("input_player_count")),
        "hidden_frame_count": _nonnegative_int(summary.get("hidden_frame_count")),
    }


def _paddle_pose_blocked_stage_metrics(reason: str, *, status: str = "blocked") -> dict[str, Any]:
    return {
        "paddle_pose": {
            "status": status,
            "reason": reason,
            "coverage": _paddle_pose_zero_coverage(),
        }
    }


def _paddle_pose_stage_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    summary_mapping = summary if isinstance(summary, Mapping) else {}
    metrics: dict[str, Any] = {
        "status": str(payload.get("status", "blocked")),
        "artifact_type": str(payload.get("artifact_type", "")),
        "source": str(payload.get("source", "")),
        "coverage": _paddle_pose_coverage(payload),
        "render_only": bool(payload.get("render_only", True)),
        "not_for_detection_metrics": bool(payload.get("not_for_detection_metrics", True)),
        "trusted_for_rkt_promotion": bool(payload.get("trusted_for_rkt_promotion", False)),
        "rkt_gate_unscoreable": bool(payload.get("rkt_gate_unscoreable", True)),
    }
    band_distribution = summary_mapping.get("band_distribution")
    if isinstance(band_distribution, Mapping):
        metrics["band_distribution"] = dict(band_distribution)
    evidence_channels = summary_mapping.get("evidence_channels")
    if isinstance(evidence_channels, Mapping):
        metrics["evidence_channels"] = dict(evidence_channels)
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        metrics["warnings"] = [str(item) for item in warnings]
    blockers = payload.get("blockers")
    if isinstance(blockers, list):
        metrics["blockers"] = [str(item) for item in blockers]
    return metrics


def _paddle_pose_blocked_reason(payload: Mapping[str, Any]) -> str:
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        return str(blockers[0])
    coverage = _paddle_pose_coverage(payload)
    if coverage["estimate_frame_count"] <= 0:
        return "no_fused_paddle_pose_frames"
    return "fused_estimator_blocked"


def _path_summary(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def resolved_best_stack_config_from_options(options: PipelineOptions) -> dict[str, Any]:
    detector_fov = {
        "detector_name": options.remote_config.body_detector_name,
        "fov_name": options.remote_config.body_fov_name,
        "fov_checkpoint_model_id": BEST_STACK_MANIFEST.value("body.detector_fov")["fov_checkpoint_model_id"],
        "fov_checkpoint_required_when_fov_enabled": BEST_STACK_MANIFEST.value("body.detector_fov")[
            "fov_checkpoint_required_when_fov_enabled"
        ],
    }
    paddle_fused = copy.deepcopy(BEST_STACK_MANIFEST.value("paddle.fused_estimator"))
    if not isinstance(paddle_fused, dict):
        paddle_fused = {"enabled": bool(paddle_fused)}
    paddle_fused["enabled"] = bool(options.paddle_pose)
    return {
        "ball.wasb_checkpoint": _path_summary(options.wasb_checkpoint),
        "ball.wasb_repo": _path_summary(options.wasb_repo),
        "ball.arc_chain": not options.no_ball_arc,
        "tracking.reid_model": _path_summary(options.reid_model),
        "tracking.global_association_profile": options.global_association_profile or DEFAULT_GLOBAL_ASSOCIATION_PROFILE,
        "confidence.calibration_curves": _path_summary(
            options.confidence_calibration_curves or DEFAULT_CONFIDENCE_CALIBRATION_CURVES
        ),
        "mesh.coverage_mode": options.mesh_coverage_mode,
        "mesh.byte_budget_mib": options.mesh_byte_budget_mib,
        "mesh.target_frame_budget": options.target_mesh_frame_budget,
        "body.detector_fov": detector_fov,
        "body.schedule": options.body_schedule,
        "paddle.fused_estimator": paddle_fused,
        "camera_motion.policy": {
            "mode": "disabled" if options.skip_camera_motion else "forced" if options.enable_camera_motion else "auto",
            "threshold": options.camera_motion_auto_threshold,
            "estimator": options.camera_motion_estimator,
            "flow_backend": options.camera_motion_flow_backend,
            "person_masks": options.camera_motion_person_masks,
            "enable_flag_default": options.enable_camera_motion,
            "disable_flag_default": options.skip_camera_motion,
        },
        "placement.undistort": options.placement_undistort,
    }


def best_stack_overrides_from_options(options: PipelineOptions) -> dict[str, Any]:
    resolved = resolved_best_stack_config_from_options(options)
    manifest_defaults = {
        "ball.wasb_checkpoint": BEST_STACK_MANIFEST.path_value("ball.wasb_checkpoint").as_posix(),
        "ball.wasb_repo": BEST_STACK_MANIFEST.path_value("ball.wasb_repo").as_posix(),
        "ball.arc_chain": BEST_STACK_MANIFEST.value("ball.arc_chain"),
        "tracking.reid_model": BEST_STACK_MANIFEST.path_value("tracking.reid_model", must_exist=False).as_posix(),
        "tracking.global_association_profile": BEST_STACK_MANIFEST.string_value("tracking.global_association_profile"),
        "confidence.calibration_curves": BEST_STACK_MANIFEST.path_value("confidence.calibration_curves").as_posix(),
        "mesh.coverage_mode": BEST_STACK_MANIFEST.string_value("mesh.coverage_mode"),
        "mesh.byte_budget_mib": BEST_STACK_MANIFEST.number_value("mesh.byte_budget_mib"),
        "mesh.target_frame_budget": BEST_STACK_MANIFEST.value("mesh.target_frame_budget"),
        "body.detector_fov": BEST_STACK_MANIFEST.value("body.detector_fov"),
        "body.schedule": BEST_STACK_MANIFEST.string_value("body.schedule"),
        "paddle.fused_estimator": copy.deepcopy(BEST_STACK_MANIFEST.value("paddle.fused_estimator")),
        "camera_motion.policy": BEST_STACK_MANIFEST.value("camera_motion.policy"),
        "placement.undistort": BEST_STACK_MANIFEST.value("placement.undistort"),
    }
    return {
        key: {"manifest": manifest_defaults[key], "resolved": value}
        for key, value in resolved.items()
        if key in manifest_defaults and value != manifest_defaults[key]
    }


def _existing_optional_path(path: Path) -> Path | None:
    return path if path.is_file() else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_json(source: Path, target: Path) -> None:
    _write_json(target, _read_json(source))


def _frame_active_player_ids(frame: Mapping[str, Any]) -> list[int]:
    ids = frame.get("active_player_ids")
    if isinstance(ids, list):
        return [int(player_id) for player_id in ids]
    return []


def _evenly_sample_frames(frames: Sequence[dict[str, Any]], *, max_count: int) -> list[dict[str, Any]]:
    ordered = sorted(frames, key=lambda frame: int(frame.get("frame_idx", 0)))
    if max_count <= 0 or len(ordered) <= max_count:
        return ordered
    if max_count == 1:
        return [ordered[len(ordered) // 2]]
    selected: list[dict[str, Any]] = []
    last_index = len(ordered) - 1
    for slot in range(max_count):
        index = round(slot * last_index / (max_count - 1))
        selected.append(ordered[index])
    return selected


def _deep_mesh_windows_from_plan_frames(frames: Sequence[Any], *, fps: float) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    last_frame_idx: int | None = None
    for frame in sorted((item for item in frames if isinstance(item, Mapping)), key=lambda item: int(item.get("frame_idx", 0))):
        frame_idx = int(frame.get("frame_idx", 0))
        if frame.get("recommended_tier") != "deep_mesh":
            if current:
                windows.append(_deep_mesh_window_from_plan_frames(current, fps=fps))
                current = []
                last_frame_idx = None
            continue
        if current and last_frame_idx is not None and frame_idx != last_frame_idx + 1:
            windows.append(_deep_mesh_window_from_plan_frames(current, fps=fps))
            current = []
        current.append(frame)
        last_frame_idx = frame_idx
    if current:
        windows.append(_deep_mesh_window_from_plan_frames(current, fps=fps))
    return windows


def _deep_mesh_window_from_plan_frames(frames: Sequence[Mapping[str, Any]], *, fps: float) -> dict[str, Any]:
    frame_start = int(frames[0]["frame_idx"])
    frame_end = int(frames[-1]["frame_idx"])
    target_player_ids = sorted(
        {
            int(target["player_id"])
            for frame in frames
            for target in frame.get("player_targets", [])
            if isinstance(target, Mapping) and target.get("target_representation") == "world_mesh"
        }
    )
    reasons = Counter(
        str(reason)
        for frame in frames
        for reason in frame.get("reasons", [])
    )
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "t0": frame_start / fps,
        "t1": (frame_end + 1) / fps,
        "frame_count": len(frames),
        "target_representation": "world_mesh",
        "fallback_representation": "sam3d_body_joints",
        "target_player_ids": target_player_ids,
        "reason_counts": dict(sorted(reasons.items())),
        "max_score": max(float(frame.get("score", 0.0)) for frame in frames),
    }


def _frame_compute_plan_summary(frames: Sequence[Any], *, deep_mesh_windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_tier: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_player_target_representation: Counter[str] = Counter()
    targeted_reviewed_contact_frame_count = 0
    coverage_incomplete_deep_mesh_frame_count = 0
    scores: list[float] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        tier = str(frame.get("recommended_tier", "unknown"))
        by_tier[tier] += 1
        scores.append(float(frame.get("score", 0.0)))
        frame_reasons = [str(reason) for reason in frame.get("reasons", [])]
        if "reviewed_contact_targeted_body" in frame_reasons:
            targeted_reviewed_contact_frame_count += 1
        if tier == "deep_mesh" and "missing_expected_players" in frame_reasons:
            coverage_incomplete_deep_mesh_frame_count += 1
        by_reason.update(frame_reasons)
        for target in frame.get("player_targets", []):
            if isinstance(target, Mapping):
                by_player_target_representation[str(target.get("target_representation", "unknown"))] += 1
    deep_mesh_frame_count = sum(int(window["frame_count"]) for window in deep_mesh_windows)
    return {
        "by_tier": dict(sorted(by_tier.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_player_target_representation": dict(sorted(by_player_target_representation.items())),
        "max_score": max(scores, default=0.0),
        "deep_mesh_window_count": len(deep_mesh_windows),
        "deep_mesh_frame_count": deep_mesh_frame_count,
        "world_mesh_frame_count": deep_mesh_frame_count,
        "human_review_frame_count": by_tier.get("human_review", 0),
        "targeted_reviewed_contact_frame_count": targeted_reviewed_contact_frame_count,
        "coverage_incomplete_deep_mesh_frame_count": coverage_incomplete_deep_mesh_frame_count,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _clip_id_from_video(video: Path) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem.strip()).strip("._-")
    return slug or "clip"


def _parse_int_tuple(value: str | Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = [str(part) for part in value]
    parsed: list[int] = []
    for raw in raw_values:
        try:
            item = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of positive integers") from exc
        if item <= 0:
            raise ValueError(f"{name} must contain only positive integers")
        parsed.append(item)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive integer")
    return tuple(parsed)


def build_options_from_args(args: argparse.Namespace) -> PipelineOptions:
    video = Path(args.video).expanduser().resolve()
    clip = args.clip or _clip_id_from_video(video)
    run_dir = Path(args.out).expanduser().resolve() if args.out else DEFAULT_RUN_ROOT / f"process_video_{clip}"
    body_postchain_raw = args.body_postchain == "raw"
    mesh_byte_budget_mib = args.mesh_byte_budget_mib
    if args.target_mesh_frame_budget is None:
        target_mesh_frame_budget = DEFAULT_TARGET_MESH_FRAME_BUDGET
    else:
        target_mesh_frame_budget = None if args.target_mesh_frame_budget == 0 else args.target_mesh_frame_budget
        mesh_byte_budget_mib = None
    if mesh_byte_budget_mib is not None and mesh_byte_budget_mib <= 0.0:
        raise ValueError("--mesh-byte-budget-mib must be positive")

    remote_config = RemoteConfig(
        host=args.remote_host,
        ssh_key=args.remote_ssh_key,
        repo=args.remote_repo,
        python=args.remote_python,
        fast_sam_python=args.remote_fast_sam_python,
        fast_sam_root=args.remote_fast_sam_root,
        lock_wait_timeout_s=args.remote_lock_wait_timeout_s,
        command_timeout_s=args.remote_command_timeout_s,
        sam3d_body_input_size_px=args.sam3d_body_input_size_px,
        sam3d_crop_bucket_sizes=_parse_int_tuple(args.sam3d_crop_bucket_sizes, name="--sam3d-crop-bucket-sizes"),
        sam3d_torch_compile=not args.no_sam3d_torch_compile,
        sam3d_compile_warmup_buckets=_parse_int_tuple(
            args.sam3d_compile_warmup_buckets,
            name="--sam3d-compile-warmup-buckets",
        ),
        sam3d_skip_tier2_mesh_vertices=not args.serialize_tier2_mesh_vertices,
        sam3d_wrist_bone_lock=not (
            args.no_sam3d_wrist_bone_lock
            or args.no_body_wrist_lock
            or body_postchain_raw
        ),
        body_postchain_mode=str(args.body_postchain),
        body_temporal_smoothing=not (args.no_body_temporal_smoothing or body_postchain_raw),
        body_foot_lock=not (args.no_body_foot_lock or body_postchain_raw),
        body_foot_pin=not (args.no_body_foot_pin or body_postchain_raw),
        body_contact_splice=not (args.no_body_contact_splice or body_postchain_raw),
        body_world_joint_visual_smoothing=not (
            args.no_body_world_joint_visual_smoothing
            or body_postchain_raw
        ),
        fetch_body_monoliths=bool(args.fetch_body_monoliths),
        target_mesh_frame_budget=target_mesh_frame_budget,
        mesh_byte_budget_mib=mesh_byte_budget_mib,
    )

    return PipelineOptions(
        video=video,
        clip=clip,
        run_dir=run_dir,
        sport=args.sport,
        max_players=args.max_players,
        max_frames=args.max_frames,
        force=args.force,
        device=args.device,
        court_corners=Path(args.court_corners).expanduser().resolve() if args.court_corners else None,
        capture_sidecar=Path(args.capture_sidecar).expanduser().resolve() if args.capture_sidecar else None,
        court_keypoints=Path(args.court_keypoints).expanduser().resolve() if args.court_keypoints else None,
        court_calibration=Path(args.court_calibration).expanduser().resolve() if args.court_calibration else None,
        allow_auto_court_corners_preview=args.allow_auto_court_corners_preview,
        court_proposals_preview=args.court_proposals_preview,
        tracks_reuse=Path(args.tracks).expanduser().resolve() if args.tracks else None,
        global_association=not args.no_global_association,
        global_association_profile=args.global_association_profile,
        reid_model=Path(args.reid_model).expanduser().resolve(),
        ball_track_reuse=Path(args.ball_track).expanduser().resolve() if args.ball_track else None,
        ball_candidates_reuse=tuple(Path(path).expanduser().resolve() for path in (args.ball_candidates or [])),
        emit_ball_candidates=not args.no_ball_candidates,
        ball_track_auto_discovery=bool(args.allow_auto_ball_track) and not args.no_auto_ball_track,
        skip_ball=args.skip_ball,
        no_ball_arc=args.no_ball_arc,
        wasb_checkpoint=Path(args.wasb_checkpoint).expanduser().resolve(),
        wasb_repo=Path(args.wasb_repo).expanduser().resolve() if args.wasb_repo else None,
        skip_audio=args.skip_audio,
        rally_gating=args.rally_gating,
        placement_keypoints_2d=Path(args.placement_keypoints_2d).expanduser().resolve()
        if args.placement_keypoints_2d
        else None,
        camera_motion_path=Path(args.camera_motion).expanduser().resolve() if args.camera_motion else None,
        enable_camera_motion=bool(args.enable_camera_motion) and not bool(args.disable_camera_motion),
        skip_camera_motion=bool(args.disable_camera_motion),
        camera_motion_estimator=args.camera_motion_estimator,
        camera_motion_flow_backend=args.camera_motion_flow_backend,
        camera_motion_person_masks=not args.no_camera_motion_person_mask,
        placement_undistort=not args.no_placement_undistort,
        mesh_coverage_mode=args.mesh_coverage_mode,
        target_mesh_frame_budget=target_mesh_frame_budget,
        mesh_byte_budget_mib=mesh_byte_budget_mib,
        ball_proximity_m=args.ball_proximity_m,
        high_confidence_swing_floor=args.high_confidence_swing_floor,
        events_selected=Path(args.events_selected).expanduser().resolve() if args.events_selected else None,
        ball_track_arc_solved=Path(args.ball_track_arc_solved).expanduser().resolve() if args.ball_track_arc_solved else None,
        no_gpu=args.no_gpu,
        body_remote=not args.body_local,
        body_schedule=args.body_schedule,
        remote_config=remote_config,
        grounding_refine=not args.no_grounding_refine,
        paddle_pose=bool(DEFAULT_PADDLE_FUSED_ESTIMATOR.get("enabled", True)) and not args.no_paddle_pose,
        confidence_gate=not args.no_confidence_gate,
        confidence_calibration_curves=Path(args.confidence_calibration_curves).expanduser().resolve()
        if args.confidence_calibration_curves
        else None,
        scene_points=not args.no_scene_points,
        manifest_path=Path(args.manifest),
        tracker_config_path=Path(args.tracker_config),
        verify_viewer=args.verify_viewer,
        vite_allow_root=Path(args.vite_allow_root).expanduser().resolve() if args.vite_allow_root else ROOT,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-command pickleball pipeline: video -> scrubber-ready bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Source video file.")
    parser.add_argument("--court-corners", help="court_corners.json with declared image_size (manual 4-corner taps).")
    parser.add_argument("--capture-sidecar", help="Pre-built capture_sidecar.json (e.g. ARKit); alternative to --court-corners.")
    parser.add_argument("--court-keypoints", help="court_keypoints.json for the no-tap ARKit metric-calibration path (pairs with --capture-sidecar).")
    parser.add_argument(
        "--court-calibration",
        help=(
            "Pre-solved court_calibration.json-shaped artifact (e.g. the metric-15pt reviewed calibration from "
            "threed.racketsport.court_calibration_metric15) to consume directly instead of re-deriving one from "
            "--court-corners/--capture-sidecar. Schema and intrinsics.source are validated; only reviewed/real-"
            "metric sources are trusted (currently metric_15pt_reviewed). Takes precedence over "
            "--court-corners/--capture-sidecar when given explicitly; if none of the three are given, "
            "auto-discovered at <video_dir>/labels/court_calibration_metric15pt.json when present."
        ),
    )
    parser.add_argument(
        "--allow-auto-court-corners-preview",
        action="store_true",
        help=(
            "If no trusted calibration seed is supplied, try white-line court detection to build unverified "
            "preview corner taps. This is for one-button demo uploads and does not mark CAL verified."
        ),
    )
    parser.add_argument(
        "--court-proposals-preview",
        action="store_true",
        help=(
            "If no trusted calibration seed is supplied, write fail-closed court_proposals.json and a "
            "court correction task. Proposal preview does not satisfy calibration."
        ),
    )
    parser.add_argument("--clip", help="Clip id; defaults to the video filename stem.")
    parser.add_argument("--out", help="Run directory; defaults to runs/process_video_<clip>/.")
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    parser.add_argument("--max-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames processed (smoke runs).")
    parser.add_argument("--force", action="store_true", help="Redo every stage even if valid artifacts already exist.")
    parser.add_argument("--device", default=None, help="Device hint for tracking/pose (e.g. cuda:0).")

    parser.add_argument("--tracks", help="Reuse an already-computed tracks.json instead of running live tracking.")
    parser.add_argument("--no-global-association", action="store_true", help="Skip the raw-pool global-association refinement step after BoT-SORT loose-pool tracking.")
    parser.add_argument(
        "--global-association-profile",
        choices=sorted(RAW_POOL_GLOBAL_ASSOCIATION_PROFILES),
        default=None,
        help=(
            "Raw-pool association tuning profile. Default uses the manifest production profile; "
            "clip-specific eval/internal-val profiles require this explicit flag."
        ),
    )
    parser.add_argument("--reid-model", default=str(DEFAULT_REID_MODEL), help="OSNet ReID checkpoint for global association.")

    parser.add_argument("--ball-track", help="Reuse an already-computed ball_track.json instead of running WASB zero-shot.")
    parser.add_argument(
        "--ball-candidates",
        action="append",
        default=[],
        help=(
            "Reuse an existing racketsport_ball_candidates sidecar for the default ball_arc chain. "
            "Repeat for WASB + TrackNet sidecars."
        ),
    )
    parser.add_argument(
        "--no-ball-candidates",
        action="store_true",
        help="Do not emit top-K ball candidate sidecars during fresh WASB/TrackNet ball inference.",
    )
    parser.add_argument(
        "--allow-auto-ball-track",
        action="store_true",
        help="Opt into clip-id based discovery of existing precomputed ball_track.json artifacts under runs/ as low-confidence preview reuse.",
    )
    parser.add_argument("--no-auto-ball-track", action="store_true", help="Keep clip-id based precomputed ball_track.json discovery disabled.")
    parser.add_argument("--skip-ball", action="store_true", help="Skip the ball stage entirely.")
    parser.add_argument("--no-ball-arc", action="store_true", help="Skip the default auto-bounce + arc-solved 3D ball stage.")
    parser.add_argument("--wasb-checkpoint", default=str(DEFAULT_WASB_CHECKPOINT))
    parser.add_argument("--wasb-repo", default=str(DEFAULT_WASB_REPO), help="WASB-SBDT repo checkout.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio-onset extraction for contact-window fusion.")
    parser.add_argument("--rally-gating", action="store_true", help="Opt in to loose rally-span gating before frame/pose/body/world stages; preserves full pre-gating artifacts with *_pre_rally_gating.json copies.")
    parser.add_argument("--placement-keypoints-2d", default=None, help="Optional native/body keypoints_2d.json for the pre-BODY placement pass.")
    parser.add_argument("--camera-motion", default=None, help="Optional camera_motion.json mapping frame pixels into the calibration-reference frame before placement homography projection.")
    parser.add_argument(
        "--enable-camera-motion",
        action="store_true",
        help="Force the preview camera_motion.json estimation stage ON, bypassing the default motion-CONDITIONAL AUTO decision.",
    )
    parser.add_argument(
        "--disable-camera-motion",
        "--skip-camera-motion",
        dest="disable_camera_motion",
        action="store_true",
        help="Force-disable the preview camera_motion.json estimation stage; wins over --enable-camera-motion.",
    )
    parser.add_argument(
        "--camera-motion-estimator",
        choices=("hardened", "legacy"),
        default="hardened",
        help="Camera-motion estimator profile for the default camera_motion stage.",
    )
    parser.add_argument(
        "--camera-motion-flow-backend",
        choices=("lk", "raft-small"),
        default="lk",
        help="Camera-motion optical-flow backend. raft-small is flag-gated and requires already-cached weights; no download is attempted.",
    )
    parser.add_argument(
        "--no-camera-motion-person-mask",
        action="store_true",
        help="Disable person-track masking in the default camera_motion stage for ablation/debug runs.",
    )
    parser.add_argument("--no-placement-undistort", action="store_true", help="Disable placement-stage pixel undistortion before homography projection.")

    parser.add_argument(
        "--mesh-coverage-mode",
        choices=MESH_COVERAGE_MODES,
        default=DEFAULT_MESH_COVERAGE_MODE,
        help=(
            "Tier-1 (SAM-3D world_mesh) frame scheduling policy for frame_compute_plan.json. "
            "'ball_aware' (owner directive 2026-07-03) triggers mesh ONLY from physically-validated "
            "ball-arc-solver contacts (--events-selected), player-ball world proximity "
            "(--ball-track-arc-solved, within --ball-proximity-m), and high-confidence swing cues "
            "(contact_windows.json events at/above --high-confidence-swing-floor) -- never raw "
            "low-confidence wrist-cue windows."
        ),
    )
    parser.add_argument(
        "--target-mesh-frame-budget",
        type=int,
        default=None,
        help=(
            "Explicit fixed-frame deep-mesh (tier-1) budget for uniform/hybrid/ball_aware scheduling. "
            "Use 0 to mean 'no cap'. When omitted, the manifest byte-budget default is used."
        ),
    )
    parser.add_argument(
        "--mesh-byte-budget-mib",
        type=float,
        default=DEFAULT_MESH_BYTE_BUDGET_MIB,
        help=(
            "Deep-mesh byte budget in MiB. The no-flag default is manifest-owned; explicit "
            "--target-mesh-frame-budget switches back to fixed-frame policy."
        ),
    )
    parser.add_argument(
        "--ball-proximity-m",
        type=float,
        default=DEFAULT_BALL_PROXIMITY_M,
        help="ball_aware mode: player-to-arc-solved-ball horizontal world distance (m) that counts as a proximity trigger.",
    )
    parser.add_argument(
        "--high-confidence-swing-floor",
        type=float,
        default=DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR,
        help="ball_aware mode: contact_windows.json confidence floor for a 'high_confidence_swing' trigger.",
    )
    parser.add_argument(
        "--events-selected",
        help=(
            "events_selected.json from scripts/racketsport/solve_ball_arcs.py (physically-validated ball-arc "
            "contact events) for ball_aware mesh scheduling. Auto-discovered as events_selected.json in the "
            "clip dir when mesh-coverage-mode=ball_aware and this is omitted."
        ),
    )
    parser.add_argument(
        "--ball-track-arc-solved",
        help=(
            "ball_track_arc_solved.json from scripts/racketsport/solve_ball_arcs.py (arc-solved ball world "
            "track) for ball_aware mesh scheduling (player-ball proximity). Auto-discovered as "
            "ball_track_arc_solved.json in the clip dir when mesh-coverage-mode=ball_aware and this is omitted."
        ),
    )

    parser.add_argument("--no-gpu", action="store_true", help="Degrade to skeleton-only: skip BODY mesh, and skip live tracking/pose unless --tracks/--ball-track reuse artifacts are given.")
    parser.add_argument("--body-local", action="store_true", help="Run BODY in-process instead of dispatching to the remote A100 (use when already running on a GPU host).")
    parser.add_argument(
        "--body-schedule",
        choices=("serial", "overlap"),
        default="serial",
        help=(
            "serial (default): run ball/events before frames so cold-run mesh scheduling is contact-dense. "
            "overlap: once frames/tracks/calibration BODY inputs are ready, dispatch BODY on a background thread running the exact same stage code "
            "path while ball/ball_arc/events/ball_fill run on the main thread, then hard-join before "
            "placement_refine/grounding_refine/world. Reuse semantics are unchanged: a no-force-valid BODY "
            "artifact still takes the plain serial path with no thread spun up."
        ),
    )
    parser.add_argument("--remote-host", default=RemoteConfig().host)
    parser.add_argument("--remote-ssh-key", default=RemoteConfig().ssh_key)
    parser.add_argument("--remote-repo", default=RemoteConfig().repo)
    parser.add_argument("--remote-python", default=RemoteConfig().python)
    parser.add_argument("--remote-fast-sam-python", default=RemoteConfig().fast_sam_python)
    parser.add_argument("--remote-fast-sam-root", default=RemoteConfig().fast_sam_root)
    parser.add_argument(
        "--sam3d-body-input-size-px",
        type=int,
        default=RemoteConfig().sam3d_body_input_size_px,
        help="Bench config: Fast SAM-3D-Body input size option for the remote/body-mode run.",
    )
    parser.add_argument(
        "--sam3d-crop-bucket-sizes",
        default=",".join(str(value) for value in RemoteConfig().sam3d_crop_bucket_sizes),
        help="Bench config: comma-separated cross-frame crop bucket sizes, e.g. 8,16.",
    )
    parser.add_argument(
        "--no-sam3d-torch-compile",
        action="store_true",
        help="Bench config: disable torch.compile for the SAM3D body-mode decoder path.",
    )
    parser.add_argument(
        "--sam3d-compile-warmup-buckets",
        default=",".join(str(value) for value in RemoteConfig().sam3d_compile_warmup_buckets),
        help="Bench config: comma-separated bucket sizes to warm up before timing torch.compile.",
    )
    parser.add_argument(
        "--serialize-tier2-mesh-vertices",
        action="store_true",
        help="Debug override: serialize mesh vertices for tier-2 body-joint frames instead of joints-only output.",
    )
    parser.add_argument(
        "--fetch-body-monoliths",
        action="store_true",
        help="Download smpl_motion.json and body_mesh.json from remote BODY. Default skips them for speed.",
    )
    parser.add_argument(
        "--no-sam3d-wrist-bone-lock",
        action="store_true",
        help="Disable the default SAM-3D post-splice canonical lower-arm wrist lock.",
    )
    parser.add_argument(
        "--body-postchain",
        choices=("default", "raw"),
        default="default",
        help="BODY post-chain preset. raw disables all post-chain stages and persists body_raw_grounded_joints.json.",
    )
    parser.add_argument("--no-body-temporal-smoothing", action="store_true")
    parser.add_argument("--no-body-foot-lock", action="store_true")
    parser.add_argument("--no-body-foot-pin", action="store_true")
    parser.add_argument("--no-body-contact-splice", action="store_true")
    parser.add_argument("--no-body-wrist-lock", action="store_true")
    parser.add_argument("--no-body-world-joint-visual-smoothing", action="store_true")
    parser.add_argument("--remote-lock-wait-timeout-s", type=int, default=RemoteConfig().lock_wait_timeout_s)
    parser.add_argument(
        "--remote-command-timeout-s",
        type=int,
        default=RemoteConfig().command_timeout_s,
        help="Overall wall-clock budget for the remote BODY run itself (separate from the shared-GPU-lock wait).",
    )

    parser.add_argument(
        "--no-grounding-refine",
        action="store_true",
        help=(
            "Skip the default render-honest BODY grounding refinement between BODY/skeleton sync and "
            "world assembly."
        ),
    )
    parser.add_argument(
        "--no-paddle-pose",
        action="store_true",
        help=(
            "Skip the default fused wrist+palm+grip render-only paddle estimate stage before world assembly."
        ),
    )
    parser.add_argument("--no-confidence-gate", action="store_true", help="Skip the default Wave-B confidence gate and point the viewer manifest at raw virtual_world.json.")
    parser.add_argument("--no-scene-points", action="store_true", help="Do not generate replay_scene.json point GLBs for the viewer manifest.")
    parser.add_argument(
        "--confidence-calibration-curves",
        default=None,
        help=(
            "calibration_curves.json for Wave-B confidence bands. Defaults to calibration_curves.json near the run "
            f"or {DEFAULT_CONFIDENCE_CALIBRATION_CURVES} when present."
        ),
    )
    parser.add_argument("--manifest", default=str(orchestrator.DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--tracker-config", default=str(orchestrator.DEFAULT_BOTSORT_REID_CONFIG))
    parser.add_argument("--verify-viewer", action="store_true", help="Run a headless web-viewer load check after the manifest is built.")
    parser.add_argument("--vite-allow-root", default=None, help="Root directory the local Vite replay server is configured to serve (default: repo root).")
    parser.add_argument("--json", action="store_true", help="Print the full JSON summary instead of a human table.")
    return parser


def _print_human_summary(summary: Mapping[str, Any]) -> None:
    print("PIPELINE SUMMARY")
    print(f"status: {summary['status']}")
    print(f"clip: {summary['clip']}")
    print(f"wall_seconds: {summary['wall_seconds']}")
    print(f"clip_dir: {summary['clip_dir']}")
    print("stages:")
    for stage in summary["stages"]:
        badge = f" [{stage['trust_badge']}]" if stage.get("trust_badge") else ""
        print(f"- {stage['stage']}: {stage['status']}{badge} ({stage['wall_seconds']}s)")
        for note in stage["notes"]:
            print(f"    {note.splitlines()[0]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        options = build_options_from_args(args)
        pipeline = ProcessVideoPipeline(options)
        summary = pipeline.run()
    except Exception as exc:  # noqa: BLE001
        payload = {"schema_version": 1, "status": "failed", "error": str(exc)}
        print(json.dumps(payload, indent=2) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human_summary(summary)
    return 0 if summary["status"] in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
