"""Fail-closed racket-sport pipeline orchestration."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from scripts.racketsport.track import build_tracks

from .ball_stage_runner import BallStageRunner
from .body_compute import build_body_compute_execution, body_frame_batches_from_execution, write_body_compute_execution
from .body_full_clip_gate import build_body_full_clip_gate
from .body_grounding_quality import build_body_grounding_quality, write_body_grounding_quality
from .body_joint_quality import build_body_joint_quality
from .body_mesh_readiness import build_body_mesh_readiness
from .ball_inflections import build_ball_inflections_from_ball_track
from .court_auto_evidence import build_auto_court_line_evidence_from_frame, build_auto_court_line_evidence_from_video
from .court_calibration import calibration_from_manual_taps, calibration_image_size, metric_calibration_from_sidecar_and_keypoints
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
from .mesh_export import build_body_mesh_export
from .pose_fast import (
    PoseCropRequest,
    RTMW3DPoseRuntime,
    build_lane_a_skeleton3d_from_rtmw3d,
)
from .pose_temporal import (
    MOTIONBERT_CONFIG_ENV,
    MOTIONBERT_PROJECT_PYTHONPATH_ENV,
    MotionBERTTemporalRuntime,
    refine_lane_a_skeleton3d,
)
from .racket_stage_runner import RacketStageRunner
from .schemas import CaptureSidecar, CourtCalibration, StrictArtifact, Tracks, validate_artifact_file
from .virtual_world import build_virtual_world_state_from_files, write_virtual_world
from .wrist_velocity_peaks import build_wrist_velocity_peaks_from_file
from .worldhmr import build_body_artifacts_from_fast_sam


YOLO26M_MODEL_ID = "yolo26m"
DEFAULT_MODEL_MANIFEST = Path("models/MANIFEST.json")
DEFAULT_BODY_MODEL_MANIFEST = DEFAULT_BODY_MANIFEST_PATH
DEFAULT_BOTSORT_REID_CONFIG = Path("configs/racketsport/botsort_reid.yaml")
DEFAULT_BODY_MAX_ROOT_SPEED_MPS = 8.0
DEFAULT_BODY_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M = 0.75
FAST_SAM_PYTHON_ENV = "FAST_SAM_PYTHON"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
BODY_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "real_model": self.real_model,
            "source_mode": self.source_mode,
            "produced_artifacts": list(self.produced_artifacts),
            "notes": list(self.notes),
            "metrics": self.metrics,
        }


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
        else:
            calibration = calibration_from_manual_taps(sidecar_path, sport=context.sport)
            source_mode = self.source_mode
            calibration_note = "manual 4-corner calibration seed; requires human-reviewed corners for product verification"
        line_evidence, evidence_notes = _calibration_line_evidence(context, calibration=calibration, net_plane=net_plane)
        artifacts = {
            "court_calibration.json": calibration,
            "court_zones.json": build_court_zones(context.sport),
            "net_plane.json": net_plane,
            "court_line_evidence.json": line_evidence,
        }
        for filename, artifact in artifacts.items():
            _write_json_artifact(context.run_dir / filename, artifact)
        _raise_if_video_evidence_not_ready(context, line_evidence)

        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=source_mode,
            produced_artifacts=tuple(artifacts),
            notes=(
                calibration_note,
                *evidence_notes,
            ),
            metrics={
                "reprojection_median_px": calibration.reprojection_error_px.median,
                "reprojection_p95_px": calibration.reprojection_error_px.p95,
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
        imgsz: int = 960,
        conf: float = 0.18,
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

        _write_json_artifact(context.run_dir / "tracks.json", tracks)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("tracks.json",),
            notes=(
                "invoked manifest yolo26m checkpoint through Ultralytics model.track with BoT-SORT ReID enabled",
                "uses Ultralytics ReID model=auto appearance encoder; not a precomputed detections adapter",
            ),
            metrics={
                **counts,
                **raw_counts,
                **scale_counts,
                "checkpoint_sha256_verified": checkpoint.name,
                "tracker_config": str(tracker_config),
                "source_video": str(video_path),
            },
        )


class PoseStageRunner:
    stage = "pose"
    real_model = True
    source_mode = "rtmw3d_x_lane_a"

    def __init__(
        self,
        *,
        manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
        runtime: Any | None = None,
        motionbert_runtime: Any | None = None,
        model_id: str = "rtmw3d_x",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self._runtime = runtime
        self._motionbert_runtime = motionbert_runtime
        self.model_id = model_id

    def run(self, context: StageContext) -> StageRun:
        tracks = validate_artifact_file("tracks", context.run_dir / "tracks.json")
        calibration = validate_artifact_file("court_calibration", context.run_dir / "court_calibration.json")
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks.json did not validate as Tracks")
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        runtime = self._runtime or RTMW3DPoseRuntime(
            manifest_path=self.manifest_path,
            model_id=self.model_id,
            device=context.device,
        )
        pose_results: list[Any] = []
        inference_frame_count = 0
        player_frame_count = 0
        for frame_idx, frame_requests in _pose_frame_batches(tracks):
            image_path = _find_body_frame_image(context, frame_idx)
            requests = [
                PoseCropRequest(
                    frame_idx=frame_idx,
                    player_id=player_id,
                    bbox_xyxy=[float(value) for value in track_frame.bbox],
                    track_world_xy=[float(value) for value in track_frame.world_xy],
                    track_confidence=float(track_frame.conf),
                )
                for player_id, track_frame in frame_requests
            ]
            if not requests:
                continue
            frame_results = runtime.infer_frame(image_path, requests)
            pose_results.extend(frame_results)
            inference_frame_count += 1
            player_frame_count += len(requests)

        world_frame = "relative_only" if calibration.reprojection_error_px.median > 3.0 else "court_Z0"
        skeleton3d = build_lane_a_skeleton3d_from_rtmw3d(
            tracks,
            pose_results,
            world_frame=world_frame,
            source_model=self.model_id,
            calibration=calibration,
        )
        motionbert_runtime = self._motionbert_runtime
        if motionbert_runtime is None and _motionbert_runtime_is_configured():
            motionbert_runtime = MotionBERTTemporalRuntime(
                manifest_path=self.manifest_path,
                device=context.device,
            )
        skeleton3d = refine_lane_a_skeleton3d(skeleton3d, fps=tracks.fps, motionbert_runtime=motionbert_runtime)
        _write_json_artifact(context.run_dir / "skeleton3d.json", skeleton3d)
        notes = ["Lane A RTMW3D skeleton output covers all tracked player frames"]
        if world_frame == "relative_only":
            notes.append("calibration reprojection median exceeded 3 px; skeleton grounding downgraded to relative_only")
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("skeleton3d.json",),
            notes=tuple(notes),
            metrics={
                "source_model": self.model_id,
                "world_frame": world_frame,
                "inference_frame_count": inference_frame_count,
                "player_frame_count": player_frame_count,
                "joint_count": len(skeleton3d["joint_names"]),
                "preview_only": skeleton3d["preview_only"],
                "temporal_refined": True,
                "motionbert": skeleton3d["provenance"]["temporal_refine"]["motionbert"],
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
        detector_name: str = "yolo",
        fov_name: str = "moge2",
        fallback_pose_runtime: Any | None = None,
        fallback_pose_manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
        fallback_pose_model_id: str = "rtmw3d_x",
        smoothing_alpha: float = 1.0,
        max_root_speed_mps: float | None = DEFAULT_BODY_MAX_ROOT_SPEED_MPS,
        max_track_anchor_smoothing_residual_m: float | None = DEFAULT_BODY_MAX_TRACK_ANCHOR_SMOOTHING_RESIDUAL_M,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.fast_sam_repo = Path(fast_sam_repo)
        self._runtime = runtime
        self.detector_name = detector_name
        self.fov_name = fov_name
        self._fallback_pose_runtime = fallback_pose_runtime
        self.fallback_pose_manifest_path = Path(fallback_pose_manifest_path)
        self.fallback_pose_model_id = fallback_pose_model_id
        self.smoothing_alpha = smoothing_alpha
        self.max_root_speed_mps = max_root_speed_mps
        self.max_track_anchor_smoothing_residual_m = max_track_anchor_smoothing_residual_m

    def run(self, context: StageContext) -> StageRun:
        body_wall_start = time.perf_counter()
        tracks = validate_artifact_file("tracks", context.run_dir / "tracks.json")
        calibration = validate_artifact_file("court_calibration", context.run_dir / "court_calibration.json")
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks.json did not validate as Tracks")
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("court_calibration.json did not validate as CourtCalibration")

        lane_b_plan = _ensure_lane_b_frame_plan_from_lane_a(context, tracks)
        body_execution = build_body_compute_execution(
            tracks,
            frame_plan_path=context.run_dir / "frame_compute_plan.json",
            max_frames=context.max_frames,
        )
        write_body_compute_execution(context.run_dir / "body_compute_execution.json", body_execution)
        frame_batches = body_frame_batches_from_execution(tracks, body_execution)
        if not frame_batches:
            raise ValueError("adaptive BODY schedule contains no world_mesh frames")

        required_model_ids = fast_sam_required_model_ids(detector_name=self.detector_name, fov_name=self.fov_name)
        assets = verify_fast_sam_manifest_assets(self.manifest_path, required_model_ids=required_model_ids)
        fast_sam_runtime_unavailable_note = ""
        runtime = self._runtime
        fast_sam_python = os.environ.get(FAST_SAM_PYTHON_ENV, "").strip()
        fast_sam_runtime_mode = "injected" if runtime is not None else ("subprocess" if fast_sam_python else "in_process")
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
                        work_dir=context.run_dir / "fast_sam_subprocess",
                    )
                else:
                    runtime = FastSam3DBodyRuntime(
                        assets=assets,
                        fast_sam_repo=self.fast_sam_repo,
                        detector_name=self.detector_name,
                        fov_name=self.fov_name,
                    )
            except RuntimeError as exc:
                fast_sam_runtime_mode = "unavailable"
                fast_sam_runtime_unavailable_note = (
                    "Fast SAM-3D-Body runtime unavailable; using RTMW3D contact fallback: "
                    f"{exc}"
                )
        samples: list[dict[str, Any]] = []
        fallback_pose_runtime = self._fallback_pose_runtime
        fallback_pose_results: list[Any] = []
        mesh_requests: list[dict[str, Any]] = []
        for frame_idx, frame_requests in frame_batches:
            image_path = _find_body_frame_image(context, frame_idx)
            track_bboxes = [list(track_frame.bbox) for _, track_frame in frame_requests]
            image_size = _image_size_from_body_frame_or_calibration(
                image_path,
                calibration=calibration,
                bboxes=track_bboxes,
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
                mesh_requests.append(
                    {
                        "frame_idx": frame_idx,
                        "player_id": player_id,
                        "track_frame": track_frame,
                        "bbox": bbox,
                        "image_path": image_path,
                        "image_size": image_size,
                    }
                )

        batched_outputs: list[list[dict[str, Any]]] | None = None
        batch_runner = getattr(runtime, "process_frame_batches", None) if runtime is not None else None
        if callable(batch_runner) and mesh_requests:
            batched_outputs = batch_runner(
                [(request["image_path"], [request["bbox"]]) for request in mesh_requests]
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
                else ([] if runtime is None else runtime.process_frame(image_path, bboxes_xyxy=[bbox]))
            )
            if not raw_outputs:
                if fallback_pose_runtime is None:
                    fallback_pose_runtime = RTMW3DPoseRuntime(
                        manifest_path=self.fallback_pose_manifest_path,
                        model_id=self.fallback_pose_model_id,
                        device=context.device,
                    )
                fallback_pose_results.extend(
                    fallback_pose_runtime.infer_frame(
                        image_path,
                        [
                            PoseCropRequest(
                                frame_idx=frame_idx,
                                player_id=player_id,
                                bbox_xyxy=[float(value) for value in bbox],
                                track_world_xy=[float(value) for value in track_frame.world_xy],
                                track_confidence=float(track_frame.conf),
                            )
                        ],
                    )
                )
                continue
            raw_output = _match_body_outputs(raw_outputs, [bbox])[0]
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
            samples.append(sample)

        body_pose_fallback = None
        if fallback_pose_results:
            body_pose_fallback = _build_body_pose_fallback_skeleton(
                tracks,
                fallback_pose_results,
                calibration=calibration,
                model_id=self.fallback_pose_model_id,
            )
            _write_json_artifact(context.run_dir / "body_pose_fallback.json", body_pose_fallback)

        if samples:
            smpl_motion, skeleton3d, grounding_metrics = build_body_artifacts_from_fast_sam(
                samples,
                calibration=calibration,
                fps=tracks.fps,
                smoothing_alpha=self.smoothing_alpha,
                max_root_speed_mps=self.max_root_speed_mps,
                max_track_anchor_smoothing_residual_m=self.max_track_anchor_smoothing_residual_m,
            )
        else:
            smpl_motion = _empty_smpl_motion(fps=tracks.fps)
            skeleton3d = _empty_body_preview_skeleton(fps=tracks.fps)
            grounding_metrics = {
                "body_samples": 0,
                "players": 0,
                "frames": 0,
                "world_frame": "court_Z0",
                "grounding": "no_fast_sam_mesh_samples_pose_fallback_only",
            }
        _write_json_artifact(context.run_dir / "smpl_motion.json", smpl_motion)
        body_grounding_quality = build_body_grounding_quality(
            clip=context.clip,
            grounding_metrics=grounding_metrics,
        )
        write_body_grounding_quality(context.run_dir / "body_grounding_quality.json", body_grounding_quality)
        smpl_motion_payload = smpl_motion.model_dump(mode="json") if hasattr(smpl_motion, "model_dump") else smpl_motion
        skeleton3d_path = context.run_dir / "skeleton3d.json"
        existing_skeleton3d = _read_optional_json(skeleton3d_path)
        preserved_lane_a_skeleton = (
            isinstance(existing_skeleton3d, Mapping)
            and existing_skeleton3d.get("preview_only") is False
        )
        if preserved_lane_a_skeleton:
            skeleton3d_payload = existing_skeleton3d
        else:
            _write_json_artifact(skeleton3d_path, skeleton3d)
            skeleton3d_payload = skeleton3d.model_dump(mode="json") if hasattr(skeleton3d, "model_dump") else skeleton3d
        body_mesh = build_body_mesh_export(
            smpl_motion_payload,
            clip=context.clip,
            body_compute_execution=body_execution,
        )
        _write_json_artifact(context.run_dir / "body_mesh.json", body_mesh)
        skeleton3d_payload, contact_splice = splice_contact_skeleton_with_body_mesh(
            skeleton3d_payload,
            body_mesh=body_mesh,
            body_compute_execution=body_execution,
            fallback_skeleton3d=body_pose_fallback,
        )
        _write_json_artifact(skeleton3d_path, skeleton3d_payload)
        _write_json_artifact(context.run_dir / "contact_splice.json", contact_splice)
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
        _write_json_artifact(context.run_dir / "body_mesh_readiness.json", body_mesh_readiness)
        produced_artifacts = (
            "body_compute_execution.json",
            "smpl_motion.json",
            "body_mesh.json",
            "contact_splice.json",
            "skeleton3d.json",
            "body_mesh_readiness.json",
            "body_joint_quality.json",
            "body_full_clip_gate.json",
            "body_grounding_quality.json",
        )
        if body_pose_fallback is not None:
            produced_artifacts = (*produced_artifacts, "body_pose_fallback.json")
        body_pose_fallback_frame_count = _skeleton_player_frame_count(body_pose_fallback)
        notes = [
            "Fast SAM-3D-Body runtime output converted to court/world coordinates with court_calibration.json",
            "BODY frame execution follows frame_compute_plan.json when present and skips manual-review/preview-only frames",
            "preserved existing Lane A skeleton3d.json" if preserved_lane_a_skeleton else "wrote mesh-derived preview skeleton3d.json",
            "spliced scheduled hitter mesh joints into Lane A skeleton3d.json at contact frames",
            (
                "used RTMW3D high-resolution crop fallback where contact mesh was unavailable"
                if body_pose_fallback_frame_count
                else "RTMW3D contact fallback was not needed"
            ),
            "BODY artifacts are real runner outputs; BODY accuracy gate still requires labeled world-MPJPE evaluation",
        ]
        if fast_sam_runtime_unavailable_note:
            notes.append(fast_sam_runtime_unavailable_note)
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=produced_artifacts,
            notes=tuple(notes),
            metrics={
                **grounding_metrics,
                "body_compute_mode": body_execution["mode"],
                "lane_b_frame_plan_source": lane_b_plan["source"],
                "lane_b_contact_event_count": lane_b_plan["contact_event_count"],
                "lane_b_frame_plan_generated_artifacts": lane_b_plan["generated_artifacts"],
                "scheduled_body_frames": body_execution["summary"]["scheduled_frame_count"],
                "scheduled_body_player_frames": body_execution["summary"]["scheduled_player_frame_count"],
                "body_mesh_frame_count": body_mesh["summary"]["mesh_frame_count"],
                "body_pose_fallback_frame_count": body_pose_fallback_frame_count,
                "fast_sam_runtime_mode": fast_sam_runtime_mode,
                "fast_sam_python": fast_sam_python,
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


def _pose_frame_batches(tracks: Tracks) -> list[tuple[int, list[tuple[int, Any]]]]:
    by_frame: dict[int, list[tuple[int, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(float(frame.t) * tracks.fps))
            by_frame.setdefault(frame_idx, []).append((int(player.id), frame))
    return [
        (frame_idx, sorted(active, key=lambda item: item[0]))
        for frame_idx, active in sorted(by_frame.items())
    ]


def _motionbert_runtime_is_configured() -> bool:
    return bool(os.environ.get(MOTIONBERT_CONFIG_ENV) or os.environ.get(MOTIONBERT_PROJECT_PYTHONPATH_ENV))


def _build_body_pose_fallback_skeleton(
    tracks: Tracks,
    pose_results: Sequence[Any],
    *,
    calibration: CourtCalibration,
    model_id: str,
) -> dict[str, Any]:
    fallback_tracks = _tracks_subset_for_pose_results(tracks, pose_results)
    world_frame = "relative_only" if calibration.reprojection_error_px.median > 3.0 else "court_Z0"
    skeleton = build_lane_a_skeleton3d_from_rtmw3d(
        fallback_tracks,
        pose_results,
        world_frame=world_frame,
        source_model=f"{model_id}_body_fallback",
        calibration=calibration,
    )
    provenance = dict(skeleton.get("provenance", {}))
    provenance["lane"] = "B_pose_fallback"
    provenance["fallback_for"] = "lane_b_contact_mesh_unavailable"
    skeleton["provenance"] = provenance
    return skeleton


def _tracks_subset_for_pose_results(tracks: Tracks, pose_results: Sequence[Any]) -> Tracks:
    result_keys = {(int(result.frame_idx), int(result.player_id)) for result in pose_results}
    payload = tracks.model_dump(mode="json")
    players: list[dict[str, Any]] = []
    for player in payload.get("players", []):
        player_id = int(player["id"])
        frames = [
            frame
            for frame in player.get("frames", [])
            if (int(round(float(frame["t"]) * tracks.fps)), player_id) in result_keys
        ]
        if frames:
            players.append({**player, "frames": frames})
    return Tracks.model_validate({**payload, "players": players})


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


def _ensure_lane_b_frame_plan_from_lane_a(context: StageContext, tracks: Tracks) -> dict[str, Any]:
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
            "source": "missing_lane_a_skeleton3d",
            "contact_event_count": 0,
            "generated_artifacts": generated_artifacts,
            "notes": ["frame_compute_plan.json not derived because skeleton3d.json is missing"],
        }
    skeleton_payload = _read_json(skeleton_path)
    if not _is_real_lane_a_skeleton3d(skeleton_payload):
        return {
            "source": "invalid_lane_a_skeleton3d",
            "contact_event_count": 0,
            "generated_artifacts": generated_artifacts,
            "notes": ["frame_compute_plan.json not derived because skeleton3d.json is not a real Lane A skeleton"],
        }

    wrist_path = _first_existing_artifact(context, "wrist_velocity_peaks.json")
    wrist_payload = _read_json(wrist_path) if wrist_path is not None else None
    if not _is_current_lane_a_wrist_velocity_peaks(wrist_payload, skeleton_path=skeleton_path):
        wrist_payload = build_wrist_velocity_peaks_from_file(skeleton_path, require_lane_a=True)
        wrist_path = context.run_dir / "wrist_velocity_peaks.json"
        _write_json(wrist_path, wrist_payload)
        generated_artifacts.append("wrist_velocity_peaks.json")
        notes.append("derived wrist_velocity_peaks.json from Lane A skeleton3d.json")
    elif wrist_path.parent != context.run_dir:
        wrist_path = context.run_dir / "wrist_velocity_peaks.json"
        _write_json(wrist_path, wrist_payload)
        generated_artifacts.append("wrist_velocity_peaks.json")
        notes.append("copied Lane A wrist_velocity_peaks.json into run output")

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
        notes.append("derived ball_inflections.json from ball_track.json for Lane B contact scheduling")

    contact_windows_path = _first_existing_artifact(context, "contact_windows.json")
    if contact_windows_path is None:
        contact_windows = fuse_contact_windows_from_cue_payloads(
            fps=tracks.fps,
            audio_onsets_payload=[],
            wrist_velocity_peaks_payload=_read_json(wrist_path),
            ball_inflections_payload=_read_json(ball_inflections_path),
            require_audio=False,
        )
        contact_windows_path = context.run_dir / "contact_windows.json"
        _write_json(contact_windows_path, contact_windows)
        generated_artifacts.append("contact_windows.json")
        notes.append("fused contact_windows.json from Lane A wrists and ball inflections")
    else:
        contact_windows = _read_json(contact_windows_path)
        if contact_windows_path.parent != context.run_dir:
            contact_windows_path = context.run_dir / "contact_windows.json"
            _write_json(contact_windows_path, contact_windows)
            generated_artifacts.append("contact_windows.json")
            notes.append("copied contact_windows.json into run output for Lane B scheduling")

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
        "source": "lane_a_wrist_velocity_peaks",
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


def _is_real_lane_a_skeleton3d(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    provenance = payload.get("provenance")
    return (
        payload.get("artifact_type") == "racketsport_skeleton3d"
        and payload.get("preview_only") is False
        and isinstance(provenance, Mapping)
        and provenance.get("lane") == "A"
    )


def _is_current_lane_a_wrist_velocity_peaks(payload: Any, *, skeleton_path: Path) -> bool:
    if not isinstance(payload, Mapping):
        return False
    source_provenance = payload.get("source_provenance")
    return (
        payload.get("artifact_type") == "racketsport_wrist_velocity_peaks"
        and payload.get("source") == "lane_a_skeleton3d_world_joints"
        and isinstance(source_provenance, Mapping)
        and source_provenance.get("lane") == "A"
        and source_provenance.get("preview_only") is False
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
) -> dict[str, Any]:
    """Run the pipeline through ``stage`` and stop rather than fabricate artifacts."""

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
        runner = registry.get(contract.stage)
        if runner is None:
            stage_runs.append(_blocked_stage(contract, f"no runner registered for stage: {contract.stage}"))
            summary_status = PIPELINE_STATUS_BLOCKED
            break

        try:
            result = runner.run(context)
            _validate_contract_artifacts(contract, run_path)
        except Exception as exc:
            stage_runs.append(
                StageRun(
                    stage=contract.stage,
                    status=PIPELINE_STATUS_FAIL,
                    real_model=getattr(runner, "real_model", False),
                    source_mode=getattr(runner, "source_mode", "unknown"),
                    notes=(f"{contract.stage} failed: {exc}",),
                ).as_dict()
            )
            summary_status = PIPELINE_STATUS_FAIL
            break
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


def _validate_contract_artifacts(contract: PipelineStageContract, run_dir: Path) -> None:
    for artifact in contract.required_artifacts:
        schema_name = ARTIFACT_SCHEMA_BY_FILENAME.get(artifact)
        if schema_name is None:
            raise ValueError(f"no schema mapping for required artifact: {artifact}")
        validate_artifact_file(schema_name, run_dir / artifact)


def _blocked_stage(contract: PipelineStageContract, note: str) -> dict[str, Any]:
    return StageRun(
        stage=contract.stage,
        status=PIPELINE_STATUS_BLOCKED,
        real_model=False,
        source_mode="unregistered",
        notes=(note,),
    ).as_dict()


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
    contact_windows_path = _existing_file(context.run_dir / "contact_windows.json")
    racket_path = _existing_file(context.run_dir / "racket_pose.json")
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

        if "body_compute_execution.json" in protected and (context.run_dir / "body_compute_execution.json").is_file():
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
                write_body_compute_execution(context.run_dir / "body_compute_execution.json", body_execution)
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


def _raise_if_video_evidence_not_ready(context: StageContext, evidence: Any) -> None:
    if _calibration_video_path(context) is None:
        return
    aggregate = getattr(evidence, "aggregate", None)
    if aggregate is None or getattr(aggregate, "auto_calibration_ready", False):
        return
    reasons = ", ".join(getattr(aggregate, "reasons", []) or ["unknown"])
    raise ValueError(f"automatic court evidence not ready for video-backed run: {reasons}")


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
    frame_size = _read_image_size(image_path)
    if frame_size is not None:
        return frame_size
    return _image_size_from_calibration_and_bboxes(calibration, bboxes)


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
