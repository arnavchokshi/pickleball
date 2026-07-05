"""BALL StageRunner integration for the current no-click prototype track."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .ball_physics3d import (
    BallSample3D,
    BounceArcReconstruction,
    detect_bounce_events,
    project_bounces_to_ball_track,
    reconstruct_bounce_arcs_from_image_track,
)
from .ball_inflections import build_ball_inflections_from_ball_track
from .contact_windows import build_contact_windows_artifact
from .event_fusion import fuse_contact_windows_from_cue_payloads
from .schemas import BallTrack, ContactWindows
from .wrist_velocity_peaks import build_wrist_velocity_peaks_from_file


DEFAULT_NO_CLICK_BALL_FILENAME = "ball_track_fusion_temporal_vball100_localtraj.json"
DEFAULT_SELECTED_TRACKS_DIR = "selected_tracks"
DEFAULT_SELECTED_BALL_FILENAME = "ball_track.json"
DEFAULT_TRACKNET_SMOKE_DIR = "tracknet_smoke_0000_0010"
DEFAULT_PROTOTYPE_GATE_ROOT = Path("runs/eval0/prototype_gate_h100_v2")
DEFAULT_AUDIO_ONSETS_FILENAME = "audio_onsets.json"
DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME = "wrist_velocity_peaks.json"
DEFAULT_BALL_INFLECTIONS_FILENAME = "ball_inflections.json"
CONTACT_FUSION_MODE_AUDIO_WRIST_BALL = "audio_wrist_ball"
CONTACT_FUSION_MODE_WRIST_BALL = "wrist_ball"
TOTNET_PREDICTIONS_FILENAME = "totnet_predictions.json"
TOTNET_RUN_METADATA_FILENAME = "totnet_run.json"
TRACKNET_RUN_METADATA_FILENAME = "tracknet_metadata.json"
TRACKNET_PREDICTION_DIR = "tracknet_predictions"
TRACKNET_RAW_BALL_FILENAME = "ball_track_tracknet_raw.json"
DEFAULT_TRACKNET_HEATMAP_VISIBLE_THRESHOLD = 0.5
BALL_LOCAL_SEARCH_SUMMARY_FILENAME = "ball_local_search_summary.json"
BALL_PHYSICS3D_SUMMARY_FILENAME = "ball_physics3d_summary.json"
BALL_BOUNCE_MIN_VERTICAL_SPEED_MPS = 0.0
BALL_BOUNCE_MIN_SEPARATION_S = 0.05

ModelRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class BallStageRun:
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


class BallStageRunner:
    stage = "ball_events"
    real_model = False
    source_mode = "no_click_fusion_temporal_vball100_localtraj"

    def __init__(
        self,
        *,
        source_path: str | Path | None = None,
        prototype_root: str | Path = DEFAULT_PROTOTYPE_GATE_ROOT,
        allow_prototype_root_fallback: bool = False,
        totnet_repo: str | Path | None = None,
        totnet_checkpoint: str | Path | None = None,
        video_path: str | Path | None = None,
        model_runner: ModelRunner | None = None,
        tracknet_repo: str | Path | None = None,
        tracknet_file: str | Path | None = None,
        inpaintnet_file: str | Path | None = None,
        tracknet_runner: ModelRunner | None = None,
        tracknet_fps: float | None = None,
        tracknet_large_video: bool = False,
        tracknet_confidence_mode: str = "heatmap_peak",
        tracknet_heatmap_visible_threshold: float = DEFAULT_TRACKNET_HEATMAP_VISIBLE_THRESHOLD,
        emit_ball_candidates: bool = True,
        ball_candidate_top_k: int = 5,
        tracknet_local_search: bool = False,
        tracknet_local_search_court_filter: bool = False,
        local_search_runner: ModelRunner | None = None,
        ball_physics3d: bool = False,
        contact_fusion_mode: str = CONTACT_FUSION_MODE_WRIST_BALL,
        confidence_threshold: float = 0.0,
        batch_size: int = 8,
    ) -> None:
        self.source_path = Path(source_path) if source_path is not None else None
        self.prototype_root = Path(prototype_root)
        self.allow_prototype_root_fallback = allow_prototype_root_fallback
        self.totnet_repo = Path(totnet_repo) if totnet_repo is not None else None
        self.totnet_checkpoint = Path(totnet_checkpoint) if totnet_checkpoint is not None else None
        self.video_path = Path(video_path) if video_path is not None else None
        self.model_runner = model_runner
        self.tracknet_repo = Path(tracknet_repo) if tracknet_repo is not None else None
        self.tracknet_file = Path(tracknet_file) if tracknet_file is not None else None
        self.inpaintnet_file = Path(inpaintnet_file) if inpaintnet_file is not None else None
        self.tracknet_runner = tracknet_runner
        self.tracknet_fps = float(tracknet_fps) if tracknet_fps is not None else None
        self.tracknet_large_video = bool(tracknet_large_video)
        if tracknet_confidence_mode not in {"legacy_visibility", "heatmap_peak"}:
            raise ValueError("tracknet_confidence_mode must be legacy_visibility or heatmap_peak")
        self.tracknet_confidence_mode = tracknet_confidence_mode
        self.tracknet_heatmap_visible_threshold = float(tracknet_heatmap_visible_threshold)
        self.emit_ball_candidates = bool(emit_ball_candidates)
        self.ball_candidate_top_k = int(ball_candidate_top_k)
        self.tracknet_local_search = bool(tracknet_local_search)
        self.tracknet_local_search_court_filter = bool(tracknet_local_search_court_filter)
        self.local_search_runner = local_search_runner
        self.ball_physics3d = bool(ball_physics3d)
        if contact_fusion_mode not in {CONTACT_FUSION_MODE_AUDIO_WRIST_BALL, CONTACT_FUSION_MODE_WRIST_BALL}:
            raise ValueError(
                f"unknown BALL contact_fusion_mode: {contact_fusion_mode}; "
                f"expected {CONTACT_FUSION_MODE_AUDIO_WRIST_BALL} or {CONTACT_FUSION_MODE_WRIST_BALL}"
            )
        self.contact_fusion_mode = contact_fusion_mode
        self.confidence_threshold = float(confidence_threshold)
        self.batch_size = int(batch_size)
        if self._totnet_configured and self._tracknet_configured:
            raise ValueError("configure only one BALL model backend: TOTNet or TrackNetV3")
        self.real_model = self._totnet_configured or self._tracknet_configured
        if self._totnet_configured:
            self.source_mode = "totnet_inference"
        elif self._tracknet_configured:
            self.source_mode = "tracknetv3_inference"
        else:
            self.source_mode = type(self).source_mode

    def run(self, context: Any) -> BallStageRun:
        if self._totnet_configured:
            ball_payload, source_path, source_mode, model_metrics = self._run_totnet_inference(context)
            selection = None
        elif self._tracknet_configured:
            ball_payload, source_path, source_mode, model_metrics = self._run_tracknet_inference(context)
            selection = None
        else:
            source_path = self._resolve_source_path(context)
            if source_path.name == "ball_points.json":
                raise ValueError("BALL StageRunner refuses to consume ball_points.json")

            ball_payload = _read_json(source_path)
            source_mode = _source_mode_for_path(source_path)
            selection = _selection_metadata_for_track(source_path)
            model_metrics = {}
            if source_mode == "selected_ball_track_prototype" and selection is None:
                raise ValueError(f"missing selected-track metadata sidecar: {source_path.parent / 'ball_track_selection.json'}")
        ball_track = BallTrack.model_validate(ball_payload)
        if ball_track.source == "tap":
            raise ValueError("BALL StageRunner refuses to consume tap/manual ball tracks")
        physics_summary: dict[str, Any] | None = None
        physics_notes: tuple[str, ...] = ()
        if self.ball_physics3d:
            ball_payload, physics_summary, physics_notes = _apply_ball_physics3d(
                ball_payload,
                summary_path=context.run_dir / BALL_PHYSICS3D_SUMMARY_FILENAME,
                source_path=source_path,
                context=context,
            )
            ball_track = BallTrack.model_validate(ball_payload)

        generated_ball_inflections, ball_inflection_notes = _derive_ball_inflections_from_current_track(
            context,
            ball_payload=ball_payload,
            enabled=self.real_model,
        )
        generated_wrist_peaks, wrist_peak_notes = _derive_wrist_velocity_peaks_from_current_skeleton(
            context,
            enabled=self.real_model,
        )
        contact_payload, contact_notes = _contact_windows_from_cues(
            context,
            fps=ball_track.fps,
            contact_fusion_mode=self.contact_fusion_mode,
        )
        ContactWindows.model_validate(contact_payload)

        _write_json(context.run_dir / "ball_track.json", ball_payload)
        _write_json(context.run_dir / "contact_windows.json", contact_payload)

        visible_count = sum(1 for frame in ball_track.frames if frame.visible)
        approx_count = sum(1 for frame in ball_track.frames if frame.approx)
        contact_event_count = len(contact_payload.get("events", []))
        metrics = {
            "source_ball_track": str(source_path),
            "frame_count": len(ball_track.frames),
            "visible_frame_count": visible_count,
            "invisible_frame_count": len(ball_track.frames) - visible_count,
            "approx_frame_count": approx_count,
            "bounce_count": len(ball_track.bounces),
            "contact_event_count": contact_event_count,
            "uses_human_clicks": False,
            "not_gate_verified": True,
            "contact_fusion_mode": self.contact_fusion_mode,
        }
        produced_artifacts = self._produced_artifacts
        if generated_ball_inflections is not None:
            metrics["ball_inflection_candidate_count"] = generated_ball_inflections["summary"]["candidate_count"]
            metrics["ball_inflection_source"] = generated_ball_inflections["source"]
            produced_artifacts = (*produced_artifacts, DEFAULT_BALL_INFLECTIONS_FILENAME)
        if generated_wrist_peaks is not None:
            metrics["wrist_velocity_peak_count"] = generated_wrist_peaks["summary"]["peak_count"]
            metrics["wrist_velocity_source"] = generated_wrist_peaks["source"]
            produced_artifacts = (*produced_artifacts, DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME)
        metrics.update(model_metrics)
        if physics_summary is not None:
            metrics["physics3d"] = _physics3d_metrics(physics_summary)
        if selection is not None:
            metrics["selection"] = selection
        status = "ran" if contact_event_count else "blocked"
        blocked_notes: tuple[str, ...] = ()
        if status == "blocked":
            blocked_notes = (
                "BALL contact windows are empty; downstream stages remain blocked until trusted cue fusion produces contacts",
            )
        return BallStageRun(
            stage=self.stage,
            status=status,
            real_model=self.real_model,
            source_mode=source_mode,
            produced_artifacts=produced_artifacts,
            notes=(
                *self._source_notes(source_path),
                *physics_notes,
                *ball_inflection_notes,
                *wrist_peak_notes,
                *contact_notes,
                *blocked_notes,
                "real BALL model inference output; not a BALL VERIFIED accuracy gate",
            ),
            metrics=metrics,
        )

    @property
    def _totnet_configured(self) -> bool:
        return self.totnet_repo is not None or self.totnet_checkpoint is not None or self.model_runner is not None

    @property
    def _tracknet_configured(self) -> bool:
        return self.tracknet_repo is not None or self.tracknet_file is not None or self.inpaintnet_file is not None

    @property
    def _produced_artifacts(self) -> tuple[str, ...]:
        if self._totnet_configured:
            artifacts = (
                "ball_track.json",
                "contact_windows.json",
                TOTNET_PREDICTIONS_FILENAME,
                TOTNET_RUN_METADATA_FILENAME,
            )
        elif self._tracknet_configured:
            if self.tracknet_local_search:
                artifacts = (
                    "ball_track.json",
                    "contact_windows.json",
                    TRACKNET_RUN_METADATA_FILENAME,
                    TRACKNET_RAW_BALL_FILENAME,
                    BALL_LOCAL_SEARCH_SUMMARY_FILENAME,
                )
            else:
                artifacts = ("ball_track.json", "contact_windows.json", TRACKNET_RUN_METADATA_FILENAME)
        else:
            artifacts = ("ball_track.json", "contact_windows.json")
        if self.ball_physics3d:
            artifacts = (*artifacts, BALL_PHYSICS3D_SUMMARY_FILENAME)
        return artifacts

    def _source_notes(self, source_path: Path) -> tuple[str, ...]:
        if self._totnet_configured:
            return ("ran TOTNet video inference locally",)
        if self._tracknet_configured:
            notes = ["ran TrackNetV3 video inference locally"]
            if self.tracknet_local_search:
                notes.append("applied TrackNetV3 local-search postprocess")
            return tuple(notes)
        return (_source_note_for_path(source_path),)

    def _run_totnet_inference(self, context: Any) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
        if self.totnet_repo is None or self.totnet_checkpoint is None:
            raise ValueError("TOTNet BALL inference requires totnet_repo and totnet_checkpoint")
        _require_dir(self.totnet_repo / "src", "TOTNet src")
        _require_file(self.totnet_checkpoint, "TOTNet checkpoint")
        video = _resolve_video_path(context, explicit=self.video_path)
        runner = self.model_runner or _default_totnet_runner()
        out = context.run_dir / "ball_track.json"
        predictions_out = context.run_dir / TOTNET_PREDICTIONS_FILENAME
        metadata_out = context.run_dir / TOTNET_RUN_METADATA_FILENAME
        summary = runner(
            video=video,
            totnet_repo=self.totnet_repo,
            checkpoint=self.totnet_checkpoint,
            out=out,
            predictions_out=predictions_out,
            metadata_out=metadata_out,
            confidence_threshold=self.confidence_threshold,
            batch_size=self.batch_size,
            device=getattr(context, "device", None),
            max_frames=getattr(context, "max_frames", None),
        )
        payload = _read_json(out)
        model_metrics = _totnet_model_metrics(summary)
        return payload, out, "totnet_inference", model_metrics

    def _run_tracknet_inference(self, context: Any) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
        if self.tracknet_repo is None or self.tracknet_file is None or self.inpaintnet_file is None:
            raise ValueError("TrackNetV3 BALL inference requires tracknet_repo, tracknet_file, and inpaintnet_file")
        _require_file(self.tracknet_repo / "predict.py", "TrackNetV3 predict.py")
        _require_file(self.tracknet_file, "TrackNetV3 TrackNet checkpoint")
        _require_file(self.inpaintnet_file, "TrackNetV3 InpaintNet checkpoint")
        video = _resolve_video_path(context, explicit=self.video_path)
        fps = self.tracknet_fps if self.tracknet_fps is not None else _video_fps(video)
        runner = self.tracknet_runner or _default_tracknet_runner()
        out = context.run_dir / "ball_track.json"
        metadata_out = context.run_dir / TRACKNET_RUN_METADATA_FILENAME
        summary = runner(
            out=out,
            fps=fps,
            metadata_out=metadata_out,
            video=video,
            tracknet_file=self.tracknet_file,
            inpaintnet_file=self.inpaintnet_file,
            tracknet_repo=self.tracknet_repo,
            prediction_dir=context.run_dir / TRACKNET_PREDICTION_DIR,
            batch_size=self.batch_size,
            large_video=self.tracknet_large_video,
            confidence_mode=self.tracknet_confidence_mode,
            heatmap_visible_threshold=self.tracknet_heatmap_visible_threshold,
            emit_candidates=self.emit_ball_candidates,
            candidate_top_k=self.ball_candidate_top_k,
        )
        model_metrics = _tracknet_model_metrics(summary)
        if self.tracknet_local_search:
            raw_out = context.run_dir / TRACKNET_RAW_BALL_FILENAME
            _copy_file(out, raw_out)
            local_summary_out = context.run_dir / BALL_LOCAL_SEARCH_SUMMARY_FILENAME
            local_summary = (self.local_search_runner or _default_local_search_runner())(
                video_path=video,
                ball_track_path=raw_out,
                out_path=out,
                summary_path=local_summary_out,
                court_calibration_path=(
                    _optional_court_calibration_path(context) if self.tracknet_local_search_court_filter else None
                ),
            )
            model_metrics["raw_tracknet_ball_track"] = str(raw_out)
            model_metrics["local_search"] = _local_search_metrics(local_summary)
        payload = _read_json(out)
        return payload, out, "tracknetv3_inference", model_metrics

    def _resolve_source_path(self, context: Any) -> Path:
        candidates = [self.source_path] if self.source_path is not None else _default_source_candidates(
            context,
            prototype_root=self.prototype_root,
            allow_prototype_root_fallback=self.allow_prototype_root_fallback,
        )
        searched: list[Path] = []
        for candidate in candidates:
            if candidate is None:
                continue
            searched.append(candidate)
            if candidate.is_file():
                return candidate

        searched_text = ", ".join(str(path) for path in searched)
        raise FileNotFoundError(
            f"missing no-click BALL source artifact: {DEFAULT_NO_CLICK_BALL_FILENAME}; "
            f"searched: {searched_text}; will not fall back to ball_points.json"
        )


def _default_totnet_runner() -> ModelRunner:
    from scripts.racketsport.run_totnet_ball import run_totnet_video

    return run_totnet_video


def _default_tracknet_runner() -> ModelRunner:
    from threed.racketsport.tracknet_adapter import run_tracknet_or_convert

    return run_tracknet_or_convert


def _default_local_search_runner() -> ModelRunner:
    from threed.racketsport.ball_local_search import write_local_search_ball_track

    return write_local_search_ball_track


def _totnet_model_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"model_family": "TOTNet"}
    for key in ("frame_count", "visible_frame_count", "confidence_threshold"):
        if key in summary:
            metrics[key] = summary[key]
    runtime = summary.get("runtime")
    if isinstance(runtime, dict):
        metrics["runtime"] = runtime
    model = summary.get("model")
    if isinstance(model, dict):
        metrics["model"] = model
    return metrics


def _tracknet_model_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"model_family": "TrackNetV3"}
    if "frame_count" in summary:
        metrics["tracknet_raw_frame_count"] = summary["frame_count"]
    if "visible_frame_count" in summary:
        metrics["tracknet_raw_visible_frame_count"] = summary["visible_frame_count"]
    if "source_mode" in summary:
        metrics["tracknet_source_mode"] = summary["source_mode"]
    runtime = summary.get("runtime")
    if isinstance(runtime, dict):
        metrics["runtime"] = runtime
    return metrics


def _local_search_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "visible_before",
        "visible_after",
        "recovered_count",
        "relocated_off_path_count",
        "suppressed_off_path_count",
        "evidence_miss_count",
        "court_rejected_count",
        "suppress_conf_threshold",
        "uses_human_clicks",
    )
    return {field: summary[field] for field in fields if field in summary}


def _apply_ball_physics3d(
    ball_payload: dict[str, Any],
    *,
    summary_path: Path,
    source_path: Path,
    context: Any,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    samples = _world_xyz_samples(ball_payload)
    existing_bounces = ball_payload.get("bounces", [])
    existing_bounce_count = len(existing_bounces) if isinstance(existing_bounces, list) else 0
    image_reconstruction: dict[str, Any] | None = None
    physics_bounces: list[dict[str, object]] = []
    sample_source = "existing_world_xyz"

    if len(samples) < 3:
        reconstruction = _reconstruct_image_ball_physics3d(ball_payload, context=context)
        image_reconstruction = reconstruction.summary()
        if reconstruction.status == "ran" and reconstruction.bounces:
            _apply_reconstructed_samples(ball_payload, reconstruction.samples, reconstruction.frame_indices)
            samples = reconstruction.samples
            physics_bounces = reconstruction.bounces
            sample_source = "image_calibration_bounce_fit"
    if not physics_bounces and len(samples) >= 3:
        events = detect_bounce_events(
            samples,
            min_vertical_speed_mps=BALL_BOUNCE_MIN_VERTICAL_SPEED_MPS,
            min_separation_s=BALL_BOUNCE_MIN_SEPARATION_S,
        )
        physics_bounces = project_bounces_to_ball_track(events)
    if physics_bounces:
        ball_payload["bounces"] = physics_bounces

    if len(samples) < 3:
        status = "insufficient_world_xyz_samples"
        notes = ("BALL 3D bounce physics skipped; fewer than 3 world_xyz ball samples",)
        if image_reconstruction is not None:
            notes = (*notes, "calibrated image-track bounce reconstruction did not produce accepted 3D samples")
    elif physics_bounces:
        status = "ran"
        if sample_source == "image_calibration_bounce_fit":
            notes = ("reconstructed BALL 3D bounce physics from image track and court calibration",)
        else:
            notes = ("applied BALL 3D bounce physics from existing world_xyz samples",)
    else:
        status = "ran_no_bounce_events"
        notes = ("BALL 3D bounce physics found no bounce events in existing world_xyz samples",)

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_physics3d_summary",
        "status": status,
        "source_ball_track": str(source_path),
        "sample_count": len(samples),
        "sample_source": sample_source,
        "existing_bounce_count": existing_bounce_count,
        "bounce_count": len(physics_bounces),
        "applied_to_ball_track": bool(physics_bounces),
        "min_vertical_speed_mps": BALL_BOUNCE_MIN_VERTICAL_SPEED_MPS,
        "min_separation_s": BALL_BOUNCE_MIN_SEPARATION_S,
        "uses_human_clicks": False,
        "not_gate_verified": True,
    }
    if image_reconstruction is not None:
        summary["image_reconstruction"] = image_reconstruction
    _write_json(summary_path, summary)
    return ball_payload, summary, notes


def _reconstruct_image_ball_physics3d(ball_payload: dict[str, Any], *, context: Any) -> BounceArcReconstruction:
    calibration_path = Path(context.run_dir) / "court_calibration.json"
    if not calibration_path.is_file():
        return BounceArcReconstruction(
            status="missing_court_calibration",
            notes=(f"missing court_calibration.json for image-track BALL 3D reconstruction: {calibration_path}",),
        )
    try:
        calibration = _read_json(calibration_path)
    except ValueError as exc:
        return BounceArcReconstruction(status="invalid_court_calibration", notes=(str(exc),))
    try:
        return reconstruct_bounce_arcs_from_image_track(
            ball_payload,
            calibration,
            image_size=_optional_source_video_size(context),
        )
    except ValueError as exc:
        return BounceArcReconstruction(status="invalid_image_reconstruction_input", notes=(str(exc),))


def _apply_reconstructed_samples(
    ball_payload: dict[str, Any],
    samples: tuple[BallSample3D, ...],
    frame_indices: tuple[int, ...],
) -> None:
    frames = ball_payload.get("frames", [])
    if not isinstance(frames, list):
        return
    for frame_index, sample in zip(frame_indices, samples, strict=True):
        if frame_index < 0 or frame_index >= len(frames):
            continue
        frame = frames[frame_index]
        if not isinstance(frame, dict):
            continue
        frame["world_xyz"] = [sample.x, sample.y, sample.z]
        if "approx" in frame:
            frame["approx"] = bool(frame["approx"])
        else:
            frame["approx"] = True


def _world_xyz_samples(ball_payload: dict[str, Any]) -> tuple[BallSample3D, ...]:
    samples: list[BallSample3D] = []
    frames = ball_payload.get("frames", [])
    if not isinstance(frames, list):
        return ()
    for frame in frames:
        if not isinstance(frame, dict) or not frame.get("visible"):
            continue
        world_xyz = frame.get("world_xyz")
        if not isinstance(world_xyz, (list, tuple)) or len(world_xyz) != 3:
            continue
        samples.append(
            BallSample3D(
                t=float(frame["t"]),
                x=float(world_xyz[0]),
                y=float(world_xyz[1]),
                z=float(world_xyz[2]),
            )
        )
    return tuple(samples)


def _physics3d_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "sample_count",
        "sample_source",
        "existing_bounce_count",
        "bounce_count",
        "applied_to_ball_track",
        "min_vertical_speed_mps",
        "min_separation_s",
        "uses_human_clicks",
        "not_gate_verified",
    )
    metrics = {field: summary[field] for field in fields if field in summary}
    image_reconstruction = summary.get("image_reconstruction")
    if isinstance(image_reconstruction, dict):
        metrics["image_reconstruction"] = {
            field: image_reconstruction[field]
            for field in (
                "status",
                "sample_count",
                "bounce_count",
                "reprojection_rmse_px",
                "max_reprojection_error_px",
                "candidate_count",
                "selected_bounce_time_s",
                "effective_accel_z_mps2",
            )
            if field in image_reconstruction
        }
    return metrics


def _resolve_video_path(context: Any, *, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    tracking_video = getattr(context, "tracking_video", None)
    if tracking_video is not None:
        candidates.append(Path(tracking_video))
    inputs_dir = Path(context.inputs_dir)
    candidates.extend(
        [
            inputs_dir / "source.mp4",
            inputs_dir / "clip.mp4",
            inputs_dir / "video.mp4",
            inputs_dir / "input.mp4",
        ]
    )
    if inputs_dir.is_dir():
        candidates.extend(sorted(path for path in inputs_dir.iterdir() if path.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"TOTNet BALL inference requires a source video; searched: {searched}")


def _video_fps(video: Path) -> float:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"TrackNetV3 BALL inference requires tracknet_fps because cv2 is unavailable: {video}") from exc
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    finally:
        cap.release()
    if fps <= 0.0:
        raise ValueError(f"could not determine FPS for TrackNetV3 BALL video: {video}")
    return fps


def _optional_source_video_size(context: Any) -> tuple[int, int] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    candidates: list[Path] = []
    tracking_video = getattr(context, "tracking_video", None)
    if tracking_video is not None:
        candidates.append(Path(tracking_video))
    inputs_dir = Path(context.inputs_dir)
    candidates.extend(
        [
            inputs_dir / "source.mp4",
            inputs_dir / "clip.mp4",
            inputs_dir / "video.mp4",
            inputs_dir / "input.mp4",
        ]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        cap = cv2.VideoCapture(str(candidate))
        try:
            if not cap.isOpened():
                continue
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width > 0 and height > 0:
            return width, height
    return None


def _optional_court_calibration_path(context: Any) -> Path | None:
    candidate = Path(context.run_dir) / "court_calibration.json"
    return candidate if candidate.is_file() else None


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _default_source_candidates(context: Any, *, prototype_root: Path, allow_prototype_root_fallback: bool) -> list[Path]:
    filename = DEFAULT_NO_CLICK_BALL_FILENAME
    smoke_dir = DEFAULT_TRACKNET_SMOKE_DIR
    selected_dir = DEFAULT_SELECTED_TRACKS_DIR
    selected_filename = DEFAULT_SELECTED_BALL_FILENAME
    candidates = [
        context.inputs_dir / selected_dir / context.clip / selected_filename,
        context.inputs_dir.parent / selected_dir / context.clip / selected_filename,
        context.run_dir / selected_dir / context.clip / selected_filename,
        context.run_dir.parent / selected_dir / context.clip / selected_filename,
        context.inputs_dir / filename,
        context.inputs_dir / smoke_dir / filename,
        context.run_dir / filename,
        context.run_dir / smoke_dir / filename,
    ]
    if allow_prototype_root_fallback:
        candidates.extend(
            [
                prototype_root / selected_dir / context.clip / selected_filename,
                prototype_root / context.clip / smoke_dir / filename,
            ]
        )
    return candidates


def _source_mode_for_path(path: Path) -> str:
    if DEFAULT_SELECTED_TRACKS_DIR in path.parts and path.name == DEFAULT_SELECTED_BALL_FILENAME:
        return "selected_ball_track_prototype"
    return BallStageRunner.source_mode


def _source_note_for_path(path: Path) -> str:
    if _source_mode_for_path(path) == "selected_ball_track_prototype":
        return (
            "consumed eval-suite selected ball track artifact; selection may point at a composite prototype and "
            "does not prove a trained PB-MAT checkpoint"
        )
    return "consumed strict no-click TrackNet/VballNet local-trajectory prototype ball track"


def _selection_metadata_for_track(path: Path) -> dict[str, Any] | None:
    sidecar = path.parent / "ball_track_selection.json"
    if not sidecar.is_file():
        return None
    payload = _read_json(sidecar)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ball_track_selection.json payload: {sidecar}")
    if payload.get("artifact_type") != "racketsport_ball_track_selection":
        raise ValueError(f"invalid ball_track_selection.json artifact_type: {sidecar}")
    fields = (
        "status",
        "clip",
        "candidate",
        "candidate_category",
        "candidate_score",
        "candidate_rank",
        "eligible_for_model_ranking",
        "trained_pbmat_checkpoint",
        "source_ball_track",
        "out",
        "not_ground_truth",
    )
    return {field: payload[field] for field in fields if field in payload}


def _derive_ball_inflections_from_current_track(
    context: Any,
    *,
    ball_payload: dict[str, Any],
    enabled: bool,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    if not enabled:
        return None, ()
    input_ball_inflections = Path(context.inputs_dir) / DEFAULT_BALL_INFLECTIONS_FILENAME
    if input_ball_inflections.is_file():
        return None, ()

    payload = build_ball_inflections_from_ball_track(ball_payload)
    _write_json(Path(context.run_dir) / DEFAULT_BALL_INFLECTIONS_FILENAME, payload)
    return payload, ("derived ball_inflections.json from current ball_track.json image motion",)


def _derive_wrist_velocity_peaks_from_current_skeleton(
    context: Any,
    *,
    enabled: bool,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    if not enabled:
        return None, ()
    input_wrist_peaks = Path(context.inputs_dir) / DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME
    if input_wrist_peaks.is_file():
        return None, ()
    skeleton_path = Path(context.run_dir) / "skeleton3d.json"
    if not skeleton_path.is_file():
        return None, ()

    payload = build_wrist_velocity_peaks_from_file(skeleton_path)
    _write_json(Path(context.run_dir) / DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME, payload)
    return payload, ("derived wrist_velocity_peaks.json from current skeleton3d.json",)


def _contact_windows_from_cues(
    context: Any,
    *,
    fps: float,
    contact_fusion_mode: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    cue_paths = {
        "audio": _first_existing(context, DEFAULT_AUDIO_ONSETS_FILENAME),
        "wrist": _first_existing(context, DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME),
        "ball": _first_existing(context, DEFAULT_BALL_INFLECTIONS_FILENAME),
    }
    require_audio = contact_fusion_mode == CONTACT_FUSION_MODE_AUDIO_WRIST_BALL
    required_cues = ("audio", "wrist", "ball") if require_audio else ("wrist", "ball")
    missing = [name for name in required_cues if cue_paths[name] is None]
    if missing:
        return (
            build_contact_windows_artifact([]),
            (f"contact_windows.json is empty because required cue artifacts are missing: {', '.join(missing)}",),
        )

    contact_payload = fuse_contact_windows_from_cue_payloads(
        fps=fps,
        audio_onsets_payload=_read_json(cue_paths["audio"]) if require_audio else [],
        wrist_velocity_peaks_payload=_read_json(cue_paths["wrist"]),
        ball_inflections_payload=_read_json(cue_paths["ball"]),
        require_audio=require_audio,
    )
    event_count = len(contact_payload.get("events", []))
    if event_count:
        if require_audio:
            note = "fused contact_windows.json from audio, wrist, and ball cue artifacts"
        else:
            note = "fused contact_windows.json from wrist and ball cue artifacts"
    else:
        note = "contact cue artifacts were present but produced zero temporally matched contact windows"
    return contact_payload, (note,)


def _first_existing(context: Any, filename: str) -> Path | None:
    for root in (context.inputs_dir, context.run_dir):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid no-click BALL source artifact JSON: {path}: {exc}") from exc


__all__ = [
    "BALL_PHYSICS3D_SUMMARY_FILENAME",
    "BallStageRunner",
    "CONTACT_FUSION_MODE_AUDIO_WRIST_BALL",
    "CONTACT_FUSION_MODE_WRIST_BALL",
    "DEFAULT_AUDIO_ONSETS_FILENAME",
    "DEFAULT_BALL_INFLECTIONS_FILENAME",
    "DEFAULT_NO_CLICK_BALL_FILENAME",
    "DEFAULT_PROTOTYPE_GATE_ROOT",
    "DEFAULT_TRACKNET_SMOKE_DIR",
    "TRACKNET_PREDICTION_DIR",
    "TRACKNET_RUN_METADATA_FILENAME",
    "TOTNET_PREDICTIONS_FILENAME",
    "TOTNET_RUN_METADATA_FILENAME",
    "DEFAULT_WRIST_VELOCITY_PEAKS_FILENAME",
]
