"""Submit-style BODY smoke runner for video-to-world-joints checks."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Mapping

from .body_compute import build_body_compute_execution, write_body_compute_execution
from .body_frame_materialization import materialize_body_frames
from .body_full_clip_gate import build_body_full_clip_gate, write_body_full_clip_gate
from .body_joint_quality import build_body_joint_quality_from_paths, write_body_joint_quality
from .body_mesh_readiness import build_body_mesh_readiness_from_paths, write_body_mesh_readiness
from .body_world_label_packet import build_body_world_label_packet_from_paths, write_body_world_label_packet
from .body_world_label_review_bundle import build_body_world_label_review_bundle_from_paths
from .body_world_label_review_overlay import INDEX_FILENAME as BODY_REVIEW_OVERLAY_INDEX
from .body_world_label_review_overlay import build_body_world_label_review_overlays_from_run
from .court_templates import Sport
from .orchestrator import BodyStageRunner, DEFAULT_BOTSORT_REID_CONFIG, DEFAULT_MODEL_MANIFEST, StageRun, run_pipeline
from .schemas import Skeleton3D, Tracks, validate_artifact_file


ARTIFACT_TYPE = "racketsport_body_video_smoke"
SCHEMA_VERSION = 1


def run_body_video_smoke(
    *,
    clip: str,
    inputs_dir: str | Path,
    video_path: str | Path,
    run_dir: str | Path,
    tracking_mode: Literal["real", "precomputed", "precomputed_tracks"] = "precomputed",
    sport: Sport = "pickleball",
    device: str | None = None,
    max_frames: int | None = None,
    manifest_path: str | Path = DEFAULT_MODEL_MANIFEST,
    tracker_config_path: str | Path = DEFAULT_BOTSORT_REID_CONFIG,
    max_players: int = 4,
    court_margin_m: float = 0.0,
    id_strategy: str = "auto",
    ball_source_path: str | Path | None = None,
    fast_sam_repo: str | Path | None = None,
    body_detector_name: str | None = None,
    body_fov_name: str | None = None,
    min_joint_count: int = 17,
    overwrite_frames: bool = True,
    diagnostic_full_track: bool = False,
    extra_runners: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a fail-closed video BODY smoke and write a single report.

    ``diagnostic_full_track`` is an explicit, diagnostic-only escape hatch
    that schedules BODY (Fast-SAM-3D-Body mesh) compute for every tracked
    player-frame instead of the contact-aware tier rule
    (``frame_rating.build_frame_compute_plan``, wired into the default
    production path via the orchestrator's Lane B frame-plan derivation).
    It exists only to reproduce past full-clip continuity/world-MPJPE
    diagnostics; a run made with it set is never representative of
    production BODY compute cost (~7x more player-frames go through the
    mesh model) and must not be cited as tier-rule promotion evidence. The
    default (``False``) path always lets the contact-aware plan through and
    additionally refuses to silently reuse a stale diagnostic-tagged
    ``frame_compute_plan.json`` that might already be sitting in
    ``inputs_dir`` from a prior diagnostic run (see ``_prepare_frame_plan``).

    ``extra_runners`` overrides/extends the default per-stage
    ``threed.racketsport.orchestrator`` ``StageRunner`` registry (merged into
    both the "tracking"-stage and "body"-stage ``run_pipeline`` calls below,
    with ``extra_runners`` taking precedence over the stage-specific
    defaults computed here). This exists for callers validating BODY against
    footage that cannot go through ``ManualCalibrationRunner`` (the default,
    unconditional "calibration" stage runner, which requires either a
    ``capture_sidecar.json`` manual-taps seed or ``court_keypoints.json`` --
    e.g. the external-ground-truth BODY validation lane
    (`threed.racketsport.external_gt_aspset510_body_inputs`), which already
    has real, trusted, non-pickleball camera calibration and needs to supply
    a pre-built ``court_calibration.json`` directly instead. Every other
    caller passes ``None`` (the default) and gets byte-identical behavior to
    before this parameter existed.
    """

    inputs = Path(inputs_dir)
    video = Path(video_path)
    out = Path(run_dir)
    if not inputs.is_dir():
        raise FileNotFoundError(f"missing inputs directory: {inputs}")
    if not video.is_file():
        raise FileNotFoundError(f"missing source video: {video}")
    out.mkdir(parents=True, exist_ok=True)

    tracking_summary = run_pipeline(
        clip=clip,
        inputs_dir=inputs,
        run_dir=out,
        stage="tracking",
        sport=sport,
        device=device,
        max_frames=max_frames,
        tracking_mode=tracking_mode,
        tracking_video=video if tracking_mode == "real" else None,
        manifest_path=manifest_path,
        tracker_config_path=tracker_config_path,
        max_players=max_players,
        court_margin_m=court_margin_m,
        id_strategy=id_strategy,
        ball_source_path=ball_source_path,
        runners=dict(extra_runners) if extra_runners else None,
    )

    frame_plan_path, frame_plan_notes = _prepare_frame_plan(inputs, out, diagnostic_full_track=diagnostic_full_track)
    body_execution_path = out / "body_compute_execution.json"
    scheduled_body_execution: dict[str, Any] | None = None
    frame_manifest: dict[str, Any] | None = None
    body_summary: dict[str, Any] | None = None
    body_error: str | None = None
    body_runtime_timing: dict[str, float] | None = None
    reuse_input_skeleton = _is_reusable_lane_a_skeleton(inputs / "skeleton3d.json")
    body_runners = _body_runners_with_pose(
        manifest_path=manifest_path,
        fast_sam_repo=fast_sam_repo,
        body_detector_name=body_detector_name,
        body_fov_name=body_fov_name,
        reuse_input_skeleton=reuse_input_skeleton,
    )
    if extra_runners:
        body_runners = {**body_runners, **extra_runners}

    try:
        tracks = validate_artifact_file("tracks", out / "tracks.json")
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks artifact did not parse as Tracks")
        if not reuse_input_skeleton:
            pose_execution_path = out / "lane_a_pose_frame_execution.json"
            pose_execution_path.write_text(
                json.dumps(_lane_a_pose_frame_execution(tracks), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            frame_manifest = materialize_body_frames(
                video_path=video,
                execution_path=pose_execution_path,
                out_dir=out / "body_frames",
                overwrite=overwrite_frames,
            )
        if frame_plan_path is not None:
            body_execution = build_body_compute_execution(
                tracks,
                frame_plan_path=frame_plan_path,
                max_frames=max_frames,
            )
            scheduled_body_execution = body_execution
            write_body_compute_execution(body_execution_path, body_execution)
            frame_manifest = materialize_body_frames(
                video_path=video,
                execution_path=body_execution_path,
                out_dir=out / "body_frames",
                overwrite=overwrite_frames,
            )
        body_wall_start = time.perf_counter()
        body_summary = run_pipeline(
            clip=clip,
            inputs_dir=inputs,
            run_dir=out,
            stage="body",
            sport=sport,
            device=device,
            max_frames=max_frames,
            tracking_mode=tracking_mode,
            tracking_video=video if tracking_mode == "real" else None,
            manifest_path=manifest_path,
            tracker_config_path=tracker_config_path,
            max_players=max_players,
            court_margin_m=court_margin_m,
            id_strategy=id_strategy,
            ball_source_path=ball_source_path,
            runners=body_runners,
        )
        if frame_plan_path is None and (out / "frame_compute_plan.json").is_file():
            frame_plan_path = out / "frame_compute_plan.json"
        body_runtime_timing = {"body_wall_seconds": max(0.0, time.perf_counter() - body_wall_start)}
    except Exception as exc:
        if "body_wall_start" in locals():
            body_runtime_timing = {"body_wall_seconds": max(0.0, time.perf_counter() - body_wall_start)}
        body_error = str(exc)

    if scheduled_body_execution is not None and (diagnostic_full_track or not _body_outputs_available(out)):
        write_body_compute_execution(body_execution_path, scheduled_body_execution)

    quality = build_body_joint_quality_from_paths(
        clip=clip,
        smpl_motion_path=out / "smpl_motion.json",
        skeleton3d_path=out / "skeleton3d.json",
        body_compute_execution_path=body_execution_path,
        min_joint_count=min_joint_count,
    )
    write_body_joint_quality(out / "body_joint_quality.json", quality)
    runtime_timing_path = out / "body_runtime_timing.json"
    if body_runtime_timing is not None:
        runtime_timing_path.write_text(json.dumps(body_runtime_timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    full_clip_gate = build_body_full_clip_gate(
        clip=clip,
        tracks=_read_optional_json(out / "tracks.json"),
        body_compute_execution=_read_optional_json(body_execution_path),
        body_joint_quality=quality,
        contact_splice=_read_optional_json(out / "contact_splice.json"),
        runtime_timing=body_runtime_timing,
        tracks_path=str(out / "tracks.json"),
        body_compute_execution_path=str(body_execution_path),
        body_joint_quality_path=str(out / "body_joint_quality.json"),
        contact_splice_path=str(out / "contact_splice.json"),
        runtime_timing_path=str(runtime_timing_path) if body_runtime_timing is not None else "",
    )
    write_body_full_clip_gate(out / "body_full_clip_gate.json", full_clip_gate)
    mesh_readiness = build_body_mesh_readiness_from_paths(
        clip=clip,
        smpl_motion_path=out / "smpl_motion.json",
        skeleton3d_path=out / "skeleton3d.json",
        frame_compute_plan_path=frame_plan_path,
        body_compute_execution_path=body_execution_path,
        body_full_clip_gate_path=out / "body_full_clip_gate.json",
    )
    write_body_mesh_readiness(out / "body_mesh_readiness.json", mesh_readiness)
    quality = _quality_after_full_clip_gate(quality, full_clip_gate)
    write_body_joint_quality(out / "body_joint_quality.json", quality)
    label_packet = build_body_world_label_packet_from_paths(
        clip=clip,
        smpl_motion_path=out / "smpl_motion.json",
        skeleton3d_path=out / "skeleton3d.json",
        body_compute_execution_path=body_execution_path,
        source_video=str(video),
        suggested_label_path="labels/body_world_joints.json",
    )
    write_body_world_label_packet(out / "body_world_label_packet.json", label_packet)
    packet_quality = build_body_joint_quality_from_paths(
        clip=clip,
        smpl_motion_path=None,
        skeleton3d_path=None,
        body_compute_execution_path=body_execution_path,
        body_world_label_packet_path=out / "body_world_label_packet.json",
        min_joint_count=min_joint_count,
    )
    write_body_joint_quality(out / "body_joint_quality_from_packet.json", packet_quality)
    label_review_bundle = build_body_world_label_review_bundle_from_paths(
        packet_path=out / "body_world_label_packet.json",
        body_frames_dir=out / "body_frames",
        out_dir=out / "body_world_label_review_bundle",
    )
    label_review_overlay = _build_label_review_overlay(out)

    body_stage = _stage(body_summary, "body")
    body_failure_note = ""
    if body_stage.get("status") == "fail":
        body_failure_note = _failure_note(body_stage)
    body_failure_note = body_failure_note or _pipeline_failure_note(body_summary) or body_error or ""
    body_runtime_ran = body_stage.get("status") == "ran" and quality.get("world_joints_available") is True
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": _status(quality=quality, body_failure_note=body_failure_note),
        "inputs_dir": str(inputs),
        "source_video": str(video),
        "run_dir": str(out),
        "tracking_mode": tracking_mode,
        "body_runtime_ran": body_runtime_ran,
        "body_failure_note": body_failure_note,
        "diagnostic_full_track_mode": diagnostic_full_track,
        "frame_plan_notes": list(frame_plan_notes),
        "paths": {
            "frame_compute_plan": str(frame_plan_path) if frame_plan_path is not None else "",
            "lane_a_pose_frame_execution": str(out / "lane_a_pose_frame_execution.json")
            if (out / "lane_a_pose_frame_execution.json").is_file()
            else "",
            "body_compute_execution": str(body_execution_path),
            "body_frames": str(out / "body_frames"),
            "smpl_motion": str(out / "smpl_motion.json"),
            "skeleton3d": str(out / "skeleton3d.json"),
            "body_joint_quality": str(out / "body_joint_quality.json"),
            "body_joint_quality_from_packet": str(out / "body_joint_quality_from_packet.json"),
            "body_full_clip_gate": str(out / "body_full_clip_gate.json"),
            "body_runtime_timing": str(runtime_timing_path) if body_runtime_timing is not None else "",
            "body_world_label_packet": str(out / "body_world_label_packet.json"),
            "body_world_label_review_bundle": str(out / "body_world_label_review_bundle"),
            "body_world_label_review_overlay": str(out / "body_world_label_review_bundle" / "overlays" / BODY_REVIEW_OVERLAY_INDEX),
        },
        "summary": _summary(quality=quality, frame_manifest=frame_manifest),
        "quality": {
            "status": quality["status"],
            "usable_for_review": quality["usable_for_review"],
            "quality_blockers": list(quality["quality_blockers"]),
            "promotion_blockers": list(quality["promotion_blockers"]),
            "warnings": list(quality.get("warnings", [])),
        },
        "full_clip_gate": {
            "passed": full_clip_gate["passed"],
            "coverage": full_clip_gate["coverage"],
            "contact_mesh_coverage": full_clip_gate["contact_mesh_coverage"],
            "latency_seconds_per_video_minute": full_clip_gate["latency_seconds_per_video_minute"],
            "blockers": list(full_clip_gate["blockers"]),
        },
        "label_packet": {
            "status": label_packet["status"],
            "sample_count": label_packet["summary"]["sample_count"],
            "required_review_sample_count": label_packet["review_plan"]["required_sample_count"],
            "selected_review_sample_count": label_packet["review_plan"]["selected_sample_count"],
            "trusted_for_world_mpjpe": label_packet["trusted_for_world_mpjpe"],
        },
        "label_review_bundle": {
            "status": label_review_bundle["status"],
            "selected_sample_count": label_review_bundle["selected_sample_count"],
            "required_sample_count": label_review_bundle["required_sample_count"],
            "missing_frame_count": label_review_bundle["missing_frame_count"],
            "queue_path": label_review_bundle["queue_path"],
            "label_template_path": label_review_bundle["label_template_path"],
        },
        "label_review_overlay": {
            "status": label_review_overlay["status"],
            "rendered_count": label_review_overlay["rendered_count"],
            "sample_count": label_review_overlay["sample_count"],
            "missing_frame_count": label_review_overlay["missing_frame_count"],
            "projection_failed_count": label_review_overlay["projection_failed_count"],
            "missing_track_bbox_count": label_review_overlay["missing_track_bbox_count"],
            "floor_anchor_projection_failed_count": int(
                label_review_overlay.get("floor_anchor_projection_failed_count", 0)
            ),
            "floor_anchor_projection_warning_count": int(
                label_review_overlay.get("floor_anchor_projection_warning_count", 0)
            ),
            "alignment_failed_count": label_review_overlay["alignment_failed_count"],
            "alignment_warning_count": label_review_overlay["alignment_warning_count"],
            "index_path": label_review_overlay["index_path"],
            "blockers": list(label_review_overlay["blockers"]),
        },
        "pipeline": {
            "tracking_status": tracking_summary.get("status") if isinstance(tracking_summary, Mapping) else "",
            "body_status": body_summary.get("status") if isinstance(body_summary, Mapping) else "",
            "body_stage_status": body_stage.get("status", ""),
        },
    }
    (out / "body_video_smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _body_runners(
    *,
    manifest_path: str | Path,
    fast_sam_repo: str | Path | None,
    body_detector_name: str | None,
    body_fov_name: str | None,
) -> dict[str, Any] | None:
    return _body_runners_with_pose(
        manifest_path=manifest_path,
        fast_sam_repo=fast_sam_repo,
        body_detector_name=body_detector_name,
        body_fov_name=body_fov_name,
        reuse_input_skeleton=False,
    )


def _body_runners_with_pose(
    *,
    manifest_path: str | Path,
    fast_sam_repo: str | Path | None,
    body_detector_name: str | None,
    body_fov_name: str | None,
    reuse_input_skeleton: bool,
) -> dict[str, Any] | None:
    runners: dict[str, Any] = {}
    if reuse_input_skeleton:
        runners["pose"] = _InputSkeletonPoseRunner()
    if fast_sam_repo is None and body_detector_name is None and body_fov_name is None:
        return runners or None
    kwargs: dict[str, Any] = {"manifest_path": manifest_path}
    if fast_sam_repo is not None:
        kwargs["fast_sam_repo"] = fast_sam_repo
    if body_detector_name is not None:
        kwargs["detector_name"] = body_detector_name
    if body_fov_name is not None:
        kwargs["fov_name"] = body_fov_name
    runners["body"] = BodyStageRunner(**kwargs)
    return runners or None


def _is_reusable_lane_a_skeleton(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        skeleton = validate_artifact_file("skeleton3d", path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(skeleton, Skeleton3D)
        and skeleton.preview_only is False
        and skeleton.provenance.get("lane") == "A"
    )


class _InputSkeletonPoseRunner:
    stage = "pose"
    real_model = False
    source_mode = "precomputed_lane_a_skeleton3d"

    def run(self, context: Any) -> StageRun:
        source = Path(context.inputs_dir) / "skeleton3d.json"
        target = Path(context.run_dir) / "skeleton3d.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing precomputed Lane A skeleton3d.json: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copy2(source, target)
        skeleton = validate_artifact_file("skeleton3d", target)
        if not isinstance(skeleton, Skeleton3D):
            raise ValueError("precomputed skeleton3d.json did not validate as Skeleton3D")
        if skeleton.preview_only:
            raise ValueError("precomputed skeleton3d.json must be a real Lane A skeleton, not preview_only")
        if skeleton.provenance.get("lane") != "A":
            raise ValueError("precomputed skeleton3d.json must have provenance.lane=A")
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=self.real_model,
            source_mode=self.source_mode,
            produced_artifacts=("skeleton3d.json",),
            notes=("reused precomputed Lane A skeleton3d.json for BODY video smoke",),
            metrics={
                "source_model": skeleton.source_model or "",
                "preview_only": skeleton.preview_only,
                "player_count": len(skeleton.players),
            },
        )


def _lane_a_pose_frame_execution(tracks: Tracks) -> dict[str, Any]:
    scheduled = []
    for frame_idx, active in _track_frames_by_index(tracks).items():
        player_ids = [player_id for player_id, _frame in active]
        scheduled.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / tracks.fps,
                "target_representation": "lane_a_skeleton",
                "target_player_ids": player_ids,
                "active_player_ids": player_ids,
                "reasons": ["lane_a_pose_full_track"],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_lane_a_pose_frame_execution",
        "fps": float(tracks.fps),
        "scheduled_frames": scheduled,
        "summary": {
            "scheduled_frame_count": len(scheduled),
            "scheduled_player_frame_count": sum(len(frame["target_player_ids"]) for frame in scheduled),
            "scheduled_by_reason": {"lane_a_pose_full_track": len(scheduled)} if scheduled else {},
        },
    }


def _prepare_frame_plan(inputs: Path, out: Path, *, diagnostic_full_track: bool = False) -> tuple[Path | None, list[str]]:
    """Resolve which ``frame_compute_plan.json`` (if any) the BODY stage should
    consume, and never let a diagnostic 100%-coverage plan silently stand in
    for the contact-aware tier rule on a non-diagnostic (production-path) run.

    - ``diagnostic_full_track=True`` explicitly opts into writing a
      100%-coverage plan (every tracked player-frame scheduled ``deep_mesh``)
      tagged ``diagnostic_full_track: true``. This is the only way to get
      full-track scheduling; it is never the default.
    - ``diagnostic_full_track=False`` (the default/production path) never
      writes a full-track plan itself. If ``inputs_dir`` already contains a
      ``frame_compute_plan.json``, it is reused *unless* it is tagged
      ``diagnostic_full_track: true`` -- e.g. left over in a shared inputs
      bundle from a prior diagnostic run -- in which case it is ignored (not
      copied into the run dir) so the orchestrator's own contact-aware
      derivation (``_ensure_lane_b_frame_plan_from_lane_a`` /
      ``frame_rating.build_frame_compute_plan``) runs instead. This is the
      concrete fail-safe against the diagnostic flag "silently overriding
      contact-aware windowing in production paths".
    """
    notes: list[str] = []
    source = inputs / "frame_compute_plan.json"
    target = out / "frame_compute_plan.json"
    if diagnostic_full_track:
        tracks = validate_artifact_file("tracks", out / "tracks.json")
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks artifact did not parse as Tracks")
        target.write_text(
            json.dumps(_diagnostic_full_track_frame_plan(tracks), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        notes.append(
            "diagnostic_full_track=True: wrote a 100%-coverage frame_compute_plan.json covering every "
            "tracked player-frame; this bypasses the contact-aware tier rule and must not be used as "
            "production BODY compute-cost or promotion evidence"
        )
        return target, notes
    if source.is_file():
        if _is_diagnostic_full_track_plan(source):
            notes.append(
                f"ignored diagnostic-tagged {source}: refusing to silently reuse a diagnostic_full_track=True "
                "plan for a non-diagnostic run; letting the contact-aware plan be derived from Lane A "
                "artifacts (ball_track.json/contact_windows.json) instead"
            )
            if target.is_file():
                target.unlink()
            return None, notes
        shutil.copy2(source, target)
        return target, notes
    if target.is_file():
        target.unlink()
    return None, notes


def _is_diagnostic_full_track_plan(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("diagnostic_full_track") is True


def _diagnostic_full_track_frame_plan(tracks: Tracks) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame_idx, active in _track_frames_by_index(tracks).items():
        player_targets = [
            {
                "player_id": player_id,
                "track_conf": round(float(track_frame.conf), 3),
                "score": 1.0,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["diagnostic_full_track_schedule"],
            }
            for player_id, track_frame in active
        ]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / tracks.fps,
                "score": 1.0,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["diagnostic_full_track_schedule"],
                "active_players": len(active),
                "active_player_ids": [player_id for player_id, _frame in active],
                "missing_players": 0,
                "min_track_conf": min((float(track_frame.conf) for _player_id, track_frame in active), default=0.0),
                "ball_conf": None,
                "player_targets": player_targets,
            }
        )
    frames = sorted(frames, key=lambda item: int(item["frame_idx"]))
    windows = _deep_mesh_windows_for_full_track(frames, fps=tracks.fps)
    player_target_count = sum(len(frame["player_targets"]) for frame in frames)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "diagnostic_full_track": True,
        "fps": float(tracks.fps),
        "expected_players": len(tracks.players),
        "frame_count": len(frames),
        "frames": frames,
        "deep_mesh_windows": windows,
        "summary": {
            "by_tier": {"deep_mesh": len(frames)} if frames else {},
            "by_reason": {"diagnostic_full_track_schedule": len(frames)} if frames else {},
            "by_player_target_representation": {"world_mesh": player_target_count} if player_target_count else {},
            "max_score": 1.0 if frames else 0.0,
            "deep_mesh_window_count": len(windows),
            "deep_mesh_frame_count": len(frames),
            "human_review_frame_count": 0,
            "targeted_reviewed_contact_frame_count": 0,
            "coverage_incomplete_deep_mesh_frame_count": 0,
        },
    }


def _track_frames_by_index(tracks: Tracks) -> dict[int, list[tuple[int, Any]]]:
    by_frame: dict[int, list[tuple[int, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            by_frame.setdefault(int(round(float(frame.t) * tracks.fps)), []).append((int(player.id), frame))
    return {frame_idx: sorted(active, key=lambda item: item[0]) for frame_idx, active in sorted(by_frame.items())}


def _deep_mesh_windows_for_full_track(frames: list[Mapping[str, Any]], *, fps: float) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    previous_idx: int | None = None
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if current and previous_idx is not None and frame_idx != previous_idx + 1:
            windows.append(_full_track_window(current, fps=fps))
            current = []
        current.append(frame)
        previous_idx = frame_idx
    if current:
        windows.append(_full_track_window(current, fps=fps))
    return windows


def _full_track_window(frames: list[Mapping[str, Any]], *, fps: float) -> dict[str, Any]:
    frame_start = int(frames[0]["frame_idx"])
    frame_end = int(frames[-1]["frame_idx"])
    target_player_ids = sorted(
        {
            int(target["player_id"])
            for frame in frames
            for target in frame.get("player_targets", [])
            if isinstance(target, Mapping)
        }
    )
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "t0": frame_start / fps,
        "t1": (frame_end + 1) / fps,
        "frame_count": len(frames),
        "target_representation": "world_mesh",
        "fallback_representation": "lane_a_skeleton",
        "target_player_ids": target_player_ids,
        "reason_counts": {"diagnostic_full_track_schedule": len(frames)},
        "max_score": 1.0,
    }


def _read_optional_json(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _quality_after_full_clip_gate(quality: dict[str, Any], full_clip_gate: Mapping[str, Any]) -> dict[str, Any]:
    if full_clip_gate.get("passed") is not True:
        return quality
    out = dict(quality)
    promotion_blockers = [
        str(blocker)
        for blocker in out.get("promotion_blockers", [])
        if str(blocker) and str(blocker) != "missing_full_clip_body_gate"
    ]
    quality_blockers = [
        str(blocker)
        for blocker in out.get("quality_blockers", [])
        if str(blocker)
    ]
    out["promotion_blockers"] = promotion_blockers
    out["blockers"] = _dedupe([*quality_blockers, *promotion_blockers])
    return out


def _build_label_review_overlay(out: Path) -> dict[str, Any]:
    overlay_dir = out / "body_world_label_review_bundle" / "overlays"
    index_path = overlay_dir / BODY_REVIEW_OVERLAY_INDEX
    required_inputs = [
        out / "body_world_label_review_bundle" / "body_world_label_review_queue.json",
        out / "tracks.json",
        out / "court_calibration.json",
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        return _write_label_review_overlay_blocked(
            overlay_dir=overlay_dir,
            status="blocked_missing_overlay_inputs",
            blockers=["missing_overlay_inputs"],
            missing_inputs=missing_inputs,
        )
    try:
        return build_body_world_label_review_overlays_from_run(run_dir=out, out_dir=overlay_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return _write_label_review_overlay_blocked(
            overlay_dir=overlay_dir,
            status="blocked_overlay_render_failed",
            blockers=["overlay_render_failed"],
            missing_inputs=[],
            error=str(exc),
        )


def _write_label_review_overlay_blocked(
    *,
    overlay_dir: Path,
    status: str,
    blockers: list[str],
    missing_inputs: list[str],
    error: str = "",
) -> dict[str, Any]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_review_overlay",
        "status": status,
        "out_dir": str(overlay_dir),
        "index_path": str(overlay_dir / BODY_REVIEW_OVERLAY_INDEX),
        "sample_count": 0,
        "rendered_count": 0,
        "missing_frame_count": 0,
        "projection_failed_count": 0,
        "missing_track_bbox_count": 0,
        "floor_anchor_projection_failed_count": 0,
        "floor_anchor_projection_warning_count": 0,
        "alignment_failed_count": 0,
        "alignment_warning_count": 0,
        "missing_inputs": missing_inputs,
        "blockers": blockers,
        "error": error,
        "overlays": [],
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "qualitative_status": "review_overlay_not_gate_verified",
    }
    (overlay_dir / BODY_REVIEW_OVERLAY_INDEX).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _stage(summary: Mapping[str, Any] | None, stage_name: str) -> dict[str, Any]:
    stages = summary.get("stages") if isinstance(summary, Mapping) else None
    if not isinstance(stages, list):
        return {}
    for stage in stages:
        if isinstance(stage, Mapping) and stage.get("stage") == stage_name:
            return dict(stage)
    return {}


def _failure_note(stage: Mapping[str, Any]) -> str:
    notes = stage.get("notes")
    if not isinstance(notes, list):
        return ""
    return "; ".join(str(note) for note in notes if note)


def _pipeline_failure_note(summary: Mapping[str, Any] | None) -> str:
    stages = summary.get("stages") if isinstance(summary, Mapping) else None
    if not isinstance(stages, list):
        return ""
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        if stage.get("status") == "fail":
            return _failure_note(stage)
    return ""


def _body_outputs_available(out: Path) -> bool:
    return (out / "smpl_motion.json").is_file() and (out / "skeleton3d.json").is_file()


def _status(*, quality: Mapping[str, Any], body_failure_note: str) -> str:
    if quality.get("status") == "quality_checked_needs_accuracy_gate":
        return "quality_checked_needs_accuracy_gate"
    quality_blockers = set(str(blocker) for blocker in quality.get("quality_blockers", []))
    missing_body_output = bool(
        quality_blockers.intersection({"missing_smpl_motion_json", "missing_skeleton3d_json", "no_world_joint_frames"})
    )
    if body_failure_note and missing_body_output:
        return "runtime_blocked"
    return "quality_blocked"


def _summary(
    *,
    quality: Mapping[str, Any],
    frame_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    quality_summary = quality.get("summary") if isinstance(quality.get("summary"), Mapping) else {}
    return {
        "scheduled_frame_count": int(quality_summary.get("scheduled_frame_count", 0)),
        "scheduled_player_frame_count": int(quality_summary.get("scheduled_player_frame_count", 0)),
        "extracted_frame_count": int(frame_manifest.get("extracted_frame_count", 0)) if frame_manifest else 0,
        "joint_frame_count": int(quality_summary.get("joint_frame_count", 0)),
        "joint_count_min": int(quality_summary.get("joint_count_min", 0)),
        "joint_count_max": int(quality_summary.get("joint_count_max", 0)),
        "schedule_coverage_ratio": float(quality_summary.get("schedule_coverage_ratio", 0.0)),
    }
