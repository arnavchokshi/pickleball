"""Benchmark person tracker variants on pickleball clips."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.racketsport.track import build_tracks

from .body_compute import build_body_compute_execution, write_body_compute_execution
from .detection_scaling import scale_detection_payload_bboxes
from .frame_rating import build_frame_compute_plan_from_files, write_frame_compute_plan
from .person_tracking_promotion_audit import build_person_tracking_promotion_audit
from .player_track_overlay import render_player_track_overlay
from .schemas import CourtCalibration, Tracks, validate_artifact_file
from .tiled_person_detector import (
    NormalizedCrop,
    parse_adaptive_crop_regions,
    parse_crop_regions,
    yolo_adaptive_tiled_detections_payload,
    yolo_tiled_detections_payload,
)

SUMMARY_NAME = "person_tracking_benchmark.json"
REPORT_NAME = "REPORT.md"
TIMING_CHART_NAME = "timing_chart.png"


@dataclass(frozen=True)
class PersonTrackerCandidate:
    name: str
    model: str
    tracker_config: Path


def parse_candidate_spec(spec: str) -> PersonTrackerCandidate:
    """Parse `name=model,tracker` into a benchmark candidate."""

    if "=" not in spec or "," not in spec:
        raise ValueError("candidate spec must be name=model,tracker")
    name, right = spec.split("=", 1)
    model, tracker = right.split(",", 1)
    if not name.strip() or not model.strip() or not tracker.strip():
        raise ValueError("candidate spec must be name=model,tracker")
    return PersonTrackerCandidate(name=name.strip(), model=model.strip(), tracker_config=Path(tracker.strip()))


def run_person_tracking_candidate(
    *,
    candidate: PersonTrackerCandidate,
    clip: str,
    video_path: str | Path,
    calibration_path: str | Path,
    out_dir: str | Path,
    max_players: int,
    max_frames: int | None = None,
    device: str | None = None,
    imgsz: int = 960,
    conf: float = 0.18,
    iou: float = 0.6,
    max_step_m: float = 2.0,
    court_margin_m: float = 0.0,
    id_strategy: str = "auto",
    batch_size: int = 32,
    half: bool | None = None,
    crop_regions: str | Sequence[NormalizedCrop] | None = None,
    adaptive_min_detections: int | None = None,
    ball_track_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    expected_players: int | None = None,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for person-tracking benchmarks") from exc

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration = _load_calibration(calibration_path)
    fps = _video_fps(video_path)
    video_width, video_height = _video_size(video_path)
    target_width, target_height = _calibration_resolution(calibration)
    scale_x = target_width / video_width
    scale_y = target_height / video_height

    model = YOLO(candidate.model)
    tiled_candidate = _is_tiled_tracker_config(candidate.tracker_config)
    adaptive_crop_regions = tiled_candidate and _is_adaptive_crop_region_spec(crop_regions)
    selected_adaptive_min_detections: int | None = None
    fallback_crop_regions: tuple[NormalizedCrop, ...] = ()
    if adaptive_crop_regions:
        primary_crop_regions, fallback_crop_regions, preset_min_detections = parse_adaptive_crop_regions(str(crop_regions))
        selected_adaptive_min_detections = int(adaptive_min_detections) if adaptive_min_detections is not None else preset_min_detections
        if selected_adaptive_min_detections <= 0:
            raise ValueError("adaptive_min_detections must be positive")
        resolved_crop_regions = primary_crop_regions + fallback_crop_regions
    else:
        resolved_crop_regions = parse_crop_regions(crop_regions) if tiled_candidate else ()
    _reset_peak_memory(device)
    start = time.perf_counter()
    if tiled_candidate:
        if adaptive_crop_regions:
            raw_detections_payload = yolo_adaptive_tiled_detections_payload(
                model=model,
                video_path=video_path,
                fps=fps,
                max_frames=max_frames,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                batch_size=batch_size,
                half=half,
                primary_crop_regions=primary_crop_regions,
                fallback_crop_regions=fallback_crop_regions,
                min_detections=selected_adaptive_min_detections,
            )
        else:
            raw_detections_payload = yolo_tiled_detections_payload(
                model=model,
                video_path=video_path,
                fps=fps,
                max_frames=max_frames,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                batch_size=batch_size,
                half=half,
                crop_regions=resolved_crop_regions,
            )
        raw_counts = _counts_from_detection_payload(raw_detections_payload)
    else:
        results = model.track(
            source=str(video_path),
            tracker=str(candidate.tracker_config),
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            persist=False,
            verbose=False,
        )
        try:
            raw_detections_payload, raw_counts = _detections_payload_from_tracked_results(results, fps=fps, max_frames=max_frames)
        finally:
            close = getattr(results, "close", None)
            if callable(close):
                close()
    wall_time_s = time.perf_counter() - start
    detections_payload = scale_detection_payload_bboxes(raw_detections_payload, scale_x=scale_x, scale_y=scale_y)

    tracks, counts = build_tracks(
        detections_payload,
        calibration,
        max_step_m=max_step_m,
        max_players=max_players,
        court_margin_m=court_margin_m,
        id_strategy=id_strategy,  # type: ignore[arg-type]
    )
    raw_detections_path = run_dir / "raw_tracked_detections.json"
    scaled_detections_path = run_dir / "tracked_detections.json"
    tracks_path = run_dir / "tracks.json"
    overlay_path = run_dir / "track_overlay.mp4"
    raw_detections_path.write_text(json.dumps(raw_detections_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scaled_detections_path.write_text(json.dumps(detections_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tracks_path.write_text(json.dumps(tracks.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    overlay = render_player_track_overlay(
        video_path=video_path,
        tracks=tracks,
        output_path=overlay_path,
        max_frames=max_frames,
        bbox_scale_x=video_width / target_width,
        bbox_scale_y=video_height / target_height,
    )

    frame_count = int(raw_counts["tracker_frames"])
    presence_score = score_track_presence(tracks, total_frames=frame_count, target_players=max_players)
    canonical_safety_audit = build_person_tracking_promotion_audit(
        tracks=tracks,
        clip=clip,
        variant=candidate.name,
        court_margin_m=court_margin_m,
        max_players=max_players,
        total_frames=frame_count,
        labeled_gate_passed=False,
        sport=getattr(calibration, "sport", "pickleball"),
    )
    adaptive_body_schedule = _maybe_write_adaptive_body_schedule(
        tracks=tracks,
        tracks_path=tracks_path,
        run_dir=run_dir,
        ball_track_path=ball_track_path,
        contact_windows_path=contact_windows_path,
        expected_players=expected_players if expected_players is not None else max_players,
    )
    metrics = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_tracker_candidate",
        "status": "ok",
        "clip": clip,
        "variant": candidate.name,
        "model": candidate.model,
        "tracker_config": str(candidate.tracker_config),
        "max_players": max_players,
        "court_margin_m": court_margin_m,
        "id_strategy": id_strategy,
        "batch_size": batch_size,
        "half": half,
        "crop_regions": [list(region) for region in resolved_crop_regions],
        "crop_region_count": len(resolved_crop_regions),
        "adaptive_crop_regions": bool(adaptive_crop_regions),
        "adaptive_min_detections": selected_adaptive_min_detections,
        "max_frames": max_frames,
        "device": device,
        "wall_time_s": round(wall_time_s, 6),
        "effective_fps": round(frame_count / wall_time_s, 6) if wall_time_s > 0 else None,
        "source_video": str(video_path),
        "raw_detections_path": str(raw_detections_path),
        "scaled_detections_path": str(scaled_detections_path),
        "tracks_path": str(tracks_path),
        "overlay_path": str(overlay_path),
        "timing": {
            "wall_time_s": round(wall_time_s, 6),
            "effective_fps": round(frame_count / wall_time_s, 6) if wall_time_s > 0 else None,
        },
        "counts": {
            **raw_counts,
            **counts,
            "source_width": int(video_width),
            "source_height": int(video_height),
            "calibration_width": int(target_width),
            "calibration_height": int(target_height),
            "bbox_scale_x": round(scale_x, 6),
            "bbox_scale_y": round(scale_y, 6),
            "batch_size": batch_size,
            "half": half,
            "crop_region_count": len(resolved_crop_regions),
            "adaptive_crop_regions": bool(adaptive_crop_regions),
        },
        "player_count": len(tracks.players),
        "track_frame_count": sum(len(player.frames) for player in tracks.players),
        "track_lengths": {str(player.id): len(player.frames) for player in tracks.players},
        "presence_score": presence_score,
        "canonical_safety_audit": canonical_safety_audit,
        "peak_memory_mb": _peak_memory_mb(device),
        "overlay": overlay,
    }
    if adaptive_body_schedule is not None:
        metrics.update(
            {
                "frame_compute_plan_path": adaptive_body_schedule["frame_compute_plan_path"],
                "body_compute_execution_path": adaptive_body_schedule["body_compute_execution_path"],
                "adaptive_body_schedule": adaptive_body_schedule,
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def build_person_tracking_report(
    rows: list[dict[str, Any]],
    *,
    device: str | None,
    max_frames: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_tracking_benchmark",
        "status": "scored_not_gate_verified",
        "device": device,
        "max_frames": max_frames,
        "candidate_count": len(rows),
        "clip_count": len({row.get("clip") for row in rows}),
        "results": rows,
        "aggregate": _aggregate(rows),
    }


def write_person_tracking_report(summary: dict[str, Any], *, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / REPORT_NAME).write_text(render_person_tracking_markdown(summary), encoding="utf-8")
    _write_timing_chart(summary, out / TIMING_CHART_NAME)


def render_person_tracking_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Person Tracking Benchmark",
        "",
        "- status: `scored_not_gate_verified`",
        f"- device: `{summary.get('device')}`",
        f"- max_frames: `{summary.get('max_frames')}`",
        "",
        "These runs compare detector/tracker/ID approaches. They are not a labeled IDF1 gate.",
        "",
        "## Timing",
        "",
        "| Clip | Variant | Model | Tracker | Crops | Players | 4-player frames | 4-player % | Mean active | Track frames | BODY frames | Safety | Wall s | FPS | Extra IDs dropped | Overlay |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["results"]:
        counts = row.get("counts", {})
        presence = row.get("presence_score") or {}
        lines.append(
            "| {clip} | `{variant}` | `{model}` | `{tracker}` | {crops} | {players} | {target_frames} | {target_rate} | {mean_active} | {track_frames} | {body_frames} | {safety} | {wall} | {fps} | {dropped} | `{overlay}` |".format(
                clip=row.get("clip"),
                variant=row.get("variant"),
                model=row.get("model"),
                tracker=Path(str(row.get("tracker_config"))).name,
                crops=row.get("crop_region_count", "n/a"),
                players=row.get("player_count"),
                target_frames=presence.get("target_player_frames", "n/a"),
                target_rate=_fmt_pct(presence.get("target_player_frame_rate")),
                mean_active=_fmt(presence.get("mean_active_players")),
                track_frames=row.get("track_frame_count"),
                body_frames=_body_schedule_cell(row),
                safety=_safety_cell(row),
                wall=_fmt(row.get("wall_time_s")),
                fps=_fmt(row.get("effective_fps")),
                dropped=counts.get("extra_players_dropped", "n/a"),
                overlay=row.get("overlay_path"),
            )
        )
    lines.extend(["", "![Timing and 4-player coverage dashboard](timing_chart.png)", ""])
    return "\n".join(lines)


def _body_schedule_cell(row: dict[str, Any]) -> str:
    schedule = row.get("adaptive_body_schedule")
    if not isinstance(schedule, dict):
        return "n/a"
    summary = schedule.get("body_execution_summary")
    if not isinstance(summary, dict):
        return "n/a"
    return f"{int(summary.get('scheduled_frame_count') or 0)} / {int(summary.get('scheduled_player_frame_count') or 0)}"


def _safety_cell(row: dict[str, Any]) -> str:
    audit = row.get("canonical_safety_audit")
    if not isinstance(audit, dict):
        return "n/a"
    status = audit.get("status")
    if status == "trusted_for_trk_promotion":
        return "trusted"
    if status == "canonical_candidate_not_gate_verified":
        return "canonical candidate"
    if status == "diagnostic_only":
        blockers = audit.get("safety_blockers")
        if isinstance(blockers, list) and blockers:
            return "diagnostic only: " + ",".join(str(item) for item in blockers)
        return "diagnostic only"
    return str(status or "unknown")


def _load_calibration(path: str | Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", Path(path))
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("calibration artifact did not parse as CourtCalibration")
    return parsed


def _maybe_write_adaptive_body_schedule(
    *,
    tracks: Tracks,
    tracks_path: Path,
    run_dir: Path,
    ball_track_path: str | Path | None,
    contact_windows_path: str | Path | None,
    expected_players: int,
) -> dict[str, Any] | None:
    if ball_track_path is None and contact_windows_path is None:
        return None
    frame_plan_path = run_dir / "frame_compute_plan.json"
    body_execution_path = run_dir / "body_compute_execution.json"
    plan = build_frame_compute_plan_from_files(
        tracks_path=tracks_path,
        ball_track_path=ball_track_path,
        contact_windows_path=contact_windows_path,
        expected_players=expected_players,
    )
    write_frame_compute_plan(frame_plan_path, plan)
    execution = build_body_compute_execution(tracks, frame_plan_path=frame_plan_path)
    write_body_compute_execution(body_execution_path, execution)
    return {
        "status": "schedule_audit_not_gate_verified",
        "expected_players": expected_players,
        "ball_track_path": str(ball_track_path) if ball_track_path is not None else None,
        "contact_windows_path": str(contact_windows_path) if contact_windows_path is not None else None,
        "frame_compute_plan_path": str(frame_plan_path),
        "body_compute_execution_path": str(body_execution_path),
        "frame_plan_summary": plan["summary"],
        "body_execution_summary": execution["summary"],
    }


def _video_fps(path: str | Path) -> float:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read video FPS") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"could not determine FPS for video: {path}")
    return fps


def _video_size(path: str | Path) -> tuple[float, float]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read video dimensions") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"could not determine video size: {path}")
    return width, height


def _calibration_resolution(calibration: CourtCalibration) -> tuple[float, float]:
    width = float(calibration.intrinsics.cx) * 2.0
    height = float(calibration.intrinsics.cy) * 2.0
    if width <= 0 or height <= 0:
        raise ValueError("calibration intrinsics must expose positive cx/cy to infer image size")
    return width, height


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for row in rows:
        variant = str(row.get("variant"))
        bucket = aggregate.setdefault(
            variant,
            {
                "variant": variant,
                "model": row.get("model"),
                "tracker_config": row.get("tracker_config"),
                "clip_count": 0,
                "mean_wall_time_s": 0.0,
                "mean_effective_fps": 0.0,
                "mean_target_player_frame_rate": 0.0,
                "mean_active_players": 0.0,
                "total_track_frame_count": 0,
                "canonical_safe_candidate_count": 0,
                "diagnostic_only_count": 0,
                "trusted_for_trk_promotion_count": 0,
            },
        )
        bucket["clip_count"] += 1
        bucket["mean_wall_time_s"] += float(row.get("wall_time_s") or 0.0)
        bucket["mean_effective_fps"] += float(row.get("effective_fps") or 0.0)
        presence = row.get("presence_score") or {}
        bucket["mean_target_player_frame_rate"] += float(presence.get("target_player_frame_rate") or 0.0)
        bucket["mean_active_players"] += float(presence.get("mean_active_players") or 0.0)
        bucket["total_track_frame_count"] += int(row.get("track_frame_count") or 0)
        audit = row.get("canonical_safety_audit")
        if isinstance(audit, dict):
            if audit.get("safe_for_canonical_review") is True:
                bucket["canonical_safe_candidate_count"] += 1
            if audit.get("diagnostic_only") is True:
                bucket["diagnostic_only_count"] += 1
            if audit.get("trusted_for_trk_promotion") is True:
                bucket["trusted_for_trk_promotion_count"] += 1
    for bucket in aggregate.values():
        clips = max(1, int(bucket["clip_count"]))
        bucket["mean_wall_time_s"] = round(float(bucket["mean_wall_time_s"]) / clips, 6)
        bucket["mean_effective_fps"] = round(float(bucket["mean_effective_fps"]) / clips, 6)
        bucket["mean_target_player_frame_rate"] = round(float(bucket["mean_target_player_frame_rate"]) / clips, 6)
        bucket["mean_active_players"] = round(float(bucket["mean_active_players"]) / clips, 6)
    return aggregate


def _is_tiled_tracker_config(path: Path) -> bool:
    name = str(path)
    return name == "tiled" or name.startswith("tiled_") or Path(name).name.startswith("tiled_")


def _is_adaptive_crop_region_spec(spec: str | Sequence[NormalizedCrop] | None) -> bool:
    return isinstance(spec, str) and spec.strip().startswith("adaptive_")


def _counts_from_detection_payload(payload: dict[str, Any]) -> dict[str, int]:
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("tiled detection produced no frames")
    tracker_boxes = 0
    tracked_person_boxes = 0
    untracked_person_boxes = 0
    for frame in frames:
        detections = frame.get("detections", []) if isinstance(frame, dict) else []
        if not isinstance(detections, list):
            continue
        tracker_boxes += len(detections)
        for detection in detections:
            if isinstance(detection, dict) and detection.get("track_id") is not None:
                tracked_person_boxes += 1
            else:
                untracked_person_boxes += 1
    counts = {
        "tracker_frames": len(frames),
        "tracker_boxes": tracker_boxes,
        "tracked_person_boxes": tracked_person_boxes,
        "untracked_person_boxes": untracked_person_boxes,
        "tracker_non_person": 0,
    }
    for key in (
        "crop_eval_count",
        "fallback_frame_count",
        "primary_crop_region_count",
        "fallback_crop_region_count",
        "adaptive_min_detections",
    ):
        value = payload.get(key)
        if value is not None:
            counts[key] = int(value)
    return counts


def _detections_payload_from_tracked_results(results: Any, *, fps: float, max_frames: int | None) -> tuple[dict[str, Any], dict[str, int]]:
    frames: list[dict[str, Any]] = []
    counts = {
        "tracker_frames": 0,
        "tracker_boxes": 0,
        "tracked_person_boxes": 0,
        "untracked_person_boxes": 0,
        "tracker_non_person": 0,
    }
    for frame_index, result in enumerate(results):
        if max_frames is not None and frame_index >= max_frames:
            break
        counts["tracker_frames"] += 1
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
    return {"fps": fps, "frames": frames}, counts


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


def score_track_presence(tracks: Tracks, *, total_frames: int, target_players: int) -> dict[str, Any]:
    """Summarize how often retained tracks show the expected player count."""

    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if target_players <= 0:
        raise ValueError("target_players must be positive")
    frame_counts = [0 for _ in range(total_frames)]
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            if 0 <= frame_index < total_frames:
                frame_counts[frame_index] += 1

    histogram: dict[str, int] = {}
    for count in frame_counts:
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    target_player_frames = sum(1 for count in frame_counts if count >= target_players)
    frames_with_any_player = sum(1 for count in frame_counts if count > 0)
    track_lengths = [len(player.frames) for player in tracks.players]
    short_track_cutoff = max(1, int(round(total_frames * 0.25))) if total_frames else 1
    return {
        "target_players": target_players,
        "total_frames": total_frames,
        "frames_with_any_player": frames_with_any_player,
        "target_player_frames": target_player_frames,
        "target_player_frame_rate": (target_player_frames / total_frames) if total_frames else 0.0,
        "mean_active_players": (sum(frame_counts) / total_frames) if total_frames else 0.0,
        "active_player_histogram": histogram,
        "id_fragmentation": {
            "selected_player_count": len(tracks.players),
            "short_track_count": sum(1 for length in track_lengths if length <= short_track_cutoff),
            "min_track_frames": min(track_lengths) if track_lengths else 0,
            "max_track_frames": max(track_lengths) if track_lengths else 0,
        },
    }


def _write_timing_chart(summary: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        _write_timing_chart_with_pillow(summary, path)
        return

    rows = list(summary["results"])
    clips = list(dict.fromkeys(str(row.get("clip")) for row in rows))
    variants = list(dict.fromkeys(str(row.get("variant")) for row in rows))
    clip_labels = [_short_clip(clip) for clip in clips]
    colors = ["#2563eb", "#16a34a", "#f97316", "#9333ea", "#dc2626", "#0891b2"]
    width = 0.78 / max(1, len(variants))
    x_positions = np.arange(len(clips))
    by_key = {(str(row.get("clip")), str(row.get("variant"))): row for row in rows}

    fig, (runtime_ax, frames_ax) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    for index, variant in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2.0) * width
        runtime_values = [float((by_key.get((clip, variant)) or {}).get("wall_time_s") or 0.0) for clip in clips]
        frame_values = [
            float(((by_key.get((clip, variant)) or {}).get("presence_score") or {}).get("target_player_frame_rate") or 0.0)
            * 100.0
            for clip in clips
        ]
        bars = runtime_ax.bar(
            x_positions + offset,
            runtime_values,
            width,
            label=variant,
            color=colors[index % len(colors)],
        )
        frames_ax.bar(x_positions + offset, frame_values, width, color=colors[index % len(colors)])
        for bar, value in zip(bars, runtime_values, strict=True):
            if value > 0:
                runtime_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    runtime_ax.set_ylabel("Wall time (s)")
    runtime_ax.set_title("Person tracking: runtime vs 4-player active-frame coverage")
    runtime_ax.grid(axis="y", alpha=0.25)
    runtime_ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    frames_ax.set_ylabel("Frames with 4 players (%)")
    frames_ax.set_ylim(0, 100)
    frames_ax.set_xticks(x_positions, clip_labels)
    frames_ax.grid(axis="y", alpha=0.25)
    frames_ax.set_xlabel("Clip")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _short_clip(clip: str) -> str:
    replacements = {
        "indoor_doubles_fwuks_0500_long_mid_baseline": "indoor doubles",
        "wolverine_mixed_0200_mid_steep_corner": "wolverine mixed",
        "burlington_gold_0300_low_steep_corner": "burlington",
        "outdoor_webcam_iynbd_1500_long_high_baseline": "outdoor webcam",
    }
    return replacements.get(clip, clip)


def _write_timing_chart_with_pillow(summary: dict[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    rows = summary["results"]
    width = max(640, 140 * max(1, len(rows)))
    height = 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = [float(row.get("wall_time_s") or 0.0) for row in rows]
    max_value = max(values) if values else 1.0
    for index, value in enumerate(values):
        x0 = 40 + index * 120
        bar_h = int((value / max_value) * 220) if max_value > 0 else 0
        draw.rectangle([x0, height - 50 - bar_h, x0 + 60, height - 50], fill=(47, 126, 216))
        draw.text((x0, height - 42), str(rows[index].get("variant")), fill=(0, 0, 0))
        draw.text((x0, height - 64 - bar_h), _fmt(value), fill=(0, 0, 0))
    image.save(path)


def _reset_peak_memory(device: str | None) -> None:
    if device is None:
        return
    try:
        import torch
    except ImportError:
        return
    if str(device).startswith("cuda") or str(device).isdigit():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str | None) -> float | None:
    if device is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    if str(device).startswith("cuda") or str(device).isdigit():
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "PersonTrackerCandidate",
    "build_person_tracking_report",
    "parse_candidate_spec",
    "render_person_tracking_markdown",
    "run_person_tracking_candidate",
    "score_track_presence",
    "write_person_tracking_report",
]
