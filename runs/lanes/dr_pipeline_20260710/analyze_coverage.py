#!/usr/bin/env python3
"""Read-only artifact coverage audit for dr_pipeline_20260710.

All outputs are written beside this script. No protected labels are read.
"""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from threed.racketsport.ball_court_filter import load_court_calibration
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.person_fast import person_detection_from_bbox
from threed.racketsport.raw_pool_person_authority import raw_pool_four_player_ceiling
from threed.racketsport.virtual_world import (
    apply_ball_track_arc_solved_overlay,
    ball_arc_segment_fail_closed_verdicts,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = {
    "wolverine_w7": ROOT
    / "runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner",
    "zwcth45s_r1": ROOT / "runs/lanes/demo_beststack_gpu_20260710/vm_pull/zwcth45s",
    "zwcth45s_r2": ROOT / "runs/lanes/demo_beststack_gpu_20260710/vm_pull/zwcth45s_r2",
}


def read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def frame_idx(frame: Mapping[str, Any], fps: float) -> int:
    if isinstance(frame.get("frame_idx"), int):
        return int(frame["frame_idx"])
    if isinstance(frame.get("frame"), int):
        return int(frame["frame"])
    return int(round(float(frame.get("t", 0.0)) * fps))


def player_frames(payload: Mapping[str, Any] | None, fps: float) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for player in (payload or {}).get("players", []):
        if not isinstance(player, Mapping):
            continue
        pid = int(player.get("id", player.get("player_id", player.get("track_id"))))
        for frame in player.get("frames", []):
            if isinstance(frame, Mapping):
                result[pid].add(frame_idx(frame, fps))
    return dict(result)


def mesh_player_frames(run_dir: Path) -> dict[int, set[int]]:
    payload = read(run_dir / "body_mesh_index/body_mesh_index.json")
    result: dict[int, set[int]] = defaultdict(set)
    for window in (payload or {}).get("windows", []):
        if not isinstance(window, Mapping):
            continue
        for player in window.get("players", []):
            if not isinstance(player, Mapping):
                continue
            pid = int(player.get("id", player.get("player_id")))
            for frame in player.get("frames", []):
                if isinstance(frame, Mapping) and isinstance(frame.get("frame_idx"), int):
                    result[pid].add(int(frame["frame_idx"]))
    return dict(result)


def greedy_nonoverlap_count(items: list[tuple[float, tuple[float, float, float, float]]], threshold: float = 0.3) -> int:
    def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0.0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1e-12)

    kept: list[tuple[float, float, float, float]] = []
    for _conf, bbox in sorted(items, key=lambda item: -item[0]):
        if all(iou(bbox, other) <= threshold for other in kept):
            kept.append(bbox)
    return len(kept)


def raw_counts_by_frame(run_dir: Path, total_frames: int) -> tuple[list[int], list[int], dict[str, Any] | None]:
    raw = read(run_dir / "raw_tracked_detections.json")
    calibration_path = run_dir / "court_calibration.json"
    if raw is None or not calibration_path.is_file():
        return [0] * total_frames, [0] * total_frames, None
    calibration = load_court_calibration(calibration_path)
    template = get_court_template(calibration.sport)
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    raw_counts = [0] * total_frames
    court_counts = [0] * total_frames
    for default_index, entry in enumerate(raw.get("frames", [])):
        if not isinstance(entry, Mapping):
            continue
        idx = int(entry.get("frame", entry.get("frame_idx", default_index)))
        if not (0 <= idx < total_frames):
            continue
        on_court: list[tuple[float, tuple[float, float, float, float]]] = []
        for det in entry.get("detections", []):
            if not isinstance(det, Mapping):
                continue
            if str(det.get("class", "person")).lower() not in {"person", "player", "0"}:
                continue
            bbox_raw = det.get("bbox") or det.get("bbox_xyxy")
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue
            bbox = tuple(float(value) for value in bbox_raw)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            raw_counts[idx] += 1
            conf = float(det.get("conf", det.get("confidence", 1.0)))
            person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=conf)
            x, y = person.foot_world_xy
            if -half_width <= x <= half_width and -half_length <= y <= half_length:
                on_court.append((conf, bbox))
        court_counts[idx] = greedy_nonoverlap_count(on_court)
    ceiling = raw_pool_four_player_ceiling(raw, calibration=calibration, expected_players=4)
    return raw_counts, court_counts, ceiling


def artifact_ball_frames(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        ball = payload.get("ball")
        frames = ball.get("frames", []) if isinstance(ball, Mapping) else []
    return [frame for frame in frames if isinstance(frame, Mapping)]


def indexed_ball(frames: list[Mapping[str, Any]], fps: float) -> dict[int, Mapping[str, Any]]:
    return {frame_idx(frame, fps): frame for frame in frames}


def stage_statuses(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in (summary or {}).get("stages", []):
        if isinstance(stage, Mapping):
            name = stage.get("stage", stage.get("name", stage.get("id")))
            if name is not None:
                result[str(name)] = {key: stage.get(key) for key in ("status", "notes", "metrics", "wall_seconds")}
    return result


def analyze_run(run_id: str, run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    tracks = read(run_dir / "tracks.json")
    ball_2d = read(run_dir / "ball_track.json")
    arc = read(run_dir / "ball_track_arc_solved.json")
    filled = read(run_dir / "ball_track_physics_filled.json")
    world = read(run_dir / "virtual_world.json")
    gated = read(run_dir / "confidence_gated_world.json")
    frame_plan = read(run_dir / "frame_compute_plan.json")
    frame_schedule = read(run_dir / "process_video_frame_schedule.json")
    body_execution = read(run_dir / "body_compute_execution.json")
    skeleton = read(run_dir / "skeleton3d.json")
    paddle = read(run_dir / "racket_pose_estimate.json")
    manifest = read(run_dir / "replay_viewer_manifest.json")
    pipeline_summary = read(run_dir / "PIPELINE_SUMMARY.json")

    fps = float((tracks or ball_2d or world or {}).get("fps", 30.0))
    candidates = [
        len(artifact_ball_frames(ball_2d)),
        int((frame_plan or {}).get("frame_count", 0)),
        len((read(run_dir / "frame_times.json") or {}).get("frames", [])),
    ]
    total_frames = max(candidates)
    if total_frames <= 0:
        raise ValueError(f"cannot infer frame count for {run_dir}")

    track_pf = player_frames(tracks, fps)
    skeleton_pf = player_frames(skeleton, fps)
    paddle_pf = player_frames(paddle, fps)
    mesh_pf = mesh_player_frames(run_dir)
    raw_counts, raw_court_counts, raw_ceiling = raw_counts_by_frame(run_dir, total_frames)

    scheduled = {int(item["frame_idx"]) for item in (body_execution or {}).get("scheduled_frames", []) if isinstance(item, Mapping)}
    materialized = {
        int(path.stem.split("_")[-1])
        for path in (run_dir / "body_frames").glob("frame_*.jpg")
        if path.stem.split("_")[-1].isdigit()
    }

    ball2 = indexed_ball(artifact_ball_frames(ball_2d), fps)
    arc_by_frame = indexed_ball(artifact_ball_frames(arc), fps)
    filled_by_frame = indexed_ball(artifact_ball_frames(filled), fps)
    world_by_frame = indexed_ball(artifact_ball_frames(world), fps)
    gated_by_frame = indexed_ball(artifact_ball_frames(gated), fps)
    overlaid = apply_ball_track_arc_solved_overlay(copy.deepcopy(filled), copy.deepcopy(arc)) if filled and arc else filled
    overlay_by_frame = indexed_ball(artifact_ball_frames(overlaid), fps)
    overlay_meta = (overlaid or {}).get("arc_solved_overlay") if isinstance(overlaid, Mapping) else None

    segment_items = [item for item in (arc or {}).get("segments", []) if isinstance(item, Mapping)]
    segment_ids = [int(item["segment_id"]) for item in segment_items if isinstance(item.get("segment_id"), int)]
    segment_status = {int(item["segment_id"]): str(item.get("status") or "") for item in segment_items if isinstance(item.get("segment_id"), int)}
    arc_segment_frame_counts: Counter[int] = Counter()
    for item in arc_by_frame.values():
        solver = item.get("arc_solver")
        if isinstance(solver, Mapping) and isinstance(solver.get("segment_id"), int):
            arc_segment_frame_counts[int(solver["segment_id"])] += 1
    verdicts = ball_arc_segment_fail_closed_verdicts((arc or {}).get("segments"))
    suppressed_ids = {sid for sid, verdict in verdicts.items() if not verdict.get("trusted")}
    fit_ids = {sid for sid, status in segment_status.items() if status == "fit"}
    strict_ukf_adjacent_fit_ids: set[int] = set()
    for position, sid in enumerate(segment_ids):
        if sid not in suppressed_ids:
            continue
        neighbors = set(segment_ids[max(0, position - 1) : position]) | set(segment_ids[position + 1 : position + 2])
        if neighbors & fit_ids:
            strict_ukf_adjacent_fit_ids.add(sid)
    strict_ukf_upper_frames = sum(arc_segment_frame_counts[sid] for sid in strict_ukf_adjacent_fit_ids)
    fallback_ids = [sid for sid in segment_ids if segment_status.get(sid) == "fit_bvp_fallback"]
    tt3d_conversions_needed = max(0, len(fallback_ids) - 4)
    tt3d_gain_candidates = sorted(
        0 if sid not in suppressed_ids else arc_segment_frame_counts[sid] for sid in fallback_ids
    )
    current_failclosed_emitted = sum(item.get("world_xyz") is not None for item in overlay_by_frame.values())
    tt3d_gain_min = sum(tt3d_gain_candidates[:tt3d_conversions_needed])
    tt3d_gain_max = sum(tt3d_gain_candidates[-tt3d_conversions_needed:]) if tt3d_conversions_needed else 0

    rows: list[dict[str, Any]] = []
    for idx in range(total_frames):
        tracked_ids = sorted(pid for pid, frames in track_pf.items() if idx in frames)
        skeleton_ids = sorted(pid for pid, frames in skeleton_pf.items() if idx in frames)
        mesh_ids = sorted(pid for pid, frames in mesh_pf.items() if idx in frames)
        paddle_ids = sorted(pid for pid, frames in paddle_pf.items() if idx in frames)
        b2 = ball2.get(idx, {})
        ba = arc_by_frame.get(idx, {})
        bf = filled_by_frame.get(idx, {})
        bw = world_by_frame.get(idx, {})
        bg = gated_by_frame.get(idx, {})
        bo = overlay_by_frame.get(idx, {})
        rows.append(
            {
                "run_id": run_id,
                "clip_id": (pipeline_summary or {}).get("clip", run_id),
                "frame_idx": idx,
                "t_s": round(idx / fps, 6),
                "expected_players": 4,
                "raw_person_boxes": raw_counts[idx],
                "raw_on_court_nonoverlap_boxes": raw_court_counts[idx],
                "tracked_player_count": len(tracked_ids),
                "tracked_player_ids": ";".join(str(value) for value in tracked_ids),
                "missing_tracked_players_vs_4": max(0, 4 - len(tracked_ids)),
                "body_scheduled": int(idx in scheduled),
                "body_frame_materialized": int(idx in materialized),
                "mesh_player_count": len(mesh_ids),
                "mesh_player_ids": ";".join(str(value) for value in mesh_ids),
                "skeleton_player_count": len(skeleton_ids),
                "skeleton_player_ids": ";".join(str(value) for value in skeleton_ids),
                "paddle_player_count": len(paddle_ids),
                "paddle_player_ids": ";".join(str(value) for value in paddle_ids),
                "ball_2d_visible": int(bool(b2.get("visible"))),
                "ball_arc_world_xyz": int(ba.get("world_xyz") is not None),
                "ball_arc_band": ba.get("band"),
                "ball_arc_segment_id": (ba.get("arc_solver") or {}).get("segment_id") if isinstance(ba.get("arc_solver"), Mapping) else None,
                "ball_physics_world_xyz": int(bf.get("world_xyz") is not None),
                "ball_physics_source": bf.get("source"),
                "ball_physics_interpolated": int(bf.get("source") == "physics_interpolated"),
                "ball_world_xyz_artifact": int(bw.get("world_xyz") is not None),
                "ball_gated_world_xyz_artifact": int(bg.get("world_xyz") is not None),
                "ball_failclosed_world_xyz_recomputed": int(bo.get("world_xyz") is not None),
                "ball_failclosed_hidden_recomputed": int(bo.get("world_xyz") is None),
                "ball_confidence_band": (bg.get("confidence_provenance") or {}).get("band") if isinstance(bg.get("confidence_provenance"), Mapping) else None,
                "ball_display_band": (bg.get("confidence_provenance") or {}).get("display_band") if isinstance(bg.get("confidence_provenance"), Mapping) else None,
            }
        )

    player_rows: list[dict[str, Any]] = []
    player_ids = sorted(set(track_pf) | set(skeleton_pf) | set(mesh_pf) | set(paddle_pf))
    for pid in player_ids:
        track_n = len(track_pf.get(pid, set()))
        player_rows.append(
            {
                "run_id": run_id,
                "player_id": pid,
                "total_clip_frames": total_frames,
                "tracked_frames": track_n,
                "tracked_pct_clip": round(100.0 * track_n / total_frames, 3),
                "body_scheduled_tracked_frames": len(track_pf.get(pid, set()) & scheduled),
                "mesh_frames": len(mesh_pf.get(pid, set())),
                "mesh_pct_clip": round(100.0 * len(mesh_pf.get(pid, set())) / total_frames, 3),
                "mesh_pct_tracked": round(100.0 * len(mesh_pf.get(pid, set())) / track_n, 3) if track_n else 0.0,
                "skeleton_frames": len(skeleton_pf.get(pid, set())),
                "skeleton_pct_clip": round(100.0 * len(skeleton_pf.get(pid, set())) / total_frames, 3),
                "skeleton_pct_tracked": round(100.0 * len(skeleton_pf.get(pid, set())) / track_n, 3) if track_n else 0.0,
                "paddle_frames": len(paddle_pf.get(pid, set())),
                "paddle_pct_clip": round(100.0 * len(paddle_pf.get(pid, set())) / total_frames, 3),
                "paddle_pct_tracked": round(100.0 * len(paddle_pf.get(pid, set())) / track_n, 3) if track_n else 0.0,
            }
        )

    tracked_counts = Counter(row["tracked_player_count"] for row in rows)
    raw_ceiling_frames = sum(row["raw_on_court_nonoverlap_boxes"] >= 4 for row in rows)
    raw_person_frames_with_4 = sum(row["raw_person_boxes"] >= 4 for row in rows)
    selected_four_frames = sum(row["tracked_player_count"] >= 4 for row in rows)
    selection_loss_frames = sum(
        row["raw_on_court_nonoverlap_boxes"] >= 4 and row["tracked_player_count"] < 4 for row in rows
    )
    detector_limited_frames = sum(row["raw_on_court_nonoverlap_boxes"] < 4 for row in rows)
    body_summary = (body_execution or {}).get("summary", {})
    plan_summary = (frame_plan or {}).get("summary", {})
    plan_policy = (frame_plan or {}).get("mesh_coverage_policy", {})
    materialization_schedule = {
        int(value) for value in (frame_schedule or {}).get("frame_indexes", []) if isinstance(value, int)
    }
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "clip_id": (pipeline_summary or {}).get("clip", run_id),
        "pipeline_status": (pipeline_summary or {}).get("status"),
        "fps": fps,
        "total_frames": total_frames,
        "tracked_player_frame_count": sum(len(value) for value in track_pf.values()),
        "tracked_frames_with_4": selected_four_frames,
        "tracked_frames_with_4_pct": round(100.0 * selected_four_frames / total_frames, 3),
        "tracked_count_histogram": {str(key): value for key, value in sorted(tracked_counts.items())},
        "raw_on_court_ceiling_frames_with_4": raw_ceiling_frames,
        "raw_on_court_ceiling_pct": round(100.0 * raw_ceiling_frames / total_frames, 3),
        "raw_person_box_frames_with_4": raw_person_frames_with_4,
        "raw_person_box_frames_with_4_pct": round(100.0 * raw_person_frames_with_4 / total_frames, 3),
        "source_only_detector_limited_frames": detector_limited_frames,
        "source_only_selection_or_association_loss_frames": selection_loss_frames,
        "raw_pool_ceiling_helper": raw_ceiling,
        "body_schedule": {
            "body_execution_required_frame_count": len(scheduled),
            "materialized_frame_count": len(materialized),
            "body_execution_required_present_count": len(scheduled & materialized),
            "body_execution_required_missing_materialized": sorted(scheduled - materialized),
            "materialization_schedule_frame_count": len(materialization_schedule),
            "materialization_schedule_missing_on_disk": sorted(materialization_schedule - materialized),
            "materialized_extra_vs_materialization_schedule": sorted(materialized - materialization_schedule),
            "frame_schedule": frame_schedule,
            "execution_summary": body_summary,
            "plan_summary": plan_summary,
            "plan_policy": plan_policy,
        },
        "mesh_player_frame_count": sum(len(value) for value in mesh_pf.values()),
        "skeleton_player_frame_count": sum(len(value) for value in skeleton_pf.values()),
        "paddle_player_frame_count": sum(len(value) for value in paddle_pf.values()),
        "ball": {
            "2d_visible_frames": sum(row["ball_2d_visible"] for row in rows),
            "arc_world_xyz_frames": sum(row["ball_arc_world_xyz"] for row in rows),
            "physics_world_xyz_frames": sum(row["ball_physics_world_xyz"] for row in rows),
            "physics_interpolated_frames": sum(row["ball_physics_interpolated"] for row in rows),
            "artifact_world_xyz_frames": sum(row["ball_world_xyz_artifact"] for row in rows),
            "artifact_gated_world_xyz_frames": sum(row["ball_gated_world_xyz_artifact"] for row in rows),
            "failclosed_world_xyz_frames_recomputed": sum(row["ball_failclosed_world_xyz_recomputed"] for row in rows),
            "failclosed_hidden_frames_recomputed": sum(row["ball_failclosed_hidden_recomputed"] for row in rows),
            "visible_2d_without_failclosed_3d_frames": sum(
                row["ball_2d_visible"] and not row["ball_failclosed_world_xyz_recomputed"] for row in rows
            ),
            "visible_2d_without_failclosed_3d_pct_of_visible_2d": round(
                100.0
                * sum(row["ball_2d_visible"] and not row["ball_failclosed_world_xyz_recomputed"] for row in rows)
                / max(1, sum(row["ball_2d_visible"] for row in rows)),
                3,
            ),
            "strict_ukf_adjacent_fit_recovery_upper_bound": {
                "eligible_suppressed_segment_ids": sorted(strict_ukf_adjacent_fit_ids),
                "additional_frame_upper_bound": strict_ukf_upper_frames,
                "emitted_frame_upper_bound": sum(row["ball_failclosed_world_xyz_recomputed"] for row in rows)
                + strict_ukf_upper_frames,
                "note": "Counterfactual ceiling only: every eligible segment would still have to pass reprojection/spatial gates.",
            },
            "tt3d_fallback_below_5_counterfactual": {
                "current_fallback_segment_count": len(fallback_ids),
                "required_conversions_to_fewer_than_5": tt3d_conversions_needed,
                "additional_emitted_frame_range": [tt3d_gain_min, tt3d_gain_max],
                "emitted_frame_range": [current_failclosed_emitted + tt3d_gain_min, current_failclosed_emitted + tt3d_gain_max],
                "note": "Combinatorial bound only. Which segments become valid cannot be known before running and scoring joint-anchor search.",
            },
            "overlay_provenance_recomputed": overlay_meta,
            "arc_summary": (arc or {}).get("summary"),
            "physics_fill_coverage": ((filled or {}).get("physics_fill") or {}).get("coverage"),
            "arc_render_summary": (read(run_dir / "ball_arc_render.json") or {}).get("summary"),
        },
        "manifest": {
            "present": manifest is not None,
            "mesh_status": (manifest or {}).get("mesh_status"),
            "body_mesh_index_url": (manifest or {}).get("body_mesh_index_url"),
            "body_mesh_url": (manifest or {}).get("body_mesh_url"),
            "virtual_world_url": (manifest or {}).get("virtual_world_url"),
        },
        "stage_statuses": stage_statuses(pipeline_summary),
    }
    return rows, summary, player_rows


def segment_rows(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    arc = read(run_dir / "ball_track_arc_solved.json") or {}
    filled = read(run_dir / "ball_track_physics_filled.json") or {}
    verdicts = ball_arc_segment_fail_closed_verdicts(arc.get("segments"))
    overlaid = apply_ball_track_arc_solved_overlay(copy.deepcopy(filled), copy.deepcopy(arc)) or {}
    fps = float(arc.get("fps", filled.get("fps", 30.0)))
    arc_frames = indexed_ball(artifact_ball_frames(arc), fps)
    overlay_frames = indexed_ball(artifact_ball_frames(overlaid), fps)
    by_segment: dict[int, list[int]] = defaultdict(list)
    for idx, frame in arc_frames.items():
        solver = frame.get("arc_solver")
        if isinstance(solver, Mapping) and isinstance(solver.get("segment_id"), int):
            by_segment[int(solver["segment_id"])].append(idx)
    rows: list[dict[str, Any]] = []
    for segment in arc.get("segments", []):
        if not isinstance(segment, Mapping) or not isinstance(segment.get("segment_id"), int):
            continue
        sid = int(segment["segment_id"])
        indexes = sorted(by_segment.get(sid, []))
        verdict = verdicts.get(sid, {})
        rows.append(
            {
                "run_id": run_id,
                "segment_id": sid,
                "status": segment.get("status"),
                "frame_start": segment.get("frame_start"),
                "frame_end": segment.get("frame_end"),
                "attributed_frame_count": len(indexes),
                "arc_world_xyz_frames": sum(arc_frames[idx].get("world_xyz") is not None for idx in indexes),
                "failclosed_emitted_frames": sum(overlay_frames.get(idx, {}).get("world_xyz") is not None for idx in indexes),
                "failclosed_suppressed_frames": sum(overlay_frames.get(idx, {}).get("world_xyz") is None for idx in indexes),
                "failclosed_trusted": verdict.get("trusted"),
                "failclosed_reasons": ";".join(str(value) for value in verdict.get("reasons", [])),
                "inlier_count": segment.get("inlier_count"),
                "outlier_count": segment.get("outlier_count"),
                "max_reprojection_error_px": segment.get("max_reprojection_error_px"),
                "physical_sanity_status": (segment.get("physical_sanity") or {}).get("status") if isinstance(segment.get("physical_sanity"), Mapping) else None,
                "physical_sanity_violations": ";".join(str(value) for value in (segment.get("physical_sanity") or {}).get("violations", [])) if isinstance(segment.get("physical_sanity"), Mapping) else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_frames: list[dict[str, Any]] = []
    all_players: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    segments: list[dict[str, Any]] = []
    paddle_quality: dict[str, Any] = {}
    for run_id, run_dir in RUNS.items():
        rows, summary, players = analyze_run(run_id, run_dir)
        all_frames.extend(rows)
        all_players.extend(players)
        summaries[run_id] = summary
        segments.extend(segment_rows(run_id, run_dir))
        paddle = read(run_dir / "racket_pose_estimate.json")
        paddle_frames = [
            frame
            for player in (paddle or {}).get("players", [])
            if isinstance(player, Mapping)
            for frame in player.get("frames", [])
            if isinstance(frame, Mapping)
        ]
        confidences = sorted(float(frame.get("conf", 0.0)) for frame in paddle_frames)
        percentile = lambda fraction: confidences[min(len(confidences) - 1, int(round(fraction * (len(confidences) - 1))))] if confidences else None
        paddle_quality[run_id] = {
            "artifact_present": paddle is not None,
            "status": (paddle or {}).get("status"),
            "world_frame": (paddle or {}).get("world_frame"),
            "translation_unit": (paddle or {}).get("translation_unit"),
            "frame_count": len(paddle_frames),
            "confidence_min": confidences[0] if confidences else None,
            "confidence_p50": percentile(0.5),
            "confidence_p95": percentile(0.95),
            "confidence_max": confidences[-1] if confidences else None,
            "reprojection_error_populated_count": sum(frame.get("reprojection_error_px") is not None for frame in paddle_frames),
            "ambiguous_frame_count": sum(bool(frame.get("ambiguous")) for frame in paddle_frames),
            "source_counts": dict(Counter(str(frame.get("source")) for frame in paddle_frames)),
            "confidence_band_counts": dict(Counter(str((frame.get("confidence_provenance") or {}).get("band")) for frame in paddle_frames)),
            "summary": (paddle or {}).get("summary"),
            "warnings": (paddle or {}).get("warnings"),
            "blockers": (paddle or {}).get("blockers"),
        }

    incomplete_owner_world = ROOT / "runs/lanes/w7_critique_20260709/world/owner_critique_zwcth45s"
    summaries["w7_owner_world_incomplete_pull"] = {
        "run_dir": str(incomplete_owner_world.relative_to(ROOT)),
        "available_files": sorted(path.name for path in incomplete_owner_world.iterdir()),
        "coverage_computable": False,
        "reason": "Only PIPELINE_SUMMARY.json and frame_times.json were pulled; no tracks/BALL/BODY/world artifacts exist in this directory.",
    }

    write_csv(OUT / "per_frame_coverage.csv", all_frames)
    write_csv(OUT / "per_player_coverage.csv", all_players)
    write_csv(OUT / "ball_segment_fate.csv", segments)
    (OUT / "coverage_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "ball_segment_fate.json").write_text(json.dumps(segments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "paddle_quality.json").write_text(json.dumps(paddle_quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": list(summaries), "per_frame_rows": len(all_frames), "player_rows": len(all_players), "segment_rows": len(segments)}, indent=2))


if __name__ == "__main__":
    main()
