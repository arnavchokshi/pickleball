#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report read-only person identity/TRK failure symptoms.")
    parser.add_argument("--tracks", type=Path, required=True, help="Input tracks.json artifact.")
    parser.add_argument("--sidecar", type=Path, help="Optional native2D/SAM3D sidecar keyed by player IDs.")
    parser.add_argument("--court-membership", type=Path, help="Optional person court-membership artifact.")
    parser.add_argument("--sam3d-identity-evidence", type=Path, help="Optional SAM3D identity evidence artifact.")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--teleport-speed-mps", type=float, default=10.0)
    args = parser.parse_args(argv)

    try:
        tracks = _read_json(args.tracks)
        report = build_identity_failure_report(
            tracks_payload=tracks,
            tracks_path=args.tracks,
            clip_id=args.clip_id,
            sidecar_payload=_read_optional_json(args.sidecar),
            court_membership_payload=_read_optional_json(args.court_membership),
            sam3d_identity_evidence_payload=_read_optional_json(args.sam3d_identity_evidence),
            teleport_speed_mps=args.teleport_speed_mps,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.out_md.write_text(render_identity_failure_markdown(report), encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: identity failure report failed: {exc}", file=sys.stderr)
        return 1

    print(args.out_json)
    print(args.out_md)
    return 0


def build_identity_failure_report(
    *,
    tracks_payload: Mapping[str, Any],
    tracks_path: Path,
    clip_id: str | None,
    sidecar_payload: Mapping[str, Any] | None = None,
    court_membership_payload: Mapping[str, Any] | None = None,
    sam3d_identity_evidence_payload: Mapping[str, Any] | None = None,
    teleport_speed_mps: float = 10.0,
) -> dict[str, Any]:
    fps = _fps(tracks_payload)
    track_frames = _track_frames(tracks_payload)
    speed_summary = _speed_teleports(track_frames, fps=fps, threshold=teleport_speed_mps)
    constant_summary = _constant_speed_spans(track_frames, fps=fps)
    sidecar_summary = _sidecar_mismatches(track_frames, sidecar_payload, fps=fps)
    membership_summary = _court_membership_violations(court_membership_payload)
    body_summary = _body_sam3d_risks(sam3d_identity_evidence_payload)
    watched = _watched_failures(
        clip_id=clip_id,
        speed_summary=speed_summary,
        constant_summary=constant_summary,
        membership_summary=membership_summary,
        body_summary=body_summary,
    )
    coverage = _coverage(track_frames)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_identity_failure_report",
        "source_only": True,
        "uses_cvat_labels": False,
        "clip_id": clip_id,
        "tracks_path": str(tracks_path),
        "track_summary": {
            "track_count": len(track_frames),
            "fps": fps,
            "per_track_frame_count": {str(player_id): len(frames) for player_id, frames in sorted(track_frames.items())},
            "coverage": coverage,
        },
        "speed_teleport_summary": speed_summary,
        "constant_speed_summary": constant_summary,
        "id_sidecar_mismatch_summary": sidecar_summary,
        "court_membership_violation_summary": membership_summary,
        "side_role_contradiction_summary": _side_role_contradictions(track_frames),
        "body_sam3d_risk_summary": body_summary,
        "watched_failures": watched,
    }


def render_identity_failure_markdown(report: Mapping[str, Any]) -> str:
    watched = report["watched_failures"]
    lines = [
        "# Person Identity Failure Report",
        "",
        f"- clip: `{report.get('clip_id')}`",
        f"- tracks: `{report.get('tracks_path')}`",
        f"- track count: `{report['track_summary']['track_count']}`",
        f"- teleport count: `{report['speed_teleport_summary']['teleport_count']}`",
        f"- constant-speed spans: `{report['constant_speed_summary']['constant_speed_span_count']}`",
        f"- sidecar mismatches: `{report['id_sidecar_mismatch_summary']['mismatch_count']}`",
        f"- inherited SAM3D/BODY anchor risks: `{report['body_sam3d_risk_summary']['inherited_anchor_risk_count']}`",
        f"- SAM3D root/track residual conflicts: `{report['body_sam3d_risk_summary']['root_track_residual_conflict_count']}`",
        "",
        "## Required Watchpoints",
        "",
        f"- Outdoor p2 frame 813: `{watched['outdoor_p2_frame_813']['status']}`",
        f"- IMG_1605 p3/p4 adjacent/constant-speed symptoms: `{watched['img1605_p3_p4_adjacent_constant_speed']['status']}`",
        "",
    ]
    return "\n".join(lines)


def _track_frames(payload: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    fps = _fps(payload)
    out: dict[int, list[dict[str, Any]]] = {}
    players = payload.get("players")
    if not isinstance(players, list):
        raise ValueError("tracks payload must contain players")
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = _int_or_none(player.get("id"))
        if player_id is None:
            continue
        rows: list[dict[str, Any]] = []
        frames = player.get("frames")
        for frame in frames if isinstance(frames, list) else []:
            if not isinstance(frame, Mapping):
                continue
            frame_idx = _frame_idx(frame, fps=fps)
            bbox = _sidecar_bbox(frame)
            world = _world_xy(frame.get("world_xy"))
            if frame_idx is None or bbox is None:
                continue
            rows.append(
                {
                    "player_id": player_id,
                    "frame_idx": frame_idx,
                    "t": _time(frame, frame_idx=frame_idx, fps=fps),
                    "bbox": bbox,
                    "world_xy": world,
                    "side": player.get("side"),
                    "role": player.get("role"),
                }
            )
        out[player_id] = sorted(rows, key=lambda row: row["frame_idx"])
    return out


def _speed_teleports(track_frames: Mapping[int, list[dict[str, Any]]], *, fps: float, threshold: float) -> dict[str, Any]:
    teleports = []
    for player_id, rows in track_frames.items():
        for previous, current in zip(rows, rows[1:]):
            if previous["world_xy"] is None or current["world_xy"] is None:
                continue
            dt = current["t"] - previous["t"]
            if dt <= 0.0:
                dt = (current["frame_idx"] - previous["frame_idx"]) / fps
            if dt <= 0.0:
                continue
            distance = math.hypot(current["world_xy"][0] - previous["world_xy"][0], current["world_xy"][1] - previous["world_xy"][1])
            speed = distance / dt
            if speed > threshold:
                teleports.append(
                    {
                        "player_id": player_id,
                        "from_frame": previous["frame_idx"],
                        "to_frame": current["frame_idx"],
                        "from_t": round(previous["t"], 6),
                        "to_t": round(current["t"], 6),
                        "speed_m_s": round(speed, 6),
                        "distance_m": round(distance, 6),
                    }
                )
    return {"teleport_speed_threshold_m_s": threshold, "teleport_count": len(teleports), "teleports": teleports}


def _constant_speed_spans(
    track_frames: Mapping[int, list[dict[str, Any]]],
    *,
    fps: float,
    min_speed_mps: float = 0.5,
    min_steps: int = 4,
    tolerance_mps: float = 0.01,
) -> dict[str, Any]:
    spans = []
    for player_id, rows in track_frames.items():
        steps = []
        for previous, current in zip(rows, rows[1:]):
            if previous["world_xy"] is None or current["world_xy"] is None:
                continue
            dt = current["t"] - previous["t"]
            if dt <= 0:
                dt = (current["frame_idx"] - previous["frame_idx"]) / fps
            if dt <= 0:
                continue
            speed = math.hypot(current["world_xy"][0] - previous["world_xy"][0], current["world_xy"][1] - previous["world_xy"][1]) / dt
            steps.append((previous["frame_idx"], current["frame_idx"], speed))
        if len(steps) < min_steps:
            continue
        current_steps: list[tuple[int, int, float]] = []
        for step in steps:
            if step[2] < min_speed_mps:
                if len(current_steps) >= min_steps:
                    spans.append(_constant_span(player_id, current_steps))
                current_steps = []
                continue
            if not current_steps or abs(step[2] - current_steps[-1][2]) <= tolerance_mps:
                current_steps.append(step)
            else:
                if len(current_steps) >= min_steps:
                    spans.append(_constant_span(player_id, current_steps))
                current_steps = [step]
        if len(current_steps) >= min_steps:
            spans.append(_constant_span(player_id, current_steps))
    return {"constant_speed_span_count": len(spans), "spans": spans}


def _constant_span(player_id: int, steps: list[tuple[int, int, float]]) -> dict[str, Any]:
    speeds = [step[2] for step in steps]
    return {
        "player_id": player_id,
        "start_frame": steps[0][0],
        "end_frame": steps[-1][1],
        "step_count": len(steps),
        "speed_m_s": round(sum(speeds) / len(speeds), 6),
        "speed_range_m_s": round(max(speeds) - min(speeds), 6),
    }


def _sidecar_mismatches(
    track_frames: Mapping[int, list[dict[str, Any]]],
    sidecar_payload: Mapping[str, Any] | None,
    *,
    fps: float,
) -> dict[str, Any]:
    if not isinstance(sidecar_payload, Mapping):
        return {"mismatch_count": 0, "mismatches": []}
    track_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rows in track_frames.values():
        for row in rows:
            track_by_frame[int(row["frame_idx"])].append(row)
    votes: dict[int, Counter[int | None]] = defaultdict(Counter)
    for player in _players(sidecar_payload):
        sidecar_id = _int_or_none(player.get("id"))
        if sidecar_id is None:
            continue
        for frame in _frames(player):
            frame_idx = _frame_idx(frame, fps=fps)
            bbox = _bbox(frame.get("bbox"))
            if frame_idx is None or bbox is None:
                continue
            best = _best_iou_track(bbox, track_by_frame.get(frame_idx, []))
            votes[sidecar_id][best] += 1
    mismatches = []
    for sidecar_id, counter in sorted(votes.items()):
        if not counter:
            continue
        majority, count = counter.most_common(1)[0]
        if majority is not None and majority != sidecar_id:
            mismatches.append({"sidecar_id": sidecar_id, "majority_track_id": majority, "vote_count": count})
    return {"mismatch_count": len(mismatches), "mismatches": mismatches}


def _court_membership_violations(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    violating = []
    if isinstance(payload, Mapping):
        fragments = payload.get("fragments")
        for fragment in fragments if isinstance(fragments, list) else []:
            if not isinstance(fragment, Mapping):
                continue
            class_name = str(fragment.get("membership_class") or "")
            if class_name in {"adjacent_court", "spectator_background", "projection_unknown"}:
                track_id = _int_or_none(fragment.get("source_tracker_id"))
                if track_id is None:
                    frag = str(fragment.get("fragment_id", ""))
                    if frag.startswith("track_"):
                        track_id = _int_or_none(frag.removeprefix("track_"))
                if track_id is not None:
                    violating.append(track_id)
    return {"violation_count": len(set(violating)), "violating_track_ids": sorted(set(violating))}


def _body_sam3d_risks(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    rows = payload.get("body_observations") if isinstance(payload, Mapping) else None
    risks = []
    root_conflicts = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        flags = [str(flag) for flag in row.get("risk_flags", [])] if isinstance(row.get("risk_flags"), list) else []
        residual = _number_or_none(row.get("root_track_residual_m"))
        has_root_conflict = residual is not None and residual > 1.0
        if has_root_conflict:
            root_conflicts.append(row)
        if row.get("transl_world_independent") is False or flags or has_root_conflict:
            risks.append(
                {
                    "body_observation_id": row.get("body_observation_id"),
                    "frame_idx": _int_or_none(row.get("frame_idx")),
                    "player_id": _int_or_none(row.get("player_id")),
                    "root_track_residual_m": residual,
                    "risk_flags": flags,
                }
            )
    inherited_count = sum(1 for row in risks if "placement_track_world_xy_anchor" in row["risk_flags"])
    return {
        "inherited_anchor_risk_count": inherited_count,
        "root_track_residual_conflict_count": len(root_conflicts),
        "risks": risks,
    }


def _watched_failures(
    *,
    clip_id: str | None,
    speed_summary: Mapping[str, Any],
    constant_summary: Mapping[str, Any],
    membership_summary: Mapping[str, Any],
    body_summary: Mapping[str, Any],
) -> dict[str, Any]:
    outdoor_flagged = any(
        int(row.get("player_id", -1)) == 2 and int(row.get("to_frame", -1)) == 813
        for row in speed_summary.get("teleports", [])
        if isinstance(row, Mapping)
    ) or any(
        int(row.get("player_id", -1)) == 2 and int(row.get("frame_idx", -1)) == 813
        for row in body_summary.get("risks", [])
        if isinstance(row, Mapping)
    )
    violating = set(membership_summary.get("violating_track_ids", []))
    constant_players = {row.get("player_id") for row in constant_summary.get("spans", []) if isinstance(row, Mapping)}
    img_applicable = clip_id is None or "img_1605" in clip_id.lower() or "owner_img_1605" in clip_id.lower()
    img_flagged = img_applicable and (bool({3, 4}.intersection(violating)) or 4 in constant_players)
    return {
        "outdoor_p2_frame_813": {
            "status": "flagged" if outdoor_flagged else "not_observed",
            "expected_frame_idx": 813,
            "expected_t_s": 13.55,
        },
        "img1605_p3_p4_adjacent_constant_speed": {
            "status": "flagged" if img_flagged else ("not_observed" if img_applicable else "not_applicable"),
            "expected_track_ids": [3, 4],
        },
    }


def _side_role_contradictions(track_frames: Mapping[int, list[dict[str, Any]]]) -> dict[str, Any]:
    contradictions = []
    for player_id, rows in track_frames.items():
        sides = {row.get("side") for row in rows if row.get("side") is not None}
        roles = {row.get("role") for row in rows if row.get("role") is not None}
        if len(sides) > 1 or len(roles) > 1:
            contradictions.append({"player_id": player_id, "sides": sorted(sides), "roles": sorted(roles)})
    return {"contradiction_count": len(contradictions), "contradictions": contradictions}


def _coverage(track_frames: Mapping[int, list[dict[str, Any]]]) -> dict[str, Any]:
    all_frames = sorted({row["frame_idx"] for rows in track_frames.values() for row in rows})
    if not all_frames:
        return {"observed_frame_count": 0, "span_frame_count": 0, "coverage_fraction": 0.0}
    span = all_frames[-1] - all_frames[0] + 1
    return {"observed_frame_count": len(all_frames), "span_frame_count": span, "coverage_fraction": round(len(all_frames) / span, 6)}


def _best_iou_track(bbox: tuple[float, float, float, float], rows: list[dict[str, Any]]) -> int | None:
    best_id = None
    best_iou = 0.0
    for row in rows:
        iou = _bbox_iou(bbox, row["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = int(row["player_id"])
    return best_id


def _sidecar_bbox(frame: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = _bbox(frame.get("bbox"))
    if bbox is not None:
        return bbox
    joints = frame.get("joints")
    if not isinstance(joints, list):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for joint in joints:
        if not isinstance(joint, Mapping):
            continue
        try:
            conf = float(joint.get("conf", joint.get("confidence", 1.0)))
            x = float(joint.get("x_px", joint.get("x")))
            y = float(joint.get("y_px", joint.get("y")))
        except (TypeError, ValueError):
            continue
        if conf <= 0.0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _players(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = payload.get("players")
    return [player for player in players if isinstance(player, Mapping)] if isinstance(players, list) else []


def _frames(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = player.get("frames")
    return [frame for frame in frames if isinstance(frame, Mapping)] if isinstance(frames, list) else []


def _fps(payload: Mapping[str, Any]) -> float:
    try:
        fps = float(payload.get("fps") or 30.0)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0.0 else 30.0


def _frame_idx(frame: Mapping[str, Any], *, fps: float) -> int | None:
    for key in ("frame_idx", "frame", "frame_index"):
        if key in frame:
            return _int_or_none(frame.get(key))
    try:
        return int(round(float(frame.get("t")) * fps))
    except (TypeError, ValueError):
        return None


def _time(frame: Mapping[str, Any], *, frame_idx: int, fps: float) -> float:
    try:
        return float(frame.get("t"))
    except (TypeError, ValueError):
        return frame_idx / fps


def _bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _world_xy(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
