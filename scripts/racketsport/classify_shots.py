#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.shot_taxonomy import classify_shots_from_payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify PB-Vision-compatible shot/outcome taxonomy from ball arc solver outputs."
    )
    parser.add_argument("--run-dir", type=Path, help="Directory containing ball_track_arc_solved.json/events_selected.json.")
    parser.add_argument("--clip-id", help="Clip id. Defaults to arc payload clip_id or run-dir name.")
    parser.add_argument("--ball-track-arc-solved", type=Path, help="Path to ball_track_arc_solved.json.")
    parser.add_argument("--events-selected", type=Path, help="Path to events_selected.json.")
    parser.add_argument("--court-zones", type=Path, help="Path to court_zones.json.")
    parser.add_argument("--net-plane", type=Path, help="Path to net_plane.json.")
    parser.add_argument("--tracks", type=Path, help="Path to tracks.json.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output shots.json path.")
    parser.add_argument("--report-md", type=Path, help="Optional single-clip Markdown report path.")
    parser.add_argument("--min-shot-type-confidence", type=float, default=0.45)
    args = parser.parse_args(argv)

    try:
        arc_path = _resolve_required(args.ball_track_arc_solved, args.run_dir, "ball_track_arc_solved.json")
        events_path = _resolve_required(args.events_selected, args.run_dir, "events_selected.json")
        arc_payload = _read_json_object(arc_path, "ball arc solved")
        events_payload = _read_json_object(events_path, "events selected")

        source_dir = _source_run_dir(arc_payload)
        court_zones_path = _resolve_optional(args.court_zones, args.run_dir, source_dir, "court_zones.json")
        net_plane_path = _resolve_optional(args.net_plane, args.run_dir, source_dir, "net_plane.json")
        tracks_path = _resolve_optional(args.tracks, args.run_dir, source_dir, "tracks.json")
        clip_id = args.clip_id or str(arc_payload.get("clip_id") or (args.run_dir.name if args.run_dir else arc_path.parent.name))

        payload = classify_shots_from_payloads(
            clip_id=clip_id,
            ball_arc_payload=arc_payload,
            events_selected_payload=events_payload,
            court_zones_payload=_read_json_object(court_zones_path, "court zones") if court_zones_path else None,
            net_plane_payload=_read_json_object(net_plane_path, "net plane") if net_plane_path else None,
            tracks_payload=_read_json_object(tracks_path, "tracks") if tracks_path else None,
            min_shot_type_confidence=args.min_shot_type_confidence,
        )

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.report_md is not None:
            args.report_md.parent.mkdir(parents=True, exist_ok=True)
            args.report_md.write_text(_single_clip_report(payload), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: shot classification failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "shot_count": payload["summary"]["shot_count"],
                "classified_count": payload["summary"]["classified_count"],
                "abstained_count": payload["summary"]["abstained_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_required(explicit: Path | None, run_dir: Path | None, filename: str) -> Path:
    path = explicit or (run_dir / filename if run_dir is not None else None)
    if path is None:
        raise ValueError(f"{filename} path is required without --run-dir")
    if not path.exists():
        raise ValueError(f"required input does not exist: {path}")
    return path


def _resolve_optional(explicit: Path | None, run_dir: Path | None, source_dir: Path | None, filename: str) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if run_dir is not None:
        candidates.append(run_dir / filename)
    if source_dir is not None:
        candidates.append(source_dir / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if explicit is not None:
        raise ValueError(f"optional input was specified but does not exist: {explicit}")
    return None


def _source_run_dir(arc_payload: Mapping[str, Any]) -> Path | None:
    inputs = arc_payload.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    for key in ("net_plane", "court_calibration", "ball_track", "contact_windows", "rally_spans"):
        value = inputs.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists():
                return path.parent
    return None


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _single_clip_report(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Shot Taxonomy Report - {payload.get('clip_id')}",
        "",
        "- Scope: internal-val taxonomy artifact only; not ground truth and not BALL/SHOT promotion evidence.",
        f"- Shot count: {payload.get('summary', {}).get('shot_count')}",
        f"- Classified: {payload.get('summary', {}).get('classified_count')}",
        f"- Abstained: {payload.get('summary', {}).get('abstained_count')}",
        f"- Distribution: `{json.dumps(payload.get('summary', {}).get('shot_type_counts', {}), sort_keys=True)}`",
        "",
        "| # | frame | player | rally | type | outcome | speed mph | peak m | confidence |",
        "|---:|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for shot in payload.get("shots", []):
        if not isinstance(shot, Mapping):
            continue
        rally = shot.get("rally_index", {})
        rally_label = rally.get("label") if isinstance(rally, Mapping) else ""
        shot_type = shot.get("shot_type", "abstain")
        outcome = shot.get("outcome", {})
        outcome_call = outcome.get("call") if isinstance(outcome, Mapping) else ""
        lines.append(
            "| {idx} | {frame} | {player} | {rally} | {shot_type} | {outcome} | {speed:.3f} | {peak:.3f} | {conf:.3f} |".format(
                idx=rally.get("contact_index", "") if isinstance(rally, Mapping) else "",
                frame=shot.get("frame", ""),
                player=shot.get("player_id", ""),
                rally=rally_label,
                shot_type=shot_type,
                outcome=outcome_call,
                speed=float(shot.get("speed_mph", 0.0)),
                peak=float(shot.get("peak_height_m", 0.0)),
                conf=float(shot.get("confidence", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
