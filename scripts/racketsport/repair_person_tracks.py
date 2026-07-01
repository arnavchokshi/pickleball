#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.player_id_repair import (  # noqa: E402
    RepairConfig,
    repair_detection_payload_to_tracks,
    repair_tracks,
)
from threed.racketsport.player_track_overlay import render_player_track_overlay  # noqa: E402
from threed.racketsport.schemas import CourtCalibration, Tracks, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline exactly-4 player ID repair for existing TRK artifacts.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Candidate directory containing tracks/detections/metrics.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output candidate directory to create.")
    parser.add_argument("--input-kind", choices=("tracks", "detections"), default="tracks")
    parser.add_argument("--tracks", type=Path, default=None, help="Input tracks.json; defaults to source-dir/tracks.json.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="Input tracked_detections.json; defaults to source-dir/tracked_detections.json.",
    )
    parser.add_argument("--calibration", type=Path, default=None, help="court_calibration.json for detections input.")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("runs/eval0/prototype_gate_h100_v2"),
        help="Fallback root containing <clip>/court_calibration.json.",
    )
    parser.add_argument("--expected-players", type=int, default=4)
    parser.add_argument("--split-gap-frames", type=int, default=24)
    parser.add_argument("--max-merge-gap-frames", type=int, default=240)
    parser.add_argument("--max-gap-fill-frames", type=int, default=24)
    parser.add_argument("--max-merge-speed-m-s", type=float, default=9.0)
    parser.add_argument("--max-gap-fill-speed-m-s", type=float, default=7.0)
    parser.add_argument("--merge-distance-slack-m", type=float, default=1.25)
    parser.add_argument("--max-merge-cost", type=float, default=2.5)
    parser.add_argument("--min-fragment-frames", type=int, default=1)
    parser.add_argument("--max-fragments-for-global", type=int, default=160)
    parser.add_argument("--court-margin-m", type=float, default=0.0)
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument("--video", type=Path, default=None, help="Source video for optional overlay.")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    source_dir = args.source_dir
    out_dir = args.out_dir
    config = RepairConfig(
        expected_players=args.expected_players,
        split_gap_frames=args.split_gap_frames,
        max_merge_gap_frames=args.max_merge_gap_frames,
        max_gap_fill_frames=args.max_gap_fill_frames,
        max_merge_speed_m_s=args.max_merge_speed_m_s,
        max_gap_fill_speed_m_s=args.max_gap_fill_speed_m_s,
        merge_distance_slack_m=args.merge_distance_slack_m,
        max_merge_cost=args.max_merge_cost,
        min_fragment_frames=args.min_fragment_frames,
        max_fragments_for_global=args.max_fragments_for_global,
        court_margin_m=args.court_margin_m,
    )

    source_metrics = _read_json(source_dir / "metrics.json")
    t0 = time.perf_counter()
    try:
        if args.input_kind == "tracks":
            tracks_path = args.tracks or source_dir / "tracks.json"
            parsed = validate_artifact_file("tracks", tracks_path)
            if not isinstance(parsed, Tracks):
                raise ValueError(f"{tracks_path} did not parse as Tracks")
            repaired, summary = repair_tracks(parsed, config=config)
            input_paths = {"tracks_path": str(tracks_path)}
        else:
            detections_path = args.detections or source_dir / "tracked_detections.json"
            calibration_path = args.calibration or _discover_calibration(source_dir, args.calibration_root)
            if calibration_path is None:
                raise ValueError("detections input requires --calibration or a discoverable calibration-root")
            parsed_calibration = validate_artifact_file("court_calibration", calibration_path)
            if not isinstance(parsed_calibration, CourtCalibration):
                raise ValueError(f"{calibration_path} did not parse as CourtCalibration")
            repaired, summary = repair_detection_payload_to_tracks(
                _read_json(detections_path),
                parsed_calibration,
                config=config,
            )
            input_paths = {"detections_path": str(detections_path), "calibration_path": str(calibration_path)}
    except Exception as exc:
        print(f"track repair failed: {exc}", file=sys.stderr)
        return 1

    wall_time_s = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_out = out_dir / "tracks.json"
    tracks_out.write_text(json.dumps(repaired.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_player_id_repair_summary",
        "input_kind": args.input_kind,
        "source_dir": str(source_dir),
        "config": config.__dict__,
        "summary": summary.to_dict(),
        "input_paths": input_paths,
        "wall_time_s": round(wall_time_s, 6),
    }
    (out_dir / "repair_summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = _build_metrics(
        source_metrics,
        source_dir=source_dir,
        out_dir=out_dir,
        tracks=repaired,
        summary_payload=summary_payload,
        wall_time_s=wall_time_s,
    )
    if args.render_overlay:
        video_path = args.video or _video_from_metrics(source_metrics)
        if video_path is None:
            print("overlay skipped: missing --video and source metrics source_video", file=sys.stderr)
        else:
            overlay_path = out_dir / "track_overlay.mp4"
            counts = metrics.get("counts", {})
            render_player_track_overlay(
                video_path=video_path,
                tracks=repaired,
                output_path=overlay_path,
                max_frames=args.max_frames,
                bbox_scale_x=_number(counts.get("source_width"), 1.0) / _number(counts.get("calibration_width"), 1.0),
                bbox_scale_y=_number(counts.get("source_height"), 1.0) / _number(counts.get("calibration_height"), 1.0),
            )
            metrics["overlay_path"] = str(overlay_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(tracks_out)
    print(out_dir / "repair_summary.json")
    return 0


def _build_metrics(
    source_metrics: dict[str, Any],
    *,
    source_dir: Path,
    out_dir: Path,
    tracks: Tracks,
    summary_payload: dict[str, Any],
    wall_time_s: float,
) -> dict[str, Any]:
    metrics = dict(source_metrics)
    counts = dict(metrics.get("counts") if isinstance(metrics.get("counts"), dict) else {})
    counts.update(
        {
            "id_repair_input_detection_count": summary_payload["summary"]["input_detection_count"],
            "id_repair_kept_detection_count": summary_payload["summary"]["kept_detection_count"],
            "id_repair_fragment_count": summary_payload["summary"]["fragment_count"],
            "id_repair_selected_fragment_count": summary_payload["summary"]["selected_fragment_count"],
            "id_repair_dropped_fragment_count": summary_payload["summary"]["dropped_fragment_count"],
            "id_repair_synthetic_frame_count": summary_payload["summary"]["synthetic_frame_count"],
        }
    )
    metrics.update(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_tracker_candidate",
            "status": "id_repair_experiment_not_promoted",
            "source_candidate_dir": str(source_dir),
            "variant": out_dir.name,
            "tracks_path": str(out_dir / "tracks.json"),
            "player_count": len(tracks.players),
            "track_frame_count": sum(len(player.frames) for player in tracks.players),
            "track_lengths": {str(player.id): len(player.frames) for player in tracks.players},
            "wall_time_s": round(wall_time_s, 6),
            "effective_fps": None,
            "timing": {"wall_time_s": round(wall_time_s, 6), "effective_fps": None},
            "counts": counts,
            "id_repair": summary_payload,
        }
    )
    return metrics


def _discover_calibration(source_dir: Path, calibration_root: Path) -> Path | None:
    clip_id = source_dir.parent.name
    candidates = [
        calibration_root / clip_id / "court_calibration.json",
        calibration_root / clip_id / "e2e_rerun_20260628_144504" / "court_calibration.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _video_from_metrics(metrics: dict[str, Any]) -> Path | None:
    value = metrics.get("source_video")
    return Path(value) if isinstance(value, str) and value else None


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and float(value) > 0 else default


if __name__ == "__main__":
    raise SystemExit(main())
