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

from threed.racketsport.player_source_selection import (  # noqa: E402
    SourceSelectionConfig,
    source_select_global_four_player_tracks,
    source_select_four_player_tracks,
)
from threed.racketsport.schemas import CourtCalibration, Tracks, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Source-only four-player selection from tracked_detections.json using model-output seed tracks."
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Candidate directory containing source artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output candidate directory to create.")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to source-dir/tracked_detections.json.")
    parser.add_argument("--seed-tracks", type=Path, default=None, help="Defaults to source-dir/tracks.json.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json for source detections.")
    parser.add_argument("--embedding-export", type=Path, default=None, help="Optional source-only person ReID embedding export JSON.")
    parser.add_argument(
        "--mode",
        choices=("global_four_player", "seeded_gap_fill"),
        default="global_four_player",
        help="global_four_player optimizes all source frames; seeded_gap_fill preserves the old conservative hole-fill path.",
    )
    parser.add_argument("--expected-players", type=int, default=4)
    parser.add_argument("--min-detection-conf", type=float, default=0.0)
    parser.add_argument("--court-margin-m", type=float, default=0.0)
    parser.add_argument("--max-gap-fill-frames", type=int, default=30)
    parser.add_argument("--max-fill-distance-m", type=float, default=1.25)
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-global-step-m", type=float, default=2.5)
    parser.add_argument("--continuity-weight", type=float, default=1.0)
    parser.add_argument("--source-id-switch-penalty", type=float, default=1.5)
    parser.add_argument("--seed-prior-weight", type=float, default=0.75)
    parser.add_argument("--cardinality-gap-penalty", type=float, default=8.0)
    parser.add_argument("--confidence-reward-weight", type=float, default=0.25)
    parser.add_argument("--embedding-weight", type=float, default=0.0)
    parser.add_argument(
        "--embedding-bbox-scale",
        type=float,
        default=1.0,
        help="Scale applied to detection bboxes before comparing them to embedding-export bboxes.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    detections_path = args.detections or source_dir / "tracked_detections.json"
    seed_tracks_path = args.seed_tracks or source_dir / "tracks.json"
    config = SourceSelectionConfig(
        expected_players=args.expected_players,
        min_detection_conf=args.min_detection_conf,
        court_margin_m=args.court_margin_m,
        max_gap_fill_frames=args.max_gap_fill_frames,
        max_fill_distance_m=args.max_fill_distance_m,
        overlap_iou_threshold=args.overlap_iou_threshold,
        max_global_step_m=args.max_global_step_m,
        continuity_weight=args.continuity_weight,
        source_id_switch_penalty=args.source_id_switch_penalty,
        seed_prior_weight=args.seed_prior_weight,
        cardinality_gap_penalty=args.cardinality_gap_penalty,
        confidence_reward_weight=args.confidence_reward_weight,
        embedding_weight=args.embedding_weight,
        embedding_bbox_scale=args.embedding_bbox_scale,
    )

    try:
        parsed_calibration = validate_artifact_file("court_calibration", args.calibration)
        if not isinstance(parsed_calibration, CourtCalibration):
            raise ValueError(f"{args.calibration} did not parse as CourtCalibration")
        parsed_tracks = validate_artifact_file("tracks", seed_tracks_path)
        if not isinstance(parsed_tracks, Tracks):
            raise ValueError(f"{seed_tracks_path} did not parse as Tracks")
        detections_payload = _read_json(detections_path)
        embedding_payload = _read_json(args.embedding_export) if args.embedding_export is not None else None
        t0 = time.perf_counter()
        if args.mode == "global_four_player":
            selected, summary = source_select_global_four_player_tracks(
                detections_payload,
                parsed_calibration,
                seed_tracks=parsed_tracks,
                embedding_payload=embedding_payload,
                config=config,
            )
        else:
            selected, summary = source_select_four_player_tracks(
                detections_payload,
                parsed_calibration,
                seed_tracks=parsed_tracks,
                embedding_payload=embedding_payload,
                config=config,
            )
        wall_time_s = time.perf_counter() - t0
    except Exception as exc:
        print(f"source track selection failed: {exc}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = args.out_dir / "tracks.json"
    tracks_path.write_text(json.dumps(selected.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_source_person_selection_summary",
        "status": "source_only_selection_not_promoted",
        "source_only": True,
        "uses_cvat_labels": False,
        "promote_trk": False,
        "mode": args.mode,
        "source_dir": str(source_dir),
        "config": config.__dict__,
        "input_paths": {
            "detections_path": str(detections_path),
            "seed_tracks_path": str(seed_tracks_path),
            "calibration_path": str(args.calibration),
            "embedding_export_path": str(args.embedding_export) if args.embedding_export is not None else None,
        },
        "summary": summary.to_dict(),
        "wall_time_s": round(wall_time_s, 6),
    }
    (args.out_dir / "source_selection_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = _candidate_metrics(
        source_dir / "metrics.json",
        source_dir=source_dir,
        out_dir=args.out_dir,
        tracks=selected,
        summary_payload=summary_payload,
        wall_time_s=wall_time_s,
    )
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(tracks_path)
    print(args.out_dir / "source_selection_summary.json")
    return 0


def _candidate_metrics(
    source_metrics_path: Path,
    *,
    source_dir: Path,
    out_dir: Path,
    tracks: Tracks,
    summary_payload: dict[str, Any],
    wall_time_s: float,
) -> dict[str, Any]:
    source_metrics = _read_json(source_metrics_path) if source_metrics_path.exists() else {}
    metrics = dict(source_metrics)
    counts = dict(metrics.get("counts") if isinstance(metrics.get("counts"), dict) else {})
    counts.update(
        {
            "source_selection_candidate_detections": summary_payload["summary"]["source_candidate_detections"],
            "source_selection_candidate_kept": summary_payload["summary"]["source_candidate_kept"],
            "source_selection_seed_frame_count": summary_payload["summary"]["seed_frame_count"],
            "source_selection_filled_frame_count": summary_payload["summary"]["filled_frame_count"],
        }
    )
    metrics.update(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_tracker_candidate",
            "status": "source_selection_experiment_not_promoted",
            "source_only": True,
            "uses_cvat_labels": False,
            "promote_trk": False,
            "mode": summary_payload["mode"],
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
            "source_selection": summary_payload,
        }
    )
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
