#!/usr/bin/env python3
"""Track L ghost diagnosis — per-FP forensics on the pulled variant-P artifacts.

Imports the FROZEN scorer's own internals (_tracks_to_predictions, _match_frame,
_best_real_player_iou, _off_court_excess_m, _overlaps_ignored, _inside_image_bounds)
so every event enumerated here is definitionally identical to what the frozen card
counted. CPU-only artifact forensics; produces NO card rows.

Usage: python3 diagnose_ghosts.py <clip_id> <tracks.json> <person_ground_truth.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/arnavchokshi/Desktop/pickleball")
sys.path.insert(0, str(REPO))

from threed.racketsport.person_track_gt_scoring import (  # noqa: E402
    DEFAULT_OFF_COURT_APRON_MARGIN_M,
    _best_real_player_iou,
    _inside_image_bounds,
    _off_court_excess_m,
    _overlaps_ignored,
    _tracks_to_predictions,
)
from threed.racketsport.person_track_gt_scoring import _match_frame  # noqa: E402
from threed.racketsport.court_templates import get_court_template  # noqa: E402
from threed.racketsport.person_track_gt_scoring import (  # noqa: E402
    PersonGroundTruth,
    Tracks,
)


def main() -> None:
    clip_id, tracks_path, gt_path, out_path = sys.argv[1:5]
    ground_truth = PersonGroundTruth.model_validate_json(Path(gt_path).read_text())
    tracks = Tracks.model_validate_json(Path(tracks_path).read_text())

    predictions, prediction_world, _outside = _tracks_to_predictions(
        ground_truth=ground_truth,
        tracks=tracks,
        candidate="diag",
        bbox_scale_x=1.0,
        bbox_scale_y=1.0,
    )
    template = get_court_template("pickleball")
    half_w = template.width_m / 2.0
    half_l = template.length_m / 2.0
    expected = 4
    iou_threshold = 0.5

    gt_by_frame = {f.frame_index: [l for l in f.labels if not l.ignored] for f in ground_truth.frames}
    ign_by_frame = {f.frame_index: [l for l in f.labels if l.ignored] for f in ground_truth.frames}
    pred_by_frame = {f.frame_index: f.detections for f in predictions.frames}

    events = []  # every unmatched, non-ignored prediction
    per_track_frames: dict[int, list[int]] = defaultdict(list)
    per_track_conf: dict[int, list[float]] = defaultdict(list)
    per_track_excess: dict[int, list[float]] = defaultdict(list)
    per_track_matched: dict[int, int] = defaultdict(int)

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_labels = gt_by_frame.get(frame_index, [])
        ign_labels = ign_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        frame_matches = _match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)
        matched_pred = {pi for _, pi, _ in frame_matches}
        matched_gt_by_pred = {pi: gi for gi, pi, _ in frame_matches}
        world_entries = prediction_world.get(frame_index, [])
        full_gt = len(gt_labels) >= expected

        for pi, pred in enumerate(pred_labels):
            track_id = None
            excess = None
            world_xy = None
            if pi < len(world_entries):
                track_id, world_xy = world_entries[pi]
                excess = _off_court_excess_m(world_xy, half_width_m=half_w, half_length_m=half_l)
            if track_id is not None:
                per_track_frames[track_id].append(frame_index)
                per_track_conf[track_id].append(float(pred.confidence))
                if excess is not None:
                    per_track_excess[track_id].append(excess)
            if pi in matched_pred:
                if track_id is not None:
                    per_track_matched[track_id] += 1
                continue
            if _overlaps_ignored(pred.bbox_xywh, ign_labels, threshold=iou_threshold):
                continue
            best_iou = _best_real_player_iou(pred.bbox_xywh, gt_labels)
            if best_iou > 0.0:
                category = "near_miss"
            elif not full_gt:
                category = "no_gt_frame"
            elif _inside_image_bounds(pred.bbox_xywh, image_width=None, image_height=None):
                category = "TRUE_SPECTATOR"
            else:
                category = "outside_image"
            events.append(
                {
                    "frame": frame_index,
                    "track_id": track_id,
                    "category": category,
                    "best_real_iou": round(best_iou, 4),
                    "conf": round(float(pred.confidence), 4),
                    "bbox_xywh": [round(v, 1) for v in pred.bbox_xywh],
                    "world_xy": [round(v, 3) for v in world_xy] if world_xy else None,
                    "off_court_excess_m": round(excess, 3) if excess is not None else None,
                    "gt_count_on_frame": len(gt_labels),
                    "matched_gt_index": matched_gt_by_pred.get(pi),
                }
            )

    ghost = [e for e in events if e["category"] == "TRUE_SPECTATOR"]
    summary = {
        "clip_id": clip_id,
        "tracks_path": tracks_path,
        "counts_by_category": {
            c: sum(1 for e in events if e["category"] == c)
            for c in ("near_miss", "no_gt_frame", "TRUE_SPECTATOR", "outside_image")
        },
        "true_spectator_events": ghost,
        "per_track": {
            str(tid): {
                "frames_present": len(per_track_frames[tid]),
                "frames_matched": per_track_matched[tid],
                "match_rate": round(per_track_matched[tid] / max(1, len(per_track_frames[tid])), 4),
                "frame_range": [min(per_track_frames[tid]), max(per_track_frames[tid])],
                "conf_min_med_max": [
                    round(min(per_track_conf[tid]), 3),
                    round(sorted(per_track_conf[tid])[len(per_track_conf[tid]) // 2], 3),
                    round(max(per_track_conf[tid]), 3),
                ],
                "excess_m_max": round(max(per_track_excess[tid]), 3) if per_track_excess[tid] else None,
                "excess_m_frames_gt0": sum(1 for e in per_track_excess[tid] if e > 0),
            }
            for tid in sorted(per_track_frames)
        },
        "all_unmatched_events": events,
    }
    Path(out_path).write_text(json.dumps(summary, indent=1))
    print(f"clip={clip_id} categories={summary['counts_by_category']}")
    for e in ghost:
        print(
            f"  GHOST frame={e['frame']} track={e['track_id']} conf={e['conf']} "
            f"world_xy={e['world_xy']} excess_m={e['off_court_excess_m']} bbox={e['bbox_xywh']}"
        )


if __name__ == "__main__":
    main()
