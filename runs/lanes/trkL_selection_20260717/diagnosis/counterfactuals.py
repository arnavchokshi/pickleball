#!/usr/bin/env python3
"""Track L counterfactual rows (CPU diagnostic, NOT card rows).

Scores fixed, VM-produced tracks.json variants with the frozen scorer:
  as_scored : sanity — must reproduce the card row exactly
  cf1       : strip track-4 synthetic bridge frames 45-86 (stitch veto, no recovery)
  cf3       : cf1 + slot re-bind (track 1's frames move into slot 4; track 4's
              post-stitch segment becomes a separate id) — simulated layer-B upper bound

Usage: python3 counterfactuals.py <wolverine_variantP_tracks.json> <frozen_gt.json>
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path("/Users/arnavchokshi/Desktop/pickleball")
sys.path.insert(0, str(REPO))

from threed.racketsport.person_track_gt_scoring import (  # noqa: E402
    PersonGroundTruth,
    Tracks,
    score_tracks_against_person_ground_truth,
)

KEYS = [
    "idf1",
    "four_player_coverage",
    "id_switches",
    "true_spectator_or_background_false_positives",
    "near_miss_false_positive_rate",
    "hota",
]


def score(gt: PersonGroundTruth, tracks_dict: dict, name: str) -> dict:
    tr = Tracks.model_validate(tracks_dict)
    row = score_tracks_against_person_ground_truth(
        ground_truth=gt, tracks=tr, candidate=name, tracks_path=name,
        iou_threshold=0.5, expected_players=4,
    )
    return {k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in KEYS}


def main() -> None:
    tracks_path, gt_path = sys.argv[1:3]
    gt = PersonGroundTruth.model_validate_json(Path(gt_path).read_text())
    raw = json.loads(Path(tracks_path).read_text())
    fps = raw["fps"]

    print("as_scored:", score(gt, raw, "as_scored"))

    cf1 = copy.deepcopy(raw)
    for p in cf1["players"]:
        if p["id"] == 4:
            p["frames"] = [fr for fr in p["frames"] if not (45 <= round(fr["t"] * fps) <= 86)]
    print("cf1_strip_bridge:", score(gt, cf1, "cf1"))

    cf3 = copy.deepcopy(raw)
    p1 = next(p for p in cf3["players"] if p["id"] == 1)
    p4 = next(p for p in cf3["players"] if p["id"] == 4)
    pre = [fr for fr in p4["frames"] if round(fr["t"] * fps) < 45]
    post = [fr for fr in p4["frames"] if round(fr["t"] * fps) > 86]
    p4["frames"] = sorted(pre + p1["frames"], key=lambda fr: fr["t"])
    tpl = {k: p4[k] for k in ("role", "role_original", "role_source", "side", "side_original", "side_source")}
    cf3["players"] = [p for p in cf3["players"] if p["id"] != 1]
    cf3["players"].append({"id": 5, "frames": post, **tpl})
    print("cf3_veto_plus_rebind:", score(gt, cf3, "cf3"))


if __name__ == "__main__":
    main()
