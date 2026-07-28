"""Recompute the wolverine foot-slide gate metric before/after the
plant_phase_plausibility_20260728 fix, directly from already-produced
artifacts (read-only inputs, CPU only, no VM, no pipeline re-run).

Why this approach instead of re-running the pipeline from skeleton3d.json:
the diagnosis lane (wolverine_slide_diag_20260728) already found that a
standalone call to worldhmr._contact_gate_stream_for_skeleton3d() on a pulled
skeleton3d.json does not exactly reproduce the committed candidate/rejection
counts, because the real orchestrator applies additional pre-processing
(root-lock, foot-pin) before this stage that is not replicated standalone.
Each target run's body_grounding_quality.json already contains the exact,
already-computed foot_lock_gate_stream.phase_rows for the real pipeline call
(player_id, foot, frame range, frame_count, max_speed_mps, min_confidence,
existing rejection_reason). This script takes those real phase_rows and:

  "before" = the metric exactly as originally computed and committed
             (accepted iff the ORIGINAL rejection_reason is None)
  "after"  = the metric recomputed using the actual shipped
             threed.racketsport.foot_contact.phase_speed_duration_plausible()
             function (imported, not re-derived) applied on top of the
             existing rejection_reason: a phase newly excluded only if it
             was previously accepted (rejection_reason is None) AND its own
             frame_count/max_speed_mps now fail the plausibility check.

This is additive by construction: it can only ever REMOVE previously-accepted
phases from the metric, never add back a phase the original run already
excluded for an unrelated reason (confidence, agreement, penetration, ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.foot_contact import (  # noqa: E402
    ContactThresholds,
    phase_plausible_for_thresholds,
)

GATE_THRESHOLDS = ContactThresholds(split_speed_mps=ContactThresholds().enter_speed_mps)
GATE_THRESHOLD_M = 0.03

CLIPS = {
    "wolverine_tonight (FAIL, alwayson_fresh_wave_20260728)": (
        "runs/alwayson_fresh_wave_20260728/split/01_wolverine_indoor_diagonal_0_10/"
        "01_wolverine_indoor_diagonal_0_10/body_grounding_quality.json"
    ),
    "wolverine_baseline (PASS, court_people_3d_completion_fresh_20260724_60631f1)": (
        "runs/court_people_3d_completion_fresh_20260724_60631f1/wolverine/"
        "01_wolverine_indoor_diagonal_0_10/body_grounding_quality.json"
    ),
    "indoor_diagonal_tonight (FAIL 0.0546, alwayson_fresh_wave_20260728)": (
        "runs/alwayson_fresh_wave_20260728/split/07_replacement_indoor_diagonal_100_110/"
        "07_replacement_indoor_diagonal_100_110/body_grounding_quality.json"
    ),
    "indoor_diagonal_baseline (FAIL, court_people_3d_completion_fresh_20260724_60631f1)": (
        "runs/court_people_3d_completion_fresh_20260724_60631f1/indoor_diagonal/"
        "07_replacement_indoor_diagonal_100_110/body_grounding_quality.json"
    ),
    "outdoor_pbvision_tonight (PASS 0.0026, alwayson_fresh_wave_20260728)": (
        "runs/alwayson_fresh_wave_20260728/split/10_replacement_outdoor_pbvision_50_60/"
        "10_replacement_outdoor_pbvision_50_60/body_grounding_quality.json"
    ),
    "outdoor_pbvision_baseline (PASS, court_people_3d_completion_fresh_20260724_60631f1)": (
        "runs/court_people_3d_completion_fresh_20260724_60631f1/outdoor_pbvision/"
        "10_replacement_outdoor_pbvision_50_60/body_grounding_quality.json"
    ),
}


def recompute(path: Path) -> dict:
    payload = json.loads(path.read_text())
    grounding_metrics = payload.get("grounding_metrics", {})
    original_max_slide_m = grounding_metrics.get("max_foot_lock_slide_m")
    gate_stream = payload.get("foot_lock_gate_stream", {})
    rows = gate_stream.get("phase_rows", [])

    before_accepted = [row for row in rows if row.get("rejection_reason") is None]
    before_max = max((float(row["slide_m"]) for row in before_accepted), default=0.0)

    newly_rejected = []
    after_accepted = []
    for row in before_accepted:
        plausible = phase_plausible_for_thresholds(
            frame_count=int(row["frame_count"]),
            max_speed_mps=float(row["max_speed_mps"]),
            thresholds=GATE_THRESHOLDS,
        )
        if plausible:
            after_accepted.append(row)
        else:
            newly_rejected.append(row)
    after_max = max((float(row["slide_m"]) for row in after_accepted), default=0.0)

    return {
        "path": str(path),
        "total_phases": len(rows),
        "originally_accepted": len(before_accepted),
        "originally_committed_max_slide_m": original_max_slide_m,
        "recomputed_before_max_slide_m": before_max,
        "recomputed_after_max_slide_m": after_max,
        "gate_before_pass": before_max <= GATE_THRESHOLD_M,
        "gate_after_pass": after_max <= GATE_THRESHOLD_M,
        "newly_rejected_phases": [
            {
                "player_id": row["player_id"],
                "foot": row["foot"],
                "frames": f"{row['start_frame_index']}-{row['end_frame_index']}",
                "frame_count": row["frame_count"],
                "slide_m": row["slide_m"],
                "max_speed_mps": row["max_speed_mps"],
            }
            for row in newly_rejected
        ],
    }


def main() -> None:
    print(
        f"gate_thresholds: enter_speed_mps={GATE_THRESHOLDS.enter_speed_mps} "
        f"plant_plausibility_min_frames={GATE_THRESHOLDS.plant_plausibility_min_frames} "
        f"plant_plausibility_speed_fraction={GATE_THRESHOLDS.plant_plausibility_speed_fraction} "
        f"-> speed_ceiling_mps={GATE_THRESHOLDS.enter_speed_mps * GATE_THRESHOLDS.plant_plausibility_speed_fraction:.4f}"
    )
    print()
    results = {}
    for name, rel_path in CLIPS.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"=== {name} === MISSING: {path}")
            continue
        result = recompute(path)
        results[name] = result
        print(f"=== {name} ===")
        print(f"  committed max_foot_lock_slide_m         : {result['originally_committed_max_slide_m']!r}")
        print(f"  recomputed BEFORE (== committed, sanity) : {result['recomputed_before_max_slide_m']:.10f}  gate_pass={result['gate_before_pass']}")
        print(f"  recomputed AFTER  (with fix)             : {result['recomputed_after_max_slide_m']:.10f}  gate_pass={result['gate_after_pass']}")
        if result["newly_rejected_phases"]:
            print(f"  newly rejected ({len(result['newly_rejected_phases'])}):")
            for phase in result["newly_rejected_phases"]:
                print(f"    - {phase}")
        else:
            print("  newly rejected (0): fix made no change for this clip")
        print()

    out_path = Path(__file__).parent / "recompute_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
