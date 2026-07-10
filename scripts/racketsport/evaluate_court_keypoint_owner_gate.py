"""CAL owner-clip gate evaluator.

Scores a frozen court-keypoint checkpoint (trained only on the external Roboflow corpus,
`runs/training_corpora_20260701/court/`) against the owner's reviewed CVAT court-keypoint
labels (`eval_clips/ball/*/labels/court_keypoints.json`).

This script is read-only inference + scoring: it never trains or mutates the checkpoint. It
exists as the single, pre-registered gate runner for CAL's "Held-out PCK@5px >= 0.95 per
viewpoint on the 4 independent CVAT frames" gate (`NORTH_STAR_ROADMAP.md`, Gate ladder). Per the binding
held-out eval protocol, a run against this script must be pre-registered in
`runs/manager/heldout_eval_ledger.md` (CAL section) BEFORE it is executed, and gets exactly one
run per pre-committed candidate.

It reports four numbers side by side (never just one):
  - raw_independent: raw per-frame PCK@5 on the 4 independently human-reviewed frames (PRIMARY).
  - raw_all: raw per-frame PCK@5 on all 32 rows, including owner-approved static-camera copies
    (SECONDARY).
  - aggregated_independent: static-camera per-clip median-aggregation PCK@5, scored against the
    4 independent frames (PRIMARY under the aggregation policy).
  - aggregated_all: same aggregation policy, scored against all 32 rows (SECONDARY).

A promotion claim may only cite whichever of raw/aggregated was pre-registered as the candidate's
evaluation mode -- see PREREGISTRATION.md in the run directory this was invoked from.

For each of the four modes above, the report's `gate_passed_per_viewpoint` field is the actual
gate ("PCK@5px >= 0.95 per viewpoint"): every clip present must individually clear the
threshold, not just the pooled average across all keypoints. `gate_passed_pooled` reports the
pooled-across-clips number for context; `gate_passed` aliases `gate_passed_per_viewpoint`, since
that is the real gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_court_keypoint_heatmap import (  # noqa: E402
    evaluate_checkpoint_against_real_labels,
    load_real_court_keypoint_labels,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--real-root",
        type=Path,
        default=ROOT / "eval_clips" / "ball",
        help="Root containing <clip>/labels/court_keypoints.json rows to score against.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pck-threshold-px", type=float, default=5.0)
    parser.add_argument("--gate-threshold", type=float, default=0.95)
    parser.add_argument(
        "--enable-homography-refinement",
        action="store_true",
        help="Apply planar court-geometry refinement to raw heatmap peaks before scoring.",
    )
    args = parser.parse_args(argv)

    rows = load_real_court_keypoint_labels(args.real_root)
    report = evaluate_checkpoint_against_real_labels(
        args.checkpoint,
        rows,
        device=args.device,
        use_homography_refinement=args.enable_homography_refinement,
        pck_threshold_px=args.pck_threshold_px,
    )
    report["gate_threshold"] = args.gate_threshold
    modes = ("raw_independent", "raw_all", "aggregated_independent", "aggregated_all")
    # Pooled: every keypoint across the selected frames counted together (informative, but not
    # the actual gate -- a viewpoint with terrible PCK@5 could be masked by good ones pooled in).
    report["gate_passed_pooled"] = {
        mode: bool(report[mode]["pck_at_5px"] is not None and report[mode]["pck_at_5px"] >= args.gate_threshold)
        for mode in modes
    }
    # Per-viewpoint: the actual CAL gate ("Held-out PCK@5px >= 0.95 per viewpoint") -- every
    # clip/viewpoint present must individually clear the threshold, not just the pooled average.
    report["gate_passed_per_viewpoint"] = {
        mode: bool(
            report[mode]["per_clip"]
            and all(
                clip_summary["pck_at_5px"] is not None and clip_summary["pck_at_5px"] >= args.gate_threshold
                for clip_summary in report[mode]["per_clip"].values()
            )
        )
        for mode in modes
    }
    # Kept for convenience/back-compat: alias to the per-viewpoint result, which is the real gate.
    report["gate_passed"] = report["gate_passed_per_viewpoint"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
