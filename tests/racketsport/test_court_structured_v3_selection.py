from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.best_stack import load_best_stack_manifest


ROOT = Path(__file__).resolve().parents[2]
WINNER_SHA256 = "bba24725dc4c184f7c2ad42c2223391d4b76829716285030e913561da2a32aae"


def test_best_stack_records_structured_v3_review_only_winner() -> None:
    entry = load_best_stack_manifest().entry("court.court_unet_v2")

    assert entry.status == "PENDING"
    assert entry.value["sha256"] == WINNER_SHA256
    assert entry.proven_against["structured_pck_at_5px"] == 0.6901408450704225
    assert entry.proven_against["baseline_structured_pck_at_5px"] == 0.647887323943662
    assert entry.proven_against["topology_valid_samples"] == 6
    assert entry.proven_against["duplicate_or_collapsed_outputs"] == 0
    assert "measurement_valid=false" in entry.notes
    assert "VERIFIED=0" in entry.notes


def test_selection_summary_matches_winning_raw_evidence() -> None:
    run_root = ROOT / "runs" / "court_structured_v3_20260723"
    summary = json.loads((run_root / "selection_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (
            run_root
            / "diagnostic_eval"
            / "curated264_task88"
            / "epoch_0006-court-keypoint-metrics.json"
        ).read_text(encoding="utf-8")
    )
    structured = metrics["structured_best_court"]

    assert summary["winner"]["checkpoint_sha256"] == WINNER_SHA256
    assert summary["winner"]["authority_state"] == "review_only"
    assert summary["winner"]["measurement_valid"] is False
    assert summary["winner"]["promotion_allowed"] is False
    assert summary["winner"]["duplicate_or_collapse_pair_count"] == 0
    assert structured["point_metrics"]["pck_at_5px"] == summary["results"][-1][
        "structured_pck_at_5px"
    ]
    assert structured["point_metrics"]["median_error_px"] == summary["results"][-1][
        "structured_median_px"
    ]
    assert structured["structure"]["topology_valid_count"] == 6
    assert structured["structure"]["duplicate_or_collapse_pair_count"] == 0
