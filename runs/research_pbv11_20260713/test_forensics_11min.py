from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_vs_pbvision as forensics


LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[2]


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cv_export=ROOT / "data/pbvision_11min_20260713/cv_export.json",
        insights=ROOT / "data/pbvision_11min_20260713/insights.json",
        stats=ROOT / "data/pbvision_11min_20260713/stats.json",
        ball_recovery_report=ROOT / "runs/lanes/ball_recovery_20260712/report.json",
        tt3d_report=ROOT / "runs/lanes/tt3d_integrate_20260712/report.json",
        source_video=ROOT / "data/pbvision_11min_20260713/source_video.mp4",
        video_provenance=ROOT / "data/pbvision_11min_20260713/video_provenance.json",
        image_size=None,
        output=output,
    )


def test_saved_metrics_match_fresh_deterministic_build(tmp_path: Path) -> None:
    actual = forensics.build(_args(tmp_path / "unused.json"))
    rendered = json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert rendered == (LANE / "forensics_metrics.json").read_text()


def test_scale_invariants_and_policy() -> None:
    payload = json.loads((LANE / "forensics_metrics.json").read_text())
    assert payload["policy"] == {
        "competitor_reference_only": True,
        "deterministic": True,
        "pbvision_is_not_ground_truth": True,
        "promotion_evidence": False,
        "training_data": False,
        "verified": False,
    }
    structure = payload["structure"]
    assert structure["cv_rally_count"] == 42
    assert structure["insights_rally_count"] == 41
    assert structure["matched_by_start_ms_count"] == 41
    assert structure["unmatched_cv_rally_indices"] == [35]
    assert structure["rally_frame_count"] == 10322
    assert structure["selected_3d_count"] == 7808
    assert payload["inputs"]["image_size_assumption"] == [1280, 720]
    assert payload["inputs"]["source_video"]["sha256"] == "272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383"


def test_hypothesis_ledger_keeps_export_flags_separate() -> None:
    payload = json.loads((LANE / "forensics_metrics.json").read_text())
    bounce = payload["hypotheses"]["bounce_radius"]
    assert bounce["ledger"]["literal_all_bounces_at_radius"] == "REFUTED"
    assert bounce["ledger"]["all_in_sequence_bounces_at_radius"] == "CONFIRMED"
    assert bounce["off_radius_count"] == 9
    assert all(row["out_of_sequence"] for row in bounce["off_radius_rows"])
    assert payload["best_stack_delta"] == "none"
    assert payload["verified"] == 0
