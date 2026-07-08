#!/usr/bin/env python3
"""Wave-6 BVP span verifier using the manager-ruled v2 whole-span policy.

This does not edit or supersede the wave-4 verifier. The wave-4 verifier is a
historical instrument for the rejected split-junction policy; this harness reads
the same independent wave-5 verifier artifacts and applies the accepted v2 rule:
frozen-baseline protected spans must remain whole with no fit/RMSE/confidence
regression, and an unverified contact prior must not force a split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[2]
W5_VERIFY = ROOT / "runs/lanes/w5_bvpspan_verify_20260707"
W4_HISTORICAL = ROOT / "runs/lanes/w4_bvp_verify_20260707/harness/bvp_verify_harness.py"
TARGET_FLOORS = {
    "burlington_gold_0300_low_steep_corner": 0.7727272727,
    "wolverine_mixed_0200_mid_steep_corner": 0.8750,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zero_delta(row: Mapping[str, Any], key: str) -> bool:
    return abs(float((row.get("delta") or {}).get(key, 0.0))) <= 1e-12


def verify(metrics_path: Path, d3e_path: Path, axis4_path: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    d3e = read_json(d3e_path)
    axis4 = read_json(axis4_path)
    protected_rows = []
    protected_pass = True
    for row in metrics.get("rows", []):
        row_pass = (
            not row.get("material_losses")
            and _zero_delta(row, "fit_coverage_fraction")
            and _zero_delta(row, "fallback_coverage_fraction")
            and _zero_delta(row, "residual_rmse_px")
            and _zero_delta(row, "viewer_confidence_weighted")
        )
        protected_pass = protected_pass and row_pass
        protected_rows.append(
            {
                "clip": row.get("clip"),
                "interval": row.get("interval"),
                "name": row.get("name"),
                "fit_coverage_delta": (row.get("delta") or {}).get("fit_coverage_fraction"),
                "fallback_coverage_delta": (row.get("delta") or {}).get("fallback_coverage_fraction"),
                "residual_rmse_px_delta": (row.get("delta") or {}).get("residual_rmse_px"),
                "viewer_confidence_weighted_delta": (row.get("delta") or {}).get("viewer_confidence_weighted"),
                "endpoint_error_max_m_delta": (row.get("delta") or {}).get("endpoint_error_max_m"),
                "material_losses": list(row.get("material_losses") or []),
                "verdict": "PASS" if row_pass else "FAIL",
            }
        )
    after_rows = [row for row in d3e.get("rows", []) if row.get("candidate") == "after_product_view"]
    d3e_rows = []
    d3e_pass = True
    for row in after_rows:
        clip = str(row.get("clip"))
        floor = TARGET_FLOORS[clip]
        f1 = float(row.get("label_f1_at_20px"))
        row_pass = f1 + 1e-12 >= floor
        d3e_pass = d3e_pass and row_pass
        d3e_rows.append(
            {
                "clip": clip,
                "label_f1_at_20px": f1,
                "floor_label_f1_at_20px": floor,
                "visible_recall_at_20px": row.get("visible_recall_at_20px"),
                "hidden_false_positive_count": row.get("hidden_false_positive_count"),
                "verdict": "PASS" if row_pass else "FAIL",
            }
        )
    axis4_payload = axis4.get("adjudication", {})
    whole_span_pass = (
        axis4_payload.get("conclusion") == "stale_harness_expectation_for_banked_v2"
        and axis4_payload.get("stale_if_policy_is_protect_span_and_ignore_unverified_contact_prior") is True
    )
    objective = (
        "PASS"
        if protected_pass and len(protected_rows) == 5 and d3e_pass and len(d3e_rows) == 2 and whole_span_pass
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "artifact_type": "w6_bvp_span_verify_harness_v2_result",
        "objective_result": objective,
        "policy": {
            "name": "bvp_span_protection_v2_whole_span",
            "rule": (
                "Frozen-baseline protected spans pass when fit/fallback/RMSE/confidence deltas are zero and no material "
                "losses occur; the historical contact-519 split-junction requirement is not part of v2."
            ),
            "historical_w4_harness": str(W4_HISTORICAL.relative_to(ROOT)),
            "historical_w4_status": "preserved_read_only_historical_instrument",
        },
        "inputs": {
            "span_metrics": str(metrics_path.relative_to(ROOT)),
            "d3e_product_eval": str(d3e_path.relative_to(ROOT)),
            "axis4_adjudication": str(axis4_path.relative_to(ROOT)),
        },
        "acceptance": {
            "protected_span_count": len(protected_rows),
            "protected_span_delta_0_0_count": sum(1 for row in protected_rows if row["verdict"] == "PASS"),
            "protected_spans": protected_rows,
            "d3e_after_rows": d3e_rows,
            "whole_span_axis4": {
                "historical_split_verdict": (metrics.get("split_497_543") or {}).get("verdict"),
                "v2_verdict": "PASS" if whole_span_pass else "FAIL",
                "manager_ruled_conclusion": axis4_payload.get("conclusion"),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify BVP protected-span v2 whole-span acceptance artifacts.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=W5_VERIFY / "w4_harness/span_equivalence_metrics.json",
    )
    parser.add_argument(
        "--d3e",
        type=Path,
        default=W5_VERIFY / "d3e_product_eval/fresh_d3e_product_eval.json",
    )
    parser.add_argument(
        "--axis4",
        type=Path,
        default=W5_VERIFY / "axis4_adjudication.json",
    )
    parser.add_argument("--out", type=Path, default=LANE / "bvp_span_v2_harness_result.json")
    args = parser.parse_args(argv)
    result = verify(args.metrics, args.d3e, args.axis4)
    write_json(args.out, result)
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))
    return 0 if result["objective_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
