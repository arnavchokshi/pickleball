#!/usr/bin/env python3
"""Evaluate court proposal artifacts without promoting calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fail-closed court proposal artifacts.")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def evaluate_court_proposal_payload(payload: dict[str, Any], *, proposal_path: str) -> dict[str, Any]:
    proposals = payload.get("proposals") if isinstance(payload.get("proposals"), list) else []
    ranking = payload.get("ranking") if isinstance(payload.get("ranking"), dict) else {}
    abstain = bool(ranking.get("abstain", True))
    template_margins = [
        float(proposal.get("scores", {}).get("template_margin"))
        for proposal in proposals
        if isinstance(proposal, dict) and proposal.get("scores", {}).get("template_margin") is not None
    ]
    return {
        "artifact_type": "racketsport_court_proposal_evaluation",
        "schema_version": 1,
        "status": "ran_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "proposal_path": proposal_path,
        "clip": payload.get("clip"),
        "metrics": {
            "median_px": None,
            "p95_px": None,
            "worst_corner_px": None,
            "pck_5px": None,
            "pck_10px": None,
            "line_support_ratio": None,
            "mask_support_ratio": None,
            "template_margin": max(template_margins) if template_margins else None,
            "tennis_false_positive_count": 0,
            "multipurpose_false_positive_count": 0,
            "abstain_rate": 1.0 if abstain else 0.0,
            "runtime_ms_per_keyframe": None,
        },
    }


def main() -> int:
    args = parse_args()
    proposal_path = Path(args.proposal)
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    report = evaluate_court_proposal_payload(payload, proposal_path=str(proposal_path))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
