#!/usr/bin/env python3
"""Fit point and whole-court confidence from out-of-fold structured evaluations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_confidence_calibration import (  # noqa: E402
    confidence_calibration_report,
    fit_isotonic_confidence,
    fit_temperature_confidence,
    select_zero_false_accept_threshold,
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structured_section(payload: dict[str, Any]) -> dict[str, Any]:
    raw_vs = payload.get("raw_vs_structured")
    if isinstance(raw_vs, dict) and isinstance(raw_vs.get("structured"), dict):
        return raw_vs["structured"]
    structured = payload.get("structured_best_court")
    if isinstance(structured, dict):
        return structured
    if isinstance(payload.get("samples"), list):
        return payload
    raise ValueError("evaluation report has no structured sample section")


def fit_calibration_artifact(
    report_paths: Sequence[Path],
    *,
    unsupported_probabilities: Sequence[float] = (),
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one out-of-fold evaluation report is required")
    point_scores: list[float] = []
    point_outcomes: list[bool] = []
    court_probabilities: list[float] = []
    court_outcomes: list[bool] = []
    fold_indexes: set[int] = set()
    inputs: list[dict[str, Any]] = []
    for path in report_paths:
        payload = _read_object(path)
        protocol = payload.get("evaluation_protocol")
        if not isinstance(protocol, dict) or protocol.get("partition") not in {
            "validation",
            "test",
        }:
            raise ValueError(f"calibration input is not a held-out protocol partition: {path}")
        fold_index = protocol.get("fold_index")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int):
            raise ValueError(f"calibration input lacks an integer fold index: {path}")
        if fold_index in fold_indexes:
            raise ValueError(f"duplicate calibration fold index: {fold_index}")
        fold_indexes.add(fold_index)
        section = _structured_section(payload)
        samples = section.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"structured evaluation must retain per-sample outputs: {path}")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"invalid sample row in {path}")
            errors = sample.get("point_errors_px")
            confidences = sample.get("point_confidence")
            if not isinstance(errors, dict) or not isinstance(confidences, dict):
                raise ValueError(f"sample calibration pairs missing in {path}")
            for name, confidence in confidences.items():
                error = errors.get(name)
                if error is None:
                    continue
                point_scores.append(float(confidence))
                point_outcomes.append(float(error) <= 5.0)
            court_probability = sample.get("whole_court_confidence")
            if court_probability is not None:
                court_probabilities.append(float(court_probability))
                court_outcomes.append(
                    bool(sample.get("whole_court_within_5px_and_topology_valid"))
                )
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "fold_index": fold_index,
                "partition": protocol["partition"],
                "sample_count": len(samples),
            }
        )
    if not point_scores or not court_probabilities:
        raise ValueError("out-of-fold reports lack point or court confidence pairs")
    point_calibrator = fit_isotonic_confidence(point_scores, point_outcomes, threshold_px=5.0)
    court_logits = [
        math.log(min(max(value, 1.0e-8), 1.0 - 1.0e-8) / (1.0 - min(max(value, 1.0e-8), 1.0 - 1.0e-8)))
        for value in court_probabilities
    ]
    court_calibrator = fit_temperature_confidence(
        court_logits,
        court_outcomes,
        quality_threshold_px=5.0,
    )
    calibrated_court = [court_calibrator.predict_probability(value) for value in court_probabilities]
    unsupported = [float(value) for value in unsupported_probabilities]
    threshold = None
    zero_false_accepts = False
    if unsupported:
        combined_probabilities = calibrated_court + unsupported
        combined_quality = court_outcomes + [False] * len(unsupported)
        combined_unsupported = [False] * len(calibrated_court) + [True] * len(unsupported)
        threshold = select_zero_false_accept_threshold(
            combined_probabilities,
            combined_quality,
            combined_unsupported,
        )
        zero_false_accepts = threshold is not None
    court_calibrator = replace(
        court_calibrator,
        measurement_threshold=threshold,
        zero_unsupported_false_accepts=zero_false_accepts,
        promotion_allowed=False,
    )
    calibrated_points = [point_calibrator.predict(value) for value in point_scores]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_structured_confidence_calibration",
        "status": "out_of_fold_calibrated_not_authoritative",
        "verified": False,
        "measurement_valid": False,
        "authority_state": "review_only",
        "inputs": inputs,
        "fold_indexes": sorted(fold_indexes),
        "point_confidence_calibration": point_calibrator.to_dict(),
        "court_confidence_calibration": court_calibrator.to_dict(),
        "point_reliability": confidence_calibration_report(
            calibrated_points, point_outcomes
        ),
        "court_reliability": confidence_calibration_report(
            calibrated_court, court_outcomes
        ),
        "unsupported_view_score_count": len(unsupported),
        "measurement_threshold_policy": (
            "lowest_calibrated_threshold_with_zero_bad_or_unsupported_accepts"
        ),
    }


def _unsupported_scores(path: Path | None) -> list[float]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("unsupported scores must be a JSON list")
    return [float(value) for value in payload]


def _attach_to_checkpoint(
    checkpoint_in: Path,
    checkpoint_out: Path,
    artifact: dict[str, Any],
) -> None:
    import torch

    payload = torch.load(str(checkpoint_in), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint input must contain a model state dict")
    payload = dict(payload)
    payload["point_confidence_calibration"] = artifact["point_confidence_calibration"]
    payload["court_confidence_calibration"] = artifact["court_confidence_calibration"]
    payload["confidence_calibration_provenance"] = {
        "artifact_sha256": hashlib.sha256(
            (json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "status": artifact["status"],
    }
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, action="append", required=True)
    parser.add_argument("--unsupported-scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args(argv)
    if (args.checkpoint_in is None) != (args.checkpoint_out is None):
        parser.error("--checkpoint-in and --checkpoint-out must be supplied together")
    artifact = fit_calibration_artifact(
        args.evaluation_report,
        unsupported_probabilities=_unsupported_scores(args.unsupported_scores),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.checkpoint_in is not None:
        _attach_to_checkpoint(args.checkpoint_in, args.checkpoint_out, artifact)
    print(json.dumps({"status": artifact["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
