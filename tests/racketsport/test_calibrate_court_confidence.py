from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.racketsport.calibrate_court_confidence import fit_calibration_artifact


def _report(path: Path, *, fold_index: int, confidence: float, correct: bool) -> Path:
    error = 2.0 if correct else 9.0
    path.write_text(
        json.dumps(
            {
                "evaluation_protocol": {
                    "fold_index": fold_index,
                    "partition": "validation",
                },
                "raw_vs_structured": {
                    "structured": {
                        "samples": [
                            {
                                "point_errors_px": {"near_left_corner": error},
                                "point_confidence": {"near_left_corner": confidence},
                                "whole_court_confidence": confidence,
                                "whole_court_within_5px_and_topology_valid": correct,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_out_of_fold_calibration_is_serializable_and_never_promotes(tmp_path: Path) -> None:
    reports = [
        _report(tmp_path / "fold0.json", fold_index=0, confidence=0.9, correct=True),
        _report(tmp_path / "fold1.json", fold_index=1, confidence=0.2, correct=False),
    ]
    artifact = fit_calibration_artifact(reports, unsupported_probabilities=[0.8])
    assert artifact["fold_indexes"] == [0, 1]
    assert artifact["point_reliability"]["sample_count"] == 2
    assert artifact["court_reliability"]["sample_count"] == 2
    assert artifact["court_confidence_calibration"]["promotion_allowed"] is False
    assert artifact["measurement_valid"] is False


def test_cli_help_is_directly_invocable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/calibrate_court_confidence.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--evaluation-report" in result.stdout
