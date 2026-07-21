from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/racketsport/abc_decision_gate.py"
SEEDS = (20260720, 20260721, 20260722)


def _eval_payload(
    *,
    arm: str,
    seed: int,
    hit_f1: float,
    bounce_f1: float,
    negative_false_positives: int = 1,
    timing_error_p90_frames: float = 0.9,
    rate: float = 0.6,
    completed_steps: int = 1000,
    target_steps: int = 1000,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "event_head_abc_arm_eval",
        "verified": False,
        "arm": arm,
        "seed": seed,
        "selection_scope": "owner_validation_41",
        "selection_rows": 41,
        "protected_50_touched": False,
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "negative_rows": 22,
        "negative_false_positives": negative_false_positives,
        "timing_error_p90_frames": timing_error_p90_frames,
        "full_video_events_per_second": rate,
        "tolerance_sweep": [{
            "tolerance_frames": 2,
            "per_class": {
                "HIT": {"f1": hit_f1},
                "BOUNCE": {"f1": bounce_f1},
            },
        }],
    }


def _write_matrix(tmp_path: Path, *, failing: bool) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {"A": [], "B": [], "C": []}
    for index, seed in enumerate(SEEDS):
        values = {
            "A": (0.50, 0.50),
            "B": ((0.49, 0.49) if failing and index == 0 else (0.62, 0.60)),
            "C": (0.54, 0.54),
        }
        for arm, (hit_f1, bounce_f1) in values.items():
            payload = _eval_payload(
                arm=arm,
                seed=seed,
                hit_f1=hit_f1,
                bounce_f1=bounce_f1,
                negative_false_positives=(3 if failing and arm == "B" and index == 1 else 1),
                timing_error_p90_frames=(
                    1.1 if failing and arm == "B" and index == 2
                    else 0.9 if arm == "B"
                    else 1.0
                ),
                rate=(1.1 if failing and arm == "B" and index == 2 else 0.6),
                completed_steps=(999 if failing and arm == "C" and index == 2 else 1000),
            )
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            paths[arm].append(path)
    return paths


def _run_gate(tmp_path: Path, paths: dict[str, list[Path]]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(CLI),
            "--arm-a", *(str(path) for path in paths["A"]),
            "--arm-b", *(str(path) for path in paths["B"]),
            "--arm-c", *(str(path) for path in paths["C"]),
            "--out", str(tmp_path / "verdict.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_synthetic_three_seed_matrix_passes_every_gate(tmp_path: Path) -> None:
    completed = _run_gate(tmp_path, _write_matrix(tmp_path, failing=False))

    assert completed.returncode == 0, completed.stderr
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "PASS"
    assert verdict["verified"] is False
    assert verdict["criteria"]["median_b_minus_a_macro_f1_at_2"]["after"] == pytest.approx(0.11)
    assert all(item["pass"] for item in verdict["criteria"].values())


def test_synthetic_matrix_fails_step_seed_negfp_and_rate_guards(tmp_path: Path) -> None:
    completed = _run_gate(tmp_path, _write_matrix(tmp_path, failing=True))

    assert completed.returncode == 1, completed.stderr
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "FAIL"
    assert verdict["criteria"]["equal_step_parity"]["pass"] is False
    assert verdict["criteria"]["all_seed_b_minus_a_nonnegative"]["pass"] is False
    assert verdict["criteria"]["paired_bootstrap_95_lower_bound"]["pass"] is False
    assert verdict["criteria"]["negative_false_positives"]["pass"] is False
    assert verdict["criteria"]["timing_p90_non_worse"]["pass"] is False
    assert verdict["criteria"]["full_video_event_rate"]["pass"] is False


def test_reviewer_counterexample_plus_two_negfp_and_one_step_fails(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, failing=False)
    for arm_paths in paths.values():
        for path in arm_paths:
            payload = json.loads(path.read_text())
            payload["completed_steps"] = payload["target_steps"] = 1
            if payload["arm"] == "A":
                payload["negative_false_positives"] = 0
            elif payload["arm"] == "B":
                payload["negative_false_positives"] = 2
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    completed = _run_gate(tmp_path, paths)

    assert completed.returncode == 1, completed.stderr
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["criteria"]["negative_false_positives"]["pass"] is True
    assert verdict["criteria"]["negative_false_positives_vs_a"]["pass"] is False
    assert verdict["criteria"]["equal_step_parity"]["pass"] is False


def test_gate_refuses_protected_or_non_owner_selection_inputs(tmp_path: Path) -> None:
    paths = _write_matrix(tmp_path, failing=False)
    payload = json.loads(paths["A"][0].read_text())
    payload["protected_50_touched"] = True
    paths["A"][0].write_text(json.dumps(payload, sort_keys=True) + "\n")

    completed = _run_gate(tmp_path, paths)

    assert completed.returncode == 2
    assert "protected-50 results are never gate inputs" in completed.stderr
