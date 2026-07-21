#!/usr/bin/env python3
"""Apply the frozen event-head A/B/C decision gate to synthetic or final eval JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


ARMS = ("A", "B", "C")
CLASSES = ("HIT", "BOUNCE")
EXPECTED_SEEDS = 3
EXPECTED_FINAL_STEPS = 1000
EXPECTED_SELECTION_ROWS = 41
EXPECTED_SELECTION_SCOPE = "owner_validation_41"
EPSILON = 1e-12


class DecisionGateInputError(ValueError):
    """Raised when an input JSON cannot support the preregistered gate."""


@dataclass(frozen=True)
class ArmEval:
    arm: str
    seed: int
    path: Path
    sha256: str
    macro_f1_at_2: float
    per_class_f1_at_2: Mapping[str, float]
    negative_false_positives: int
    negative_rows: int
    timing_error_p90_frames: float
    full_video_events_per_second: float
    completed_steps: int
    target_steps: int


def _finite_score(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionGateInputError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise DecisionGateInputError(f"{field} must be finite and in [0,1]")
    return parsed


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecisionGateInputError(f"{field} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionGateInputError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise DecisionGateInputError(f"{field} must be finite and nonnegative")
    return parsed


def _first_present(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value
    joined = " or ".join(".".join(path) for path in paths)
    raise DecisionGateInputError(f"missing required field: {joined}")


def _tolerance_two_metrics(payload: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    sweep = payload.get("tolerance_sweep")
    if not isinstance(sweep, list):
        raise DecisionGateInputError("tolerance_sweep must be an array")
    matches = [
        item for item in sweep
        if isinstance(item, Mapping) and item.get("tolerance_frames") == 2
    ]
    if len(matches) != 1:
        raise DecisionGateInputError(
            "tolerance_sweep must contain exactly one tolerance_frames=2 result"
        )
    per_class = matches[0].get("per_class")
    if not isinstance(per_class, Mapping):
        raise DecisionGateInputError("tolerance_frames=2 result lacks per_class")
    class_f1: dict[str, float] = {}
    for name in CLASSES:
        metrics = per_class.get(name)
        if not isinstance(metrics, Mapping):
            raise DecisionGateInputError(f"tolerance_frames=2 lacks {name} metrics")
        class_f1[name] = _finite_score(
            metrics.get("f1"), field=f"tolerance_frames=2.{name}.f1"
        )
    return sum(class_f1.values()) / len(class_f1), class_f1


def _parse_seed_path(value: str) -> tuple[int | None, Path]:
    prefix, separator, remainder = value.partition("=")
    if separator and prefix.isdecimal():
        return int(prefix), Path(remainder)
    return None, Path(value)


def _load_arm_eval(arm: str, value: str) -> ArmEval:
    seed_override, path = _parse_seed_path(value)
    if not path.is_file():
        raise DecisionGateInputError(f"{arm} eval JSON is absent: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionGateInputError(f"{arm} eval JSON is invalid: {path}") from exc
    if not isinstance(payload, Mapping):
        raise DecisionGateInputError(f"{arm} eval JSON must be an object: {path}")
    selection_scope = _first_present(
        payload, ("selection_scope",), ("evaluation", "selection_scope")
    )
    selection_rows = _integer(
        _first_present(payload, ("selection_rows",), ("evaluation", "selection_rows")),
        field=f"{path}.selection_rows",
    )
    if selection_scope != EXPECTED_SELECTION_SCOPE or selection_rows != EXPECTED_SELECTION_ROWS:
        raise DecisionGateInputError(
            f"{path} must contain only the frozen {EXPECTED_SELECTION_SCOPE} selection set "
            f"({EXPECTED_SELECTION_ROWS} rows)"
        )
    if payload.get("protected_50_touched") is not False:
        raise DecisionGateInputError(
            f"{path}.protected_50_touched must be false; protected-50 results are never gate inputs"
        )
    declared_arm = payload.get("arm")
    if declared_arm is not None and str(declared_arm).upper() != arm:
        raise DecisionGateInputError(
            f"{path} declares arm={declared_arm!r}, expected {arm}"
        )
    seed_value = seed_override if seed_override is not None else payload.get("seed")
    seed = _integer(seed_value, field=f"{path}.seed")
    if seed_override is not None and payload.get("seed") is not None:
        declared_seed = _integer(payload["seed"], field=f"{path}.seed")
        if declared_seed != seed_override:
            raise DecisionGateInputError(
                f"{path} declares seed={declared_seed}, CLI paired it with {seed_override}"
            )
    macro, per_class = _tolerance_two_metrics(payload)
    rate = _finite_nonnegative(
        _first_present(
            payload,
            ("full_video_events_per_second",),
            ("full_video_event_rate", "events_per_second"),
            ("full_video", "events_per_second"),
        ),
        field=f"{path}.full_video_events_per_second",
    )
    return ArmEval(
        arm=arm,
        seed=seed,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        macro_f1_at_2=macro,
        per_class_f1_at_2=per_class,
        negative_false_positives=_integer(
            _first_present(payload, ("negative_false_positives",)),
            field=f"{path}.negative_false_positives",
        ),
        negative_rows=_integer(
            _first_present(payload, ("negative_rows",)),
            field=f"{path}.negative_rows",
        ),
        timing_error_p90_frames=_finite_nonnegative(
            _first_present(
                payload,
                ("timing_error_p90_frames",),
                ("timing_p90_frames",),
                ("timing", "p90_frames"),
            ),
            field=f"{path}.timing_error_p90_frames",
        ),
        full_video_events_per_second=rate,
        completed_steps=_integer(
            _first_present(
                payload,
                ("completed_steps",),
                ("checkpoint_completed_steps",),
                ("training", "completed_steps"),
            ),
            field=f"{path}.completed_steps",
        ),
        target_steps=_integer(
            _first_present(payload, ("target_steps",), ("training", "target_steps")),
            field=f"{path}.target_steps",
        ),
    )


def _criterion(*, passed: bool, target: str, after: Any) -> dict[str, Any]:
    return {"pass": passed, "target": target, "after": after}


def _paired_bootstrap_lower_bound(deltas: Sequence[float]) -> tuple[float, int]:
    """Return the conservative empirical 2.5% bound over paired-seed medians.

    Three seeds have only 3**3=27 ordered bootstrap resamples, so enumerate the
    complete distribution rather than adding Monte Carlo noise to the gate.
    """

    if not deltas:
        raise DecisionGateInputError("paired bootstrap requires at least one paired delta")
    values = sorted(
        median([deltas[index] for index in indices])
        for indices in product(range(len(deltas)), repeat=len(deltas))
    )
    lower_index = max(0, math.ceil(0.025 * len(values)) - 1)
    return values[lower_index], len(values)


def evaluate_abc_gate(arms: Mapping[str, Sequence[ArmEval]]) -> dict[str, Any]:
    if set(arms) != set(ARMS):
        raise DecisionGateInputError("gate requires exactly arms A, B, and C")
    indexed: dict[str, dict[int, ArmEval]] = {}
    for arm in ARMS:
        values = list(arms[arm])
        if len(values) != EXPECTED_SEEDS:
            raise DecisionGateInputError(
                f"arm {arm} requires exactly {EXPECTED_SEEDS} eval JSONs"
            )
        if any(value.arm != arm for value in values):
            raise DecisionGateInputError(f"arm {arm} contains a mismatched result")
        by_seed = {value.seed: value for value in values}
        if len(by_seed) != EXPECTED_SEEDS:
            raise DecisionGateInputError(f"arm {arm} contains duplicate seeds")
        indexed[arm] = by_seed
    seeds = sorted(indexed["A"])
    if any(set(indexed[arm]) != set(seeds) for arm in ARMS[1:]):
        raise DecisionGateInputError("A/B/C seed sets must match exactly")

    b_minus_a = [
        indexed["B"][seed].macro_f1_at_2 - indexed["A"][seed].macro_f1_at_2
        for seed in seeds
    ]
    b_minus_c = [
        indexed["B"][seed].macro_f1_at_2 - indexed["C"][seed].macro_f1_at_2
        for seed in seeds
    ]
    bootstrap_lower_bound, bootstrap_resamples = _paired_bootstrap_lower_bound(b_minus_a)
    class_deltas = {
        str(seed): {
            name: (
                indexed["B"][seed].per_class_f1_at_2[name]
                - indexed["A"][seed].per_class_f1_at_2[name]
            )
            for name in CLASSES
        }
        for seed in seeds
    }
    step_counts = {
        arm: {str(seed): indexed[arm][seed].completed_steps for seed in seeds}
        for arm in ARMS
    }
    target_counts = {
        arm: {str(seed): indexed[arm][seed].target_steps for seed in seeds}
        for arm in ARMS
    }
    all_results = [indexed[arm][seed] for arm in ARMS for seed in seeds]
    equal_steps = (
        len({result.completed_steps for result in all_results}) == 1
        and len({result.target_steps for result in all_results}) == 1
        and all(
            result.completed_steps == result.target_steps == EXPECTED_FINAL_STEPS
            for result in all_results
        )
    )
    negative_fp = {
        str(seed): {
            arm: {
                "false_positives": indexed[arm][seed].negative_false_positives,
                "negative_rows": indexed[arm][seed].negative_rows,
            }
            for arm in ("A", "B")
        }
        for seed in seeds
    }
    timing_p90 = {
        str(seed): {
            arm: indexed[arm][seed].timing_error_p90_frames for arm in ("A", "B")
        }
        for seed in seeds
    }
    b_rates = {
        str(seed): indexed["B"][seed].full_video_events_per_second for seed in seeds
    }
    criteria = {
        "equal_step_parity": _criterion(
            passed=equal_steps,
            target=f"all 9 arms complete exactly {EXPECTED_FINAL_STEPS} target steps",
            after={"completed_steps": step_counts, "target_steps": target_counts},
        ),
        "median_b_minus_a_macro_f1_at_2": _criterion(
            passed=median(b_minus_a) + EPSILON >= 0.10,
            target=">= 0.10",
            after=median(b_minus_a),
        ),
        "all_seed_b_minus_a_nonnegative": _criterion(
            passed=all(delta + EPSILON >= 0.0 for delta in b_minus_a),
            target="every paired seed >= 0.0",
            after={str(seed): delta for seed, delta in zip(seeds, b_minus_a, strict=True)},
        ),
        "paired_bootstrap_95_lower_bound": _criterion(
            passed=bootstrap_lower_bound > 0.0,
            target="> 0.0",
            after={
                "lower_bound": bootstrap_lower_bound,
                "confidence": 0.95,
                "statistic": "median_paired_seed_B_minus_A_macro_f1_at_2",
                "method": "complete_ordered_resample_enumeration_conservative_empirical_quantile",
                "resamples": bootstrap_resamples,
            },
        ),
        "b_beats_c": _criterion(
            passed=all(delta > EPSILON for delta in b_minus_c),
            target="B > C for every paired seed",
            after={str(seed): delta for seed, delta in zip(seeds, b_minus_c, strict=True)},
        ),
        "per_class_regression": _criterion(
            passed=all(
                delta + EPSILON >= -0.03
                for seed_deltas in class_deltas.values()
                for delta in seed_deltas.values()
            ),
            target="every paired B-A HIT/BOUNCE F1 delta >= -0.03",
            after=class_deltas,
        ),
        "negative_false_positives": _criterion(
            passed=all(
                indexed["B"][seed].negative_rows == 22
                and indexed["B"][seed].negative_false_positives <= 2
                for seed in seeds
            ),
            target="B <= 2 false positives on exactly 22 negative rows for every seed",
            after=negative_fp,
        ),
        "negative_false_positives_vs_a": _criterion(
            passed=all(
                indexed["A"][seed].negative_rows == 22
                and indexed["B"][seed].negative_rows == 22
                and indexed["B"][seed].negative_false_positives
                <= indexed["A"][seed].negative_false_positives + 1
                for seed in seeds
            ),
            target="B negative false positives <= A + 1 on the same 22 rows for every seed",
            after=negative_fp,
        ),
        "timing_p90_non_worse": _criterion(
            passed=all(
                indexed["B"][seed].timing_error_p90_frames
                <= indexed["A"][seed].timing_error_p90_frames + EPSILON
                for seed in seeds
            ),
            target="B timing-error p90 <= A for every paired seed",
            after=timing_p90,
        ),
        "full_video_event_rate": _criterion(
            passed=all(
                0.3 - EPSILON
                <= indexed["B"][seed].full_video_events_per_second
                <= 1.0 + EPSILON
                for seed in seeds
            ),
            target="B in [0.3, 1.0] events/s for every seed",
            after=b_rates,
        ),
    }
    per_seed = {
        str(seed): {
            arm: {
                "macro_f1_at_2": indexed[arm][seed].macro_f1_at_2,
                "per_class_f1_at_2": dict(indexed[arm][seed].per_class_f1_at_2),
                "negative_false_positives": indexed[arm][seed].negative_false_positives,
                "negative_rows": indexed[arm][seed].negative_rows,
                "timing_error_p90_frames": indexed[arm][seed].timing_error_p90_frames,
                "full_video_events_per_second": (
                    indexed[arm][seed].full_video_events_per_second
                ),
                "completed_steps": indexed[arm][seed].completed_steps,
                "target_steps": indexed[arm][seed].target_steps,
            }
            for arm in ARMS
        }
        for seed in seeds
    }
    return {
        "schema_version": 2,
        "artifact_type": "event_head_abc_decision_gate",
        "verified": False,
        "selection_scope": EXPECTED_SELECTION_SCOPE,
        "selection_rows": EXPECTED_SELECTION_ROWS,
        "protected_50_touched": False,
        "verdict": "PASS" if all(item["pass"] for item in criteria.values()) else "FAIL",
        "seed_count": len(seeds),
        "seeds": seeds,
        "criteria": criteria,
        "per_seed": per_seed,
        "inputs": {
            arm: [
                {
                    "seed": indexed[arm][seed].seed,
                    "path": str(indexed[arm][seed].path),
                    "sha256": indexed[arm][seed].sha256,
                }
                for seed in seeds
            ]
            for arm in ARMS
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-a", nargs="+", required=True, metavar="[SEED=]EVAL_JSON",
        help="Exactly three A eval JSONs; prefix paths with SEED= if JSON omits seed",
    )
    parser.add_argument("--arm-b", nargs="+", required=True, metavar="[SEED=]EVAL_JSON")
    parser.add_argument("--arm-c", nargs="+", required=True, metavar="[SEED=]EVAL_JSON")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.out.suffix != ".json":
            raise DecisionGateInputError("--out must be a .json verdict artifact")
        arms = {
            "A": [_load_arm_eval("A", value) for value in args.arm_a],
            "B": [_load_arm_eval("B", value) for value in args.arm_b],
            "C": [_load_arm_eval("C", value) for value in args.arm_c],
        }
        verdict = evaluate_abc_gate(arms)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    except (DecisionGateInputError, OSError) as exc:
        parser.exit(2, f"ABC decision gate input rejected: {exc}\n")
    print(json.dumps({"out": str(args.out), "verdict": verdict["verdict"]}, sort_keys=True))
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
