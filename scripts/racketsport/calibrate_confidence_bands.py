#!/usr/bin/env python3
"""Calibrate confidence-band curves from internal-val PHYS artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.confidence_gate import horizon_bucket  # noqa: E402


HORIZON_BUCKETS = ("1-3", "4-8", "9-15", "16+")
BUCKET_REPRESENTATIVE_HORIZONS = (2, 6, 12, 16)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Chain-style run directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output calibration_curves.json path.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: run dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    virtual_world = _load_optional_json(run_dir / "virtual_world.json") or {}
    ball_filled = _load_optional_json(run_dir / "ball_track_physics_filled.json") or {}

    curves = {
        "schema_version": 1,
        "artifact_type": "racketsport_confidence_calibration_curves",
        "generated_at": _utc_now(),
        "run_dir": str(run_dir),
        "policy": {
            "internal_val_only": True,
            "protected_eval_labels_used": False,
            "outdoor_indoor_labels_read": False,
            "high_confidence_frames_only": True,
        },
        "ball": _calibrate_ball(ball_filled),
        "player_joints": _calibrate_player_joints(virtual_world, threshold=args.confidence_threshold),
        "paddle": {
            "status": "no_prediction_possible",
            "predictor": "PaddleNullPredictor",
            "horizon_buckets": {},
            "reason": "Current chain has no estimable paddle frames; 0/63 contacts estimable on Wolverine.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "entities": ["ball", "player_joints", "paddle"]}, indent=2, sort_keys=True))
    return 0


def _calibrate_ball(ball_filled: Mapping[str, Any]) -> dict[str, Any]:
    loo = _find_ball_leave_one_out(ball_filled)
    if not loo:
        return {
            "status": "insufficient_ball_loo",
            "predictor": "BallBallisticAdapter",
            "horizon_buckets": _default_buckets(p50=0.1649, p90=0.4893, p95=0.5621, source="known_wolverine_loo_fallback"),
            "known_loo_comparison": {"median_m": 0.1649, "p95_m": 0.5621, "source": "manager_prompt_known_input"},
        }
    median = float(loo.get("median", loo.get("p50", 0.1649)))
    p90 = float(loo.get("p90", median))
    p95 = float(loo.get("p95", p90))
    count = int(loo.get("count", 0) or 0)
    return {
        "status": "calibrated_from_existing_ballfill_loo",
        "predictor": "BallBallisticAdapter",
        "horizon_buckets": _default_buckets(p50=median, p90=p90, p95=p95, source="phys_ballfill_leave_one_out"),
        "known_loo_comparison": {
            "count": count,
            "median_m": median,
            "p90_m": p90,
            "p95_m": p95,
            "manager_expected_median_m": 0.1649,
            "manager_expected_p95_m": 0.5621,
        },
        "notes": [
            "The upstream PHYS-BALLFILL report exposes aggregate LOO, not per-horizon LOO.",
            "Buckets are anchored to the measured aggregate LOO and inflated monotonically with horizon.",
        ],
    }


def _calibrate_player_joints(virtual_world: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    fps = float(virtual_world.get("fps") or 30.0)
    players = virtual_world.get("players", [])
    if not isinstance(players, list):
        players = []
    errors_by_bucket: dict[str, list[float]] = {bucket: [] for bucket in HORIZON_BUCKETS}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = [frame for frame in player.get("frames", []) or [] if isinstance(frame, Mapping)]
        indexed = {_frame_index(frame, fps): frame for frame in frames}
        for horizon in BUCKET_REPRESENTATIVE_HORIZONS:
            bucket = horizon_bucket(horizon)
            for target_index, target in indexed.items():
                prev = indexed.get(target_index - horizon)
                prev2 = indexed.get(target_index - 2 * horizon)
                if not prev or not prev2:
                    continue
                if not (_high_joint_conf(target, threshold) and _high_joint_conf(prev, threshold) and _high_joint_conf(prev2, threshold)):
                    continue
                predicted = _predict_joint_frame(prev2.get("joints_world"), prev.get("joints_world"))
                target_joints = target.get("joints_world")
                if predicted is None or not isinstance(target_joints, list):
                    continue
                error = _mean_joint_error(predicted, target_joints)
                if error is not None:
                    errors_by_bucket[bucket].append(error)

    buckets: dict[str, dict[str, Any]] = {}
    for bucket, errors in errors_by_bucket.items():
        if errors:
            buckets[bucket] = {
                "sample_count": len(errors),
                "p50_m": _quantile(errors, 0.50),
                "p90_m": _quantile(errors, 0.90),
            }
        else:
            buckets[bucket] = {"sample_count": 0, "p50_m": 0.05, "p90_m": 0.15, "status": "fallback_no_samples"}
    return {
        "status": "calibrated_from_virtual_world_high_confidence_joints" if any(v["sample_count"] for v in buckets.values()) else "fallback_no_joint_samples",
        "predictor": "JointKinematicPredictor",
        "horizon_buckets": buckets,
        "notes": ["Kinematic LOO uses high-confidence existing joint frames only; it is internal-val calibration, not BODY verification."],
    }


def _find_ball_leave_one_out(ball_filled: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Any] = []
    physics_fill = ball_filled.get("physics_fill")
    if isinstance(physics_fill, Mapping):
        candidates.append(physics_fill.get("validation"))
    candidates.append(ball_filled.get("validation"))
    for validation in candidates:
        if not isinstance(validation, Mapping):
            continue
        loo = validation.get("leave_one_out")
        if not isinstance(loo, Mapping):
            continue
        errors = loo.get("error_3d_m")
        if isinstance(errors, Mapping):
            return errors
    return None


def _default_buckets(*, p50: float, p90: float, p95: float, source: str) -> dict[str, dict[str, Any]]:
    factors = {"1-3": 1.0, "4-8": 1.25, "9-15": 1.5, "16+": 2.0}
    return {
        bucket: {
            "sample_count": None,
            "p50_m": p50 * factor,
            "p90_m": p90 * factor,
            "p95_m": p95 * factor,
            "source": source,
        }
        for bucket, factor in factors.items()
    }


def _predict_joint_frame(prev2: Any, prev: Any) -> list[list[float]] | None:
    if not isinstance(prev2, list) or not isinstance(prev, list) or len(prev2) != len(prev):
        return None
    predicted: list[list[float]] = []
    for older_joint, prev_joint in zip(prev2, prev):
        if not isinstance(older_joint, Sequence) or not isinstance(prev_joint, Sequence):
            return None
        if len(older_joint) != 3 or len(prev_joint) != 3:
            return None
        predicted.append([float(prev_joint[axis]) + (float(prev_joint[axis]) - float(older_joint[axis])) for axis in range(3)])
    return predicted


def _mean_joint_error(predicted: Sequence[Sequence[float]], target: Sequence[Sequence[float]]) -> float | None:
    if len(predicted) != len(target) or not predicted:
        return None
    errors = []
    for pred_joint, target_joint in zip(predicted, target):
        if not isinstance(target_joint, Sequence) or len(target_joint) != 3:
            return None
        errors.append(sum((float(pred_joint[axis]) - float(target_joint[axis])) ** 2 for axis in range(3)) ** 0.5)
    return sum(errors) / len(errors)


def _high_joint_conf(frame: Mapping[str, Any], threshold: float) -> bool:
    values = frame.get("joint_conf")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return False
    numeric = [float(value) for value in values if not isinstance(value, bool)]
    return bool(numeric) and sum(numeric) / len(numeric) >= threshold


def _frame_index(frame: Mapping[str, Any], fps: float) -> int:
    if "frame_index" in frame:
        return int(frame["frame_index"])
    return int(round(float(frame.get("t", 0.0)) * fps))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
