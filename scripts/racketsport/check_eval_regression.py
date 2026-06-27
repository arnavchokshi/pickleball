#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_DROP_PERCENT = 2.0


@dataclass(frozen=True)
class MetricRegression:
    path: str
    baseline: float | int
    current: float | int
    drop_percent: float


@dataclass(frozen=True)
class RegressionResult:
    status: str
    max_drop_percent: float
    checked_metrics: int
    failures: list[MetricRegression]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "max_drop_percent": self.max_drop_percent,
            "checked_metrics": self.checked_metrics,
            "failures": [asdict(failure) for failure in self.failures],
        }


def compare_phase_metrics(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    max_drop_percent: float = DEFAULT_MAX_DROP_PERCENT,
) -> RegressionResult:
    if max_drop_percent < 0:
        raise ValueError("max_drop_percent must be non-negative")

    current_metrics = _measured_numeric_metrics(current)
    baseline_metrics = _measured_numeric_metrics(baseline)
    failures: list[MetricRegression] = []

    for path, baseline_value in sorted(baseline_metrics.items()):
        if path not in current_metrics:
            continue
        current_value = current_metrics[path]
        if current_value >= baseline_value:
            continue

        drop_percent = _drop_percent(baseline_value=baseline_value, current_value=current_value)
        if drop_percent > max_drop_percent:
            failures.append(
                MetricRegression(
                    path=path,
                    baseline=baseline_value,
                    current=current_value,
                    drop_percent=drop_percent,
                )
            )

    return RegressionResult(
        status="fail" if failures else "pass",
        max_drop_percent=max_drop_percent,
        checked_metrics=len(set(current_metrics).intersection(baseline_metrics)),
        failures=failures,
    )


def _measured_numeric_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    for name, metric in _as_mapping(payload.get("metrics")).items():
        value = _numeric_measured_value(metric)
        if value is not None:
            metrics[f"metrics.{name}"] = value

    for index, clip in enumerate(_as_list(payload.get("clips"))):
        if not isinstance(clip, dict):
            continue
        clip_name = str(clip.get("clip") or index)
        for name, metric in _as_mapping(clip.get("metrics")).items():
            value = _numeric_measured_value(metric)
            if value is not None:
                metrics[f"clips[{clip_name}].metrics.{name}"] = value

    return metrics


def _numeric_measured_value(metric: Any) -> float | int | None:
    if not isinstance(metric, dict):
        return None
    if metric.get("status") != "measured":
        return None

    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _drop_percent(*, baseline_value: float | int, current_value: float | int) -> float:
    if baseline_value == 0:
        return 100.0 if current_value < 0 else 0.0
    return ((baseline_value - current_value) / abs(baseline_value)) * 100.0


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current and baseline phase metrics JSON for regressions.")
    parser.add_argument("--current", type=Path, required=True, help="Current phase metrics JSON path.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline phase metrics JSON path.")
    parser.add_argument(
        "--max-drop-percent",
        type=float,
        default=DEFAULT_MAX_DROP_PERCENT,
        help=f"Maximum allowed measured numeric metric drop percentage. Defaults to {DEFAULT_MAX_DROP_PERCENT}.",
    )
    args = parser.parse_args()

    result = compare_phase_metrics(
        current=_load_json(args.current),
        baseline=_load_json(args.baseline),
        max_drop_percent=args.max_drop_percent,
    )
    print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
