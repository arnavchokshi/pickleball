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


@dataclass(frozen=True)
class MetricArtifactPair:
    relative_path: str
    baseline: Path
    current: Path


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


def _discover_metric_artifacts(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"{root} must be a directory")

    artifacts: dict[str, Path] = {}
    for path in sorted(root.rglob("metrics.json")):
        if not path.is_file():
            continue
        artifacts[path.relative_to(root).as_posix()] = path
    return artifacts


def _discover_paired_metric_artifacts(*, current_root: Path, baseline_root: Path) -> list[MetricArtifactPair]:
    current_artifacts = _discover_metric_artifacts(current_root)
    baseline_artifacts = _discover_metric_artifacts(baseline_root)
    if not baseline_artifacts:
        raise ValueError(f"No metrics.json artifacts found under baseline root {baseline_root}")
    if not current_artifacts:
        raise ValueError(f"No metrics.json artifacts found under current root {current_root}")

    missing_current = sorted(set(baseline_artifacts) - set(current_artifacts))
    if missing_current:
        missing = ", ".join(missing_current)
        raise ValueError(f"Current root {current_root} is missing baseline metrics artifacts: {missing}")

    return [
        MetricArtifactPair(
            relative_path=relative_path,
            baseline=baseline_path,
            current=current_artifacts[relative_path],
        )
        for relative_path, baseline_path in sorted(baseline_artifacts.items())
    ]


def _prefixed_failure(*, relative_path: str, failure: MetricRegression) -> MetricRegression:
    return MetricRegression(
        path=f"{relative_path}:{failure.path}",
        baseline=failure.baseline,
        current=failure.current,
        drop_percent=failure.drop_percent,
    )


def _compare_metric_artifact_pairs(
    pairs: list[MetricArtifactPair],
    *,
    max_drop_percent: float,
) -> dict[str, Any]:
    checked_metrics = 0
    failures: list[MetricRegression] = []

    for pair in pairs:
        result = compare_phase_metrics(
            current=_load_json(pair.current),
            baseline=_load_json(pair.baseline),
            max_drop_percent=max_drop_percent,
        )
        checked_metrics += result.checked_metrics
        failures.extend(
            _prefixed_failure(relative_path=pair.relative_path, failure=failure) for failure in result.failures
        )

    return {
        "status": "fail" if failures else "pass",
        "max_drop_percent": max_drop_percent,
        "checked_artifacts": len(pairs),
        "checked_metrics": checked_metrics,
        "failures": [asdict(failure) for failure in failures],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current and baseline phase metrics JSON for regressions.")
    parser.add_argument("--current", type=Path, help="Current phase metrics JSON path.")
    parser.add_argument("--baseline", type=Path, help="Baseline phase metrics JSON path.")
    parser.add_argument("--current-root", type=Path, help="Directory containing current phase metrics JSON artifacts.")
    parser.add_argument("--baseline-root", type=Path, help="Directory containing baseline phase metrics JSON artifacts.")
    parser.add_argument(
        "--max-drop-percent",
        type=float,
        default=DEFAULT_MAX_DROP_PERCENT,
        help=f"Maximum allowed measured numeric metric drop percentage. Defaults to {DEFAULT_MAX_DROP_PERCENT}.",
    )
    args = parser.parse_args()

    direct_files_requested = args.current is not None or args.baseline is not None
    roots_requested = args.current_root is not None or args.baseline_root is not None
    if direct_files_requested == roots_requested:
        parser.error("Provide either --current/--baseline or --current-root/--baseline-root.")
    if direct_files_requested and (args.current is None or args.baseline is None):
        parser.error("--current and --baseline must be provided together.")
    if roots_requested and (args.current_root is None or args.baseline_root is None):
        parser.error("--current-root and --baseline-root must be provided together.")

    try:
        if direct_files_requested:
            current_path = args.current
            baseline_path = args.baseline
            assert current_path is not None
            assert baseline_path is not None
            result = compare_phase_metrics(
                current=_load_json(current_path),
                baseline=_load_json(baseline_path),
                max_drop_percent=args.max_drop_percent,
            )
            payload = result.to_json_dict()
        else:
            current_root = args.current_root
            baseline_root = args.baseline_root
            assert current_root is not None
            assert baseline_root is not None
            pairs = _discover_paired_metric_artifacts(current_root=current_root, baseline_root=baseline_root)
            payload = _compare_metric_artifact_pairs(pairs, max_drop_percent=args.max_drop_percent)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
