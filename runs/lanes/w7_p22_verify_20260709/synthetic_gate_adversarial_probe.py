#!/usr/bin/env python3
"""Adversarial probes for the w7 synthetic BODY decode gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.racketsport import synthetic_body_decode_gate as gate


ROOT = Path("runs/lanes/w7_p22_verify_20260709")


def _args(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        out=ROOT / f"{name}.json",
        render_dir=ROOT / f"{name}_renders",
        samples=2,
        seed=123,
        decoder="mock",
    )


def _scaled(points: list[list[float]], scale: float) -> list[list[float]]:
    return [[float(value) * scale for value in point] for point in points]


def main() -> int:
    original_mock_decode = gate._mock_decode
    original_apply_pred_cam_t_once = gate.mhr_decode.apply_pred_cam_t_once
    results: dict[str, Any] = {}

    def mis_scaled_decode(sample: dict[str, Any]) -> dict[str, list[list[float]]]:
        decoded = original_mock_decode(sample)
        return {
            "joints_world": _scaled(decoded["joints_world"], 1.10),
            "vertices_world": _scaled(decoded["vertices_world"], 1.10),
        }

    def wrong_double_apply(
        points_camera: Any,
        *,
        pred_cam_t: list[float] | tuple[float, ...] | None = None,
        already_applied: bool = False,
    ) -> list[list[float]]:
        if points_camera is None:
            return []
        points = [[float(value) for value in point] for point in points_camera]
        if pred_cam_t is None or already_applied:
            return points
        cam = [float(pred_cam_t[idx]) for idx in range(3)]
        return [[point[idx] + 2.0 * cam[idx] for idx in range(3)] for point in points]

    try:
        gate._mock_decode = mis_scaled_decode
        mis_scaled_report = gate.run(_args("synthetic_gate_mis_scaled_decoder_report"))
        results["mis_scaled_decoder"] = {
            "gate_1b_passed": mis_scaled_report["gate_1b_world_round_trip"]["passed"],
            "joints_world_p95_abs_error_mm": mis_scaled_report["gate_1b_world_round_trip"][
                "joints_world_p95_abs_error_mm"
            ],
            "mesh_skeleton_divergence_passed": mis_scaled_report["mesh_skeleton_divergence"]["passed"],
        }
    finally:
        gate._mock_decode = original_mock_decode

    try:
        gate.mhr_decode.apply_pred_cam_t_once = wrong_double_apply
        self_referential_report = gate.run(_args("synthetic_gate_wrong_helper_report"))
        results["wrong_helper_used_for_truth_and_mock_decode"] = {
            "gate_1b_passed": self_referential_report["gate_1b_world_round_trip"]["passed"],
            "joints_world_p95_abs_error_mm": self_referential_report["gate_1b_world_round_trip"][
                "joints_world_p95_abs_error_mm"
            ],
            "mesh_skeleton_divergence_passed": self_referential_report["mesh_skeleton_divergence"]["passed"],
            "interpretation": (
                "The gate still passes when the helper used to author truth is wrong, because the mock decode "
                "uses the same helper."
            ),
        }
    finally:
        gate.mhr_decode.apply_pred_cam_t_once = original_apply_pred_cam_t_once

    (ROOT / "synthetic_gate_adversarial_probe_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
