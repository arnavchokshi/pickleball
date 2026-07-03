from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from threed.racketsport.court_detector_v2_model import (
    make_court_detector_v2_model,
    make_mobilenet_v3_court_keypoint_regressor,
    make_resnet50_court_keypoint_regressor,
)


def test_detector_v2_model_output_shapes() -> None:
    model = make_court_detector_v2_model(keypoint_count=15, line_count=8, net_count=3)
    x = torch.zeros((2, 3, 256, 456))

    out = model(x)

    assert out["keypoint_heatmaps"].shape == (2, 15, 256, 456)
    assert out["line_masks"].shape == (2, 8, 256, 456)
    assert out["net_masks"].shape == (2, 3, 256, 456)
    assert out["visibility_logits"].shape == (2, 15)


def test_resnet50_keypoint_regressor_outputs_xy_and_visibility_channels() -> None:
    model = make_resnet50_court_keypoint_regressor(keypoint_count=12, weights=None)
    x = torch.zeros((2, 3, 96, 96), dtype=torch.float32)

    out = model(x)

    assert tuple(out.shape) == (2, 36)
    assert model.court_keypoint_count == 12
    assert model.court_keypoint_output_layout == "x_y_visibility_per_keypoint"


def test_mobilenet_v3_keypoint_regressor_is_lightweight_and_outputs_xy_visibility_channels() -> None:
    model = make_mobilenet_v3_court_keypoint_regressor(keypoint_count=15, weights=None)
    x = torch.zeros((2, 3, 96, 96), dtype=torch.float32)

    out = model(x)

    assert tuple(out.shape) == (2, 45)
    assert model.court_keypoint_count == 15
    assert model.court_keypoint_output_layout == "x_y_visibility_per_keypoint"
    assert sum(parameter.numel() for parameter in model.parameters()) < 4_000_000


def test_train_court_detector_v2_multitask_dry_run_writes_not_verified_report(tmp_path: Path) -> None:
    out = tmp_path / "court_detector_v2_train_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_detector_v2_multitask.py",
            "--eval-root",
            "eval_clips/ball",
            "--out",
            str(out),
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["artifact_type"] == "court_detector_v2_multitask_training_report"
    assert payload["status"] == "trained_not_cal3_verified"
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["model_heads"] == ["keypoint_heatmaps", "line_masks", "net_masks", "visibility_logits"]
