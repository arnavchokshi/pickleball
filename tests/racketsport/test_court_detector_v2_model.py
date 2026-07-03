from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from threed.racketsport.court_detector_v2_model import (
    evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint,
    make_court_detector_v2_model,
    make_mobilenet_v3_court_keypoint_regressor,
    make_resnet50_court_keypoint_regressor,
    train_mobilenet_v3_court_keypoint_regressor,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


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


def test_mobilenet_v3_checkpoint_eval_fails_closed_when_checkpoint_missing(tmp_path: Path) -> None:
    report = evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint(
        checkpoint_path=tmp_path / "missing_mobilenet_v3.pt",
        rows=[],
        device="cpu",
    )

    assert report["artifact_type"] == "mobilenet_v3_court_keypoint_regressor_eval"
    assert report["status"] == "unavailable"
    assert report["reason"] == "missing_checkpoint"
    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert report["diagnostic_only"] is True
    assert report["promotes_calibration"] is False
    assert report["gate_passed"] is False


def test_mobilenet_v3_checkpoint_eval_scores_reviewed_rows_from_normalized_xy_logits(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (96, 96), (12, 34, 56)).save(image_path)
    keypoint_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    keypoints = {
        name: [12.0 + index * 3.0, 20.0 + index * 2.0]
        for index, name in enumerate(keypoint_names)
    }
    model = make_mobilenet_v3_court_keypoint_regressor(keypoint_count=len(keypoint_names), weights=None)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        logits = []
        for name in keypoint_names:
            x, y = keypoints[name]
            for value, scale in ((x, 96.0), (y, 96.0), (0.99, 1.0)):
                normalized = max(1e-6, min(1.0 - 1e-6, value / scale))
                logits.append(torch.logit(torch.tensor(normalized)).item())
        model.classifier[-1].bias.copy_(torch.tensor(logits, dtype=torch.float32))
    checkpoint_path = tmp_path / "mobilenet_v3_regressor.pt"
    torch.save(
        {
            "architecture": "mobilenet_v3_small_regressor",
            "model_state_dict": model.state_dict(),
            "keypoint_names": keypoint_names,
            "input_size": [96, 96],
            "coordinate_mode": "sigmoid_normalized_xy",
        },
        checkpoint_path,
    )

    report = evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint(
        checkpoint_path=checkpoint_path,
        rows=[
            {
                "image_path": str(image_path),
                "source_video_size": [96, 96],
                "keypoints": keypoints,
                "label_status": "reviewed",
                "clip": "synthetic_clip",
            }
        ],
        device="cpu",
        pck_threshold_px=5.0,
    )

    assert report["status"] == "scored"
    assert report["architecture"] == "mobilenet_v3_small_regressor"
    assert report["coordinate_mode"] == "sigmoid_normalized_xy"
    assert report["reviewed_row_count"] == 1
    assert report["evaluated_keypoint_count"] == 15
    assert report["mean_error_px"] == pytest.approx(0.0, abs=1e-4)
    assert report["median_error_px"] == pytest.approx(0.0, abs=1e-4)
    assert report["pck_at_5px"] == 1.0
    assert report["gate_passed"] is True
    assert report["diagnostic_only"] is True
    assert report["promotes_calibration"] is False


def test_train_mobilenet_v3_regressor_writes_checkpoint_and_scores_holdout(tmp_path: Path) -> None:
    from PIL import Image

    keypoint_names = [point.name for point in PICKLEBALL_KEYPOINTS]
    rows = []
    for index, clip in enumerate(("train_clip", "holdout_clip")):
        image_path = tmp_path / f"{clip}.png"
        Image.new("RGB", (64, 64), (20 + index * 40, 30, 60)).save(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "source_video_size": [64, 64],
                "keypoints": {
                    name: [10.0 + point_index * 2.0 + index, 12.0 + point_index * 1.5]
                    for point_index, name in enumerate(keypoint_names)
                },
                "label_status": "reviewed",
                "clip": clip,
            }
        )

    report = train_mobilenet_v3_court_keypoint_regressor(
        rows=rows,
        out_dir=tmp_path / "mobilenet_run",
        holdout_clip_names={"holdout_clip"},
        input_size=(64, 64),
        epochs=1,
        learning_rate=0.01,
        device="cpu",
        seed=7,
    )

    checkpoint = Path(report["checkpoint"])
    assert report["artifact_type"] == "mobilenet_v3_court_keypoint_regressor_training_report"
    assert report["status"] == "trained_not_cal3_verified"
    assert report["verified"] is False
    assert report["not_cal3_verified"] is True
    assert report["diagnostic_only"] is True
    assert report["promotes_calibration"] is False
    assert report["train_row_count"] == 1
    assert report["holdout_row_count"] == 1
    assert report["training"]["epoch_count"] == 1
    assert report["training"]["final_loss"] is not None
    assert checkpoint.name == "mobilenet_v3_court_keypoint_regressor.pt"
    assert checkpoint.is_file()
    assert Path(report["metrics_path"]).is_file()
    assert report["evaluation"]["status"] == "scored"
    assert report["evaluation"]["evaluated_keypoint_count"] == 15
    assert report["evaluation"]["diagnostic_only"] is True
    assert report["evaluation"]["promotes_calibration"] is False


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
