from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport.train_court_keypoint_heatmap import run_training
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_court_unet_v2_checkpoint(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")

    from threed.racketsport.court_keypoint_net import make_court_keypoint_heatmap_model

    checkpoint_path = tmp_path / "court_unet_v2.pt"
    model = make_court_keypoint_heatmap_model(len(PICKLEBALL_KEYPOINTS), architecture="court_unet_v2")
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": [32, 18],
            "model_architecture": "court_unet_v2",
            "network_architecture": "court_unet_v2",
            "keypoint_names": [point.name for point in PICKLEBALL_KEYPOINTS],
        },
        checkpoint_path,
    )
    return checkpoint_path


def _build_checkpoint_and_owner_labels(tmp_path: Path) -> tuple[Path, Path]:
    """Train a throwaway checkpoint on one synthetic "external corpus" clip, then write a
    second, disjoint clip shaped like the real `eval_clips/ball/*/labels/court_keypoints.json`
    (1 independent review + 3 owner-approved static-camera copies) to stand in for the owner
    gate's real_root. Returns (checkpoint_path, owner_real_root).
    """
    cv2 = __import__("cv2")

    train_root = tmp_path / "external_corpus"
    train_clip = train_root / "some_external_dataset"
    train_clip.mkdir(parents=True)
    train_video = train_clip / "source.mp4"
    writer = cv2.VideoWriter(str(train_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.release()
    _write_json(
        train_clip / "labels" / "court_keypoints.json",
        {
            "schema_version": 1,
            "annotation": {
                "items": [
                    {
                        "frame": "frame_000000.jpg",
                        "status": "reviewed",
                        "keypoints": {
                            point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                            for index, point in enumerate(PICKLEBALL_KEYPOINTS)
                        },
                    }
                ]
            },
            "frames": {
                "frame_dir": "runs/label_frames/some_external_dataset",
                "source_resolution": [64, 36],
                "label_coordinate_space": [64, 36],
            },
            "review": {"status": "reviewed", "reviewer": "court-label-review"},
        },
    )

    out = tmp_path / "checkpoint_run"
    summary = run_training(
        Namespace(
            real_root=train_root,
            out=out,
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=0,
            batch_size=1,
            image_width=32,
            image_height=18,
            sigma=1.5,
            learning_rate=1e-3,
            real_finetune_start_epoch=0,
            eval_every=1,
            seed=13,
            device="cpu",
            skip_holdout_artifacts=True,
            static_camera_aggregate=False,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
        )
    )
    checkpoint_path = Path(summary["checkpoint"])
    assert checkpoint_path.is_file()

    owner_root = tmp_path / "eval_clips" / "ball"
    owner_clip = owner_root / "some_owner_clip"
    owner_clip.mkdir(parents=True)
    owner_video = owner_clip / "source.mp4"
    writer = cv2.VideoWriter(str(owner_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(4):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (10 + idx * 20, 40, 70)
        writer.write(frame)
    writer.release()
    payload = {
        "annotation": {
            "items": [
                {
                    "frame": f"frame_{frame_index:06d}.jpg",
                    "status": "reviewed" if frame_index == 0 else "reviewed_static_camera_copy",
                    "keypoints": {
                        point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
                    },
                }
                for frame_index in range(4)
            ]
        },
        "frames": {
            "frame_dir": "runs/label_frames/some_owner_clip",
            "source_resolution": [64, 36],
            "label_coordinate_space": [64, 36],
        },
        "review": {"status": "reviewed", "reviewer": "owner-review"},
    }
    _write_json(owner_clip / "labels" / "court_keypoints.json", payload)

    return checkpoint_path, owner_root


def test_evaluate_court_keypoint_owner_gate_cli_accepts_court_unet_v2_checkpoint(tmp_path: Path) -> None:
    """Current CALV1 checkpoints use the `court_unet_v2` network, whose forward pass returns
    a dict with `keypoint_heatmaps`. The owner-gate CLI still owns the pre-registered gate
    contract, so it must decode that output shape instead of crashing before scoring."""
    _legacy_checkpoint_path, owner_root = _build_checkpoint_and_owner_labels(tmp_path)
    checkpoint_path = _write_court_unet_v2_checkpoint(tmp_path)
    out_path = tmp_path / "gate_report_unet_v2.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_owner_gate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--real-root",
            str(owner_root),
            "--out",
            str(out_path),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "court_keypoint_owner_gate_report"
    assert report["checkpoint"] == str(checkpoint_path)
    assert report["raw_independent"]["keypoint_error_summary"]["count"] == len(PICKLEBALL_KEYPOINTS)
    assert json.loads(completed.stdout)["checkpoint"] == str(checkpoint_path)


def test_evaluate_court_keypoint_owner_gate_cli_writes_gate_report(tmp_path: Path) -> None:
    """End-to-end CLI smoke test for `scripts/racketsport/evaluate_court_keypoint_owner_gate.py`
    -- the single, read-only, pre-registered runner for the CAL owner-clip gate. Runs the real
    script as a subprocess (not just the underlying function) so the argparse wiring and JSON
    output path are exercised, not only the library call.
    """
    checkpoint_path, owner_root = _build_checkpoint_and_owner_labels(tmp_path)
    out_path = tmp_path / "gate_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_owner_gate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--real-root",
            str(owner_root),
            "--out",
            str(out_path),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.is_file()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "court_keypoint_owner_gate_report"
    assert report["independent_frame_count"] == 1
    assert report["all_frame_count"] == 4
    # COURT-LOADER-1 regression: the owner-gate's existing full-15 rows still score all 15
    # points per row; partial/external masking cannot alter this path.
    assert [row["keypoint_count"] for row in report["raw_all"]["per_row"]] == [15, 15, 15, 15]
    assert report["raw_all"]["keypoint_error_summary"]["count"] == 60
    expected_modes = {"raw_independent", "raw_all", "aggregated_independent", "aggregated_all"}
    assert set(report["gate_passed"]) == expected_modes
    assert set(report["gate_passed_pooled"]) == expected_modes
    assert set(report["gate_passed_per_viewpoint"]) == expected_modes
    # "gate_passed" is an alias for the real per-viewpoint gate (every clip must individually
    # clear the threshold), not the pooled-across-all-clips number.
    assert report["gate_passed"] == report["gate_passed_per_viewpoint"]
    assert report["gate_threshold"] == pytest.approx(0.95)
    # stdout should also carry the same JSON payload for quick human inspection.
    assert json.loads(completed.stdout)["artifact_type"] == "court_keypoint_owner_gate_report"
