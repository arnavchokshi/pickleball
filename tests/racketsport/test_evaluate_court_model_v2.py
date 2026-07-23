from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from threed.racketsport.court_keypoint_net import make_court_keypoint_heatmap_model  # noqa: E402
from scripts.racketsport.evaluate_court_model_v2 import (  # noqa: E402
    _row_native_image_bgr,
    evaluate_checkpoint_against_real_labels,
    evaluate_structured_checkpoint_against_real_labels,
    predict_row_keypoints_source_px,
)

KEYPOINT_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checkpoint(tmp_path: Path) -> Path:
    model = make_court_keypoint_heatmap_model(len(KEYPOINT_NAMES), architecture="court_unet_v2")
    checkpoint_path = tmp_path / "court_model_v2.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": [64, 36],
            "model_architecture": "court_unet_v2",
            "network_architecture": "court_unet_v2",
            "keypoint_names": KEYPOINT_NAMES,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_owner_clip(
    root: Path,
    clip_name: str,
    *,
    loaded_size: tuple[int, int],
    source_size: tuple[int, int],
    frame_count: int = 4,
) -> None:
    """Build one `<clip>/labels/court_keypoints.json` row set shaped like the real
    `eval_clips/ball/*/labels/court_keypoints.json` rows: a saved preview JPEG at `loaded_size`,
    keypoint labels scaled up to a *different* `source_size` (mirroring the real
    1280x720-preview-vs-1920x1080-source_video_size case this eval harness must rescale
    correctly), and a `source.mp4` at `source_size` as the fallback video path.
    """

    from threed.racketsport.court_keypoint_net import keypoint_labels_from_court_corners

    clip_dir = root / clip_name
    frames_dir = clip_dir / "labels" / "court_keypoint_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    loaded_width, loaded_height = loaded_size
    source_width, source_height = source_size
    labels_source_space = keypoint_labels_from_court_corners(
        {
            "near_left": [source_width * 0.1, source_height * 0.9],
            "near_right": [source_width * 0.9, source_height * 0.9],
            "far_right": [source_width * 0.65, source_height * 0.15],
            "far_left": [source_width * 0.35, source_height * 0.15],
        }
    )

    items = []
    for frame_index in range(frame_count):
        frame_name = f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frames_dir / frame_name), np.full((loaded_height, loaded_width, 3), 50, dtype=np.uint8))
        items.append(
            {
                "frame": frame_name,
                "status": "reviewed" if frame_index == 0 else "reviewed_static_camera_copy",
                "keypoints": labels_source_space,
            }
        )

    _write_json(
        clip_dir / "labels" / "court_keypoints.json",
        {
            "annotation": {"items": items},
            "frames": {
                "frame_dir": str(frames_dir),
                "source_resolution": [source_width, source_height],
                "label_coordinate_space": [loaded_width, loaded_height],
            },
            "review": {"status": "reviewed", "reviewer": "test"},
        },
    )

    writer = cv2.VideoWriter(str(clip_dir / "source.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, source_size)
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.full((source_height, source_width, 3), 80, dtype=np.uint8))
    writer.release()


def test_row_native_image_bgr_reports_rescale_matching_declared_source_size(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _write_owner_clip(owner_root, "some_clip", loaded_size=(128, 72), source_size=(256, 144))

    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(owner_root)
    image_bgr, (scale_x, scale_y) = _row_native_image_bgr(rows[0])

    assert image_bgr.shape == (72, 128, 3)  # the cached preview JPEG's own resolution
    assert scale_x == pytest.approx(2.0)
    assert scale_y == pytest.approx(2.0)


def test_predict_row_keypoints_source_px_lands_in_declared_source_space(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _write_owner_clip(owner_root, "some_clip", loaded_size=(128, 72), source_size=(256, 144))
    checkpoint_path = _write_checkpoint(tmp_path)

    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(owner_root)
    predicted = predict_row_keypoints_source_px(rows[0], checkpoint_path, device="cpu")

    for name in KEYPOINT_NAMES:
        x, y = predicted[name]
        # Predictions are rescaled into the row's declared 256x144 source space, not the loaded
        # 128x72 preview's own pixel space.
        assert 0.0 <= x <= 256.0
        assert 0.0 <= y <= 144.0


def test_evaluate_checkpoint_against_real_labels_reports_independent_and_all_modes(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _write_owner_clip(owner_root, "clip_a", loaded_size=(64, 36), source_size=(64, 36))
    _write_owner_clip(owner_root, "clip_b", loaded_size=(64, 36), source_size=(128, 72))
    checkpoint_path = _write_checkpoint(tmp_path)

    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(owner_root)
    assert len(rows) == 8  # 2 clips x 4 rows

    report = evaluate_checkpoint_against_real_labels(checkpoint_path, rows, device="cpu")

    assert report["independent_frame_count"] == 2  # 1 independent-reviewed row per clip
    assert report["all_frame_count"] == 8
    assert report["independent"]["keypoint_error_summary"]["count"] > 0
    assert set(report["independent"]["per_clip"]) == {"clip_a", "clip_b"}
    assert set(report["all"]["per_clip"]) == {"clip_a", "clip_b"}


def test_structured_eval_scores_floor_names_and_stays_review_only(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _write_owner_clip(owner_root, "clip_a", loaded_size=(64, 36), source_size=(64, 36))
    checkpoint_path = _write_checkpoint(tmp_path)

    from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

    rows = load_real_court_keypoint_labels(owner_root)[:1]
    report = evaluate_structured_checkpoint_against_real_labels(checkpoint_path, rows, device="cpu")

    assert report["evaluated_taxonomy"] == "12_canonical_floor_points_exact_name"
    assert report["point_metrics"]["labeled_count"] == 12
    assert report["authority_state"] == "review_only"
    assert report["measurement_valid"] is False
    assert report["promotion_allowed"] is False


def test_evaluate_court_model_v2_cli_writes_scanner_compatible_report(tmp_path: Path) -> None:
    """End-to-end CLI smoke test for `scripts/racketsport/evaluate_court_model_v2.py` (also this
    script's scaffold-index direct CLI reference test). Verifies the output JSON carries the
    exact top-level fields the existing CAL evidence scanner
    (`overlapping_court_calibration._neural_keypoint_checkpoint_evidence`, which globs
    `runs/**/court_keypoint_metrics.json` for `checkpoint`/`gate`/`after`) reads, with
    `after.real_keypoint_median_px` populated and the gate fixed at PCK@5px>=0.95."""

    owner_root = tmp_path / "owner"
    _write_owner_clip(owner_root, "some_clip", loaded_size=(64, 36), source_size=(64, 36))
    checkpoint_path = _write_checkpoint(tmp_path)
    out_path = tmp_path / "court_keypoint_metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_model_v2.py",
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
    assert report["artifact_type"] == "court_keypoint_owner_gate_report_v2"
    assert report["checkpoint"] == str(checkpoint_path)
    assert report["gate"]["pck_threshold_px"] == 5.0
    assert report["gate"]["threshold"] == pytest.approx(0.95)
    assert "value" in report["gate"] and "passed" in report["gate"]
    assert report["after"]["real_keypoint_median_px"] is not None
    assert report["independent_frame_count"] == 1
    assert report["all_frame_count"] == 4
    expected_modes = {"independent", "all"}
    assert set(report["gate_passed_pooled"]) == expected_modes
    assert set(report["gate_passed_per_viewpoint"]) == expected_modes
    assert report["gate_passed"] == report["gate_passed_per_viewpoint"]
    assert json.loads(completed.stdout)["artifact_type"] == "court_keypoint_owner_gate_report_v2"
