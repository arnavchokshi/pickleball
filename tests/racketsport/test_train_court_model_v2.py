from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import COURT_UNET_V2_HEATMAP_STRIDE, PICKLEBALL_KEYPOINTS

torch = pytest.importorskip("torch")

import scripts.racketsport.train_court_model_v2 as train_court_model_v2  # noqa: E402

KEYPOINT_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


# ---------------------------------------------------------------------------------------------
# Fallback sampler + sample -> tensor conversion (fully self-contained, never needs CAL-SYNTH)
# ---------------------------------------------------------------------------------------------


def test_fallback_synthetic_sample_matches_cal_synth_shape_contract() -> None:
    import random

    rng = random.Random(7)
    sample = train_court_model_v2._fallback_synthetic_sample(rng, (320, 180))

    assert sample["image_bgr"].shape == (180, 320, 3)
    assert sample["image_bgr"].dtype == np.uint8
    assert sample["keypoints_xy"].shape == (15, 2)
    assert sample["keypoints_xy"].dtype == np.float32
    assert sample["keypoints_vis"].shape == (15,)
    assert set(np.unique(sample["keypoints_vis"]).tolist()) <= {0, 1, 2}
    assert sample["line_family_mask"].shape == (180, 320)
    assert set(np.unique(sample["line_family_mask"]).tolist()) <= {0, 1, 2, 3}
    assert sample["surface_mask"].shape == (180, 320)
    assert set(np.unique(sample["surface_mask"]).tolist()) <= {0, 1, 2}
    # The court interior must actually be drawn (non-trivial surface mask), otherwise the
    # merge-into-5-class-seg-target step would silently supervise an all-background image.
    assert int((sample["surface_mask"] == 2).sum()) > 0


def test_sample_to_training_arrays_produces_expected_shapes_and_masks() -> None:
    import random

    rng = random.Random(3)
    sample = train_court_model_v2._fallback_synthetic_sample(rng, (640, 360))

    arrays = train_court_model_v2.sample_to_training_arrays(
        sample,
        model_width=640,
        model_height=360,
        heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE,
        sigma_px=1.5,
        has_seg_target=True,
    )

    assert arrays["image"].shape == (3, 360, 640)
    assert arrays["heatmaps"].shape == (15, 90, 160)
    assert arrays["heatmap_mask"].shape == (15,)
    assert arrays["vis_target"].shape == (15,)
    assert arrays["seg_target"].shape == (90, 160)
    assert float(arrays["seg_mask"]) == 1.0
    # Every non-off-frame channel's heatmap must peak near its labeled keypoint location.
    for index in range(15):
        if arrays["heatmap_mask"][index] <= 0:
            continue
        peak_y, peak_x = np.unravel_index(np.argmax(arrays["heatmaps"][index]), arrays["heatmaps"][index].shape)
        expected_x = sample["keypoints_xy"][index, 0] / COURT_UNET_V2_HEATMAP_STRIDE
        expected_y = sample["keypoints_xy"][index, 1] / COURT_UNET_V2_HEATMAP_STRIDE
        assert abs(peak_x - expected_x) <= 1.5
        assert abs(peak_y - expected_y) <= 1.5


def test_sample_to_training_arrays_without_seg_target_masks_seg_loss() -> None:
    import random

    rng = random.Random(4)
    sample = train_court_model_v2._fallback_synthetic_sample(rng, (640, 360))
    arrays = train_court_model_v2.sample_to_training_arrays(
        sample, model_width=640, model_height=360, has_seg_target=False
    )
    assert float(arrays["seg_mask"]) == 0.0


# ---------------------------------------------------------------------------------------------
# CAL-SYNTH contract integration: real module when present, clean skip/fallback otherwise
# ---------------------------------------------------------------------------------------------


def test_real_cal_synth_contract_produces_valid_samples_when_available() -> None:
    """Integration test against CAL-SYNTH's actual stable contract. Skips cleanly if the module
    (or the specific attribute) is not importable -- per the CAL-MODEL spec, this lane must never
    hard-fail on CAL-SYNTH's landing order."""

    pytest.importorskip("threed.racketsport.court_synth_stream")
    module = importlib.import_module("threed.racketsport.court_synth_stream")
    if not hasattr(module, "iter_synthetic_court_samples"):
        pytest.skip("court_synth_stream.iter_synthetic_court_samples not present yet")

    samples = train_court_model_v2._iter_synthetic_samples(
        {"image_size": [640, 360]}, seed=11, force_fallback=False
    )
    sample = next(iter(samples))
    assert sample["image_bgr"].shape == (360, 640, 3)
    assert sample["keypoints_xy"].shape == (15, 2)
    assert sample["keypoints_vis"].shape == (15,)


def test_missing_cal_synth_module_falls_back_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates CAL-SYNTH's module/attribute being missing (it lands in this same worktree in
    parallel) by monkeypatching the trainer's guarded import handle to None -- the exact state it
    would be in if the `except Exception` import guard at module load time had triggered. The
    trainer must keep working via the built-in fallback sampler, never raise."""

    monkeypatch.setattr(train_court_model_v2, "_iter_synthetic_court_samples", None)

    samples = train_court_model_v2._iter_synthetic_samples(
        {"image_size": [320, 180]}, seed=5, force_fallback=False
    )
    sample = next(iter(samples))
    assert sample["image_bgr"].shape == (180, 320, 3)
    assert sample["meta"]["source"] == "cal_model_synthetic_fallback"


def test_synthetic_fallback_flag_forces_fallback_even_when_module_available() -> None:
    samples = train_court_model_v2._iter_synthetic_samples(
        {"image_size": [320, 180]}, seed=5, force_fallback=True
    )
    sample = next(iter(samples))
    assert sample["meta"]["source"] == "cal_model_synthetic_fallback"


# ---------------------------------------------------------------------------------------------
# IterableDataset per-worker seeding
# ---------------------------------------------------------------------------------------------


class _FakeWorkerInfo:
    def __init__(self, worker_id: int) -> None:
        self.id = worker_id


def test_synthetic_court_iterable_dataset_per_worker_seeding_is_deterministic_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = train_court_model_v2.SyntheticCourtIterableDataset(
        config={"image_size": [320, 180]},
        base_seed=42,
        model_width=320,
        model_height=180,
        sigma_px=1.5,
        force_fallback=True,
        samples_per_epoch=2,
    )

    def _first_image(worker_id: int) -> np.ndarray:
        import torch.utils.data as data_module

        monkeypatch.setattr(data_module, "get_worker_info", lambda: _FakeWorkerInfo(worker_id))
        rows = list(iter(dataset))
        assert len(rows) == 2
        return rows[0]["image"]

    worker0_run1 = _first_image(0)
    worker0_run2 = _first_image(0)
    worker1_run1 = _first_image(1)

    # Same worker id + same base_seed -> identical stream (determinism).
    assert np.allclose(worker0_run1, worker0_run2)
    # Different worker id -> a different offset seed, so a different stream.
    assert not np.allclose(worker0_run1, worker1_run1)


# ---------------------------------------------------------------------------------------------
# On-disk real corpora conversion: correct rescale even when the loaded image resolution differs
# from the row's declared source_video_size (the 1280x720-vs-1920x1080 case).
# ---------------------------------------------------------------------------------------------


def test_real_row_to_sample_arrays_rescales_by_actual_loaded_image_size(tmp_path: Path) -> None:
    import cv2

    from threed.racketsport.court_keypoint_net import keypoint_labels_from_court_corners

    # Loaded preview image is 128x72 ("label" resolution); the row declares a *different*
    # source_video_size (256x144, exactly 2x). Keypoints are stored in that 256x144 space.
    loaded_width, loaded_height = 128, 72
    source_width, source_height = 256, 144
    image_path = tmp_path / "frame_000000.jpg"
    cv2.imwrite(str(image_path), np.full((loaded_height, loaded_width, 3), 60, dtype=np.uint8))

    labels_source_space = keypoint_labels_from_court_corners(
        {
            "near_left": [20.0, 130.0],
            "near_right": [236.0, 130.0],
            "far_right": [170.0, 20.0],
            "far_left": [86.0, 20.0],
        }
    )
    row = {
        "image_path": str(image_path),
        "video_path": None,
        "frame_index": 0,
        "source_video_size": [source_width, source_height],
        "keypoints": labels_source_space,
    }

    arrays = train_court_model_v2.real_row_to_sample_arrays(row, model_width=128, model_height=72, sigma_px=1.5)

    assert arrays["image"].shape == (3, 72, 128)
    assert float(arrays["seg_mask"]) == 0.0  # real rows never carry segmentation ground truth
    assert bool((arrays["vis_target"] == 1.0).all())  # every labeled real keypoint is "visible"

    # The near_left_corner heatmap peak (in model-input pixel space) must land at the label's
    # position rescaled DOWN into the actually-loaded 128x72 image space (i.e. halved), not at
    # the raw 256x144 source-space coordinate.
    near_left_index = KEYPOINT_NAMES.index("near_left_corner")
    peak_y, peak_x = np.unravel_index(
        np.argmax(arrays["heatmaps"][near_left_index]), arrays["heatmaps"][near_left_index].shape
    )
    expected_x = (20.0 * (loaded_width / source_width)) / COURT_UNET_V2_HEATMAP_STRIDE
    expected_y = (130.0 * (loaded_height / source_height)) / COURT_UNET_V2_HEATMAP_STRIDE
    assert abs(peak_x - expected_x) <= 1.5
    assert abs(peak_y - expected_y) <= 1.5


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------


def test_class_balanced_seg_loss_ignores_rows_with_no_seg_target() -> None:
    logits = torch.randn(2, 5, 4, 4, requires_grad=True)
    target = torch.zeros(2, 4, 4, dtype=torch.int64)
    sample_mask = torch.tensor([0.0, 0.0])

    loss = train_court_model_v2.class_balanced_seg_loss(logits, target, sample_mask, torch=torch)
    assert float(loss) == pytest.approx(0.0)


def test_class_balanced_seg_loss_upweights_rare_classes() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(1, 5, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.int64)
    target[0, 0, 0] = 3  # one rare "net" pixel among 63 background pixels
    sample_mask = torch.tensor([1.0])

    loss = train_court_model_v2.class_balanced_seg_loss(logits, target, sample_mask, torch=torch)
    assert float(loss.detach()) > 0.0
    assert torch.isfinite(loss)


def test_visibility_bce_loss_is_finite_and_zero_for_perfect_logits() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0]])
    perfect_logits = torch.tensor([[20.0, -20.0, 20.0]])
    loss = train_court_model_v2.visibility_bce_loss(perfect_logits, target, torch=torch)
    assert float(loss) < 1e-6


# ---------------------------------------------------------------------------------------------
# CPU smoke acceptance proof (subprocess CLI run; also this script's direct-cli-reference test)
# ---------------------------------------------------------------------------------------------


def test_train_court_model_v2_cli_cpu_smoke_beats_random_init(tmp_path: Path) -> None:
    """CAL-MODEL CPU-provable acceptance proof: a 2-epoch run (batch-size 64, i.e. 64
    synthetic-fallback training samples per epoch) at the real 640x360 architecture resolution
    must strictly decrease loss and beat random-init PCK@40px on a held-out 16-sample synthetic
    set generated once from a separate, fixed validation seed. Runs the actual CLI (this is also
    `scripts/racketsport/train_court_model_v2.py`'s scaffold-index direct CLI reference test)."""

    out_dir = tmp_path / "cal_model_v2_smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "2",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "64",
            "--image-width",
            "640",
            "--image-height",
            "360",
            "--val-samples",
            "16",
            "--synthetic-fallback",
            "--device",
            "cpu",
            "--eval-every",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=540,
    )

    metrics_path = out_dir / "court_keypoint_metrics.json"
    assert metrics_path.is_file()
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert summary["architecture"]["name"] == "court_unet_v2"
    assert 10_000_000 <= summary["architecture"]["param_count"] <= 35_000_000
    assert summary["architecture"]["heatmap_stride"] == COURT_UNET_V2_HEATMAP_STRIDE
    assert len(summary["history"]) == 2

    # Loss strictly decreases across the 2 epochs.
    assert summary["history"][1]["train_loss_last"] < summary["history"][0]["train_loss_last"]

    # Beats random-init PCK@40px on the fixed held-out synthetic set.
    before_pck40 = summary["before"]["pck_at_40px"]
    after_pck40 = summary["after"]["pck_at_40px"]
    assert before_pck40 is not None and after_pck40 is not None
    assert after_pck40 > before_pck40

    checkpoint_path = Path(summary["checkpoint"])
    assert checkpoint_path.is_file()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_payload["network_architecture"] == "court_unet_v2"
    assert checkpoint_payload["keypoint_names"] == KEYPOINT_NAMES

    # gate/checkpoint/after are all present -- this run's court_keypoint_metrics.json is scanned
    # by the existing CAL evidence scanner (overlapping_court_calibration
    # ._neural_keypoint_checkpoint_evidence) even though this particular gate is synthetic-only.
    assert summary["gate"]["pck_threshold_px"] == 5.0
    assert summary["gate"]["threshold"] == pytest.approx(0.95)
    assert "checkpoint" in summary and "after" in summary

    assert completed.returncode == 0
    cli_summary = json.loads(completed.stdout)
    assert cli_summary["checkpoint"] == summary["checkpoint"]


def test_train_court_model_v2_resume_continues_from_saved_epoch(tmp_path: Path) -> None:
    out_dir = tmp_path / "cal_model_v2_resume"
    base_args = [
        sys.executable,
        "scripts/racketsport/train_court_model_v2.py",
        "--out",
        str(out_dir),
        "--steps-per-epoch",
        "1",
        "--batch-size",
        "4",
        "--image-width",
        "160",
        "--image-height",
        "96",
        "--val-samples",
        "4",
        "--synthetic-fallback",
        "--device",
        "cpu",
    ]
    subprocess.run(base_args + ["--epochs", "1"], check=True, capture_output=True, text=True, timeout=120)
    checkpoint_path = out_dir / "court_model_v2.pt"
    assert checkpoint_path.is_file()

    completed = subprocess.run(
        base_args + ["--epochs", "2", "--resume", str(checkpoint_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    summary = json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    # Resumed from epoch 1 and only ran the remaining epoch (epoch 2), not both epochs again.
    assert [row["epoch"] for row in summary["history"]] == [2]


def _build_tiny_external_corpus(root: Path) -> None:
    import cv2

    clip = root / "some_external_clip"
    clip.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(clip / "source.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (128, 72))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((72, 128, 3), 90, dtype=np.uint8))
    writer.release()

    keypoints = {name: [float(10 + index * 5), float(10 + index * 3)] for index, name in enumerate(KEYPOINT_NAMES)}
    labels_dir = clip / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "court_keypoints.json").write_text(
        json.dumps(
            {
                "annotation": {
                    "items": [
                        {"frame": "frame_000000.jpg", "status": "reviewed", "keypoints": keypoints},
                        {"frame": "frame_000001.jpg", "status": "reviewed", "keypoints": keypoints},
                    ]
                },
                "frames": {
                    "frame_dir": str(labels_dir),
                    "source_resolution": [128, 72],
                    "label_coordinate_space": [128, 72],
                },
                "review": {"status": "reviewed", "reviewer": "test"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_train_court_model_v2_cli_mixes_in_on_disk_real_corpus_end_to_end(tmp_path: Path) -> None:
    """End-to-end smoke test for the on-disk real-corpora path (`--real-root` +
    `--real-weight`/`--synthetic-weight` mixed sampling): must not crash, must compute a
    real-holdout eval block with seg-IoU correctly reported as unavailable (real rows carry no
    segmentation ground truth), must never touch the protected eval_clips (this fixture uses an
    unrelated synthetic external-corpus clip name, never a protected clip id), and -- the fixture
    has exactly 2 rows, `--real-val-samples 1` -- the held-out row must be EXCLUDED from the
    training pool (real_train_row_count == 1, not 2): a finite on-disk corpus must never let the
    same row serve as both a training example and a holdout eval example."""

    external_root = tmp_path / "external_corpus"
    _build_tiny_external_corpus(external_root)
    out_dir = tmp_path / "cal_model_v2_real_mix"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "2",
            "--batch-size",
            "4",
            "--image-width",
            "160",
            "--image-height",
            "96",
            "--val-samples",
            "4",
            "--synthetic-fallback",
            "--device",
            "cpu",
            "--real-root",
            str(external_root),
            "--real-weight",
            "1.0",
            "--synthetic-weight",
            "1.0",
            "--real-batch-size",
            "1",
            "--real-val-samples",
            "1",
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0

    summary = json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    assert summary["training"]["real_train_row_count"] == 1
    assert summary["training"]["real_fraction"] == pytest.approx(0.5)
    assert summary["real_holdout_count"] == 1
    assert summary["real_holdout_after"] is not None
    assert summary["real_holdout_after"]["keypoint_count"] == 15
    # Real rows carry no segmentation ground truth -- seg-IoU must be reported as unavailable
    # (None), never a fabricated number.
    assert summary["real_holdout_after"]["seg_mean_iou"] is None
