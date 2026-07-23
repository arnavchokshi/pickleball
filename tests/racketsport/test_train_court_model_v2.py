from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


def test_synthetic_training_dataloader_worker0_matches_single_process_materialization() -> None:
    loader = train_court_model_v2._make_synthetic_training_dataloader(
        config={"image_size": [96, 64]},
        base_seed=42,
        batch_size=3,
        steps_per_epoch=1,
        global_step_offset=0,
        model_width=96,
        model_height=64,
        sigma_px=1.5,
        force_fallback=True,
        synthetic_workers=0,
        torch=torch,
    )
    [batch] = list(train_court_model_v2._iter_ordered_synthetic_batches(loader, expected_steps=1))
    direct = train_court_model_v2.materialize_synthetic_batch(
        config={"image_size": [96, 64]},
        seed=43,
        count=3,
        model_width=96,
        model_height=64,
        sigma_px=1.5,
        force_fallback=True,
        torch=torch,
    )

    for key in ("image", "heatmaps", "heatmap_mask", "vis_target", "seg_target", "seg_mask"):
        assert torch.equal(batch[key], direct[key])
    assert batch["loader_meta"]["global_step_index"] == 0
    assert batch["loader_meta"]["synthetic_seed"] == 43


def test_synthetic_training_dataloader_multi_worker_has_stamped_step_mapping() -> None:
    loader = train_court_model_v2._make_synthetic_training_dataloader(
        config={"image_size": [96, 64]},
        base_seed=42,
        batch_size=2,
        steps_per_epoch=3,
        global_step_offset=0,
        model_width=96,
        model_height=64,
        sigma_px=1.5,
        force_fallback=True,
        synthetic_workers=2,
        torch=torch,
    )

    batches = list(train_court_model_v2._iter_ordered_synthetic_batches(loader, expected_steps=3))
    assert [batch["loader_meta"]["global_step_index"] for batch in batches] == [0, 1, 2]
    assert [batch["loader_meta"]["synthetic_seed"] for batch in batches] == [43, 44, 45]
    assert {batch["loader_meta"]["worker_id"] for batch in batches} == {0, 1}
    for batch in batches:
        direct = train_court_model_v2.materialize_synthetic_batch(
            config={"image_size": [96, 64]},
            seed=batch["loader_meta"]["synthetic_seed"],
            count=2,
            model_width=96,
            model_height=64,
            sigma_px=1.5,
            force_fallback=True,
            torch=torch,
        )
        assert torch.equal(batch["image"], direct["image"])


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


def _real_row_fixture(tmp_path: Path, *, null_names: set[str] | None = None) -> dict[str, object]:
    import cv2

    image_path = tmp_path / "real_row.jpg"
    cv2.imwrite(str(image_path), np.full((72, 128, 3), 80, dtype=np.uint8))
    null_names = null_names or set()
    keypoints = {
        name: None if name in null_names else [float(8 + index * 6), float(6 + index * 3)]
        for index, name in enumerate(KEYPOINT_NAMES)
    }
    return {
        "image_path": str(image_path),
        "video_path": None,
        "frame_index": 0,
        "source_video_size": [128, 72],
        "keypoints": keypoints,
    }


def _loss_args() -> SimpleNamespace:
    return SimpleNamespace(
        vis_loss_weight=0.2,
        seg_loss_weight=1.0,
        geometric_loss_weight=0.0,
        geometric_colinearity_weight=1.0,
        geometric_homography_weight=1.0,
    )


class _FixedCourtOutputs(torch.nn.Module):
    def __init__(self, heatmap_logits: torch.Tensor, vis_logits: torch.Tensor) -> None:
        super().__init__()
        self.heatmap_logits = torch.nn.Parameter(heatmap_logits)
        self.vis_logits = torch.nn.Parameter(vis_logits)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = image.shape[0]
        return {
            "keypoint_heatmaps": self.heatmap_logits.expand(batch, -1, -1, -1),
            "keypoint_vis_logits": self.vis_logits.expand(batch, -1),
            "line_family_logits": torch.zeros(
                batch,
                5,
                self.heatmap_logits.shape[-2],
                self.heatmap_logits.shape[-1],
                dtype=image.dtype,
                device=image.device,
            ),
        }


def test_null_masked_real_row_has_zero_loss_and_metric_contribution_under_garbage_predictions(
    tmp_path: Path,
) -> None:
    null_name = KEYPOINT_NAMES[-1]
    arrays = train_court_model_v2.real_row_to_sample_arrays(
        _real_row_fixture(tmp_path, null_names={null_name}),
        model_width=128,
        model_height=72,
        sigma_px=1.5,
    )
    batch = train_court_model_v2._stack_arrays([arrays], torch)
    null_index = KEYPOINT_NAMES.index(null_name)
    assert arrays["heatmap_mask"][null_index] == 0.0
    assert arrays["vis_mask"][null_index] == 0.0

    clean_heatmaps = torch.zeros((1, 15, 18, 32))
    clean_visibility = torch.zeros((1, 15))
    clean_model = _FixedCourtOutputs(clean_heatmaps.clone(), clean_visibility.clone())
    clean_loss, _ = train_court_model_v2._compute_batch_loss(
        clean_model, batch, device=torch.device("cpu"), args=_loss_args(), torch=torch, skip_geometric=True
    )

    garbage_heatmaps = clean_heatmaps.clone()
    garbage_heatmaps[0, null_index, 0, 0] = 1000.0
    garbage_heatmaps[0, null_index, -1, -1] = -1000.0
    garbage_visibility = clean_visibility.clone()
    garbage_visibility[0, null_index] = 1000.0
    garbage_model = _FixedCourtOutputs(garbage_heatmaps, garbage_visibility)
    garbage_loss, _ = train_court_model_v2._compute_batch_loss(
        garbage_model, batch, device=torch.device("cpu"), args=_loss_args(), torch=torch, skip_geometric=True
    )
    garbage_loss.backward()

    assert torch.equal(garbage_loss.detach(), clean_loss.detach())
    assert torch.count_nonzero(garbage_model.heatmap_logits.grad[:, null_index]).item() == 0
    assert torch.count_nonzero(garbage_model.vis_logits.grad[:, null_index]).item() == 0

    metrics = train_court_model_v2.evaluate_on_batch(
        garbage_model,
        batch,
        device=torch.device("cpu"),
        heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE,
        torch=torch,
    )
    assert metrics["keypoint_count"] == 14


def test_full_15_real_row_total_loss_is_identical_to_pre_mask_formula(tmp_path: Path) -> None:
    arrays = train_court_model_v2.real_row_to_sample_arrays(
        _real_row_fixture(tmp_path), model_width=128, model_height=72, sigma_px=1.5
    )
    batch = train_court_model_v2._stack_arrays([arrays], torch)
    torch.manual_seed(1709)
    heatmap_logits = torch.randn((1, 15, 18, 32))
    vis_logits = torch.randn((1, 15))
    model = _FixedCourtOutputs(heatmap_logits.clone(), vis_logits.clone())

    actual, _ = train_court_model_v2._compute_batch_loss(
        model, batch, device=torch.device("cpu"), args=_loss_args(), torch=torch, skip_geometric=True
    )
    legacy_heatmap = train_court_model_v2.court_keypoint_heatmap_loss(
        heatmap_logits,
        batch["heatmaps"],
        batch["heatmap_mask"].unsqueeze(-1).unsqueeze(-1).expand_as(batch["heatmaps"]),
    )
    legacy = legacy_heatmap + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
        vis_logits, batch["vis_target"]
    ) + 1.0 * heatmap_logits.new_zeros(())

    assert bool((batch["vis_mask"] == 1).all())
    assert torch.equal(actual, legacy)


def test_real_batch_can_receive_geometry_regularization(tmp_path: Path) -> None:
    arrays = train_court_model_v2.real_row_to_sample_arrays(
        _real_row_fixture(tmp_path), model_width=128, model_height=72, sigma_px=1.5
    )
    batch = train_court_model_v2._stack_arrays([arrays], torch)
    model = _FixedCourtOutputs(
        torch.randn((1, 15, 18, 32)),
        torch.zeros((1, 15)),
    )
    args = _loss_args()
    args.geometric_loss_weight = 0.05

    loss, components = train_court_model_v2._compute_batch_loss(
        model,
        batch,
        device=torch.device("cpu"),
        args=args,
        torch=torch,
    )

    assert torch.isfinite(loss)
    assert "geometric_loss" in components
    assert components["geometric_loss"] >= 0.0


def test_real_photometric_aug_changes_only_pixels_and_is_seeded(tmp_path: Path) -> None:
    import random

    row = _real_row_fixture(tmp_path, null_names={KEYPOINT_NAMES[-1]})
    plain = train_court_model_v2.real_row_to_sample_arrays(
        row, model_width=128, model_height=72, sigma_px=1.5
    )
    augmented_a = train_court_model_v2.real_row_to_sample_arrays(
        row,
        model_width=128,
        model_height=72,
        sigma_px=1.5,
        photometric_aug=True,
        rng=random.Random(7),
    )
    augmented_b = train_court_model_v2.real_row_to_sample_arrays(
        row,
        model_width=128,
        model_height=72,
        sigma_px=1.5,
        photometric_aug=True,
        rng=random.Random(7),
    )
    assert not np.array_equal(plain["image"], augmented_a["image"])
    assert np.array_equal(augmented_a["image"], augmented_b["image"])
    for key in ("heatmaps", "heatmap_mask", "vis_target", "vis_mask", "keypoint_supervision_mask"):
        assert np.array_equal(plain[key], augmented_a[key])


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda row: row["keypoints"].update({"not_canonical": [1.0, 2.0]}), "unexpected canonical"),
        (lambda row: row.update({"keypoints": {name: None for name in KEYPOINT_NAMES}}), "at least one"),
        (lambda row: row["keypoints"].update({KEYPOINT_NAMES[0]: [float("nan"), 2.0]}), "finite"),
    ],
)
def test_real_row_to_sample_arrays_fails_loudly_on_malformed_rows(
    tmp_path: Path, mutator: object, match: str
) -> None:
    row = _real_row_fixture(tmp_path)
    mutator(row)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        train_court_model_v2.real_row_to_sample_arrays(
            row, model_width=128, model_height=72, sigma_px=1.5
        )


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
            "--synthetic-workers",
            "0",
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


@pytest.mark.parametrize("synthetic_workers", [0, 1])
def test_train_court_model_v2_cli_tiny_cpu_smoke_with_worker_modes(
    tmp_path: Path, synthetic_workers: int
) -> None:
    out_dir = tmp_path / f"cal_model_v2_workers_{synthetic_workers}"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "2",
            "--image-width",
            "96",
            "--image-height",
            "64",
            "--val-samples",
            "2",
            "--synthetic-fallback",
            "--device",
            "cpu",
            "--synthetic-workers",
            str(synthetic_workers),
            "--eval-every",
            "1",
            "--geometric-loss-weight",
            "0.0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0
    summary = json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    assert summary["training"]["synthetic_workers"] == synthetic_workers
    assert summary["training"]["synthetic_loader"] == "SyntheticCourtIterableDataset+DataLoader"
    assert len(summary["history"]) == 1
    assert summary["history"][0]["synthetic_worker_count"] >= 1


def test_train_court_model_v2_periodic_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    def _args(out_dir: Path, *, epochs: int, resume: Path | None = None) -> list[str]:
        args = [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "2",
            "--image-width",
            "96",
            "--image-height",
            "64",
            "--val-samples",
            "2",
            "--synthetic-fallback",
            "--device",
            "cpu",
            "--synthetic-workers",
            "0",
            "--eval-every",
            "1",
            "--keep-last-checkpoints",
            "1",
            "--geometric-loss-weight",
            "0.0",
        ]
        if resume is not None:
            args.extend(["--resume", str(resume)])
        return args

    def _run(out_dir: Path, *, epochs: int, resume: Path | None = None) -> dict[str, object]:
        args = _args(out_dir, epochs=epochs, resume=resume)
        completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0
        return json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))

    def _run_until_first_epoch_checkpoint_then_kill(out_dir: Path) -> None:
        process = subprocess.Popen(
            _args(out_dir, epochs=2),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stderr is not None
        deadline = time.time() + 180
        saw_epoch_one = False
        stderr_lines: list[str] = []
        while time.time() < deadline:
            line = process.stderr.readline()
            if line:
                stderr_lines.append(line)
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("epoch") == 1:
                    saw_epoch_one = True
                    break
            if process.poll() is not None:
                break
        if not saw_epoch_one:
            process.kill()
            stdout, stderr = process.communicate(timeout=30)
            raise AssertionError(f"trainer did not reach epoch-1 checkpoint before exit; stdout={stdout}; stderr={''.join(stderr_lines)}{stderr}")
        process.kill()
        process.communicate(timeout=30)

    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    _run(uninterrupted_dir, epochs=2)
    _run_until_first_epoch_checkpoint_then_kill(resumed_dir)

    first_epoch_checkpoint = resumed_dir / "court_model_v2_epoch_0001.pt"
    assert first_epoch_checkpoint.is_file()
    resumed_summary = _run(resumed_dir, epochs=2, resume=first_epoch_checkpoint)

    retained_epoch_checkpoints = sorted(path.name for path in resumed_dir.glob("court_model_v2_epoch_*.pt"))
    assert retained_epoch_checkpoints == ["court_model_v2_epoch_0002.pt"]
    assert [row["epoch"] for row in resumed_summary["history"]] == [2]

    uninterrupted_state = torch.load(
        uninterrupted_dir / "court_model_v2.pt", map_location="cpu", weights_only=False
    )["model"]
    resumed_state = torch.load(resumed_dir / "court_model_v2.pt", map_location="cpu", weights_only=False)["model"]
    assert uninterrupted_state.keys() == resumed_state.keys()
    for key in uninterrupted_state:
        assert torch.allclose(uninterrupted_state[key], resumed_state[key], atol=1e-7, rtol=1e-7), key


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
        "--synthetic-workers",
        "0",
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


def test_encoder_weights_path_missing_fails_loudly_without_random_init(tmp_path: Path) -> None:
    out_dir = tmp_path / "cal_model_v2_missing_encoder"
    missing = tmp_path / "missing_resnet34.pth"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "2",
            "--image-width",
            "96",
            "--image-height",
            "64",
            "--val-samples",
            "2",
            "--synthetic-fallback",
            "--device",
            "cpu",
            "--encoder-weights-path",
            str(missing),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 2
    assert "encoder weights were requested" in completed.stderr
    assert "does not exist" in completed.stderr
    assert not (out_dir / "court_model_v2.pt").exists()


def test_encoder_weights_imagenet_requires_local_torchvision_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty_torch_home"))
    with pytest.raises(FileNotFoundError, match="does not download"):
        train_court_model_v2._resolve_encoder_weights_path("imagenet")


def test_encoder_weights_path_rejects_wrong_resnet34_shape(tmp_path: Path) -> None:
    bogus_checkpoint = tmp_path / "not_resnet34.pth"
    torch.save({"conv1.weight": torch.zeros(1)}, bogus_checkpoint)

    with pytest.raises(ValueError, match="does not match expected torchvision resnet34"):
        train_court_model_v2._resolve_encoder_weights_path(str(bogus_checkpoint))


def test_pinned_imagenet_resnet34_checkpoint_validates_and_initializes_exact_weights() -> None:
    checkpoint_path = Path("models/checkpoints/court_external/torchvision/resnet34-b627a593.pth")
    assert train_court_model_v2._resolve_encoder_weights_path(checkpoint_path) == checkpoint_path
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = train_court_model_v2.make_court_keypoint_heatmap_model(
        15,
        architecture="court_unet_v2",
        encoder_weights_path=checkpoint_path,
    )
    assert torch.equal(model.stem[0].weight.detach(), source["conv1.weight"])
    assert torch.equal(model.stem[1].running_mean.detach(), source["bn1.running_mean"])
    assert int(model.stem[1].num_batches_tracked) == 0


def _build_tiny_external_corpus(root: Path, *, partial: bool = False) -> None:
    import cv2

    clip = root / "some_external_clip"
    clip.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(clip / "source.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (128, 72))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((72, 128, 3), 90, dtype=np.uint8))
    writer.release()

    keypoints = {name: [float(10 + index * 5), float(10 + index * 3)] for index, name in enumerate(KEYPOINT_NAMES)}
    if partial:
        for name in KEYPOINT_NAMES[-3:]:
            keypoints[name] = None
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


def test_direct_cli_five_step_partial_real_plus_synthetic_fallback_round_trip(tmp_path: Path) -> None:
    external_root = tmp_path / "partial_external_corpus"
    _build_tiny_external_corpus(external_root, partial=True)
    out_dir = tmp_path / "five_step_partial_mix"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "5",
            "--batch-size",
            "1",
            "--image-width",
            "96",
            "--image-height",
            "64",
            "--val-samples",
            "1",
            "--synthetic-fallback",
            "--synthetic-workers",
            "0",
            "--device",
            "cpu",
            "--real-root",
            str(external_root),
            "--real-weight",
            "1.0",
            "--synthetic-weight",
            "0.0",
            "--real-batch-size",
            "1",
            "--real-val-samples",
            "1",
            "--geometric-loss-weight",
            "0.0",
            "--real-photometric-aug",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0
    summary = json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    sampling = summary["training"]["real_sampling"]
    assert sampling["batches"] == 5
    assert sampling["null_channels"] == 15
    assert sampling["null_heatmap_supervision_count"] == 0
    assert sampling["null_visibility_supervision_count"] == 0
    assert summary["training"]["real_photometric_aug"] is True
    checkpoint = torch.load(out_dir / "court_model_v2.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["network_architecture"] == "court_unet_v2"


def test_init_from_real_court_model_checkpoint_is_fresh_and_owner_gate_cli_accepts_output(
    tmp_path: Path,
) -> None:
    source_checkpoint = Path("models/checkpoints/court_unet_v2/court_model_v2.pt")
    assert source_checkpoint.is_file()
    fixture_root = tmp_path / "two_frame_fixture"
    _build_tiny_external_corpus(fixture_root, partial=True)
    out_dir = tmp_path / "initialized_run"

    train = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_model_v2.py",
            "--out",
            str(out_dir),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--batch-size",
            "1",
            "--image-width",
            "96",
            "--image-height",
            "64",
            "--val-samples",
            "1",
            "--synthetic-fallback",
            "--synthetic-workers",
            "0",
            "--device",
            "cpu",
            "--geometric-loss-weight",
            "0.0",
            "--real-root",
            str(fixture_root),
            "--real-weight",
            "1.0",
            "--synthetic-weight",
            "0.0",
            "--real-batch-size",
            "1",
            "--real-val-samples",
            "1",
            "--init-from-checkpoint",
            str(source_checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert train.returncode == 0
    summary = json.loads((out_dir / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    initialization = summary["training"]["initialization"]
    assert initialization["mode"] == "model_checkpoint_fresh_optimizer"
    assert initialization["checkpoint"]["path"] == str(source_checkpoint)
    assert initialization["checkpoint"]["optimizer_restored"] is False
    assert initialization["checkpoint"]["start_epoch"] == 0
    produced_checkpoint = out_dir / "court_model_v2.pt"
    assert torch.load(produced_checkpoint, map_location="cpu", weights_only=False)["epoch"] == 1

    gate_out = tmp_path / "owner_gate_fixture_report.json"
    gate = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_keypoint_owner_gate.py",
            "--checkpoint",
            str(produced_checkpoint),
            "--real-root",
            str(fixture_root),
            "--out",
            str(gate_out),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert gate.returncode == 0
    report = json.loads(gate_out.read_text(encoding="utf-8"))
    assert report["all_frame_count"] == 2
    assert report["raw_all"]["keypoint_error_summary"]["count"] == 24
