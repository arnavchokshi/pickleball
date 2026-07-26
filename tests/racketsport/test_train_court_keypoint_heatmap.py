from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import scripts.racketsport.train_court_keypoint_heatmap as trainer_module

from scripts.racketsport.train_court_keypoint_heatmap import (
    aux_real_sample_training_tensors,
    aux_synthetic_stream_config,
    aux_synthetic_sample_training_tensors,
    choose_torch_device_name,
    court_keypoint_label_rows,
    court_keypoint_heatmap_loss,
    court_keypoint_probabilities,
    curriculum_synthetic_fraction,
    evaluate_checkpoint_against_real_labels,
    heatmaps_for_points,
    keypoint_checkpoint_metadata,
    legacy_synthetic_training_sample,
    legacy_training_image_tensor,
    load_real_corner_labels,
    load_real_court_keypoint_labels,
    make_court_keypoint_heatmap_model,
    predict_source_keypoints,
    run_training,
    resolve_keypoint_training_contract,
    sample_curriculum_real_batch,
    training_cli_summary,
)
from threed.racketsport.court_keypoint_net import (
    ALL_PICKLEBALL_KEYPOINTS,
    AUX_PICKLEBALL_KEYPOINTS,
    COURT_UNET_V2_AUX_ARCHITECTURE,
    COURT_UNET_V2_HEATMAP_STRIDE,
    PICKLEBALL_KEYPOINTS,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMps(mps_available)


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)


def _production_preprocessor_tensor(
    image_bgr: np.ndarray,
    *,
    model_size: tuple[int, int],
    keypoint_names: list[str],
    torch: object,
) -> object:
    from threed.racketsport.court_model_infer import _infer_court_model_with_loaded_model

    class _CaptureModel(torch.nn.Module):
        heatmap_stride = COURT_UNET_V2_HEATMAP_STRIDE

        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, tensor):
            self.seen = tensor.detach().cpu().clone()
            batch = tensor.shape[0]
            head_height = (model_size[1] + COURT_UNET_V2_HEATMAP_STRIDE - 1) // COURT_UNET_V2_HEATMAP_STRIDE
            head_width = (model_size[0] + COURT_UNET_V2_HEATMAP_STRIDE - 1) // COURT_UNET_V2_HEATMAP_STRIDE
            return {
                "keypoint_heatmaps": torch.zeros(
                    (batch, len(keypoint_names), head_height, head_width), dtype=tensor.dtype
                ),
                "keypoint_vis_logits": torch.zeros((batch, len(keypoint_names)), dtype=tensor.dtype),
                "line_family_logits": torch.zeros((batch, 5, head_height, head_width), dtype=tensor.dtype),
            }

    model = _CaptureModel()
    _infer_court_model_with_loaded_model(
        image_bgr,
        model=model,
        keypoint_names=keypoint_names,
        model_size=model_size,
        device="cpu",
    )
    assert model.seen is not None
    return model.seen


def test_choose_torch_device_name_honors_mps_when_available() -> None:
    assert choose_torch_device_name("mps", _FakeTorch(cuda_available=False, mps_available=True)) == "mps"
    assert choose_torch_device_name("mps", _FakeTorch(cuda_available=False, mps_available=False)) == "cpu"
    assert choose_torch_device_name("cuda", _FakeTorch(cuda_available=True, mps_available=True)) == "cuda"
    assert choose_torch_device_name("cuda", _FakeTorch(cuda_available=False, mps_available=True)) == "cpu"


def test_aux_training_contract_is_canonical_first_and_line_mode_fails_closed() -> None:
    legacy = resolve_keypoint_training_contract(aux_keypoints=False, line_segmentation=False)
    assert legacy["network_architecture"] == "encoder_decoder_v1"
    assert legacy["heatmap_stride"] == 1
    assert legacy["keypoint_names"] == tuple(point.name for point in PICKLEBALL_KEYPOINTS)

    aux = resolve_keypoint_training_contract(aux_keypoints=True, line_segmentation=False)
    assert aux["network_architecture"] == COURT_UNET_V2_AUX_ARCHITECTURE
    assert aux["heatmap_stride"] == COURT_UNET_V2_HEATMAP_STRIDE
    assert len(aux["keypoint_names"]) == 33
    assert aux["keypoint_names"][: len(PICKLEBALL_KEYPOINTS)] == tuple(
        point.name for point in PICKLEBALL_KEYPOINTS
    )
    assert aux["keypoint_names"][len(PICKLEBALL_KEYPOINTS) :] == tuple(
        point.name for point in AUX_PICKLEBALL_KEYPOINTS
    )
    assert aux["keypoint_names"] == tuple(point.name for point in ALL_PICKLEBALL_KEYPOINTS)
    assert keypoint_checkpoint_metadata(aux) == {
        "aux_keypoints": True,
        "canonical_keypoint_count": 15,
        "aux_keypoint_count": 18,
        "heatmap_stride": 4,
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_keypoint_training_contract(aux_keypoints=True, line_segmentation=True)


def test_aux_real_rows_mask_all_aux_channels_and_aux_gradients_are_zero() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    image_bgr = np.zeros((72, 128, 3), dtype=np.uint8)
    points = {
        point.name: [16.0 + index * 4.0, 18.0 + (index % 3) * 8.0]
        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
    }

    image, target, mask = aux_real_sample_training_tensors(
        image_bgr,
        points,
        source_width=128.0,
        source_height=72.0,
        width=64,
        height=36,
        sigma=1.5,
        cv2=cv2,
        np=np,
        torch=torch,
    )

    assert image.shape == (3, 36, 64)
    assert target.shape == (33, 9, 16)
    assert mask.shape == target.shape
    assert torch.all(mask[:15] == 1)
    assert torch.count_nonzero(mask[15:]).item() == 0
    assert torch.count_nonzero(target[15:]).item() == 0

    logits = torch.zeros((1, *target.shape), dtype=torch.float32, requires_grad=True)
    loss = court_keypoint_heatmap_loss(logits, target.unsqueeze(0), mask.unsqueeze(0))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :15]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 15:]).item() == 0


def test_aux_training_tensor_is_exactly_production_preprocessor_tensor() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from threed.racketsport.court_synth_stream import iter_synthetic_court_samples

    # Exercise the existing CLI defaults, including ceil-divided 90 / 4 = 23 rows.
    target_size = (160, 90)
    sample = next(
        iter_synthetic_court_samples(
            {
                "count": 1,
                "image_size": list(target_size),
                "scenarios": ["dedicated_indoor"],
                "aux_keypoints": True,
                "paint_texture_randomization": True,
                "aux_partial_visibility": True,
            },
            seed=20260722,
        )
    )

    training_image, _, _ = aux_synthetic_sample_training_tensors(
        sample,
        width=target_size[0],
        height=target_size[1],
        cv2=cv2,
        np=np,
        torch=torch,
    )
    production_image = _production_preprocessor_tensor(
        sample["image_bgr"],
        model_size=target_size,
        keypoint_names=[point.name for point in ALL_PICKLEBALL_KEYPOINTS],
        torch=torch,
    )

    assert sample["keypoints_xy"].shape == (33, 2)
    assert sample["keypoint_heatmaps"].shape == (33, 23, 40)
    assert sample["keypoint_heatmap_mask"].shape == (33, 23, 40)
    assert torch.equal(training_image.unsqueeze(0), production_image)


def test_aux_evaluation_primitive_uses_production_preprocessor(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from PIL import Image
    from threed.racketsport.court_synth_stream import iter_synthetic_court_samples

    source_size = (320, 180)
    model_size = (160, 90)
    sample = next(
        iter_synthetic_court_samples(
            {
                "count": 1,
                "image_size": list(source_size),
                "scenarios": ["dedicated_indoor"],
                "aux_keypoints": True,
                "paint_texture_randomization": True,
                "aux_partial_visibility": True,
            },
            seed=20260722,
        )
    )
    image_rgb = Image.fromarray(cv2.cvtColor(sample["image_bgr"], cv2.COLOR_BGR2RGB))
    monkeypatch.setattr(
        trainer_module,
        "load_label_image",
        lambda row, *, cv2, image_module: image_rgb.copy(),
    )

    class _CaptureAuxModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, tensor):
            self.seen = tensor.detach().cpu().clone()
            return {
                "keypoint_heatmaps": torch.zeros(
                    (tensor.shape[0], len(ALL_PICKLEBALL_KEYPOINTS), 23, 40),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            }

    model = _CaptureAuxModel()
    keypoint_names = [point.name for point in ALL_PICKLEBALL_KEYPOINTS]
    predict_source_keypoints(
        model,
        {
            "source_video_size": list(source_size),
            "keypoints": {keypoint_names[0]: [0.0, 0.0]},
        },
        cv2=cv2,
        np=np,
        torch=torch,
        image_module=Image,
        device="cpu",
        width=model_size[0],
        height=model_size[1],
        keypoint_names=keypoint_names,
        heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE,
        production_preprocessing=True,
    )
    expected = _production_preprocessor_tensor(
        sample["image_bgr"],
        model_size=model_size,
        keypoint_names=keypoint_names,
        torch=torch,
    )

    assert model.seen is not None
    assert torch.equal(model.seen, expected)

    legacy_model = _CaptureAuxModel()
    predict_source_keypoints(
        legacy_model,
        {
            "source_video_size": list(source_size),
            "keypoints": {keypoint_names[0]: [0.0, 0.0]},
        },
        cv2=cv2,
        np=np,
        torch=torch,
        image_module=Image,
        device="cpu",
        width=model_size[0],
        height=model_size[1],
        keypoint_names=keypoint_names,
        heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE,
    )
    legacy_expected = legacy_training_image_tensor(
        image_rgb,
        width=model_size[0],
        height=model_size[1],
        np=np,
        torch=torch,
    ).unsqueeze(0)
    assert legacy_model.seen is not None
    assert torch.equal(legacy_model.seen, legacy_expected)


def test_aux_homography_refinement_preserves_unrefined_aux_predictions(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from PIL import Image

    canonical_name = PICKLEBALL_KEYPOINTS[0].name
    aux_name = AUX_PICKLEBALL_KEYPOINTS[0].name
    source_rgb = np.zeros((90, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        trainer_module,
        "load_label_image",
        lambda row, *, cv2, image_module: Image.fromarray(source_rgb.copy()),
    )

    class _ZeroAuxModel(torch.nn.Module):
        def forward(self, tensor):
            return {
                "keypoint_heatmaps": torch.zeros(
                    (tensor.shape[0], len(ALL_PICKLEBALL_KEYPOINTS), 23, 40),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            }

    monkeypatch.setattr(
        trainer_module,
        "refine_keypoint_xy_with_planar_homography",
        lambda points: {canonical_name: [12.5, 34.5]},
    )
    predicted = predict_source_keypoints(
        _ZeroAuxModel(),
        {
            "source_video_size": [160, 90],
            "keypoints": {canonical_name: [0.0, 0.0], aux_name: [0.0, 0.0]},
        },
        cv2=cv2,
        np=np,
        torch=torch,
        image_module=Image,
        device="cpu",
        width=160,
        height=90,
        keypoint_names=[point.name for point in ALL_PICKLEBALL_KEYPOINTS],
        use_homography_refinement=True,
        heatmap_stride=COURT_UNET_V2_HEATMAP_STRIDE,
        production_preprocessing=True,
    )

    assert predicted[canonical_name] == [12.5, 34.5]
    assert aux_name in predicted
    assert predicted[aux_name] == [0.0, 0.0]


def test_aux_checkpoint_top_level_gate_stays_canonical_with_divergent_aux_error(
    tmp_path: Path, monkeypatch
) -> None:
    keypoint_names = [point.name for point in ALL_PICKLEBALL_KEYPOINTS]
    canonical_name = PICKLEBALL_KEYPOINTS[0].name
    aux_name = AUX_PICKLEBALL_KEYPOINTS[0].name
    row = {
        "clip": "canonical-only",
        "frame_index": 0,
        "label_status": "reviewed",
        "keypoints": {
            canonical_name: [12.0, 34.0],
            aux_name: [50.0, 60.0],
        },
    }
    payload = {"network_architecture": COURT_UNET_V2_AUX_ARCHITECTURE}
    monkeypatch.setattr(
        trainer_module,
        "load_court_keypoint_checkpoint",
        lambda checkpoint_path, *, device: payload,
    )
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_checkpoint",
        lambda checkpoint_payload, *, device: (object(), keypoint_names, 160, 90),
    )

    preprocessing_flags = []

    def fake_predict(model, selected_row, **kwargs):
        preprocessing_flags.append(kwargs["production_preprocessing"])
        return {
            canonical_name: [12.0, 34.0],
            aux_name: [100.0, 60.0],
        }

    def fake_aggregate(model, rows, **kwargs):
        preprocessing_flags.append(kwargs["production_preprocessing"])
        return {
            "canonical-only": {
                canonical_name: [12.0, 34.0],
                aux_name: [100.0, 60.0],
            }
        }

    monkeypatch.setattr(trainer_module, "predict_source_keypoints", fake_predict)
    monkeypatch.setattr(trainer_module, "aggregate_static_camera_predictions", fake_aggregate)

    report = evaluate_checkpoint_against_real_labels(tmp_path / "dormant_aux_checkpoint.pt", [row])

    assert preprocessing_flags == [True, True]
    for mode in ("raw_independent", "raw_all", "aggregated_independent", "aggregated_all"):
        assert report[mode]["keypoint_error_summary"]["count"] == 1
        assert report[mode]["pck_at_5px"] == 1.0
        assert report[mode]["canonical_keypoints"]["keypoint_count"] == 1
        assert report[mode]["canonical_keypoints"]["pck_at_5px"] == 1.0
        assert report[mode]["aux_keypoints"]["keypoint_count"] == 1
        assert report[mode]["aux_keypoints"]["pck_at_5px"] == 0.0
        assert report[mode]["combined_keypoints"]["keypoint_count"] == 2
        assert report[mode]["combined_keypoints"]["pck_at_5px"] == 0.5


def test_aux_synthetic_stream_config_propagates_trainer_sigma() -> None:
    from threed.racketsport.court_synth_stream import iter_synthetic_court_samples

    narrow_config = aux_synthetic_stream_config(width=160, height=90, sigma=0.75)
    wide_config = aux_synthetic_stream_config(width=160, height=90, sigma=2.75)
    assert narrow_config["heatmap_sigma_px"] == 0.75
    assert wide_config["heatmap_sigma_px"] == 2.75

    narrow = next(iter_synthetic_court_samples(narrow_config, seed=20260722))
    wide = next(iter_synthetic_court_samples(wide_config, seed=20260722))
    assert np.array_equal(narrow["image_bgr"], wide["image_bgr"])
    assert np.array_equal(narrow["keypoints_xy"], wide["keypoints_xy"])
    assert np.array_equal(narrow["keypoint_heatmap_mask"], wide["keypoint_heatmap_mask"])
    assert not np.array_equal(narrow["keypoint_heatmaps"], wide["keypoint_heatmaps"])
    assert float(wide["keypoint_heatmaps"].sum()) > float(narrow["keypoint_heatmaps"].sum())


def test_run_training_aux_zero_epoch_default_size_smoke(tmp_path: Path, monkeypatch) -> None:
    actual_iter = trainer_module.iter_synthetic_court_samples
    captured_configs = []

    def recording_iter(config, *, seed):
        captured_configs.append(dict(config))
        return actual_iter(config, seed=seed)

    monkeypatch.setattr(trainer_module, "iter_synthetic_court_samples", recording_iter)
    summary = run_training(
        Namespace(
            real_root=None,
            out=tmp_path / "aux_zero_epoch",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=0,
            batch_size=1,
            image_width=160,
            image_height=90,
            sigma=1.875,
            learning_rate=1e-3,
            real_finetune_start_epoch=0,
            eval_every=1,
            seed=20260722,
            device="cpu",
            skip_holdout_artifacts=True,
            static_camera_aggregate=False,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
            aux_keypoints=True,
        )
    )

    assert captured_configs == [
        {
            "image_size": [160, 90],
            "aux_keypoints": True,
            "paint_texture_randomization": True,
            "aux_partial_visibility": True,
            "heatmap_sigma_px": 1.875,
        }
    ]
    assert summary["history"] == []
    assert summary["gate"]["value"] is None
    assert summary["gate"]["passed"] is False
    assert summary["architecture"]["network_architecture"] == COURT_UNET_V2_AUX_ARCHITECTURE
    assert summary["architecture"]["canonical_keypoint_count"] == 15
    assert summary["architecture"]["aux_keypoint_count"] == 18
    assert summary["after"]["synthetic_canonical_keypoints"]["keypoint_count"] > 0
    assert summary["after"]["synthetic_aux_keypoints"]["keypoint_count"] > 0

    checkpoint = trainer_module.load_court_keypoint_checkpoint(Path(summary["checkpoint"]))
    assert checkpoint["network_architecture"] == COURT_UNET_V2_AUX_ARCHITECTURE
    assert checkpoint["aux_keypoints"] is True
    assert checkpoint["canonical_keypoint_count"] == 15
    assert checkpoint["aux_keypoint_count"] == 18
    assert checkpoint["heatmap_stride"] == 4
    assert checkpoint["image_size"] == [160, 90]
    assert len(checkpoint["keypoint_names"]) == 33


def test_flag_off_pil_to_production_resize_mapping_is_stamped() -> None:
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from PIL import Image
    from threed.racketsport.court_synth_stream import iter_synthetic_court_samples

    sample = next(
        iter_synthetic_court_samples(
            {
                "count": 1,
                "scenarios": ["dedicated_indoor"],
                "image_size": [320, 180],
                "focal_px_range": [250, 1000],
            },
            seed=20260722,
        )
    )
    source = sample["image_bgr"]
    image_rgb = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    legacy = legacy_training_image_tensor(
        image_rgb, width=160, height=90, np=np, torch=torch
    ).unsqueeze(0)
    production = _production_preprocessor_tensor(
        source,
        model_size=(160, 90),
        keypoint_names=[point.name for point in PICKLEBALL_KEYPOINTS],
        torch=torch,
    )

    def sha256(value) -> str:
        return hashlib.sha256(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()).hexdigest()

    assert hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest() == (
        "53ff4c751eae1525d68535643a9856d19967f749e2470824679ea866f7a79d71"
    )
    assert sha256(legacy) == "22c2fec39882bf11cc1bda3c381c6ed25dedca3ccbc84a09eab7f390ebd9a613"
    assert sha256(production) == "c15a60047485b1fe1b695a1f47fc92312e8c136afa4497ddac72356dfd2b3940"
    difference = (legacy - production).abs()
    assert torch.count_nonzero(difference).item() == 22648
    assert difference.max().item() == pytest.approx(0.16862744092941284)
    assert difference.mean().item() == pytest.approx(0.005633624270558357)


def test_flag_off_legacy_synthetic_trainer_tensors_match_pre_a2_golden() -> None:
    torch = pytest.importorskip("torch")
    from PIL import Image, ImageDraw

    image, target, mask = legacy_synthetic_training_sample(
        width=160,
        height=90,
        sigma=2.5,
        line_width=3,
        use_line_segmentation=False,
        rng=random.Random(20260722),
        np=np,
        torch=torch,
        image_module=Image,
        image_draw_module=ImageDraw,
    )

    def sha256(value) -> str:
        return hashlib.sha256(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()).hexdigest()

    assert image.shape == (3, 90, 160)
    assert target.shape == (15, 90, 160)
    assert mask.shape == (15, 90, 160)
    assert sha256(image) == "eeca7cd464a35aca8731b7c3bb0fdf46936422bcd50e3a17b0ab49a2f6e2a6d3"
    assert sha256(target) == "eeb2c27718e06df47345678db9c9f975b76d6efc8445e37e2c1c8cb9d51f8aab"
    assert sha256(mask) == "94cf6d77b76fc11ec1934a8abe22c97aba76ed51a9c348d36258583d610c1e99"


def test_aux_keypoints_cli_flag_defaults_off_without_running_training(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run_training(args):
        captured["args"] = args
        return {}

    monkeypatch.setattr(trainer_module, "run_training", fake_run_training)
    monkeypatch.setattr(trainer_module, "training_cli_summary", lambda summary: summary)

    assert trainer_module.main(["--out", str(tmp_path / "unused")]) == 0
    assert captured["args"].aux_keypoints is False


def test_court_keypoint_heatmap_loss_prioritizes_labeled_peaks() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    target[0, 0, 4, 4] = 1.0
    mask = torch.ones_like(target)

    peak_missed = torch.full_like(target, -6.0)
    background_false_positive = torch.full_like(target, -6.0)
    background_false_positive[0, 0, 4, 4] = 6.0
    background_false_positive[0, 0, 0, 0] = 6.0

    missed_loss = court_keypoint_heatmap_loss(peak_missed, target, mask)
    false_positive_loss = court_keypoint_heatmap_loss(background_false_positive, target, mask)

    assert missed_loss.item() > false_positive_loss.item() * 4.0

    probabilities = court_keypoint_probabilities(torch.tensor([[[[-6.0, 0.0, 6.0]]]]))
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0, 0, 0].tolist() == pytest.approx([0.00000612898, 0.002472608, 0.9975212])


def test_full_15_reviewed_loss_tensor_is_byte_identical_to_legacy_path() -> None:
    """COURT-LOADER-1 default-path proof: a full-15 mask uses the exact pre-change formula."""
    torch = pytest.importorskip("torch")
    fixture_row = court_keypoint_label_rows(
        _reviewed_court_keypoint_label_payload(
            source_resolution=[64, 36],
            label_coordinate_space=[64, 36],
        )
    )[0]
    target_array, mask_array = heatmaps_for_points(
        fixture_row["keypoints"],
        [point.name for point in PICKLEBALL_KEYPOINTS],
        64,
        36,
        sigma=1.5,
    )
    torch.manual_seed(1709)
    target = torch.from_numpy(target_array).unsqueeze(0)
    mask = torch.from_numpy(mask_array).unsqueeze(0)
    logits = torch.randn_like(target)

    actual = court_keypoint_heatmap_loss(logits, target, mask)

    target_flat = target.clamp(0.0, 1.0).reshape(1, len(PICKLEBALL_KEYPOINTS), -1)
    logits_flat = logits.reshape(1, len(PICKLEBALL_KEYPOINTS), -1)
    target_distribution = target_flat / target_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    per_channel = -(target_distribution * torch.nn.functional.log_softmax(logits_flat, dim=-1)).sum(dim=-1)
    legacy = per_channel.sum() / torch.tensor(len(PICKLEBALL_KEYPOINTS), dtype=logits.dtype)

    assert fixture_row["label_status"] == "reviewed"
    assert set(fixture_row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert torch.equal(actual, legacy)


def test_masked_unlabeled_channel_has_zero_loss_and_gradient_contribution() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
    target[0, 0, 2, 2] = 1.0
    mask = torch.zeros_like(target)
    mask[:, 0] = 1.0

    logits = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        logits[0, 1, 0, 0] = 1000.0
        logits[0, 1, 4, 4] = -1000.0
    loss_with_garbage = court_keypoint_heatmap_loss(logits, target, mask)
    loss_with_garbage.backward()

    clean_logits = torch.zeros_like(target)
    clean_loss = court_keypoint_heatmap_loss(clean_logits, target, mask)
    assert torch.equal(loss_with_garbage.detach(), clean_loss)
    assert torch.count_nonzero(logits.grad[:, 1]).item() == 0


def test_court_keypoint_heatmap_model_uses_encoder_decoder_context() -> None:
    torch = pytest.importorskip("torch")

    model = make_court_keypoint_heatmap_model(3)
    output = model(torch.zeros((2, 3, 90, 160), dtype=torch.float32))

    assert tuple(output.shape) == (2, 3, 90, 160)
    assert any(isinstance(module, torch.nn.Conv2d) and module.stride == (2, 2) for module in model.modules())
    assert any(isinstance(module, torch.nn.Upsample) for module in model.modules())


def test_load_real_corner_labels_uses_committed_video_frame_when_label_frames_are_absent(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_corners.json",
        {
            "schema_version": 1,
            "annotation": {
                "items": [
                    {
                        "frame": "frame_000002.jpg",
                        "court_corners": {
                            "near_left": [8, 32],
                            "near_right": [56, 32],
                            "far_right": [44, 6],
                            "far_left": [20, 6],
                        },
                    }
                ]
            },
            "frames": {"frame_dir": "runs/label_frames/clip_a", "source_resolution": [128, 72]},
        },
    )

    rows = load_real_corner_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    row = rows[0]
    assert row["clip"] == "clip_a"
    assert row["video_path"] == str(clip_root / "source.mp4")
    assert row["frame_index"] == 2
    assert row["image_path"] is None
    assert set(row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert row["label_coordinate_space"] == [64, 36]
    assert row["source_video_size"] == [128, 72]
    assert row["keypoints"]["near_left_corner"] == pytest.approx([16.0, 64.0])
    assert row["keypoints"]["near_right_corner"] == pytest.approx([112.0, 64.0])


def _reviewed_court_keypoint_label_payload(
    frame: str = "frame_000002.jpg",
    *,
    source_resolution: list[int] | None = None,
    label_coordinate_space: list[int] | None = None,
    item_status: str = "reviewed",
) -> dict:
    return {
        "schema_version": 1,
        "annotation": {
            "items": [
                {
                    "frame": frame,
                    "status": item_status,
                    "keypoints": {
                        point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
                    },
                }
            ]
        },
        "frames": {
            "frame_dir": "runs/label_frames/clip_a",
            "source_resolution": source_resolution or [128, 72],
            "label_coordinate_space": label_coordinate_space or [64, 36],
        },
        "review": {"status": "reviewed", "reviewer": "court-label-review"},
    }


def _partial_external_payload() -> dict:
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000001.jpg",
        source_resolution=[64, 36],
        label_coordinate_space=[64, 36],
        item_status="reviewed_external_dataset",
    )
    for name in ("net_left_sideline", "net_center", "net_right_sideline"):
        payload["annotation"]["items"][0]["keypoints"][name] = None
    return payload


def _write_mixed_partial_training_fixture(tmp_path: Path) -> Path:
    cv2 = __import__("cv2")
    real_root = tmp_path / "real_corpus"
    clip_id = "0tmdeghtfvjx"
    clip_root = real_root / clip_id
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.write(np.full((36, 64, 3), 40, dtype=np.uint8))
    writer.release()

    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg",
        source_resolution=[64, 36],
        label_coordinate_space=[64, 36],
    )
    partial_item = _partial_external_payload()["annotation"]["items"][0]
    payload["annotation"]["items"].append(partial_item)
    payload["clip"] = clip_id
    payload["status"] = trainer_module.OWNER_APPROVED_STATUS
    payload["training_eligibility"] = {
        "queued": True,
        "owner_adjudication": {
            "path": trainer_module.OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
            "sha256": trainer_module.OWNER_ELIGIBILITY_ACT_SHA256,
            "video_id": clip_id,
            "decision": "APPROVE",
        },
    }
    for item in payload["annotation"]["items"]:
        item["pseudo_label_status"] = trainer_module.OWNER_APPROVED_STATUS
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)
    return real_root


def test_load_real_court_keypoint_labels_requires_reviewed_full_15_point_labels(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_corners.json",
        {
            "schema_version": 1,
            "annotation": {
                "items": [
                    {
                        "frame": "frame_000002.jpg",
                        "court_corners": {
                            "near_left": [8, 32],
                            "near_right": [56, 32],
                            "far_right": [44, 6],
                            "far_left": [20, 6],
                        },
                    }
                ]
            },
            "frames": {"frame_dir": "runs/label_frames/clip_a", "source_resolution": [128, 72]},
        },
    )

    with pytest.raises(ValueError, match="reviewed 15-keypoint court labels"):
        load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")


def test_load_real_court_keypoint_labels_reads_reviewed_full_15_point_labels(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(clip_root / "labels" / "court_keypoints.json", _reviewed_court_keypoint_label_payload())

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    row = rows[0]
    assert row["clip"] == "clip_a"
    assert row["video_path"] == str(clip_root / "source.mp4")
    assert row["frame_index"] == 2
    assert row["image_path"] is None
    assert set(row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert row["label_coordinate_space"] == [64, 36]
    assert row["source_video_size"] == [128, 72]
    assert row["keypoints"]["near_left_corner"] == pytest.approx([20.0, 10.0])
    assert row["keypoints"]["far_nvz_right"] == pytest.approx([104.0, 66.0])
    assert row["label_status"] == "reviewed"


def test_load_real_court_keypoint_labels_preserves_absolute_and_repo_relative_frame_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    real_root = tmp_path / "real_corpus"
    absolute_frames = tmp_path / "absolute_frames"
    repo_relative_frames = Path("repo_relative_frames")
    absolute_frames.mkdir()
    repo_relative_frames.mkdir()
    (absolute_frames / "frame_000002.jpg").write_bytes(b"absolute")
    (repo_relative_frames / "frame_000002.jpg").write_bytes(b"repo-relative")

    absolute_payload = _reviewed_court_keypoint_label_payload()
    absolute_payload["frames"].update(
        {"frame_dir": str(absolute_frames), "path_base": "repo_root"}
    )
    repo_relative_payload = _reviewed_court_keypoint_label_payload()
    repo_relative_payload["frames"].update(
        {"frame_dir": repo_relative_frames.as_posix(), "path_base": "repo_root"}
    )
    _write_json(real_root / "absolute_clip" / "labels" / "court_keypoints.json", absolute_payload)
    _write_json(
        real_root / "repo_relative_clip" / "labels" / "court_keypoints.json",
        repo_relative_payload,
    )

    rows = {row["clip"]: row for row in load_real_court_keypoint_labels(real_root)}

    assert Path(rows["absolute_clip"]["image_path"]).resolve() == (
        absolute_frames / "frame_000002.jpg"
    ).resolve()
    assert Path(rows["repo_relative_clip"]["image_path"]).resolve() == (
        repo_relative_frames / "frame_000002.jpg"
    ).resolve()


def test_load_real_court_keypoint_labels_reads_all_reviewed_items(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    payload = _reviewed_court_keypoint_label_payload("frame_000001.jpg")
    second_item = {
        "frame": "frame_000002.jpg",
        # Owner-approved static-camera copy of the independent review above -- must stay
        # distinct from "reviewed" all the way through to the loaded row's label_status.
        "status": "reviewed_static_camera_copy",
        "keypoints": {
            point.name: [float(index * 5 + 20), float(index * 7 + 30)]
            for index, point in enumerate(PICKLEBALL_KEYPOINTS)
        },
    }
    payload["annotation"]["items"].append(second_item)
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 2
    assert [row["frame_index"] for row in rows] == [1, 2]
    assert [row["clip"] for row in rows] == ["clip_a", "clip_a"]
    assert rows[1]["keypoints"]["near_left_corner"] == pytest.approx([40.0, 60.0])
    assert rows[1]["keypoints"]["far_nvz_right"] == pytest.approx([180.0, 256.0])
    assert rows[0]["label_status"] == "reviewed"
    assert rows[1]["label_status"] == "reviewed_static_camera_copy"


def test_load_real_court_keypoint_labels_rejects_unknown_item_status(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(item_status="reviewed_by_a_robot"),
    )

    with pytest.raises(ValueError, match="status must be one of"):
        load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")


def test_load_real_court_keypoint_labels_accepts_static_camera_copy_status(tmp_path: Path) -> None:
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(item_status="reviewed_static_camera_copy"),
    )

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    assert rows[0]["label_status"] == "reviewed_static_camera_copy"


def test_load_real_court_keypoint_labels_accepts_synthetic_status(tmp_path: Path) -> None:
    """CAL-R2 provenance fix: 'synthetic' is its own accepted status, distinct from
    'reviewed_static_camera_copy' -- see the SYNTHETIC_STATUS comment in
    train_court_keypoint_heatmap.py for why conflating the two was a real bug (it let synthetic
    rows silently inflate a count meant only for owner-approved REAL human-review copies)."""
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    (clip_root / "source.mp4").write_bytes(b"fake video")
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(item_status="synthetic"),
    )

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")

    assert len(rows) == 1
    assert rows[0]["label_status"] == "synthetic"


def test_partial_null_schema_loads_only_labeled_keypoints_and_builds_channel_mask() -> None:
    payload = _partial_external_payload()

    row = court_keypoint_label_rows(payload, allow_pending_diagnostic_only=True)[0]

    assert row["label_status"] == "reviewed_external_dataset"
    assert row["label_source"] == "reviewed_partial_court_keypoint_labels"
    assert len(row["keypoints"]) == 12
    assert set(row["keypoints"]).isdisjoint({"net_left_sideline", "net_center", "net_right_sideline"})
    _, mask = heatmaps_for_points(
        row["keypoints"],
        [point.name for point in PICKLEBALL_KEYPOINTS],
        64,
        36,
        sigma=1.5,
    )
    assert mask.shape == (15, 36, 64)
    assert mask.sum() == pytest.approx(12 * 36 * 64)
    names = [point.name for point in PICKLEBALL_KEYPOINTS]
    for name in ("net_left_sideline", "net_center", "net_right_sideline"):
        assert mask[names.index(name)].sum() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda keypoints: keypoints.pop("net_center"), "exactly the 15 canonical keypoints"),
        (
            lambda keypoints: keypoints.__setitem__("net_center", {"labeled": False}),
            "must be a two-item image coordinate",
        ),
        (lambda keypoints: [keypoints.__setitem__(name, None) for name in list(keypoints)], "at least one"),
    ],
)
def test_loader_rejects_malformed_partial_schemas_fail_loud(mutate: object, match: str) -> None:
    payload = _partial_external_payload()
    mutate(payload["annotation"]["items"][0]["keypoints"])

    with pytest.raises(ValueError, match=match):
        court_keypoint_label_rows(payload, allow_pending_diagnostic_only=True)


def test_training_summary_counts_external_dataset_rows_separately(tmp_path: Path) -> None:
    real_root = _write_mixed_partial_training_fixture(tmp_path)
    out = tmp_path / "count_summary"

    summary = run_training(
        Namespace(
            real_root=real_root,
            out=out,
            holdout_clip=["not_a_real_clip"],
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

    assert summary["labels_independent_human_frames"] == 1
    assert summary["labels_external_dataset_frame_count"] == 1
    assert summary["gate"]["independent_reviewed_frame_count"] == 1
    assert summary["gate"]["external_dataset_frame_count"] == 1
    assert training_cli_summary(summary)["labels_external_dataset_frame_count"] == 1
    assert summary["after"]["real_keypoint_count"] == 12

    rows = load_real_court_keypoint_labels(real_root)
    report = evaluate_checkpoint_against_real_labels(Path(summary["checkpoint"]), rows, device="cpu")
    assert report["raw_all"]["keypoint_error_summary"]["count"] == 27
    assert [item["keypoint_count"] for item in report["raw_all"]["per_row"]] == [15, 12]
    assert report["independent_frame_count"] == 1
    assert report["raw_independent"]["keypoint_error_summary"]["count"] == 15


def test_direct_cli_two_epoch_mixed_full15_and_partial_external_smoke(tmp_path: Path) -> None:
    real_root = _write_mixed_partial_training_fixture(tmp_path)
    out = tmp_path / "direct_cli_smoke"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/train_court_keypoint_heatmap.py",
            "--real-root",
            str(real_root),
            "--out",
            str(out),
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--image-width",
            "32",
            "--image-height",
            "18",
            "--real-finetune-start-epoch",
            "0",
            "--eval-every",
            "1",
            "--device",
            "cpu",
            "--skip-holdout-artifacts",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out / "court_keypoint_heatmap.pt").is_file()
    metrics = json.loads((out / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 2
    assert metrics["labels_independent_human_frames"] == 1
    assert metrics["labels_external_dataset_frame_count"] == 1
    assert '"checkpoint"' in completed.stdout


def test_run_training_writes_holdout_predictions_overlay_and_gate_metric(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + idx * 10, 60, 90)
        writer.write(frame)
    writer.release()
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg",
            source_resolution=[64, 36],
            label_coordinate_space=[64, 36],
        ),
    )

    out = tmp_path / "court_run"
    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=out,
            holdout_clip=["clip_a"],
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
            static_camera_aggregate=False,
            enable_homography_refinement=True,
            disable_homography_refinement=False,
        )
    )

    assert summary["gate"]["metric"] == "heldout_pck_at_5px"
    assert summary["gate"]["threshold"] == 0.95
    assert summary["gate"]["pck_threshold_px"] == 5.0
    assert summary["gate"]["value"] == summary["after"]["real_keypoint_pck_at_5px"]
    assert summary["postprocess"]["homography_refinement"] is True
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["keypoint_count"] == 15
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["pck_at_5px"] == summary["after"]["real_keypoint_pck_at_5px"]
    assert summary["after"]["real_corner_median_px"] is not None
    assert summary["after"]["real_corner_median_model_input_px"] is not None
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(summary["after"]["real_corner_median_px"])
    assert summary["after"]["real_corner_median_source_px"] == pytest.approx(
        summary["after"]["real_corner_median_model_input_px"] * 2.0,
        rel=0.15,
    )
    assert summary["holdout_artifacts"][0]["clip"] == "clip_a"
    assert summary["holdout_artifacts"][0]["prediction_artifact"].endswith("clip_a_court_keypoints.json")
    assert summary["holdout_artifacts"][0]["overlay_artifact"].endswith("clip_a_court_keypoints_overlay.mp4")
    assert summary["holdout_artifacts"][0]["overlay_frame_count"] == 3
    assert summary["holdout_artifacts"][0]["heldout_label_frame_index"] == 0
    assert summary["holdout_artifacts"][0]["heldout_label_frame_indices"] == [0]
    assert (out / "holdout_predictions" / "clip_a_court_keypoints.json").is_file()
    assert (out / "holdout_overlays" / "clip_a_court_keypoints_overlay.mp4").is_file()
    prediction_payload = json.loads((out / "holdout_predictions" / "clip_a_court_keypoints.json").read_text(encoding="utf-8"))
    near_left_prediction = prediction_payload["frames"][0]["keypoints"]["near_left_corner"]
    assert near_left_prediction["postprocess"] == "planar_homography_ransac_v1"
    assert len(near_left_prediction["raw_xy"]) == 2
    # A single independently-reviewed frame, no static-camera copies in this fixture.
    assert summary["labels_independent_human_frames"] == 1
    assert summary["labels_static_camera_copy_frame_count"] == 0
    assert summary["gate"]["independent_reviewed_frame_count"] == 1
    assert summary["gate"]["copied_frame_count"] == 0
    assert "Independent human-verified frames = 1" in summary["gate"]["human_verification_note"]
    assert "Independent human-verified frames = 1" in summary["note"]


def test_run_training_reports_synthetic_frame_count_separately_from_human_counts(tmp_path: Path) -> None:
    """CAL-R2 provenance fix regression test: a run_training summary must break out
    labels_synthetic_frame_count as its own field, distinct from
    labels_independent_human_frames / labels_static_camera_copy_frame_count, so a gate can never
    silently count synthetic renders as any form of human verification."""
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + idx * 10, 60, 90)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
    )
    statuses = ["reviewed", "reviewed_static_camera_copy", "synthetic"]
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "status": statuses[frame_index],
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(3)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_synthetic_status",
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

    assert summary["labels_independent_human_frames"] == 1
    assert summary["labels_static_camera_copy_frame_count"] == 1
    assert summary["labels_synthetic_frame_count"] == 1
    assert summary["gate"]["independent_reviewed_frame_count"] == 1
    assert summary["gate"]["copied_frame_count"] == 1
    assert summary["gate"]["synthetic_frame_count"] == 1
    assert "1 additional frame(s) are synthetic domain-randomized renders" in summary["gate"]["human_verification_note"]

    cli_summary = training_cli_summary(summary)
    assert cli_summary["labels_synthetic_frame_count"] == 1


def test_run_training_can_hold_out_frames_per_viewpoint(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(6):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (30 + idx * 5, 70, 100)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000001.jpg",
        source_resolution=[64, 36],
        label_coordinate_space=[64, 36],
    )
    # Alternate independent reviews and owner-approved static-camera copies so the test
    # exercises the provenance split, not just the frame-stride holdout split.
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "status": "reviewed" if frame_index % 2 == 1 else "reviewed_static_camera_copy",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(1, 5)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_frame_split",
            holdout_clip=["clip_b"],
            holdout_frame_stride=2,
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
            static_camera_aggregate=True,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
        )
    )

    assert summary["holdout_strategy"] == {"type": "frame_stride", "stride": 2}
    assert summary["real_train_count"] == 2
    assert summary["real_holdout_count"] == 2
    assert summary["gate"]["metric"] == "heldout_static_camera_aggregate_pck_at_5px"
    assert summary["postprocess"]["static_camera_aggregation"] is True
    assert summary["postprocess"]["static_camera_aggregation_row_count"] == 2
    assert summary["after"]["real_keypoint_pck_per_clip"]["clip_a"]["keypoint_count"] == 30
    assert len(summary["holdout_artifacts"]) == 1
    assert summary["holdout_artifacts"][0]["clip"] == "clip_a"
    assert summary["holdout_artifacts"][0]["heldout_label_frame_index"] is None
    assert summary["holdout_artifacts"][0]["heldout_label_frame_indices"] == [2, 4]
    assert summary["holdout_artifacts"][0]["heldout_keypoint_count"] == 30
    # 2 independently-reviewed frames (odd frame indices) + 2 static-camera-copy frames
    # (even frame indices), counted across all loaded rows regardless of train/holdout split.
    assert summary["labels_independent_human_frames"] == 2
    assert summary["labels_static_camera_copy_frame_count"] == 2
    assert summary["gate"]["independent_reviewed_frame_count"] == 2
    assert summary["gate"]["copied_frame_count"] == 2


def test_run_training_can_skip_holdout_artifacts_and_still_write_metrics(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.release()
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg",
            source_resolution=[64, 36],
            label_coordinate_space=[64, 36],
        ),
    )

    out = tmp_path / "court_run_skip_artifacts"
    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=out,
            holdout_clip=["clip_a"],
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

    metrics = json.loads((out / "court_keypoint_metrics.json").read_text(encoding="utf-8"))
    assert summary["holdout_artifacts"] == []
    assert metrics["gate"]["metric"] == "heldout_pck_at_5px"
    assert metrics["holdout_artifacts"] == []
    assert not (out / "holdout_overlays").exists()


def test_curriculum_synthetic_fraction_ramps_linearly_from_start_to_end() -> None:
    assert curriculum_synthetic_fraction(0, 5, start_fraction=0.8, end_fraction=0.2) == pytest.approx(0.8)
    assert curriculum_synthetic_fraction(4, 5, start_fraction=0.8, end_fraction=0.2) == pytest.approx(0.2)
    assert curriculum_synthetic_fraction(2, 5, start_fraction=0.8, end_fraction=0.2) == pytest.approx(0.5)
    # Degenerate total_epochs (<=1) must not divide by zero.
    assert curriculum_synthetic_fraction(0, 1, start_fraction=0.8, end_fraction=0.2) == pytest.approx(0.2)
    assert curriculum_synthetic_fraction(0, 0, start_fraction=0.8, end_fraction=0.2) == pytest.approx(0.2)


def _rows_with_status(status: str, count: int) -> list[dict]:
    return [{"label_status": status, "id": f"{status}_{i}"} for i in range(count)]


def test_sample_curriculum_real_batch_without_curriculum_flags_is_uniform_over_all_rows() -> None:
    rng = __import__("random").Random(7)
    rows = _rows_with_status("reviewed", 6) + _rows_with_status("synthetic", 6)

    sample = sample_curriculum_real_batch(
        rows,
        epoch=0,
        total_epochs=10,
        real_batch_size=4,
        synthetic_curriculum_start_fraction=0.0,
        synthetic_curriculum_end_fraction=0.0,
        rng=rng,
    )

    assert len(sample) == 4
    assert all(row in rows for row in sample)


def test_sample_curriculum_real_batch_is_synthetic_heavy_early_and_real_heavy_late() -> None:
    rng = __import__("random").Random(7)
    rows = _rows_with_status("reviewed", 50) + _rows_with_status("synthetic", 50)

    early_sample = sample_curriculum_real_batch(
        rows,
        epoch=0,
        total_epochs=100,
        real_batch_size=20,
        synthetic_curriculum_start_fraction=0.9,
        synthetic_curriculum_end_fraction=0.1,
        rng=rng,
    )
    late_sample = sample_curriculum_real_batch(
        rows,
        epoch=99,
        total_epochs=100,
        real_batch_size=20,
        synthetic_curriculum_start_fraction=0.9,
        synthetic_curriculum_end_fraction=0.1,
        rng=rng,
    )

    early_synthetic_count = sum(1 for row in early_sample if row["label_status"] == "synthetic")
    late_synthetic_count = sum(1 for row in late_sample if row["label_status"] == "synthetic")
    assert len(early_sample) == 20
    assert len(late_sample) == 20
    assert early_synthetic_count >= 17  # ~90% of 20
    assert late_synthetic_count <= 3  # ~10% of 20
    assert early_synthetic_count > late_synthetic_count


def test_sample_curriculum_real_batch_falls_back_to_uniform_without_a_synthetic_pool() -> None:
    rng = __import__("random").Random(7)
    rows = _rows_with_status("reviewed", 10)

    sample = sample_curriculum_real_batch(
        rows,
        epoch=0,
        total_epochs=10,
        real_batch_size=4,
        synthetic_curriculum_start_fraction=0.9,
        synthetic_curriculum_end_fraction=0.1,
        rng=rng,
    )

    assert len(sample) == 4
    assert all(row["label_status"] == "reviewed" for row in sample)


def test_run_training_applies_geometric_consistency_loss_and_logs_components(tmp_path: Path) -> None:
    """End-to-end plumbing check for --geometric-loss-weight: training must run to completion,
    add the geometric loss to the heatmap loss every step, and log the per-component breakdown
    in history so a run's REPORT.md can compare geometric-loss-on vs off."""
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.release()
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
        ),
    )

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_geometric_loss",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=2,
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
            geometric_loss_weight=0.1,
            geometric_colinearity_weight=1.0,
            geometric_homography_weight=1.0,
        )
    )

    assert Path(summary["checkpoint"]).is_file()
    assert summary["geometric_loss"] == {
        "enabled": True,
        "weight": 0.1,
        "colinearity_weight": 1.0,
        "homography_weight": 1.0,
    }
    assert len(summary["history"]) == 2
    for row in summary["history"]:
        assert "geometric_loss" in row
        assert "geometric_colinearity" in row
        assert "geometric_homography" in row
        assert "heatmap_loss" in row
        assert row["geometric_loss"] >= 0.0


def test_run_training_without_geometric_loss_weight_omits_geometric_history_fields(tmp_path: Path) -> None:
    """Default (--geometric-loss-weight 0.0, or the flag entirely unset via getattr) must skip
    the extra soft-argmax/DLT computation, not just zero-weight it -- this is the backward
    compatibility guarantee for existing Namespace()-based callers/tests that never set the new
    geometric_loss_weight attribute at all."""
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.release()
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
        ),
    )

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_no_geometric_loss",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=1,
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

    assert summary["geometric_loss"] == {
        "enabled": False,
        "weight": 0.0,
        "colinearity_weight": 1.0,
        "homography_weight": 1.0,
    }
    assert "geometric_loss" not in summary["history"][0]


def test_run_training_curriculum_mixes_synthetic_and_real_rows(tmp_path: Path) -> None:
    """End-to-end plumbing check for the synthetic/real curriculum flags against a mixed
    real+synthetic train_real pool, exercising the sample_curriculum_real_batch path inside
    run_training's actual epoch loop (not just the unit-tested helper in isolation)."""
    cv2 = __import__("cv2")
    clip_root = tmp_path / "training_corpora" / "mixed_clip"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(10):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (10 + idx * 5, 40, 70)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
    )
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "status": "reviewed" if frame_index < 2 else "synthetic",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(10)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "training_corpora",
            out=tmp_path / "court_run_curriculum",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=3,
            batch_size=1,
            image_width=32,
            image_height=18,
            sigma=1.5,
            learning_rate=1e-3,
            real_finetune_start_epoch=0,
            real_batch_size=4,
            eval_every=1,
            seed=13,
            device="cpu",
            skip_holdout_artifacts=True,
            static_camera_aggregate=False,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
            synthetic_curriculum_start_fraction=0.8,
            synthetic_curriculum_end_fraction=0.2,
        )
    )

    assert summary["real_train_count"] == 10
    assert summary["curriculum"] == {
        "synthetic_curriculum_start_fraction": 0.8,
        "synthetic_curriculum_end_fraction": 0.2,
    }
    assert Path(summary["checkpoint"]).is_file()
    assert len(summary["history"]) == 3


def test_run_training_line_segmentation_mode_requires_640px_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 640px wide"):
        run_training(
            Namespace(
                real_root=None,
                out=tmp_path / "court_run_line_too_small",
                holdout_clip=["nonexistent_clip"],
                holdout_frame_stride=0,
                epochs=0,
                batch_size=1,
                image_width=160,
                image_height=90,
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
                model_architecture="line_segmentation_intersection_v1",
            )
        )


def test_run_training_line_segmentation_mode_logs_line_intersection_metrics(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "external_corpus" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    assert writer.isOpened()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (20, 60, 90)
    writer.write(frame)
    writer.release()
    _write_json(
        clip_root / "labels" / "court_keypoints.json",
        _reviewed_court_keypoint_label_payload(
            "frame_000000.jpg",
            source_resolution=[640, 360],
            label_coordinate_space=[640, 360],
        ),
    )

    summary = run_training(
        Namespace(
            real_root=tmp_path / "external_corpus",
            out=tmp_path / "court_run_line_mode",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=0,
            batch_size=1,
            image_width=640,
            image_height=360,
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
            model_architecture="line_segmentation_intersection_v1",
            line_width=3,
        )
    )

    assert summary["architecture"]["name"] == "line_segmentation_intersection_v1"
    assert summary["round3_input_resolution"]["image_width"] == 640
    assert summary["gate"]["metric"] == "heldout_line_intersection_pck_at_5px"
    assert summary["after"]["prediction_mode"] == "line_segmentation_intersection"
    assert "line_fit_rms_px" in summary["after"]
    assert "homography_self_consistency_px" in summary["after"]
    assert Path(summary["checkpoint"]).is_file()


def test_real_finetune_batch_size_caps_per_epoch_real_row_count(tmp_path: Path) -> None:
    """Without a cap, every real-finetune epoch processes the ENTIRE train_real set as one
    full-batch step, so per-epoch cost scales with corpus size -- fine for a handful of rows,
    but a real bottleneck once train_real is a multi-hundred-row external-corpus tier (this
    was measured taking projected hours on CPU for a ~750-image tier before this fix). This
    exercises the `--real-batch-size` mini-batch path end-to-end (small real epoch count, more
    train rows than the cap) and asserts it completes and still writes a valid checkpoint --
    a regression test against the full-batch-only code path silently coming back.
    """
    cv2 = __import__("cv2")
    clip_root = tmp_path / "external_corpus" / "some_dataset"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(6):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (10 + idx * 10, 40, 70)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
    )
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "status": "reviewed",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(6)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "external_corpus",
            out=tmp_path / "court_run_real_batch_cap",
            holdout_clip=["nonexistent_clip"],
            holdout_frame_stride=0,
            epochs=2,
            batch_size=1,
            image_width=32,
            image_height=18,
            sigma=1.5,
            learning_rate=1e-3,
            real_finetune_start_epoch=0,
            real_batch_size=2,
            eval_every=1,
            seed=13,
            device="cpu",
            skip_holdout_artifacts=True,
            static_camera_aggregate=False,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
        )
    )

    assert summary["real_train_count"] == 6
    assert Path(summary["checkpoint"]).is_file()
    assert len(summary["history"]) == 2


def test_static_camera_aggregation_scores_only_holdout_rows_not_train_rows(tmp_path: Path) -> None:
    """Eval-integrity regression test for the CAL static-camera aggregation policy.

    `NORTH_STAR_ROADMAP.md`'s "CAL static-camera aggregation policy" note flags a prior uncommitted
    one-off check that aggregated over *training* rows and scored the result as a held-out gate
    number. This constructs a fixture where the train/holdout row counts differ (4 vs 1) and
    asserts the aggregation row count always matches the holdout count, never the (larger)
    train count.
    """
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(5):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + idx * 10, 60, 90)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
    )
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            "status": "reviewed" if frame_index == 0 else "reviewed_static_camera_copy",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(5)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
            out=tmp_path / "court_run_agg_split",
            holdout_clip=["clip_b"],
            holdout_frame_stride=5,
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
            static_camera_aggregate=True,
            enable_homography_refinement=False,
            disable_homography_refinement=False,
        )
    )

    assert summary["real_train_count"] == 4
    assert summary["real_holdout_count"] == 1
    assert summary["postprocess"]["static_camera_aggregation_row_count"] == 1
    assert summary["postprocess"]["static_camera_aggregation_row_count"] != summary["real_train_count"]
    assert summary["postprocess"]["static_camera_aggregation_source"] == "holdout_rows_self_referential"


def test_evaluate_checkpoint_against_real_labels_reports_raw_and_aggregated_modes(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    clip_root = tmp_path / "eval_clips" / "ball" / "clip_a"
    clip_root.mkdir(parents=True)
    video = clip_root / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 36))
    assert writer.isOpened()
    for idx in range(4):
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[:, :] = (20 + idx * 15, 50, 80)
        writer.write(frame)
    writer.release()
    payload = _reviewed_court_keypoint_label_payload(
        "frame_000000.jpg", source_resolution=[64, 36], label_coordinate_space=[64, 36]
    )
    payload["annotation"]["items"] = [
        {
            "frame": f"frame_{frame_index:06d}.jpg",
            # Exactly one independent human review per clip; the rest are owner-approved
            # copies of it, mirroring the real eval_clips/ball/*/labels/court_keypoints.json
            # shape (4 independent frames total, 28 static-camera copies).
            "status": "reviewed" if frame_index == 0 else "reviewed_static_camera_copy",
            "keypoints": {
                point.name: [float(index * 3 + 10), float(index * 2 + 5)]
                for index, point in enumerate(PICKLEBALL_KEYPOINTS)
            },
        }
        for frame_index in range(4)
    ]
    _write_json(clip_root / "labels" / "court_keypoints.json", payload)

    out = tmp_path / "court_run_checkpoint_only"
    summary = run_training(
        Namespace(
            real_root=tmp_path / "eval_clips" / "ball",
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

    rows = load_real_court_keypoint_labels(tmp_path / "eval_clips" / "ball")
    assert len(rows) == 4

    report = evaluate_checkpoint_against_real_labels(checkpoint_path, rows, device="cpu")

    assert report["independent_frame_count"] == 1
    assert report["all_frame_count"] == 4
    assert report["raw_independent"]["frame_count"] == 1
    assert report["raw_all"]["frame_count"] == 4
    assert report["aggregated_independent"]["frame_count"] == 1
    assert report["aggregated_all"]["frame_count"] == 4
    # All 4 rows share the exact same target keypoints (owner-copied labels for a static
    # camera), so the aggregated prediction (one median per clip) scores identically whether
    # compared against the 1 independent frame or all 4 frames.
    assert report["aggregated_independent"]["pck_at_5px"] == report["aggregated_all"]["pck_at_5px"]
    assert report["aggregated_independent"]["keypoint_error_summary"]["mean"] == pytest.approx(
        report["aggregated_all"]["keypoint_error_summary"]["mean"]
    )
    assert "clip_a" in report["raw_independent"]["per_clip"]
    assert "clip_a" in report["aggregated_all"]["per_clip"]

    # Downstream homography self-consistency proxy: all 15 keypoints are predicted for
    # clip_a, so both the raw-representative and aggregated per-clip fits should succeed
    # (not None) and report a finite reprojection error.
    homography = report["homography_self_consistency"]
    assert homography["raw_representative_per_clip"]["clip_a"] is not None
    assert homography["raw_representative_per_clip"]["clip_a"]["count"] == 15
    assert homography["aggregated_per_clip"]["clip_a"] is not None
    assert homography["aggregated_per_clip"]["clip_a"]["count"] == 15


def test_homography_self_consistency_returns_none_below_four_points() -> None:
    from scripts.racketsport.train_court_keypoint_heatmap import _homography_self_consistency_px

    assert _homography_self_consistency_px({"near_left_corner": [10.0, 20.0]}) is None
    assert (
        _homography_self_consistency_px(
            {
                "near_left_corner": [10.0, 20.0],
                "near_right_corner": [200.0, 20.0],
                "far_left_corner": [10.0, 200.0],
                "far_right_corner": [200.0, 200.0],
            }
        )
        is not None
    )


def test_training_cli_summary_prints_gate_metric_and_artifact_paths() -> None:
    summary = training_cli_summary(
        {
            "checkpoint": "run/court_keypoint_heatmap.pt",
            "gate": {
                "metric": "heldout_pck_at_5px",
                "value": 0.9,
                "threshold": 0.95,
                "pck_threshold_px": 5.0,
                "passed": False,
            },
            "before": {"real_keypoint_median_px": 80.0},
            "after": {"real_keypoint_median_px": 40.0},
            "holdout_artifacts": [
                {
                    "clip": "clip_a",
                    "prediction_artifact": "run/holdout_predictions/clip_a.json",
                    "overlay_artifact": "run/holdout_overlays/clip_a.mp4",
                    "median_keypoint_reprojection_px": 12.5,
                }
            ],
            "labels_independent_human_frames": 4,
            "labels_static_camera_copy_frame_count": 28,
        }
    )

    assert summary["checkpoint"] == "run/court_keypoint_heatmap.pt"
    assert summary["gate"]["metric"] == "heldout_pck_at_5px"
    assert summary["gate"]["value"] == 0.9
    assert summary["holdout_artifacts"][0]["overlay_artifact"] == "run/holdout_overlays/clip_a.mp4"
    # CLI-visible summary must keep independent human-verified frames distinguishable from
    # owner-approved static-camera copies (4 vs 28 for the current court-keypoint dataset).
    assert summary["labels_independent_human_frames"] == 4
    assert summary["labels_static_camera_copy_frame_count"] == 28
