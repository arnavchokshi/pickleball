from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS

torch = pytest.importorskip("torch")

from threed.racketsport.court_model_infer import (  # noqa: E402
    build_court_model_from_checkpoint,
    infer_court_model,
    load_court_model_checkpoint,
)
from threed.racketsport.court_keypoint_net import make_court_keypoint_heatmap_model  # noqa: E402
from threed.racketsport.court_structured_model import (  # noqa: E402
    COURT_STRUCTURED_V3_ARCHITECTURE,
    STRUCTURED_FLOOR_KEYPOINT_NAMES,
    make_court_structured_v3_model,
)
from threed.racketsport.court_confidence_calibration import (  # noqa: E402
    fit_isotonic_confidence,
)

KEYPOINT_NAMES = [point.name for point in PICKLEBALL_KEYPOINTS]


def _write_checkpoint(tmp_path: Path, *, image_size: tuple[int, int] = (640, 360)) -> Path:
    model = make_court_keypoint_heatmap_model(len(KEYPOINT_NAMES), architecture="court_unet_v2")
    checkpoint_path = tmp_path / "court_model_v2.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": list(image_size),
            "model_architecture": "court_unet_v2",
            "network_architecture": "court_unet_v2",
            "keypoint_names": KEYPOINT_NAMES,
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_infer_court_model_returns_stable_contract_keys_and_shapes(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    image_bgr = (np.random.RandomState(0).rand(720, 1280, 3) * 255).astype(np.uint8)

    result = infer_court_model(image_bgr, checkpoint_path, device="cpu")

    assert set(result) == {
        "keypoints_xy",
        "keypoints_conf",
        "keypoints_vis",
        "line_family_mask",
        "line_distance_maps",
        "surface_mask",
        "structured_observations",
        "best_court",
    }
    assert set(result["keypoints_xy"]) == set(KEYPOINT_NAMES)
    assert set(result["keypoints_conf"]) == set(KEYPOINT_NAMES)
    assert set(result["keypoints_vis"]) == set(KEYPOINT_NAMES)
    for name in KEYPOINT_NAMES:
        x, y = result["keypoints_xy"][name]
        assert 0.0 <= x <= 1280.0
        assert 0.0 <= y <= 720.0
        assert 0.0 <= result["keypoints_conf"][name] <= 1.0
        assert 0.0 <= result["keypoints_vis"][name] <= 1.0

    assert result["line_family_mask"].shape == (720, 1280)
    assert result["surface_mask"].shape == (720, 1280)
    assert result["line_family_mask"].dtype == np.uint8
    assert result["surface_mask"].dtype == np.uint8
    # This checkpoint's 5-class head has no separate "apron" class -- surface_mask is always
    # restricted to {background, interior}, never {apron}=1 (see split_line_family_segmentation).
    assert set(np.unique(result["surface_mask"]).tolist()) <= {0, 2}
    assert set(np.unique(result["line_family_mask"]).tolist()) <= {0, 1, 2, 3}
    assert len(result["structured_observations"]) == 12
    assert result["best_court"]["floor_only"] is True
    assert result["best_court"]["measurement_valid"] is False
    assert result["best_court"]["authority_state"] == "review_only"
    assert result["best_court"]["confidence_status"] == "uncalibrated"
    assert result["best_court"]["point_confidence"] == result["best_court"]["point_confidence_raw"]
    assert set(result["best_court"]["keypoints_xy"]) == {
        name for name in KEYPOINT_NAMES if not name.startswith("net_")
    }


def test_infer_court_model_rescales_to_the_callers_own_image_resolution(tmp_path: Path) -> None:
    """The adapter must return coordinates in whatever resolution `image_bgr` itself has, not the
    checkpoint's fixed model-input resolution -- this is the whole point of the STABLE CONTRACT
    (callers should never need to know the model's native resolution)."""

    checkpoint_path = _write_checkpoint(tmp_path, image_size=(640, 360))
    small_image = (np.random.RandomState(1).rand(90, 160, 3) * 255).astype(np.uint8)
    large_image = (np.random.RandomState(1).rand(1080, 1920, 3) * 255).astype(np.uint8)

    small_result = infer_court_model(small_image, checkpoint_path, device="cpu")
    large_result = infer_court_model(large_image, checkpoint_path, device="cpu")

    for name in KEYPOINT_NAMES:
        sx, sy = small_result["keypoints_xy"][name]
        assert 0.0 <= sx <= 160.0
        assert 0.0 <= sy <= 90.0
        lx, ly = large_result["keypoints_xy"][name]
        assert 0.0 <= lx <= 1920.0
        assert 0.0 <= ly <= 1080.0
    assert small_result["line_family_mask"].shape == (90, 160)
    assert large_result["line_family_mask"].shape == (1080, 1920)


def test_infer_court_model_rejects_non_court_unet_v2_checkpoints(tmp_path: Path) -> None:
    legacy_model = make_court_keypoint_heatmap_model(len(KEYPOINT_NAMES), architecture="encoder_decoder_v1")
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": legacy_model.state_dict(),
            "image_size": [160, 90],
            "model_architecture": "keypoint_heatmap_v1",
            "network_architecture": "encoder_decoder_v1",
            "keypoint_names": KEYPOINT_NAMES,
        },
        checkpoint_path,
    )
    payload = load_court_model_checkpoint(checkpoint_path, device="cpu")
    with pytest.raises(ValueError, match="court_unet_v2"):
        build_court_model_from_checkpoint(payload, device="cpu")


def test_build_adapter_accepts_review_only_structured_v3_checkpoint(tmp_path: Path) -> None:
    model = make_court_structured_v3_model()
    checkpoint_path = tmp_path / "court_structured_v3.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": [160, 96],
            "model_architecture": COURT_STRUCTURED_V3_ARCHITECTURE,
            "network_architecture": COURT_STRUCTURED_V3_ARCHITECTURE,
            "keypoint_names": list(STRUCTURED_FLOOR_KEYPOINT_NAMES),
            "status": "trained_not_phase_verified",
        },
        checkpoint_path,
    )

    payload = load_court_model_checkpoint(checkpoint_path, device="cpu")
    loaded, names, image_size = build_court_model_from_checkpoint(payload, device="cpu")

    assert loaded.architecture == COURT_STRUCTURED_V3_ARCHITECTURE
    assert tuple(names) == STRUCTURED_FLOOR_KEYPOINT_NAMES
    assert image_size == (160, 96)


def test_v3_inference_uses_all_30_floor_channels_and_learned_covariance(tmp_path: Path) -> None:
    model = make_court_structured_v3_model()
    checkpoint_path = tmp_path / "court_structured_v3.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "image_size": [160, 96],
            "model_architecture": COURT_STRUCTURED_V3_ARCHITECTURE,
            "keypoint_names": list(STRUCTURED_FLOOR_KEYPOINT_NAMES),
            "heatmap_decoder": "dark",
            "coordinate_transform": "udp",
        },
        checkpoint_path,
    )
    image_bgr = (np.random.RandomState(11).rand(96, 160, 3) * 255).astype(np.uint8)

    result = infer_court_model(image_bgr, checkpoint_path, device="cpu")

    assert len(result["structured_observations"]) == 30
    assert all(
        row["covariance_policy"]["kind"] == "learned_positive_definite_head"
        for row in result["structured_observations"]
    )
    assert set(result["best_court"]["keypoints_xy"]) == {
        point.name for point in PICKLEBALL_KEYPOINTS if not point.name.startswith("net_")
    }


def test_infer_applies_serialized_point_confidence_calibration(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path, image_size=(160, 96))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    calibrator = fit_isotonic_confidence(
        [0.0, 0.1, 0.9, 1.0],
        [0, 0, 1, 1],
    )
    payload["point_confidence_calibration"] = calibrator.to_dict()
    torch.save(payload, checkpoint_path)

    image_bgr = (np.random.RandomState(7).rand(96, 160, 3) * 255).astype(np.uint8)
    result = infer_court_model(image_bgr, checkpoint_path, device="cpu")

    assert result["best_court"]["confidence_status"] == "calibrated_source_disjoint_dev"
    assert result["best_court"]["point_confidence_calibration"]["sample_count"] == 4
    assert set(result["best_court"]["point_confidence"]) == set(
        result["best_court"]["point_confidence_raw"]
    )


def test_checkpoint_sidecar_selects_measured_decoder_without_rewriting_weights(
    tmp_path: Path,
) -> None:
    checkpoint_path = _write_checkpoint(tmp_path, image_size=(160, 96))
    (tmp_path / "PROVENANCE.json").write_text(
        '{"inference_defaults":{"heatmap_decoder":"dark",'
        '"coordinate_transform":"legacy_stride"}}',
        encoding="utf-8",
    )

    payload = load_court_model_checkpoint(checkpoint_path, device="cpu")
    model, _names, _size = build_court_model_from_checkpoint(payload, device="cpu")

    assert model._heatmap_decoder == "dark"
    assert model._coordinate_transform == "legacy_stride"
    assert payload["inference_defaults_provenance"].endswith("PROVENANCE.json")


def test_infer_court_model_rejects_malformed_image(tmp_path: Path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="HxWx3"):
        infer_court_model(np.zeros((10, 10), dtype=np.uint8), checkpoint_path, device="cpu")
