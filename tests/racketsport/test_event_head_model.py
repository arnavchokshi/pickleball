from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from threed.racketsport.event_head.model import (
    EventHead,
    checkpoint_payload,
    load_checkpoint,
    masked_cross_entropy,
    upgrade_event_head_with_offset,
)


def test_event_head_shapes_and_masked_cross_entropy() -> None:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    logits = model(torch.randn(2, 4, 3, 32, 32))
    assert logits.shape == (2, 4, 3)
    targets = torch.tensor([[0, 2, 0, 2], [0, 1, 0, 1]])
    masks = torch.tensor([[True, False, True], [True, True, False]])
    loss = masked_cross_entropy(logits, targets, masks)
    assert torch.isfinite(loss)


def test_masked_cross_entropy_class_weights_match_torch_and_none_is_unchanged() -> None:
    logits = torch.tensor(
        [[
            [2.0, -1.0, 0.5], [1.5, 0.0, -0.5], [0.7, -0.2, 0.1],
            [1.2, -0.3, 0.4], [0.3, 1.1, -0.8], [0.2, -0.5, 1.4],
        ]] * 2,
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 1, 2]])
    validity_mask = torch.tensor([
        [True, True, False],
        [True, False, True],
    ])
    valid_target = validity_mask.gather(1, targets).bool()
    masked_logits = logits.masked_fill(~validity_mask[:, None, :], -1e4)

    original_losses = F.cross_entropy(
        masked_logits.flatten(0, 1), targets.flatten(), reduction="none",
    ).reshape_as(targets)
    expected_unweighted = original_losses[valid_target].mean()
    actual_unweighted = masked_cross_entropy(
        logits, targets, validity_mask, class_weights=None,
    )
    assert actual_unweighted.detach().cpu().numpy().tobytes() == (
        expected_unweighted.detach().cpu().numpy().tobytes()
    )

    weights = torch.tensor([1.0, 5.0, 5.0])
    flat_valid = valid_target.flatten()
    expected_weighted = F.cross_entropy(
        masked_logits.flatten(0, 1)[flat_valid],
        targets.flatten()[flat_valid],
        weight=weights,
    )
    actual_weighted = masked_cross_entropy(
        logits, targets, validity_mask, class_weights=weights,
    )
    assert torch.equal(actual_weighted, expected_weighted)


def test_imagenet_flag_fails_loudly_when_loader_fails(monkeypatch) -> None:
    import threed.racketsport.event_head.model as module

    monkeypatch.setattr(module, "mobilenet_v3_small", lambda **_: (_ for _ in ()).throw(OSError("offline")))
    try:
        module.EventHead(weights="imagenet")
    except RuntimeError as exc:
        assert "could not be loaded" in str(exc)
    else:
        raise AssertionError("ImageNet load failure was swallowed")


def test_default_forward_preserves_legacy_math_and_state_surface() -> None:
    torch.manual_seed(20260722)
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    model.eval()
    frames = torch.randn(2, 4, 3, 32, 32)

    batch, time, channels, height, width = frames.shape
    encoded = model.frame_backbone(
        frames.reshape(batch * time, channels, height, width)
    )
    encoded = model.pool(encoded).flatten(1)
    encoded = model.frame_projection(encoded).reshape(batch, time, -1)
    temporal, _ = model.temporal(encoded)
    legacy_logits = model.classifier(temporal)

    current_logits = model(frames)
    assert current_logits.detach().numpy().tobytes() == (
        legacy_logits.detach().numpy().tobytes()
    )
    assert model.config["offset_regression_head"] is False
    assert model.offset_regressor is None
    assert not any(key.startswith("offset_regressor.") for key in model.state_dict())
    with pytest.raises(RuntimeError, match="offset regression head is disabled"):
        model.forward_with_aux(frames)


def test_offset_aux_shapes_and_gradients_without_changing_forward_contract() -> None:
    torch.manual_seed(20260722)
    legacy = EventHead(weights="none", feature_dim=8, hidden_dim=8).eval()
    torch.manual_seed(20260722)
    model = EventHead(
        weights="none", feature_dim=8, hidden_dim=8,
        offset_regression_head=True,
    )
    model.eval()
    frames = torch.randn(2, 4, 3, 32, 32)

    logits, offsets = model.forward_with_aux(frames)
    legacy_api_logits = model(frames)
    legacy_model_logits = legacy(frames)
    assert logits.shape == (2, 4, 3)
    assert offsets.shape == (2, 4, 2)
    assert legacy_api_logits.detach().numpy().tobytes() == (
        logits.detach().numpy().tobytes()
    )
    # The optional module is initialized after every legacy module, so it does
    # not perturb seeded initialization or classification output.
    assert legacy_model_logits.detach().numpy().tobytes() == (
        logits.detach().numpy().tobytes()
    )

    (logits.square().mean() + offsets.square().mean()).backward()
    assert model.offset_regressor is not None
    assert model.offset_regressor.weight.grad is not None
    assert torch.isfinite(model.offset_regressor.weight.grad).all()
    assert model.offset_regressor.weight.grad.abs().sum() > 0


def test_offset_checkpoint_roundtrip_is_strict(tmp_path) -> None:
    torch.manual_seed(20260722)
    model = EventHead(
        weights="none", feature_dim=8, hidden_dim=8,
        offset_regression_head=True,
    ).eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    expected_logits, expected_offsets = model.forward_with_aux(frames)
    checkpoint = tmp_path / "offset_event_head.pt"
    torch.save(checkpoint_payload(model, step=17), checkpoint)

    loaded, payload = load_checkpoint(checkpoint)
    actual_logits, actual_offsets = loaded.forward_with_aux(frames)
    assert payload["step"] == 17
    assert loaded.config["offset_regression_head"] is True
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_offsets, expected_offsets)


def test_old_checkpoint_without_offset_config_loads_strictly(tmp_path) -> None:
    torch.manual_seed(20260722)
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8).eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    expected = model(frames)
    payload = checkpoint_payload(model)
    model_config = dict(payload["model_config"])
    model_config.pop("offset_regression_head")
    payload["model_config"] = model_config
    checkpoint = tmp_path / "legacy_event_head.pt"
    torch.save(payload, checkpoint)

    loaded, loaded_payload = load_checkpoint(checkpoint)
    assert "offset_regression_head" not in loaded_payload["model_config"]
    assert loaded.config["offset_regression_head"] is False
    assert loaded.offset_regressor is None
    assert torch.equal(loaded(frames), expected)


def test_model_only_upgrade_copies_every_legacy_tensor_and_preserves_logits() -> None:
    torch.manual_seed(20260722)
    legacy = EventHead(weights="none", feature_dim=8, hidden_dim=8).eval()
    frames = torch.randn(2, 4, 3, 32, 32)
    expected_logits = legacy(frames)

    torch.manual_seed(7)
    upgraded = upgrade_event_head_with_offset(legacy)
    assert upgraded.config["offset_regression_head"] is True
    assert upgraded.offset_regressor is not None
    for key, value in legacy.state_dict().items():
        assert torch.equal(upgraded.state_dict()[key], value), key
    assert torch.equal(upgraded(frames), expected_logits)
    assert upgraded.forward_with_aux(frames)[1].shape == (2, 4, 2)

    with pytest.raises(ValueError, match="already has offset regression enabled"):
        upgrade_event_head_with_offset(upgraded)
