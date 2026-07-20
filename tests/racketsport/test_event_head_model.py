from __future__ import annotations

import torch
from torch.nn import functional as F

from threed.racketsport.event_head.model import EventHead, masked_cross_entropy


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
