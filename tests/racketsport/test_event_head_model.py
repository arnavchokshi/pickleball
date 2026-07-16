from __future__ import annotations

import torch

from threed.racketsport.event_head.model import EventHead, masked_cross_entropy


def test_event_head_shapes_and_masked_cross_entropy() -> None:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    logits = model(torch.randn(2, 4, 3, 32, 32))
    assert logits.shape == (2, 4, 3)
    targets = torch.tensor([[0, 2, 0, 2], [0, 1, 0, 1]])
    masks = torch.tensor([[True, False, True], [True, True, False]])
    loss = masked_cross_entropy(logits, targets, masks)
    assert torch.isfinite(loss)


def test_imagenet_flag_fails_loudly_when_loader_fails(monkeypatch) -> None:
    import threed.racketsport.event_head.model as module

    monkeypatch.setattr(module, "mobilenet_v3_small", lambda **_: (_ for _ in ()).throw(OSError("offline")))
    try:
        module.EventHead(weights="imagenet")
    except RuntimeError as exc:
        assert "could not be loaded" in str(exc)
    else:
        raise AssertionError("ImageNet load failure was swallowed")
