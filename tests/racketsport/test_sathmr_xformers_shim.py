from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def test_xformers_shim_exposes_memory_attention_and_block_diagonal_mask(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "third_party_shims"))
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    ops = importlib.import_module("xformers.ops")

    q = torch.randn(1, 5, 2, 4)
    k = torch.randn(1, 7, 2, 4)
    v = torch.randn(1, 7, 2, 3)
    mask = ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(q_seqlen=[2, 3], kv_seqlen=[4, 3])

    out = ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    assert out.shape == (1, 5, 2, 3)
    first, second = mask.split(out)
    assert first.shape == (1, 2, 2, 3)
    assert second.shape == (1, 3, 2, 3)

    nested = ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([3, 3, 5])
    nested._batch_sizes = [2, 1]
    first_group, second_group = nested.split(torch.randn(1, 11, 4))
    assert first_group.shape == (2, 3, 4)
    assert second_group.shape == (1, 5, 4)


def test_xformers_shim_exposes_unbind_swiglu_and_index_helpers(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "third_party_shims"))
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)

    ops = importlib.import_module("xformers.ops")

    values = torch.arange(24).reshape(2, 3, 4)
    assert len(ops.unbind(values, 1)) == 3

    selected = ops.index_select_cat(
        [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])],
        [torch.tensor([1]), torch.tensor([0, 1])],
    )
    assert selected.tolist() == [[3, 4], [5, 6], [7, 8]]

    target = torch.zeros(3, 2)
    source = torch.ones(2, 2)
    scaling = torch.tensor([2.0, 3.0])
    added = ops.scaled_index_add(target, torch.tensor([0, 2]), source, scaling=scaling, alpha=0.5)
    assert added.tolist() == [[1.0, 1.5], [0.0, 0.0], [1.0, 1.5]]

    layer = ops.SwiGLU(in_features=4, hidden_features=8, out_features=2)
    assert layer(torch.randn(3, 4)).shape == (3, 2)
