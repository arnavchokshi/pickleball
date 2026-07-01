from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def memory_efficient_attention(q: Tensor, k: Tensor, v: Tensor, attn_bias: object | None = None) -> Tensor:
    bias = attn_bias.materialize(q, k) if hasattr(attn_bias, "materialize") else attn_bias
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=bias,
    )
    return out.transpose(1, 2)


def unbind(value: Tensor, dim: int) -> tuple[Tensor, ...]:
    return torch.unbind(value, dim=dim)


def index_select_cat(values: list[Tensor], indices: list[Tensor]) -> Tensor:
    if len(values) != len(indices):
        raise ValueError("values and indices must have the same length")
    selected = [value.index_select(0, index.to(device=value.device)) for value, index in zip(values, indices, strict=True)]
    return torch.cat(selected, dim=0)


def scaled_index_add(
    target: Tensor,
    index: Tensor,
    source: Tensor,
    *,
    scaling: Tensor | None = None,
    alpha: float = 1.0,
) -> Tensor:
    update = source
    if scaling is not None:
        update = update * scaling.to(device=source.device, dtype=source.dtype)
    return torch.index_add(target, 0, index.to(device=target.device), update.to(dtype=target.dtype), alpha=alpha)


class SwiGLU(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class BlockDiagonalMask:
    def __init__(self, q_seqlen: list[int], kv_seqlen: list[int]) -> None:
        if len(q_seqlen) != len(kv_seqlen):
            raise ValueError("q_seqlen and kv_seqlen must have the same length")
        self.q_seqlen = [int(value) for value in q_seqlen]
        self.kv_seqlen = [int(value) for value in kv_seqlen]
        self._batch_sizes: list[int] | None = None

    @classmethod
    def from_seqlens(
        cls,
        seqlens: list[int] | tuple[int, ...] | None = None,
        *,
        q_seqlen: list[int] | tuple[int, ...] | None = None,
        kv_seqlen: list[int] | tuple[int, ...] | None = None,
    ) -> BlockDiagonalMask:
        if seqlens is not None:
            q_seqlen = list(seqlens)
            kv_seqlen = list(seqlens)
        if q_seqlen is None or kv_seqlen is None:
            raise ValueError("from_seqlens requires seqlens or q_seqlen and kv_seqlen")
        return cls(list(q_seqlen), list(kv_seqlen))

    def materialize(self, q: Tensor, k: Tensor) -> Tensor:
        q_total = int(q.shape[1])
        kv_total = int(k.shape[1])
        bias = torch.full((1, 1, q_total, kv_total), float("-inf"), device=q.device, dtype=q.dtype)
        q_start = 0
        kv_start = 0
        for q_len, kv_len in zip(self.q_seqlen, self.kv_seqlen, strict=True):
            bias[:, :, q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0.0
            q_start += q_len
            kv_start += kv_len
        return bias

    def split(self, value: Tensor) -> tuple[Tensor, ...]:
        chunks = []
        start = 0
        for length in self.q_seqlen:
            chunks.append(value[:, start : start + length])
            start += length
        if self._batch_sizes is None:
            return tuple(chunks)

        grouped = []
        offset = 0
        for batch_size in self._batch_sizes:
            group = chunks[offset : offset + batch_size]
            if not group:
                grouped.append(value.new_empty((0, 0, *value.shape[2:])))
            else:
                first_len = group[0].shape[1]
                if any(chunk.shape[1] != first_len for chunk in group):
                    raise ValueError("cannot regroup variable sequence lengths into a dense tensor")
                grouped.append(torch.cat(group, dim=0))
            offset += batch_size
        return tuple(grouped)


fmha = SimpleNamespace(attn_bias=SimpleNamespace(BlockDiagonalMask=BlockDiagonalMask))
