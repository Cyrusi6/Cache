"""Exact content-space RoPE transforms for FPCT.

The source and receiver caches store keys after their own rotary embedding.  The
math FPCT path removes that rotation, runs the shared candidate fuser in content
space, and applies the receiver parent rotation exactly once to every fused
candidate.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


FPCT_POSITION_MODES = frozenset({"legacy", "math"})


def normalize_fpct_position_mode(value: str | None) -> str:
    normalized = "legacy" if value is None else str(value).strip().lower()
    if normalized not in FPCT_POSITION_MODES:
        raise ValueError(
            "fpct_position_mode must be one of "
            f"{sorted(FPCT_POSITION_MODES)}, got {value!r}"
        )
    return normalized


def rotate_half(value: Tensor) -> Tensor:
    """The half-rotation used by Llama/Qwen rotary embeddings."""

    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _broadcast_rotary(value: Tensor, cosine: Tensor, sine: Tensor) -> Tuple[Tensor, Tensor]:
    if cosine.shape != sine.shape:
        raise ValueError("FPCT rotary cosine/sine shapes differ")
    if cosine.shape[0] != value.shape[0] or cosine.shape[-1] != value.shape[-1]:
        raise ValueError(
            "FPCT rotary embedding is incompatible with the key tensor: "
            f"key={tuple(value.shape)}, cos={tuple(cosine.shape)}"
        )
    if cosine.ndim + 1 != value.ndim:
        raise ValueError(
            "FPCT rotary embedding must omit only the attention-head axis: "
            f"key={tuple(value.shape)}, cos={tuple(cosine.shape)}"
        )
    return cosine.unsqueeze(1), sine.unsqueeze(1)


def apply_fpct_rope(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Apply the exact model-provided rotary embedding to a key tensor."""

    cosine, sine = _broadcast_rotary(value, cosine, sine)
    working = value.float()
    output = working * cosine.float() + rotate_half(working) * sine.float()
    return output.to(dtype=value.dtype)


def remove_fpct_rope(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Invert RoPE, including non-unit attention scaling when present."""

    cosine, sine = _broadcast_rotary(value, cosine, sine)
    working = value.float()
    cosine = cosine.float()
    sine = sine.float()
    denominator = (cosine.square() + sine.square()).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    output = (working * cosine - rotate_half(working) * sine) / denominator
    return output.to(dtype=value.dtype)


def model_rotary_cos_sin(
    model: nn.Module,
    reference: Tensor,
    position_ids: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Evaluate the actual source/receiver rotary module at exact position IDs."""

    backbone = getattr(model, "model", None)
    rotary = getattr(backbone, "rotary_emb", None)
    if rotary is None or not callable(rotary):
        raise TypeError("FPCT math position mode requires model.model.rotary_emb")
    if position_ids.ndim != 2:
        raise ValueError("FPCT rotary position_ids must have shape [B,N]")
    position_ids = position_ids.to(device=reference.device, dtype=torch.long)
    cosine, sine = rotary(reference, position_ids)
    if cosine.shape != (*position_ids.shape, reference.shape[-1]):
        raise ValueError(
            "FPCT model rotary output has unexpected shape: "
            f"positions={tuple(position_ids.shape)}, cos={tuple(cosine.shape)}, "
            f"head_dim={reference.shape[-1]}"
        )
    return cosine.to(dtype=reference.dtype), sine.to(dtype=reference.dtype)
