from __future__ import annotations

from types import SimpleNamespace

import torch

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    bind_fpct_layout_layer_semantics,
    build_fpct_packed_layout,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    pack_fpct_memory,
)


def _segment(
    start: int,
    *,
    batch: int = 1,
    length: int = 3,
    parent_equivalent: torch.Tensor | None = None,
    parent_force_native: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> FPCTSidecarSegment:
    key = torch.zeros(batch, 1, length, 2, 4, dtype=dtype)
    value = torch.zeros_like(key)
    prior = torch.full((batch, length, 2), 0.5, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    return FPCTSidecarSegment(
        start,
        key,
        value,
        prior,
        valid,
        parent_force_native=parent_force_native,
        parent_equivalent=parent_equivalent,
    )


def _bound_map(
    source_length: int,
    segments: list[FPCTSidecarSegment],
    *,
    layer: int = 0,
) -> torch.Tensor:
    layout = build_fpct_packed_layout(source_length, segments)
    bound = bind_fpct_layout_layer_semantics(layout, [(layer, segments)])
    value = bound.semantic_parent_equivalent(layer)
    assert value is not None
    return value


def test_red_green_equivalent_middle_sidecar_keeps_native_edges_true() -> None:
    equivalent = torch.ones(1, 3, dtype=torch.bool)
    semantic = _bound_map(9, [_segment(2, parent_equivalent=equivalent)])
    assert semantic.dtype == torch.bool
    assert semantic.tolist() == [[True] * 9]


def test_one_false_sidecar_parent_does_not_poison_native_positions() -> None:
    equivalent = torch.tensor([[True, False, True]])
    semantic = _bound_map(9, [_segment(2, parent_equivalent=equivalent)])
    expected = torch.ones(1, 9, dtype=torch.bool)
    expected[:, 3] = False
    assert torch.equal(semantic, expected)


def test_unknown_sidecar_metadata_fails_closed_only_inside_span() -> None:
    semantic = _bound_map(9, [_segment(2)])
    expected = torch.ones(1, 9, dtype=torch.bool)
    expected[:, 2:5] = False
    assert torch.equal(semantic, expected)


def test_discontinuous_sidecars_preserve_native_gaps() -> None:
    first = _segment(
        1,
        length=2,
        parent_equivalent=torch.tensor([[True, False]]),
    )
    second = _segment(
        6,
        length=2,
        parent_force_native=torch.tensor([[False, True]]),
    )
    semantic = _bound_map(10, [first, second])
    expected = torch.ones(1, 10, dtype=torch.bool)
    expected[:, 2] = False
    expected[:, 6] = False
    assert torch.equal(semantic, expected)


@torch.no_grad()
def test_mixed_batch_exact_sample_selects_parent_and_active_sample_stays_flat() -> None:
    generator = torch.Generator().manual_seed(20260722)
    batch, hkv, source_length, dimension = 2, 1, 9, 4
    query = torch.randn(batch, 2, 2, dimension, generator=generator)
    parent_key = torch.randn(
        batch, hkv, source_length, dimension, generator=generator
    )
    parent_value = torch.randn(
        batch, hkv, source_length, dimension, generator=generator
    )
    candidate_key = parent_key[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
    candidate_value = parent_value[:, :, 2:5, None, :].expand(-1, -1, -1, 2, -1).clone()
    candidate_key[1, :, 1, 0] += 1.5
    candidate_value[1, :, 1, 0] -= 0.75
    prior = torch.full((batch, 3, 2), 0.5, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    equivalent = torch.tensor(
        [[True, True, True], [True, False, True]], dtype=torch.bool
    )
    segment = FPCTSidecarSegment(
        2,
        candidate_key,
        candidate_value,
        prior,
        valid,
        parent_equivalent=equivalent,
    )
    layout = build_fpct_packed_layout(source_length, [segment])
    layout = bind_fpct_layout_layer_semantics(layout, [(0, [segment])])
    semantic = layout.semantic_parent_equivalent(0)
    assert semantic is not None
    assert semantic[0].all()
    assert not semantic[1, 3]

    mask = torch.zeros(batch, 1, query.shape[2], source_length)
    packed = pack_fpct_memory(
        parent_key,
        parent_value,
        mask,
        [segment],
        query_length=query.shape[2],
        layout=layout,
        semantic_parent_equivalent=semantic,
    )
    assert packed.all_parent_equivalent.tolist() == [True, False]
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    parent_output, _ = fpct_qwen_eager_attention_forward(
        module,
        query,
        parent_key,
        parent_value,
        mask,
        scaling=0.5,
    )
    factorized_output, _ = fpct_qwen_hierarchical_attention_forward(
        module,
        query,
        packed,
        parent_key,
        parent_value,
        mask,
        scaling=0.5,
    )
    assert torch.equal(factorized_output[0], parent_output[0])
    assert not torch.equal(factorized_output[1], parent_output[1])


def test_mixed_batch_training_dropout_and_active_candidate_gradients() -> None:
    generator = torch.Generator().manual_seed(404)
    batch, source_length, dimension = 2, 9, 4
    query = torch.randn(batch, 2, 2, dimension, generator=generator, requires_grad=True)
    parent_key = torch.randn(batch, 1, source_length, dimension, generator=generator)
    parent_value = torch.randn(batch, 1, source_length, dimension, generator=generator)
    candidate_key = (
        parent_key[:, :, 2:5, None, :]
        .expand(-1, -1, -1, 2, -1)
        .clone()
        .requires_grad_(True)
    )
    candidate_value = (
        parent_value[:, :, 2:5, None, :]
        .expand(-1, -1, -1, 2, -1)
        .clone()
        .requires_grad_(True)
    )
    with torch.no_grad():
        candidate_key[1, :, 1, 0] += 0.5
        candidate_value[1, :, 1, 0] -= 0.25
    prior = torch.full((batch, 3, 2), 0.5, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    segment = FPCTSidecarSegment(
        2,
        candidate_key,
        candidate_value,
        prior,
        valid,
        parent_equivalent=torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        ),
    )
    layout = bind_fpct_layout_layer_semantics(
        build_fpct_packed_layout(source_length, [segment]), [(0, [segment])]
    )
    semantic = layout.semantic_parent_equivalent(0)
    assert semantic is not None
    mask = torch.zeros(batch, 1, query.shape[2], source_length)
    packed = pack_fpct_memory(
        parent_key,
        parent_value,
        mask,
        [segment],
        query_length=query.shape[2],
        layout=layout,
        semantic_parent_equivalent=semantic,
    )
    module = SimpleNamespace(num_key_value_groups=2, training=True)
    torch.manual_seed(12345)
    first, _ = fpct_qwen_hierarchical_attention_forward(
        module,
        query,
        packed,
        parent_key,
        parent_value,
        mask,
        scaling=0.5,
        dropout=0.1,
    )
    torch.manual_seed(12345)
    second, _ = fpct_qwen_hierarchical_attention_forward(
        module,
        query,
        packed,
        parent_key,
        parent_value,
        mask,
        scaling=0.5,
        dropout=0.1,
    )
    assert torch.equal(first, second)
    first[1].float().square().mean().backward()
    assert candidate_key.grad is not None and torch.isfinite(candidate_key.grad).all()
    assert candidate_value.grad is not None and torch.isfinite(candidate_value.grad).all()
    assert candidate_key.grad[1].abs().sum() > 0
    assert candidate_value.grad[1].abs().sum() > 0
