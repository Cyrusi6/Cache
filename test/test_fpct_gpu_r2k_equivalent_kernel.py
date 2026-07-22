from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rosetta.model import fpct_attention
from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    bind_fpct_layout_layer_semantics,
    build_fpct_packed_layout,
    fpct_eager_attention,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    pack_fpct_memory,
)
from rosetta.model import wrapper as fpct_wrapper


def _case(dtype: torch.dtype):
    generator = torch.Generator().manual_seed(20260722)
    q = torch.randn(2, 4, 3, 4, generator=generator, dtype=dtype)
    native_key = torch.randn(2, 2, 5, 4, generator=generator, dtype=dtype)
    native_value = torch.randn(2, 2, 5, 4, generator=generator, dtype=dtype)
    candidate_key = torch.randn(2, 2, 5, 4, 4, generator=generator, dtype=dtype)
    candidate_value = torch.randn(2, 2, 5, 4, 4, generator=generator, dtype=dtype)
    prior = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.5, 0.3, 0.2, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.6, 0.4, 0.0, 0.0],
                [0.6, 0.25, 0.15, 0.0],
                [0.55, 0.2, 0.15, 0.1],
            ],
        ],
        dtype=torch.float32,
    )
    valid = prior > 0
    weights = prior[:, None, :, :, None].to(dtype=dtype)
    collapsed_key = (candidate_key * weights).sum(dim=3)
    collapsed_value = (candidate_value * weights).sum(dim=3)
    supported = valid.any(dim=-1)[:, None, :, None]
    cache_key = torch.where(supported, collapsed_key, native_key)
    cache_value = torch.where(supported, collapsed_value, native_value)
    attention_mask = torch.zeros(2, 1, 3, 5, dtype=torch.float32)
    attention_mask[:, :, :, -1] = torch.tensor([0.0, -torch.inf, 0.0])[None, :]
    return q, cache_key, cache_value, candidate_key, candidate_value, prior, valid, attention_mask


def test_fp64_grouped_beta_gamma_equals_flat_global_softmax() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 3, 4, 5, 4, generator=generator, dtype=torch.float64)
    valid = torch.rand(2, 5, 4, generator=generator) > 0.35
    valid[:, :, 0] = True
    masked = torch.where(
        valid[:, None, None], logits, torch.full_like(logits, -torch.inf)
    )
    flat = torch.softmax(masked.flatten(-2), dim=-1).unflatten(-1, (5, 4))
    gamma = torch.softmax(masked, dim=-1)
    group = torch.logsumexp(masked, dim=-1)
    beta = torch.softmax(group, dim=-1)
    hierarchical = beta[..., None] * gamma
    torch.testing.assert_close(flat, hierarchical, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [(torch.float32, 2e-5, 2e-5), (torch.bfloat16, 2e-2, 2e-2)],
)
def test_flat_kernel_matches_dense_output_and_gradients(dtype, atol, rtol) -> None:
    values = _case(dtype)

    def run(use_production: bool):
        q, ck, cv, fk, fv, prior, valid, mask = values
        q = q.detach().clone().requires_grad_(True)
        ck = ck.detach().clone().requires_grad_(True)
        cv = cv.detach().clone().requires_grad_(True)
        fk = fk.detach().clone().requires_grad_(True)
        fv = fv.detach().clone().requires_grad_(True)
        sidecar = FPCTSidecarSegment(0, fk, fv, prior, valid)
        packed = pack_fpct_memory(ck, cv, mask, [sidecar], query_length=q.shape[2])
        if use_production:
            module = SimpleNamespace(num_key_value_groups=2, training=False)
            output, _ = fpct_qwen_hierarchical_attention_forward(
                module,
                q,
                packed,
                ck,
                cv,
                mask,
                scaling=0.5,
            )
            output = output.transpose(1, 2)
        else:
            output, _ = fpct_eager_attention(q, packed)
        loss = output.float().square().mean()
        loss.backward()
        return output.detach(), [tensor.grad.detach() for tensor in (q, ck, cv, fk, fv)]

    actual, actual_gradients = run(True)
    expected, expected_gradients = run(False)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), atol=atol, rtol=rtol
        )


def test_m0_through_m4_are_packed_as_frozen_strata() -> None:
    q, ck, cv, fk, fv, prior, valid, mask = _case(torch.float32)
    packed = pack_fpct_memory(
        ck, cv, mask, [FPCTSidecarSegment(0, fk, fv, prior, valid)], query_length=3
    )
    assert valid[0].sum(dim=-1).tolist() == [0, 1, 2, 3, 4]
    assert packed.expanded_slots.tolist() == [11, 11]
    assert packed.extra_slots.tolist() == [6, 6]
    assert torch.isfinite(packed.attention_mask).any(dim=-1).all()


def test_parent_logits_are_computed_once_and_reused(monkeypatch) -> None:
    q, ck, cv, fk, fv, prior, valid, mask = _case(torch.float32)
    candidate_key = fk.clone()
    candidate_value = fv.clone()
    candidate_key[0] = ck[0, :, :, None, :]
    candidate_value[0] = cv[0, :, :, None, :]
    sidecar = FPCTSidecarSegment(0, candidate_key, candidate_value, prior, valid)
    packed = pack_fpct_memory(ck, cv, mask, [sidecar], query_length=3)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    qk_rhs = []
    original = torch.matmul

    def traced(left, right, *args, **kwargs):
        if left.ndim == 4 and right.ndim == 4 and left.shape[-1] == q.shape[-1] and right.shape[-2] == q.shape[-1]:
            qk_rhs.append(right.detach().clone())
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(fpct_attention.torch, "matmul", traced)
    actual, _ = fpct_qwen_hierarchical_attention_forward(
        module, q, packed, ck, cv, mask, scaling=0.5
    )
    parent, _ = fpct_qwen_eager_attention_forward(
        module, q, ck, cv, mask, scaling=0.5
    )
    assert len(qk_rhs) == 3  # two production QK calls plus the explicit test reference
    atom_rhs = qk_rhs[1]
    equivalent_slots = (
        packed.active
        & torch.gather(
            packed.parent_equivalent, 1, packed.parent_index.clamp_min(0)
        )
    )
    assert torch.count_nonzero(
        atom_rhs.transpose(2, 3)[equivalent_slots[:, None, :, None].expand_as(atom_rhs.transpose(2, 3))]
    ) == 0
    torch.testing.assert_close(actual[0], parent[0], atol=0, rtol=0)


def test_layer_semantic_metadata_is_built_once_and_selected_by_layer() -> None:
    q, ck, cv, fk, fv, prior, valid, _mask = _case(torch.float32)
    first = FPCTSidecarSegment(
        0, fk, fv, prior, valid, parent_force_native=torch.ones(2, 5, dtype=torch.bool)
    )
    second = FPCTSidecarSegment(
        0, fk, fv, prior, valid, parent_force_native=torch.zeros(2, 5, dtype=torch.bool)
    )
    layout = build_fpct_packed_layout(ck.shape[2], [first])
    layout = bind_fpct_layout_layer_semantics(layout, [(3, [first]), (7, [second])])
    assert layout.safe_parent.data_ptr() != 0
    assert layout.safe_candidate_flat_index.data_ptr() != 0
    assert torch.all(layout.semantic_parent_equivalent(3))
    assert not torch.any(layout.semantic_parent_equivalent(7))
    assert not layout.semantic_parent_equivalent_by_layer.requires_grad


def test_instrumentation_off_scope_does_not_call_record_function(monkeypatch) -> None:
    def forbidden(_name):
        raise AssertionError("record_function was called with scopes disabled")

    monkeypatch.setattr(fpct_wrapper, "record_function", forbidden)
    with fpct_wrapper._fpct_scope(False, "fpct.attention"):
        pass


def test_hot_path_source_has_no_group_value_or_host_sync_api() -> None:
    source = inspect.getsource(fpct_qwen_hierarchical_attention_forward)
    assert "group_value" not in source
    for forbidden in (".item(", ".tolist(", ".cpu(", ".numpy("):
        assert forbidden not in source
    assert source.count("fpct_qwen_eager_attention_forward(") == 1
    assert "return_fp32_logits=True" in source
