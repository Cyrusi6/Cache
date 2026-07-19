from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/analysis/fpct_reference_operator.py"
SPEC = importlib.util.spec_from_file_location("fpct_reference_operator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ref = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ref
SPEC.loader.exec_module(ref)


ATOL64 = 1e-10
RTOL64 = 1e-8
ATOL32 = 2e-5
RTOL32 = 2e-5


def affine_fuser(base_k, base_v, source_k, source_v):
    return 0.7 * base_k + 0.3 * source_k, 0.4 * base_v + 0.6 * source_v


def nonlinear_fuser(base_k, base_v, source_k, source_v):
    return torch.tanh(base_k + 0.8 * source_k), torch.sin(base_v + 0.5 * source_v)


def tensors(dtype=torch.float64, *, hq=4, hkv=2, n=3, k=4, t=2, d=3):
    generator = torch.Generator().manual_seed(20260719)
    q = torch.randn(2, hq, t, d, generator=generator, dtype=dtype)
    native_k = torch.randn(2, hkv, n, d, generator=generator, dtype=dtype)
    native_v = torch.randn(2, hkv, n, d, generator=generator, dtype=dtype)
    source_k = torch.randn(2, hkv, n, k, d, generator=generator, dtype=dtype)
    source_v = torch.randn(2, hkv, n, k, d, generator=generator, dtype=dtype)
    prior_full = torch.tensor(
        [
            [[0.6, 0.4, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0], [0.1, 0.2, 0.3, 0.4]],
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )
    prior = prior_full[:, :n, :k].clone()
    valid = prior > 0
    bias = torch.randn(2, 1, t, n, generator=generator, dtype=dtype) * 0.05
    mask = torch.zeros(2, 1, t, n, dtype=dtype)
    if n > 2:
        mask[:, :, 0, 2] = -torch.inf
    return q, native_k, native_v, source_k, source_v, prior, valid, bias, mask


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float64, ATOL64, RTOL64), (torch.float32, ATOL32, RTOL32)])
def test_flat_global_equals_hierarchical_beta_gamma(dtype, atol, rtol) -> None:
    args = tensors(dtype)
    flat = ref.f_flat(*args[:7], nonlinear_fuser, args[7], args[8])
    hierarchical = ref.f_hierarchical(*args[:7], nonlinear_fuser, args[7], args[8])
    torch.testing.assert_close(flat.output, hierarchical.output, atol=atol, rtol=rtol)
    torch.testing.assert_close(flat.native_probability, hierarchical.native_probability, atol=atol, rtol=rtol)
    torch.testing.assert_close(flat.transported_probability, hierarchical.transported_probability, atol=atol, rtol=rtol)


def test_all_parents_m_le_one_degenerate_cpre_cpost_f() -> None:
    q, nk, nv, sk, sv, _a, _valid, bias, mask = tensors()
    prior = torch.zeros(2, 3, 4, dtype=q.dtype)
    prior[0, 0, 1] = 1
    prior[0, 2, 3] = 1
    prior[1, 1, 0] = 1
    valid = prior > 0
    pre = ref.c_pre(q, nk, nv, sk, sv, prior, valid, nonlinear_fuser, bias, mask)
    post = ref.c_post(q, nk, nv, sk, sv, prior, valid, nonlinear_fuser, bias, mask)
    factorized = ref.f_flat(q, nk, nv, sk, sv, prior, valid, nonlinear_fuser, bias, mask)
    torch.testing.assert_close(pre.output, post.output, atol=ATOL64, rtol=RTOL64)
    torch.testing.assert_close(post.output, factorized.output, atol=ATOL64, rtol=RTOL64)


def test_affine_fuser_cpre_equals_cpost() -> None:
    args = tensors()
    pre = ref.c_pre(*args[:7], affine_fuser, args[7], args[8])
    post = ref.c_post(*args[:7], affine_fuser, args[7], args[8])
    torch.testing.assert_close(pre.output, post.output, atol=ATOL64, rtol=RTOL64)


def test_cpost_and_f_share_identical_candidate_fuser_outputs() -> None:
    args = tensors()
    post = ref.c_post(*args[:7], nonlinear_fuser, args[7], args[8])
    factorized = ref.f_flat(*args[:7], nonlinear_fuser, args[7], args[8])
    torch.testing.assert_close(post.fused_key, factorized.fused_key, atol=0, rtol=0)
    torch.testing.assert_close(post.fused_value, factorized.fused_value, atol=0, rtol=0)


def test_replicated_collapse_equals_cpost() -> None:
    args = tensors()
    post = ref.c_post(*args[:7], nonlinear_fuser, args[7], args[8])
    replicated = ref.replicated_collapse(*args[:7], nonlinear_fuser, args[7], args[8])
    torch.testing.assert_close(post.output, replicated.output, atol=ATOL64, rtol=RTOL64)


def test_refinement_duplicate_invariance() -> None:
    q, nk, nv, sk, sv, _prior, _valid, bias, mask = tensors(n=1)
    single_prior = torch.zeros(2, 1, 4, dtype=q.dtype)
    single_prior[..., 0] = 1.0
    single_valid = single_prior > 0
    single = ref.f_flat(q, nk, nv, sk, sv, single_prior, single_valid, nonlinear_fuser, bias, mask)

    refined_k = sk.clone()
    refined_v = sv.clone()
    refined_k[..., 1, :] = refined_k[..., 0, :]
    refined_v[..., 1, :] = refined_v[..., 0, :]
    refined_prior = torch.zeros_like(single_prior)
    refined_prior[..., 0] = 0.37
    refined_prior[..., 1] = 0.63
    refined = ref.f_flat(q, nk, nv, refined_k, refined_v, refined_prior, refined_prior > 0, nonlinear_fuser, bias, mask)
    torch.testing.assert_close(single.output, refined.output, atol=ATOL64, rtol=RTOL64)


def test_candidate_permutation_with_prior_is_invariant() -> None:
    args = tensors()
    permutation = torch.tensor([2, 0, 3, 1])
    base = ref.f_flat(*args[:7], nonlinear_fuser, args[7], args[8])
    permuted = ref.f_flat(
        args[0], args[1], args[2], args[3][:, :, :, permutation],
        args[4][:, :, :, permutation], args[5][:, :, permutation],
        args[6][:, :, permutation], nonlinear_fuser, args[7], args[8],
    )
    torch.testing.assert_close(base.output, permuted.output, atol=ATOL64, rtol=RTOL64)


def test_causal_parent_mask_is_inherited_by_children() -> None:
    args = tensors()
    result = ref.f_flat(*args[:7], nonlinear_fuser, args[7], args[8])
    assert torch.equal(result.native_probability[:, :, 0, 2], torch.zeros_like(result.native_probability[:, :, 0, 2]))
    assert torch.equal(result.transported_probability[:, :, 0, 2], torch.zeros_like(result.transported_probability[:, :, 0, 2]))


def test_invalid_candidate_probability_and_gradient_are_exact_zero() -> None:
    q, nk, nv, sk, sv, prior, valid, bias, mask = tensors(n=1)
    valid = valid.clone()
    valid[..., 2:] = False
    prior = prior.clone()
    prior[..., 2:] = 9.0  # positive but explicitly invalid
    sk = sk.clone().requires_grad_(True)
    sv = sv.clone().requires_grad_(True)
    result = ref.f_flat(q, nk, nv, sk, sv, prior, valid, nonlinear_fuser, bias, mask)
    assert torch.equal(result.transported_probability[..., 2:], torch.zeros_like(result.transported_probability[..., 2:]))
    result.output.sum().backward()
    assert torch.equal(sk.grad[..., 2:, :], torch.zeros_like(sk.grad[..., 2:, :]))
    assert torch.equal(sv.grad[..., 2:, :], torch.zeros_like(sv.grad[..., 2:, :]))


def test_zero_support_uses_native_fallback_and_all_invalid_is_stable() -> None:
    q, nk, nv, sk, sv, prior, valid, bias, mask = tensors(n=1)
    prior.zero_()
    valid.zero_()
    result = ref.f_flat(q, nk, nv, sk, sv, prior, valid, nonlinear_fuser, bias, mask)
    assert torch.isfinite(result.output).all()
    assert torch.equal(result.transported_probability, torch.zeros_like(result.transported_probability))
    torch.testing.assert_close(result.native_probability.sum(dim=-1), torch.ones_like(result.native_probability.sum(dim=-1)))


def test_native_and_source_atoms_share_one_global_denominator() -> None:
    args = tensors()
    result = ref.f_flat(*args[:7], nonlinear_fuser, args[7], args[8])
    total = result.native_probability.sum(dim=-1) + result.transported_probability.sum(dim=(-1, -2))
    torch.testing.assert_close(total, torch.ones_like(total), atol=ATOL64, rtol=RTOL64)


@pytest.mark.parametrize("hq,hkv", [(4, 2), (4, 1), (2, 2)])
def test_gqa_mqa_shapes(hq: int, hkv: int) -> None:
    args = tensors(hq=hq, hkv=hkv)
    result = ref.f_flat(*args[:7], affine_fuser, args[7], args[8])
    assert result.output.shape == args[0].shape


def test_flat_hierarchical_forward_and_gradients_match() -> None:
    base = tensors(n=2, k=3, t=1, d=2)
    differentiable = [x.clone().requires_grad_(True) for x in base[:6]]
    valid, bias, mask = base[6], base[7], base[8]
    flat = ref.f_flat(*differentiable, valid, nonlinear_fuser, bias, mask).output.sum()
    grad_flat = torch.autograd.grad(flat, differentiable, retain_graph=False)

    differentiable_h = [x.detach().clone().requires_grad_(True) for x in base[:6]]
    hierarchical = ref.f_hierarchical(*differentiable_h, valid, nonlinear_fuser, bias, mask).output.sum()
    grad_h = torch.autograd.grad(hierarchical, differentiable_h, retain_graph=False)
    torch.testing.assert_close(flat, hierarchical, atol=ATOL64, rtol=RTOL64)
    for left, right in zip(grad_flat, grad_h):
        torch.testing.assert_close(left, right, atol=ATOL64, rtol=RTOL64)


def test_torch_gradcheck() -> None:
    dtype = torch.float64
    q = torch.randn(1, 1, 1, 2, dtype=dtype, requires_grad=True)
    nk = torch.randn(1, 1, 1, 2, dtype=dtype, requires_grad=True)
    nv = torch.randn(1, 1, 1, 2, dtype=dtype, requires_grad=True)
    sk = torch.randn(1, 1, 1, 2, 2, dtype=dtype, requires_grad=True)
    sv = torch.randn(1, 1, 1, 2, 2, dtype=dtype, requires_grad=True)
    prior = torch.tensor([[[0.4, 0.6]]], dtype=dtype, requires_grad=True)
    valid = torch.ones_like(prior, dtype=torch.bool)

    def function(*values):
        return ref.f_flat(*values, valid, affine_fuser).output

    assert torch.autograd.gradcheck(function, (q, nk, nv, sk, sv, prior), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_jensen_gap_nonnegative_and_strict_for_distinct_logits() -> None:
    logits = torch.tensor([[[0.0, 2.0, -1.0]]], dtype=torch.float64)
    prior = torch.tensor([[[0.2, 0.5, 0.3]]], dtype=torch.float64)
    gap = ref.jensen_gap(logits, prior, torch.ones_like(prior, dtype=torch.bool))
    assert torch.all(gap >= 0)
    assert torch.all(gap > 0)
    equal = ref.jensen_gap(torch.ones_like(logits), prior, torch.ones_like(prior, dtype=torch.bool))
    torch.testing.assert_close(equal, torch.zeros_like(equal), atol=ATOL64, rtol=RTOL64)


def test_future_g_zero_exact_receiver_recovery() -> None:
    q, nk, nv, sk, sv, prior, valid, bias, mask = tensors()
    fused_k, fused_v, _a, _legal, _support = ref.candidate_fused(nk, nv, sk, sv, prior, valid, nonlinear_fuser)
    recovered, native_p, child_p = ref.general_f_with_native_sibling(
        q, nk, nv, fused_k, fused_v, prior, valid,
        torch.zeros(2, 1, 1, 3, dtype=q.dtype), bias, mask,
    )
    zero_prior = torch.zeros_like(prior)
    native_only = ref.c_pre(q, nk, nv, sk, sv, zero_prior, zero_prior > 0, nonlinear_fuser, bias, mask)
    torch.testing.assert_close(recovered, native_only.output, atol=ATOL64, rtol=RTOL64)
    assert torch.equal(child_p, torch.zeros_like(child_p))
    torch.testing.assert_close(native_p, native_only.native_probability, atol=ATOL64, rtol=RTOL64)
