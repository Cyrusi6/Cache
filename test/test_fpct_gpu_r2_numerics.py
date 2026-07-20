from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    canonical_log_prior,
    fpct_qwen_eager_attention_forward,
)
from rosetta.train.dataset_adapters import RosettaDataCollator


@pytest.mark.parametrize("cardinality", [2, 3, 4])
@pytest.mark.parametrize("extreme", [False, True])
def test_canonical_fp32_prior_invariants(cardinality: int, extreme: bool) -> None:
    prior = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    if extreme:
        values = torch.tensor(
            [1.0 - 3e-4] + [3e-4 / (cardinality - 1)] * (cardinality - 1)
        )
    else:
        values = torch.full((cardinality,), 1.0 / cardinality)
    prior[:, :, :cardinality] = values.to(torch.bfloat16)
    valid = prior > 0
    canonical, log_prior, legal = canonical_log_prior(prior, valid)
    assert canonical.dtype == torch.float32
    assert log_prior.dtype == torch.float32
    assert torch.equal(torch.isneginf(log_prior), ~legal)
    torch.testing.assert_close(
        torch.logsumexp(log_prior, dim=-1), torch.zeros(2, 3), atol=2e-7, rtol=0
    )
    torch.testing.assert_close(
        canonical.sum(dim=-1), torch.ones(2, 3), atol=2e-7, rtol=0
    )


def test_cpu_collator_certification_hashes_and_layout_hint() -> None:
    sections = [
        {
            "source_indices": torch.tensor([[[0, 1, -1, -1], [2, -1, -1, -1]]]),
            "source_weights": torch.tensor([[[0.4, 0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
            "fpct_sharer_mask": 1,
        },
        {
            "source_indices": torch.tensor([[[3, 4, 5, -1]]]),
            "source_weights": torch.tensor([[[0.2, 0.3, 0.5, 0.0]]]),
            "fpct_sharer_mask": 1,
        },
    ]
    RosettaDataCollator._certify_fpct_soft_alignment(
        sections, source_attention=torch.ones(1, 6), target_length=3
    )
    assert {section["fpct_prior_sha256"] for section in sections} == {
        sections[0]["fpct_prior_sha256"]
    }
    assert all(section["fpct_prior_certified"] is True for section in sections)
    assert all(section["fpct_max_slots_hint"] == 6 for section in sections)
    assert all(section["fpct_target_length_hint"] == 3 for section in sections)
    for section in sections:
        assert section["source_weights"].dtype == torch.float32


def test_cpu_certifier_rejects_duplicate_and_invalid_positive_mass() -> None:
    duplicate = [{
        "source_indices": torch.tensor([[[0, 0]]]),
        "source_weights": torch.tensor([[[0.5, 0.5]]]),
    }]
    with pytest.raises(ValueError, match="duplicate"):
        RosettaDataCollator._certify_fpct_soft_alignment(
            duplicate, source_attention=torch.ones(1, 2), target_length=1
        )
    invalid = [{
        "source_indices": torch.tensor([[[0, 3]]]),
        "source_weights": torch.tensor([[[0.5, 0.5]]]),
    }]
    with pytest.raises(ValueError, match="invalid/padded"):
        RosettaDataCollator._certify_fpct_soft_alignment(
            invalid, source_attention=torch.ones(1, 2), target_length=1
        )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_shared_qwen_adapter_uses_fp32_accumulation_and_gqa(groups: int) -> None:
    generator = torch.Generator().manual_seed(29)
    hkv = 2
    module = SimpleNamespace(num_key_value_groups=groups, training=False)
    query = torch.randn(1, hkv * groups, 3, 8, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, hkv, 4, 8, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, hkv, 4, 8, generator=generator, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 3, 4, dtype=torch.float32)
    output, probability = fpct_qwen_eager_attention_forward(
        module, query, key, value, mask, scaling=8 ** -0.5
    )
    assert output.shape == (1, 3, hkv * groups, 8)
    assert probability.shape == (1, hkv * groups, 3, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        probability.float().sum(dim=-1),
        torch.ones(1, hkv * groups, 3),
        atol=5e-3,
        rtol=0,
    )


def test_fpct_hot_paths_do_not_materialize_cuda_scalars() -> None:
    from rosetta.model.wrapper import RosettaModel
    from rosetta.model.fpct_attention import build_fpct_packed_layout, pack_fpct_memory

    for function in (
        RosettaModel._fpct_legal_prior,
        build_fpct_packed_layout,
        pack_fpct_memory,
    ):
        source = inspect.getsource(function)
        assert ".item(" not in source
        assert ".cpu(" not in source
        assert ".numpy(" not in source


def test_certified_sidecar_requires_fp32_prior() -> None:
    with pytest.raises(ValueError, match="float32"):
        FPCTSidecarSegment(
            0,
            torch.zeros(1, 1, 1, 2, 2),
            torch.zeros(1, 1, 1, 2, 2),
            torch.tensor([[[0.5, 0.5]]], dtype=torch.float64),
            torch.ones(1, 1, 2, dtype=torch.bool),
            max_slots_hint=2,
            source_length_hint=1,
            prior_sha256="0" * 64,
            certified=True,
        ).validate()
