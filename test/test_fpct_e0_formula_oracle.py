from __future__ import annotations

import pytest
import torch

from script.analysis.fpct_e0_formula_oracle import (
    canonical_prior,
    global_factorized_attention,
    run_parity,
)


def test_canonical_prior_rejects_illegal_mass() -> None:
    with pytest.raises(ValueError, match="illegal FPCT prior"):
        canonical_prior(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[True, False]]]))
    with pytest.raises(ValueError, match="illegal FPCT prior"):
        canonical_prior(torch.tensor([[[1.0, -0.1]]]), torch.ones(1, 1, 2, dtype=torch.bool))


def test_global_attention_invalid_atom_is_zero() -> None:
    q = torch.ones(1, 1, 1, 2)
    parent_k = torch.ones(1, 1, 1, 2)
    parent_v = torch.ones(1, 1, 1, 2)
    cand_k = torch.ones(1, 1, 1, 2, 2)
    cand_v = torch.ones(1, 1, 1, 2, 2)
    prior = torch.tensor([[[1.0, 0.0]]])
    valid = torch.tensor([[[True, False]]])
    output, probability = global_factorized_attention(
        q, parent_k, parent_v, cand_k, cand_v, prior, valid, torch.zeros(1, 1, 1, 1)
    )
    assert torch.equal(probability[..., 1], torch.zeros_like(probability[..., 1]))
    torch.testing.assert_close(output, torch.ones_like(output))


def test_full_production_parity_contract() -> None:
    result = run_parity()
    assert result["status"] == "GO"
    assert all(value is True or isinstance(value, int) for value in result["checks"].values())
    assert result["maximum_fp32_error"] <= 1e-6
