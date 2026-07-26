from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from script.analysis.fpct_e1_mechanism_audit import (
    FLOAT32_ATOL,
    LAMBDA_GRID,
    OUTPUT_COLUMNS,
    GammaQueryAccumulator,
    causal_gold_log_probs,
    centered_candidates,
    centered_factorized_attention,
    classify_candidate_topology,
    collapsed_attention,
    main,
    prepare_rows,
    synthetic_rows,
    validate_and_derive_row,
    verify_artifacts,
    write_artifacts,
)


def test_centered_family_preserves_prior_mean_and_grid_endpoints() -> None:
    candidates = torch.tensor([[[[1.0, 2.0], [5.0, 6.0], [99.0, 99.0]]]], dtype=torch.float64)
    prior = torch.tensor([[[0.25, 0.75, 0.0]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True, False]]])
    expected_mean = torch.tensor([[[4.0, 5.0]]], dtype=torch.float64)
    for value in LAMBDA_GRID:
        transformed, mean = centered_candidates(candidates, prior, valid, value)
        torch.testing.assert_close(mean, expected_mean, atol=1e-12, rtol=0)
        weighted = (transformed * prior.unsqueeze(-1)).sum(dim=-2)
        torch.testing.assert_close(weighted, expected_mean, atol=1e-12, rtol=0)
    zero, _ = centered_candidates(candidates, prior, valid, 0.0)
    torch.testing.assert_close(zero[..., :2, :], expected_mean.unsqueeze(-2).expand_as(zero[..., :2, :]))
    one, _ = centered_candidates(candidates, prior, valid, 1.0)
    torch.testing.assert_close(one[..., :2, :], candidates[..., :2, :])
    two, _ = centered_candidates(candidates, prior, valid, 2.0)
    torch.testing.assert_close(two[..., 0, :], torch.tensor([[[[-2.0, -1.0]]]], dtype=torch.float64).squeeze(-2))


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float64, 1e-10, 1e-8), (torch.float32, 2e-5, 2e-5)])
def test_lambda_zero_is_cpost_global_softmax_oracle(dtype: torch.dtype, atol: float, rtol: float) -> None:
    generator = torch.Generator().manual_seed(20260726)
    query = torch.randn(2, 2, 3, 5, generator=generator, dtype=dtype)
    key = torch.randn(2, 2, 4, 3, 5, generator=generator, dtype=dtype)
    value = torch.randn(2, 2, 4, 3, 5, generator=generator, dtype=dtype)
    prior = torch.tensor(
        [
            [[0.7, 0.3, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.4, 0.6, 0.0]],
            [[0.6, 0.4, 0.0], [0.1, 0.2, 0.7], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        ],
        dtype=dtype,
    )
    valid = prior > 0
    mask = torch.zeros(2, 1, 3, 4, dtype=dtype)
    mask[:, :, 0, 3] = -torch.inf
    factorized, child_probability = centered_factorized_attention(query, key, value, prior, valid, 0.0, mask)
    collapsed, parent_probability = collapsed_attention(query, key, value, prior, valid, mask)
    torch.testing.assert_close(factorized, collapsed, atol=atol, rtol=rtol)
    torch.testing.assert_close(child_probability.sum(dim=-1), parent_probability, atol=atol, rtol=rtol)


def test_lambda_zero_is_cpost_with_native_atoms_in_one_global_denominator() -> None:
    generator = torch.Generator().manual_seed(2026072601)
    query = torch.randn(2, 2, 3, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 2, 3, 3, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 2, 3, 3, 4, generator=generator, dtype=torch.float64)
    prior = torch.tensor(
        [
            [[0.7, 0.3, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0]],
            [[0.6, 0.4, 0.0], [0.1, 0.2, 0.7], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    valid = prior > 0
    native_key = torch.randn(2, 2, 2, 4, generator=generator, dtype=torch.float64)
    native_value = torch.randn(2, 2, 2, 4, generator=generator, dtype=torch.float64)
    native_valid = torch.tensor([[True, False], [True, True]])
    native_mask = torch.zeros(2, 1, 3, 2, dtype=torch.float64)
    native_mask[:, :, 0, 1] = -torch.inf

    factorized, child_probability, factorized_native_probability = centered_factorized_attention(
        query,
        key,
        value,
        prior,
        valid,
        0.0,
        native_key=native_key,
        native_value=native_value,
        native_valid=native_valid,
        native_mask=native_mask,
        return_native_probability=True,
    )
    collapsed, parent_probability, collapsed_native_probability = collapsed_attention(
        query,
        key,
        value,
        prior,
        valid,
        native_key=native_key,
        native_value=native_value,
        native_valid=native_valid,
        native_mask=native_mask,
        return_native_probability=True,
    )
    torch.testing.assert_close(factorized, collapsed, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(child_probability.sum(dim=-1), parent_probability, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(factorized_native_probability, collapsed_native_probability, atol=1e-10, rtol=1e-8)
    total = child_probability.sum(dim=(-1, -2)) + factorized_native_probability.sum(dim=-1)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-12, rtol=1e-10)
    assert bool((child_probability.sum(dim=(-1, -2)) < 1).all())
    assert torch.equal(
        factorized_native_probability[0, ..., 1],
        torch.zeros_like(factorized_native_probability[0, ..., 1]),
    )


def test_centered_oracle_permutation_and_invalid_gradient_exact_zero() -> None:
    query = torch.tensor([[[[0.5, -0.25]]]], dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0], [300.0, -500.0]]]]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[[[[2.0, 0.0], [0.0, 4.0], [700.0, 900.0]]]]], dtype=torch.float64, requires_grad=True)
    prior = torch.tensor([[[0.7, 0.3, 0.0]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True, False]]])
    output, probability = centered_factorized_attention(query, key, value, prior, valid, 1.0)
    assert torch.equal(probability[..., 2], torch.zeros_like(probability[..., 2]))
    output.sum().backward()
    assert torch.equal(key.grad[..., 2, :], torch.zeros_like(key.grad[..., 2, :]))
    assert torch.equal(value.grad[..., 2, :], torch.zeros_like(value.grad[..., 2, :]))

    permutation = torch.tensor([1, 0, 2])
    permuted_output, _ = centered_factorized_attention(
        query.detach(), key.detach().index_select(-2, permutation), value.detach().index_select(-2, permutation),
        prior.index_select(-1, permutation), valid.index_select(-1, permutation), 1.0,
    )
    torch.testing.assert_close(permuted_output, output.detach(), atol=1e-12, rtol=1e-10)


def test_centered_oracle_duplicate_refinement_invariance() -> None:
    query = torch.tensor([[[[0.2, -0.7], [0.8, 0.1]]]], dtype=torch.float64)
    key = torch.tensor([[[[[1.0, 0.5], [-0.5, 2.0]]]]], dtype=torch.float64)
    value = torch.tensor([[[[[2.0, -1.0], [0.25, 3.0]]]]], dtype=torch.float64)
    prior = torch.tensor([[[0.6, 0.4]]], dtype=torch.float64)
    valid = torch.ones_like(prior, dtype=torch.bool)
    refined_key = torch.cat((key[..., :1, :], key[..., :1, :], key[..., 1:, :]), dim=-2)
    refined_value = torch.cat((value[..., :1, :], value[..., :1, :], value[..., 1:, :]), dim=-2)
    refined_prior = torch.tensor([[[0.2, 0.4, 0.4]]], dtype=torch.float64)
    refined_valid = torch.ones_like(refined_prior, dtype=torch.bool)
    for lambda_value in LAMBDA_GRID:
        original, _ = centered_factorized_attention(query, key, value, prior, valid, lambda_value)
        refined, _ = centered_factorized_attention(
            query, refined_key, refined_value, refined_prior, refined_valid, lambda_value
        )
        torch.testing.assert_close(refined, original, atol=1e-12, rtol=1e-10)


def test_causal_gold_logp_scores_next_token_and_honors_ignore_mask() -> None:
    logits = torch.full((1, 5, 7), -20.0)
    labels = torch.tensor([[-100, -100, 3, 4, -100]])
    logits[0, 1, 3] = 20.0  # Query position 1 predicts target position 2.
    logits[0, 2, 4] = 20.0  # Query position 2 predicts target position 3.
    logits[0, 3, 0] = 20.0  # Ignored target position 4 must not appear.
    result = causal_gold_log_probs(logits, labels)
    assert result["query_position"].tolist() == [1, 2]
    assert result["target_position"].tolist() == [2, 3]
    assert result["target_token_id"].tolist() == [3, 4]
    assert torch.all(result["gold_logp"] > -1e-6)


def test_query_variance_accumulates_across_decode_steps_without_overwrite() -> None:
    accumulator = GammaQueryAccumulator(2)
    accumulator.update([0.9, 0.1])
    accumulator.update([0.1, 0.9])
    assert accumulator.count == 2
    assert accumulator.mean_candidate_variance > 0
    assert accumulator.top1_changed
    variance_after_two = accumulator.mean_candidate_variance
    accumulator.update([0.5, 0.5])
    assert accumulator.count == 3
    assert accumulator.mean_candidate_variance != variance_after_two


def test_topology_taxonomy_is_mutually_exclusive() -> None:
    assert classify_candidate_topology((0, 4), [(0, 2)]) == "not_applicable"
    assert classify_candidate_topology((0, 4), [(0, 2), (2, 4)]) == "partition_compositional"
    assert classify_candidate_topology((0, 4), [(0, 3), (1, 4)]) == "taxonomy_unresolved"
    assert classify_candidate_topology(
        (0, 4), [(0, 3), (1, 4)], independent_competitors=True
    ) == "competing_overlap"
    assert classify_candidate_topology(
        (0, 4), [(0, 3), (1, 4)], candidate_origins=["independent_overlap", "independent_overlap"]
    ) == "competing_overlap"
    assert classify_candidate_topology((0, 4), [(0, 1), (3, 4)]) == "taxonomy_unresolved"
    assert classify_candidate_topology((0, 4), [(0, 2), (2, 4)], boundary_or_fallback=True) == "boundary_fallback"
    assert classify_candidate_topology(
        (0, 4), [(0, 2), (2, 4)], candidate_origins=["span_overlap", "window_neighbor"]
    ) == "neighbor_expansion"


def test_row_firewall_causal_contract_and_lambda_zero_gate() -> None:
    row = synthetic_rows()[0]
    derived = validate_and_derive_row(row)
    assert derived["delta_gold_logp"] == 0
    assert derived["output_delta_l2"] == 0

    pilot = copy.deepcopy(row)
    pilot["split_role"] = "e1_pilot"
    with pytest.raises(ValueError, match="data firewall"):
        validate_and_derive_row(pilot)

    shifted = copy.deepcopy(row)
    shifted["target_position"] += 1
    with pytest.raises(ValueError, match="causal shift"):
        validate_and_derive_row(shifted)

    broken_zero = copy.deepcopy(row)
    broken_zero["gold_logp"] += 10 * FLOAT32_ATOL
    with pytest.raises(ValueError, match="lambda=0"):
        validate_and_derive_row(broken_zero)


def test_prepare_rows_computes_answer_query_variance() -> None:
    rows = [row for row in synthetic_rows() if row["cell"] == "Y_CF" and row["lambda_value"] == 1.0]
    prepared = prepare_rows(rows)
    assert len(prepared) == 2
    assert all(row["gamma_query_variance"] > 0 for row in prepared)
    assert all(row["posterior_top1_changed"] for row in prepared)


def test_dry_run_artifacts_reproduce_and_have_frozen_schema(tmp_path: Path) -> None:
    summary = write_artifacts(synthetic_rows(), tmp_path)
    assert summary["integrity"]["only_e0_design_consumed"] is True
    table = pq.read_table(tmp_path / "e1_mechanism_rows.parquet")
    assert tuple(table.column_names) == OUTPUT_COLUMNS
    verification = verify_artifacts(tmp_path)
    assert verification["status"] == "GO"
    assert verification["e1_pilot_consumed"] is False


def test_cli_dry_run_then_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["dry-run", "--output-dir", str(tmp_path)]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "COMPLETE"
    assert main(["verify", "--output-dir", str(tmp_path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "GO"
