from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from script.analysis.fpct_e1_mechanism_audit import (
    FLOAT32_ATOL,
    LAMBDA_GRID,
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    GammaQueryAccumulator,
    _csv_bytes,
    build_summary,
    causal_gold_log_probs,
    centered_candidates,
    centered_factorized_attention,
    classify_candidate_topology,
    collapsed_attention,
    contraction_records,
    lambda_records,
    layer_head_records,
    main,
    prepare_rows,
    raw_topology_aggregate_rows,
    read_raw_topology_sidecar,
    read_input_rows,
    synthetic_rows,
    sha256_file,
    topology_records,
    validate_and_derive_row,
    validate_raw_topology_row,
    verify_artifacts,
    verify_raw_topology_artifacts,
    write_artifacts,
    write_raw_topology_artifacts,
)
from script.experiment.fpct_e1_prepare_input_lock import raw_topology_ledger


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


def test_finite_float32_min_mask_has_exact_zero_probability() -> None:
    """Qwen uses a finite sentinel, which must still mean hard-masked."""

    query = torch.tensor([[[[0.25, -0.5]]]], dtype=torch.float32)
    key = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], dtype=torch.float32)
    value = torch.tensor([[[[[2.0, 0.0], [0.0, 3.0]]]]], dtype=torch.float32)
    prior = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    parent_mask = torch.full(
        (1, 1, 1, 1), torch.finfo(torch.float32).min, dtype=torch.float32
    )
    native_key = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
    native_value = torch.tensor([[[[4.0, 5.0]]]], dtype=torch.float32)
    native_valid = torch.ones(1, 1, dtype=torch.bool)

    output, child_probability, native_probability = centered_factorized_attention(
        query,
        key,
        value,
        prior,
        valid,
        1.0,
        parent_mask,
        native_key=native_key,
        native_value=native_value,
        native_valid=native_valid,
        return_native_probability=True,
    )
    assert torch.equal(child_probability, torch.zeros_like(child_probability))
    assert torch.equal(native_probability, torch.ones_like(native_probability))
    torch.testing.assert_close(output, native_value, atol=0, rtol=0)


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
    assert classify_candidate_topology(
        (0, 4), [(0, 2), (2, 4)], certified_partition=True
    ) == "partition_compositional"
    assert classify_candidate_topology(
        (0, 4), [(0, 2), (2, 4)]
    ) == "taxonomy_unresolved"
    assert classify_candidate_topology((0, 4), [(0, 3), (1, 4)]) == "taxonomy_unresolved"
    assert classify_candidate_topology(
        (0, 4), [(0, 3), (1, 4)], independent_competitors=True
    ) == "taxonomy_unresolved"
    assert classify_candidate_topology(
        (0, 4), [(0, 4), (0, 5)], independent_competitors=True
    ) == "competing_overlap"
    assert classify_candidate_topology(
        (0, 4), [(0, 4), (0, 4)], independent_competitors=True
    ) == "taxonomy_unresolved"
    assert classify_candidate_topology(
        (0, 4),
        [(0, 4), (0, 5)],
        independent_competitors=True,
        duplicate_or_overlap_alias=True,
    ) == "boundary_fallback"
    assert classify_candidate_topology((0, 4), [(0, 1), (3, 4)]) == "taxonomy_unresolved"
    assert classify_candidate_topology((0, 4), [(0, 2), (2, 4)], boundary_or_fallback=True) == "boundary_fallback"
    assert classify_candidate_topology(
        (0, 4), [(0, 2), (2, 4)], candidate_origins=["span_overlap", "window_neighbor"]
    ) == "neighbor_expansion"
    assert classify_candidate_topology(
        (0, 4), [(2, 4), (0, 2)], certified_partition=True
    ) == "partition_compositional"


def test_certified_partition_topology_is_candidate_permutation_invariant() -> None:
    source_spans = [(0, 1), (1, 3), (3, 4)]
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    for permutation in permutations:
        permuted_spans = [source_spans[index] for index in permutation]
        assert classify_candidate_topology(
            (0, 4), permuted_spans, certified_partition=True
        ) == "partition_compositional"
        assert classify_candidate_topology(
            (0, 4), permuted_spans
        ) == "taxonomy_unresolved"


def test_raw_topology_ledger_preserves_permuted_certified_candidate_slots() -> None:
    raw = {
        "message_mask": [True],
        "slm_ids": [101],
        "llm_ids": [201, 202],
        "slm_offsets": [(0, 4)],
        "llm_offsets": [(0, 2), (2, 4)],
        "content_spans_slm": [(0, 4)],
        "content_spans_llm": [(0, 4)],
        "sections": [
            {"type": "message", "slm_range": (0, 1), "llm_range": (0, 2)}
        ],
        "soft_alignment": {
            "source_indices": [[1, 0]],
            "source_weights": [[0.7, 0.3]],
        },
    }
    sanitized = {
        "soft_alignment": {
            "source_indices": [[1, 0]],
            "source_weights": [[0.7, 0.3]],
            "fpct_certified_mask": [True],
            "fpct_certification_reason": ["certified_disjoint_partition"],
        }
    }

    rows = raw_topology_ledger(
        raw_details=raw,
        sanitized_details=sanitized,
        instruction_end=1,
        task="ai2-arc",
        sample_sha256="1" * 64,
        content_group_sha256="2" * 64,
        candidate_window=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["taxonomy"] == "partition_compositional"
    assert row["raw_candidate_indices"] == [1, 0]
    assert row["runtime_candidate_indices"] == [1, 0]
    assert row["raw_weights"] == [0.7, 0.3]
    assert row["runtime_weights"] == [0.7, 0.3]
    assert [candidate["slot"] for candidate in row["candidates"]] == [0, 1]
    assert [candidate["source_index"] for candidate in row["candidates"]] == [1, 0]
    assert [candidate["source_span"] for candidate in row["candidates"]] == [
        [2, 4],
        [0, 2],
    ]
    assert [candidate["intersection"] for candidate in row["candidates"]] == [
        [2, 4],
        [0, 2],
    ]
    assert [candidate["raw_weight"] for candidate in row["candidates"]] == [
        0.7,
        0.3,
    ]
    assert [candidate["runtime_weight"] for candidate in row["candidates"]] == [
        0.7,
        0.3,
    ]
    assert validate_raw_topology_row(row) == row


def test_raw_topology_sidecar_export_is_separate_hashed_and_reproducible(
    tmp_path: Path,
) -> None:
    raw = {
        "message_mask": [True, True],
        "slm_ids": [101, 102],
        "llm_ids": [201, 202, 203, 203],
        "slm_offsets": [(0, 2), (2, 4)],
        "llm_offsets": [(0, 1), (1, 2), (2, 3), (2, 3)],
        "content_spans_slm": [(0, 4)],
        "content_spans_llm": [(0, 4)],
        "sections": [
            {
                "type": "message",
                "slm_range": (0, 2),
                "llm_range": (0, 4),
            }
        ],
        "soft_alignment": {
            "source_indices": [[0, 1], [2, 3]],
            "source_weights": [[0.5, 0.5], [0.5, 0.5]],
        },
    }
    sanitized = {
        "soft_alignment": {
            "source_indices": [[0, 1], [2, -1]],
            "source_weights": [[0.5, 0.5], [1.0, 0.0]],
            "fpct_certified_mask": [True, False],
            "fpct_certification_reason": [
                "certified_disjoint_partition",
                "exact_duplicate_source_offsets",
            ],
        }
    }
    rows = raw_topology_ledger(
        raw_details=raw,
        sanitized_details=sanitized,
        instruction_end=2,
        task="ai2-arc",
        sample_sha256="1" * 64,
        content_group_sha256="2" * 64,
        candidate_window=0,
    )
    sidecar = tmp_path / "input_lock.pt"
    torch.save(
        {
            "protocol_id": "fpct_e1_e0_design_input_lock_v1",
            "split_role": "e0_design",
            "e1_pilot_consumed": False,
            "model_or_checkpoint_loaded": False,
            "items": [
                {
                    "task": "ai2-arc",
                    "sample_sha256": "1" * 64,
                    "content_group_sha256": "2" * 64,
                    "raw_topology_ledger": rows,
                }
            ],
        },
        sidecar,
    )
    observed = read_raw_topology_sidecar(sidecar)
    assert observed == rows
    aggregates = raw_topology_aggregate_rows(observed)
    assert {row["taxonomy"] for row in aggregates} == {
        "partition_compositional",
        "boundary_fallback",
    }
    assert sum(row["raw_minus_runtime_candidate_count"] for row in aggregates) == 1
    assert sum(row["raw_minus_runtime_extra_slot_count"] for row in aggregates) == 1
    uncertified = next(
        row for row in aggregates if row["taxonomy"] == "boundary_fallback"
    )
    assert (
        uncertified["functional_metric_contract"]
        == "UNAVAILABLE_UNCERTIFIED_NO_FUNCTIONAL_METRIC"
    )
    output_dir = tmp_path / "raw-audit"
    manifest = write_raw_topology_artifacts(sidecar, output_dir)
    assert manifest["status"] == "GO_MODEL_OUTPUT_FREE"
    assert manifest["functional_metrics_present"] is False
    verified = verify_raw_topology_artifacts(sidecar, output_dir)
    assert verified["status"] == "GO_MODEL_OUTPUT_FREE"
    assert verified["row_count"] == 2
    summary = json.loads(
        (output_dir / "e1_raw_topology_aggregates.json").read_text()
    )
    assert summary["task"]["ai2-arc"]["offset_uncertified_parent_count"] == 1
    assert summary["task"]["ai2-arc"]["raw_minus_runtime_extra_slot_count"] == 1
    assert "no functional metric" in summary["functional_metric_contract"]
    cli_dir = tmp_path / "raw-audit-cli"
    assert main(
        [
            "raw-topology",
            "--input-lock-sidecar",
            str(sidecar),
            "--output-dir",
            str(cli_dir),
        ]
    ) == 0
    assert main(
        [
            "verify-raw-topology",
            "--input-lock-sidecar",
            str(sidecar),
            "--output-dir",
            str(cli_dir),
        ]
    ) == 0


def test_raw_topology_preserves_zero_length_boundary_geometry() -> None:
    raw = {
        "message_mask": [True],
        "slm_ids": [101],
        "llm_ids": [201, 202],
        "slm_offsets": [(0, 0)],
        "llm_offsets": [(0, 1), (1, 2)],
        "content_spans_slm": [(0, 2)],
        "content_spans_llm": [(0, 2)],
        "sections": [
            {"type": "message", "slm_range": (0, 1), "llm_range": (0, 2)}
        ],
        "soft_alignment": {
            "source_indices": [[0, 1]],
            "source_weights": [[0.5, 0.5]],
        },
    }
    sanitized = {
        "soft_alignment": {
            "source_indices": [[0, -1]],
            "source_weights": [[1.0, 0.0]],
            "fpct_certified_mask": [False],
            "fpct_certification_reason": ["zero_length_receiver_interval"],
        }
    }
    rows = raw_topology_ledger(
        raw_details=raw,
        sanitized_details=sanitized,
        instruction_end=1,
        task="ai2-arc",
        sample_sha256="1" * 64,
        content_group_sha256="2" * 64,
        candidate_window=0,
    )
    assert len(rows) == 1
    assert rows[0]["receiver_span"] == [0, 0]
    assert rows[0]["taxonomy"] == "boundary_fallback"
    assert validate_raw_topology_row(rows[0])["offset_uncertified"] is True


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

    wrong_posterior = copy.deepcopy(row)
    wrong_posterior["gamma"] = [0.9, 0.1]
    with pytest.raises(ValueError, match="lambda=0"):
        validate_and_derive_row(wrong_posterior)

    wrong_task_outcome = copy.deepcopy(row)
    wrong_task_outcome["end_task_correct"] = not row["cpost_end_task_correct"]
    with pytest.raises(ValueError, match="lambda=0"):
        validate_and_derive_row(wrong_task_outcome)


def test_prepare_rows_computes_answer_query_variance() -> None:
    rows = [row for row in synthetic_rows() if row["cell"] == "Y_CF" and row["lambda_value"] == 1.0]
    prepared = prepare_rows(rows)
    assert len(prepared) == 2
    assert all(row["gamma_query_variance"] > 0 for row in prepared)
    assert all(row["posterior_top1_changed"] for row in prepared)


def test_response_logp_and_end_task_accuracy_are_not_parent_head_weighted(
    tmp_path: Path,
) -> None:
    base = [
        row
        for row in synthetic_rows()
        if row["cell"] == "Y_CF" and row["lambda_value"] == 1.0
    ]
    duplicated = []
    for row in base:
        duplicated.append(row)
        copy_row = copy.deepcopy(row)
        copy_row["parent_position"] += 1
        duplicated.append(copy_row)
    summary = write_artifacts(duplicated, tmp_path)
    primary = summary["primary_teacher_forced_gold_logp"]
    assert len(primary) == 1
    record = primary[0]
    assert record["answer_query_count"] == 2
    assert record["content_group_count"] == 1
    expected = sum(row["gold_logp"] - row["cpost_gold_logp"] for row in base)
    assert record["group_equal_mean_response_delta_gold_logp"] == pytest.approx(
        expected
    )
    assert record["group_equal_end_task_accuracy"] == 1.0


def test_bounded_streaming_matches_in_memory_numerical_semantics(
    tmp_path: Path,
) -> None:
    by_partition: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for original in synthetic_rows():
        key = (
            original["seed"],
            original["checkpoint_arm"],
            original["inference_operator"],
            original["cell"],
            original["task"],
            original["lambda_value"],
        )
        by_partition.setdefault(key, []).append(original)
    streamed: list[dict[str, object]] = []
    identities = sorted(
        (
            hashlib.sha256(f"semantic-sample-{ordinal}".encode()).hexdigest(),
            hashlib.sha256(f"semantic-group-{ordinal}".encode()).hexdigest(),
        )
        for ordinal in range(2)
    )
    for key in sorted(by_partition):
        for sample, group in identities:
            for original in by_partition[key]:
                row = copy.deepcopy(original)
                row["sample_sha256"] = sample
                row["content_group_sha256"] = group
                streamed.append(row)

    merged = tmp_path / "merged-stage.parquet"
    pq.write_table(
        pa.Table.from_pylist(streamed).select(list(INPUT_COLUMNS)), merged
    )
    output = tmp_path / "semantic-equivalence"
    observed_summary = write_artifacts(read_input_rows(merged), output)
    prepared = prepare_rows(streamed)
    expected_csv = {
        "e1_layer_head_summary.csv": layer_head_records(prepared),
        "e1_projector_contraction.csv": contraction_records(prepared),
        "e1_candidate_topology.csv": topology_records(prepared),
        "e1_centered_lambda_summary.csv": lambda_records(prepared),
    }
    for filename, records in expected_csv.items():
        assert (output / filename).read_bytes() == _csv_bytes(records)
    hashes = {
        "e1_mechanism_rows.parquet": sha256_file(
            output / "e1_mechanism_rows.parquet"
        ),
        **{
            filename: sha256_file(output / filename) for filename in expected_csv
        },
    }
    assert observed_summary == build_summary(prepared, hashes)


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


def test_functional_stage_is_bounded_beyond_one_parquet_batch(
    tmp_path: Path,
) -> None:
    base = synthetic_rows()[0]

    def rows() -> object:
        for parent_position in range(4097):
            row = copy.deepcopy(base)
            row["parent_position"] = parent_position
            yield row

    output = tmp_path / "bounded"
    summary = write_artifacts(rows(), output)
    assert summary["row_count"] == 4097
    parquet = pq.ParquetFile(output / "e1_mechanism_rows.parquet")
    assert parquet.metadata.num_row_groups == 2
    iterator = read_input_rows(output / "e1_mechanism_rows.parquet")
    assert iter(iterator) is iterator
    first = next(iterator)
    assert set(first) == set(base)
    verified = verify_artifacts(output)
    assert verified["row_count"] == 4097


def test_functional_analyzer_source_has_no_unbounded_reader_api() -> None:
    source = Path(
        "script/analysis/fpct_e1_mechanism_audit.py"
    ).read_text(encoding="utf-8")
    for forbidden in (".read_text(", "pq.read_table(", ".to_pylist("):
        assert forbidden not in source
