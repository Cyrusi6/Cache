from __future__ import annotations

from types import SimpleNamespace

import torch

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    fpct_mechanism_diagnostics,
    pack_fpct_memory,
)
from rosetta.model.fpct_instrumentation import FPCTCaptureAccumulator
from script.experiment.fpct_e1_fast_rca import (
    TASK_COUNTS,
    TASKS,
    VARIANTS,
    overlap_composition_weights,
    read_manifest,
    git_head,
    _move_to_device,
    _active_section_labels,
)
from script.analysis import fpct_e1_fast_rca_analysis as fast_analysis


def _packed_case():
    generator = torch.Generator().manual_seed(20260820)
    query = torch.randn(1, 4, 3, 4, generator=generator)
    candidate_key = torch.randn(1, 2, 3, 3, 4, generator=generator)
    candidate_value = torch.randn(1, 2, 3, 3, 4, generator=generator)
    prior = torch.tensor(
        [[[0.6, 0.4, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]]
    )
    valid = prior > 0
    weight = prior[:, None, :, :, None]
    parent_key = (candidate_key * weight).sum(dim=3)
    parent_value = (candidate_value * weight).sum(dim=3)
    mask = torch.zeros(1, 1, 3, 3)
    packed = pack_fpct_memory(
        parent_key,
        parent_value,
        mask,
        [FPCTSidecarSegment(0, candidate_key, candidate_value, prior, valid)],
        query_length=3,
    )
    return query, parent_key, parent_value, mask, packed


def test_summary_only_capture_never_materializes_expanded_rows() -> None:
    query, _parent_key, _parent_value, _mask, packed = _packed_case()
    capture = FPCTCaptureAccumulator(
        "teacher_forced_response",
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        detail_mode="summary_only",
    )
    metrics, payload = fpct_mechanism_diagnostics(
        query, packed, return_capture_payload=True
    )
    payload["query_metrics"] = {
        "output_delta_l2": torch.ones(1, 4, 3),
    }
    capture.update(0, metrics, payload)
    report = capture.finalize()
    assert report["long_form_row_count"] == 0
    assert "long_form_primitives" not in report
    assert "fused_d_k" in report["metrics"]
    assert "fused_d_v" in report["metrics"]
    assert report["layers"]["0"]["heads"]


def test_parent_mass_intervention_preserves_each_cpost_parent_probability() -> None:
    query, parent_key, parent_value, mask, packed = _packed_case()
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    output, atom_probability = fpct_qwen_hierarchical_attention_forward(
        module,
        query,
        packed,
        parent_key,
        parent_value,
        mask,
        scaling=0.5,
        preserve_parent_mass=True,
    )
    _parent_output, parent_probability = fpct_qwen_eager_attention_forward(
        module, query, parent_key, parent_value, mask, scaling=0.5
    )
    parent_index = packed.parent_index[:, None, None, :].expand_as(atom_probability)
    grouped = torch.zeros_like(parent_probability).scatter_add_(
        3, parent_index, atom_probability
    )
    torch.testing.assert_close(grouped, parent_probability, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(output).all()
    invalid = ~packed.active[:, None, None, :]
    assert torch.equal(
        torch.where(invalid, atom_probability, torch.zeros_like(atom_probability)),
        torch.zeros_like(atom_probability),
    )


def test_overlap_composition_uses_intersection_length_without_reordering() -> None:
    indices = torch.tensor([[10, 11, -1, -1], [12, -1, -1, -1]])
    weights = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    item = {
        "feature": {
            "soft_alignment": {
                "source_indices": indices,
                "source_weights": weights,
            }
        },
        "raw_topology_ledger": [
            {
                "parent_position": 0,
                "taxonomy": "partition_compositional",
                "certified": True,
                "candidates": [
                    {
                        "source_index": 10,
                        "intersection_length": 1,
                        "runtime_retained": True,
                    },
                    {
                        "source_index": 11,
                        "intersection_length": 3,
                        "runtime_retained": True,
                    },
                ],
            }
        ],
    }
    actual = overlap_composition_weights(item)
    assert torch.equal(actual["soft_alignment"]["source_indices"], indices)
    torch.testing.assert_close(
        actual["soft_alignment"]["source_weights"][0],
        torch.tensor([0.25, 0.75, 0.0, 0.0]),
    )
    assert torch.equal(item["feature"]["soft_alignment"]["source_weights"], weights)


def test_fast_manifest_freezes_population_interventions_and_firewall() -> None:
    manifest = read_manifest(__import__("pathlib").Path(__file__).parents[1])
    assert tuple(manifest["tasks"]["order"]) == TASKS
    assert manifest["tasks"]["counts"] == TASK_COUNTS
    assert tuple(manifest["interventions"]["execution_order"]) == VARIANTS
    assert manifest["interventions"]["rope_frame_correction"]["selectable"] is False
    assert manifest["capture"]["expanded_long_form_rows_materialized"] is False
    assert manifest["firewall"]["e1_pilot"] == "SEALED_NOT_RUN_NOT_READ"


def test_hierarchical_bootstrap_treats_three_seeds_as_top_level(monkeypatch) -> None:
    rows = []
    for seed in (1, 2, 3):
        for arm in ("c_post_trained", "f_trained"):
            for task in TASKS:
                for group in range(4):
                    rows.append(
                        {
                            "seed": seed,
                            "checkpoint_arm": arm,
                            "task": task,
                            "content_group_sha256": f"{task}-{group}",
                            "positive": 0.1 + 0.01 * seed,
                        }
                    )
    monkeypatch.setattr(fast_analysis, "BOOTSTRAP_REPLICATES", 200)
    lower, center, upper = fast_analysis.hierarchical_lcb(rows, "positive")
    assert 0 < lower <= center <= upper


def test_container_execution_sha_fallback_is_strict(monkeypatch, tmp_path) -> None:
    import script.experiment.fpct_e1_fast_rca as runner

    monkeypatch.setenv("FPCT_EXECUTION_SHA", "a" * 40)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert git_head(tmp_path) == "a" * 40
    monkeypatch.setenv("FPCT_EXECUTION_SHA", "not-a-sha")
    try:
        git_head(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid injected execution SHA was accepted")


def test_recursive_device_move_preserves_message_metadata() -> None:
    value = {
        "input_ids": [torch.tensor([1, 2])],
        "messages": [[{"role": "user", "content": "x"}]],
    }
    actual = _move_to_device(value, torch.device("cpu"))
    assert actual["input_ids"][0].device.type == "cpu"
    assert actual["messages"] == value["messages"]


def test_active_section_labels_match_wrapper_returned_logits() -> None:
    labels = torch.full((1, 12), -100, dtype=torch.long)
    labels[:, 9:] = torch.tensor([4, 5, 6])
    sections = [
        torch.zeros(1, 3, 2, dtype=torch.long),
        torch.zeros(1, 4, 2, dtype=torch.long),
        torch.zeros(1, 5, 2, dtype=torch.long),
    ]
    active = _active_section_labels(labels, sections)
    assert active.shape == (1, 5)
    assert active.tolist() == [[-100, -100, 4, 5, 6]]
