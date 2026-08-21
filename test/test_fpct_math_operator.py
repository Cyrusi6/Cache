from __future__ import annotations

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rosetta.model.fpct_position import (
    apply_fpct_rope,
    model_rotary_cos_sin,
    normalize_fpct_position_mode,
    remove_fpct_rope,
)
from rosetta.model.projector import C2CProjector
from rosetta.model.wrapper import RosettaModel


def _qwen(num_key_value_heads: int) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _projector() -> C2CProjector:
    projector = C2CProjector(
        source_dim=8,
        target_dim=8,
        source_num_heads=4,
        target_num_heads=2,
        intermediate_dim=64,
        hidden_dim=32,
        num_layers=3,
        dropout=0.0,
        dtype=torch.float32,
    )
    projector.legacy_scalar_gate_eval_mode = "forced_on"
    projector.eval()
    return projector


def _inputs(base: Qwen3ForCausalLM, source: Qwen3ForCausalLM):
    generator = torch.Generator().manual_seed(20260821)
    source_content_key = torch.randn(1, 4, 5, 8, generator=generator)
    source_value = torch.randn(1, 4, 5, 8, generator=generator)
    target_content_key = torch.randn(1, 2, 3, 8, generator=generator)
    target_value = torch.randn(1, 2, 3, 8, generator=generator)
    source_positions = torch.arange(5).unsqueeze(0)
    target_positions = torch.tensor([[7, 8, 9]])
    source_cosine, source_sine = model_rotary_cos_sin(
        source, source_content_key, source_positions
    )
    target_cosine, target_sine = model_rotary_cos_sin(
        base, target_content_key, target_positions
    )
    source_key = apply_fpct_rope(
        source_content_key, source_cosine, source_sine
    )
    target_key = apply_fpct_rope(
        target_content_key, target_cosine, target_sine
    )
    indices = torch.tensor([[[0, 1], [3, 4], [2, -1]]])
    weights = torch.tensor([[[0.75, 0.25], [0.4, 0.6], [1.0, 0.0]]])
    soft = {
        "source_indices": indices,
        "source_weights": weights,
        "source_confidence": torch.ones(1, 3),
        "source_entropy": torch.zeros(1, 3),
        "source_entropy_override": torch.zeros(1, 3, dtype=torch.bool),
        "fpct_prior_certified": True,
    }
    return {
        "source_content_key": source_content_key,
        "source_key": source_key,
        "source_value": source_value,
        "target_content_key": target_content_key,
        "target_key": target_key,
        "target_value": target_value,
        "target_positions": target_positions,
        "indices": indices,
        "weights": weights,
        "soft": soft,
        "target_cosine": target_cosine,
        "target_sine": target_sine,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_rope_round_trip_and_gradient(dtype: torch.dtype) -> None:
    model = _qwen(2).to(dtype=dtype)
    value = torch.randn(2, 2, 5, 8, dtype=dtype, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])
    cosine, sine = model_rotary_cos_sin(model, value, positions)
    recovered = remove_fpct_rope(
        apply_fpct_rope(value, cosine, sine), cosine, sine
    )
    # Transformers deliberately evaluates rotary frequencies in FP32 even when
    # the surrounding model is FP64, so the inverse is bounded by that source
    # precision rather than an artificial FP64 oracle.
    tolerance = 2e-5 if dtype == torch.float32 else 2e-6
    torch.testing.assert_close(recovered, value, atol=tolerance, rtol=tolerance)
    recovered.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_math_candidate_path_is_content_fusion_then_parent_rope() -> None:
    torch.manual_seed(31)
    base = _qwen(2)
    source = _qwen(4)
    projector = _projector()
    wrapper = RosettaModel(
        [base, source],
        projector_list=[projector],
        fpct_operator="f",
        fpct_position_mode="math",
        fpct_trace=True,
    )
    data = _inputs(base, source)
    record = wrapper._project_fpct_candidates(
        projector=projector,
        source_model_idx=1,
        source_key_cache=data["source_key"],
        source_value_cache=data["source_value"],
        base_kv=(data["target_key"], data["target_value"]),
        source_indices=data["indices"],
        source_weights=data["weights"],
        soft_section=data["soft"],
        target_position_ids=data["target_positions"],
        target_layer_idx=0,
    )
    trace = wrapper._fpct_candidate_trace_tensors[0][0]
    safe = data["indices"].clamp_min(0)
    gather = safe[:, None, :, :, None].expand(1, 4, 3, 2, 8)
    expected_source_content = torch.gather(
        data["source_content_key"][:, :, :, None, :].expand(-1, -1, -1, 2, -1),
        2,
        gather,
    )
    expected_source_content = torch.where(
        (data["indices"] >= 0)[:, None, :, :, None],
        expected_source_content,
        torch.zeros_like(expected_source_content),
    )
    expected_source_value = torch.gather(
        data["source_value"][:, :, :, None, :].expand(-1, -1, -1, 2, -1),
        2,
        gather,
    )
    expected_source_value = torch.where(
        (data["indices"] >= 0)[:, None, :, :, None],
        expected_source_value,
        torch.zeros_like(expected_source_value),
    )
    torch.testing.assert_close(
        trace["source_candidate_key"], expected_source_content, atol=2e-5, rtol=2e-5
    )

    candidate_cosine = data["target_cosine"].unsqueeze(2).expand(1, 3, 2, 8)
    candidate_sine = data["target_sine"].unsqueeze(2).expand_as(candidate_cosine)
    fused_content_key = remove_fpct_rope(
        record[2], candidate_cosine, candidate_sine
    )
    nuisance = record[7]
    for candidate in range(2):
        expected_key, expected_value = projector.forward(
                (
                    expected_source_content[:, :, :, candidate, :],
                    expected_source_value[:, :, :, candidate, :],
                ),
            (data["target_content_key"], data["target_value"]),
            fpct_parent_nuisance=nuisance,
        )
        torch.testing.assert_close(
            fused_content_key[:, :, :, candidate, :],
            expected_key,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            record[3][:, :, :, candidate, :],
            expected_value,
            atol=2e-5,
            rtol=2e-5,
        )


def test_math_cpost_and_f_share_identical_candidates_and_state_dict() -> None:
    torch.manual_seed(47)
    base = _qwen(2)
    source = _qwen(4)
    projector = _projector()
    projector_state = copy.deepcopy(projector.state_dict())
    data = _inputs(base, source)
    records = []
    keys = []
    for operator in ("c_post", "f"):
        current_projector = _projector()
        current_projector.load_state_dict(projector_state)
        wrapper = RosettaModel(
            [_qwen(2), _qwen(4)],
            projector_list=[current_projector],
            fpct_operator=operator,
            fpct_position_mode="math",
        )
        records.append(
            wrapper._project_fpct_candidates(
                projector=current_projector,
                source_model_idx=1,
                source_key_cache=data["source_key"],
                source_value_cache=data["source_value"],
                base_kv=(data["target_key"], data["target_value"]),
                source_indices=data["indices"],
                source_weights=data["weights"],
                soft_section=data["soft"],
                target_position_ids=data["target_positions"],
            )
        )
        keys.append(set(wrapper.state_dict()))
    torch.testing.assert_close(records[0][2], records[1][2], atol=0.0, rtol=0.0)
    torch.testing.assert_close(records[0][3], records[1][3], atol=0.0, rtol=0.0)
    assert keys[0] == keys[1]


def test_position_mode_validation_and_config_contract() -> None:
    assert normalize_fpct_position_mode(None) == "legacy"
    assert normalize_fpct_position_mode("MATH") == "math"
    with pytest.raises(ValueError):
        normalize_fpct_position_mode("approximate")
    with pytest.raises(ValueError):
        RosettaModel([_qwen(2)], fpct_operator=None, fpct_position_mode="math")
    wrapper = RosettaModel(
        [_qwen(2)], fpct_operator="f", fpct_position_mode="math"
    )
    assert wrapper.fpct_config_dict()["position_mode"] == "math"


def test_actual_random_qwen_math_prefill_response_and_projector_gradient() -> None:
    torch.manual_seed(83)
    base = _qwen(2)
    source = _qwen(4)
    projector = _projector()
    wrapper = RosettaModel(
        [base, source],
        projector_list=[projector],
        fpct_operator="f",
        fpct_position_mode="math",
    )
    wrapper.set_projector_config(1, 0, 0, 0, 0)
    first_indices = torch.tensor([[[0, 1], [2, -1], [3, 4]]])
    first_weights = torch.tensor([[[0.6, 0.4], [1.0, 0.0], [0.25, 0.75]]])

    def section(indices, weights, sharer):
        length = indices.shape[1]
        return {
            "source_indices": indices,
            "source_weights": weights,
            "source_confidence": torch.ones(1, length),
            "source_entropy": torch.zeros(1, length),
            "source_entropy_override": torch.zeros(1, length, dtype=torch.bool),
            "fpct_sharer_mask": sharer,
            "fpct_prior_certified": True,
            "fpct_prior_sha256": "a" * 64,
            "fpct_max_slots_hint": 6,
            "fpct_target_length_hint": 4,
            "fpct_source_lengths": (5,),
        }

    output = wrapper(
        input_ids=[torch.tensor([[4, 5, 6, 7]]), torch.tensor([[8, 9, 10, 11, 12]])],
        attention_mask=[torch.ones(1, 4), torch.ones(1, 5)],
        position_ids=torch.arange(4).unsqueeze(0),
        kv_cache_index=[torch.zeros(1, 3, 2, dtype=torch.long), torch.zeros(1, 1, 2, dtype=torch.long)],
        soft_alignment=[
            section(first_indices, first_weights, 1),
            section(torch.full((1, 1, 2), -1), torch.zeros(1, 1, 2), -1),
        ],
        labels=torch.tensor([[-100, -100, -100, 7]]),
        use_cache=True,
    )
    assert output.logits.shape == (1, 1, 64)
    assert torch.isfinite(output.logits).all()
    output.logits.float().sum().backward()
    projector_gradients = [
        parameter.grad
        for parameter in projector.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert projector_gradients
    assert all(torch.isfinite(gradient).all() for gradient in projector_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in projector_gradients) > 0
