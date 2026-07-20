from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.wrapper import RosettaModel


def _config(num_key_value_heads: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    return config


def _model(num_key_value_heads: int, state=None) -> Qwen3ForCausalLM:
    torch.manual_seed(1701)
    model = Qwen3ForCausalLM(_config(num_key_value_heads))
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _sidecar(
    wrapper: RosettaModel,
    *,
    batch_size: int,
    source_length: int,
    num_key_value_heads: int,
    ambiguous: bool,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(719)
    key = torch.randn(
        batch_size,
        num_key_value_heads,
        source_length,
        4,
        8,
        generator=generator,
        requires_grad=requires_grad,
    )
    value = torch.randn(
        batch_size,
        num_key_value_heads,
        source_length,
        4,
        8,
        generator=generator,
        requires_grad=requires_grad,
    )
    prior = torch.zeros(batch_size, source_length, 4)
    prior[..., 0] = 1.0
    if ambiguous:
        prior[:, 0] = torch.tensor([0.6, 0.4, 0.0, 0.0])
        prior[:, 2] = 0.0  # explicit m=0 native fallback row
        prior[1, 1] = torch.tensor([0.0, 0.7, 0.3, 0.0])
    valid = prior > 0
    wrapper._store_fpct_sidecar(0, 0, key, value, prior, valid)
    return key, value


def _forward(
    wrapper: RosettaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: DynamicCache | None = None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values or DynamicCache(),
        use_cache=True,
        return_dict=True,
    )


@pytest.mark.parametrize("num_key_value_heads", [1, 2])
def test_actual_qwen3_eager_prefill_decode_padding_gqa_mqa_and_gradients(
    num_key_value_heads: int,
) -> None:
    model = _model(num_key_value_heads)
    wrapper = RosettaModel([model], fpct_operator="f")
    key, value = _sidecar(
        wrapper,
        batch_size=2,
        source_length=4,
        num_key_value_heads=num_key_value_heads,
        ambiguous=True,
        requires_grad=True,
    )
    input_ids = torch.tensor([[4, 5, 6, 7], [0, 0, 8, 9]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    output = _forward(wrapper, input_ids, attention_mask)
    assert output.logits.shape == (2, 4, 64)
    assert torch.isfinite(output.logits).all()
    output.logits.float().sum().backward()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert key.grad.abs().sum() > 0
    assert value.grad.abs().sum() > 0

    with torch.no_grad():
        decode = _forward(
            wrapper,
            torch.tensor([[10], [11]]),
            torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
            output.past_key_values,
        )
    assert decode.logits.shape == (2, 1, 64)
    assert decode.past_key_values.get_seq_length() == 5
    assert torch.isfinite(decode.logits).all()
    assert wrapper._fpct_packed_layout is not None
    assert wrapper._fpct_packed_layout.source_length == 5


@pytest.mark.parametrize("operator", ["c_pre", "c_post", "f"])
def test_actual_qwen3_runtime_operator_switch_and_config_roundtrip(
    operator: str, tmp_path: Path
) -> None:
    model = _model(2)
    wrapper = RosettaModel([model], fpct_operator=operator)
    path = tmp_path / f"{operator}.json"
    path.write_text(json.dumps(wrapper.fpct_config_dict(), sort_keys=True) + "\n")
    assert json.loads(path.read_text())["operator"] == operator
    output = _forward(
        wrapper,
        torch.tensor([[4, 5, 6]]),
        torch.ones(1, 3, dtype=torch.long),
    )
    assert torch.isfinite(output.logits).all()


def test_actual_qwen3_m1_and_replicated_collapse_equal_cpost() -> None:
    base = _model(2)
    state = base.state_dict()
    input_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    cpost = RosettaModel([_model(2, state)], fpct_operator="c_post")
    cpost_output = _forward(cpost, input_ids, attention_mask).logits

    m1 = RosettaModel([_model(2, state)], fpct_operator="f")
    _sidecar(
        m1,
        batch_size=2,
        source_length=4,
        num_key_value_heads=2,
        ambiguous=False,
    )
    m1_output = _forward(m1, input_ids, attention_mask).logits
    torch.testing.assert_close(m1_output, cpost_output, atol=0, rtol=0)
    assert m1._fpct_packed_layout is not None
    assert torch.equal(
        m1._fpct_packed_layout.extra_slots,
        torch.zeros_like(m1._fpct_packed_layout.extra_slots),
    )

    replicated = RosettaModel(
        [_model(2, state)],
        fpct_operator="f",
        fpct_replicated_collapse=True,
    )
    _sidecar(
        replicated,
        batch_size=2,
        source_length=4,
        num_key_value_heads=2,
        ambiguous=True,
    )
    replicated_output = _forward(replicated, input_ids, attention_mask).logits
    torch.testing.assert_close(replicated_output, cpost_output, atol=2e-6, rtol=2e-6)
    assert replicated.fpct_config_dict()["replicated_collapse"] is True

    bypass = RosettaModel(
        [_model(2, state)],
        fpct_operator="f",
        fpct_collapse_to_parent_bypass=True,
    )
    _sidecar(
        bypass,
        batch_size=2,
        source_length=4,
        num_key_value_heads=2,
        ambiguous=True,
    )
    bypass_output = _forward(bypass, input_ids, attention_mask).logits
    torch.testing.assert_close(bypass_output, cpost_output, atol=0, rtol=0)
    assert bypass.fpct_config_dict()["collapse_to_parent_bypass"] is True


def test_actual_qwen3_ambiguous_f_activates_but_has_no_new_parameters() -> None:
    base = _model(2)
    state = base.state_dict()
    input_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    cpost = RosettaModel([_model(2, state)], fpct_operator="c_post")
    factorized = RosettaModel([_model(2, state)], fpct_operator="f")
    _sidecar(
        factorized,
        batch_size=2,
        source_length=4,
        num_key_value_heads=2,
        ambiguous=True,
    )
    cpost_output = _forward(cpost, input_ids, attention_mask).logits
    factorized_output = _forward(factorized, input_ids, attention_mask).logits
    assert not torch.allclose(cpost_output, factorized_output)
    assert cpost.state_dict().keys() == factorized.state_dict().keys()


def test_actual_qwen3_off_by_default_mechanism_instrumentation() -> None:
    model = _model(2)
    wrapper = RosettaModel(
        [model], fpct_operator="f", fpct_instrumentation=True
    )
    _sidecar(
        wrapper,
        batch_size=2,
        source_length=4,
        num_key_value_heads=2,
        ambiguous=True,
    )
    _forward(
        wrapper,
        torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]]),
        torch.ones(2, 4, dtype=torch.long),
    )
    expected = {
        "gamma_kl_prior",
        "gamma_query_variance",
        "candidate_logit_variance",
        "candidate_logit_range",
        "jensen_gap",
        "d_k",
        "d_v",
        "expanded_slot_ratio",
        "extra_slots",
        "output_delta_l2",
    }
    assert expected.issubset(wrapper._fpct_mechanism_metrics)
    assert set(wrapper._fpct_layer_metrics) == {0}
    assert all(
        torch.isfinite(value).all()
        for value in wrapper._fpct_mechanism_metrics.values()
    )
    assert wrapper.fpct_config_dict()["instrumentation"] is True
