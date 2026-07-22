from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.wrapper import RosettaModel


def _config(*, layers: int, kv_heads: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    return config


def _model(*, layers: int, kv_heads: int, dtype: torch.dtype, state=None):
    torch.manual_seed(20260722)
    model = Qwen3ForCausalLM(_config(layers=layers, kv_heads=kv_heads)).to(dtype=dtype)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _tensor_sha(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _ulp_max(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.dtype == torch.float32:
        left_bits = left.contiguous().view(torch.int32).to(torch.int64)
        right_bits = right.contiguous().view(torch.int32).to(torch.int64)
    elif left.dtype == torch.bfloat16:
        left_bits = left.contiguous().view(torch.int16).to(torch.int32)
        right_bits = right.contiguous().view(torch.int16).to(torch.int32)
    else:
        raise AssertionError(f"unsupported exact-null dtype: {left.dtype}")
    return int((left_bits - right_bits).abs().max()) if left.numel() else 0


def _assert_bitwise(left: torch.Tensor, right: torch.Tensor) -> None:
    assert torch.equal(left, right)
    assert _tensor_sha(left) == _tensor_sha(right)
    assert float((left.float() - right.float()).abs().max()) == 0.0
    assert _ulp_max(left, right) == 0


def _store_partial_sidecars(
    wrapper: RosettaModel,
    *,
    layers: int,
    batch: int,
    kv_heads: int,
    dtype: torch.dtype,
    equivalent: torch.Tensor,
) -> None:
    generator = torch.Generator().manual_seed(991)
    prior = torch.full((batch, 3, 2), 0.5, dtype=torch.float32)
    valid = torch.ones_like(prior, dtype=torch.bool)
    for layer in range(layers):
        key = torch.randn(
            batch, kv_heads, 3, 2, 8, generator=generator, dtype=dtype
        )
        value = torch.randn(
            batch, kv_heads, 3, 2, 8, generator=generator, dtype=dtype
        )
        wrapper._store_fpct_sidecar(
            layer,
            2,
            key,
            value,
            prior,
            valid,
            parent_equivalent=equivalent,
            prior_sha256="r2l-test-canonical-prior",
            max_slots_hint=9,
            source_length_hint=6,
            certified=True,
        )


def _forward(
    wrapper: RosettaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache or DynamicCache(),
        use_cache=True,
        return_dict=True,
    )


@dataclass
class _Capture:
    hidden: dict[int, list[torch.Tensor]]
    handles: list


def _capture_layers(wrapper: RosettaModel) -> _Capture:
    hidden: dict[int, list[torch.Tensor]] = {}
    handles = []
    for index, layer in enumerate(wrapper.model_list[0].model.layers):
        def hook(_module, _inputs, output, *, layer_index=index):
            value = output[0] if isinstance(output, tuple) else output
            hidden.setdefault(layer_index, []).append(value.detach().clone())

        handles.append(layer.register_forward_hook(hook))
    return _Capture(hidden=hidden, handles=handles)


def _assert_cache_equal(left: DynamicCache, right: DynamicCache) -> None:
    assert len(left.key_cache) == len(right.key_cache)
    for left_key, right_key, left_value, right_value in zip(
        left.key_cache, right.key_cache, left.value_cache, right.value_cache
    ):
        _assert_bitwise(left_key, right_key)
        _assert_bitwise(left_value, right_value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_qwen3_28_layer_mixed_memory_bitwise_exact_null_decode4(
    dtype: torch.dtype,
) -> None:
    layers, kv_heads, batch = 28, 2, 2
    base = _model(layers=layers, kv_heads=kv_heads, dtype=dtype)
    state = base.state_dict()
    cpost = RosettaModel(
        [_model(layers=layers, kv_heads=kv_heads, dtype=dtype, state=state)],
        fpct_operator="c_post",
        fpct_trace=True,
    )
    factorized = RosettaModel(
        [_model(layers=layers, kv_heads=kv_heads, dtype=dtype, state=state)],
        fpct_operator="f",
        fpct_trace=True,
    )
    assert cpost.state_dict().keys() == factorized.state_dict().keys()
    _store_partial_sidecars(
        factorized,
        layers=layers,
        batch=batch,
        kv_heads=kv_heads,
        dtype=dtype,
        equivalent=torch.ones(batch, 3, dtype=torch.bool),
    )
    cpost_capture = _capture_layers(cpost)
    factorized_capture = _capture_layers(factorized)
    try:
        cpost_cache = None
        factorized_cache = None
        input_ids = torch.tensor([[4, 5, 6, 7, 8, 9], [0, 0, 10, 11, 12, 13]])
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.long
        )
        for step in range(5):
            cpost_output = _forward(cpost, input_ids, attention_mask, cpost_cache)
            factorized_output = _forward(
                factorized, input_ids, attention_mask, factorized_cache
            )
            _assert_bitwise(cpost_output.logits, factorized_output.logits)
            _assert_cache_equal(
                cpost_output.past_key_values, factorized_output.past_key_values
            )
            assert factorized._fpct_packed_layout is not None
            for layer in range(layers):
                semantic = factorized._fpct_packed_layout.semantic_parent_equivalent(layer)
                assert semantic is not None and semantic.all()
                cpost_trace = cpost._fpct_attention_trace_tensors[layer][-1]
                factorized_trace = factorized._fpct_attention_trace_tensors[layer][-1]
                _assert_bitwise(cpost_trace["pre_o_proj"], factorized_trace["pre_o_proj"])
                _assert_bitwise(cpost_trace["post_o_proj"], factorized_trace["post_o_proj"])
                _assert_bitwise(
                    cpost_capture.hidden[layer][-1], factorized_capture.hidden[layer][-1]
                )
            cpost_cache = cpost_output.past_key_values
            factorized_cache = factorized_output.past_key_values
            if step < 4:
                input_ids = torch.tensor([[14 + step], [19 + step]])
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(batch, 1, dtype=attention_mask.dtype),
                    ),
                    dim=1,
                )
    finally:
        for handle in cpost_capture.handles + factorized_capture.handles:
            handle.remove()


@torch.no_grad()
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_actual_qwen3_mixed_batch_exact_active_isolation(kv_heads: int) -> None:
    layers, batch, dtype = 2, 2, torch.float32
    base = _model(layers=layers, kv_heads=kv_heads, dtype=dtype)
    state = base.state_dict()
    cpost = RosettaModel(
        [_model(layers=layers, kv_heads=kv_heads, dtype=dtype, state=state)],
        fpct_operator="c_post",
    )
    factorized = RosettaModel(
        [_model(layers=layers, kv_heads=kv_heads, dtype=dtype, state=state)],
        fpct_operator="f",
    )
    _store_partial_sidecars(
        factorized,
        layers=layers,
        batch=batch,
        kv_heads=kv_heads,
        dtype=dtype,
        equivalent=torch.tensor(
            [[True, True, True], [True, False, True]], dtype=torch.bool
        ),
    )
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]])
    mask = torch.ones(batch, 6, dtype=torch.long)
    cpost_logits = _forward(cpost, input_ids, mask, None).logits
    factorized_logits = _forward(factorized, input_ids, mask, None).logits
    _assert_bitwise(cpost_logits[0], factorized_logits[0])
    assert not torch.equal(cpost_logits[1], factorized_logits[1])
    assert factorized._fpct_packed_layout is not None
    assert factorized._fpct_packed_layout.semantic_parent_equivalent(0).all(dim=-1).tolist() == [True, False]
