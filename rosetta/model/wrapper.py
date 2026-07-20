"""
The ensemble of multiple standard transformers LLM models, with automatic kv-cache projection. It shares the same interface as the standard transformers LLM models.
"""

from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from torch.profiler import record_function
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json
import hashlib

from rosetta.model.projector import Projector
from rosetta.model.fpct_attention import (
    FPCTPackedLayout,
    FPCTSidecarSegment,
    build_fpct_packed_layout,
    fpct_eager_attention,
    fpct_mechanism_diagnostics,
    fpct_qwen_eager_attention_forward,
    normalize_fpct_operator,
    normalize_prior as normalize_fpct_prior,
    pack_fpct_memory,
)
from rosetta.model.sampling import sample_token
from transformers.utils import ModelOutput

try:
    from transformers.generation.utils import (
        GreedySearchDecoderOnlyOutput,
        SampleDecoderOnlyOutput,
    )
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache


def hybrid_to_dynamic(hybrid_cache):
    if hybrid_cache is None:
        return None
    if isinstance(hybrid_cache, DynamicCache):
        return hybrid_cache

    # 手动从 HybridCache 提取
    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache
        values = hybrid_cache.value_cache
        assert len(keys) == len(values), "key/value 层数不一致"

        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")


class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM
    """

    def __init__(
        self,
        model_list: List[PreTrainedModel],
        base_model_idx=0,
        projector_list: List[Projector] = [],
        include_response: bool = False,
        multi_source_fusion_mode: str = "parallel",
        fpct_operator: Optional[str] = None,
        fpct_replicated_collapse: bool = False,
        fpct_instrumentation: bool = False,
        fpct_collapse_to_parent_bypass: bool = False,
        fpct_profile_scopes: bool = False,
        fpct_trace: bool = False,
    ):
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(
            device=device, dtype=dtype
        )

        self.projector_dict = {}
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []
        self.fpct_operator = normalize_fpct_operator(fpct_operator)
        self._fpct_sidecars: Dict[int, List[FPCTSidecarSegment]] = {}
        self._fpct_structure_segments: Dict[
            tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, str, int, int, bool]
        ] = {}
        self._fpct_packed_layout: Optional[FPCTPackedLayout] = None
        # Historical name retained for config/state compatibility. In R2 this
        # means expanded replicated atoms, not the exact parent bypass.
        self.fpct_replicated_collapse = bool(fpct_replicated_collapse)
        self.fpct_instrumentation = bool(fpct_instrumentation)
        self._fpct_mechanism_metrics: Dict[str, torch.Tensor] = {}
        self._fpct_layer_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
        self._fpct_input_prior_sha256: Optional[str] = None
        self.fpct_collapse_to_parent_bypass = bool(
            fpct_collapse_to_parent_bypass
        )
        self.fpct_profile_scopes = bool(fpct_profile_scopes)
        self.fpct_trace = bool(fpct_trace)
        self._fpct_candidate_trace_tensors: Dict[
            int, List[Dict[str, torch.Tensor]]
        ] = {}
        self._fpct_attention_trace_tensors: Dict[
            int, List[Dict[str, torch.Tensor]]
        ] = {}
        if self.fpct_replicated_collapse and self.fpct_operator != "f":
            raise ValueError("replicated collapse is an F-only inference control")
        if self.fpct_collapse_to_parent_bypass and self.fpct_operator != "f":
            raise ValueError("collapse-to-parent bypass is an F-only control")
        if self.fpct_operator in {"c_post", "f"}:
            for projector in self.projector_list:
                projector.suppress_host_diagnostics = True

        # Multi-source fusion mode:
        # "sequential" (default): each source updates base cache iteratively
        # "parallel": all sources project from clean base cache, then sum projections
        self.include_response = include_response
        if self.fpct_operator == "f" and include_response:
            raise ValueError("FPCT F does not support include_response hooks")
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(
                f"multi_source_fusion_mode must be 'sequential' or 'parallel', got '{multi_source_fusion_mode}'"
            )
        self.multi_source_fusion_mode = multi_source_fusion_mode

    def fpct_config_dict(self) -> Dict[str, Optional[str]]:
        config = {"operator": self.fpct_operator, "position_mode": "legacy", "a": "1", "g": "1"}
        if self.fpct_replicated_collapse:
            config["replicated_atoms"] = True
            config["replicated_collapse"] = True
        if self.fpct_collapse_to_parent_bypass:
            config["collapse_to_parent_bypass"] = True
        if self.fpct_instrumentation:
            config["instrumentation"] = True
        return config

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device

    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        """
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        return self

    # set projector
    def set_projector_config(
        self,
        source_model_idx: int,
        source_model_layer_idx: int,
        target_model_idx: int,
        target_model_layer_idx: int,
        projector_idx: int,
    ):
        """
        Set the projector configuration
        Args:
            source_model_idx: int, the index of the source model
            source_model_layer_idx: int, the index of the source model layer
            target_model_idx: int, the index of the target model
            target_model_layer_idx: int, the index of the target model layer
            projector_idx: int, the index of the projector

        The projector dict structure supports multiple projectors per target layer.
        Structure:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(
            target_model_layer_idx
        )
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][
                target_model_layer_idx
            ] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))

    def load_projector(self, projector_list):
        self.projector_list: List[Projector] = projector_list

    def get_projector(
        self,
        source_model_idx,
        source_model_layer_idx,
        target_model_idx,
        target_model_layer_idx,
    ):
        pair_list = self.projector_dict[target_model_idx][source_model_idx][
            target_model_layer_idx
        ]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result

    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip("-").isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj

    def save_projector_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            self.kv_cache_dict[target_model_idx][
                source_model_idx
            ] = DynamicCache()  # noqa, maybe we should use RosettaCache here
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(
        attn_module,
        new_k_cache,
        new_v_cache,
        fpct_sidecars: Optional[List[FPCTSidecarSegment]] = None,
        fpct_layout: Optional[FPCTPackedLayout] = None,
        fpct_replicated_collapse: bool = False,
        fpct_metric_sink: Optional[Dict[str, torch.Tensor]] = None,
        fpct_layer_metric_sink: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        fpct_collapse_to_parent_bypass: bool = False,
        fpct_profile_scopes: bool = False,
        fpct_attention_trace_sink: Optional[
            Dict[int, List[Dict[str, torch.Tensor]]]
        ] = None,
    ):
        """
        Monkeypatch Qwen3Attention.forward so that *current step* attention uses the
        provided key/value (in cache space) before computing attention.

        This avoids editing transformers' Qwen3 code while ensuring the modified KV
        is used in the same forward pass (not just for the next token).

        new_k_cache/new_v_cache: (B, kv_heads, q_len, head_dim) in the SAME space as
        Qwen3Attention's key_states/value_states AFTER k_norm + RoPE (k) and reshape (v).
        """
        import types

        # Lazy imports to avoid hard dependency at module import time
        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            apply_rotary_pos_emb,
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )

        orig_forward = attn_module.forward

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # This is essentially Qwen3Attention.forward with one injection point.
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # === Injection point (before cache update & attention) ===
            # Replace current-token key/value with provided cache-space tensors.
            # Expect same shape as key_states/value_states at this moment:
            # (B, kv_heads, q_len, head_dim)
            if new_k_cache is not None and new_v_cache is not None:
                # Only replace if compatible
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            if fpct_sidecars:
                if fpct_layout is None:
                    raise ValueError("FPCT sidecars require a prebuilt packed layout")
                base_key_states = key_states
                base_value_states = value_states
                parent_attention_mask = attention_mask
                if fpct_collapse_to_parent_bypass:
                    fpct_layout.validate_runtime(base_key_states, fpct_sidecars)
                    packed = None
                else:
                    with record_function("fpct.pack"):
                        packed = pack_fpct_memory(
                            base_key_states,
                            base_value_states,
                            parent_attention_mask,
                            fpct_sidecars,
                            query_length=query_states.shape[-2],
                            layout=fpct_layout,
                            replicated_collapse=fpct_replicated_collapse,
                        )
                    key_states = packed.key
                    value_states = packed.value
                    attention_mask = packed.attention_mask
                if fpct_metric_sink is not None and packed is not None:
                    with record_function("fpct.diagnostics"):
                        metrics = fpct_mechanism_diagnostics(query_states, packed)
                        replicated = pack_fpct_memory(
                            base_key_states,
                            base_value_states,
                            parent_attention_mask,
                            fpct_sidecars,
                            query_length=query_states.shape[-2],
                            layout=fpct_layout,
                            replicated_collapse=True,
                        )
                        factorized_output, _ = fpct_eager_attention(
                            query_states, packed
                        )
                        collapsed_output, _ = fpct_eager_attention(
                            query_states, replicated
                        )
                        metrics["output_delta_l2"] = (
                            factorized_output.float() - collapsed_output.float()
                        ).square().mean().sqrt()
                    detached_metrics = {
                        name: value.detach() for name, value in metrics.items()
                    }
                    if fpct_layer_metric_sink is not None:
                        layer_bucket = fpct_layer_metric_sink.setdefault(
                            self.layer_idx, {}
                        )
                        for name, value in detached_metrics.items():
                            sum_key, max_key, count_key = (
                                f"{name}/sum", f"{name}/max", f"{name}/count"
                            )
                            layer_bucket[sum_key] = (
                                layer_bucket.get(sum_key, torch.zeros_like(value))
                                + value
                            )
                            layer_bucket[max_key] = torch.maximum(
                                layer_bucket.get(max_key, value), value
                            )
                            layer_bucket[count_key] = (
                                layer_bucket.get(count_key, torch.zeros_like(value))
                                + torch.ones_like(value)
                            )
                    for name, value in detached_metrics.items():
                        sum_key, max_key, count_key = (
                            f"{name}/sum", f"{name}/max", f"{name}/count"
                        )
                        fpct_metric_sink[sum_key] = (
                            fpct_metric_sink.get(sum_key, torch.zeros_like(value)) + value
                        )
                        fpct_metric_sink[max_key] = torch.maximum(
                            fpct_metric_sink.get(max_key, value), value
                        )
                        fpct_metric_sink[count_key] = (
                            fpct_metric_sink.get(count_key, torch.zeros_like(value))
                            + torch.ones_like(value)
                        )
                        fpct_metric_sink[name] = (
                            fpct_metric_sink[sum_key]
                            / fpct_metric_sink[count_key].clamp_min(1)
                        )

            if self.config._attn_implementation != "eager":
                raise RuntimeError("FPCT R2 requires eager attention at runtime")
            attention_interface = fpct_qwen_eager_attention_forward

            with record_function("receiver_attention"):
                with record_function("fpct.attention"):
                    attn_output, attn_weights = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        scaling=self.scaling,
                        sliding_window=self.sliding_window,
                        **kwargs,
                    )

            if fpct_attention_trace_sink is not None:
                trace_row = {"pre_o_proj": attn_output.detach()}
                if packed is not None:
                    invalid = ~packed.active[:, None, None, :]
                    invalid_probability = torch.where(
                        invalid,
                        attn_weights.float(),
                        torch.zeros_like(attn_weights, dtype=torch.float32),
                    )
                    trace_row["invalid_probability_max"] = (
                        invalid_probability.abs().amax().detach()
                    )
                fpct_attention_trace_sink.setdefault(self.layer_idx, []).append(
                    trace_row
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            if fpct_attention_trace_sink is not None:
                fpct_attention_trace_sink[self.layer_idx][-1]["post_o_proj"] = (
                    attn_output.detach()
                )
            return attn_output, attn_weights

        attn_module.forward = types.MethodType(patched_forward, attn_module)
        return orig_forward

    def _install_fpct_attention_hooks(self, source_length: int):
        if self.fpct_operator not in {"c_post", "f"}:
            return []
        if self.model_list[self.base_model_idx].config._attn_implementation != "eager":
            raise RuntimeError("FPCT R2 requires receiver eager attention")
        if self.fpct_operator == "c_post":
            handlers = []
            for layer_idx, layer in enumerate(
                self.model_list[self.base_model_idx].model.layers
            ):
                attn = layer.self_attn
                original = self._monkeypatch_qwen3_attention_forward(
                    attn, None, None, fpct_profile_scopes=self.fpct_profile_scopes,
                    fpct_attention_trace_sink=(
                        self._fpct_attention_trace_tensors if self.fpct_trace else None
                    ),
                )
                handlers.append((attn, original))
            return handlers
        if not self._fpct_sidecars:
            return []
        ordered = [
            (layer_idx, segments)
            for layer_idx, segments in sorted(self._fpct_sidecars.items())
            if segments
        ]
        if not ordered:
            return []
        first_specs = [
            (segment.parent_start, segment.key.shape[2], segment.key.shape[3])
            for segment in ordered[0][1]
        ]
        for _layer_idx, segments in ordered[1:]:
            specs = [
                (segment.parent_start, segment.key.shape[2], segment.key.shape[3])
                for segment in segments
            ]
            if specs != first_specs:
                raise ValueError("FPCT sidecar structure differs across layers")
        if (
            self._fpct_packed_layout is None
            or self._fpct_packed_layout.source_length != source_length
        ):
            with record_function("fpct.layout_prepare"):
                self._fpct_packed_layout = build_fpct_packed_layout(
                    source_length, ordered[0][1]
                )
        handlers = []
        for layer_idx, segments in ordered:
            attn = self.model_list[self.base_model_idx].model.layers[
                layer_idx
            ].self_attn
            original = self._monkeypatch_qwen3_attention_forward(
                attn,
                None,
                None,
                fpct_sidecars=segments,
                fpct_layout=self._fpct_packed_layout,
                fpct_replicated_collapse=self.fpct_replicated_collapse,
                fpct_metric_sink=(
                    self._fpct_mechanism_metrics
                    if self.fpct_instrumentation
                    else None
                ),
                fpct_layer_metric_sink=(
                    self._fpct_layer_metrics if self.fpct_instrumentation else None
                ),
                fpct_collapse_to_parent_bypass=(
                    self.fpct_collapse_to_parent_bypass
                ),
                fpct_profile_scopes=self.fpct_profile_scopes,
                fpct_attention_trace_sink=(
                    self._fpct_attention_trace_tensors if self.fpct_trace else None
                ),
            )
            handlers.append((attn, original))
        return handlers

    def _base_model_forward_with_fpct(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            source_length = int(attention_mask.shape[-1])
        else:
            input_value = kwargs.get("input_ids")
            if input_value is None:
                input_value = kwargs.get("inputs_embeds")
            current_length = int(input_value.shape[-2] if input_value.ndim == 3 else input_value.shape[-1])
            past = kwargs.get("past_key_values")
            past_length = int(past.get_seq_length()) if past is not None else 0
            source_length = past_length + current_length
        handlers = self._install_fpct_attention_hooks(source_length)
        try:
            return self.model_list[self.base_model_idx].forward(**kwargs)
        finally:
            self.remove_hooks(handlers)

    def register_hooks(
        self,
        input_ids,
        attention_mask,
        position_ids,
        base_kv_cache,
        source_model_idx,
        source_kv_cache,
    ):

        base_kv_copy = clone_kv_cache(base_kv_cache)
        source_kv_copy = clone_kv_cache(source_kv_cache)

        new_length = input_ids.shape[1]

        base_output_kv_cache = (
            self.model_list[self.base_model_idx]
            .forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=base_kv_copy,
                labels=None,
                use_cache=True,
            )
            .past_key_values
        )
        source_output_kv_cache = (
            self.model_list[source_model_idx]
            .forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=source_kv_copy,
                labels=None,
                use_cache=True,
            )
            .past_key_values
        )
        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][
            source_model_idx
        ].items():
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            pair_list = entry

            projected_kv_list = []
            source_kv_list = []
            for source_model_layer_idx, projector_idx in pair_list:
                source_key_cache, source_value_cache = source_output_kv_cache[
                    source_model_layer_idx
                ]
                new_source_key_cache = source_key_cache[:, :, -new_length:, :]
                new_source_value_cache = source_value_cache[:, :, -new_length:, :]
                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                projected_key, projected_value = self.projector_list[
                    projector_idx
                ].forward(
                    new_source_kv_cache,  # tuple of (key, value), each of shape (B, N, H, D)
                    new_base_kv_cache,
                )
                projected_kv_list.append((projected_key, projected_value))
                source_kv_list.append(new_source_kv_cache)

            # Use first projector result
            agg_key, agg_value = projected_kv_list[0]

            # Update cache
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][
                :, :, -new_length:, :
            ] = agg_value

        # Monkeypatch attention forward so the modified KV is used in *this* forward pass.
        hook_handlers = []  # list of (attn_module, orig_forward)
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(
                attn, new_k, new_v
            )
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache

    def remove_hooks(self, hook_handlers):
        # Restore monkeypatched forwards
        for attn, orig_forward in hook_handlers:
            attn.forward = orig_forward

    def _prefill_soft_source_caches(
        self,
        input_ids: List[torch.LongTensor],
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> None:
        """Build full source-model KV caches for soft cross-token alignment."""
        if self.base_model_idx not in self.kv_cache_dict:
            self.kv_cache_dict[self.base_model_idx] = {}

        for source_model_idx in range(1, len(self.model_list)):
            source_input_ids = input_ids[source_model_idx]
            source_attention_mask = None
            if isinstance(attention_mask, list):
                source_attention_mask = attention_mask[source_model_idx]

            model = self.model_list[source_model_idx]
            was_training = model.training
            had_gc = getattr(model, "is_gradient_checkpointing", False)

            try:
                if was_training:
                    model.eval()
                if had_gc:
                    model.gradient_checkpointing_disable()

                with torch.no_grad():
                    out = model(
                        input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    source_cache = hybrid_to_dynamic(out.past_key_values)
            finally:
                if had_gc:
                    model.gradient_checkpointing_enable()
                if was_training:
                    model.train()

            self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(
                source_cache
            )

    @staticmethod
    def _weighted_source_kv_from_indices(
        source_key_cache: torch.Tensor,
        source_value_cache: torch.Tensor,
        source_indices: torch.Tensor,
        source_weights: torch.Tensor,
    ) -> tuple:
        """
        Gather source KV by top-k token indices and reduce them with source_weights.
        source cache: (B, H, S, D), indices/weights: (B, N, K).
        """
        B, H, S, D = source_key_cache.shape
        if S == 0:
            raise ValueError(
                "Cannot gather weighted source KV from an empty source cache"
            )

        source_indices = source_indices.to(source_key_cache.device)
        source_weights = source_weights.to(
            source_key_cache.device, dtype=source_key_cache.dtype
        )
        valid = (source_indices >= 0) & (source_indices < S)
        safe_indices = source_indices.clamp(min=0, max=S - 1)

        row_sums = source_weights.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
        source_weights = torch.where(
            row_sums > 0,
            source_weights.masked_fill(~valid, 0.0) / row_sums.clamp_min(1e-12),
            torch.zeros_like(source_weights),
        )

        _, N, K = safe_indices.shape
        gather_index = safe_indices[:, None, :, :, None].expand(B, H, N, K, D)
        expanded_key = source_key_cache[:, :, None, :, :].expand(B, H, N, S, D)
        expanded_value = source_value_cache[:, :, None, :, :].expand(B, H, N, S, D)
        gathered_key = torch.gather(expanded_key, dim=3, index=gather_index)
        gathered_value = torch.gather(expanded_value, dim=3, index=gather_index)
        weights = source_weights[:, None, :, :, None]

        return (
            (gathered_key * weights).sum(dim=3),
            (gathered_value * weights).sum(dim=3),
        )

    @staticmethod
    def _source_kv_candidates_from_indices(
        source_key_cache: torch.Tensor,
        source_value_cache: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> tuple:
        """
        Gather source KV candidates without reducing the top-k dimension.
        source cache: (B, H, S, D), indices: (B, N, K).
        returns key/value candidates (B, H, N, K, D) and valid mask (B, N, K).
        """
        B, H, S, D = source_key_cache.shape
        if S == 0:
            raise ValueError("Cannot gather source KV candidates from an empty cache")

        source_indices = source_indices.to(source_key_cache.device)
        valid = (source_indices >= 0) & (source_indices < S)
        safe_indices = source_indices.clamp(min=0, max=S - 1)

        _, N, K = safe_indices.shape
        gather_index = safe_indices[:, None, :, :, None].expand(B, H, N, K, D)
        expanded_key = source_key_cache[:, :, None, :, :].expand(B, H, N, S, D)
        expanded_value = source_value_cache[:, :, None, :, :].expand(B, H, N, S, D)
        gathered_key = torch.gather(expanded_key, dim=3, index=gather_index)
        gathered_value = torch.gather(expanded_value, dim=3, index=gather_index)
        valid_expanded = valid[:, None, :, :, None].to(dtype=gathered_key.dtype)
        return gathered_key * valid_expanded, gathered_value * valid_expanded, valid

    @staticmethod
    def _fpct_legal_prior(
        source_indices: torch.Tensor,
        source_weights: torch.Tensor,
        source_length: int,
        device: torch.device,
        *,
        certified: bool,
    ) -> tuple:
        indices = source_indices.to(device=device)
        weights = source_weights.to(device=device, dtype=torch.float32)
        index_valid = (indices >= 0) & (indices < source_length)
        if certified:
            return weights, index_valid & (weights > 0)
        if device.type == "cuda":
            raise ValueError("CUDA FPCT prior must be CPU-certified before forward")
        finite = torch.isfinite(weights)
        if torch.any(~finite):
            raise ValueError("FPCT source prior contains nonfinite mass")
        if torch.any(weights < 0):
            raise ValueError("FPCT source prior contains negative mass")
        if torch.any((weights > 0) & ~index_valid):
            raise ValueError("FPCT source prior assigns positive mass to an invalid index")
        legal = index_valid & (weights > 0)
        safe = indices.masked_fill(~legal, -1)
        sorted_indices = safe.sort(dim=-1).values
        duplicate = (
            (sorted_indices[..., 1:] == sorted_indices[..., :-1])
            & (sorted_indices[..., 1:] >= 0)
        )
        if torch.any(duplicate):
            raise ValueError("FPCT source prior contains a duplicate legal index")
        normalized, legal = normalize_fpct_prior(weights, legal)
        return normalized, legal

    @staticmethod
    def _fpct_projector_is_supported(projector: Projector) -> bool:
        return (
            hasattr(projector, "_compute_alignment_confidence")
            and getattr(projector, "alignment_weight_calibration_mode", "none")
            == "none"
            and getattr(projector, "learned_alignment_mode", "none") == "none"
            and getattr(projector, "learned_alignment_injection_gate_mode", "none")
            == "none"
            and getattr(projector, "learned_alignment_transfer_gate_mode", "none")
            == "none"
        )

    def _project_fpct_candidates(
        self,
        *,
        projector: Projector,
        source_key_cache: torch.Tensor,
        source_value_cache: torch.Tensor,
        base_kv: tuple,
        source_indices: torch.Tensor,
        source_weights: torch.Tensor,
        soft_section: dict,
        target_layer_idx: Optional[int] = None,
    ) -> tuple:
        if not self._fpct_projector_is_supported(projector):
            raise ValueError(
                "FPCT c_post/f requires the frozen C2C projector path and rejects "
                "learned alignment/router/injection/transfer gates"
            )
        source_candidates_k, source_candidates_v, index_valid = (
            self._source_kv_candidates_from_indices(
                source_key_cache,
                source_value_cache,
                source_indices,
            )
        )
        prior, legal = self._fpct_legal_prior(
            source_indices,
            source_weights,
            source_key_cache.shape[2],
            source_key_cache.device,
            certified=soft_section.get("fpct_prior_certified") is True,
        )
        legal = legal & index_valid.to(device=legal.device)
        weights = prior[:, None, :, :, None].float()
        averaged_source = (
            (source_candidates_k.float() * weights).sum(dim=3).to(
                dtype=source_candidates_k.dtype
            ),
            (source_candidates_v.float() * weights).sum(dim=3).to(
                dtype=source_candidates_v.dtype
            ),
        )
        projector_uses_confidence = (
            hasattr(projector, "uses_internal_source_confidence")
            and projector.uses_internal_source_confidence()
        )
        projector_kwargs = {}
        if projector_uses_confidence:
            projector_kwargs = {
                "source_confidence": soft_section.get("source_confidence"),
                "source_weights": prior,
                "source_entropy": soft_section.get("source_entropy"),
                "source_entropy_override": soft_section.get(
                    "source_entropy_override"
                ),
            }
        parent_projected = projector.forward(
            averaged_source,
            base_kv,
            fpct_capture_parent_nuisance=True,
            **projector_kwargs,
        )
        parent_confidence_aux = getattr(
            projector, "_last_alignment_confidence_aux_loss", None
        )
        parent_residual_aux = getattr(
            projector, "_last_alignment_residual_scale_aux_loss", None
        )
        nuisance = getattr(projector, "_fpct_last_parent_nuisance", None)
        if not isinstance(nuisance, dict):
            raise ValueError("FPCT projector did not expose parent nuisance")
        if not projector_uses_confidence:
            parent_projected = self._apply_source_confidence_to_projected_kv(
                parent_projected[0],
                parent_projected[1],
                base_kv[0],
                base_kv[1],
                soft_section,
            )

        candidate_keys = []
        candidate_values = []
        for candidate_idx in range(source_candidates_k.shape[3]):
            projected_key, projected_value = projector.forward(
                (
                    source_candidates_k[:, :, :, candidate_idx, :],
                    source_candidates_v[:, :, :, candidate_idx, :],
                ),
                base_kv,
                fpct_parent_nuisance=nuisance,
                **projector_kwargs,
            )
            if not projector_uses_confidence:
                projected_key, projected_value = (
                    self._apply_source_confidence_to_projected_kv(
                        projected_key,
                        projected_value,
                        base_kv[0],
                        base_kv[1],
                        soft_section,
                    )
                )
            candidate_keys.append(projected_key)
            candidate_values.append(projected_value)
        fused_key = torch.stack(candidate_keys, dim=3)
        fused_value = torch.stack(candidate_values, dim=3)
        projector._last_alignment_confidence_aux_loss = parent_confidence_aux
        projector._last_alignment_residual_scale_aux_loss = parent_residual_aux
        collapsed_key = (fused_key.float() * weights).sum(dim=3).to(
            dtype=base_kv[0].dtype
        )
        collapsed_value = (fused_value.float() * weights).sum(dim=3).to(
            dtype=base_kv[1].dtype
        )
        candidate_count = legal.sum(dim=-1)
        first_candidate = legal.to(torch.long).argmax(dim=-1)
        gather_index = first_candidate[:, None, :, None, None].expand(
            fused_key.shape[0], fused_key.shape[1], fused_key.shape[2], 1,
            fused_key.shape[-1],
        )
        single_key = torch.gather(fused_key, 3, gather_index).squeeze(3)
        single_value = torch.gather(fused_value, 3, gather_index).squeeze(3)
        singleton = (candidate_count == 1)[:, None, :, None]
        collapsed_key = torch.where(singleton, single_key, collapsed_key)
        collapsed_value = torch.where(singleton, single_value, collapsed_value)
        has_support = legal.any(dim=-1)[:, None, :, None]
        collapsed_key = torch.where(has_support, collapsed_key, base_kv[0])
        collapsed_value = torch.where(has_support, collapsed_value, base_kv[1])
        if self.fpct_trace:
            if target_layer_idx is None:
                raise ValueError("FPCT trace requires target_layer_idx")
            self._fpct_candidate_trace_tensors.setdefault(
                int(target_layer_idx), []
            ).append(
                {
                    "source_candidate_key": source_candidates_k.detach(),
                    "source_candidate_value": source_candidates_v.detach(),
                    "native_parent_key": base_kv[0].detach(),
                    "native_parent_value": base_kv[1].detach(),
                    "fused_candidate_key": fused_key.detach(),
                    "fused_candidate_value": fused_value.detach(),
                    "collapsed_key": collapsed_key.detach(),
                    "collapsed_value": collapsed_value.detach(),
                    "prior": prior.detach(),
                    "legal": legal.detach(),
                    "legacy_key_gate": nuisance["legacy_key_gate"].detach(),
                    "legacy_value_gate": nuisance["legacy_value_gate"].detach(),
                    "key_alignment_confidence": nuisance[
                        "key_alignment_confidence"
                    ].detach(),
                    "value_alignment_confidence": nuisance[
                        "value_alignment_confidence"
                    ].detach(),
                }
            )
        return (
            collapsed_key,
            collapsed_value,
            fused_key,
            fused_value,
            prior,
            legal,
            parent_projected,
            nuisance,
        )

    def _store_fpct_sidecar(
        self,
        target_layer_idx: int,
        parent_start: int,
        fused_key: torch.Tensor,
        fused_value: torch.Tensor,
        prior: torch.Tensor,
        legal: torch.Tensor,
        prior_sha256: str = "",
        max_slots_hint: int = -1,
        source_length_hint: int = -1,
        certified: bool = False,
    ) -> None:
        if not certified:
            if prior.device.type == "cuda":
                raise ValueError("CUDA FPCT sidecar requires CPU-certified metadata")
            prior, legal = normalize_fpct_prior(prior, legal)
            digest = hashlib.sha256()
            digest.update(b"fpct-test-prior-v1\0")
            digest.update(prior.contiguous().numpy().tobytes())
            prior_sha256 = digest.hexdigest()
            counts = legal.sum(dim=-1)
            slots = torch.where(
                counts >= 2, counts, torch.ones_like(counts)
            ).sum(dim=-1)
            max_slots_hint = int(slots.max()) if slots.numel() else 0
            source_length_hint = parent_start + int(fused_key.shape[2])
            certified = True
        if not prior_sha256 or max_slots_hint < 0 or source_length_hint < 0:
            raise ValueError("FPCT sidecar certification metadata is incomplete")
        structure_key = (parent_start, fused_key.shape[2], fused_key.shape[3])
        canonical = self._fpct_structure_segments.get(structure_key)
        if canonical is None:
            canonical = (
                prior.detach(), legal.detach(), prior_sha256,
                int(max_slots_hint), int(source_length_hint), True,
            )
            self._fpct_structure_segments[structure_key] = canonical
        elif canonical[2:] != (
            prior_sha256, int(max_slots_hint), int(source_length_hint), True
        ):
            raise ValueError("FPCT sidecar prior provenance differs across layers")
        segment = FPCTSidecarSegment(
            parent_start=parent_start,
            key=fused_key,
            value=fused_value,
            prior=canonical[0],
            valid=canonical[1],
            prior_sha256=canonical[2],
            max_slots_hint=canonical[3],
            source_length_hint=canonical[4],
            certified=canonical[5],
        )
        segment.validate()
        segments = self._fpct_sidecars.setdefault(target_layer_idx, [])
        new_end = parent_start + fused_key.shape[2]
        for existing in segments:
            existing_end = existing.parent_start + existing.key.shape[2]
            if parent_start < existing_end and existing.parent_start < new_end:
                raise ValueError("overlapping FPCT sidecar segments")
        segments.append(segment)

    @staticmethod
    def _apply_source_confidence_to_projected_kv(
        projected_key: torch.Tensor,
        projected_value: torch.Tensor,
        base_key: torch.Tensor,
        base_value: torch.Tensor,
        soft_section: dict,
    ) -> tuple:
        """Scale the projector residual by optional per-token source confidence."""
        source_confidence = soft_section.get("source_confidence")
        if source_confidence is None:
            return projected_key, projected_value

        confidence = source_confidence.to(
            projected_key.device,
            dtype=projected_key.dtype,
        )
        if confidence.dim() == 2:
            confidence = confidence[:, None, :, None]
        elif confidence.dim() == 3:
            confidence = confidence[:, None, :, :]
        else:
            raise ValueError(
                "source_confidence must have shape (B, N) or (B, N, 1), "
                f"got {tuple(confidence.shape)}"
            )
        confidence = confidence.clamp(min=0.0, max=1.0)

        return (
            base_key + confidence * (projected_key - base_key),
            base_value + confidence * (projected_value - base_value),
        )

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        soft_alignment: Optional[List[Dict[str, torch.Tensor]]] = None,
        # **kwargs: Unpack[KwargsForCausalLM],
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass

        kv_cache_index: List of tensors with shape (B, sec_seq_len, 2).
            The first element [i][0][0][0] controls sharer selection:
            - -1: No projection (receiver only, skip all sharers)
            - 0: Self projection (receiver projects from itself) - not currently used
            - >0: Bitmask selecting sharers (1 (001)=sharer1, 2 (010)=sharer2, 3 (011)=both, 7 (111)=all three)
            Each bit corresponds to a sharer: bit i selects sharer at model_list[i+1].

        input_ids: If LongTensor, same input for all models. If List, per-model inputs.
        """

        # Handle different input formats: if input_ids is a list, use per-model inputs
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            base_input_ids = (
                input_ids[self.base_model_idx] if input_ids is not None else None
            )
            base_attention_mask = (
                attention_mask[self.base_model_idx]
                if attention_mask is not None
                else None
            )
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()
            self._fpct_sidecars = {}
            self._fpct_structure_segments = {}
            self._fpct_packed_layout = None
        self._fpct_mechanism_metrics.clear()
        self._fpct_layer_metrics.clear()
        self._fpct_candidate_trace_tensors.clear()
        self._fpct_attention_trace_tensors.clear()

        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1
        use_soft_alignment = soft_alignment is not None
        if use_soft_alignment:
            if not isinstance(input_ids, list):
                raise ValueError("soft_alignment requires per-model input_ids")
            if len(soft_alignment) != num_sections:
                raise ValueError(
                    f"soft_alignment section count {len(soft_alignment)} does not match "
                    f"kv_cache_index section count {num_sections}"
                )
            if self.fpct_operator in {"c_post", "f"}:
                if not all(
                    section.get("fpct_prior_certified") is True
                    for section in soft_alignment
                ):
                    raise ValueError("FPCT R2 requires CPU-certified alignment metadata")
                prior_hashes = {
                    section.get("fpct_prior_sha256") for section in soft_alignment
                }
                if len(prior_hashes) != 1 or None in prior_hashes:
                    raise ValueError("FPCT sections do not share one canonical prior SHA")
                self._fpct_input_prior_sha256 = prior_hashes.pop()
            if seqlen > 1:
                self._prefill_soft_source_caches(input_ids, attention_mask)

        section_lengths = (
            [kv_cache_index[i].shape[1] for i in range(num_sections)]
            if kv_cache_index is not None
            else [seqlen]
        )
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)

        curr_base_kv_cache = past_key_values

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = (
                base_input_ids[:, start:end] if base_input_ids is not None else None
            )
            prefill_attention_mask = (
                base_attention_mask[:, :end]
                if base_attention_mask is not None
                else None
            )
            prefill_position_ids = (
                position_ids[:, start:end] if position_ids is not None else None
            )
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:

                if self.include_response:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = (
                        self.register_hooks(
                            input_ids=prefill_input_ids,
                            attention_mask=prefill_attention_mask,
                            position_ids=prefill_position_ids,
                            base_kv_cache=self.kv_cache_dict[self.base_model_idx][
                                self.base_model_idx
                            ],
                            source_model_idx=1,
                            source_kv_cache=self.kv_cache_dict[self.base_model_idx][1],
                        )
                    )

                # calculate target model kvcache
                output = self._base_model_forward_with_fpct(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs,
                )

                if self.include_response:
                    self.remove_hooks(hook_handlers)

                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = (
                        clone_kv_cache(base_output_kv_cache)
                    )
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(
                        source_output_kv_cache
                    )

            else:

                output = self._base_model_forward_with_fpct(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs,
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = (
                    clone_kv_cache(output.past_key_values)
                )

                curr_base_kv_cache: DynamicCache = output.past_key_values

                if not use_soft_alignment:
                    for source_model_idx in range(1, len(self.model_list)):
                        if self.base_model_idx not in self.kv_cache_dict:
                            self.kv_cache_dict[self.base_model_idx] = {}
                        if (
                            source_model_idx
                            not in self.kv_cache_dict[self.base_model_idx]
                        ):
                            self.kv_cache_dict[self.base_model_idx][
                                source_model_idx
                            ] = None

                        # Get model-specific input_ids and attention_mask
                        if isinstance(input_ids, list):
                            source_input_ids = input_ids[source_model_idx]
                            source_attention_mask = (
                                attention_mask[source_model_idx]
                                if attention_mask is not None
                                else None
                            )
                            source_prefill_input_ids = (
                                source_input_ids[:, start:end]
                                if source_input_ids is not None
                                else None
                            )
                            source_prefill_attention_mask = (
                                source_attention_mask[:, :end]
                                if source_attention_mask is not None
                                else None
                            )
                        else:
                            # Backward compatibility: use same input for all models
                            source_prefill_input_ids = prefill_input_ids
                            source_prefill_attention_mask = prefill_attention_mask

                        model = self.model_list[source_model_idx]
                        was_training = model.training
                        had_gc = getattr(model, "is_gradient_checkpointing", False)

                        try:
                            if was_training:
                                model.eval()
                            if had_gc:
                                model.gradient_checkpointing_disable()

                            with torch.no_grad():
                                out = model(
                                    input_ids=source_prefill_input_ids,
                                    attention_mask=source_prefill_attention_mask,
                                    position_ids=prefill_position_ids,
                                    past_key_values=self.kv_cache_dict[
                                        self.base_model_idx
                                    ][source_model_idx],
                                    use_cache=True,
                                    return_dict=True,
                                )
                                curr_source_kv_cache = out.past_key_values
                        finally:
                            if had_gc:
                                model.gradient_checkpointing_enable()
                            if was_training:
                                model.train()

                        curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = (
                            clone_kv_cache(curr_source_kv_cache)
                        )

                # calculate source model kvcache and apply projections
                if self.base_model_idx in self.projector_dict:
                    # Iterate over all source models in projector_dict
                    if self.fpct_operator in {"c_post", "f"}:
                        sharer_mask = soft_alignment[i].get("fpct_sharer_mask")
                        if not isinstance(sharer_mask, int):
                            raise ValueError(
                                "FPCT R2 requires CPU scalar sharer metadata"
                            )
                    else:
                        sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        base_cache = clone_kv_cache(curr_base_kv_cache)

                        # For parallel mode, accumulate residuals for each target layer
                        parallel_delta_cache = (
                            {} if self.multi_source_fusion_mode == "parallel" else None
                        )

                        # Compute and apply projections (shared logic for both modes)
                        for source_model_idx in self.projector_dict[
                            self.base_model_idx
                        ].keys():
                            # Check if this sharer is selected: bit (source_model_idx - 1)
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue
                            if self.multi_source_fusion_mode == "sequential":
                                base_cache_ref = curr_base_kv_cache
                            else:
                                # Parallel: always project from the clean cloned base cache
                                base_cache_ref = base_cache

                            for target_layer_idx, entry in self.projector_dict[
                                self.base_model_idx
                            ][source_model_idx].items():
                                # Get base KV cache slice for projection
                                base_key_cache, base_value_cache = base_cache_ref[
                                    target_layer_idx
                                ]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]
                                new_base_value_cache = base_value_cache[
                                    :, :, start:end, :
                                ]
                                new_base_kv_cache = (
                                    new_base_key_cache,
                                    new_base_value_cache,
                                )

                                pair_list = entry

                                projected_kv_list = []
                                source_kv_list = []
                                fpct_candidate_records = []
                                for source_model_layer_idx, projector_idx in pair_list:
                                    source_key_cache, source_value_cache = (
                                        self.kv_cache_dict[self.base_model_idx][
                                            source_model_idx
                                        ][source_model_layer_idx]
                                    )
                                    projector = self.projector_list[projector_idx]
                                    if use_soft_alignment:
                                        soft_section = soft_alignment[i]
                                        source_weights = soft_section["source_weights"]
                                        if self.fpct_operator in {"c_post", "f"}:
                                            with record_function(
                                                "fpct.project_candidates"
                                            ):
                                                fpct_record = self._project_fpct_candidates(
                                                    projector=projector,
                                                    source_key_cache=source_key_cache,
                                                    source_value_cache=source_value_cache,
                                                    base_kv=new_base_kv_cache,
                                                    source_indices=soft_section[
                                                        "source_indices"
                                                    ],
                                                    source_weights=source_weights,
                                                    soft_section=soft_section,
                                                    target_layer_idx=target_layer_idx,
                                                )
                                            projected_kv_list.append(
                                                (fpct_record[0], fpct_record[1])
                                            )
                                            source_kv_list.append(fpct_record[6])
                                            fpct_candidate_records.append(fpct_record)
                                            continue
                                        projector_uses_learned_alignment = (
                                            hasattr(
                                                projector,
                                                "uses_learned_source_alignment",
                                            )
                                            and projector.uses_learned_source_alignment()
                                        )
                                        if projector_uses_learned_alignment:
                                            (
                                                source_key_candidates,
                                                source_value_candidates,
                                                valid_mask,
                                            ) = self._source_kv_candidates_from_indices(
                                                source_key_cache=source_key_cache,
                                                source_value_cache=source_value_cache,
                                                source_indices=soft_section[
                                                    "source_indices"
                                                ],
                                            )
                                            new_source_kv_cache = (
                                                projector.align_source_kv(
                                                    source_kv_candidates=(
                                                        source_key_candidates,
                                                        source_value_candidates,
                                                    ),
                                                    target_kv=new_base_kv_cache,
                                                    valid_mask=valid_mask,
                                                    source_weights=source_weights,
                                                )
                                            )
                                        else:
                                            if hasattr(
                                                projector, "calibrate_source_weights"
                                            ):
                                                source_weights = (
                                                    projector.calibrate_source_weights(
                                                        source_weights=source_weights,
                                                        source_indices=soft_section[
                                                            "source_indices"
                                                        ],
                                                        source_confidence=soft_section.get(
                                                            "source_confidence"
                                                        ),
                                                        source_entropy=soft_section.get(
                                                            "source_entropy"
                                                        ),
                                                        source_entropy_override=soft_section.get(
                                                            "source_entropy_override"
                                                        ),
                                                    )
                                                )
                                            new_source_kv_cache = (
                                                self._weighted_source_kv_from_indices(
                                                    source_key_cache=source_key_cache,
                                                    source_value_cache=source_value_cache,
                                                    source_indices=soft_section[
                                                        "source_indices"
                                                    ],
                                                    source_weights=source_weights,
                                                )
                                            )
                                    else:
                                        new_source_key_cache = source_key_cache[
                                            :, :, start:end, :
                                        ]
                                        new_source_value_cache = source_value_cache[
                                            :, :, start:end, :
                                        ]
                                        new_source_kv_cache = (
                                            new_source_key_cache,
                                            new_source_value_cache,
                                        )
                                    projector_uses_confidence = (
                                        use_soft_alignment
                                        and hasattr(
                                            projector,
                                            "uses_internal_source_confidence",
                                        )
                                        and projector.uses_internal_source_confidence()
                                    )
                                    projector_kwargs = {}
                                    if projector_uses_confidence:
                                        projector_kwargs = {
                                            "source_confidence": soft_section.get(
                                                "source_confidence"
                                            ),
                                            "source_weights": source_weights,
                                            "source_entropy": soft_section.get(
                                                "source_entropy"
                                            ),
                                            "source_entropy_override": soft_section.get(
                                                "source_entropy_override"
                                            ),
                                        }
                                    projected_key, projected_value = projector.forward(
                                        new_source_kv_cache,
                                        new_base_kv_cache,
                                        **projector_kwargs,
                                    )
                                    if (
                                        use_soft_alignment
                                        and not projector_uses_confidence
                                    ):
                                        projected_key, projected_value = (
                                            self._apply_source_confidence_to_projected_kv(
                                                projected_key=projected_key,
                                                projected_value=projected_value,
                                                base_key=new_base_key_cache,
                                                base_value=new_base_value_cache,
                                                soft_section=soft_section,
                                            )
                                        )
                                    projected_kv_list.append(
                                        (projected_key, projected_value)
                                    )
                                    source_kv_list.append(new_source_kv_cache)

                                # Use first projector result
                                agg_key, agg_value = projected_kv_list[0]
                                if self.fpct_operator == "f":
                                    if not fpct_candidate_records:
                                        raise ValueError(
                                            "FPCT F requires soft-alignment candidate records"
                                        )
                                    fpct_record = fpct_candidate_records[0]
                                    self._store_fpct_sidecar(
                                        target_layer_idx=target_layer_idx,
                                        parent_start=start,
                                        fused_key=fpct_record[2],
                                        fused_value=fpct_record[3],
                                        prior=fpct_record[4],
                                        legal=fpct_record[5],
                                        prior_sha256=soft_section.get(
                                            "fpct_prior_sha256", ""
                                        ),
                                        max_slots_hint=int(
                                            soft_section.get(
                                                "fpct_max_slots_hint", -1
                                            )
                                        ),
                                        source_length_hint=int(
                                            soft_section.get(
                                                "fpct_target_length_hint", -1
                                            )
                                        ),
                                        certified=(
                                            soft_section.get(
                                                "fpct_prior_certified"
                                            )
                                            is True
                                        ),
                                    )

                                # Collect or apply projection based on mode
                                if self.multi_source_fusion_mode == "sequential":
                                    # Sequential: apply immediately so next source sees updated cache
                                    curr_base_kv_cache.key_cache[target_layer_idx][
                                        :, :, start:end, :
                                    ] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][
                                        :, :, start:end, :
                                    ] = agg_value
                                else:
                                    # Parallel: accumulate residuals (agg - base) for this target layer
                                    if target_layer_idx not in parallel_delta_cache:
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[
                                        target_layer_idx
                                    ]
                                    delta_key = delta_key + (
                                        agg_key - new_base_key_cache
                                    )
                                    delta_value = delta_value + (
                                        agg_value - new_base_value_cache
                                    )
                                    parallel_delta_cache[target_layer_idx] = (
                                        delta_key,
                                        delta_value,
                                    )

                        # For parallel mode, apply all accumulated residuals in one shot
                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (
                                delta_key,
                                delta_value,
                            ) in parallel_delta_cache.items():
                                base_key_cache, base_value_cache = base_cache[
                                    target_layer_idx
                                ]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                curr_base_kv_cache.key_cache[target_layer_idx][
                                    :, :, start:end, :
                                ] = (base_key_slice + delta_key)
                                curr_base_kv_cache.value_cache[target_layer_idx][
                                    :, :, start:end, :
                                ] = (base_value_slice + delta_value)

                output.past_key_values = curr_base_kv_cache

        return output

    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        streamer=None,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """

        self.kv_cache_dict = dict()
        prefill_soft_alignment = kwargs.pop("soft_alignment", None)

        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = (
            gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        )
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = (
                eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]
            )

        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        # Resolve base inputs
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = (
                attention_mask[self.base_model_idx]
                if attention_mask is not None
                else None
            )
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(
                base_input_ids, dtype=torch.long, device=base_input_ids.device
            )

        batch_size = base_input_ids.size(0)

        # Prefill to build caches and obtain initial logits
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            soft_alignment=prefill_soft_alignment,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        # Initialize streamer with prompt if provided
        if streamer is not None:
            streamer.put(base_input_ids)

        # EOS handling setup
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(
                eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            )
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=all_input_ids.device
        )

        # Start from last prefill logits
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # Apply repetition/presence/frequency penalties to logits before sampling
            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0)
                or (presence_penalty is not None and presence_penalty != 0.0)
                or (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()
                vocab_size = adjusted_logits.size(-1)
                # Per-batch penalty application for clarity and correctness
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]
                    if seq_tokens.numel() == 0:
                        continue
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype != torch.float32 and counts.dtype != torch.float64:
                        counts = counts.to(adjusted_logits.dtype)
                    # Presence penalty: penalize any token that has appeared
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = (
                                adjusted_logits[b, presence_mask] - presence_penalty
                            )
                    # Frequency penalty: penalize proportionally to frequency
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = (
                            adjusted_logits[b] - frequency_penalty * counts
                        )
                    # Repetition penalty (HF-style): divide positive logits, multiply negative logits
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = (
                                    adjusted_logits[b, pos_mask] / repetition_penalty
                                )
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = (
                                    adjusted_logits[b, neg_mask] * repetition_penalty
                                )

            # Sample next token
            next_token = sample_token(
                adjusted_logits,
                temperature=effective_temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor(
                    [next_token], device=all_input_ids.device, dtype=torch.long
                ).repeat(batch_size)

            # Apply EOS logic
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= next_token == eid
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(
                            pad_token_id,
                            device=next_token.device,
                            dtype=next_token.dtype,
                        ),
                        next_token,
                    )

            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        device=current_attention_mask.device,
                        dtype=current_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            # Stream the new token if streamer provided
            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [
                torch.tensor([-1, 0], dtype=torch.long)
                .repeat(1, 1)
                .unsqueeze(0)
                .to(all_input_ids.device)
            ]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            last_logits = decode_output.logits[:, -1, :]

        # End streaming if streamer provided
        if streamer is not None:
            streamer.end()

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if (
                GreedySearchDecoderOnlyOutput is not None
                and SampleDecoderOnlyOutput is not None
            ):
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids
