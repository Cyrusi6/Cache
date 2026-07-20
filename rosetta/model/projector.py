"""
Projector nn module for the unified memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Cache, DynamicCache
from typing import Dict, Optional, Tuple, Literal, Union
import copy
import math

from rosetta.utils.registry import (
    register_model,
    get_projector_class,
    PROJECTOR_REGISTRY,
    capture_init_args,
    save_object,
    load_object,
)


class Projector(nn.Module):
    """Base projector class for unified memory"""

    def uses_internal_source_confidence(self) -> bool:
        """Whether this projector consumes source confidence inside forward."""
        return False

    def uses_learned_source_alignment(self) -> bool:
        """Whether this projector learns top-k source-token aggregation."""
        return False

    def align_source_kv(
        self,
        source_kv_candidates: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        valid_mask: Tensor,
        source_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Optionally reduce source KV candidates before projection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement learned source alignment"
        )

    def calibrate_source_weights(
        self,
        source_weights: Tensor,
        source_indices: Optional[Tensor] = None,
        source_confidence: Optional[Tensor] = None,
        source_entropy: Optional[Tensor] = None,
        source_entropy_override: Optional[Tensor] = None,
    ) -> Tensor:
        """Optionally calibrate top-k source weights before source KV gathering."""
        return source_weights

    def forward(
        self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Project and combine the source key-value tensors to the target key-value tensors
        Args:
            source_kv: Tuple of (key, value) tensors, each (..., D_s) where ... are arbitrary leading dimensions
            target_kv: Tuple of (key, value) tensors, each (..., D_t) where ... are arbitrary leading dimensions
        Returns:
            Tuple of (key, value) tensors, each (..., D_t) with same leading dimensions as input
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def cache_project(self, source_kv_cache: Cache, target_kv_cache: Cache) -> Cache:
        """
        Project the source kv cache to the target kv cache
        """
        if not isinstance(source_kv_cache, DynamicCache) or not isinstance(
            target_kv_cache, DynamicCache
        ):
            raise ValueError("Only DynamicCache is supported")

        projected_cache = DynamicCache()

        # Process each layer
        for layer_idx in range(len(source_kv_cache.key_cache)):
            source_key = source_kv_cache.key_cache[layer_idx]  # (B, H, N, D_s)
            source_value = source_kv_cache.value_cache[layer_idx]  # (B, H, N, D_s)

            # Get corresponding target tensors (for reference/combination)
            if layer_idx < len(target_kv_cache.key_cache):
                target_key = target_kv_cache.key_cache[layer_idx]  # (B, H, N, D_t)
                target_value = target_kv_cache.value_cache[layer_idx]  # (B, H, N, D_t)
            else:
                # If target cache doesn't have this layer, create dummy tensors
                B, H, N, D_s = source_key.shape
                D_t = source_key.shape[-1]  # Assume same dimension for simplicity
                target_key = torch.zeros(
                    B, H, N, D_t, device=source_key.device, dtype=source_key.dtype
                )
                target_value = torch.zeros(
                    B, H, N, D_t, device=source_value.device, dtype=source_value.dtype
                )

            # Reshape for forward pass: DynamicCache format (B, H, N, D) -> projector format (B, N, H, D)
            source_key_reshaped = source_key.transpose(1, 2)
            source_value_reshaped = source_value.transpose(1, 2)
            target_key_reshaped = target_key.transpose(1, 2)
            target_value_reshaped = target_value.transpose(1, 2)

            # Project using forward method with tuple input/output
            source_kv = (source_key_reshaped, source_value_reshaped)
            target_kv = (target_key_reshaped, target_value_reshaped)
            projected_key, projected_value = self.forward(source_kv, target_kv)

            # Reshape back: projector format (B, N, H, D) -> DynamicCache format (B, H, N, D)
            projected_key = projected_key.transpose(1, 2)
            projected_value = projected_value.transpose(1, 2)

            # Update cache
            projected_cache.update(projected_key, projected_value, layer_idx)

        return projected_cache


class ModernMLP(nn.Module):
    """
    Modern MLP with residual connections, layer normalization, and configurable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_swiglu = use_swiglu

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer_output_dim = output_dim if i == num_layers - 1 else hidden_dim

            if (
                self.use_swiglu and i < num_layers - 1
            ):  # Don't use SwiGLU on output layer
                layer = SwiGLUBlock(layer_input_dim, layer_output_dim, dtype=dtype)
            else:
                layer = nn.Linear(layer_input_dim, layer_output_dim, dtype=dtype)

            self.layers.append(layer)

            # Add layer norm after each layer except the last one
            if use_layer_norm and i < num_layers - 1:
                self.layers.append(nn.LayerNorm(layer_output_dim, dtype=dtype))

            # Add activation after each layer except the last one
            if i < num_layers - 1 and not self.use_swiglu:
                self.layers.append(copy.deepcopy(self.activation))

            # Add dropout after activation
            if dropout > 0 and i < num_layers - 1:
                self.layers.append(nn.Dropout(dropout))

        # Residual projection if dimensions don't match
        if self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        else:
            self.residual_proj = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        residual = x

        for layer in self.layers:
            x = layer(x)

        # Add residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        return x


class SwiGLUBlock(nn.Module):
    """SwiGLU activation block for modern transformer architectures."""

    def __init__(
        self, input_dim: int, output_dim: int, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.up_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return gate * up


@register_model
@capture_init_args
class AllInOneProjector(Projector):
    """
    Unified projector that consolidates all projection functionalities with modern patterns.

    Features:
    1. Gate logit granularity: scalar, token-wise, head-wise, or value-wise
    2. Key/Value weight granularity: scalar, token-wise, head-wise, or value-wise
    3. Input-dependent gates and weights via MLP or parameters
    4. Optional concatenation with combiner networks
    5. Modern MLP architecture with residual connections and SwiGLU
    6. Configurable target preservation: choose between traditional blending or simplified projection
    7. Optional adding of target (self) signal to outputs via add_self

    Target Preservation Modes:
    - preserve_target_weight=True (default): output = (1-weight)*target + gate*weight*projected
    - preserve_target_weight=False: output = target + gate*weight*projected (no weight coefficient on target)
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_swiglu: bool = False,
        # Gate configuration
        gate_granularity: Literal[
            "scalar", "token", "head", "head_merged", "value"
        ] = "scalar",
        gate_depends_on_input: bool = False,
        gate_input_features: Optional[
            str
        ] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        gate_init_value: float = 0.0,
        # Weight configuration
        weight_granularity: Literal[
            "scalar", "token", "head", "head_merged", "value"
        ] = "scalar",
        weight_depends_on_input: bool = False,
        weight_input_features: Optional[
            str
        ] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        weight_init_value: float = 0.0,
        # Target preservation configuration
        preserve_target_weight: bool = True,  # If False, target won't be multiplied by (1 - normalized_weight)
        add_self: bool = True,  # If False, target (self) won't be added to outputs
        # Concat configuration
        use_concat: bool = False,
        # combiner_hidden_dim: int = 128,
        weight_hidden_dim: int = 1024,
        # Temperature and gumbel
        use_gumbel: bool = True,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        anneal_steps: int = 1360,
        scalar_temperature: float = 0.005,
        # Sequence length configuration
        max_sequence_length: int = 8192,  # Maximum sequence length for token-level parameters
        pos_emb: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.hidden_dim = hidden_dim
        self.weight_hidden_dim = weight_hidden_dim
        self.max_sequence_length = max_sequence_length

        # Configuration
        self.gate_granularity = gate_granularity
        self.gate_depends_on_input = gate_depends_on_input
        self.gate_input_features = gate_input_features
        self.weight_granularity = weight_granularity
        self.weight_depends_on_input = weight_depends_on_input
        self.weight_input_features = weight_input_features
        self.preserve_target_weight = preserve_target_weight
        self.add_self = add_self
        self.use_concat = use_concat
        self.use_gumbel = use_gumbel
        self.scalar_temperature = scalar_temperature

        # Temperature annealing for gate
        self.register_buffer(
            "gate_temperature", torch.tensor(initial_temperature, dtype=dtype)
        )
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps

        # Build projection networks
        self.key_projection = self._build_projection_mlp(
            source_dim * source_num_heads,
            target_dim * target_num_heads,
            hidden_dim,
            num_layers,
            activation,
            use_layer_norm,
            use_residual,
            dropout,
            use_swiglu,
            dtype,
        )
        self.value_projection = self._build_projection_mlp(
            source_dim * source_num_heads,
            target_dim * target_num_heads,
            hidden_dim,
            num_layers,
            activation,
            use_layer_norm,
            use_residual,
            dropout,
            use_swiglu,
            dtype,
        )

        # Build gate components
        self._build_gate_components(dtype)

        # Build weight components
        self._build_weight_components(weight_init_value, dtype)

        # Build concat components if needed
        if self.use_concat:
            in_dim = target_dim * target_num_heads * 2
            out_dim = target_dim * target_num_heads
            self.key_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)
            self.value_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)

    def _build_projection_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        use_layer_norm: bool,
        use_residual: bool,
        dropout: float,
        use_swiglu: bool,
        dtype: torch.dtype,
    ) -> ModernMLP:
        """Build modern MLP for projection."""
        return ModernMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            use_swiglu=use_swiglu,
            dtype=dtype,
        )

    def _build_gate_components(self, dtype: torch.dtype):
        """Build gate logit components based on configuration."""
        if not self.gate_depends_on_input:
            # Parameter-based gate
            gate_shape = self._get_parameter_shape(self.gate_granularity)
            self.gate_logit = nn.Parameter(torch.zeros(gate_shape, dtype=dtype))
        else:
            # Input-dependent gate via MLP
            input_dim = self._get_gate_input_dim()
            output_dim = self._get_gate_output_dim()

            self.gate_generator = ModernMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype,
            )

    def _build_weight_components(self, weight_init_value: float, dtype: torch.dtype):
        """Build weight components based on configuration."""
        if not self.weight_depends_on_input:
            # Parameter-based weights
            weight_shape = self._get_parameter_shape(self.weight_granularity)
            self.key_weight = nn.Parameter(
                torch.full(weight_shape, weight_init_value, dtype=dtype)
            )
            self.value_weight = nn.Parameter(
                torch.full(weight_shape, weight_init_value, dtype=dtype)
            )
        else:
            # Input-dependent weights via MLP
            input_dim = self._get_weight_input_dim()
            output_dim = self._get_weight_output_dim()

            # Shared hidden layer for efficiency
            self.weight_hidden = ModernMLP(
                input_dim=input_dim,
                output_dim=self.weight_hidden_dim,
                hidden_dim=self.weight_hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype,
            )

            # Separate heads for key and value weights
            self.key_weight_head = nn.Linear(
                self.weight_hidden_dim, output_dim, dtype=dtype
            )
            self.value_weight_head = nn.Linear(
                self.weight_hidden_dim, output_dim, dtype=dtype
            )

    def _get_parameter_shape(self, granularity: str) -> tuple:
        """Get parameter shape based on granularity."""
        if granularity == "scalar":
            return ()  # Scalar
        elif granularity == "token":
            return (
                self.max_sequence_length,
            )  # Token-level parameters with max sequence length
        elif granularity == "head":
            return (
                self.max_sequence_length,
                self.target_num_heads,
            )  # Token and head level parameters
        elif granularity == "head_merged":
            return (
                self.max_sequence_length,
                self.target_num_heads,
            )  # Token and head level parameters
        elif granularity == "value":
            return (
                self.max_sequence_length,
                self.target_num_heads,
                self.target_dim,
            )  # Token, head and value level parameters
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def _get_gate_input_dim(self) -> int:
        """Get input dimension for gate generator."""
        base_dim = 0
        if self.gate_input_features == "target_key":
            base_dim = self.target_dim
        elif self.gate_input_features == "target_value":
            base_dim = self.target_dim
        elif self.gate_input_features == "both":
            base_dim = self.target_dim * 2
        elif self.gate_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key
        elif self.gate_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value
        elif self.gate_input_features == "target_projected_both":
            base_dim = (
                self.target_dim * 4
            )  # target_key + target_value + projected_key + projected_value
        else:
            raise ValueError(f"Invalid gate input features: {self.gate_input_features}")

        # Adjust for granularity processing strategy
        if self.gate_granularity == "scalar":
            # Scalar: process aggregated features across all heads
            return base_dim  # Use pooled features
        elif self.gate_granularity == "token":
            # Token: process merged head dimensions
            return base_dim * self.target_num_heads  # Flatten (H, D) to (H*D)
        elif self.gate_granularity == "head_merged":
            # Head-merged: similar to token granularity, merge H and D
            return base_dim * self.target_num_heads  # (B, N, H*D)
        elif self.gate_granularity == "head":
            # Head-local: per head processing, do not merge heads
            return base_dim  # (B, H, N, D)
        else:  # value
            # Value: process per-head features
            return base_dim  # Keep per-head processing (B, H, N, D)

    def _get_gate_output_dim(self) -> int:
        """Get output dimension for gate generator."""
        if self.gate_granularity == "scalar":
            return 1
        elif self.gate_granularity == "token":
            return 1  # Per token
        elif self.gate_granularity == "head_merged":
            # Per token per head after merge: output one value per head
            return self.target_num_heads
        elif self.gate_granularity == "head":
            # Per token per head: scalar per head
            return 1
        elif self.gate_granularity == "value":
            return (
                self.target_dim
            )  # Per token per head per value (but processed per-head, so output D per head)
        else:
            raise ValueError(f"Invalid gate granularity: {self.gate_granularity}")

    def _get_weight_input_dim(self) -> int:
        """Get input dimension for weight generator."""
        base_dim = 0
        if self.weight_input_features == "target_key":
            base_dim = self.target_dim
        elif self.weight_input_features == "target_value":
            base_dim = self.target_dim
        elif self.weight_input_features == "both":
            base_dim = self.target_dim * 2
        elif self.weight_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key
        elif self.weight_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value
        elif self.weight_input_features == "target_projected_both":
            base_dim = (
                self.target_dim * 4
            )  # target_key + target_value + projected_key + projected_value
        else:
            raise ValueError(
                f"Invalid weight input features: {self.weight_input_features}"
            )

        # Adjust for granularity processing strategy
        if self.weight_granularity == "scalar":
            # Scalar: process aggregated features across all heads
            return base_dim  # Use pooled features
        elif self.weight_granularity == "token":
            # Token: process merged head dimensions
            return base_dim * self.target_num_heads  # Flatten (H, D) to (H*D)
        elif self.weight_granularity == "head_merged":
            # Head-merged: similar to token granularity, merge H and D
            return base_dim * self.target_num_heads  # (B, N, H*D)
        elif self.weight_granularity == "head":
            # Head-local: per head processing, do not merge heads
            return base_dim  # (B, H, N, D)
        else:  # value
            # Value: process per-head features
            return base_dim  # Keep per-head processing (B, H, N, D)

    def _get_weight_output_dim(self) -> int:
        """Get output dimension for weight generator."""
        if self.weight_granularity == "scalar":
            return 1
        elif self.weight_granularity == "token":
            return 1  # Per token
        elif self.weight_granularity == "head_merged":
            # Per token per head after merge: output one value per head
            return self.target_num_heads
        elif self.weight_granularity == "head":
            # Per token per head: scalar per head
            return 1
        elif self.weight_granularity == "value":
            return (
                self.target_dim
            )  # Per token per head per value (but processed per-head, so output D per head)
        else:
            raise ValueError(f"Invalid weight granularity: {self.weight_granularity}")

    def _generate_gates(
        self,
        target_key: Tensor,
        target_value: Tensor,
        projected_key: Tensor = None,
        projected_value: Tensor = None,
    ) -> Tensor:
        """Generate gate logits based on configuration."""
        if not self.gate_depends_on_input:
            # Use parameter-based gate
            return self.gate_logit
        else:
            # Generate input-dependent gate
            # First, prepare the base input features
            if self.gate_input_features == "target_key":
                base_input = target_key
            elif self.gate_input_features == "target_value":
                base_input = target_value
            elif self.gate_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)
            elif self.gate_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError(
                        "projected_key is required for target_projected_key input features"
                    )
                base_input = torch.cat([target_key, projected_key], dim=-1)
            elif self.gate_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError(
                        "projected_value is required for target_projected_value input features"
                    )
                base_input = torch.cat([target_value, projected_value], dim=-1)
            elif self.gate_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError(
                        "Both projected_key and projected_value are required for target_projected_both input features"
                    )
                base_input = torch.cat(
                    [target_key, target_value, projected_key, projected_value], dim=-1
                )

            # Now process based on granularity
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape

            if self.gate_granularity == "scalar":
                # For scalar granularity, aggregate all dimensions: (B, H, N, D_input) -> (B, D_input)
                gate_input = base_input.mean(
                    dim=(1, 2)
                )  # Average over heads and tokens
            elif self.gate_granularity == "token":
                # For token granularity, merge H and D_input dimensions: (B, H, N, D_input) -> (B, N, H*D_input)
                gate_input = (
                    base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
                )
            elif self.gate_granularity == "head_merged":
                # For head granularity, merge H and D like token: (B, H, N, D_in) -> (B, N, H*D_in)
                gate_input = (
                    base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
                )
            elif self.gate_granularity == "head":
                # For head granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input
            elif self.gate_granularity == "value":
                # For value granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input

            return self.gate_generator(gate_input)

    def _generate_weights(
        self,
        target_key: Tensor,
        target_value: Tensor,
        projected_key: Tensor = None,
        projected_value: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate weights based on configuration."""
        if not self.weight_depends_on_input:
            # Use parameter-based weights
            return self.key_weight, self.value_weight
        else:
            # Generate input-dependent weights
            # First, prepare the base input features
            if self.weight_input_features == "target_key":
                base_input = target_key
            elif self.weight_input_features == "target_value":
                base_input = target_value
            elif self.weight_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)
            elif self.weight_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError(
                        "projected_key is required for target_projected_key input features"
                    )
                base_input = torch.cat([target_key, projected_key], dim=-1)
            elif self.weight_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError(
                        "projected_value is required for target_projected_value input features"
                    )
                base_input = torch.cat([target_value, projected_value], dim=-1)
            elif self.weight_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError(
                        "Both projected_key and projected_value are required for target_projected_both input features"
                    )
                base_input = torch.cat(
                    [target_key, target_value, projected_key, projected_value], dim=-1
                )

            # Now process based on granularity
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape

            if self.weight_granularity == "scalar":
                # For scalar granularity, aggregate all dimensions: (B, H, N, D_input) -> (B, D_input)
                weight_input = base_input.mean(
                    dim=(1, 2)
                )  # Average over heads and tokens
            elif self.weight_granularity == "token":
                # For token granularity, merge H and D_input dimensions: (B, H, N, D_input) -> (B, N, H*D_input)
                weight_input = (
                    base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
                )
            elif self.weight_granularity == "head_merged":
                # For head granularity, merge H and D like token: (B, H, N, D_in) -> (B, N, H*D_in)
                weight_input = (
                    base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
                )
            elif self.weight_granularity == "head":
                # For head granularity, keep per-head processing: (B, H, N, D_input)
                weight_input = base_input
            elif self.weight_granularity == "value":
                # For value granularity, keep per-head processing: (B, H, N, D_input)
                weight_input = base_input

            weight_hidden = self.weight_hidden(weight_input)
            key_weight = self.key_weight_head(weight_hidden)
            value_weight = self.value_weight_head(weight_hidden)

            return key_weight, value_weight

    def _apply_gumbel_sigmoid(self, gate_logit: Tensor) -> Tensor:
        """Apply Gumbel sigmoid trick for training."""
        if self.training and self.use_gumbel:
            gumbel_noise = self._sample_gumbel(
                gate_logit.shape, gate_logit.device, gate_logit.dtype
            )
            return torch.sigmoid((gate_logit + gumbel_noise) / self.gate_temperature)
        else:
            return (gate_logit > 0).float()

    @staticmethod
    def _sample_gumbel(
        shape: tuple, device: torch.device, dtype: torch.dtype, eps: float = 1e-20
    ) -> Tensor:
        """Sample from Gumbel distribution."""
        u = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(u + eps) + eps)

    def _reshape_for_granularity(
        self, tensor: Tensor, granularity: str, target_shape: tuple
    ) -> Tensor:
        """Reshape tensor to match target shape based on granularity."""
        B, H, N, D = target_shape

        if granularity == "scalar":
            # Scalar -> (B, H, N, D)
            return tensor.view(1, 1, 1, 1).expand(B, H, N, D)
        elif granularity == "token":
            # (max_seq_len,) -> (B, H, N, D) - slice to actual sequence length
            token_params = tensor[:N]  # Take first N tokens
            return token_params.view(1, 1, N, 1).expand(B, H, N, D)
        elif granularity == "head":
            # (max_seq_len, H) -> (B, H, N, D) - slice to actual sequence length, each token each head independent
            head_params = tensor[:N, :]  # Take first N tokens, all heads: (N, H)
            return (
                head_params.view(1, N, H, 1).transpose(1, 2).expand(B, H, N, D)
            )  # (1, N, H, 1) -> (1, H, N, 1) -> (B, H, N, D)
        elif granularity == "head_merged":
            raise NotImplementedError
        elif granularity == "value":
            # (max_seq_len, H, D) -> (B, H, N, D) - slice to actual sequence length, each token each head each value independent
            value_params = tensor[:N, :, :]  # Take first N tokens: (N, H, D)
            return (
                value_params.view(1, N, H, D).transpose(1, 2).expand(B, H, N, D)
            )  # (1, N, H, D) -> (1, H, N, D) -> (B, H, N, D)
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def update_temperature(self, step: int):
        """Update temperature using exponential annealing schedule for gate only."""
        # Update gate temperature
        gate_ratio = min(step / self.anneal_steps, 1.0)
        gate_temp = (
            self.initial_temperature
            * (self.final_temperature / self.initial_temperature) ** gate_ratio
        )
        self.gate_temperature.fill_(gate_temp)

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with unified projection logic.

        Args:
            source_kv: Tuple of (key, value) tensors, each (B, H_s, N, D_s)
            target_kv: Tuple of (key, value) tensors, each (B, H_t, N, D_t)
            position_ids: Position ids tensor (B, N), optional, required if pos_emb=True
        Returns:
            Tuple of (key, value) tensors, each (B, H_t, N, D_t)
        """
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        # Get shapes
        B, H_s, N, D_s = source_key.shape
        _, H_t, _, D_t = target_key.shape

        # Reshape for projection: (B, H, N, D) -> (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, H_s * D_s)
        source_value_flat = (
            source_value.transpose(1, 2).contiguous().view(B, N, H_s * D_s)
        )

        # Project source to target dimension
        projected_key_flat = self.key_projection(source_key_flat)  # (B, N, H_t * D_t)
        projected_value_flat = self.value_projection(
            source_value_flat
        )  # (B, N, H_t * D_t)

        # Handle concatenation if enabled
        if self.use_concat:
            target_key_flat = (
                target_key.transpose(1, 2).contiguous().view(B, N, H_t * D_t)
            )
            target_value_flat = (
                target_value.transpose(1, 2).contiguous().view(B, N, H_t * D_t)
            )

            # Concatenate and combine
            combined_key = torch.cat([projected_key_flat, target_key_flat], dim=-1)
            combined_value = torch.cat(
                [projected_value_flat, target_value_flat], dim=-1
            )

            final_projected_key_flat = self.key_combiner(combined_key)
            final_projected_value_flat = self.value_combiner(combined_value)
        else:
            final_projected_key_flat = projected_key_flat
            final_projected_value_flat = projected_value_flat

        # Reshape back: (B, N, H_t * D_t) -> (B, H_t, N, D_t)
        projected_key = final_projected_key_flat.view(B, N, H_t, D_t).transpose(1, 2)
        projected_value = final_projected_value_flat.view(B, N, H_t, D_t).transpose(
            1, 2
        )

        # Generate gates and weights (may need projected tensors for input features)
        needs_projected_for_gate = (
            self.gate_depends_on_input
            and self.gate_input_features
            in [
                "target_projected_key",
                "target_projected_value",
                "target_projected_both",
            ]
        )
        needs_projected_for_weight = (
            self.weight_depends_on_input
            and self.weight_input_features
            in [
                "target_projected_key",
                "target_projected_value",
                "target_projected_both",
            ]
        )

        if needs_projected_for_gate or needs_projected_for_weight:
            gate_logit = self._generate_gates(
                target_key, target_value, projected_key, projected_value
            )
            key_weight, value_weight = self._generate_weights(
                target_key, target_value, projected_key, projected_value
            )
        else:
            gate_logit = self._generate_gates(target_key, target_value)
            key_weight, value_weight = self._generate_weights(target_key, target_value)

        # Reshape gates and weights to match target shape
        target_shape = (B, H_t, N, D_t)
        if self.gate_depends_on_input:
            # Reshape based on gate granularity - all preserve token dimension N
            if self.gate_granularity == "scalar":
                # For scalar, gate_logit is already (B, 1) from MLP, just expand
                gate_logit = gate_logit.view(B, 1, 1, 1).expand(target_shape)
            elif self.gate_granularity == "token":
                gate_logit = (
                    gate_logit.unsqueeze(1).unsqueeze(-1).expand(target_shape)
                )  # (B, N, 1) -> (B, H, N, D)
            elif self.gate_granularity == "head_merged":
                # (B, N, H) -> (B, H, N, D) - per token per head, broadcast over D
                gate_logit = (
                    gate_logit.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
                )
            elif self.gate_granularity == "head":
                # (B, H, N, 1) -> (B, H, N, D) - per token per head scalar, broadcast over D
                gate_logit = gate_logit.expand(B, H_t, N, D_t)
            elif self.gate_granularity == "value":
                # (B, H, N, D) -> (B, H, N, D) - each token each head each value has one value
                pass  # Already in correct shape
        else:
            gate_logit = self._reshape_for_granularity(
                gate_logit, self.gate_granularity, target_shape
            )

        if self.weight_depends_on_input:
            # Reshape weights based on granularity - all preserve token dimension N
            if self.weight_granularity == "scalar":
                # For scalar, weights are already (B, 1) from MLP, just expand
                key_weight = key_weight.view(B, 1, 1, 1).expand(target_shape)
                value_weight = value_weight.view(B, 1, 1, 1).expand(target_shape)
            elif self.weight_granularity == "token":
                key_weight = key_weight.unsqueeze(1).expand(
                    target_shape
                )  # (B, N, 1) -> (B, H, N, D)
                value_weight = value_weight.unsqueeze(1).expand(target_shape)
            elif self.weight_granularity == "head_merged":
                # (B, N, H) -> (B, H, N, D) - per token per head, broadcast over D
                key_weight = (
                    key_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
                )
                value_weight = (
                    value_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
                )
            elif self.weight_granularity == "head":
                # (B, H, N, 1) -> (B, H, N, D) - per token per head scalar, broadcast over D
                key_weight = key_weight.expand(B, H_t, N, D_t)
                value_weight = value_weight.expand(B, H_t, N, D_t)
            elif self.weight_granularity == "value":
                # (B, H, N, D) -> (B, H, N, D) - each token each head each value has one value
                pass  # Already in correct shape
        else:
            key_weight = self._reshape_for_granularity(
                key_weight, self.weight_granularity, target_shape
            )
            value_weight = self._reshape_for_granularity(
                value_weight, self.weight_granularity, target_shape
            )

        # Apply gating and selection
        gate = self._apply_gumbel_sigmoid(gate_logit)

        # Normalize weights using dynamic temperature
        normalized_key_weight = torch.sigmoid(key_weight / self.scalar_temperature)
        normalized_value_weight = torch.sigmoid(value_weight / self.scalar_temperature)

        # Final combination
        # Compute projected contribution (always present)
        projected_key_term = gate * normalized_key_weight * projected_key
        projected_value_term = gate * normalized_value_weight * projected_value

        # Compute target (self) contribution depending on flags
        if self.add_self:
            if self.preserve_target_weight:
                target_key_term = (1 - normalized_key_weight) * target_key
                target_value_term = (1 - normalized_value_weight) * target_value
            else:
                target_key_term = target_key
                target_value_term = target_value
        else:
            target_key_term = torch.zeros_like(target_key)
            target_value_term = torch.zeros_like(target_value)

        # Final outputs
        output_key = target_key_term + projected_key_term
        output_value = target_value_term + projected_value_term

        return (output_key, output_value)


class QwenStyleLayer(nn.Module):
    """
    One Qwen3-style MLP sublayer:
      y = x + Dropout( down( SiLU(gate(LN(x))) * up(LN(x)) ) )
    - Pre-norm with RMSNorm
    - Bias-free linears
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.gate(h)) * self.up(h)  # SwiGLU
        h = self.down(h)
        h = self.drop(h)
        return x + h


class StandardFFNLayer(nn.Module):
    """
    Pre-norm RMSNorm, classic MLP:
      y = x + Dropout( W2( Act( W1( RMSNorm(x) ) ) ) )
    - No SwiGLU: single hidden nonlinearity (GELU/ReLU/SiLU)
    - Bias-free linears (common in modern LLM FFNs)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.act(self.w1(h))
        h = self.w2(h)
        h = self.drop(h)
        return x + h


class RegularMLP(nn.Module):
    """
    Qwen3-style stacked MLP operating at a fixed hidden size.
    - No input/output projections; caller is responsible for projections.
    - num_layers repeats of Qwen-style FFN sublayer (pre-RMSNorm, SwiGLU, bias-free)
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 3,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.blocks = nn.ModuleList(
            [
                StandardFFNLayer(
                    hidden_size=hidden_dim,
                    intermediate_size=intermediate_dim,
                    dropout=dropout,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


@register_model
@capture_init_args
class C2CProjector(Projector):
    """
    Concise projector specialized to a fixed C2C configuration using StandardMLP.
    - Projections: StandardMLP (pre-RMSNorm, SwiGLU, residual per sublayer)
    - Concat: enabled, followed by linear combiner to target size
    - Gate: scalar parameter with Gumbel-sigmoid during training
    - Weights: input-dependent, head_merged granularity using target and projected key
    - Target preservation: add_self=True, preserve_target_weight=False
    - Temperatures: annealed gate temperature (1.0 -> 0.001 over 1929 steps), scalar_temperature=1.0
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        alignment_confidence_gate_mode: Literal[
            "none",
            "learned_affine",
            "token_mlp",
        ] = "none",
        alignment_confidence_feature_mode: Literal["none", "quality"] = "none",
        alignment_confidence_max_delta: float = 2.0,
        alignment_confidence_eps: float = 1e-4,
        alignment_confidence_delta_l2_weight: float = 0.0,
        alignment_confidence_delta_l2_mode: Literal["all", "uncertain"] = "all",
        alignment_confidence_delta_l2_confidence_threshold: float = 0.999,
        alignment_confidence_delta_l2_entropy_threshold: float = 0.0,
        alignment_confidence_layer_scale_mode: Literal[
            "none",
            "early_key_late_value",
            "learned",
        ] = "none",
        alignment_confidence_layer_idx: Optional[int] = None,
        alignment_confidence_num_layers: Optional[int] = None,
        alignment_confidence_key_layer_scale_start: float = 1.0,
        alignment_confidence_key_layer_scale_end: float = 1.0,
        alignment_confidence_value_layer_scale_start: float = 1.0,
        alignment_confidence_value_layer_scale_end: float = 1.0,
        alignment_confidence_key_layer_scale_init: float = 1.0,
        alignment_confidence_value_layer_scale_init: float = 1.0,
        alignment_residual_scale_mode: Literal["none", "learned"] = "none",
        alignment_residual_scale_max_delta: float = 1.0,
        alignment_residual_scale_l2_weight: float = 0.0,
        alignment_residual_key_scale_init: float = 1.0,
        alignment_residual_value_scale_init: float = 1.0,
        alignment_weight_calibration_mode: Literal["none", "span_mlp"] = "none",
        alignment_weight_calibration_max_delta: float = 1.0,
        alignment_weight_calibration_eps: float = 1e-8,
        alignment_weight_calibration_apply_mode: Literal[
            "all",
            "ambiguous",
        ] = "all",
        alignment_weight_calibration_entropy_threshold: float = 0.0,
        alignment_weight_calibration_confidence_threshold: float = 0.999,
        alignment_weight_calibration_delta_l2_weight: float = 0.0,
        alignment_weight_calibration_entropy_l2_weight: float = 0.0,
        learned_alignment_mode: Literal["none", "kv_router"] = "none",
        learned_alignment_hidden_dim: int = 128,
        learned_alignment_temperature: float = 1.0,
        learned_alignment_init: Literal["anchor", "uniform"] = "anchor",
        learned_alignment_anchor_logit: float = 2.0,
        learned_alignment_dropout: float = 0.0,
        learned_alignment_prior_mode: Literal["none", "span_log_prior"] = "none",
        learned_alignment_prior_strength: float = 1.0,
        learned_alignment_delta_max: float = 0.0,
        learned_alignment_delta_l2_weight: float = 0.0,
        learned_alignment_prior_ce_weight: float = 0.0,
        learned_alignment_injection_gate_mode: Literal["none", "token_mlp"] = "none",
        learned_alignment_injection_init_logit: float = 0.0,
        learned_alignment_injection_max_delta: float = 2.0,
        learned_alignment_transfer_gate_mode: Literal[
            "none",
            "router_quality",
        ] = "none",
        learned_alignment_transfer_gate_floor: float = 0.0,
        learned_alignment_transfer_gate_entropy_threshold: float = 0.9,
        learned_alignment_transfer_gate_margin_threshold: float = 0.1,
        learned_alignment_transfer_gate_temperature: float = 0.1,
        learned_alignment_transfer_gate_min_valid: int = 2,
        learned_alignment_aux_loss_mode: Literal[
            "none",
            "span_ce",
            "grad_ce",
            "replay_ce",
            "grad_ce_margin_rank",
        ] = "none",
        learned_alignment_aux_loss_weight: float = 0.0,
        learned_alignment_margin_rank_loss_weight: float = 0.0,
        learned_alignment_margin_rank_threshold: float = 0.0,
        learned_alignment_margin_rank_temperature: float = 1.0,
        learned_alignment_margin_rank_scope: Literal[
            "row",
            "batch_mean",
        ] = "row",
        learned_alignment_aux_apply_mode: Literal["all", "ambiguous"] = "ambiguous",
        learned_alignment_aux_target_mode: Literal[
            "source_weights",
            "valid_uniform",
        ] = "source_weights",
        learned_alignment_aux_uniform_mix: float = 0.0,
        learned_alignment_aux_score_temperature: float = 1.0,
        learned_alignment_aux_score_normalize: bool = True,
        learned_alignment_aux_grad_clip: float = 5.0,
        learned_alignment_aux_span_mix: float = 0.0,
        learned_alignment_aux_top_r: int = 0,
        learned_alignment_aux_score_margin_threshold: float = 0.0,
        learned_alignment_aux_eps: float = 1e-8,
        capture_alignment_diagnostics: bool = False,
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"
        valid_confidence_modes = {"none", "learned_affine", "token_mlp"}
        if alignment_confidence_gate_mode not in valid_confidence_modes:
            raise ValueError(
                f"alignment_confidence_gate_mode must be one of "
                f"{sorted(valid_confidence_modes)}, got "
                f"{alignment_confidence_gate_mode}"
            )
        valid_confidence_feature_modes = {"none", "quality"}
        if alignment_confidence_feature_mode not in valid_confidence_feature_modes:
            raise ValueError(
                "alignment_confidence_feature_mode must be one of "
                f"{sorted(valid_confidence_feature_modes)}, got "
                f"{alignment_confidence_feature_mode}"
            )
        if (
            alignment_confidence_feature_mode != "none"
            and alignment_confidence_gate_mode != "token_mlp"
        ):
            raise ValueError(
                "alignment_confidence_feature_mode requires "
                "alignment_confidence_gate_mode='token_mlp'"
            )
        valid_weight_calibration_modes = {"none", "span_mlp"}
        if alignment_weight_calibration_mode not in valid_weight_calibration_modes:
            raise ValueError(
                "alignment_weight_calibration_mode must be one of "
                f"{sorted(valid_weight_calibration_modes)}, got "
                f"{alignment_weight_calibration_mode}"
            )
        valid_learned_alignment_modes = {"none", "kv_router"}
        if learned_alignment_mode not in valid_learned_alignment_modes:
            raise ValueError(
                "learned_alignment_mode must be one of "
                f"{sorted(valid_learned_alignment_modes)}, got "
                f"{learned_alignment_mode}"
            )
        valid_learned_injection_modes = {"none", "token_mlp"}
        if learned_alignment_injection_gate_mode not in valid_learned_injection_modes:
            raise ValueError(
                "learned_alignment_injection_gate_mode must be one of "
                f"{sorted(valid_learned_injection_modes)}, got "
                f"{learned_alignment_injection_gate_mode}"
            )
        if (
            learned_alignment_injection_gate_mode != "none"
            and learned_alignment_mode == "none"
        ):
            raise ValueError(
                "learned_alignment_injection_gate_mode requires "
                "learned_alignment_mode != 'none'"
            )
        valid_learned_transfer_gate_modes = {"none", "router_quality"}
        if (
            learned_alignment_transfer_gate_mode
            not in valid_learned_transfer_gate_modes
        ):
            raise ValueError(
                "learned_alignment_transfer_gate_mode must be one of "
                f"{sorted(valid_learned_transfer_gate_modes)}, got "
                f"{learned_alignment_transfer_gate_mode}"
            )
        if (
            learned_alignment_transfer_gate_mode != "none"
            and learned_alignment_mode == "none"
        ):
            raise ValueError(
                "learned_alignment_transfer_gate_mode requires "
                "learned_alignment_mode != 'none'"
            )
        valid_learned_aux_modes = {
            "none",
            "span_ce",
            "grad_ce",
            "replay_ce",
            "grad_ce_margin_rank",
        }
        if learned_alignment_aux_loss_mode not in valid_learned_aux_modes:
            raise ValueError(
                "learned_alignment_aux_loss_mode must be one of "
                f"{sorted(valid_learned_aux_modes)}, got "
                f"{learned_alignment_aux_loss_mode}"
            )
        valid_learned_aux_apply_modes = {"all", "ambiguous"}
        if learned_alignment_aux_apply_mode not in valid_learned_aux_apply_modes:
            raise ValueError(
                "learned_alignment_aux_apply_mode must be one of "
                f"{sorted(valid_learned_aux_apply_modes)}, got "
                f"{learned_alignment_aux_apply_mode}"
            )
        valid_learned_aux_target_modes = {"source_weights", "valid_uniform"}
        if learned_alignment_aux_target_mode not in valid_learned_aux_target_modes:
            raise ValueError(
                "learned_alignment_aux_target_mode must be one of "
                f"{sorted(valid_learned_aux_target_modes)}, got "
                f"{learned_alignment_aux_target_mode}"
            )
        if (
            learned_alignment_aux_loss_mode != "none"
            and learned_alignment_mode == "none"
        ):
            raise ValueError(
                "learned_alignment_aux_loss_mode requires "
                "learned_alignment_mode != 'none'"
            )
        valid_learned_alignment_inits = {"anchor", "uniform"}
        if learned_alignment_init not in valid_learned_alignment_inits:
            raise ValueError(
                "learned_alignment_init must be one of "
                f"{sorted(valid_learned_alignment_inits)}, got "
                f"{learned_alignment_init}"
            )
        valid_learned_prior_modes = {"none", "span_log_prior"}
        if learned_alignment_prior_mode not in valid_learned_prior_modes:
            raise ValueError(
                "learned_alignment_prior_mode must be one of "
                f"{sorted(valid_learned_prior_modes)}, got "
                f"{learned_alignment_prior_mode}"
            )
        if learned_alignment_prior_mode != "none" and learned_alignment_mode == "none":
            raise ValueError(
                "learned_alignment_prior_mode requires "
                "learned_alignment_mode != 'none'"
            )
        valid_weight_calibration_apply_modes = {"all", "ambiguous"}
        if (
            alignment_weight_calibration_apply_mode
            not in valid_weight_calibration_apply_modes
        ):
            raise ValueError(
                "alignment_weight_calibration_apply_mode must be one of "
                f"{sorted(valid_weight_calibration_apply_modes)}, got "
                f"{alignment_weight_calibration_apply_mode}"
            )
        valid_layer_scale_modes = {"none", "early_key_late_value", "learned"}
        if alignment_confidence_layer_scale_mode not in valid_layer_scale_modes:
            raise ValueError(
                "alignment_confidence_layer_scale_mode must be one of "
                f"{sorted(valid_layer_scale_modes)}, got "
                f"{alignment_confidence_layer_scale_mode}"
            )
        valid_residual_scale_modes = {"none", "learned"}
        if alignment_residual_scale_mode not in valid_residual_scale_modes:
            raise ValueError(
                "alignment_residual_scale_mode must be one of "
                f"{sorted(valid_residual_scale_modes)}, got "
                f"{alignment_residual_scale_mode}"
            )
        valid_delta_l2_modes = {"all", "uncertain"}
        if alignment_confidence_delta_l2_mode not in valid_delta_l2_modes:
            raise ValueError(
                "alignment_confidence_delta_l2_mode must be one of "
                f"{sorted(valid_delta_l2_modes)}, got "
                f"{alignment_confidence_delta_l2_mode}"
            )

        # Dimensions
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.source_num_heads = source_num_heads
        self.target_num_heads = target_num_heads
        self.alignment_confidence_gate_mode = alignment_confidence_gate_mode
        # Evaluation-only causal intervention. This is intentionally not a
        # constructor argument and therefore never becomes part of a training
        # recipe or checkpoint architecture.
        self.alignment_confidence_eval_mode = "learned"
        self.legacy_scalar_gate_eval_mode = "checkpoint_native"
        self.alignment_confidence_feature_mode = alignment_confidence_feature_mode
        self.alignment_confidence_quality_feature_dim = (
            4 if alignment_confidence_feature_mode == "quality" else 0
        )
        self.alignment_confidence_max_delta = float(alignment_confidence_max_delta)
        self.alignment_confidence_eps = float(alignment_confidence_eps)
        self.alignment_confidence_delta_l2_weight = float(
            alignment_confidence_delta_l2_weight
        )
        self.alignment_confidence_delta_l2_mode = alignment_confidence_delta_l2_mode
        self.alignment_confidence_delta_l2_confidence_threshold = float(
            alignment_confidence_delta_l2_confidence_threshold
        )
        self.alignment_confidence_delta_l2_entropy_threshold = float(
            alignment_confidence_delta_l2_entropy_threshold
        )
        self.alignment_confidence_layer_scale_mode = (
            alignment_confidence_layer_scale_mode
        )
        self.alignment_confidence_layer_idx = alignment_confidence_layer_idx
        self.alignment_confidence_num_layers = alignment_confidence_num_layers
        self.alignment_confidence_key_layer_scale_start = float(
            alignment_confidence_key_layer_scale_start
        )
        self.alignment_confidence_key_layer_scale_end = float(
            alignment_confidence_key_layer_scale_end
        )
        self.alignment_confidence_value_layer_scale_start = float(
            alignment_confidence_value_layer_scale_start
        )
        self.alignment_confidence_value_layer_scale_end = float(
            alignment_confidence_value_layer_scale_end
        )
        self.alignment_confidence_key_layer_scale_init = float(
            alignment_confidence_key_layer_scale_init
        )
        self.alignment_confidence_value_layer_scale_init = float(
            alignment_confidence_value_layer_scale_init
        )
        self.alignment_residual_scale_mode = alignment_residual_scale_mode
        self.alignment_residual_scale_max_delta = float(
            alignment_residual_scale_max_delta
        )
        self.alignment_residual_scale_l2_weight = float(
            alignment_residual_scale_l2_weight
        )
        self.alignment_residual_key_scale_init = float(
            alignment_residual_key_scale_init
        )
        self.alignment_residual_value_scale_init = float(
            alignment_residual_value_scale_init
        )
        self.alignment_weight_calibration_mode = alignment_weight_calibration_mode
        self.alignment_weight_calibration_max_delta = float(
            alignment_weight_calibration_max_delta
        )
        self.alignment_weight_calibration_eps = float(alignment_weight_calibration_eps)
        self.alignment_weight_calibration_apply_mode = (
            alignment_weight_calibration_apply_mode
        )
        self.alignment_weight_calibration_entropy_threshold = float(
            alignment_weight_calibration_entropy_threshold
        )
        self.alignment_weight_calibration_confidence_threshold = float(
            alignment_weight_calibration_confidence_threshold
        )
        self.alignment_weight_calibration_delta_l2_weight = float(
            alignment_weight_calibration_delta_l2_weight
        )
        self.alignment_weight_calibration_entropy_l2_weight = float(
            alignment_weight_calibration_entropy_l2_weight
        )
        self.learned_alignment_mode = learned_alignment_mode
        self.learned_alignment_hidden_dim = int(learned_alignment_hidden_dim)
        self.learned_alignment_temperature = float(learned_alignment_temperature)
        self.learned_alignment_init = learned_alignment_init
        self.learned_alignment_anchor_logit = float(learned_alignment_anchor_logit)
        self.learned_alignment_dropout = float(learned_alignment_dropout)
        self.learned_alignment_prior_mode = learned_alignment_prior_mode
        self.learned_alignment_prior_strength = float(learned_alignment_prior_strength)
        self.learned_alignment_delta_max = float(learned_alignment_delta_max)
        self.learned_alignment_delta_l2_weight = float(
            learned_alignment_delta_l2_weight
        )
        self.learned_alignment_prior_ce_weight = float(
            learned_alignment_prior_ce_weight
        )
        self.learned_alignment_injection_gate_mode = (
            learned_alignment_injection_gate_mode
        )
        self.learned_alignment_injection_init_logit = float(
            learned_alignment_injection_init_logit
        )
        self.learned_alignment_injection_max_delta = float(
            learned_alignment_injection_max_delta
        )
        self.learned_alignment_transfer_gate_mode = learned_alignment_transfer_gate_mode
        self.learned_alignment_transfer_gate_floor = float(
            learned_alignment_transfer_gate_floor
        )
        self.learned_alignment_transfer_gate_entropy_threshold = float(
            learned_alignment_transfer_gate_entropy_threshold
        )
        self.learned_alignment_transfer_gate_margin_threshold = float(
            learned_alignment_transfer_gate_margin_threshold
        )
        self.learned_alignment_transfer_gate_temperature = float(
            learned_alignment_transfer_gate_temperature
        )
        self.learned_alignment_transfer_gate_min_valid = int(
            learned_alignment_transfer_gate_min_valid
        )
        self.learned_alignment_aux_loss_mode = learned_alignment_aux_loss_mode
        self.learned_alignment_aux_loss_weight = float(
            learned_alignment_aux_loss_weight
        )
        self.learned_alignment_margin_rank_loss_weight = float(
            learned_alignment_margin_rank_loss_weight
        )
        self.learned_alignment_margin_rank_threshold = float(
            learned_alignment_margin_rank_threshold
        )
        self.learned_alignment_margin_rank_temperature = float(
            learned_alignment_margin_rank_temperature
        )
        self.learned_alignment_margin_rank_scope = learned_alignment_margin_rank_scope
        self.learned_alignment_aux_apply_mode = learned_alignment_aux_apply_mode
        self.learned_alignment_aux_target_mode = learned_alignment_aux_target_mode
        self.learned_alignment_aux_uniform_mix = float(
            learned_alignment_aux_uniform_mix
        )
        self.learned_alignment_aux_score_temperature = float(
            learned_alignment_aux_score_temperature
        )
        self.learned_alignment_aux_score_normalize = bool(
            learned_alignment_aux_score_normalize
        )
        self.learned_alignment_aux_grad_clip = float(learned_alignment_aux_grad_clip)
        self.learned_alignment_aux_span_mix = float(learned_alignment_aux_span_mix)
        self.learned_alignment_aux_top_r = int(learned_alignment_aux_top_r)
        self.learned_alignment_aux_score_margin_threshold = float(
            learned_alignment_aux_score_margin_threshold
        )
        self.learned_alignment_aux_eps = float(learned_alignment_aux_eps)
        self.capture_alignment_diagnostics = bool(capture_alignment_diagnostics)
        if self.alignment_confidence_delta_l2_weight < 0:
            raise ValueError(
                "alignment_confidence_delta_l2_weight must be non-negative, got "
                f"{alignment_confidence_delta_l2_weight}"
            )
        if not 0.0 <= self.alignment_confidence_delta_l2_confidence_threshold <= 1.0:
            raise ValueError(
                "alignment_confidence_delta_l2_confidence_threshold must be in "
                "[0, 1], got "
                f"{alignment_confidence_delta_l2_confidence_threshold}"
            )
        if self.alignment_confidence_delta_l2_entropy_threshold < 0:
            raise ValueError(
                "alignment_confidence_delta_l2_entropy_threshold must be "
                "non-negative, got "
                f"{alignment_confidence_delta_l2_entropy_threshold}"
            )
        if self.alignment_residual_scale_max_delta < 0:
            raise ValueError(
                "alignment_residual_scale_max_delta must be non-negative, got "
                f"{alignment_residual_scale_max_delta}"
            )
        if self.alignment_residual_scale_l2_weight < 0:
            raise ValueError(
                "alignment_residual_scale_l2_weight must be non-negative, got "
                f"{alignment_residual_scale_l2_weight}"
            )
        if self.alignment_residual_key_scale_init < 0:
            raise ValueError(
                "alignment_residual_key_scale_init must be non-negative, got "
                f"{alignment_residual_key_scale_init}"
            )
        if self.alignment_residual_value_scale_init < 0:
            raise ValueError(
                "alignment_residual_value_scale_init must be non-negative, got "
                f"{alignment_residual_value_scale_init}"
            )
        if self.alignment_weight_calibration_max_delta < 0:
            raise ValueError(
                "alignment_weight_calibration_max_delta must be non-negative, got "
                f"{alignment_weight_calibration_max_delta}"
            )
        if self.alignment_weight_calibration_eps <= 0:
            raise ValueError(
                "alignment_weight_calibration_eps must be positive, got "
                f"{alignment_weight_calibration_eps}"
            )
        if self.alignment_weight_calibration_entropy_threshold < 0:
            raise ValueError(
                "alignment_weight_calibration_entropy_threshold must be "
                "non-negative, got "
                f"{alignment_weight_calibration_entropy_threshold}"
            )
        if not 0.0 <= self.alignment_weight_calibration_confidence_threshold <= 1.0:
            raise ValueError(
                "alignment_weight_calibration_confidence_threshold must be in "
                "[0, 1], got "
                f"{alignment_weight_calibration_confidence_threshold}"
            )
        if self.alignment_weight_calibration_delta_l2_weight < 0:
            raise ValueError(
                "alignment_weight_calibration_delta_l2_weight must be "
                "non-negative, got "
                f"{alignment_weight_calibration_delta_l2_weight}"
            )
        if self.alignment_weight_calibration_entropy_l2_weight < 0:
            raise ValueError(
                "alignment_weight_calibration_entropy_l2_weight must be "
                "non-negative, got "
                f"{alignment_weight_calibration_entropy_l2_weight}"
            )
        if self.learned_alignment_hidden_dim <= 0:
            raise ValueError(
                "learned_alignment_hidden_dim must be positive, got "
                f"{learned_alignment_hidden_dim}"
            )
        if self.learned_alignment_temperature <= 0:
            raise ValueError(
                "learned_alignment_temperature must be positive, got "
                f"{learned_alignment_temperature}"
            )
        if self.learned_alignment_dropout < 0:
            raise ValueError(
                "learned_alignment_dropout must be non-negative, got "
                f"{learned_alignment_dropout}"
            )
        if self.learned_alignment_prior_strength < 0:
            raise ValueError(
                "learned_alignment_prior_strength must be non-negative, got "
                f"{learned_alignment_prior_strength}"
            )
        if self.learned_alignment_delta_max < 0:
            raise ValueError(
                "learned_alignment_delta_max must be non-negative, got "
                f"{learned_alignment_delta_max}"
            )
        if self.learned_alignment_delta_l2_weight < 0:
            raise ValueError(
                "learned_alignment_delta_l2_weight must be non-negative, got "
                f"{learned_alignment_delta_l2_weight}"
            )
        if self.learned_alignment_prior_ce_weight < 0:
            raise ValueError(
                "learned_alignment_prior_ce_weight must be non-negative, got "
                f"{learned_alignment_prior_ce_weight}"
            )
        if self.learned_alignment_injection_max_delta < 0:
            raise ValueError(
                "learned_alignment_injection_max_delta must be non-negative, got "
                f"{learned_alignment_injection_max_delta}"
            )
        if not 0.0 <= self.learned_alignment_transfer_gate_floor <= 1.0:
            raise ValueError(
                "learned_alignment_transfer_gate_floor must be in [0, 1], got "
                f"{learned_alignment_transfer_gate_floor}"
            )
        if not 0.0 <= self.learned_alignment_transfer_gate_entropy_threshold <= 1.0:
            raise ValueError(
                "learned_alignment_transfer_gate_entropy_threshold must be in "
                "[0, 1], got "
                f"{learned_alignment_transfer_gate_entropy_threshold}"
            )
        if not 0.0 <= self.learned_alignment_transfer_gate_margin_threshold <= 1.0:
            raise ValueError(
                "learned_alignment_transfer_gate_margin_threshold must be in "
                "[0, 1], got "
                f"{learned_alignment_transfer_gate_margin_threshold}"
            )
        if self.learned_alignment_transfer_gate_temperature <= 0:
            raise ValueError(
                "learned_alignment_transfer_gate_temperature must be positive, got "
                f"{learned_alignment_transfer_gate_temperature}"
            )
        if self.learned_alignment_transfer_gate_min_valid <= 0:
            raise ValueError(
                "learned_alignment_transfer_gate_min_valid must be positive, got "
                f"{learned_alignment_transfer_gate_min_valid}"
            )
        if self.learned_alignment_aux_loss_weight < 0:
            raise ValueError(
                "learned_alignment_aux_loss_weight must be non-negative, got "
                f"{learned_alignment_aux_loss_weight}"
            )
        if self.learned_alignment_margin_rank_loss_weight < 0:
            raise ValueError(
                "learned_alignment_margin_rank_loss_weight must be non-negative, got "
                f"{learned_alignment_margin_rank_loss_weight}"
            )
        if self.learned_alignment_margin_rank_threshold < 0:
            raise ValueError(
                "learned_alignment_margin_rank_threshold must be non-negative, got "
                f"{learned_alignment_margin_rank_threshold}"
            )
        if self.learned_alignment_margin_rank_temperature <= 0:
            raise ValueError(
                "learned_alignment_margin_rank_temperature must be positive, got "
                f"{learned_alignment_margin_rank_temperature}"
            )
        valid_margin_rank_scopes = {"row", "batch_mean"}
        if self.learned_alignment_margin_rank_scope not in valid_margin_rank_scopes:
            raise ValueError(
                "learned_alignment_margin_rank_scope must be one of "
                f"{sorted(valid_margin_rank_scopes)}, got "
                f"{learned_alignment_margin_rank_scope}"
            )
        if not 0.0 <= self.learned_alignment_aux_uniform_mix <= 1.0:
            raise ValueError(
                "learned_alignment_aux_uniform_mix must be in [0, 1], got "
                f"{learned_alignment_aux_uniform_mix}"
            )
        if self.learned_alignment_aux_score_temperature <= 0:
            raise ValueError(
                "learned_alignment_aux_score_temperature must be positive, got "
                f"{learned_alignment_aux_score_temperature}"
            )
        if self.learned_alignment_aux_grad_clip < 0:
            raise ValueError(
                "learned_alignment_aux_grad_clip must be non-negative, got "
                f"{learned_alignment_aux_grad_clip}"
            )
        if not 0.0 <= self.learned_alignment_aux_span_mix <= 1.0:
            raise ValueError(
                "learned_alignment_aux_span_mix must be in [0, 1], got "
                f"{learned_alignment_aux_span_mix}"
            )
        if self.learned_alignment_aux_top_r < 0:
            raise ValueError(
                "learned_alignment_aux_top_r must be non-negative, got "
                f"{learned_alignment_aux_top_r}"
            )
        if self.learned_alignment_aux_score_margin_threshold < 0:
            raise ValueError(
                "learned_alignment_aux_score_margin_threshold must be non-negative, got "
                f"{learned_alignment_aux_score_margin_threshold}"
            )
        if self.learned_alignment_aux_eps <= 0:
            raise ValueError(
                "learned_alignment_aux_eps must be positive, got "
                f"{learned_alignment_aux_eps}"
            )
        (
            self.alignment_confidence_key_token_delta_scale,
            self.alignment_confidence_value_token_delta_scale,
        ) = self._resolve_alignment_layer_scales()
        if self.alignment_confidence_layer_scale_mode == "learned":
            self.alignment_confidence_key_token_delta_scale_param = nn.Parameter(
                torch.tensor(
                    self.alignment_confidence_key_layer_scale_init,
                    dtype=torch.float32,
                )
            )
            self.alignment_confidence_value_token_delta_scale_param = nn.Parameter(
                torch.tensor(
                    self.alignment_confidence_value_layer_scale_init,
                    dtype=torch.float32,
                )
            )
        self.last_alignment_key_layer_scale = (
            self.alignment_confidence_key_token_delta_scale
        )
        self.last_alignment_value_layer_scale = (
            self.alignment_confidence_value_token_delta_scale
        )
        self._last_alignment_confidence_aux_loss: Optional[Tensor] = None
        self._last_alignment_residual_scale_aux_loss: Optional[Tensor] = None
        self._last_alignment_weight_calibration_aux_loss: Optional[Tensor] = None
        self._last_learned_alignment_aux_loss: Optional[Tensor] = None
        self._last_learned_alignment_margin_rank_aux_loss: Optional[Tensor] = None
        self._last_learned_alignment_key_weights_for_aux: Optional[Tensor] = None
        self._last_learned_alignment_value_weights_for_aux: Optional[Tensor] = None
        self._last_learned_alignment_valid_mask_for_aux: Optional[Tensor] = None
        self._last_learned_alignment_source_weights_for_aux: Optional[Tensor] = None
        self._last_learned_alignment_key_transfer_gate: Optional[Tensor] = None
        self._last_learned_alignment_value_transfer_gate: Optional[Tensor] = None
        self._last_learned_alignment_prior_aux_loss: Optional[Tensor] = None
        self._learned_alignment_forced_rank: Optional[int] = None
        self._learned_alignment_replay_target: Optional[Tensor] = None
        self._learned_alignment_replay_utility: Optional[Tensor] = None
        self._learned_alignment_replay_utility_valid: Optional[Tensor] = None
        self._learned_alignment_replay_scoring_mode = False

        # Sizes
        in_dim = source_dim * source_num_heads
        out_dim = target_dim * target_num_heads

        # 1) concat(source_X, target_X) then project to hidden_dim
        self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)

        # 2) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        self.key_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_mlp1 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )

        # 3a) intermediate representation → (L-2)-layer MLP for weights → project to head dim
        self.key_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_scalar_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim,
            num_layers=1,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        # 3b) intermediate representation → (L-2)-layer MLP for projected_X → finally project hidden_dim → out_dim
        self.key_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.value_proj_mlp2 = RegularMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers - 2,
            dropout=dropout,
            dtype=dtype,
        )
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)

        if zero_init:
            print("Initializing projector weights to zero")
            nn.init.zeros_(self.key_proj_out.weight)
            nn.init.zeros_(self.key_proj_out.bias)
            nn.init.zeros_(self.value_proj_out.weight)
            nn.init.zeros_(self.value_proj_out.bias)

        # Scalar key/value gate parameters and temperature schedule
        self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.use_gumbel = True
        self.register_buffer(
            "gate_temperature", torch.tensor(initial_temperature, dtype=dtype)
        )
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps

        # Temperature for weight normalization
        self.scalar_temperature = 1.0

        if self.alignment_confidence_gate_mode != "none":
            self.key_alignment_confidence_bias = nn.Parameter(
                torch.tensor(0.0, dtype=dtype)
            )
            self.value_alignment_confidence_bias = nn.Parameter(
                torch.tensor(0.0, dtype=dtype)
            )
            self.key_alignment_entropy_scale = nn.Parameter(
                torch.tensor(0.0, dtype=dtype)
            )
            self.value_alignment_entropy_scale = nn.Parameter(
                torch.tensor(0.0, dtype=dtype)
            )

        if self.alignment_confidence_gate_mode == "token_mlp":
            confidence_head_input_dim = (
                hidden_dim + self.alignment_confidence_quality_feature_dim
            )
            self.key_alignment_confidence_head = nn.Linear(
                confidence_head_input_dim,
                target_num_heads,
                dtype=dtype,
            )
            self.value_alignment_confidence_head = nn.Linear(
                confidence_head_input_dim,
                target_num_heads,
                dtype=dtype,
            )
            nn.init.zeros_(self.key_alignment_confidence_head.weight)
            nn.init.zeros_(self.key_alignment_confidence_head.bias)
            nn.init.zeros_(self.value_alignment_confidence_head.weight)
            nn.init.zeros_(self.value_alignment_confidence_head.bias)

        if self.alignment_residual_scale_mode == "learned":
            self.alignment_residual_key_scale_delta = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32)
            )
            self.alignment_residual_value_scale_delta = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32)
            )

        if self.alignment_weight_calibration_mode == "span_mlp":
            self.alignment_weight_calibration_head = nn.Linear(
                6,
                1,
                dtype=torch.float32,
            )
            nn.init.zeros_(self.alignment_weight_calibration_head.weight)
            nn.init.zeros_(self.alignment_weight_calibration_head.bias)

        if self.learned_alignment_mode == "kv_router":
            router_dim = self.learned_alignment_hidden_dim
            self.learned_key_alignment_source = nn.Linear(
                in_dim, router_dim, dtype=dtype
            )
            self.learned_key_alignment_target = nn.Linear(
                out_dim, router_dim, dtype=dtype
            )
            self.learned_key_alignment_score = nn.Linear(router_dim, 1, dtype=dtype)
            self.learned_value_alignment_source = nn.Linear(
                in_dim, router_dim, dtype=dtype
            )
            self.learned_value_alignment_target = nn.Linear(
                out_dim, router_dim, dtype=dtype
            )
            self.learned_value_alignment_score = nn.Linear(router_dim, 1, dtype=dtype)
            self.learned_alignment_dropout_module = (
                nn.Dropout(self.learned_alignment_dropout)
                if self.learned_alignment_dropout > 0
                else nn.Identity()
            )
            for module in (
                self.learned_key_alignment_score,
                self.learned_value_alignment_score,
            ):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        if self.learned_alignment_injection_gate_mode == "token_mlp":
            self.learned_key_injection_gate_bias = nn.Parameter(
                torch.tensor(
                    self.learned_alignment_injection_init_logit,
                    dtype=dtype,
                )
            )
            self.learned_value_injection_gate_bias = nn.Parameter(
                torch.tensor(
                    self.learned_alignment_injection_init_logit,
                    dtype=dtype,
                )
            )
            self.learned_key_injection_gate_head = nn.Linear(
                hidden_dim,
                target_num_heads,
                dtype=dtype,
            )
            self.learned_value_injection_gate_head = nn.Linear(
                hidden_dim,
                target_num_heads,
                dtype=dtype,
            )
            nn.init.zeros_(self.learned_key_injection_gate_head.weight)
            nn.init.zeros_(self.learned_key_injection_gate_head.bias)
            nn.init.zeros_(self.learned_value_injection_gate_head.weight)
            nn.init.zeros_(self.learned_value_injection_gate_head.bias)

    def uses_internal_source_confidence(self) -> bool:
        return self.alignment_confidence_gate_mode != "none"

    def set_alignment_confidence_eval_mode(self, mode: str) -> None:
        """Select an evaluation-only view of a trained confidence gate.

        ``learned`` preserves checkpoint behavior and ``static`` uses native
        source confidence without learned deltas. ``forced_on`` forces both
        alignment-confidence and legacy scalar K/V gates; the two optional
        ``*_forced_on`` diagnostic modes isolate either component. Projection
        weights remain untouched in every mode.
        """
        normalized = str(mode).replace("-", "_")
        valid_modes = {
            "learned",
            "static",
            "forced_on",
            "alignment_forced_on",
            "legacy_forced_on",
        }
        if normalized not in valid_modes:
            raise ValueError(
                "alignment confidence eval mode must be one of "
                f"{sorted(valid_modes)}, got {mode!r}"
            )
        alignment_mode = {
            "learned": "learned",
            "static": "static",
            "forced_on": "forced_on",
            "alignment_forced_on": "forced_on",
            "legacy_forced_on": "learned",
        }[normalized]
        if (
            alignment_mode != "learned"
            and self.alignment_confidence_gate_mode == "none"
        ):
            raise ValueError(
                f"cannot apply {normalized!r} to a projector trained without "
                "an alignment confidence gate"
            )
        self.alignment_confidence_eval_mode = alignment_mode
        # The formal forced-on control must remove both gate layers. Static
        # keeps the trained legacy scalar K/V gate exactly as checkpointed.
        self.legacy_scalar_gate_eval_mode = (
            "forced_on"
            if normalized in {"forced_on", "legacy_forced_on"}
            else "checkpoint_native"
        )
        self.gate_eval_intervention_mode = normalized

    def uses_learned_source_alignment(self) -> bool:
        return self.learned_alignment_mode != "none"

    def uses_learned_alignment_injection_gate(self) -> bool:
        return (
            self.learned_alignment_injection_gate_mode != "none"
            or self.learned_alignment_transfer_gate_mode != "none"
        )

    def set_learned_alignment_forced_rank(self, rank: Optional[int]) -> None:
        """Force Route-3 candidate rank during teacher-forced replay."""
        self._learned_alignment_forced_rank = None if rank is None else int(rank)

    def set_learned_alignment_replay_target(
        self,
        target: Optional[Tensor],
    ) -> None:
        """Set batch-level candidate replay target for replay CE."""
        self._learned_alignment_replay_target = (
            None if target is None else target.detach()
        )

    def set_learned_alignment_replay_utility(
        self,
        utility: Optional[Tensor],
    ) -> None:
        """Set batch-level forced-candidate utilities for pairwise router ranking."""
        self._learned_alignment_replay_utility = (
            None if utility is None else utility.detach()
        )
        if utility is None:
            self._learned_alignment_replay_utility_valid = None

    def set_learned_alignment_replay_utility_valid(
        self,
        valid: Optional[Tensor],
    ) -> None:
        """Set a mask for cached forced-candidate utilities."""
        self._learned_alignment_replay_utility_valid = (
            None if valid is None else valid.detach().to(dtype=torch.bool)
        )

    def set_learned_alignment_replay_scoring_mode(self, enabled: bool) -> None:
        """Use deterministic soft gates while scoring forced replay ranks."""
        self._learned_alignment_replay_scoring_mode = bool(enabled)

    @staticmethod
    def _forced_candidate_weights(valid_mask: Tensor, rank: int) -> Tensor:
        valid = valid_mask.to(dtype=torch.bool)
        B, N, K = valid.shape
        weights = torch.zeros(
            B,
            N,
            K,
            device=valid_mask.device,
            dtype=torch.float32,
        )
        if K == 0:
            return weights

        forced_rank = min(max(int(rank), 0), K - 1)
        anchor_rank = torch.zeros(B, N, device=valid_mask.device, dtype=torch.long)
        forced_rank_tensor = torch.full_like(anchor_rank, forced_rank)
        selected_rank = torch.where(
            valid[..., forced_rank], forced_rank_tensor, anchor_rank
        )
        has_selected = valid.gather(-1, selected_rank.unsqueeze(-1)).squeeze(-1)
        weights.scatter_(
            -1, selected_rank.unsqueeze(-1), has_selected.unsqueeze(-1).float()
        )
        return weights

    def _learned_alignment_logits(
        self,
        source_candidates: Tensor,
        target_cache: Tensor,
        source_proj: nn.Linear,
        target_proj: nn.Linear,
        score_head: nn.Linear,
    ) -> Tensor:
        """
        Score top-k source-token KV candidates for each receiver token.

        source_candidates: (B, Hs, N, K, Ds)
        target_cache: (B, Ht, N, Dt)
        returns: (B, N, K)
        """
        B, Hs, N, K, Ds = source_candidates.shape
        _, Ht, _, Dt = target_cache.shape
        source_flat = (
            source_candidates.permute(0, 2, 3, 1, 4).contiguous().view(B, N, K, Hs * Ds)
        )
        target_flat = target_cache.transpose(1, 2).contiguous().view(B, N, Ht * Dt)

        source_hidden = source_proj(source_flat)
        target_hidden = target_proj(target_flat).unsqueeze(2)
        hidden = torch.tanh(source_hidden + target_hidden)
        hidden = self.learned_alignment_dropout_module(hidden)
        return score_head(hidden).squeeze(-1)

    def _masked_candidate_softmax(
        self,
        logits: Tensor,
        valid_mask: Tensor,
        source_weights: Optional[Tensor] = None,
    ) -> Tensor:
        compute_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        logits = logits.to(dtype=compute_dtype)
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        logits = self._bounded_learned_alignment_delta(logits)
        if self.learned_alignment_prior_mode == "span_log_prior":
            prior = self._learned_alignment_auxiliary_target(
                source_weights=source_weights,
                valid_mask=valid_mask,
                compute_dtype=compute_dtype,
            ).to(device=logits.device)
            prior_logits = prior.clamp_min(self.learned_alignment_aux_eps).log()
            prior_logits = prior_logits.masked_fill(~valid, 0.0)
            logits = logits + self.learned_alignment_prior_strength * prior_logits
        elif self.learned_alignment_init == "anchor" and logits.shape[-1] > 0:
            logits = logits.clone()
            logits[..., 0] = logits[..., 0] + self.learned_alignment_anchor_logit
        logits = logits / self.learned_alignment_temperature

        has_valid = valid.any(dim=-1, keepdim=True)
        masked_logits = logits.masked_fill(~valid, -torch.finfo(compute_dtype).max)
        masked_logits = torch.where(
            has_valid,
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        weights = torch.softmax(masked_logits, dim=-1)
        return torch.where(has_valid, weights, torch.zeros_like(weights))

    def _learned_alignment_prior_target(
        self,
        source_weights: Optional[Tensor],
        valid_mask: Tensor,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        return self._learned_alignment_auxiliary_target(
            source_weights=source_weights,
            valid_mask=valid_mask,
            compute_dtype=compute_dtype,
        )

    def _bounded_learned_alignment_delta(self, logits: Tensor) -> Tensor:
        if self.learned_alignment_delta_max <= 0:
            return logits
        return self.learned_alignment_delta_max * torch.tanh(
            logits / self.learned_alignment_delta_max
        )

    def _compute_learned_alignment_prior_regularization(
        self,
        key_logits: Tensor,
        value_logits: Tensor,
        key_weights: Tensor,
        value_weights: Tensor,
        source_weights: Optional[Tensor],
        valid_mask: Tensor,
    ) -> None:
        self._last_learned_alignment_prior_aux_loss = None
        if self.learned_alignment_prior_mode == "none" or (
            self.learned_alignment_delta_l2_weight <= 0
            and self.learned_alignment_prior_ce_weight <= 0
        ):
            return

        compute_dtype = (
            torch.float32
            if key_logits.dtype in (torch.float16, torch.bfloat16)
            else key_logits.dtype
        )
        valid = valid_mask.to(device=key_logits.device, dtype=torch.bool)
        valid_count = valid.sum(dim=-1)
        selected_rows = valid_count > 0
        if self.learned_alignment_aux_apply_mode == "ambiguous":
            selected_rows = selected_rows & (valid_count > 1)
        selected_float = selected_rows.to(dtype=compute_dtype)
        selected_count = selected_float.sum()

        zero = key_logits.to(dtype=compute_dtype).sum() * 0.0
        losses = []
        if self.learned_alignment_delta_l2_weight > 0:
            key_delta = self._bounded_learned_alignment_delta(
                key_logits.to(dtype=compute_dtype)
            ).masked_fill(~valid, 0.0)
            value_delta = self._bounded_learned_alignment_delta(
                value_logits.to(dtype=compute_dtype)
            ).masked_fill(~valid, 0.0)
            row_delta_l2 = 0.5 * (
                key_delta.square().mean(dim=-1) + value_delta.square().mean(dim=-1)
            )
            delta_l2 = (
                (row_delta_l2 * selected_float).sum() / selected_count.clamp_min(1.0)
                if row_delta_l2.numel() > 0
                else zero
            )
            if selected_count.detach().item() == 0:
                delta_l2 = zero
            losses.append(self.learned_alignment_delta_l2_weight * delta_l2)
        else:
            delta_l2 = zero

        if self.learned_alignment_prior_ce_weight > 0:
            prior = self._learned_alignment_prior_target(
                source_weights=source_weights,
                valid_mask=valid_mask,
                compute_dtype=compute_dtype,
            ).to(device=key_weights.device)
            key_probs = key_weights.to(dtype=compute_dtype)
            value_probs = value_weights.to(dtype=compute_dtype)
            key_ce = -(
                prior * key_probs.clamp_min(self.learned_alignment_aux_eps).log()
            ).sum(dim=-1)
            value_ce = -(
                prior * value_probs.clamp_min(self.learned_alignment_aux_eps).log()
            ).sum(dim=-1)
            row_ce = 0.5 * (key_ce + value_ce)
            prior_ce = (
                (row_ce * selected_float).sum() / selected_count.clamp_min(1.0)
                if row_ce.numel() > 0
                else zero
            )
            if selected_count.detach().item() == 0:
                prior_ce = zero
            losses.append(self.learned_alignment_prior_ce_weight * prior_ce)
        else:
            prior = None
            prior_ce = zero

        self._last_learned_alignment_prior_aux_loss = torch.stack(losses).sum()
        try:
            self.last_learned_alignment_prior_aux_loss = float(
                self._last_learned_alignment_prior_aux_loss.detach()
                .float()
                .cpu()
                .item()
            )
            self.last_learned_alignment_delta_l2 = float(
                delta_l2.detach().float().cpu().item()
            )
            self.last_learned_alignment_prior_ce = float(
                prior_ce.detach().float().cpu().item()
            )
            self.last_learned_alignment_prior_selected_rate = float(
                selected_rows.detach().float().mean().cpu().item()
            )
            if prior is not None:
                prior_entropy = self._normalized_source_weight_entropy(prior)
                selected_denom = selected_count.detach().clamp_min(1.0)
                self.last_learned_alignment_prior_entropy = float(
                    (
                        (prior_entropy.detach().float() * selected_float.detach()).sum()
                        / selected_denom
                    )
                    .cpu()
                    .item()
                )
                self.last_learned_alignment_prior_top1 = float(
                    (
                        (
                            prior.max(dim=-1).values.detach().float()
                            * selected_float.detach()
                        ).sum()
                        / selected_denom
                    )
                    .cpu()
                    .item()
                )
        except Exception:
            pass

    def _learned_alignment_auxiliary_target(
        self,
        source_weights: Optional[Tensor],
        valid_mask: Tensor,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        valid = valid_mask.to(dtype=torch.bool)
        valid_float = valid.to(dtype=compute_dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True)
        has_valid = valid_count > 0
        uniform_target = torch.where(
            has_valid,
            valid_float / valid_count.clamp_min(1.0),
            torch.zeros_like(valid_float),
        )

        if (
            self.learned_alignment_aux_target_mode == "valid_uniform"
            or source_weights is None
        ):
            target = uniform_target
        else:
            span_target = source_weights.to(
                device=valid_mask.device,
                dtype=compute_dtype,
            )
            span_target = span_target.masked_fill(~valid, 0.0).clamp_min(0.0)
            span_sum = span_target.sum(dim=-1, keepdim=True)
            target = torch.where(
                span_sum > self.learned_alignment_aux_eps,
                span_target / span_sum.clamp_min(self.learned_alignment_aux_eps),
                uniform_target,
            )

        if self.learned_alignment_aux_uniform_mix > 0:
            mix = self.learned_alignment_aux_uniform_mix
            target = (1.0 - mix) * target + mix * uniform_target
            target = torch.where(has_valid, target, torch.zeros_like(target))

        return target

    def _compute_learned_alignment_auxiliary_loss(
        self,
        key_weights: Tensor,
        value_weights: Tensor,
        source_weights: Optional[Tensor],
        valid_mask: Tensor,
    ) -> None:
        self._last_learned_alignment_aux_loss = None
        if (
            self.learned_alignment_aux_loss_mode == "none"
            or self.learned_alignment_aux_loss_weight <= 0
        ):
            return

        compute_dtype = (
            torch.float32
            if key_weights.dtype in (torch.float16, torch.bfloat16)
            else key_weights.dtype
        )
        key_probs = key_weights.to(dtype=compute_dtype)
        value_probs = value_weights.to(dtype=compute_dtype)
        target = self._learned_alignment_auxiliary_target(
            source_weights=source_weights,
            valid_mask=valid_mask,
            compute_dtype=compute_dtype,
        )
        valid_count = valid_mask.to(dtype=torch.bool).sum(dim=-1)
        selected_rows = valid_count > 0
        if self.learned_alignment_aux_apply_mode == "ambiguous":
            selected_rows = selected_rows & (valid_count > 1)

        key_ce = -(
            target * key_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        value_ce = -(
            target * value_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        row_ce = 0.5 * (key_ce + value_ce)
        selected_float = selected_rows.to(dtype=compute_dtype)
        selected_count = selected_float.sum()
        ce_loss = (
            (row_ce * selected_float).sum() / selected_count.clamp_min(1.0)
            if row_ce.numel() > 0
            else key_probs.sum() * 0.0
        )
        if selected_count.detach().item() == 0:
            ce_loss = row_ce.sum() * 0.0

        weighted_loss = self.learned_alignment_aux_loss_weight * ce_loss
        self._last_learned_alignment_aux_loss = weighted_loss

        try:
            selected_denom = selected_count.detach().clamp_min(1.0)
            target_entropy = self._normalized_source_weight_entropy(target)
            self.last_learned_alignment_aux_loss = float(
                weighted_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_ce = float(
                ce_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_selected_rate = float(
                selected_rows.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_aux_target_entropy = float(
                (
                    (target_entropy.detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_anchor = float(
                (
                    (target[..., 0].detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_top1 = float(
                (
                    (
                        target.max(dim=-1).values.detach().float()
                        * selected_float.detach()
                    ).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_aux_target_tensor = (
                    target.detach().float().cpu()
                )
                self.last_learned_alignment_aux_selected_tensor = (
                    selected_rows.detach().float().cpu()
                )
        except Exception:
            pass

    def _compute_learned_alignment_replay_auxiliary_loss(
        self,
        key_weights: Tensor,
        value_weights: Tensor,
        valid_mask: Tensor,
    ) -> None:
        self._last_learned_alignment_aux_loss = None
        if (
            self.learned_alignment_aux_loss_mode != "replay_ce"
            or self.learned_alignment_aux_loss_weight <= 0
            or self._learned_alignment_replay_target is None
        ):
            return

        compute_dtype = (
            torch.float32
            if key_weights.dtype in (torch.float16, torch.bfloat16)
            else key_weights.dtype
        )
        K = key_weights.shape[-1]
        target = self._learned_alignment_replay_target.to(
            device=key_weights.device,
            dtype=compute_dtype,
        )
        if target.dim() == 1:
            target = target.view(1, 1, -1)
        elif target.dim() == 2:
            target = target.unsqueeze(1)
        if target.shape[-1] < K:
            target = F.pad(target, (0, K - target.shape[-1]))
        elif target.shape[-1] > K:
            target = target[..., :K]

        valid = valid_mask.to(device=key_weights.device, dtype=torch.bool)
        while target.dim() < valid.dim():
            target = target.unsqueeze(0)
        target = target.expand_as(key_weights).masked_fill(~valid, 0.0).clamp_min(0.0)
        target_sum = target.sum(dim=-1, keepdim=True)
        target = torch.where(
            target_sum > self.learned_alignment_aux_eps,
            target / target_sum.clamp_min(self.learned_alignment_aux_eps),
            torch.zeros_like(target),
        )

        valid_count = valid.sum(dim=-1)
        selected_rows = valid_count > 0
        if self.learned_alignment_aux_apply_mode == "ambiguous":
            selected_rows = selected_rows & (valid_count > 1)
        selected_rows = selected_rows & (
            target_sum.squeeze(-1) > self.learned_alignment_aux_eps
        )

        key_probs = key_weights.to(dtype=compute_dtype)
        value_probs = value_weights.to(dtype=compute_dtype)
        key_ce = -(
            target * key_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        value_ce = -(
            target * value_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        row_ce = 0.5 * (key_ce + value_ce)
        selected_float = selected_rows.to(dtype=compute_dtype)
        selected_count = selected_float.sum()
        ce_loss = (
            (row_ce * selected_float).sum() / selected_count.clamp_min(1.0)
            if row_ce.numel() > 0
            else key_probs.sum() * 0.0
        )
        if selected_count.detach().item() == 0:
            ce_loss = row_ce.sum() * 0.0

        weighted_loss = self.learned_alignment_aux_loss_weight * ce_loss
        self._last_learned_alignment_aux_loss = weighted_loss

        try:
            selected_denom = selected_count.detach().clamp_min(1.0)
            target_entropy = self._normalized_source_weight_entropy(target)
            target_top2 = torch.topk(target, k=min(2, K), dim=-1).values
            if target_top2.shape[-1] == 1:
                target_margin = torch.zeros_like(target_top2[..., 0])
            else:
                target_margin = target_top2[..., 0] - target_top2[..., 1]
            self.last_learned_alignment_aux_loss = float(
                weighted_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_ce = float(
                ce_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_selected_rate = float(
                selected_rows.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_aux_target_entropy = float(
                (
                    (target_entropy.detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_anchor = float(
                (
                    (target[..., 0].detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_top1 = float(
                (
                    (
                        target.max(dim=-1).values.detach().float()
                        * selected_float.detach()
                    ).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_score_margin = float(
                (
                    (target_margin.detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_aux_target_tensor = (
                    target.detach().float().cpu()
                )
                self.last_learned_alignment_aux_selected_tensor = (
                    selected_rows.detach().float().cpu()
                )
        except Exception:
            pass

    def _clear_learned_alignment_aux_state(self) -> None:
        self._last_learned_alignment_aux_loss = None
        self._last_learned_alignment_margin_rank_aux_loss = None
        self._last_learned_alignment_key_weights_for_aux = None
        self._last_learned_alignment_value_weights_for_aux = None
        self._last_learned_alignment_valid_mask_for_aux = None
        self._last_learned_alignment_source_weights_for_aux = None
        self._last_learned_alignment_key_transfer_gate = None
        self._last_learned_alignment_value_transfer_gate = None
        self._last_learned_alignment_prior_aux_loss = None

    def _store_learned_alignment_aux_state(
        self,
        key_weights: Tensor,
        value_weights: Tensor,
        valid_mask: Tensor,
        source_weights: Optional[Tensor],
    ) -> None:
        if (
            self.learned_alignment_aux_loss_mode
            not in {"grad_ce", "grad_ce_margin_rank"}
            or self.learned_alignment_aux_loss_weight <= 0
        ):
            return
        self._last_learned_alignment_key_weights_for_aux = key_weights
        self._last_learned_alignment_value_weights_for_aux = value_weights
        self._last_learned_alignment_valid_mask_for_aux = valid_mask.detach()
        self._last_learned_alignment_source_weights_for_aux = (
            source_weights.detach() if source_weights is not None else None
        )

    def _cached_replay_target_for_candidates(
        self,
        weights: Tensor,
        valid_mask: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if self._learned_alignment_replay_target is None:
            return None

        compute_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        K = weights.shape[-1]
        target = self._learned_alignment_replay_target.to(
            device=weights.device,
            dtype=compute_dtype,
        )
        if target.dim() == 1:
            target = target.view(1, 1, -1)
        elif target.dim() == 2:
            target = target.unsqueeze(1)
        if target.shape[-1] < K:
            target = F.pad(target, (0, K - target.shape[-1]))
        elif target.shape[-1] > K:
            target = target[..., :K]

        valid = valid_mask.to(device=weights.device, dtype=torch.bool)
        while target.dim() < valid.dim():
            target = target.unsqueeze(0)
        target = target.expand_as(weights).masked_fill(~valid, 0.0).clamp_min(0.0)
        target_sum = target.sum(dim=-1, keepdim=True)
        target = torch.where(
            target_sum > self.learned_alignment_aux_eps,
            target / target_sum.clamp_min(self.learned_alignment_aux_eps),
            torch.zeros_like(target),
        )
        return target.detach(), (
            target_sum.squeeze(-1) > self.learned_alignment_aux_eps
        )

    def _candidate_target_from_loss_gradient(
        self,
        weights: Tensor,
        grad: Tensor,
        valid_mask: Tensor,
        source_weights: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        compute_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        valid = valid_mask.to(device=weights.device, dtype=torch.bool)
        score = -grad.detach().to(device=weights.device, dtype=compute_dtype)
        score = score.masked_fill(~valid, -torch.finfo(compute_dtype).max)

        if self.learned_alignment_aux_score_normalize:
            finite_score = torch.where(valid, score, torch.zeros_like(score))
            valid_count = valid.to(dtype=compute_dtype).sum(dim=-1, keepdim=True)
            mean = finite_score.sum(dim=-1, keepdim=True) / valid_count.clamp_min(1.0)
            centered = torch.where(valid, score - mean, torch.zeros_like(score))
            variance = centered.square().sum(
                dim=-1, keepdim=True
            ) / valid_count.clamp_min(1.0)
            normalized = (
                centered / variance.clamp_min(self.learned_alignment_aux_eps).sqrt()
            )
            score = torch.where(valid, normalized, score)

        if self.learned_alignment_aux_grad_clip > 0:
            clip_value = self.learned_alignment_aux_grad_clip
            score = torch.where(valid, score.clamp(-clip_value, clip_value), score)

        has_valid = valid.any(dim=-1, keepdim=True)
        target = torch.softmax(
            score / self.learned_alignment_aux_score_temperature,
            dim=-1,
        )
        target = torch.where(has_valid, target, torch.zeros_like(target))

        if self.learned_alignment_aux_top_r > 0 and target.shape[-1] > 1:
            top_r = min(self.learned_alignment_aux_top_r, target.shape[-1])
            top_indices = torch.topk(
                score,
                k=top_r,
                dim=-1,
            ).indices
            sparse_mask = torch.zeros_like(target, dtype=torch.bool)
            sparse_mask.scatter_(-1, top_indices, True)
            sparse_mask = sparse_mask & valid
            sparse_target = torch.where(sparse_mask, target, torch.zeros_like(target))
            sparse_denom = sparse_target.sum(dim=-1, keepdim=True)
            target = torch.where(
                sparse_denom > 0,
                sparse_target / sparse_denom.clamp_min(self.learned_alignment_aux_eps),
                target,
            )

        if self.learned_alignment_aux_span_mix > 0:
            span_target = self._learned_alignment_auxiliary_target(
                source_weights=source_weights,
                valid_mask=valid_mask,
                compute_dtype=compute_dtype,
            )
            mix = self.learned_alignment_aux_span_mix
            target = (1.0 - mix) * target + mix * span_target
            target = torch.where(has_valid, target, torch.zeros_like(target))

        return target.detach(), score.detach()

    def _candidate_score_margin(self, score: Tensor, valid_mask: Tensor) -> Tensor:
        valid = valid_mask.to(device=score.device, dtype=torch.bool)
        masked_score = score.masked_fill(~valid, -torch.finfo(score.dtype).max)
        if score.shape[-1] <= 1:
            return torch.zeros_like(score[..., 0])
        top2 = torch.topk(masked_score, k=2, dim=-1).values
        return top2[..., 0] - top2[..., 1]

    def compute_learned_alignment_grad_auxiliary_loss(
        self,
        task_loss: Tensor,
    ) -> Optional[Tensor]:
        if (
            self.learned_alignment_aux_loss_mode
            not in {"grad_ce", "grad_ce_margin_rank"}
            or self.learned_alignment_aux_loss_weight <= 0
            or self._last_learned_alignment_key_weights_for_aux is None
            or self._last_learned_alignment_value_weights_for_aux is None
            or self._last_learned_alignment_valid_mask_for_aux is None
        ):
            return None

        key_weights = self._last_learned_alignment_key_weights_for_aux
        value_weights = self._last_learned_alignment_value_weights_for_aux
        valid_mask = self._last_learned_alignment_valid_mask_for_aux.to(
            device=key_weights.device
        )
        source_weights = self._last_learned_alignment_source_weights_for_aux
        if source_weights is not None:
            source_weights = source_weights.to(device=key_weights.device)

        key_grad, value_grad = torch.autograd.grad(
            task_loss,
            (key_weights, value_weights),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        if key_grad is None or value_grad is None:
            return None

        cached_target_state = self._cached_replay_target_for_candidates(
            weights=key_weights,
            valid_mask=valid_mask,
        )
        if cached_target_state is None:
            key_target, key_score = self._candidate_target_from_loss_gradient(
                weights=key_weights,
                grad=key_grad,
                valid_mask=valid_mask,
                source_weights=source_weights,
            )
            value_target, value_score = self._candidate_target_from_loss_gradient(
                weights=value_weights,
                grad=value_grad,
                valid_mask=valid_mask,
                source_weights=source_weights,
            )
            selected_from_cache = None
        else:
            key_target, selected_from_cache = cached_target_state
            value_target = key_target
            key_score = key_target
            value_score = value_target

        compute_dtype = (
            torch.float32
            if key_weights.dtype in (torch.float16, torch.bfloat16)
            else key_weights.dtype
        )
        key_probs = key_weights.to(dtype=compute_dtype)
        value_probs = value_weights.to(dtype=compute_dtype)
        key_ce = -(
            key_target * key_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        value_ce = -(
            value_target * value_probs.clamp_min(self.learned_alignment_aux_eps).log()
        ).sum(dim=-1)
        row_ce = 0.5 * (key_ce + value_ce)

        valid_count = valid_mask.to(dtype=torch.bool).sum(dim=-1)
        selected_rows = valid_count > 0
        if self.learned_alignment_aux_apply_mode == "ambiguous":
            selected_rows = selected_rows & (valid_count > 1)
        if selected_from_cache is not None:
            selected_rows = selected_rows & selected_from_cache.to(
                device=selected_rows.device,
                dtype=torch.bool,
            )
        key_margin = self._candidate_score_margin(key_score, valid_mask)
        value_margin = self._candidate_score_margin(value_score, valid_mask)
        margin = 0.5 * (key_margin + value_margin)
        if self.learned_alignment_aux_score_margin_threshold > 0:
            selected_rows = selected_rows & (
                margin >= self.learned_alignment_aux_score_margin_threshold
            )
        selected_float = selected_rows.to(dtype=compute_dtype)
        selected_count = selected_float.sum()
        ce_loss = (
            (row_ce * selected_float).sum() / selected_count.clamp_min(1.0)
            if row_ce.numel() > 0
            else key_probs.sum() * 0.0
        )
        if selected_count.detach().item() == 0:
            ce_loss = row_ce.sum() * 0.0

        weighted_loss = self.learned_alignment_aux_loss_weight * ce_loss
        self._last_learned_alignment_aux_loss = weighted_loss

        try:
            selected_denom = selected_count.detach().clamp_min(1.0)
            target = 0.5 * (key_target + value_target)
            target_entropy = self._normalized_source_weight_entropy(target)
            self.last_learned_alignment_aux_loss = float(
                weighted_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_ce = float(
                ce_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_aux_selected_rate = float(
                selected_rows.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_aux_target_entropy = float(
                (
                    (target_entropy.detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_anchor = float(
                (
                    (target[..., 0].detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_target_top1 = float(
                (
                    (
                        target.max(dim=-1).values.detach().float()
                        * selected_float.detach()
                    ).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_aux_score_margin = float(
                (
                    (margin.detach().float() * selected_float.detach()).sum()
                    / selected_denom
                )
                .cpu()
                .item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_aux_target_tensor = (
                    target.detach().float().cpu()
                )
                self.last_learned_alignment_aux_selected_tensor = (
                    selected_rows.detach().float().cpu()
                )
        except Exception:
            pass

        return weighted_loss

    def _expand_replay_utility_for_candidates(
        self,
        weights: Tensor,
        valid_mask: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if self._learned_alignment_replay_utility is None:
            return None

        compute_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        K = weights.shape[-1]
        utility = self._learned_alignment_replay_utility.to(
            device=weights.device,
            dtype=compute_dtype,
        )
        utility_valid: Optional[Tensor] = None
        if self._learned_alignment_replay_utility_valid is not None:
            utility_valid = self._learned_alignment_replay_utility_valid.to(
                device=weights.device,
                dtype=torch.bool,
            )
        if utility.dim() == 1:
            utility = utility.view(1, 1, -1)
            if utility_valid is not None:
                utility_valid = utility_valid.view(1, 1, -1)
        elif utility.dim() == 2:
            utility = utility.unsqueeze(1)
            if utility_valid is not None:
                utility_valid = utility_valid.unsqueeze(1)
        if utility_valid is None:
            utility_valid = torch.ones_like(utility, dtype=torch.bool)
        if utility.shape[-1] < K:
            pad = K - utility.shape[-1]
            utility = F.pad(utility, (0, pad))
            utility_valid = F.pad(utility_valid, (0, pad), value=False)
        elif utility.shape[-1] > K:
            utility = utility[..., :K]
            utility_valid = utility_valid[..., :K]

        valid = valid_mask.to(device=weights.device, dtype=torch.bool)
        while utility.dim() < valid.dim():
            utility = utility.unsqueeze(0)
            utility_valid = utility_valid.unsqueeze(0)
        return utility.expand_as(weights), utility_valid.expand_as(weights)

    def _pairwise_margin_rank_loss(
        self,
        probs: Tensor,
        utility: Tensor,
        valid_mask: Tensor,
        selected_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        compute_dtype = (
            torch.float32
            if probs.dtype in (torch.float16, torch.bfloat16)
            else probs.dtype
        )
        probs = probs.to(dtype=compute_dtype)
        utility = utility.to(device=probs.device, dtype=compute_dtype)
        valid = valid_mask.to(device=probs.device, dtype=torch.bool)
        log_probs = probs.clamp_min(self.learned_alignment_aux_eps).log()

        utility_diff = utility.unsqueeze(-1) - utility.unsqueeze(-2)
        row_logit_diff = log_probs.unsqueeze(-1) - log_probs.unsqueeze(-2)
        valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        pair_mask = (
            valid_pair
            & selected_rows.unsqueeze(-1).unsqueeze(-1)
            & (utility_diff > self.learned_alignment_margin_rank_threshold)
        )

        pair_loss = F.softplus(
            -row_logit_diff / self.learned_alignment_margin_rank_temperature
        )
        pair_float = pair_mask.to(dtype=compute_dtype)
        pair_count = pair_float.sum()
        if pair_count.detach().item() == 0:
            return pair_loss.sum() * 0.0, pair_count
        return (pair_loss * pair_float).sum() / pair_count.clamp_min(1.0), pair_count

    def _batch_mean_margin_rank_loss(
        self,
        probs: Tensor,
        utility: Tensor,
        valid_mask: Tensor,
        selected_rows: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        compute_dtype = (
            torch.float32
            if probs.dtype in (torch.float16, torch.bfloat16)
            else probs.dtype
        )
        probs = probs.to(dtype=compute_dtype)
        utility = utility.to(device=probs.device, dtype=compute_dtype)
        valid = valid_mask.to(device=probs.device, dtype=torch.bool)
        row_weight = (valid & selected_rows.unsqueeze(-1)).to(dtype=compute_dtype)
        denom = row_weight.sum(dim=(0, 1)).clamp_min(1.0)
        mass = (probs * row_weight).sum(dim=(0, 1)) / denom
        mean_utility = (utility * row_weight).sum(dim=(0, 1)) / denom
        present = row_weight.sum(dim=(0, 1)) > 0

        utility_diff = mean_utility.unsqueeze(-1) - mean_utility.unsqueeze(-2)
        logit_diff = mass.clamp_min(self.learned_alignment_aux_eps).log().unsqueeze(
            -1
        ) - mass.clamp_min(self.learned_alignment_aux_eps).log().unsqueeze(-2)
        pair_mask = (
            present.unsqueeze(-1)
            & present.unsqueeze(-2)
            & (utility_diff > self.learned_alignment_margin_rank_threshold)
        )
        pair_loss = F.softplus(
            -logit_diff / self.learned_alignment_margin_rank_temperature
        )
        pair_float = pair_mask.to(dtype=compute_dtype)
        pair_count = pair_float.sum()
        if pair_count.detach().item() == 0:
            return pair_loss.sum() * 0.0, pair_count
        return (pair_loss * pair_float).sum() / pair_count.clamp_min(1.0), pair_count

    def _compute_learned_alignment_margin_rank_auxiliary_loss(
        self,
        key_weights: Tensor,
        value_weights: Tensor,
        valid_mask: Tensor,
    ) -> None:
        self._last_learned_alignment_margin_rank_aux_loss = None
        if (
            self.learned_alignment_aux_loss_mode != "grad_ce_margin_rank"
            or self.learned_alignment_margin_rank_loss_weight <= 0
        ):
            return

        utility_state = self._expand_replay_utility_for_candidates(
            weights=key_weights,
            valid_mask=valid_mask,
        )
        if utility_state is None:
            return
        utility, utility_valid = utility_state

        compute_dtype = (
            torch.float32
            if key_weights.dtype in (torch.float16, torch.bfloat16)
            else key_weights.dtype
        )
        valid = valid_mask.to(device=key_weights.device, dtype=torch.bool)
        rank_valid = valid & utility_valid.to(device=valid.device, dtype=torch.bool)
        valid_count = rank_valid.sum(dim=-1)
        selected_rows = valid_count > 1
        if self.learned_alignment_aux_apply_mode == "all":
            selected_rows = valid_count > 0

        rank_loss_fn = (
            self._batch_mean_margin_rank_loss
            if self.learned_alignment_margin_rank_scope == "batch_mean"
            else self._pairwise_margin_rank_loss
        )
        key_rank_loss, key_pair_count = rank_loss_fn(
            probs=key_weights,
            utility=utility,
            valid_mask=rank_valid,
            selected_rows=selected_rows,
        )
        value_rank_loss, value_pair_count = rank_loss_fn(
            probs=value_weights,
            utility=utility,
            valid_mask=rank_valid,
            selected_rows=selected_rows,
        )
        rank_loss = 0.5 * (key_rank_loss + value_rank_loss)
        weighted_loss = self.learned_alignment_margin_rank_loss_weight * rank_loss
        self._last_learned_alignment_margin_rank_aux_loss = weighted_loss

        try:
            selected_float = selected_rows.to(dtype=compute_dtype)
            selected_count = selected_float.sum().detach().clamp_min(1.0)
            masked_utility = utility.masked_fill(
                ~rank_valid,
                -torch.finfo(compute_dtype).max,
            )
            top2 = torch.topk(
                masked_utility, k=min(2, masked_utility.shape[-1]), dim=-1
            ).values
            if top2.shape[-1] == 1:
                utility_margin = torch.zeros_like(top2[..., 0])
            else:
                utility_margin = top2[..., 0] - top2[..., 1]
            utility_top1 = masked_utility.argmax(dim=-1).to(dtype=compute_dtype)
            self.last_learned_alignment_margin_rank_aux_loss = float(
                weighted_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_margin_rank_loss = float(
                rank_loss.detach().float().cpu().item()
            )
            self.last_learned_alignment_margin_rank_selected_rate = float(
                selected_rows.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_margin_rank_pair_count = float(
                (0.5 * (key_pair_count + value_pair_count))
                .detach()
                .float()
                .cpu()
                .item()
            )
            self.last_learned_alignment_margin_rank_utility_margin = float(
                (
                    (utility_margin.detach().float() * selected_float.detach()).sum()
                    / selected_count
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_margin_rank_utility_top1 = float(
                (
                    (utility_top1.detach().float() * selected_float.detach()).sum()
                    / selected_count
                )
                .cpu()
                .item()
            )
            self.last_learned_alignment_margin_rank_utility_anchor = float(
                (
                    (utility[..., 0].detach().float() * selected_float.detach()).sum()
                    / selected_count
                )
                .cpu()
                .item()
            )
        except Exception:
            pass

    def align_source_kv(
        self,
        source_kv_candidates: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        valid_mask: Tensor,
        source_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.learned_alignment_mode != "kv_router":
            return super().align_source_kv(
                source_kv_candidates=source_kv_candidates,
                target_kv=target_kv,
                valid_mask=valid_mask,
                source_weights=source_weights,
            )

        self._clear_learned_alignment_aux_state()
        source_key_candidates, source_value_candidates = source_kv_candidates
        target_key, target_value = target_kv
        if self._learned_alignment_forced_rank is not None:
            key_weights = self._forced_candidate_weights(
                valid_mask,
                self._learned_alignment_forced_rank,
            )
            value_weights = key_weights
            key = (
                source_key_candidates
                * key_weights.to(dtype=source_key_candidates.dtype)[:, None, :, :, None]
            ).sum(dim=3)
            value = (
                source_value_candidates
                * value_weights.to(dtype=source_value_candidates.dtype)[
                    :, None, :, :, None
                ]
            ).sum(dim=3)
            self._store_learned_alignment_transfer_gate(
                key_weights=key_weights,
                value_weights=value_weights,
                valid_mask=valid_mask,
                target_shape=target_key.shape,
                dtype=target_key.dtype,
                device=target_key.device,
            )
            return key, value

        key_logits = self._learned_alignment_logits(
            source_candidates=source_key_candidates,
            target_cache=target_key,
            source_proj=self.learned_key_alignment_source,
            target_proj=self.learned_key_alignment_target,
            score_head=self.learned_key_alignment_score,
        )
        value_logits = self._learned_alignment_logits(
            source_candidates=source_value_candidates,
            target_cache=target_value,
            source_proj=self.learned_value_alignment_source,
            target_proj=self.learned_value_alignment_target,
            score_head=self.learned_value_alignment_score,
        )
        key_weights = self._masked_candidate_softmax(
            key_logits,
            valid_mask,
            source_weights=source_weights,
        )
        value_weights = self._masked_candidate_softmax(
            value_logits,
            valid_mask,
            source_weights=source_weights,
        )
        self._compute_learned_alignment_prior_regularization(
            key_logits=key_logits,
            value_logits=value_logits,
            key_weights=key_weights,
            value_weights=value_weights,
            source_weights=source_weights,
            valid_mask=valid_mask,
        )
        self._store_learned_alignment_transfer_gate(
            key_weights=key_weights,
            value_weights=value_weights,
            valid_mask=valid_mask,
            target_shape=target_key.shape,
            dtype=target_key.dtype,
            device=target_key.device,
        )
        if self.learned_alignment_aux_loss_mode in {"grad_ce", "grad_ce_margin_rank"}:
            self._store_learned_alignment_aux_state(
                key_weights=key_weights,
                value_weights=value_weights,
                valid_mask=valid_mask,
                source_weights=source_weights,
            )
            self._compute_learned_alignment_margin_rank_auxiliary_loss(
                key_weights=key_weights,
                value_weights=value_weights,
                valid_mask=valid_mask,
            )
        elif self.learned_alignment_aux_loss_mode == "replay_ce":
            self._compute_learned_alignment_replay_auxiliary_loss(
                key_weights=key_weights,
                value_weights=value_weights,
                valid_mask=valid_mask,
            )
        else:
            self._compute_learned_alignment_auxiliary_loss(
                key_weights=key_weights,
                value_weights=value_weights,
                source_weights=source_weights,
                valid_mask=valid_mask,
            )

        key = (
            source_key_candidates
            * key_weights.to(dtype=source_key_candidates.dtype)[:, None, :, :, None]
        ).sum(dim=3)
        value = (
            source_value_candidates
            * value_weights.to(dtype=source_value_candidates.dtype)[:, None, :, :, None]
        ).sum(dim=3)

        try:
            key_entropy = self._normalized_source_weight_entropy(key_weights)
            value_entropy = self._normalized_source_weight_entropy(value_weights)
            self.last_learned_alignment_key_entropy = float(
                key_entropy.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_entropy = float(
                value_entropy.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_top1_mean = float(
                key_weights.max(dim=-1).values.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_top1_mean = float(
                value_weights.max(dim=-1).values.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_anchor_mean = float(
                key_weights[..., 0].detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_anchor_mean = float(
                value_weights[..., 0].detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_valid_rate = float(
                valid_mask.detach().float().mean().cpu().item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_key_weights_tensor = (
                    key_weights.detach().float().cpu()
                )
                self.last_learned_alignment_value_weights_tensor = (
                    value_weights.detach().float().cpu()
                )
        except Exception:
            pass

        return key, value

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "alignment_confidence_key_token_delta_scale_param"):
            self.alignment_confidence_key_token_delta_scale_param.data = (
                self.alignment_confidence_key_token_delta_scale_param.data.float()
            )
            self.alignment_confidence_value_token_delta_scale_param.data = (
                self.alignment_confidence_value_token_delta_scale_param.data.float()
            )
            if self.alignment_confidence_key_token_delta_scale_param.grad is not None:
                self.alignment_confidence_key_token_delta_scale_param.grad.data = (
                    self.alignment_confidence_key_token_delta_scale_param.grad.data.float()
                )
            if self.alignment_confidence_value_token_delta_scale_param.grad is not None:
                self.alignment_confidence_value_token_delta_scale_param.grad.data = (
                    self.alignment_confidence_value_token_delta_scale_param.grad.data.float()
                )
        if hasattr(self, "alignment_residual_key_scale_delta"):
            self.alignment_residual_key_scale_delta.data = (
                self.alignment_residual_key_scale_delta.data.float()
            )
            self.alignment_residual_value_scale_delta.data = (
                self.alignment_residual_value_scale_delta.data.float()
            )
            if self.alignment_residual_key_scale_delta.grad is not None:
                self.alignment_residual_key_scale_delta.grad.data = (
                    self.alignment_residual_key_scale_delta.grad.data.float()
                )
            if self.alignment_residual_value_scale_delta.grad is not None:
                self.alignment_residual_value_scale_delta.grad.data = (
                    self.alignment_residual_value_scale_delta.grad.data.float()
                )
        if hasattr(self, "alignment_weight_calibration_head"):
            self.alignment_weight_calibration_head.weight.data = (
                self.alignment_weight_calibration_head.weight.data.float()
            )
            self.alignment_weight_calibration_head.bias.data = (
                self.alignment_weight_calibration_head.bias.data.float()
            )
            if self.alignment_weight_calibration_head.weight.grad is not None:
                self.alignment_weight_calibration_head.weight.grad.data = (
                    self.alignment_weight_calibration_head.weight.grad.data.float()
                )
            if self.alignment_weight_calibration_head.bias.grad is not None:
                self.alignment_weight_calibration_head.bias.grad.data = (
                    self.alignment_weight_calibration_head.bias.grad.data.float()
                )
        return self

    def alignment_regularization_loss(self) -> Optional[Tensor]:
        losses = [
            loss
            for loss in (
                self._last_alignment_confidence_aux_loss,
                self._last_alignment_residual_scale_aux_loss,
                self._last_alignment_weight_calibration_aux_loss,
                self._last_learned_alignment_aux_loss,
                self._last_learned_alignment_margin_rank_aux_loss,
                self._last_learned_alignment_prior_aux_loss,
            )
            if loss is not None
        ]
        if not losses:
            return None
        return torch.stack(losses).sum()

    def update_temperature(self, step: int):
        ratio = min(step / self.anneal_steps, 1.0)
        temp = (
            self.initial_temperature
            * (self.final_temperature / self.initial_temperature) ** ratio
        )
        self.gate_temperature.fill_(temp)

    def _resolve_alignment_layer_scales(self) -> Tuple[float, float]:
        if self.alignment_confidence_layer_scale_mode == "none":
            return 1.0, 1.0

        if self.alignment_confidence_layer_scale_mode == "learned":
            return (
                self.alignment_confidence_key_layer_scale_init,
                self.alignment_confidence_value_layer_scale_init,
            )

        if self.alignment_confidence_layer_idx is None:
            raise ValueError(
                "alignment_confidence_layer_idx is required when "
                "alignment_confidence_layer_scale_mode is not 'none'"
            )
        if self.alignment_confidence_num_layers is None:
            raise ValueError(
                "alignment_confidence_num_layers is required when "
                "alignment_confidence_layer_scale_mode is not 'none'"
            )
        if self.alignment_confidence_num_layers <= 0:
            raise ValueError(
                "alignment_confidence_num_layers must be positive, got "
                f"{self.alignment_confidence_num_layers}"
            )
        if (
            not 0
            <= self.alignment_confidence_layer_idx
            < self.alignment_confidence_num_layers
        ):
            raise ValueError(
                "alignment_confidence_layer_idx must be in "
                f"[0, {self.alignment_confidence_num_layers}), got "
                f"{self.alignment_confidence_layer_idx}"
            )

        denom = max(self.alignment_confidence_num_layers - 1, 1)
        layer_position = float(self.alignment_confidence_layer_idx) / float(denom)
        key_scale = (
            self.alignment_confidence_key_layer_scale_start
            + (
                self.alignment_confidence_key_layer_scale_end
                - self.alignment_confidence_key_layer_scale_start
            )
            * layer_position
        )
        value_scale = (
            self.alignment_confidence_value_layer_scale_start
            + (
                self.alignment_confidence_value_layer_scale_end
                - self.alignment_confidence_value_layer_scale_start
            )
            * layer_position
        )
        return key_scale, value_scale

    def _current_alignment_layer_scales(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        if self.alignment_confidence_layer_scale_mode == "learned":
            return (
                self.alignment_confidence_key_token_delta_scale_param.to(
                    device=device,
                    dtype=dtype,
                ),
                self.alignment_confidence_value_token_delta_scale_param.to(
                    device=device,
                    dtype=dtype,
                ),
            )

        return (
            torch.tensor(
                self.alignment_confidence_key_token_delta_scale,
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                self.alignment_confidence_value_token_delta_scale,
                device=device,
                dtype=dtype,
            ),
        )

    def _current_alignment_residual_scales(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        self._last_alignment_residual_scale_aux_loss = None
        if self.alignment_residual_scale_mode == "none":
            key_scale = torch.tensor(1.0, device=device, dtype=dtype)
            value_scale = torch.tensor(1.0, device=device, dtype=dtype)
            self.last_alignment_key_residual_scale = 1.0
            self.last_alignment_value_residual_scale = 1.0
            self.last_alignment_residual_scale_delta_l2 = 0.0
            self.last_alignment_residual_scale_aux_loss = 0.0
            return key_scale, value_scale

        compute_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        key_delta = self.alignment_residual_key_scale_delta.to(
            device=device,
            dtype=compute_dtype,
        )
        value_delta = self.alignment_residual_value_scale_delta.to(
            device=device,
            dtype=compute_dtype,
        )
        if self.alignment_residual_scale_max_delta > 0:
            max_delta = torch.tensor(
                self.alignment_residual_scale_max_delta,
                device=device,
                dtype=compute_dtype,
            )
            key_delta = max_delta * torch.tanh(key_delta / max_delta)
            value_delta = max_delta * torch.tanh(value_delta / max_delta)

        key_scale = (
            torch.tensor(
                self.alignment_residual_key_scale_init,
                device=device,
                dtype=compute_dtype,
            )
            + key_delta
        ).clamp_min(0.0)
        value_scale = (
            torch.tensor(
                self.alignment_residual_value_scale_init,
                device=device,
                dtype=compute_dtype,
            )
            + value_delta
        ).clamp_min(0.0)
        delta_l2 = 0.5 * (key_delta.square() + value_delta.square())
        if self.alignment_residual_scale_l2_weight > 0:
            self._last_alignment_residual_scale_aux_loss = (
                self.alignment_residual_scale_l2_weight * delta_l2
            )

        try:
            self.last_alignment_key_residual_scale = float(
                key_scale.detach().float().cpu().item()
            )
            self.last_alignment_value_residual_scale = float(
                value_scale.detach().float().cpu().item()
            )
            self.last_alignment_residual_scale_delta_l2 = float(
                delta_l2.detach().float().cpu().item()
            )
            self.last_alignment_residual_scale_aux_loss = float(
                (
                    self._last_alignment_residual_scale_aux_loss
                    if self._last_alignment_residual_scale_aux_loss is not None
                    else delta_l2.new_zeros(())
                )
                .detach()
                .float()
                .cpu()
                .item()
            )
        except Exception:
            pass

        return key_scale.to(dtype=dtype), value_scale.to(dtype=dtype)

    @staticmethod
    def _normalized_source_weight_entropy(source_weights: Tensor) -> Tensor:
        positive = source_weights.clamp_min(0.0)
        totals = positive.sum(dim=-1, keepdim=True)
        probs = torch.where(
            totals > 0,
            positive / totals.clamp_min(torch.finfo(source_weights.dtype).eps),
            torch.zeros_like(positive),
        )
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(
            dim=-1
        )
        active = (positive > 0).sum(dim=-1)
        denom = active.clamp_min(2).to(dtype=source_weights.dtype).log()
        return torch.where(active > 1, entropy / denom, torch.zeros_like(entropy))

    def _learned_alignment_transfer_gate_from_weights(
        self,
        weights: Tensor,
        valid_mask: Tensor,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, Ht, N, _ = target_shape
        ones = torch.ones(B, Ht, N, 1, dtype=dtype, device=device)
        zero_rows = torch.zeros(B, N, dtype=torch.float32, device=device)
        if self.learned_alignment_transfer_gate_mode == "none":
            return (
                ones,
                zero_rows,
                zero_rows,
                zero_rows,
                torch.zeros_like(zero_rows, dtype=torch.bool),
            )

        compute_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        probs = weights.detach().to(device=device, dtype=compute_dtype).clamp_min(0.0)
        valid = valid_mask.to(device=device, dtype=torch.bool)
        if probs.shape[-1] == 0:
            return (
                ones,
                zero_rows,
                zero_rows,
                zero_rows,
                torch.zeros_like(zero_rows, dtype=torch.bool),
            )

        probs = probs.masked_fill(~valid, 0.0)
        totals = probs.sum(dim=-1, keepdim=True)
        has_valid = totals.squeeze(-1) > 0
        eps = torch.finfo(compute_dtype).eps
        probs = torch.where(
            has_valid.unsqueeze(-1),
            probs / totals.clamp_min(eps),
            torch.zeros_like(probs),
        )

        entropy = self._normalized_source_weight_entropy(probs)
        masked_probs = probs.masked_fill(~valid, -torch.finfo(compute_dtype).max)
        topk = torch.topk(masked_probs, k=min(2, probs.shape[-1]), dim=-1).values
        top1 = topk[..., 0].clamp_min(0.0)
        if topk.shape[-1] == 1:
            margin = torch.ones_like(top1)
        else:
            margin = top1 - topk[..., 1].clamp_min(0.0)

        temperature = max(self.learned_alignment_transfer_gate_temperature, eps)
        entropy_score = torch.sigmoid(
            (self.learned_alignment_transfer_gate_entropy_threshold - entropy)
            / temperature
        )
        margin_score = torch.sigmoid(
            (margin - self.learned_alignment_transfer_gate_margin_threshold)
            / temperature
        )
        quality = (entropy_score * margin_score).clamp_min(0.0).sqrt()

        active_count = valid.sum(dim=-1)
        calibrated_rows = has_valid & (
            active_count >= self.learned_alignment_transfer_gate_min_valid
        )
        quality = torch.where(calibrated_rows, quality, torch.ones_like(quality))
        floor = self.learned_alignment_transfer_gate_floor
        gate = floor + (1.0 - floor) * quality
        gate = gate.clamp(min=floor, max=1.0)
        expanded_gate = gate[:, None, :, None].expand(B, Ht, N, 1).to(dtype=dtype)
        return expanded_gate, entropy, margin, top1, calibrated_rows

    def _store_learned_alignment_transfer_gate(
        self,
        key_weights: Tensor,
        value_weights: Tensor,
        valid_mask: Tensor,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.learned_alignment_transfer_gate_mode == "none":
            self._last_learned_alignment_key_transfer_gate = None
            self._last_learned_alignment_value_transfer_gate = None
            return

        (
            key_gate,
            key_entropy,
            key_margin,
            key_top1,
            key_rows,
        ) = self._learned_alignment_transfer_gate_from_weights(
            weights=key_weights,
            valid_mask=valid_mask,
            target_shape=target_shape,
            dtype=dtype,
            device=device,
        )
        (
            value_gate,
            value_entropy,
            value_margin,
            value_top1,
            value_rows,
        ) = self._learned_alignment_transfer_gate_from_weights(
            weights=value_weights,
            valid_mask=valid_mask,
            target_shape=target_shape,
            dtype=dtype,
            device=device,
        )
        self._last_learned_alignment_key_transfer_gate = key_gate
        self._last_learned_alignment_value_transfer_gate = value_gate

        try:
            self.last_learned_alignment_key_transfer_gate = float(
                key_gate.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_transfer_gate = float(
                value_gate.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_transfer_gate_std = float(
                key_gate.detach().float().std(unbiased=False).cpu().item()
            )
            self.last_learned_alignment_value_transfer_gate_std = float(
                value_gate.detach().float().std(unbiased=False).cpu().item()
            )
            self.last_learned_alignment_key_transfer_margin = float(
                key_margin.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_transfer_margin = float(
                value_margin.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_transfer_top1 = float(
                key_top1.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_transfer_top1 = float(
                value_top1.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_transfer_entropy = float(
                key_entropy.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_transfer_entropy = float(
                value_entropy.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_transfer_gate_selected_rate = float(
                (key_rows | value_rows).detach().float().mean().cpu().item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_key_transfer_gate_tensor = (
                    key_gate.detach().float().cpu()
                )
                self.last_learned_alignment_value_transfer_gate_tensor = (
                    value_gate.detach().float().cpu()
                )
        except Exception:
            pass

    def _current_learned_alignment_transfer_gate(
        self,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        B, Ht, N, _ = target_shape
        ones = torch.ones(B, Ht, N, 1, dtype=dtype, device=device)
        if self.learned_alignment_transfer_gate_mode == "none":
            return ones, ones

        def materialize(gate: Optional[Tensor]) -> Tensor:
            if gate is None:
                return ones
            gate = gate.to(device=device, dtype=dtype)
            if gate.shape == (B, Ht, N, 1):
                return gate
            if gate.shape == (B, 1, N, 1):
                return gate.expand(B, Ht, N, 1)
            return ones

        return (
            materialize(self._last_learned_alignment_key_transfer_gate),
            materialize(self._last_learned_alignment_value_transfer_gate),
        )

    def _resolve_source_entropy(
        self,
        source_weights: Optional[Tensor],
        source_entropy: Optional[Tensor],
        source_entropy_override: Optional[Tensor],
        fallback: Tensor,
        compute_dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Use controlled entropy where requested, otherwise preserve native math."""
        native_entropy = fallback
        if source_weights is not None:
            native_entropy = self._normalized_source_weight_entropy(
                source_weights.to(device=device, dtype=compute_dtype)
            )
        if source_entropy is None:
            return native_entropy

        explicit_entropy = source_entropy.to(device=device, dtype=compute_dtype)
        if explicit_entropy.dim() == 3 and explicit_entropy.shape[-1] == 1:
            explicit_entropy = explicit_entropy.squeeze(-1)
        if explicit_entropy.shape != native_entropy.shape:
            raise ValueError(
                "source_entropy must have shape (B, N) or (B, N, 1), "
                f"got {tuple(source_entropy.shape)}"
            )
        explicit_entropy = explicit_entropy.clamp(min=0.0, max=1.0)

        if source_entropy_override is None:
            override = torch.ones_like(explicit_entropy, dtype=torch.bool)
        else:
            override = source_entropy_override.to(device=device, dtype=torch.bool)
            if override.dim() == 3 and override.shape[-1] == 1:
                override = override.squeeze(-1)
            if override.shape != native_entropy.shape:
                raise ValueError(
                    "source_entropy_override must have shape (B, N) or "
                    f"(B, N, 1), got {tuple(source_entropy_override.shape)}"
                )
        return torch.where(override, explicit_entropy, native_entropy)

    def _alignment_quality_features(
        self,
        confidence: Tensor,
        source_weights: Optional[Tensor],
        source_entropy: Optional[Tensor],
        source_entropy_override: Optional[Tensor],
        compute_dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Build per-target alignment quality features for the token gate."""
        confidence_feature = confidence.squeeze(1).squeeze(-1)
        zero = torch.zeros_like(confidence_feature)
        one = torch.ones_like(confidence_feature)

        if source_weights is None:
            entropy_weights = None
            top1 = one
            active_fraction = one
        else:
            weights = source_weights.to(device=device, dtype=compute_dtype)
            positive = weights.clamp_min(0.0)
            active_mask = positive > 0
            totals = positive.sum(dim=-1, keepdim=True)
            probs = torch.where(
                totals > 0,
                positive / totals.clamp_min(torch.finfo(compute_dtype).eps),
                torch.zeros_like(positive),
            )
            entropy_weights = probs
            top1 = probs.max(dim=-1).values
            active_fraction = active_mask.sum(dim=-1).to(dtype=compute_dtype) / float(
                max(positive.shape[-1], 1)
            )
        entropy = self._resolve_source_entropy(
            source_weights=entropy_weights,
            source_entropy=source_entropy,
            source_entropy_override=source_entropy_override,
            fallback=zero,
            compute_dtype=compute_dtype,
            device=device,
        )

        return torch.stack(
            (
                confidence_feature,
                entropy,
                top1,
                active_fraction,
            ),
            dim=-1,
        )

    def calibrate_source_weights(
        self,
        source_weights: Tensor,
        source_indices: Optional[Tensor] = None,
        source_confidence: Optional[Tensor] = None,
        source_entropy: Optional[Tensor] = None,
        source_entropy_override: Optional[Tensor] = None,
    ) -> Tensor:
        self._last_alignment_weight_calibration_aux_loss = None
        if self.alignment_weight_calibration_mode == "none" or source_weights is None:
            return source_weights

        if self.alignment_weight_calibration_mode != "span_mlp":
            return source_weights

        original_dtype = source_weights.dtype
        device = source_weights.device
        compute_dtype = (
            torch.float32
            if source_weights.dtype in (torch.float16, torch.bfloat16)
            else source_weights.dtype
        )
        weights = source_weights.to(device=device, dtype=compute_dtype)
        positive_mask = weights > 0
        if source_indices is not None:
            index_mask = source_indices.to(device=device) >= 0
            valid_mask = positive_mask & index_mask
        else:
            valid_mask = positive_mask

        positive_weights = weights.masked_fill(~valid_mask, 0.0).clamp_min(0.0)
        row_sums = positive_weights.sum(dim=-1, keepdim=True)
        has_valid = row_sums > 0
        probs = torch.where(
            has_valid,
            positive_weights
            / row_sums.clamp_min(self.alignment_weight_calibration_eps),
            torch.zeros_like(positive_weights),
        )

        if probs.shape[-1] <= 1:
            return probs.to(dtype=original_dtype)

        entropy = self._resolve_source_entropy(
            source_weights=probs,
            source_entropy=source_entropy,
            source_entropy_override=source_entropy_override,
            fallback=torch.zeros_like(probs[..., 0]),
            compute_dtype=compute_dtype,
            device=device,
        )
        active_count = valid_mask.sum(dim=-1)
        if self.alignment_weight_calibration_apply_mode == "all":
            calibration_rows = has_valid.squeeze(-1)
        else:
            calibration_rows = (active_count > 1) & (
                entropy > self.alignment_weight_calibration_entropy_threshold
            )
            if source_confidence is not None:
                confidence = source_confidence.to(device=device, dtype=compute_dtype)
                if confidence.dim() == 3 and confidence.shape[-1] == 1:
                    confidence = confidence.squeeze(-1)
                calibration_rows = calibration_rows | (
                    (active_count > 1)
                    & (
                        confidence
                        < self.alignment_weight_calibration_confidence_threshold
                    )
                )
        max_prob = probs.max(dim=-1, keepdim=True).values
        k = probs.shape[-1]
        if k == 1:
            rank_feature = torch.ones_like(probs)
        else:
            rank_feature = torch.linspace(
                1.0,
                0.0,
                steps=k,
                device=device,
                dtype=compute_dtype,
            ).view(*([1] * (probs.dim() - 1)), k)
            rank_feature = rank_feature.expand_as(probs)

        safe_probs = probs.clamp_min(self.alignment_weight_calibration_eps)
        entropy_feature = entropy.unsqueeze(-1).expand_as(probs)
        max_prob_feature = max_prob.expand_as(probs)
        features = torch.stack(
            [
                probs,
                safe_probs.log(),
                rank_feature,
                entropy_feature,
                max_prob_feature,
                rank_feature * entropy_feature,
            ],
            dim=-1,
        )
        head_weight = self.alignment_weight_calibration_head.weight.to(
            device=device,
            dtype=compute_dtype,
        )
        head_bias = self.alignment_weight_calibration_head.bias.to(
            device=device,
            dtype=compute_dtype,
        )
        delta = F.linear(features, head_weight, head_bias).squeeze(-1)
        if self.alignment_weight_calibration_max_delta > 0:
            max_delta = torch.tensor(
                self.alignment_weight_calibration_max_delta,
                device=device,
                dtype=compute_dtype,
            )
            delta = max_delta * torch.tanh(delta / max_delta)

        logits = safe_probs.log() + delta
        logits = logits.masked_fill(~valid_mask, -torch.finfo(compute_dtype).max)
        calibrated = torch.softmax(logits, dim=-1)
        calibrated = torch.where(has_valid, calibrated, torch.zeros_like(calibrated))
        calibrated = torch.where(
            calibration_rows.unsqueeze(-1),
            calibrated,
            probs,
        )

        selected_valid_mask = valid_mask & calibration_rows.unsqueeze(-1)
        selected_valid = selected_valid_mask.to(dtype=compute_dtype)
        selected_row = calibration_rows.to(dtype=compute_dtype)
        selected_token_count = selected_row.sum().clamp_min(1.0)
        delta_l2 = (
            delta.square() * selected_valid
        ).sum() / selected_valid.sum().clamp_min(1.0)
        calibrated_entropy = self._normalized_source_weight_entropy(calibrated)
        entropy_l2 = (
            (calibrated_entropy - entropy).square() * selected_row
        ).sum() / selected_token_count
        if (
            self.alignment_weight_calibration_delta_l2_weight > 0
            or self.alignment_weight_calibration_entropy_l2_weight > 0
        ):
            self._last_alignment_weight_calibration_aux_loss = (
                self.alignment_weight_calibration_delta_l2_weight * delta_l2
                + self.alignment_weight_calibration_entropy_l2_weight * entropy_l2
            )

        try:
            self.last_alignment_weight_entropy = float(
                entropy.detach().float().mean().cpu().item()
            )
            self.last_alignment_weight_calibrated_entropy = float(
                calibrated_entropy.detach().float().mean().cpu().item()
            )
            self.last_alignment_weight_entropy_l2 = float(
                entropy_l2.detach().float().cpu().item()
            )
            self.last_alignment_weight_delta_l2 = float(
                delta_l2.detach().float().cpu().item()
            )
            self.last_alignment_weight_aux_loss = float(
                (
                    self._last_alignment_weight_calibration_aux_loss
                    if self._last_alignment_weight_calibration_aux_loss is not None
                    else delta_l2.new_zeros(())
                )
                .detach()
                .float()
                .cpu()
                .item()
            )
            self.last_alignment_weight_calibration_selected_rate = float(
                calibration_rows.detach().float().mean().cpu().item()
            )
            self.last_alignment_weight_delta_abs_mean = float(
                delta.detach().abs().float().mean().cpu().item()
            )
            self.last_alignment_weight_delta_abs_max = float(
                delta.detach().abs().float().max().cpu().item()
            )
            self.last_alignment_weight_top1_mean = float(
                calibrated[..., 0].detach().float().mean().cpu().item()
            )
            if self.capture_alignment_diagnostics:
                self.last_alignment_weight_calibrated_tensor = (
                    calibrated.detach().float().cpu()
                )
                self.last_alignment_weight_delta_tensor = delta.detach().float().cpu()
        except Exception:
            pass

        return calibrated.to(dtype=original_dtype)

    def _limit_alignment_confidence_delta(self, delta: Tensor) -> Tensor:
        if self.alignment_confidence_max_delta <= 0:
            return delta
        return self.alignment_confidence_max_delta * torch.tanh(
            delta / self.alignment_confidence_max_delta
        )

    def _limit_learned_alignment_injection_delta(self, delta: Tensor) -> Tensor:
        if self.learned_alignment_injection_max_delta <= 0:
            return delta
        return self.learned_alignment_injection_max_delta * torch.tanh(
            delta / self.learned_alignment_injection_max_delta
        )

    def _compute_learned_alignment_injection_gate(
        self,
        key_hidden: Tensor,
        value_hidden: Tensor,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        B, Ht, N, _ = target_shape
        if self.learned_alignment_injection_gate_mode == "none":
            ones = torch.ones(B, Ht, N, 1, dtype=dtype, device=device)
            return ones, ones

        compute_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        key_delta = self.learned_key_injection_gate_head(key_hidden).to(
            dtype=compute_dtype
        )
        value_delta = self.learned_value_injection_gate_head(value_hidden).to(
            dtype=compute_dtype
        )
        key_delta = key_delta.permute(0, 2, 1).unsqueeze(-1)
        value_delta = value_delta.permute(0, 2, 1).unsqueeze(-1)
        key_delta = self._limit_learned_alignment_injection_delta(key_delta)
        value_delta = self._limit_learned_alignment_injection_delta(value_delta)

        key_logit = (
            self.learned_key_injection_gate_bias.to(dtype=compute_dtype).view(
                1, 1, 1, 1
            )
            + key_delta
        )
        value_logit = (
            self.learned_value_injection_gate_bias.to(dtype=compute_dtype).view(
                1, 1, 1, 1
            )
            + value_delta
        )
        key_gate = torch.sigmoid(key_logit).to(dtype=dtype).expand(B, Ht, N, 1)
        value_gate = torch.sigmoid(value_logit).to(dtype=dtype).expand(B, Ht, N, 1)

        try:
            self.last_learned_alignment_key_injection = float(
                key_gate.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_injection = float(
                value_gate.detach().float().mean().cpu().item()
            )
            self.last_learned_alignment_key_injection_std = float(
                key_gate.detach().float().std(unbiased=False).cpu().item()
            )
            self.last_learned_alignment_value_injection_std = float(
                value_gate.detach().float().std(unbiased=False).cpu().item()
            )
            self.last_learned_alignment_key_injection_delta_abs_mean = float(
                key_delta.detach().abs().float().mean().cpu().item()
            )
            self.last_learned_alignment_value_injection_delta_abs_mean = float(
                value_delta.detach().abs().float().mean().cpu().item()
            )
            if self.capture_alignment_diagnostics:
                self.last_learned_alignment_key_injection_tensor = (
                    key_gate.detach().float().cpu()
                )
                self.last_learned_alignment_value_injection_tensor = (
                    value_gate.detach().float().cpu()
                )
        except Exception:
            pass

        return key_gate, value_gate

    def _alignment_delta_l2_regularization(
        self,
        key_delta: Tensor,
        value_delta: Tensor,
        confidence: Tensor,
        entropy: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        key_delta_l2 = key_delta.square().mean()
        value_delta_l2 = value_delta.square().mean()
        raw_delta_l2 = 0.5 * (key_delta_l2 + value_delta_l2)
        if self.alignment_confidence_delta_l2_mode == "all":
            return raw_delta_l2, None

        selected = confidence < self.alignment_confidence_delta_l2_confidence_threshold
        if entropy is not None:
            selected = selected | (
                entropy > self.alignment_confidence_delta_l2_entropy_threshold
            )
        selected_mask = selected.to(dtype=key_delta.dtype)

        def masked_square_mean(delta: Tensor) -> Tensor:
            masked_square = delta.square() * selected_mask
            broadcast_mask = torch.broadcast_to(selected_mask, masked_square.shape)
            denom = broadcast_mask.sum()
            return masked_square.sum() / denom.clamp_min(1.0)

        selected_delta_l2 = 0.5 * (
            masked_square_mean(key_delta) + masked_square_mean(value_delta)
        )
        return selected_delta_l2, selected_mask

    def _eval_alignment_confidence_override(
        self,
        source_confidence: Optional[Tensor],
        source_weights: Optional[Tensor],
        source_entropy: Optional[Tensor],
        source_entropy_override: Optional[Tensor],
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Return a static/forced-on eval gate without changing checkpoint state."""
        eval_mode = getattr(self, "alignment_confidence_eval_mode", "learned")
        if eval_mode == "learned":
            return None

        B, Ht, N, _ = target_shape
        compute_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        if eval_mode == "forced_on" or source_confidence is None:
            token_confidence = torch.ones(
                B, 1, N, 1, dtype=compute_dtype, device=device
            )
        else:
            token_confidence = source_confidence.to(
                device=device, dtype=compute_dtype
            )
            if token_confidence.dim() == 2:
                token_confidence = token_confidence[:, None, :, None]
            elif token_confidence.dim() == 3:
                token_confidence = token_confidence[:, None, :, :]
            else:
                raise ValueError(
                    "source_confidence must have shape (B, N) or (B, N, 1), "
                    f"got {tuple(token_confidence.shape)}"
                )
            token_confidence = token_confidence.clamp(min=0.0, max=1.0)

        key_confidence = token_confidence.expand(B, Ht, N, 1).to(dtype=dtype)
        value_confidence = key_confidence
        zero_delta = torch.zeros_like(key_confidence, dtype=torch.float32)
        if getattr(self, "suppress_host_diagnostics", False):
            self._last_alignment_confidence_aux_loss = None
            return key_confidence, value_confidence
        source_for_record = token_confidence.detach().float().cpu()
        entropy_for_record = torch.zeros_like(source_for_record)
        if source_weights is not None or source_entropy is not None:
            entropy_for_record = self._resolve_source_entropy(
                source_weights=source_weights,
                source_entropy=source_entropy,
                source_entropy_override=source_entropy_override,
                fallback=torch.zeros(B, N, device=device, dtype=compute_dtype),
                compute_dtype=compute_dtype,
                device=device,
            )[:, None, :, None].detach().float().cpu()

        self._last_alignment_confidence_aux_loss = None
        self.last_alignment_delta_l2 = 0.0
        self.last_alignment_regularized_delta_l2 = 0.0
        self.last_alignment_regularization_selected_rate = 0.0
        self.last_alignment_aux_loss = 0.0
        self.last_alignment_key_delta_abs_mean = 0.0
        self.last_alignment_value_delta_abs_mean = 0.0
        self.last_alignment_key_delta_abs_max = 0.0
        self.last_alignment_value_delta_abs_max = 0.0
        self.last_alignment_key_confidence_std = float(
            key_confidence.detach().float().std(unbiased=False).cpu().item()
        )
        self.last_alignment_value_confidence_std = (
            self.last_alignment_key_confidence_std
        )
        if self.capture_alignment_diagnostics:
            self.last_alignment_key_delta_tensor = zero_delta.detach().cpu()
            self.last_alignment_value_delta_tensor = zero_delta.detach().cpu()
            self.last_alignment_key_confidence_tensor = (
                key_confidence.detach().float().cpu()
            )
            self.last_alignment_value_confidence_tensor = (
                value_confidence.detach().float().cpu()
            )
            self.last_alignment_source_confidence_tensor = source_for_record
            self.last_alignment_entropy_tensor = entropy_for_record
            self.last_alignment_regularization_mask_tensor = torch.zeros_like(
                source_for_record
            )
            if not hasattr(self, "alignment_diagnostic_records"):
                self.alignment_diagnostic_records = []
            self.alignment_diagnostic_records.append(
                {
                    "key_delta": self.last_alignment_key_delta_tensor,
                    "value_delta": self.last_alignment_value_delta_tensor,
                    "key_confidence": self.last_alignment_key_confidence_tensor,
                    "value_confidence": self.last_alignment_value_confidence_tensor,
                    "source_confidence": self.last_alignment_source_confidence_tensor,
                    "entropy": self.last_alignment_entropy_tensor,
                    "regularization_mask": (
                        self.last_alignment_regularization_mask_tensor
                    ),
                    "quality_features": torch.empty(0),
                }
            )
        return key_confidence, value_confidence

    def _compute_alignment_confidence(
        self,
        source_confidence: Optional[Tensor],
        source_weights: Optional[Tensor],
        key_hidden: Tensor,
        value_hidden: Tensor,
        target_shape: Tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
        source_entropy: Optional[Tensor] = None,
        source_entropy_override: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, Ht, N, _ = target_shape
        self._last_alignment_confidence_aux_loss = None
        eval_override = self._eval_alignment_confidence_override(
            source_confidence=source_confidence,
            source_weights=source_weights,
            source_entropy=source_entropy,
            source_entropy_override=source_entropy_override,
            target_shape=target_shape,
            dtype=dtype,
            device=device,
        )
        if eval_override is not None:
            return eval_override
        if self.alignment_confidence_gate_mode == "none" or source_confidence is None:
            ones = torch.ones(B, Ht, N, 1, dtype=dtype, device=device)
            return ones, ones

        compute_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        confidence = source_confidence.to(device=device, dtype=compute_dtype)
        if confidence.dim() == 2:
            confidence = confidence[:, None, :, None]
        elif confidence.dim() == 3:
            confidence = confidence[:, None, :, :]
        else:
            raise ValueError(
                "source_confidence must have shape (B, N) or (B, N, 1), "
                f"got {tuple(confidence.shape)}"
            )

        confidence = confidence.clamp(
            min=self.alignment_confidence_eps,
            max=1.0 - self.alignment_confidence_eps,
        )
        base_logit = torch.logit(confidence)
        key_delta = self.key_alignment_confidence_bias.to(dtype=compute_dtype).view(
            1, 1, 1, 1
        )
        value_delta = self.value_alignment_confidence_bias.to(dtype=compute_dtype).view(
            1, 1, 1, 1
        )
        entropy: Optional[Tensor] = None
        quality_features: Optional[Tensor] = None

        if source_weights is not None or source_entropy is not None:
            entropy = self._resolve_source_entropy(
                source_weights=source_weights,
                source_entropy=source_entropy,
                source_entropy_override=source_entropy_override,
                fallback=torch.zeros(B, N, device=device, dtype=compute_dtype),
                compute_dtype=compute_dtype,
                device=device,
            )[:, None, :, None]
            key_delta = (
                key_delta
                + self.key_alignment_entropy_scale.to(dtype=compute_dtype).view(
                    1, 1, 1, 1
                )
                * entropy
            )
            value_delta = (
                value_delta
                + self.value_alignment_entropy_scale.to(dtype=compute_dtype).view(
                    1, 1, 1, 1
                )
                * entropy
            )

        if self.alignment_confidence_gate_mode == "token_mlp":
            key_gate_hidden = key_hidden
            value_gate_hidden = value_hidden
            if self.alignment_confidence_feature_mode == "quality":
                quality_features = self._alignment_quality_features(
                    confidence=confidence,
                    source_weights=source_weights,
                    source_entropy=source_entropy,
                    source_entropy_override=source_entropy_override,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                key_gate_hidden = torch.cat(
                    [key_hidden, quality_features.to(dtype=key_hidden.dtype)],
                    dim=-1,
                )
                value_gate_hidden = torch.cat(
                    [value_hidden, quality_features.to(dtype=value_hidden.dtype)],
                    dim=-1,
                )

            key_token_delta = self.key_alignment_confidence_head(key_gate_hidden).to(
                dtype=compute_dtype
            )
            value_token_delta = self.value_alignment_confidence_head(
                value_gate_hidden
            ).to(dtype=compute_dtype)
            key_token_delta = key_token_delta.permute(0, 2, 1).unsqueeze(-1)
            value_token_delta = value_token_delta.permute(0, 2, 1).unsqueeze(-1)
            key_layer_scale, value_layer_scale = self._current_alignment_layer_scales(
                dtype=compute_dtype,
                device=device,
            )
            key_token_delta = key_token_delta * key_layer_scale.view(1, 1, 1, 1)
            value_token_delta = value_token_delta * value_layer_scale.view(1, 1, 1, 1)
            key_delta = key_delta + key_token_delta
            value_delta = value_delta + value_token_delta
        else:
            key_layer_scale, value_layer_scale = self._current_alignment_layer_scales(
                dtype=compute_dtype,
                device=device,
            )

        key_delta = self._limit_alignment_confidence_delta(key_delta)
        value_delta = self._limit_alignment_confidence_delta(value_delta)
        raw_delta_l2 = 0.5 * (key_delta.square().mean() + value_delta.square().mean())
        regularized_delta_l2, regularization_mask = (
            self._alignment_delta_l2_regularization(
                key_delta=key_delta,
                value_delta=value_delta,
                confidence=confidence,
                entropy=entropy,
            )
        )
        if self.alignment_confidence_delta_l2_weight > 0:
            self._last_alignment_confidence_aux_loss = (
                self.alignment_confidence_delta_l2_weight * regularized_delta_l2
            )
        key_confidence = torch.sigmoid(base_logit + key_delta).to(dtype=dtype)
        value_confidence = torch.sigmoid(base_logit + value_delta).to(dtype=dtype)
        key_confidence = key_confidence.expand(B, Ht, N, 1)
        value_confidence = value_confidence.expand(B, Ht, N, 1)

        if getattr(self, "suppress_host_diagnostics", False):
            return key_confidence, value_confidence

        try:
            self.last_alignment_delta_l2 = float(raw_delta_l2.detach().cpu().item())
            self.last_alignment_regularized_delta_l2 = float(
                regularized_delta_l2.detach().cpu().item()
            )
            if regularization_mask is None:
                selected_rate = 1.0
            else:
                selected_rate = float(
                    regularization_mask.detach().float().mean().cpu().item()
                )
            self.last_alignment_regularization_selected_rate = selected_rate
            self.last_alignment_aux_loss = float(
                (
                    self._last_alignment_confidence_aux_loss
                    if self._last_alignment_confidence_aux_loss is not None
                    else raw_delta_l2.new_zeros(())
                )
                .detach()
                .cpu()
                .item()
            )
            self.last_alignment_key_delta_abs_mean = float(
                key_delta.detach().abs().mean().cpu().item()
            )
            self.last_alignment_value_delta_abs_mean = float(
                value_delta.detach().abs().mean().cpu().item()
            )
            self.last_alignment_key_delta_abs_max = float(
                key_delta.detach().abs().max().cpu().item()
            )
            self.last_alignment_value_delta_abs_max = float(
                value_delta.detach().abs().max().cpu().item()
            )
            self.last_alignment_key_layer_scale = float(
                key_layer_scale.detach().cpu().item()
            )
            self.last_alignment_value_layer_scale = float(
                value_layer_scale.detach().cpu().item()
            )
            self.last_alignment_key_confidence_std = float(
                key_confidence.detach().float().std(unbiased=False).cpu().item()
            )
            self.last_alignment_value_confidence_std = float(
                value_confidence.detach().float().std(unbiased=False).cpu().item()
            )
            if quality_features is not None:
                detached_quality_features = quality_features.detach().float()
                self.last_alignment_quality_confidence_mean = float(
                    detached_quality_features[..., 0].mean().cpu().item()
                )
                self.last_alignment_quality_entropy_mean = float(
                    detached_quality_features[..., 1].mean().cpu().item()
                )
                self.last_alignment_quality_top1_mean = float(
                    detached_quality_features[..., 2].mean().cpu().item()
                )
                self.last_alignment_quality_active_fraction_mean = float(
                    detached_quality_features[..., 3].mean().cpu().item()
                )
            if self.capture_alignment_diagnostics:
                self.last_alignment_key_delta_tensor = key_delta.detach().float().cpu()
                self.last_alignment_value_delta_tensor = (
                    value_delta.detach().float().cpu()
                )
                self.last_alignment_key_confidence_tensor = (
                    key_confidence.detach().float().cpu()
                )
                self.last_alignment_value_confidence_tensor = (
                    value_confidence.detach().float().cpu()
                )
                self.last_alignment_source_confidence_tensor = (
                    confidence.detach().float().cpu()
                )
                self.last_alignment_entropy_tensor = (
                    entropy.detach().float().cpu()
                    if entropy is not None
                    else torch.zeros_like(confidence.detach().float().cpu())
                )
                self.last_alignment_regularization_mask_tensor = (
                    regularization_mask.detach().float().cpu()
                    if regularization_mask is not None
                    else torch.ones_like(confidence.detach().float().cpu())
                )
                quality_tensor = (
                    quality_features.detach().float().cpu()
                    if quality_features is not None
                    else torch.empty(0)
                )
                record = {
                    "key_delta": self.last_alignment_key_delta_tensor,
                    "value_delta": self.last_alignment_value_delta_tensor,
                    "key_confidence": self.last_alignment_key_confidence_tensor,
                    "value_confidence": self.last_alignment_value_confidence_tensor,
                    "source_confidence": self.last_alignment_source_confidence_tensor,
                    "entropy": self.last_alignment_entropy_tensor,
                    "regularization_mask": self.last_alignment_regularization_mask_tensor,
                    "quality_features": quality_tensor,
                }
                if not hasattr(self, "alignment_diagnostic_records"):
                    self.alignment_diagnostic_records = []
                self.alignment_diagnostic_records.append(record)
        except Exception:
            pass
        return key_confidence, value_confidence

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
        source_confidence: Optional[Tensor] = None,
        source_weights: Optional[Tensor] = None,
        source_entropy: Optional[Tensor] = None,
        source_entropy_override: Optional[Tensor] = None,
        fpct_parent_nuisance: Optional[Dict[str, Tensor]] = None,
        fpct_capture_parent_nuisance: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        source_key, source_value = source_kv
        target_key, target_value = target_kv
        self._last_alignment_confidence_aux_loss = None
        self._last_alignment_residual_scale_aux_loss = None

        B, Hs, N, Ds = source_key.shape
        _, Ht, _, Dt = target_key.shape

        # Flatten heads
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        source_value_flat = (
            source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds)
        )
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        target_value_flat = (
            target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt)
        )

        # 1) concat source and target features along channel
        key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)
        value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)

        # 2) project to hidden dim
        key_hidden = self.key_in(key_cat)
        value_hidden = self.value_in(value_cat)

        # 3) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        key_hidden = self.key_mlp1(key_hidden)
        value_hidden = self.value_mlp1(value_hidden)

        # 4b) intermediate representation -> projected feature path
        key_proj_hidden = self.key_proj_out(
            self.key_proj_mlp2(key_hidden)
        )  # (B, N, Ht * Dt)
        value_proj_hidden = self.value_proj_out(
            self.value_proj_mlp2(value_hidden)
        )  # (B, N, Ht * Dt)
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(
            1, 2
        )  # (B, Ht, N, Dt)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(
            1, 2
        )  # (B, Ht, N, Dt)

        # 4a) intermediate representation -> scalar path
        key_scalar = self.key_scalar_head(
            self.key_scalar_mlp2(key_hidden)
        )  # (B, N, Ht)
        value_scalar = self.value_scalar_head(
            self.value_scalar_mlp2(value_hidden)
        )  # (B, N, Ht)
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)

        # Key/value gates: element-wise Gumbel noise with scalar logits (broadcast over channels)
        key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)
        value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)
        if fpct_parent_nuisance is not None:
            key_gate = fpct_parent_nuisance["legacy_key_gate"].to(
                device=target_key.device, dtype=target_key.dtype
            )
            value_gate = fpct_parent_nuisance["legacy_value_gate"].to(
                device=target_value.device, dtype=target_value.dtype
            )
            expected_gate_shape = (B, Ht, N, 1)
            try:
                key_gate = torch.broadcast_to(key_gate, expected_gate_shape)
                value_gate = torch.broadcast_to(value_gate, expected_gate_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "FPCT parent legacy gates must have shape "
                    f"broadcastable to {expected_gate_shape}"
                ) from exc
        elif getattr(self, "legacy_scalar_gate_eval_mode", "checkpoint_native") == "forced_on":
            key_gate = torch.ones(
                B, Ht, N, 1, device=target_key.device, dtype=target_key.dtype
            )
            value_gate = torch.ones_like(key_gate)
        elif self._learned_alignment_replay_scoring_mode:
            key_gate = torch.sigmoid(key_gate_logit / self.gate_temperature)
            value_gate = torch.sigmoid(value_gate_logit / self.gate_temperature)
        elif self.training and self.use_gumbel:
            u1 = torch.rand(
                B, Ht, N, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype
            )
            u2 = torch.rand(
                B,
                Ht,
                N,
                1,
                device=value_gate_logit.device,
                dtype=value_gate_logit.dtype,
            )
            g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)
            value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)
        else:
            key_gate = (key_gate_logit > 0).float()
            value_gate = (value_gate_logit > 0).float()

        # Normalize scalars (scalar_temperature=1.0)
        norm_key_scalar = torch.sigmoid(key_scalar)
        norm_value_scalar = torch.sigmoid(value_scalar)

        if fpct_parent_nuisance is not None:
            key_alignment_confidence = fpct_parent_nuisance[
                "key_alignment_confidence"
            ].to(device=target_key.device, dtype=target_key.dtype)
            value_alignment_confidence = fpct_parent_nuisance[
                "value_alignment_confidence"
            ].to(device=target_value.device, dtype=target_value.dtype)
            expected_confidence_shape = (B, Ht, N, 1)
            if (
                key_alignment_confidence.shape != expected_confidence_shape
                or value_alignment_confidence.shape != expected_confidence_shape
            ):
                raise ValueError(
                    "FPCT parent confidence must have shape "
                    f"{expected_confidence_shape}"
                )
        else:
            key_alignment_confidence, value_alignment_confidence = (
                self._compute_alignment_confidence(
                    source_confidence=source_confidence,
                    source_weights=source_weights,
                    source_entropy=source_entropy,
                    source_entropy_override=source_entropy_override,
                    key_hidden=key_hidden,
                    value_hidden=value_hidden,
                    target_shape=(B, Ht, N, Dt),
                    dtype=target_key.dtype,
                    device=target_key.device,
                )
            )
        key_residual_scale, value_residual_scale = (
            self._current_alignment_residual_scales(
                dtype=target_key.dtype,
                device=target_key.device,
            )
        )
        key_learned_injection_gate, value_learned_injection_gate = (
            self._compute_learned_alignment_injection_gate(
                key_hidden=key_hidden,
                value_hidden=value_hidden,
                target_shape=(B, Ht, N, Dt),
                dtype=target_key.dtype,
                device=target_key.device,
            )
        )
        key_transfer_gate, value_transfer_gate = (
            self._current_learned_alignment_transfer_gate(
                target_shape=(B, Ht, N, Dt),
                dtype=target_key.dtype,
                device=target_key.device,
            )
        )

        if fpct_capture_parent_nuisance:
            self._fpct_last_parent_nuisance = {
                "legacy_key_gate": key_gate,
                "legacy_value_gate": value_gate,
                "key_alignment_confidence": key_alignment_confidence,
                "value_alignment_confidence": value_alignment_confidence,
            }

        # Combine (preserve_target_weight=False, add_self=True)
        output_key = (
            target_key
            + key_residual_scale.view(1, 1, 1, 1)
            * key_alignment_confidence
            * key_learned_injection_gate
            * key_transfer_gate
            * key_gate
            * norm_key_scalar
            * projected_key
        )
        output_value = (
            target_value
            + value_residual_scale.view(1, 1, 1, 1)
            * value_alignment_confidence
            * value_learned_injection_gate
            * value_transfer_gate
            * value_gate
            * norm_value_scalar
            * projected_value
        )

        if getattr(self, "suppress_host_diagnostics", False):
            return output_key, output_value

        # Expose capture attributes for downstream analysis scripts
        try:
            # Store normalized scalars (detach to avoid autograd, keep device-agnostic via CPU)
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            # Store gate logits as python floats (parameters are scalar)
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(
                self.value_gate_logit.detach().cpu().item()
            )
            self.last_legacy_key_gate = key_gate.detach().mean().cpu()
            self.last_legacy_value_gate = value_gate.detach().mean().cpu()
            self.last_legacy_scalar_gate_eval_mode = getattr(
                self, "legacy_scalar_gate_eval_mode", "checkpoint_native"
            )
            self.last_alignment_key_confidence = (
                key_alignment_confidence.detach().mean().cpu()
            )
            self.last_alignment_value_confidence = (
                value_alignment_confidence.detach().mean().cpu()
            )
        except Exception:
            # Best-effort capture; never break forward path
            pass

        return output_key, output_value


def save_projector(obj: Projector, file_path: str) -> None:
    save_object(obj, file_path)


def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    return load_object(file_path, get_projector_class, override_args)


def create_projector(projector_type: str, **kwargs) -> Projector:
    """
    Factory function to create a projector based on type.

    Args:
        projector_type: String indicating the type of projector
        **kwargs: Additional arguments to pass to the projector constructor

    Returns:
        An instance of the appropriate projector
    """
    # Prefer using the unified registry getter (handles case-insensitive keys)
    try:
        cls = get_projector_class(projector_type)
    except ValueError as e:
        raise e
    return cls(**kwargs)
