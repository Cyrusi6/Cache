# Route-1 V2.2 Train Recipes

These recipes extend the best v2.1 setting, `adaptive_entropy050`, with
learnable confidence calibration inside `C2CProjector`.

Common setup:

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `soft_span_overlap_v2`
- Score mode: `uniform`
- Static confidence seed: entropy, alpha `0.5`, floor `0.5`, fallback `0.25`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`

Variants:

- `learned_affine_entropy050`: learns global key/value bias and entropy scale.
- `token_mlp_entropy050`: learns token/head confidence deltas from projector hidden states.
