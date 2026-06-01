# Route-1 V2.2 Small Loop Summary

## Goal

Explore stronger route-1 variants without modifying or overwriting the preserved
v2 and v2.1 result sets. V2.2 keeps the best v2.1 static entropy confidence
setting and adds a learnable confidence gate inside `C2CProjector`.

## Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `soft_span_overlap_v2`
- Score mode: `uniform`
- Static confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC Challenge, OpenBookQA
- Baseline: v2.1 `adaptive_entropy050`

## Variants

- `v2.2_token_mlp_entropy050`: token/head confidence deltas are predicted from
  projector hidden states.
- `v2.2_learned_affine_entropy050`: global key/value confidence bias and entropy
  scale are learned.

## Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.1 entropy050 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 | +0.00 |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +1.05 |
| `v2.2_learned_affine_entropy050` | 45.03 | 51.65 | 42.80 | 46.49 | -3.28 |

## Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 0.3779 | 0.1337 | 0.1336 |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.2_learned_affine_entropy050` | 0.3822 | 0.1254 | 0.1243 |

## Artifacts

- Score CSV: `local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v22/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v22/`
- Checkpoints: `local/checkpoints/route1_alignment_v22/`
- Token MLP eval summaries: `local/final_results/route1_alignment_v22/qwen3_tinyllama_token_mlp_entropy050_small2048/`
- Learned affine eval summaries: `local/final_results/route1_alignment_v22/qwen3_tinyllama_learned_affine_entropy050_small2048/`
- Preserved pre-v2.2 snapshot: `local/snapshots/route1_v21_20260428_before_v22`

## Notes

- `v2.2_token_mlp_entropy050` is the current best route-1 small-loop result. It
  improves all three downstream tasks over v2.1 `adaptive_entropy050`.
- `v2.2_learned_affine_entropy050` has the best validation loss but much worse
  downstream accuracy, especially on OpenBookQA. For this codebase and small
  loop, validation loss is not a reliable proxy for route-1 alignment quality.
- The useful v2.2 signal is local adaptive confidence, not a global scalar
  calibration. The next route-1 extension should build on `token_mlp`, with
  regularization and confidence diagnostics rather than switching to
  `learned_affine`.
