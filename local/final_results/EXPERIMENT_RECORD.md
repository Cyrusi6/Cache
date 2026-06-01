# C2C Experiment Record

This file is the top-level structured record for local C2C experiments.
Append new runs here after each completed training/evaluation cycle, and keep
task-level raw artifacts under their original result directories.

## Record Index

| ID | Date | Purpose | Status | Primary Record |
| --- | --- | --- | --- | --- |
| `E0-native-baseline` | 2026-04-25 to 2026-04-26 | Reproduce paper-style native and Rosetta baseline metrics | done | `local/final_results/paper_native_baseline/README.md` |
| `E1-route1-tinyllama-hard` | 2026-04-26 | First true cross-tokenizer route-1 loop with `longest` vs hard `span_overlap` | done | `local/final_results/route1_alignment/README.md` |
| `E2-route1-llama32-soft` | 2026-04-27 | Local Llama-3.2 pair with `longest` / `span_overlap` / `soft_span_overlap` | done | `local/final_results/route1_alignment/EXPERIMENT_LOG.md` |
| `E3-route1-stress-tinyllama-soft` | 2026-04-27 | Strong mismatch stress test and consolidated comparison | done | `local/final_results/route1_alignment/stress_test_summary/route1_mismatch_stress_summary.md` |
| `E4-route1-v2-soft-scoring` | 2026-04-27 | Route-1 v2 soft scoring ablation on strong tokenizer mismatch | done | `local/final_results/route1_alignment_v2/small_loop_summary/route1_v2_small_loop_summary.md` |
| `E5-route1-v21-adaptive-confidence` | 2026-04-28 | Route-1 v2.1 entropy confidence gate on strong tokenizer mismatch | done | `local/final_results/route1_alignment_v21/small_loop_summary/route1_v21_small_loop_summary.md` |
| `E6-route1-v22-learnable-confidence` | 2026-04-28 | Route-1 v2.2 learnable confidence gates on strong tokenizer mismatch | done | `local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_summary.md` |
| `E7-route1-v23-delta-l2` | 2026-04-28 | Route-1 v2.3 delta L2 regularization for token MLP confidence | done | `local/final_results/route1_alignment_v23/small_loop_summary/route1_v23_small_loop_summary.md` |
| `E8-route1-v24-selective-delta-l2` | 2026-04-28 | Route-1 v2.4 selective delta L2 regularization on uncertain rows | done | `local/final_results/route1_alignment_v24/small_loop_summary/route1_v24_small_loop_summary.md` |
| `E9-route1-v25-token-mlp-diagnostics` | 2026-04-28 | Diagnose v2.2 token MLP confidence gate by layer, head, and alignment bucket | done | `local/final_results/route1_alignment_v25/diagnostics/v22_token_mlp_gate_mmlu_aux64/diagnostic_summary.md` |
| `E10-route1-v25-layer-gate` | 2026-04-28 | Route-1 v2.5 static layer-aware key/value token confidence gate | done | `local/final_results/route1_alignment_v25/small_loop_summary/route1_v25_small_loop_summary.md` |
| `E11-route1-v26-learned-layer-scale` | 2026-04-29 | Route-1 v2.6 learned per-layer scalar confidence scale | done | `local/final_results/route1_alignment_v26b/small_loop_summary/route1_v26b_small_loop_summary.md` |
| `E12-route1-v27-adaptive-overlap` | 2026-04-29 | Route-1 v2.7 fixed adaptive overlap top-k reweighting | done | `local/final_results/route1_alignment_v27/small_loop_summary/route1_v27_small_loop_summary.md` |
| `E13-route1-v28-span-mlp-calibrator` | 2026-04-29 | Route-1 v2.8 learnable top-k span weight calibration before KV gather | done | `local/final_results/route1_alignment_v28/small_loop_summary/route1_v28_small_loop_summary.md` |
| `E14-route1-v28-flip-diagnostics` | 2026-04-29 | Diagnose v2.2 vs v2.8 answer flips, output format, and answer-prior shift | done | `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/diagnostic_summary.md` |
| `E15-route1-v28b-constrained-calibrator` | 2026-04-29 | Route-1 v2.8b ambiguous-only span calibration with no-op regularization | done | `local/final_results/route1_alignment_v28b/small_loop_summary/route1_v28b_small_loop_summary.md` |
| `E16-route1-v29-residual-scale` | 2026-04-30 | Route-1 v2.9 learned per-layer key/value residual injection scale | done | `local/final_results/route1_alignment_v29/small_loop_summary/route1_v29_small_loop_summary.md` |
| `E17-route1-v210-answer-prior` | 2026-04-30 | Route-1 v2.10 smoothed answer-prior regularization on supervised ABCD positions | done | `local/final_results/route1_alignment_v210/small_loop_summary/route1_v210_small_loop_summary.md` |
| `E18-route1-v211-answer-margin-routing` | 2026-04-30 | Route-1 v2.11 benchmark-aligned answer-margin routing on supervised ABCD positions | done | `local/final_results/route1_alignment_v211/small_loop_summary/route1_v211_small_loop_summary.md` |
| `E19-route1-v211b-answer-margin-cehinge` | 2026-05-01 | Route-1 v2.11b softer CE+hinge answer-margin routing on supervised ABCD positions | done | `local/final_results/route1_alignment_v211b/small_loop_summary/route1_v211b_small_loop_summary.md` |
| `E20-route3-learned-alignment` | 2026-05-03 | Route-3 learned cross-tokenizer KV candidate routing from paper-native C2C | done | `local/final_results/route3_learned_alignment/README.md` |

## E0 Native / Paper-Style Baseline

### Goal

Establish paper-aligned reference metrics before modifying route-1 alignment.

### Models and Data

- Base/native receivers: `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B-Instruct`
- Rosetta/C2C baseline: original fuser path from existing code
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Extra early GSM8K check stored separately under `baseline_align_longest_gsm8k`

### Main Evaluation Results

| Model / Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B` native | 35.07 | 40.17 | 39.60 | 38.28 |
| `Qwen2.5-0.5B-Instruct` native | 38.73 | 41.48 | 45.80 | 42.00 |
| Rosetta fuser baseline | 43.06 | 54.52 | 52.60 | 50.06 |

### Artifacts

- Summary directory: `local/final_results/paper_native_baseline/`
- Detailed README: `local/final_results/paper_native_baseline/README.md`
- Early GSM8K baseline directory: `local/final_results/baseline_align_longest_gsm8k/`

### Notes

- These results are reference baselines, not route-1 method evidence.
- The early GSM8K Rosetta-longest result was poor and should not be used as a main claim without rerunning under the final route-1 setup.

## E1 Route-1 TinyLlama Hard Alignment

### Goal

Run the first executable true cross-tokenizer route-1 closed loop before the local Llama-3.2 model was available.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Strategies: `longest`, hard `span_overlap`
- Train data: MMLU `auxiliary_train`, `num_samples=15000`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA

### Results

| Strategy | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs `longest` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `longest` | 45.53 | 56.09 | 48.80 | 50.14 | +0.00 |
| hard `span_overlap` | 43.48 | 52.61 | 40.80 | 45.63 | -4.51 |

### Artifacts

- Record: `local/final_results/route1_alignment/README.md`
- Checkpoints:
  - `local/checkpoints/route1_alignment/qwen3_0.6b+tinyllama1.1b_longest_mmlu15k/final`
  - `local/checkpoints/route1_alignment/qwen3_0.6b+tinyllama1.1b_span_overlap_mmlu15k/final`
- Eval summaries:
  - `local/final_results/route1_alignment/qwen3_tinyllama_longest/`
  - `local/final_results/route1_alignment/qwen3_tinyllama_span_overlap/`

### Notes

- Hard `span_overlap` underperformed `longest`, indicating that hard single-index span selection can amplify tokenizer mismatch noise.
- This result motivated moving from hard span alignment to soft weighted span alignment.

## E2 Route-1 Qwen3 + Llama-3.2 Soft Alignment

### Goal

Test the full three-way alignment comparison on the initially preferred Llama-family sharer after the local gated model became available.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `meta-llama/Llama-3.2-1B-Instruct`
- Local sharer path: `/home/lijunsi/projects/KVcache/models/c2c/Llama-3.2-1B-Instruct`
- Strategies: `longest`, hard `span_overlap`, `soft_span_overlap`
- Soft alignment: `top_k=4`
- Train data: MMLU `auxiliary_train`, `num_samples=15000`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA

### Alignment Diagnostics

| Metric | Value |
| --- | ---: |
| `hard_longest_span_change_rate` | 3.34% |
| `soft_one_to_many_rate` | 0.00% |
| `soft_many_to_one_source_rate` | 1.38% |
| `soft_avg_nonzero_candidates` | 1.0000 |
| `soft_avg_entropy` | 0.0000 |
| Avg receiver tokens | 287.77 |
| Avg sharer tokens | 304.33 |

### Results

| Strategy | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs `longest` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `longest` | 44.41 | 53.39 | 44.80 | 47.53 | +0.00 |
| hard `span_overlap` | 44.90 | 54.00 | 48.00 | 48.97 | +1.44 |
| `soft_span_overlap` | 47.46 | 55.39 | 45.60 | 49.48 | +1.95 |

### Training Stability

| Strategy | Final train loss | Final eval loss | Non-finite train windows |
| --- | ---: | ---: | ---: |
| `longest` | `nan` | 0.1076 | 3 |
| hard `span_overlap` | `nan` | 0.1047 | 3 |
| `soft_span_overlap` | 0.2714 | 0.1030 | 0 |

### Artifacts

- Record: `local/final_results/route1_alignment/EXPERIMENT_LOG.md`
- Diagnostic: `local/final_results/route1_alignment/diagnostics/qwen3_llama32_local/route1_alignment_diag_topk4_n256.json`
- Checkpoints:
  - `local/checkpoints/route1_alignment/qwen3_0.6b+llama3.2_1b_longest_mmlu15k_local/final`
  - `local/checkpoints/route1_alignment/qwen3_0.6b+llama3.2_1b_span_overlap_mmlu15k_local/final`
  - `local/checkpoints/route1_alignment/qwen3_0.6b+llama3.2_1b_soft_span_overlap_mmlu15k_local/final`
- Eval summaries:
  - `local/final_results/route1_alignment/qwen3_llama_longest_local/`
  - `local/final_results/route1_alignment/qwen3_llama_span_overlap_local/`
  - `local/final_results/route1_alignment/qwen3_llama_soft_span_overlap_local/`

### Notes

- This is a mild tokenizer mismatch pair.
- `soft_span_overlap` improved average accuracy and training stability, but the diagnostic shows that it often degenerates to one active source token on this pair.

## E3 Route-1 Qwen3 + TinyLlama Strong Mismatch Stress Test

### Goal

Validate whether route-1 soft alignment is more useful in a genuinely difficult cross-tokenizer setting than in a mild mismatch pair.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Strategies: `longest`, hard `span_overlap`, `soft_span_overlap`
- Soft alignment: `top_k=4`
- Train data: MMLU `auxiliary_train`, `num_samples=15000`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- This summary was produced from existing completed runs; no rerun was performed.

### Alignment Diagnostics

| Metric | Value |
| --- | ---: |
| `hard_longest_span_change_rate` | 23.87% |
| `soft_one_to_many_rate` | 14.05% |
| `soft_many_to_one_source_rate` | 0.27% |
| `soft_avg_nonzero_candidates` | 1.1569 |
| `soft_avg_entropy` | 0.0980 |
| Avg receiver tokens | 287.77 |
| Avg sharer tokens | 333.40 |

### Results

| Strategy | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs `longest` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `longest` | 45.53 | 56.09 | 48.80 | 50.14 | +0.00 |
| hard `span_overlap` | 43.48 | 52.61 | 40.80 | 45.63 | -4.51 |
| `soft_span_overlap` | 47.07 | 58.17 | 53.80 | 53.01 | +2.88 |

### Disagreement Analysis

| Bucket | Count |
| --- | ---: |
| `soft_correct_both_hard_wrong` | 609 |
| `soft_correct_longest_wrong` | 80 |
| `soft_correct_span_wrong` | 544 |
| `soft_wrong_both_hard_correct` | 434 |
| `soft_wrong_longest_correct` | 119 |
| `soft_wrong_span_correct` | 388 |

### Artifacts

- Stress summary: `local/final_results/route1_alignment/stress_test_summary/route1_mismatch_stress_summary.md`
- Score CSV: `local/final_results/route1_alignment/stress_test_summary/route1_mismatch_stress_scores.csv`
- Diagnostic CSV: `local/final_results/route1_alignment/stress_test_summary/route1_mismatch_stress_diagnostics.csv`
- Disagreement CSV: `local/final_results/route1_alignment/stress_test_summary/qwen3_tinyllama_strategy_disagreements.csv`
- Diagnostic JSON: `local/final_results/route1_alignment/diagnostics/route1_alignment_diag_topk4_n256.json`
- Checkpoints:
  - `local/checkpoints/route1_alignment/qwen3_0.6b+tinyllama1.1b_longest_mmlu15k/final`
  - `local/checkpoints/route1_alignment/qwen3_0.6b+tinyllama1.1b_span_overlap_mmlu15k/final`
  - `local/checkpoints/route1_alignment/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_mmlu15k/final`

### Notes

- This is the strongest current evidence for the route-1 paper story.
- Hard `span_overlap` is worse than `longest`, while `soft_span_overlap` wins clearly.
- The key claim should be that soft weighted KV gathering mitigates cross-tokenizer hard-index mismatch, not that span overlap alone is sufficient.

## E4 Route-1 V2 Soft Scoring Ablation

### Goal

Try stronger route-1 v2 soft alignment scoring variants while preserving the existing v1 code/results as a snapshot.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Variants: v1 `soft_span_overlap` control, v2 `uniform`, v2 `overlap_power2`, v2 `boundary_power2`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Soft top-k: `4`
- V1 snapshot: `local/snapshots/route1_v1_20260427_175055`

### Diagnostics

| Method | Entropy | Top-1 weight | Top-1 boundary hit | Avg candidates | One-to-many |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soft_span_overlap_control` | 0.097984 | 0.940817 | 0.997203 | 1.156869 | 0.140546 |
| `soft_span_overlap_v2_uniform` | 0.103911 | 0.927096 | 0.997203 | 1.156869 | 0.140546 |
| `soft_span_overlap_v2_overlap_power2` | 0.086990 | 0.950703 | 0.997203 | 1.156869 | 0.140546 |
| `soft_span_overlap_v2_boundary_power2` | 0.085863 | 0.951531 | 0.998459 | 1.156869 | 0.140546 |

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soft_span_overlap_control` | 42.97 | 49.39 | 41.60 | 44.65 | +0.00 |
| `soft_span_overlap_v2_uniform` | 44.83 | 52.61 | 47.00 | 48.15 | +3.49 |
| `soft_span_overlap_v2_overlap_power2` | 42.99 | 51.91 | 44.60 | 46.50 | +1.85 |
| `soft_span_overlap_v2_boundary_power2` | 43.79 | 51.74 | 43.60 | 46.37 | +1.71 |

### Training

| Method | Final train loss | Final eval loss |
| --- | ---: | ---: |
| `soft_span_overlap_control` | 0.3767 | 0.1284 |
| `soft_span_overlap_v2_uniform` | 0.3663 | 0.1266 |
| `soft_span_overlap_v2_overlap_power2` | 0.3844 | 0.1370 |
| `soft_span_overlap_v2_boundary_power2` | 0.3795 | 0.1335 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v2/small_loop_summary/route1_v2_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v2/small_loop_summary/route1_v2_small_loop_scores.csv`
- Diagnostic CSV: `local/final_results/route1_alignment_v2/small_loop_summary/route1_v2_small_loop_diagnostics.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v2/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v2/`
- Checkpoints: `local/checkpoints/route1_alignment_v2/`

### Notes

- `soft_span_overlap_v2_uniform` is the current best v2 small-loop variant.
- Sharper weighting improves top-1 concentration but hurts downstream accuracy relative to uniform weighting.
- This suggests the next route-1 extension should keep the multi-token evidence soft and focus on adaptive weighting/gating rather than stronger hardening.

## E5 Route-1 V2.1 Adaptive Confidence

### Goal

Add an entropy-based source confidence gate on top of the best v2 uniform soft span-overlap setting, while preserving the v2 snapshot and result set.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap`, `soft_alignment_score_mode=uniform`, `top_k=4`
- New mechanism: entropy confidence gate applied to the projected KV residual
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Preserved v2 snapshot: `local/snapshots/route1_v2_20260427_231217`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2 uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| `soft_span_overlap_v2_uniform_control` | 44.83 | 52.61 | 47.00 | 48.15 | +0.00 |
| `adaptive_entropy025` | 43.77 | 52.26 | 48.00 | 48.01 | -0.14 |
| `adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 | +1.62 |
| `adaptive_entropy075` | 44.42 | 53.04 | 48.60 | 48.69 | +0.54 |

### Diagnostics

| Method | Alpha | Floor | Avg source confidence | Gated rate |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_entropy025` | 0.25 | 0.75 | 0.964864 | 14.0546% |
| `adaptive_entropy050` | 0.50 | 0.50 | 0.929727 | 14.0546% |
| `adaptive_entropy075` | 0.75 | 0.25 | 0.894591 | 14.0546% |

### Artifacts

- Summary: `local/final_results/route1_alignment_v21/small_loop_summary/route1_v21_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v21/small_loop_summary/route1_v21_small_loop_scores.csv`
- Diagnostic CSV: `local/final_results/route1_alignment_v21/small_loop_summary/route1_v21_small_loop_diagnostics.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v21/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v21/`
- Checkpoints: `local/checkpoints/route1_alignment_v21/`

### Notes

- `adaptive_entropy050` is the best current v2.1 small-loop variant.
- The result supports confidence-gated soft span alignment: too little gating is close to v2 uniform, while too much gating suppresses useful fused signal.

## E6 Route-1 V2.2 Learnable Confidence Gates

### Goal

Preserve the v2.1 best setting and test whether a learnable confidence gate can
improve strong cross-tokenizer mismatch alignment beyond static entropy
confidence.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Static confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- New mechanisms:
  - `token_mlp`: token/head confidence deltas from projector hidden states
  - `learned_affine`: global key/value confidence bias and entropy scale
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Preserved pre-v2.2 snapshot: `local/snapshots/route1_v21_20260428_before_v22`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.1 entropy050 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 | +0.00 |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +1.05 |
| `v2.2_learned_affine_entropy050` | 45.03 | 51.65 | 42.80 | 46.49 | -3.28 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 0.3779 | 0.1337 | 0.1336 |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.2_learned_affine_entropy050` | 0.3822 | 0.1254 | 0.1243 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v22/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v22/`
- Checkpoints: `local/checkpoints/route1_alignment_v22/`

### Notes

- `v2.2_token_mlp_entropy050` is the current best route-1 small-loop result. It
  improves all three downstream tasks over v2.1 `adaptive_entropy050`.
- `v2.2_learned_affine_entropy050` is a negative result: it has the best eval
  loss but poor downstream transfer, especially on OpenBookQA.
- The result supports a route-1 v2 direction based on local adaptive confidence
  over soft span alignment, rather than global scalar confidence calibration.

## E7 Route-1 V2.3 Delta L2 Regularization

### Goal

Test whether a simple trust-region regularizer on the v2.2 `token_mlp`
confidence deltas improves strong cross-tokenizer route-1 alignment.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence gate: `token_mlp`
- Regularizer: `alignment_confidence_delta_l2_weight=0.01`
- Static confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Preserved pre-v2.3 snapshot: `local/snapshots/route1_v22_20260428_before_v23`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 | -1.05 |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.3_token_mlp_delta_l2_0p01` | 45.40 | 53.30 | 49.40 | 49.37 | -1.45 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 0.3779 | 0.1337 | 0.1336 |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.3_token_mlp_delta_l2_0p01` | 0.3706 | 0.1783 | 0.1747 |

### Diagnostics

| Metric | Value |
| --- | ---: |
| `projector/alignment_aux_loss` | 0.00018 |
| `projector/alignment_delta_l2` | 0.01774 |
| `projector/key_confidence_mean` | 0.93750 |
| `projector/value_confidence_mean` | 0.93750 |
| `projector/key_confidence_std` | 0.16476 |
| `projector/value_confidence_std` | 0.16549 |
| `projector/key_delta_abs_mean` | 0.08562 |
| `projector/value_delta_abs_mean` | 0.02242 |
| `projector/key_delta_abs_max` | 0.47789 |
| `projector/value_delta_abs_max` | 0.44671 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v23/small_loop_summary/route1_v23_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v23/small_loop_summary/route1_v23_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v23/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v23/`
- Checkpoints: `local/checkpoints/route1_alignment_v23/`
- Eval summaries: `local/final_results/route1_alignment_v23/qwen3_tinyllama_token_mlp_delta_l2_0p01_small2048/`
- Wandb offline run: `wandb/offline-run-20260428_152311-zcdhlqvy`

### Notes

- `v2.3_token_mlp_delta_l2_0p01` is a negative ablation. It underperforms
  `v2.2_token_mlp_entropy050` on all three downstream tasks.
- The result suggests that useful token-level confidence learning needs local
  confident deviations from the entropy prior.
- Future v2.3 variants should focus on selective or data-dependent confidence
  calibration, not a global L2 penalty on every confidence delta.

## E8 Route-1 V2.4 Selective Delta L2 Regularization

### Goal

Test whether the v2.3 delta L2 regularizer becomes useful when applied only to
uncertain soft-alignment rows instead of every token.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence gate: `token_mlp`
- Regularizer: `alignment_confidence_delta_l2_weight=0.01`
- Regularizer mode: `alignment_confidence_delta_l2_mode=uncertain`
- Uncertain rows: source confidence `<0.999` or normalized source-weight entropy `>0.0`
- Static confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Preserved pre-v2.4 snapshot: `local/snapshots/route1_v23_20260428_before_v24`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 | -1.05 |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.3_token_mlp_delta_l2_0p01` | 45.40 | 53.30 | 49.40 | 49.37 | -1.45 |
| `v2.4_token_mlp_selective_delta_l2_0p01` | 45.28 | 52.61 | 46.20 | 48.03 | -2.79 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 0.3779 | 0.1337 | 0.1336 |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.3_token_mlp_delta_l2_0p01` | 0.3706 | 0.1783 | 0.1747 |
| `v2.4_token_mlp_selective_delta_l2_0p01` | 0.3797 | 0.2108 | 0.1992 |

### Diagnostics

| Metric | Value |
| --- | ---: |
| `projector/alignment_aux_loss` | 0.00024 |
| `projector/alignment_delta_l2` | 0.02192 |
| `projector/alignment_regularized_delta_l2` | 0.02386 |
| `projector/alignment_regularization_selected_rate` | 0.12625 |
| `projector/key_confidence_mean` | 0.93806 |
| `projector/value_confidence_mean` | 0.93750 |
| `projector/key_confidence_std` | 0.16462 |
| `projector/value_confidence_std` | 0.16579 |
| `projector/key_delta_abs_mean` | 0.08640 |
| `projector/value_delta_abs_mean` | 0.01882 |
| `projector/key_delta_abs_max` | 0.50095 |
| `projector/value_delta_abs_max` | 0.24196 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v24/small_loop_summary/route1_v24_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v24/small_loop_summary/route1_v24_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v24/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v24/`
- Checkpoints: `local/checkpoints/route1_alignment_v24/`
- Eval summaries: `local/final_results/route1_alignment_v24/qwen3_tinyllama_token_mlp_selective_delta_l2_0p01_small2048/`
- Wandb offline run: `wandb/offline-run-20260428_185543-8loxzg7x`

### Notes

- `v2.4_token_mlp_selective_delta_l2_0p01` is a negative ablation.
- Selective regularization avoided global pressure on all rows, but still hurt
  all three downstream tasks compared with v2.2.
- The trust-region regularization path is now weak: both global and selective
  delta L2 underperform the unconstrained v2.2 token MLP.

## E9 Route-1 V2.5 Token MLP Confidence Diagnostics

### Goal

Diagnose why `v2.2_token_mlp_entropy050` is the best current route-1
small-loop method before designing the next route-1 v2.5 candidate.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Checkpoint: `local/checkpoints/route1_alignment_v22/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v22_token_mlp_entropy050_small2048/final`
- Eval config: `local/tmp/eval_configs/route1_alignment_v22/route1_v22_qwen3_tinyllama_token_mlp_entropy050_small2048_mmlu-redux.yaml`
- Dataset: MMLU `auxiliary_train`
- Samples: 64
- This is a diagnostic run only; no new training checkpoint was produced.
- Preserved pre-diagnostic snapshot: `local/snapshots/route1_v24_20260428_before_v25_diag`

### Diagnostics

| Metric | Value |
| --- | ---: |
| Mean forward loss | 0.155936 |
| Mean absolute key confidence delta | 0.629674 |
| Mean absolute value confidence delta | 0.105145 |
| Mean key confidence | 0.929608 |
| Mean value confidence | 0.931608 |
| Mean source confidence | 0.929203 |
| Mean normalized source entropy | 0.141422 |

### Alignment Buckets

| Bucket | Count per layer | Ratio |
| --- | ---: | ---: |
| `confident_1to1` | 110784 | 85.86% |
| `entropy_ambiguous` | 18248 | 14.14% |
| `fallback` | 0 | 0.00% |
| `low_confidence_nonentropy` | 0 | 0.00% |

### Layer Pattern

| Layer group | Key delta abs mean | Value delta abs mean |
| --- | ---: | ---: |
| Early layers 0-7 | 1.2230 | 0.0196 |
| Middle layers 8-19 | 0.4295 | 0.0650 |
| Late layers 20-27 | 0.3366 | 0.2508 |

Top key-delta layers are early (`0`, `1`, `5`, `4`, `6`), while top
value-delta layers are late (`27`, `26`, `24`, `25`, `21`). The same pattern is
visible at head level: top key heads concentrate in layers `0` and `1`, while
top value heads concentrate in layer `27`.

### Artifacts

- Diagnostic summary: `local/final_results/route1_alignment_v25/diagnostics/v22_token_mlp_gate_mmlu_aux64/diagnostic_summary.md`
- Summary JSON: `local/final_results/route1_alignment_v25/diagnostics/v22_token_mlp_gate_mmlu_aux64/summary.json`
- Layer-bucket CSV: `local/final_results/route1_alignment_v25/diagnostics/v22_token_mlp_gate_mmlu_aux64/layer_bucket_stats.csv`
- Layer-head CSV: `local/final_results/route1_alignment_v25/diagnostics/v22_token_mlp_gate_mmlu_aux64/layer_head_stats.csv`
- Diagnostic script: `script/analysis/route1_confidence_gate_diagnostics.py`

### Notes

- v2.2 token MLP behaves like a layer-aware, key/value-asymmetric confidence
  controller, not like a uniform global confidence calibrator.
- This explains the v2.3 and v2.4 negative results: delta L2 regularization
  suppresses large local corrections that appear to be useful.
- The next route-1 v2.5 candidate should preserve v2.2 local confidence
  freedom and add explicit layer awareness, such as separate key/value
  per-layer schedules or learnable per-layer scales.

## E10 Route-1 V2.5 Static Layer-Aware Gate

### Goal

Test whether the layer pattern found in E9 can improve route-1 by explicitly
scaling token-level confidence deltas by target layer.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: static layer-aware token delta scaling
  - key scale: `1.5 -> 0.5` from target layer `0` to `27`
  - value scale: `0.5 -> 1.5` from target layer `0` to `27`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`
- Preserved pre-implementation snapshot: `local/snapshots/route1_v25_20260428_195452_before_layer_gate`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.5_layer_gate_eklv150050` | 47.02 | 54.26 | 49.60 | 50.29 | -0.53 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.5_layer_gate_eklv150050` | 0.3813 | 0.1810 | 0.1794 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v25/small_loop_summary/route1_v25_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v25/small_loop_summary/route1_v25_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v25/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v25/`
- Checkpoint: `local/checkpoints/route1_alignment_v25/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v25_layer_gate_eklv150050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v25/qwen3_tinyllama_layer_gate_eklv150050_small2048/`
- W&B offline run: `wandb/offline-run-20260428_200233-xoezqhlg`

### Notes

- `v2.5_layer_gate_eklv150050` is a near-neutral but negative ablation compared
  with v2.2.
- The result suggests that simply hard-coding the diagnosed early-key/late-value
  pattern is too restrictive or too coarse.
- The layer-aware direction remains plausible, but the next variant should use
  learnable or weakly-constrained per-layer scales instead of a fixed strong
  schedule.

## E11 Route-1 V2.6 Learned Layer Scale

### Goal

Test whether v2.2's diagnosed early-key/late-value asymmetry can be captured by
learnable per-layer key/value confidence-delta scales while preserving the
token-level flexibility of the `token_mlp` confidence gate.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: learned per-projector key/value scalar confidence-delta scales
- Scale initialization: key `1.0`, value `1.0`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Debugging Note

The first v2.6a implementation accidentally allowed the scalar scale parameters
to be cast to bf16 during `RosettaModel` initialization. That run completed but
the checkpoint scales stayed exactly at `1.0`, so it is not a valid learned-scale
ablation. It is preserved as a debugging artifact:
`local/snapshots/route1_v26a_20260429_bf16_scale_stuck`.

The corrected v2.6b implementation keeps the learned scalar parameters in
`torch.float32` even after module-level dtype casting. Smoke testing confirmed
the scales moved away from initialization before the full 2048-sample run.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.5_layer_gate_eklv150050` | 47.02 | 54.26 | 49.60 | 50.29 | -0.53 |
| `v2.6b_learned_layer_scale_fp32_init1` | 44.96 | 51.13 | 41.60 | 45.90 | -4.92 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.5_layer_gate_eklv150050` | 0.3813 | 0.1810 | 0.1794 |
| `v2.6a_learned_layer_scale_bf16_stuck` | 0.3768 | 0.1849 | 0.1833 |
| `v2.6b_learned_layer_scale_fp32_init1` | 0.3685 | 0.2065 | 0.2041 |

### Learned Scale Check

| Scale | Min | Max | Mean | Dtype |
| --- | ---: | ---: | ---: | --- |
| Key | 0.99893814 | 1.00126684 | 1.00023809 | `torch.float32` |
| Value | 0.99939036 | 1.00173926 | 1.00058392 | `torch.float32` |

### Artifacts

- Summary: `local/final_results/route1_alignment_v26b/small_loop_summary/route1_v26b_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v26b/small_loop_summary/route1_v26b_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v26b/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v26b/`
- Checkpoint: `local/checkpoints/route1_alignment_v26b/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v26b_learned_layer_scale_fp32_init1_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v26b/qwen3_tinyllama_learned_layer_scale_fp32_init1_small2048/`
- Invalid v2.6a checkpoint: `local/checkpoints/route1_alignment_v26/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v26_learned_layer_scale_init1_small2048/final`
- Invalid v2.6a snapshot: `local/snapshots/route1_v26a_20260429_bf16_scale_stuck`

### Validation

- `python -m py_compile rosetta/model/projector.py script/train/SFT_train.py rosetta/train/model_utils.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `19 passed`

### Notes

- v2.6b is a clear negative ablation. It lowers train loss but worsens held-out
  eval loss and all three downstream benchmarks.
- The learned scales moved only slightly from `1.0`, which suggests scalar
  layer gates are too weak or too indirect under this small-loop setup.
- Route-1 should keep v2.2 as the current best candidate. The next variant
  should act directly on token/span-level alignment confidence or top-k weighting
  rather than adding another global scalar gate on top of `token_mlp`.

## E12 Route-1 V2.7 Adaptive Overlap Reweighting

### Goal

Test a direct token/span-level extension of v2.2 by reweighting the selected
top-k soft span alignment candidates with overlap evidence. This tests whether a
fixed span-evidence calibration rule can improve cross-tokenizer generalization
without changing the current best confidence gate.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: adaptive overlap reweighting of top-k soft span weights
- Reweighting: `soft_alignment_reweight_mode=adaptive_overlap`, strength `1.0`, power `2.0`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.5_layer_gate_eklv150050` | 47.02 | 54.26 | 49.60 | 50.29 | -0.53 |
| `v2.6b_learned_layer_scale_fp32_init1` | 44.96 | 51.13 | 41.60 | 45.90 | -4.92 |
| `v2.7_adaptive_overlap_p2s1_entropy050` | 46.50 | 54.61 | 50.80 | 50.64 | -0.18 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.5_layer_gate_eklv150050` | 0.3813 | 0.1810 | 0.1794 |
| `v2.6b_learned_layer_scale_fp32_init1` | 0.3685 | 0.2065 | 0.2041 |
| `v2.7_adaptive_overlap_p2s1_entropy050` | 0.3647 | 0.1931 | 0.1895 |

### Implementation

- `rosetta/model/aligner.py`: added optional `adaptive_overlap` reweighting for
  post-normalized soft span top-k weights.
- `script/train/SFT_train.py`: forwards reweighting config into `TokenAligner`.
- `script/evaluation/unified_evaluator.py`: forwards reweighting config into
  `TokenAligner` during benchmark evaluation.
- `test/test_aligner_span_overlap.py`: adds unit coverage for adaptive
  reweighting and equal-overlap no-op behavior.

### Artifacts

- Summary: `local/final_results/route1_alignment_v27/small_loop_summary/route1_v27_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v27/small_loop_summary/route1_v27_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v27/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v27/`
- Checkpoint: `local/checkpoints/route1_alignment_v27/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v27_adaptive_overlap_p2s1_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v27/qwen3_tinyllama_adaptive_overlap_p2s1_entropy050_small2048/`
- W&B offline smoke run: `wandb/offline-run-20260429_154349-xt1z7mnr`
- W&B offline full run: `wandb/offline-run-20260429_154520-eka1ar0s`

### Validation

- `python -m py_compile rosetta/model/aligner.py rosetta/model/projector.py script/train/SFT_train.py script/evaluation/unified_evaluator.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `21 passed`

### Notes

- v2.7 is mixed but negative relative to v2.2. It improves train loss and
  OpenBookQA, but MMLU-Redux, AI2-ARC, held-out eval loss, and mean accuracy are
  worse.
- Direct fixed overlap sharpening appears too brittle under the 2048-sample
  small loop. It may overfit the training objective while weakening
  generalization.
- This does not invalidate route-1. It suggests the next token/span-level
  variant should use learnable or context-conditioned calibration instead of a
  fixed overlap-power rule.

## E13 Route-1 V2.8 Span MLP Weight Calibration

### Goal

Test whether a learnable, context-conditioned calibration of top-k soft span
alignment weights improves strong cross-tokenizer route-1 transfer. Unlike v2.7
fixed overlap-power reweighting, v2.8 initializes as a no-op and learns a small
per-candidate logit delta before source KV gathering.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `alignment_weight_calibration_mode=span_mlp`
- Calibration max delta: `1.0`
- Reweight mode: `none`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.5_layer_gate_eklv150050` | 47.02 | 54.26 | 49.60 | 50.29 | -0.53 |
| `v2.6b_learned_layer_scale_fp32_init1` | 44.96 | 51.13 | 41.60 | 45.90 | -4.92 |
| `v2.7_adaptive_overlap_p2s1_entropy050` | 46.50 | 54.61 | 50.80 | 50.64 | -0.18 |
| `v2.8_span_mlp_calibrator_entropy050` | 44.94 | 54.26 | 51.00 | 50.07 | -0.75 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.5_layer_gate_eklv150050` | 0.3813 | 0.1810 | 0.1794 |
| `v2.6b_learned_layer_scale_fp32_init1` | 0.3685 | 0.2065 | 0.2041 |
| `v2.7_adaptive_overlap_p2s1_entropy050` | 0.3647 | 0.1931 | 0.1895 |
| `v2.8_span_mlp_calibrator_entropy050` | 0.3696 | 0.1370 | 0.1374 |

### Calibration Movement Check

| Metric | Value |
| --- | ---: |
| Number of projector modules | 28 |
| Weight absolute mean | 0.00013957 |
| Weight absolute max | 0.00138573 |
| Bias absolute mean | 0.00000007 |
| Bias absolute max | 0.00000090 |

### Implementation

- `rosetta/model/projector.py`: added `Projector.calibrate_source_weights`
  default no-op and `C2CProjector` `span_mlp` calibration mode.
- `rosetta/model/wrapper.py`: calls the projector calibration before weighted
  source KV gathering for soft alignment.
- `test/test_aligner_span_overlap.py`: added unit coverage for no-op
  initialization, fp32 parameter retention, and learnable rank-bias behavior.

### Artifacts

- Summary: `local/final_results/route1_alignment_v28/small_loop_summary/route1_v28_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v28/small_loop_summary/route1_v28_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v28/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v28/`
- Checkpoint: `local/checkpoints/route1_alignment_v28/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v28_span_mlp_calibrator_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v28/qwen3_tinyllama_span_mlp_calibrator_entropy050_small2048/`
- W&B offline smoke run: `wandb/offline-run-20260429_164641-0azum4ds`
- W&B offline full run: `wandb/offline-run-20260429_164746-tnh43qzv`

### Validation

- `python -m py_compile rosetta/model/aligner.py rosetta/model/projector.py rosetta/model/wrapper.py script/train/SFT_train.py script/evaluation/unified_evaluator.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `23 passed`

### Notes

- v2.8 is a mixed negative ablation relative to v2.2. It strongly improves the
  small held-out eval loss, but benchmark mean drops by `0.75`.
- OpenBookQA improves over v2.2 by `0.40`, while MMLU-Redux drops by `2.13`
  and AI2-ARC drops by `0.52`.
- This indicates a validation/benchmark mismatch: the span MLP can optimize the
  small validation objective, but that signal is not yet robust across
  downstream benchmarks.
- v2.2 remains the best current stress-pair route-1 baseline.

## E14 Route-1 V2.8 Flip Diagnostics

### Goal

Compare the current best route-1 stress-pair baseline
`v2.2_token_mlp_entropy050` against the v2.8 span MLP calibrator to identify
whether v2.8 failed because of malformed outputs, answer extraction issues, or
answer-level flip balance.

### Setup

- Baseline run: `local/final_results/route1_alignment_v22/qwen3_tinyllama_token_mlp_entropy050_small2048`
- Candidate run: `local/final_results/route1_alignment_v28/qwen3_tinyllama_span_mlp_calibrator_entropy050_small2048`
- Compared files: completed `*_cot.csv` outputs for MMLU-Redux, AI2-ARC, and OpenBookQA
- Diagnostic script: `script/analysis/route1_eval_flip_diagnostics.py`

### Dataset Summary

| Dataset | Base Acc | Candidate Acc | Delta | Regression | Improvement | Net | Pred Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AI2-ARC | 54.78 | 54.26 | -0.52 | 78 | 72 | -6 | 18.35 |
| MMLU-Redux | 47.07 | 44.94 | -2.13 | 547 | 427 | -120 | 27.79 |
| OpenBookQA | 50.60 | 51.00 | +0.40 | 28 | 30 | +2 | 17.80 |

### Output Format Check

| Dataset | Baseline blank | Candidate blank | Baseline invalid | Candidate invalid | Baseline gen len | Candidate gen len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AI2-ARC | 0 | 0 | 0 | 0 | 7.00 | 6.36 |
| MMLU-Redux | 0 | 0 | 0 | 0 | 6.98 | 6.55 |
| OpenBookQA | 0 | 0 | 0 | 0 | 7.00 | 6.53 |

### Answer Distribution Shift

| Dataset | Distribution | A | B | C | D |
| --- | --- | ---: | ---: | ---: | ---: |
| AI2-ARC | true | 22.9 | 26.2 | 26.3 | 24.6 |
| AI2-ARC | v2.2 pred | 24.2 | 18.2 | 26.7 | 31.0 |
| AI2-ARC | v2.8 pred | 33.0 | 18.9 | 22.5 | 25.6 |
| MMLU-Redux | true | 22.8 | 24.5 | 25.6 | 27.1 |
| MMLU-Redux | v2.2 pred | 26.0 | 15.0 | 25.7 | 33.3 |
| MMLU-Redux | v2.8 pred | 43.6 | 13.5 | 18.4 | 24.4 |
| OpenBookQA | true | 27.6 | 25.2 | 26.4 | 20.8 |
| OpenBookQA | v2.2 pred | 33.4 | 13.2 | 24.2 | 29.2 |
| OpenBookQA | v2.8 pred | 35.4 | 14.8 | 22.2 | 27.6 |

### Artifacts

- Diagnostic summary: `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/diagnostic_summary.md`
- Dataset summary CSV: `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/dataset_summary.csv`
- Subject summary CSV: `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/subject_summary.csv`
- Flip examples CSV: `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/flip_examples.csv`
- Output checks JSON: `local/final_results/route1_alignment_v28/diagnostics/v22_vs_v28_flip_diagnostics/output_checks.json`

### Notes

- v2.8 is not failing because of blank or invalid answer extraction; those
  counts are zero for all three datasets.
- The main loss comes from answer-level flip balance. MMLU-Redux has 547
  regressions versus 427 improvements, producing a net `-120` flip count.
- v2.8 introduces a strong answer-prior shift toward option `A`, especially on
  MMLU-Redux where predicted `A` rises from 26.0% in v2.2 to 43.6% in v2.8.
- The most likely issue is calibration-induced answer bias, not route-1 soft
  alignment itself. v2.8 should not replace v2.2 unless a constrained variant
  removes this answer-prior shift.

## E15 Route-1 V2.8b Constrained Span Weight Calibration

### Goal

Test whether a constrained version of v2.8 can remove the answer-prior shift by
only calibrating ambiguous / low-confidence soft alignment rows, limiting the
maximum top-k logit delta, and adding no-op regularization.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `alignment_weight_calibration_mode=span_mlp`
- Calibration apply mode: `ambiguous`
- Entropy threshold: `0.05`
- Confidence threshold: `0.999`
- Calibration max delta: `0.25`
- Delta L2 weight: `0.01`
- Entropy L2 weight: `0.01`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.8_span_mlp_calibrator_entropy050` | 44.94 | 54.26 | 51.00 | 50.07 | -0.75 |
| `v2.8b_ambig_reg01_max025_entropy050` | 44.94 | 54.26 | 51.00 | 50.07 | -0.75 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.8_span_mlp_calibrator_entropy050` | 0.3696 | 0.1370 | 0.1374 |
| `v2.8b_ambig_reg01_max025_entropy050` | 0.3696 | 0.1370 | 0.1374 |

### Diagnostics

| Diagnostic | Result |
| --- | --- |
| Smoke selected calibration rows | `projector/weight_calibration_selected_rate = 0.13429` |
| Full selected calibration rows | `projector/weight_calibration_selected_rate = 0.11111` |
| Full max weight delta | `projector/weight_delta_abs_max = 0.00149` |
| Full mean weight delta | `projector/weight_delta_abs_mean = 0.00107` |
| v2.8 vs v2.8b prediction changes | `0.00%` on MMLU-Redux, AI2-ARC, OpenBookQA |

### Implementation

- `rosetta/model/projector.py`: added ambiguous-only calibration selection,
  confidence-aware row selection, max-delta constrained span MLP calibration,
  and delta/entropy no-op regularization.
- `rosetta/model/wrapper.py`: passes soft-alignment source confidence into the
  projector calibration step.
- `script/train/SFT_train.py`: logs span-weight calibration entropy, selected
  row rate, delta magnitude, and auxiliary loss diagnostics.
- `test/test_aligner_span_overlap.py`: adds coverage for ambiguous-row targeting
  and regularization backpropagation.

### Artifacts

- Summary: `local/final_results/route1_alignment_v28b/small_loop_summary/route1_v28b_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v28b/small_loop_summary/route1_v28b_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v28b/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v28b/`
- Checkpoint: `local/checkpoints/route1_alignment_v28b/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v28b_ambig_reg01_max025_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v28b/qwen3_tinyllama_ambig_reg01_max025_entropy050_small2048/`
- v2.2 vs v2.8b diagnostics: `local/final_results/route1_alignment_v28b/diagnostics/v22_vs_v28b_flip_diagnostics/diagnostic_summary.md`
- v2.8 vs v2.8b diagnostics: `local/final_results/route1_alignment_v28b/diagnostics/v28_vs_v28b_output_diff/diagnostic_summary.md`
- W&B offline smoke run: `wandb/offline-run-20260429_175433-airltrin`
- W&B offline full run: `wandb/offline-run-20260429_175531-6ywq8tgm`

### Validation

- `python -m black rosetta/model/projector.py rosetta/model/wrapper.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- `python -m py_compile rosetta/model/projector.py rosetta/model/wrapper.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `25 passed`

### Notes

- v2.8b is a negative ablation. It keeps the v2.8 small-loop validation loss
  but does not recover any benchmark accuracy.
- Direct v2.8 vs v2.8b diagnostics show zero prediction changes on all three
  benchmarks, so the constrained calibrator did not alter downstream behavior.
- The calibration branch is active, but the learned movement is too small to
  change generation. This suggests span-weight calibrator variants should not
  be the next main route-1 direction unless paired with a stronger objective.

## E16 Route-1 V2.9 Learned Residual Scale

### Goal

Test whether a learnable per-layer key/value residual injection scale can
improve route-1 transfer while preserving v2.2's token-level confidence freedom.
The scale initializes as a no-op, so the initial behavior is compatible with
v2.2.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `alignment_residual_scale_mode=learned`
- Residual max delta: `1.0`
- Residual scale L2 weight: `0.01`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.7_adaptive_overlap_p2s1_entropy050` | 46.50 | 54.61 | 50.80 | 50.64 | -0.18 |
| `v2.8_span_mlp_calibrator_entropy050` | 44.94 | 54.26 | 51.00 | 50.07 | -0.75 |
| `v2.9_residual_scale_l2p01_entropy050` | 45.40 | 52.43 | 44.60 | 47.48 | -3.34 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.9_residual_scale_l2p01_entropy050` | 0.3687 | 0.1916 | 0.1895 |

### Diagnostics

| Diagnostic | Result |
| --- | --- |
| Key delta abs mean | `0.000601` |
| Value delta abs mean | `0.000519` |
| Key scale mean | `0.999479` |
| Value scale mean | `1.000389` |
| v2.2 vs v2.9 total regressions / improvements | `589 / 438` |
| v2.2 vs v2.9 net flip count | `-151` |
| v2.9 MMLU-Redux predicted D rate | `50.7%` |
| v2.9 OpenBookQA predicted D rate | `54.8%` |

### Implementation

- `rosetta/model/projector.py`: added `alignment_residual_scale_mode=learned`,
  bounded key/value residual scale deltas, fp32 scalar parameter retention, and
  residual-scale auxiliary regularization.
- `script/train/SFT_train.py`: logs residual scale, residual delta L2, and
  residual auxiliary loss.
- `test/test_aligner_span_overlap.py`: adds tests for residual-scale output
  modulation, regularization backpropagation, and fp32 parameter retention.
- `script/analysis/route1_eval_flip_diagnostics.py`: generalized the diagnostic
  markdown wording beyond v2.8-specific labels.

### Artifacts

- Summary: `local/final_results/route1_alignment_v29/small_loop_summary/route1_v29_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v29/small_loop_summary/route1_v29_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v29/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v29/`
- Checkpoint: `local/checkpoints/route1_alignment_v29/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v29_residual_scale_l2p01_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v29/qwen3_tinyllama_residual_scale_l2p01_entropy050_small2048/`
- Flip diagnostics: `local/final_results/route1_alignment_v29/diagnostics/v22_vs_v29_flip_diagnostics/diagnostic_summary.md`
- W&B offline smoke run: `wandb/offline-run-20260430_151341-93n3m5qm`
- W&B offline full run: `wandb/offline-run-20260430_151436-wxwsef28`

### Validation

- `python -m py_compile rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- `python -m black rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `28 passed`

### Notes

- v2.9 is a negative ablation. It underperforms v2.2 on MMLU-Redux, AI2-ARC,
  and OpenBookQA, with mean accuracy dropping by `3.34`.
- The residual scalar parameters barely move from no-op initialization, but the
  generated answers shift strongly toward option `D`. This makes the branch
  unpromising as a route-1 mainline.
- v2.2 remains the current best stress-pair method.

## E17 Route-1 V2.10 Answer-Prior Regularization

### Goal

Test whether a smoothed answer-choice prior regularizer can reduce
multiple-choice answer bias and improve downstream transfer for the current
best route-1 stress-pair setup.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `answer_prior_regularization`
- Answer-prior target: `smoothed_gold`
- Label smoothing: `0.7`
- Auxiliary weight: `0.03`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 45.12 | 53.04 | 47.40 | 48.52 | -2.30 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 0.3831 | 0.1714 | 0.1673 |

### Diagnostics

| Dataset | Regression | Improvement | Net flip | Pred changed | v2.10 Pred D |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI2-ARC | 65 | 45 | -20 | 15.22 | 42.6 |
| MMLU-Redux | 444 | 334 | -110 | 23.28 | 50.1 |
| OpenBookQA | 50 | 34 | -16 | 25.60 | 52.0 |

### Implementation

- `rosetta/train/answer_prior.py`: added answer-option token resolution and a
  causal-shifted KL auxiliary loss over supervised ABCD answer positions.
- `script/train/SFT_train.py`: applies the answer-prior loss during training,
  logs answer-prior metrics, and keeps evaluation loss free of this auxiliary
  term.
- `test/test_answer_prior_regularization.py`: adds unit coverage for the helper.

### Artifacts

- Summary: `local/final_results/route1_alignment_v210/small_loop_summary/route1_v210_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v210/small_loop_summary/route1_v210_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v210/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v210/`
- Checkpoint: `local/checkpoints/route1_alignment_v210/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v210_answer_prior_smooth070_w003_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v210/qwen3_tinyllama_answer_prior_smooth070_w003_entropy050_small2048/`
- Flip diagnostics: `local/final_results/route1_alignment_v210/diagnostics/v22_vs_v210_flip_diagnostics/diagnostic_summary.md`
- W&B offline smoke run: `wandb/offline-run-20260430_173038-dwvne2c3`
- W&B offline full run: `wandb/offline-run-20260430_173137-vjvs7036`

### Validation

- `python -m black rosetta/train/answer_prior.py script/train/SFT_train.py test/test_answer_prior_regularization.py`
- `python -m py_compile rosetta/train/answer_prior.py script/train/SFT_train.py test/test_answer_prior_regularization.py test/test_aligner_span_overlap.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_answer_prior_regularization.py test/test_aligner_span_overlap.py -q`
- Result: `31 passed`

### Notes

- v2.10 is a negative ablation. It slightly improves small-loop final eval loss
  but is worse than v2.2 on MMLU-Redux, AI2-ARC, and OpenBookQA.
- Blank and invalid prediction counts are zero, so this is not an output-format
  collapse.
- The main failure mode is stronger answer-prior shift toward option `D`.
  Answer-position-only regularization should not be the next route-1 mainline.

## E18 Route-1 V2.11 Answer-Margin Routing

### Goal

Test whether a benchmark-aligned answer-margin auxiliary objective can improve
route-1 transfer by directly supervising ABCD answer positions with a hinge
margin over answer-option logits.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `answer_margin_routing`
- Objective: hinge margin `0.5`
- Auxiliary weight: `0.03`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 45.12 | 53.04 | 47.40 | 48.52 | -2.30 |
| `v2.11_answer_margin_hinge050_w003_entropy050` | 45.86 | 54.61 | 51.00 | 50.49 | -0.33 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 0.3831 | 0.1714 | 0.1673 |
| `v2.11_answer_margin_hinge050_w003_entropy050` | 0.3886 | 0.1801 | 0.1800 |

### Diagnostics

| Dataset | Regression | Improvement | Net flip | Pred changed | v2.11 Pred D |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI2-ARC | 58 | 56 | -2 | 14.96 | 36.3 |
| MMLU-Redux | 429 | 361 | -68 | 22.80 | 35.6 |
| OpenBookQA | 25 | 27 | +2 | 16.20 | 37.6 |

### Implementation

- `rosetta/train/answer_margin.py`: added causal-shifted answer-option margin
  loss over supervised ABCD answer positions.
- `script/train/SFT_train.py`: applies the answer-margin loss during training,
  logs answer-margin metrics, and keeps evaluation loss free of this auxiliary
  term.
- `test/test_answer_margin_routing.py`: adds unit coverage for the helper.

### Artifacts

- Summary: `local/final_results/route1_alignment_v211/small_loop_summary/route1_v211_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v211/small_loop_summary/route1_v211_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v211/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v211/`
- Checkpoint: `local/checkpoints/route1_alignment_v211/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v211_answer_margin_hinge050_w003_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v211/qwen3_tinyllama_answer_margin_hinge050_w003_entropy050_small2048/`
- Flip diagnostics: `local/final_results/route1_alignment_v211/diagnostics/v22_vs_v211_flip_diagnostics/diagnostic_summary.md`
- W&B offline smoke run: `wandb/offline-run-20260430_183210-s4opxi8n`
- W&B offline full run: `wandb/offline-run-20260430_183258-clrr54xb`

### Validation

- `python -m black rosetta/train/answer_margin.py script/train/SFT_train.py test/test_answer_margin_routing.py`
- `python -m py_compile rosetta/train/answer_margin.py script/train/SFT_train.py test/test_answer_margin_routing.py test/test_answer_prior_regularization.py test/test_aligner_span_overlap.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_answer_margin_routing.py test/test_answer_prior_regularization.py test/test_aligner_span_overlap.py -q`
- Result: `34 passed`

### Notes

- v2.11 recovers much of the v2.10 loss and slightly improves OpenBookQA over
  v2.2, but it still underperforms v2.2 on benchmark mean by 0.33.
- Blank and invalid prediction counts are zero, so this is not an output-format
  collapse.
- The main failure mode is a stronger aggregate answer-prior shift toward
  option `D`, especially on MMLU-Redux. The answer-margin idea is a partial
  positive branch, but this exact hinge-only objective should not replace v2.2
  as the current route-1 mainline.

## E19 Route-1 V2.11b Softer Answer-Margin

### Goal

Test whether the v2.11 answer-margin branch can be made less biased by replacing
hinge-only supervision with a softer CE+hinge objective over supervised ABCD
answer positions.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`, `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `answer_margin_routing`
- Objective: `ce_hinge`
- Margin: `0.5`
- CE weight: `1.0`
- Hinge weight: `0.5`
- Auxiliary weight: `0.02`
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 45.12 | 53.04 | 47.40 | 48.52 | -2.30 |
| `v2.11_answer_margin_hinge050_w003_entropy050` | 45.86 | 54.61 | 51.00 | 50.49 | -0.33 |
| `v2.11b_answer_margin_cehinge050_h050_w002_entropy050` | 46.00 | 53.91 | 49.60 | 49.84 | -0.98 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.10_answer_prior_smooth070_w003_entropy050` | 0.3831 | 0.1714 | 0.1673 |
| `v2.11_answer_margin_hinge050_w003_entropy050` | 0.3886 | 0.1801 | 0.1800 |
| `v2.11b_answer_margin_cehinge050_h050_w002_entropy050` | 0.4048 | 0.1782 | 0.1766 |

### Diagnostics

| Dataset | Regression | Improvement | Net flip | Pred changed | v2.11b Pred C | v2.11b Pred D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AI2-ARC | 47 | 37 | -10 | 10.09 | 29.1 | 27.0 |
| MMLU-Redux | 331 | 271 | -60 | 16.96 | 29.8 | 26.8 |
| OpenBookQA | 20 | 15 | -5 | 10.00 | 24.4 | 31.8 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v211b/small_loop_summary/route1_v211b_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v211b/small_loop_summary/route1_v211b_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v211b/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v211b/`
- Checkpoint: `local/checkpoints/route1_alignment_v211b/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v211b_answer_margin_cehinge050_h050_w002_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v211b/qwen3_tinyllama_answer_margin_cehinge050_h050_w002_entropy050_small2048/`
- Flip diagnostics: `local/final_results/route1_alignment_v211b/diagnostics/v22_vs_v211b_flip_diagnostics/diagnostic_summary.md`
- W&B offline smoke run: `wandb/offline-run-20260501_115633-h6d9qf0l`
- W&B offline full run: `wandb/offline-run-20260501_115723-ut9b0zx5`

### Validation

- `jq empty local/tmp/train_recipes/route1_alignment_v211b/*.json`
- YAML config load check for all v2.11b eval configs
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest test/test_answer_margin_routing.py -q`
- Result: `3 passed`
- Smoke32 training completed
- Full small2048 training completed
- MMLU-Redux / AI2-ARC / OpenBookQA evaluation completed
- v2.2 vs v2.11b flip diagnostics completed

### Notes

- v2.11b reduces the v2.11 hinge-only D bias, especially on MMLU-Redux and
  AI2-ARC, but it does not improve benchmark accuracy.
- Blank and invalid prediction counts remain zero, so this is not an
  output-format collapse.
- Regressions outnumber improvements by 75 across the compared benchmarks.
- The answer-margin branch should remain a diagnostic branch for now; v2.2 is
  still the current route-1 mainline on the Qwen3/TinyLlama stress pair.

## E20 Route-1 V2.12a Alignment-Quality-Aware Token Gate

### Goal

Test whether the current v2.2 token-level confidence gate can generalize better
when the gate also receives local alignment-quality features from the soft span
mapping.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Base alignment: `soft_span_overlap_v2`, `soft_alignment_score_mode=uniform`,
  `top_k=4`
- Confidence seed: entropy alpha `0.5`, floor `0.5`, fallback `0.25`
- Confidence gate: `token_mlp`
- New mechanism: `alignment_confidence_feature_mode=quality`
- Quality features: source confidence, normalized source-weight entropy, top-1
  normalized source weight, active source candidate fraction
- Train data: MMLU `auxiliary_train`, `num_samples=2048`
- Evaluation: MMLU-Redux, AI2-ARC, OpenBookQA
- Seed: `42`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `v2.12a_alignment_quality_token_mlp_entropy050` | 43.36 | 53.74 | 49.20 | 48.77 | -2.05 |

### Training

| Method | Final train loss | Mid eval loss | Final eval loss |
| --- | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.3731 | 0.1714 | 0.1694 |
| `v2.12a_alignment_quality_token_mlp_entropy050` | 0.3509 | 0.0967 | 0.1007 |

### Projector Diagnostics

| Metric | Value |
| --- | ---: |
| `projector/quality_confidence_mean` | 0.90781 |
| `projector/quality_entropy_mean` | 0.18421 |
| `projector/quality_top1_mean` | 0.90789 |
| `projector/quality_active_fraction_mean` | 0.29605 |
| `projector/key_delta_abs_mean` | 0.54704 |
| `projector/value_delta_abs_mean` | 0.13093 |
| `projector/key_delta_abs_max` | 1.17382 |
| `projector/value_delta_abs_max` | 0.78145 |

### Artifacts

- Summary: `local/final_results/route1_alignment_v212/small_loop_summary/route1_v212_small_loop_summary.md`
- Score CSV: `local/final_results/route1_alignment_v212/small_loop_summary/route1_v212_small_loop_scores.csv`
- Train configs: `local/tmp/train_recipes/route1_alignment_v212/`
- Eval configs: `local/tmp/eval_configs/route1_alignment_v212/`
- Checkpoint: `local/checkpoints/route1_alignment_v212/qwen3_0.6b+tinyllama1.1b_soft_span_overlap_v212_alignment_quality_token_mlp_entropy050_small2048/final`
- Eval summaries: `local/final_results/route1_alignment_v212/qwen3_tinyllama_alignment_quality_token_mlp_entropy050_small2048/`
- W&B offline smoke run: `wandb/offline-run-20260501_145131-iav1kjbk`
- W&B offline full run: `wandb/offline-run-20260501_145237-4x3hb95w`

### Validation

- `python -m black rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- `python -m pytest test/test_aligner_span_overlap.py -q`
- Result: `30 passed`
- Smoke32 training completed
- Full small2048 training completed
- MMLU-Redux / AI2-ARC / OpenBookQA evaluation completed

### Notes

- v2.12a greatly improves the small-loop train/eval losses but hurts all three
  downstream benchmarks.
- The branch is likely overfitting the local validation objective or learning a
  dataset-specific alignment-quality shortcut that does not transfer to
  downstream benchmark prompting.
- Treat v2.12a as a negative ablation. The current route-1 mainline remains
  `v2.2_token_mlp_entropy050`.

## Current Research Interpretation

- Route-1 should remain the main direction for now.
- The current strongest claim is: cross-tokenizer C2C needs many-to-many, span-aware soft alignment when tokenizer mismatch is large.
- Qwen3/Llama-3.2 is useful as a mild mismatch control; Qwen3/TinyLlama is useful as a stress case.
- The best current route-1 candidate is `v2.2_token_mlp_entropy050` on the Qwen3/TinyLlama stress pair.
- `v2.3_token_mlp_delta_l2_0p01` and `v2.4_token_mlp_selective_delta_l2_0p01` are negative ablations, so delta L2 trust-region regularization should not be the main next method.
- v2.2 confidence diagnostics show a stable asymmetric layer pattern: key
  confidence corrections are strongest in early layers, while value confidence
  corrections are strongest in late layers.
- Static layer-aware scaling in v2.5 did not beat v2.2, so the next useful
  layer-aware experiment should preserve local confidence freedom through
  learnable or weakly-constrained per-layer scales rather than fixed schedules.
- v2.6b learned scalar layer scaling also did not beat v2.2. Simple scalar
  layer gates should be treated as a negative branch for now.
- v2.7 adaptive overlap reweighting modified token/span-level top-k weighting
  directly but still did not beat v2.2. Fixed overlap-power sharpening should be
  treated as a negative or boundary ablation.
- v2.8 learnable span MLP weight calibration substantially improved the
  small-loop eval loss but did not improve benchmark mean. Treat it as evidence
  of a validation/benchmark mismatch rather than as a new mainline result.
- v2.8 flip diagnostics show no broad format failure. The main failure mode is
  answer-prior shift toward option `A`, especially on MMLU-Redux.
- v2.8b constrained span calibration did not change downstream predictions
  relative to v2.8. Treat the span-weight calibrator branch as a negative
  ablation for now, not as the route-1 mainline.
- v2.9 learned residual scaling also did not beat v2.2. Simple scalar
  layer/KV-level fusion should be treated as a negative branch for now.
- v2.10 answer-prior regularization slightly improved the small-loop eval loss
  but hurt all downstream benchmarks and strengthened prediction bias toward
  option `D`. Treat answer-position-only regularization as a negative branch.
- v2.11 benchmark-aligned answer-margin routing is a partial positive branch:
  it is much better than v2.10 and improves OpenBookQA, but it is still 0.33
  points below v2.2 mean and shifts predictions too strongly toward option `D`.
- v2.11b CE+hinge answer-margin softens that D shift but drops to a 49.84 mean
  and shifts prediction mass toward option `C`; it is a negative/diagnostic
  branch, not a replacement for v2.2.
- v2.12a alignment-quality-aware token gating strongly improves small-loop
  eval loss but drops benchmark mean to 48.77. Treat alignment-quality features
  inside the token gate as a negative ablation unless paired with stronger
  generalization control.
- The next route-1 step should keep v2.2 as the current best stress-pair
  baseline and move away from simple scalar gates or top-k source-weight
  calibration. If continuing answer-aware work, pure answer-position auxiliary
  losses should be paired with explicit distribution control or treated as
  diagnostics rather than as the main route-1 mechanism.

## E20 Route-3 Learned Alignment

### Goal

Start a new Route-3 branch from paper-native C2C rather than Route-1 v2.2.
The method keeps C2C residual projection/fusion and makes cross-tokenizer
candidate aggregation learnable from KV features.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `top_k=8`, candidate window `1`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Smoke train data: MMLU `auxiliary_train`, `num_samples=32`
- Small-loop train data: MMLU `auxiliary_train`, `num_samples=2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA
- Primary record: `local/final_results/route3_learned_alignment/README.md`

### Status

- Code and recipes created.
- Unit tests passed.
- Smoke training/evaluation passed.
- Small-loop training/evaluation completed.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 |
| `route3_learned_alignment_kv_router_small2048` | 42.92 | 53.30 | 50.00 | 48.74 | -2.08 |

### Diagnostics

- Small-loop train loss: `0.3895`
- Small-loop eval loss: `0.1273`
- Learned router anchor mean remains high at about `0.772`, with entropy about
  `0.617`. The router is still close to the anchor-biased initialization rather
  than learning a strongly different source-token selection policy.
- MMLU-Redux is the main regression (`-4.15` vs v2.2). AI2-ARC (`-1.48`) and
  OpenBookQA (`-0.60`) are closer but still below v2.2.

### Interpretation

Route-3 v0 is a negative benchmark result. The learned candidate router is
technically viable and runs through smoke/small-loop training and evaluation,
but candidate selection alone does not recover the token/head-level trust
control that made Route-1 v2.2 strong. The next Route-3 variant should add
either a learned injection gate on top of the routed KV, or an auxiliary
contrastive/likelihood objective that makes non-anchor candidates useful enough
to overcome the conservative anchor initialization.

## E21 Route-3.1 Learned Injection Gate

### Goal

Continue Route-3 learned alignment by adding a learned token/head injection
gate after the learned KV router. This tests whether Route-3 v0 mainly failed
because it lacked Route-1 v2.2 style trust control over transferred KV.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `top_k=8`, candidate window `1`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- New module: `learned_alignment_injection_gate_mode=token_mlp`
- Gate init/max delta: `learned_alignment_injection_init_logit=1.0`,
  `learned_alignment_injection_max_delta=2.0`
- Smoke train data: MMLU `auxiliary_train`, `num_samples=32`
- Small-loop train data: MMLU `auxiliary_train`, `num_samples=2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA
- Primary record: `local/final_results/route3_learned_alignment/README.md`

### Status

- Code and recipes created.
- Unit tests passed.
- Smoke training/evaluation passed.
- Small-loop training/evaluation completed.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp | Delta vs Route-3 v0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +2.08 |
| `route3_learned_alignment_kv_router_small2048` | 42.92 | 53.30 | 50.00 | 48.74 | -2.08 | +0.00 |
| `route3_learned_alignment_kv_router_injection_gate_small2048` | 42.84 | 49.65 | 44.80 | 45.77 | -5.05 | -2.97 |

### Diagnostics

- Smoke train loss: `4.2657`
- Smoke eval loss: `2.5491`
- Smoke MMLU-Redux abstract_algebra accuracy: `26.32%`
- Small-loop train loss: `0.3741`
- Small-loop eval loss: `0.1098`
- Router anchor mean remains high at about `0.77285`, with entropy about
  `0.61626` and valid rate about `0.39906`.
- Key injection mean/std/delta abs mean: `0.68819` / `0.10729` / `0.53452`
- Value injection mean/std/delta abs mean: `0.74360` / `0.02461` / `0.12346`

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_injection_gate_smoke32.json`
  and
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_injection_gate_small2048.json`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_injection_gate_smoke_mmlu-redux.yaml`
  and
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_injection_gate_small2048_*.yaml`
- Checkpoints:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_injection_gate_smoke32/final`
  and
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_injection_gate_small2048/final`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_injection_gate_small2048/`

### Interpretation

Route-3.1 is a negative benchmark result. The learned injection gate is
trainable and active, and it improves the small-loop eval loss from Route-3 v0
(`0.1273` to `0.1098`). However, benchmark transfer gets worse, especially on
AI2-ARC and OpenBookQA. This means the gate is learning local calibration on the
MMLU auxiliary slice rather than a robust cross-tokenizer alignment rule.

The key issue is that the learned candidate router is still anchor-biased.
Anchor probability stays near `0.773`, so the post-router gate can only scale an
already conservative aggregate. It cannot fix wrong candidate selection, and it
does not add a direct learning signal for when non-anchor candidates should be
used. Route-3 should therefore shift from post-hoc injection calibration to
explicit candidate-level learning, such as contrastive candidate routing,
teacher-forced likelihood distillation, or router confidence regularization
based on candidate entropy and margin.

## E22 Route-3.2 Candidate-Level Span CE Mix

### Goal

Continue Route-3 by adding an explicit candidate-level auxiliary objective for
the learned KV router. This tests whether the router needs direct supervision
over top-k source-token candidates before adding more injection-strength
calibration.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `top_k=8`, candidate window `1`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary loss: `learned_alignment_aux_loss_mode=span_ce`
- Auxiliary loss weight: `learned_alignment_aux_loss_weight=0.05`
- Auxiliary apply/target: `ambiguous` / `source_weights`
- Uniform target mix: `learned_alignment_aux_uniform_mix=0.5`
- Smoke train data: MMLU `auxiliary_train`, `num_samples=32`
- Small-loop train data: MMLU `auxiliary_train`, `num_samples=2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA
- Primary record: `local/final_results/route3_learned_alignment/README.md`

### Status

- Code and recipes created.
- Unit tests passed.
- Smoke training/evaluation passed.
- Small-loop training/evaluation completed.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp | Delta vs Route-3 v0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +2.08 |
| `route3_learned_alignment_kv_router_small2048` | 42.92 | 53.30 | 50.00 | 48.74 | -2.08 | +0.00 |
| `route3_learned_alignment_kv_router_injection_gate_small2048` | 42.84 | 49.65 | 44.80 | 45.77 | -5.05 | -2.97 |
| `route3_learned_alignment_kv_router_span_ce_mix050_small2048` | 44.28 | 52.09 | 45.60 | 47.32 | -3.49 | -1.42 |

### Diagnostics

- Syntax checks passed for `rosetta/model/projector.py` and
  `script/train/SFT_train.py`.
- JSON parsing passed for the Route-3.2 smoke32 and small2048 recipes.
- Unit test command:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
- Unit test result: `35 passed`
- Black check passed for `rosetta/model/projector.py`,
  `script/train/SFT_train.py`, and `test/test_aligner_span_overlap.py`.
- Smoke train loss/eval loss: `4.5649` / `1.9324`
- Smoke auxiliary CE/loss: `0.93688` / `0.04684`
- Smoke MMLU-Redux abstract_algebra accuracy: `5.26%`
- Small-loop train loss/final eval loss: `0.4343` / `0.1279`
- Small-loop auxiliary CE/loss: `0.94138` / `0.04707`
- Auxiliary target anchor/top1/entropy: `0.65923` / `0.65923` / `0.78674`
- Router key/value anchor mean remains high: `0.77199` / `0.77198`
- Router key/value entropy: `0.61686` / `0.61688`
- Valid rate: `0.40054`

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_mix050_smoke32.json`
  and
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_mix050_small2048.json`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_span_ce_mix050_smoke_mmlu-redux.yaml`
  and
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_span_ce_mix050_small2048_*.yaml`
- Checkpoints:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_span_ce_mix050_smoke32/final`
  and
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_span_ce_mix050_small2048/final`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_span_ce_mix050_small2048/`

### Interpretation

Route-3.2 is a partial positive diagnostic but still a negative benchmark
result. Candidate-level span CE improves MMLU-Redux over Route-3 v0 (`42.92` to
`44.28`), so explicit candidate supervision has some useful signal. However,
AI2-ARC drops from `53.30` to `52.09`, OpenBookQA drops from `50.00` to `45.60`,
and the mean drops from `48.74` to `47.32`. It remains `3.49` points below
Route-1 v2.2.

The main reason is visible in diagnostics: the mixed auxiliary target has an
anchor probability near `0.659`, but the learned router stays near `0.772`.
The auxiliary objective is active, yet too weak to overcome anchor-biased
initialization and the downstream LM objective. The result is enough to move
MMLU a little, but not enough to learn a transferable cross-tokenizer candidate
policy.

Next Route-3 should make candidate learning stronger and measurable before
adding another gate:

- Raise or schedule the span CE weight and require router anchor probability to
  move below `0.70` before running full benchmark evaluation.
- Reduce or anneal `learned_alignment_anchor_logit` so early optimization is not
  pinned to the anchor candidate.
- Replace weak span CE with teacher-forced candidate likelihood distillation or
  contrastive candidate scoring.
- Use candidate dropout or entropy/margin regularization for controlled
  non-anchor exploration, then re-test injection gating after candidate routing
  actually changes.

## E23 Route-3.3 Strong Span CE and Lower Anchor Prior

### Goal

Continue Route-3.2 by making candidate learning strong enough to visibly move
the learned router. This tests whether the previous negative result was caused
by weak auxiliary pressure and a too-large anchor initialization.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `top_k=8`, candidate window `1`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary loss: `learned_alignment_aux_loss_mode=span_ce`
- Auxiliary loss weight: `learned_alignment_aux_loss_weight=0.2`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Auxiliary apply/target: `ambiguous` / `source_weights`
- Uniform target mix: `learned_alignment_aux_uniform_mix=0.5`
- Smoke train data: MMLU `auxiliary_train`, `num_samples=32`
- Small-loop train data: MMLU `auxiliary_train`, `num_samples=2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA
- Primary record: `local/final_results/route3_learned_alignment/README.md`

### Status

- Recipes and eval configs created.
- Unit tests passed.
- Smoke training/evaluation passed.
- Small-loop training/evaluation completed.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp | Delta vs Route-3 v0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +2.08 |
| `route3_learned_alignment_kv_router_small2048` | 42.92 | 53.30 | 50.00 | 48.74 | -2.08 | +0.00 |
| `route3_learned_alignment_kv_router_span_ce_mix050_small2048` | 44.28 | 52.09 | 45.60 | 47.32 | -3.49 | -1.42 |
| `route3_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048` | 44.07 | 52.17 | 47.60 | 47.95 | -2.87 | -0.79 |

### Diagnostics

- JSON parsing passed for the Route-3.3 train recipes and eval YAML files.
- Unit test command:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
- Unit test result: `35 passed`
- Smoke train command:
  `env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m torch.distributed.run --nproc_per_node=4 script/train/SFT_train.py --config local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_smoke32.json`
- Smoke train loss/eval loss: `4.6792` / `2.5909`
- Smoke auxiliary CE/loss: `0.92402` / `0.18480`
- Smoke router key/value anchor mean: `0.55951` / `0.55951`
- Smoke router key/value entropy: `0.89217` / `0.89217`
- Smoke MMLU-Redux abstract_algebra accuracy: `15.79%`
- Small-loop train command:
  `env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m torch.distributed.run --nproc_per_node=4 script/train/SFT_train.py --config local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048.json`
- Small-loop train loss/final eval loss: `0.5717` / `0.1210`
- Small-loop auxiliary CE/loss: `0.93008` / `0.18602`
- Auxiliary target anchor/top1/entropy: `0.65923` / `0.65923` / `0.78674`
- Router key/value anchor mean: `0.55724` / `0.55714`
- Router key/value entropy: `0.89259` / `0.89268`
- Valid rate: `0.40054`

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_smoke32.json`
  and
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048.json`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_span_ce_w020_anchor1p0_smoke_mmlu-redux.yaml`
  and
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048_*.yaml`
- Checkpoints:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_smoke32/final`
  and
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048/final`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260505_125001_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260505_125504_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260505_125818_summary.json`

### Interpretation

Route-3.3 is a diagnostic improvement but still a negative benchmark result.
It improves the Route-3.2 mean from `47.32` to `47.95`, mostly by recovering
OpenBookQA from `45.60` to `47.60`. MMLU-Redux is roughly flat/slightly lower
(`44.28` to `44.07`), AI2-ARC is roughly flat (`52.09` to `52.17`), and the
mean remains below both Route-3 v0 (`48.74`) and Route-1 v2.2 (`50.82`).

The main positive signal is that the router is no longer anchor-locked. Anchor
probability drops from about `0.772` in Route-3.2 to about `0.557` in Route-3.3,
and entropy rises from about `0.617` to about `0.893`. This means the previous
diagnostic gate, "move router anchor below 0.70," was satisfied.

The new failure mode is target quality. The span/uniform auxiliary target can
force non-anchor exploration, but it does not know which source candidate
actually improves target-token likelihood or answer accuracy. Therefore Route-3
should not keep increasing span CE weight blindly. The next credible direction
is candidate quality supervision: teacher-forced likelihood distillation,
contrastive candidate ranking with hard negatives, or reintroducing an
injection gate only after routing is trained from a likelihood-aware signal.

## E24 Route-3.4 Gradient CE Likelihood-Aware Candidate Target

### Goal

Replace Route-3.3's span/uniform candidate supervision with a downstream-loss
aware candidate target. This tests the paper-level hypothesis that cross-
tokenizer cache communication needs learned candidate quality, not just
character-span overlap.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `top_k=8`, candidate window `1`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary loss: `learned_alignment_aux_loss_mode=grad_ce`
- Auxiliary loss weight: `learned_alignment_aux_loss_weight=0.05`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Auxiliary target: first-order candidate score from `-d(task_loss)/d(router_prob)`
- Auxiliary normalization: enabled, score clip `3.0`, span mix `0.1`
- Smoke train data: MMLU `auxiliary_train`, `num_samples=32`
- Small-loop train data: MMLU `auxiliary_train`, `num_samples=2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA
- Primary record: `local/final_results/route3_learned_alignment/README.md`

### Status

- Code implemented in `rosetta/model/projector.py` and `script/train/SFT_train.py`.
- Recipes and eval configs created under `local/tmp/.../route3_learned_alignment/`.
- Unit tests passed.
- Smoke training/evaluation passed.
- Small-loop training/evaluation completed.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp | Delta vs Route-3 v0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +2.08 |
| `route3_learned_alignment_kv_router_small2048` | 42.92 | 53.30 | 50.00 | 48.74 | -2.08 | +0.00 |
| `route3_learned_alignment_kv_router_span_ce_mix050_small2048` | 44.28 | 52.09 | 45.60 | 47.32 | -3.49 | -1.42 |
| `route3_learned_alignment_kv_router_span_ce_w020_anchor1p0_small2048` | 44.07 | 52.17 | 47.60 | 47.95 | -2.87 | -0.79 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.80 |

### Diagnostics

- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py script/train/SFT_train.py`
- Unit test command:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
- Unit test result: `36 passed`
- Black check passed for `rosetta/model/projector.py`,
  `script/train/SFT_train.py`, and `test/test_aligner_span_overlap.py`.
- Smoke train command:
  `env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m torch.distributed.run --nproc_per_node=4 script/train/SFT_train.py --config local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_smoke32.json`
- Smoke train loss/eval loss: `4.5536` / `2.5822`
- Smoke auxiliary CE/loss: `1.19733` / `0.05987`
- Smoke auxiliary score margin: `0.33928`
- Smoke auxiliary target anchor/top1/entropy: `0.38672` / `0.41056` / `0.97281`
- Smoke router key/value anchor mean: `0.55951` / `0.55951`
- Smoke MMLU-Redux abstract_algebra accuracy: `15.79%`
- Small-loop train command:
  `env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m torch.distributed.run --nproc_per_node=4 script/train/SFT_train.py --config local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048.json`
- Small-loop train loss/final eval loss: `0.4173` / `0.1436`
- Small-loop auxiliary CE/loss: `1.19985` / `0.05999`
- Small-loop auxiliary score margin: `0.08411`
- Auxiliary target anchor/top1/entropy: `0.38681` / `0.39062` / `0.98802`
- Router key/value anchor mean: `0.55809` / `0.55811`
- Router key/value entropy: `0.89259` / `0.89257`
- Valid rate: `0.39856`

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_smoke32.json`
  and
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048.json`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_smoke_mmlu-redux.yaml`
  and
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048_*.yaml`
- Checkpoints:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_smoke32/final`
  and
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048/final`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260505_150441_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260505_174549_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260505_174820_summary.json`

### Interpretation

Route-3.4 is the strongest Route-3 result so far, but still does not beat the
Route-1 v2.2 benchmark target. It improves the Route-3 v0 mean from `48.74` to
`49.54` and Route-3.3 from `47.95` to `49.54`, with gains on all three
benchmarks. This is evidence that the learned alignment direction is not dead:
candidate-quality supervision from downstream CE is more useful than pure
span/uniform supervision.

The reason it still misses v2.2 is the quality and sharpness of the training
signal. The gradient-derived target is much less anchor-biased than Route-3.3
(`0.387` target anchor vs `0.659`), but it is too diffuse after training:
target entropy is about `0.988`, target top1 is only about `0.391`, and score
margin drops to `0.084`. The router stays near `0.558` anchor probability, so
it is no longer locked but also not learning a decisive candidate ranking. The
method also makes training substantially slower because it needs
`autograd.grad` through the task loss.

Next Route-3 should keep likelihood-aware candidate learning, but sharpen and
cheapen it before larger training:

- Use lower target temperature or top-r sparse projection for `grad_ce`.
- Raise/schedule `grad_ce` weight only after the target margin is made larger.
- Score only selected layers or selected target positions to reduce cost.
- Try offline teacher-forced candidate replay on a small calibration set, then
  use online `grad_ce` as a regularizer instead of the only supervision.

## E25 Route-3.5/3.6 Gradient Target Sharpening

### Goal

Test the most direct follow-up to Route-3.4: if online likelihood-aware
`grad_ce` helps but remains too diffuse, can simple target sharpening produce a
stronger learned alignment policy?

### Setup

- Base: Route-3.4 `grad_ce_w005_span010_anchor1p0`
- New code path: `rosetta/model/projector.py`
- New controls:
  - `learned_alignment_aux_top_r`
  - `learned_alignment_aux_score_margin_threshold`
- Train data: MMLU `auxiliary_train`, smoke `32`, small-loop `2048` only for
  the milder top-r variant
- Eval data: MMLU-Redux `abstract_algebra` smoke, with full benchmarks skipped
  for variants that failed smoke/small-loop diagnostics

### Variants

| Variant | Main Change | Status |
| --- | --- | --- |
| `grad_ce_topr2_t050_w010_span005_anchor1p0` | aggressive top-r `2`, temp `0.5`, loss weight `0.1` | smoke only |
| `grad_ce_topr4_t075_w005_span005_anchor1p0` | milder top-r `4`, temp `0.75`, loss weight `0.05` | smoke + small-loop |
| `grad_ce_margin010_w005_span010_anchor1p0` | score-margin threshold `0.1` | smoke only |

### Diagnostics

- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py script/train/SFT_train.py`
- Unit test command:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
- Unit test result: `38 passed`
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
- Black result: passed

### Results

| Method | Train Loss | Eval Loss | Aux CE/Loss | Score Margin | Target Anchor/Top1/Entropy | Smoke Eval | Decision |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| Route-3.4 smoke | 4.5536 | 2.5822 | 1.19733 / 0.05987 | 0.33928 | 0.38672 / 0.41056 / 0.97281 | 15.79% abstract_algebra | baseline for this round |
| Route-3.5 `topr2_t050_w010_span005` smoke | 4.6171 | 2.5977 | 1.22496 / 0.12250 | 0.33928 | 0.35909 / 0.51768 / 0.91285 | 26.60% broad/non-comparable smoke | stop |
| Route-3.5b `topr4_t075_w005_span005` smoke | 4.5554 | 2.5741 | 1.23177 / 0.06159 | 0.33928 | 0.35228 / 0.40300 / 0.96761 | 10.53% abstract_algebra | small-loop only |
| Route-3.6 `margin010_w005_span010` smoke | 4.5541 | 2.6052 | 1.20692 / 0.06035 | 0.44589 | 0.38171 / 0.43529 / 0.95286 | 10.53% abstract_algebra | stop |

Route-3.5b small-loop:

- Train loss/final eval loss: `0.4200` / `0.1500`
- Mid eval loss at step 50: `0.1501`
- Auxiliary CE/loss: `1.23313` / `0.06166`
- Auxiliary selected rate: `1.00000`
- Auxiliary score margin: `0.02958`
- Auxiliary target anchor/top1/entropy: `0.35352` / `0.35952` / `0.99422`
- Router key/value anchor mean: `0.55809` / `0.55811`
- Valid rate: `0.39856`

No full benchmark was run for Route-3.5b or Route-3.6. Route-3.5b's final eval
loss is worse than Route-3.4 (`0.1500` vs `0.1436`), and its target signal
degrades after small-loop training: score margin falls to `0.02958`, target
top1 drops to `0.35952`, and entropy rises to `0.99422`. Route-3.6 fails at
smoke scale with worse eval loss and smoke accuracy despite a cleaner selected
subset.

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_topr2_t050_w010_span005_anchor1p0_*.json`
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_topr4_t075_w005_span005_anchor1p0_*.json`
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_margin010_w005_span010_anchor1p0_*.json`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_topr2_t050_w010_span005_anchor1p0_*.yaml`
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_topr4_t075_w005_span005_anchor1p0_*.yaml`
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_margin010_w005_span010_anchor1p0_*.yaml`
- Smoke summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_topr2_t050_w010_span005_anchor1p0_smoke32/mmlu-redux_abstract_algebra/Rosetta_mmlu-redux_generate_20260506_113306_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_topr4_t075_w005_span005_anchor1p0_smoke32/mmlu-redux_abstract_algebra/Rosetta_mmlu-redux_generate_20260506_113845_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_margin010_w005_span010_anchor1p0_smoke32/mmlu-redux_abstract_algebra/Rosetta_mmlu-redux_generate_20260506_114459_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Interpretation

E25 is a negative result, but it is useful because it rules out a cheap fix.
Simple online sharpening does not make Route-3 exceed Route-3.4 or v2.2.

Top-r sparsification initially sharpens the target, but it also amplifies noisy
first-order gradients. In the aggressive top-r `2` run, target top1 rises to
`0.51768` and entropy falls to `0.91285`, but train/eval loss gets worse and
generation quality is unstable. In the milder top-r `4` run, smoke eval loss is
slightly lower than Route-3.4, but after small-loop training the diagnostic
collapses: entropy becomes `0.99422` and score margin only `0.02958`.

Margin filtering produces a cleaner selected set, not better candidate labels.
The selected rate drops to `0.41346` and margin rises to `0.44589`, but smoke
eval loss (`2.6052`) and abstract_algebra accuracy (`10.53%`) are worse than
Route-3.4. The likely issue is coverage: many ambiguous token rows are simply
removed, so the router gets less cross-tokenizer learning signal exactly where
alignment is hard.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next credible direction: stop spending more cycles on post-processing the
online `autograd.grad` target. Build explicit candidate likelihood labels with
offline teacher-forced candidate replay on a small calibration set, then train
the router with a true candidate ranking objective. Online `grad_ce` can remain
as regularization, but should not be the only supervision source.

## E26 Route-3.7 Teacher-Forced Candidate Replay

### Goal

Implement the next credible Route-3 learned-alignment experiment after E25:
replace noisy online first-order `grad_ce` labels with explicit candidate
ranking labels. For each batch, force candidate ranks `0..3`, measure the
teacher-forced task loss for each forced KV transfer, convert losses to a soft
target, and train the router with replay CE.

### Setup

- Base: Route-3 learned alignment from paper-native C2C, not v2.2 stacking
- Alignment: `learned_span_alignment`, `soft_alignment_top_k=8`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- New auxiliary mode: `learned_alignment_aux_loss_mode=replay_ce`
- Replay scoring:
  - `candidate_replay_alignment.enabled=true`
  - `num_candidates=4`
  - `target_temperature=0.5`
  - `score_normalize=true`
  - `score_clip=3.0`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Auxiliary weight: `learned_alignment_aux_loss_weight=0.1`
- Train data: MMLU `auxiliary_train`, smoke `16/32`, small-loop `2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

### Implementation

- `rosetta/model/projector.py`
  - Added forced-rank learned-alignment mode for candidate replay scoring.
  - Added batch-level replay target injection.
  - Added `replay_ce` auxiliary loss over key/value router candidate weights.
  - Added deterministic soft-gate mode for replay scoring.
- `script/train/SFT_train.py`
  - Added candidate replay scoring before the main train forward.
  - Added replay-target normalization and replay diagnostics logging.
- `test/test_aligner_span_overlap.py`
  - Added tests for forced-rank candidate selection and replay CE target usage.

Important implementation finding: the first replay smoke exposed a real bug in
the scoring setup. In eval mode the projector gate used hard `(logit > 0)`
thresholding, so the initial closed gate made all forced-rank logits identical.
After adding deterministic soft-gate scoring, forced candidates produced
different task losses and replay labels became meaningful.

### Diagnostics

- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py script/train/SFT_train.py`
- Unit test:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
  - Result: `40 passed`
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
  - Result: passed

Forced-rank diagnostic after the soft-gate fix:

| Forced Rank | Task Loss |
| ---: | ---: |
| 0 | 4.4828658104 |
| 1 | 4.4419541359 |
| 2 | 4.3232240677 |
| 3 | 4.4995603561 |

Smoke32:

- Train loss/final eval loss: `4.6303` / `2.5982`
- Candidate replay best rank: `1.16667`
- Candidate replay loss margin: `0.09684`
- Rank losses: `4.33396` / `4.26648` / `4.26689` / `4.33011`
- Candidate target anchor/top1/entropy/margin:
  `0.19779` / `0.82249` / `0.36542` / `0.66943`
- Projector auxiliary CE/loss: `1.29399` / `0.12940`
- Projector target anchor/top1/entropy/margin:
  `0.29006` / `0.65702` / `0.69842` / `0.36696`
- Smoke MMLU-Redux `abstract_algebra`: `15.79%`

Small-loop 2048:

- Train loss/final eval loss: `0.5313` / `0.1375`
- Candidate replay best rank: `1.50000`
- Candidate replay loss margin: `0.00095`
- Rank losses: `0.11583` / `0.11575` / `0.11703` / `0.11506`
- Candidate target anchor/top1/entropy/margin:
  `0.38818` / `0.63565` / `0.52358` / `0.39674`
- Projector auxiliary CE/loss: `1.20044` / `0.12004`
- Projector target anchor/top1/entropy/margin:
  `0.40318` / `0.56557` / `0.65502` / `0.16266`
- Router key/value anchor mean: `0.55205` / `0.55214`
- Router key/value entropy: `0.89382` / `0.89373`
- Valid rate: `0.41369`

Small-loop eval loss improved over Route-3.4 (`0.1375` vs `0.1436`), so full
benchmark evaluation was run.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048` | 43.79 | 51.57 | 45.60 | 46.98 | -3.84 | -2.56 |

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_smoke16.json`
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_smoke32.json`
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048.json`
- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048/final`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048_*.yaml`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260506_143132_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260506_144045_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260506_144504_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Interpretation

E26 is a negative full-benchmark result. It answers the key question from E25:
explicit candidate replay labels are feasible and non-degenerate, but the
current replay objective does not improve C2C benchmark behavior.

Why it looked promising: replay CE made the candidate target much sharper than
online `grad_ce`. Smoke target top1 reached `0.82249`, entropy fell to
`0.36542`, and small-loop teacher-forced eval loss improved from Route-3.4's
`0.1436` to `0.1375`.

Why it failed on benchmark: the replay label optimizes teacher-forced
next-token likelihood under forced KV transfer, not answer correctness or
generation robustness. After small-loop training, candidate loss margin
collapsed to `0.00095`, so the rank labels became nearly ambiguous even though
the target remained artificially sharp. The model then underperformed on all
three generation benchmarks, especially MMLU-Redux and OpenBookQA.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next credible Route-3 idea: keep replay infrastructure, but change the
supervision target from full-token CE to answer-aware candidate labels. Score
candidates on answer option likelihood, choice-letter span likelihood, or
answer margin; combine this with a format-preservation loss. Replay labels
should calibrate candidate routing, not be the only alignment objective.

## E27 Route-3.8 Answer-Aware Candidate Replay

### Goal

Follow the E26 conclusion by changing replay labels from full-token
teacher-forced CE to answer-aware candidate scoring. The hypothesis was that
Route-3.7 failed because it optimized generic next-token likelihood instead of
answer behavior.

### Setup

- Base: paper-native Route-3 learned alignment, not v2.2 stacking
- Alignment: `learned_span_alignment`, `soft_alignment_top_k=8`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary mode: `learned_alignment_aux_loss_mode=replay_ce`
- Replay candidates: `num_candidates=4`
- Replay temperature: `target_temperature=0.75`
- Replay normalization/clipping: `score_normalize=true`, `score_clip=3.0`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Auxiliary weight: `learned_alignment_aux_loss_weight=0.05`
- Train data: MMLU `auxiliary_train`, smoke `32`, small-loop `2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

Variants:

- `answer_token_replay_ce_c4_t075_w005_anchor1p0`
  - `candidate_replay_alignment.score_mode=answer_token_ce`
  - Scores only option-token shifted labels.
- `answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0`
  - `candidate_replay_alignment.score_mode=answer_suffix_ce`
  - `candidate_replay_alignment.answer_suffix_tokens=3`
  - Scores option token plus two following supervised shifted labels.

Both variants used `fallback_score_mode=suffix_ce`,
`fallback_suffix_tokens=4`, and `min_score_positions=1`.

### Implementation

- `script/train/SFT_train.py`
  - Added answer-aware candidate replay score modes:
    `answer_token_ce`, `answer_suffix_ce`, and `suffix_ce`.
  - Added automatic option-token id resolution via the main tokenizer.
  - Added score-position and fallback diagnostics.
  - Kept answer-aware scoring limited to replay-target construction; normal
    training loss is unchanged.
- `test/test_aligner_span_overlap.py`
  - Added tests for option-token CE scoring and fallback-to-suffix scoring.

### Diagnostics

- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile script/train/SFT_train.py rosetta/model/projector.py`
- Unit test:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
  - Result: `42 passed`
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check script/train/SFT_train.py rosetta/model/projector.py test/test_aligner_span_overlap.py`
  - Result: passed

### Smoke Results

| Variant | Train Loss | Eval Loss | Score Positions | Fallback | Replay Target Anchor/Top1/Entropy/Margin | Smoke MMLU-Redux abstract_algebra |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `answer_token_replay_ce_c4_t075_w005_anchor1p0_smoke32` | 4.5658 | 2.5839 | 1 | 0 | 0.13607 / 0.59129 / 0.63949 / 0.29120 | 10.53 |
| `answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_smoke32` | 4.5685 | 2.6090 | 3 | 0 | 0.08332 / 0.65911 / 0.61432 / 0.40560 | 15.79 |

Answer-token replay was smoke-negative and stopped. Answer-suffix recovered the
Route-3.4 smoke accuracy and had no fallback, so it was scaled to the small
loop.

### Small-Loop 2048

`answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048`:

- Train loss/final eval loss: `0.4641` / `0.1367`
- Candidate replay best rank: `1.12500`
- Candidate replay loss margin: `0.01086`
- Rank losses: `0.29266` / `0.29080` / `0.29306` / `0.29446`
- Candidate target anchor/top1/entropy/margin:
  `0.36859` / `0.65562` / `0.59882` / `0.42446`
- Score positions/fallback: `3` / `0`
- Projector auxiliary CE/loss: `0.84400` / `0.04220`
- Projector target anchor/top1/entropy/margin:
  `0.75990` / `0.76093` / `0.52096` / `0.56766`
- Router key/value anchor mean: `0.55210` / `0.55214`
- Router key/value entropy: `0.89377` / `0.89373`
- Valid rate: `0.41369`

Small-loop eval loss improved over Route-3.4 (`0.1436`) and Route-3.7
(`0.1375`), so full benchmark evaluation was run.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048` | 43.79 | 51.57 | 45.60 | 46.98 | -3.84 | -2.56 |
| `route3_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048` | 41.96 | 48.87 | 45.00 | 45.28 | -5.54 | -4.26 |

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_answer_token_replay_ce_c4_t075_w005_anchor1p0_*.json`
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_*.json`
- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048/final`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048_*.yaml`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260506_164208_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260506_160743_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260506_160229_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Interpretation

E27 is a negative full-benchmark result. It gives a useful answer to the
question "does answer-aware replay fix Route-3.7?" The answer is no.

What worked: labels were valid and non-degenerate. Score coverage matched the
intended answer window (`1` position for answer-token, `3` positions for
answer-suffix), and fallback was never used. Answer-suffix also produced a
cleaner projector target after small-loop training: target top1 `0.76093`,
entropy `0.52096`, and margin `0.56766`.

What failed: the improved local replay target did not become better generation
behavior. Small-loop teacher-forced eval loss improved to `0.1367`, but full
benchmark accuracy dropped on all three tasks: MMLU-Redux `41.96`, AI2-ARC
`48.87`, and OpenBookQA `45.00`.

Likely reason: answer-aware replay is still a local teacher-forced objective.
It scores how a forced candidate affects likelihood around the final answer
tokens, but it does not directly train the model to preserve reasoning path,
answer-format robustness, or broad task correctness under generation. It also
replaces Route-3.4's softer `grad_ce` signal rather than combining with it.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next ideas:

- Use answer-margin candidate labels, not answer-token CE: score correct-option
  logprob minus best-wrong-option logprob under each forced candidate.
- Use pairwise/listwise ranking loss over forced candidates instead of replay CE.
- Mix answer-aware replay with the Route-3.4 `grad_ce` target rather than
  replacing it.
- Add explicit short-answer format preservation, because local answer CE alone
  does not guarantee generation behavior.

## Update Template

Use this template for the next experiment:

## E28 Route-3.9 Utility-Calibrated Margin-Rank Router

### Goal

Follow the Route-3.8 conclusion without doing another replay-CE variant.
Route-3.9 keeps the current best Route-3 signal, Route-3.4 `grad_ce`, and adds
answer-margin forced-candidate utility only as a weak pairwise/listwise ranking
calibration term.

The hypothesis was: answer-aware replay failed because it replaced the soft
`grad_ce` objective with local replay CE. A better version should combine
`grad_ce` with answer-margin ranking rather than training the router to copy a
sharp replay target.

### Setup

- Base: paper-native Route-3 learned alignment, not v2.2 stacking
- Alignment: `learned_span_alignment`, `soft_alignment_top_k=8`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary mode: `learned_alignment_aux_loss_mode=grad_ce_margin_rank`
- Base `grad_ce` auxiliary weight: `0.05`
- Span weak-prior mix: `0.10`
- Answer-margin rank weight: `0.02`
- Rank threshold: `0.10`
- Rank scope: `batch_mean`
- Forced candidates: `4`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Train data: MMLU `auxiliary_train`, smoke `32`, small-loop `2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

Main variant:

`route3_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048`

### Implementation

- `script/train/SFT_train.py`
  - Added answer-margin candidate scoring:
    `log p(correct option) - logsumexp(log p(wrong options))`.
  - Returned candidate utilities alongside replay targets.
  - Passed utilities to the projector for ranking calibration.
  - Fixed `grad_ce` to use the raw task loss, not the total loss after other
    projector auxiliary terms.
- `rosetta/model/projector.py`
  - Added `grad_ce_margin_rank`.
  - Added row-wise and `batch_mean` ranking scopes.
  - Added valid-utility masking for unscored padded candidates.
- `test/test_aligner_span_overlap.py`
  - Added tests for answer-margin scoring, rank-gradient behavior, and padded
    candidate masking.

Important bug fixed: replay scoring only covers `num_candidates=4`, while the
router has `top_k=8`. Missing utilities were initially padded with zero, which
can make unscored ranks look better than scored ranks when answer margins are
negative. Route-3.9 uses a valid-utility mask and ignores unscored padded ranks.

### Diagnostics

- Unit test:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
  - Result: `46 passed`
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check rosetta/model/projector.py test/test_aligner_span_overlap.py script/train/SFT_train.py`
  - Result: passed
- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py script/train/SFT_train.py`
  - Result: passed

### Smoke Results

| Variant | Status | Train Loss | Eval Loss | Rank Scope | Pair Count | Smoke MMLU-Redux abstract_algebra |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `grad_ce_marginrank_c4_wgrad005_wrank002_tau010_span010_anchor1p0_smoke32` | diagnostic, pre-fix row-wise | 4.5706 | 2.5919 | row | 56 | 10.53 |
| `grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_smoke32` | valid post-fix | 4.5708 | 2.5893 | batch_mean | 1 | 21.05 |

The row-wise version was stopped because answer-margin utility was too broad as
a per-row label. The post-fix `batch_mean` version exceeded the Route-3.4 smoke
level (`15.79%`), so it was scaled.

### Small-Loop 2048

`grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048`:

- Train loss/final eval loss: `0.4250` / `0.1382`
- Candidate replay fallback: `0`
- Candidate replay score positions: `1`
- Candidate replay best rank: `1.4`
- Candidate replay loss margin: `0.01797`
- Candidate target anchor/top1/entropy/margin:
  `0.05183` / `0.49258` / `0.76805` / `0.18823`
- Projector `grad_ce` auxiliary CE/loss: `1.19949` / `0.05997`
- Projector answer-margin score margin: `0.04420`
- Projector target anchor/top1/entropy:
  `0.38717` / `0.39093` / `0.98802`
- Router key/value anchor mean: `0.55809` / `0.55812`
- Router key/value entropy: `0.89259` / `0.89256`
- Valid rate: `0.39856`
- Final margin-rank pair count/loss/utility margin:
  `0` / `0` / `0.00045`

The final pair count collapse is the main diagnostic: after small-loop
training, answer-margin differences among forced candidates are below the rank
threshold, so the new calibration term becomes mostly inactive.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_replay_ce_c4_t050_w010_anchor1p0_small2048` | 43.79 | 51.57 | 45.60 | 46.98 | -3.84 | -2.56 |
| `route3_learned_alignment_kv_router_answer_suffix_replay_ce_c4_t075_w005_suf3_anchor1p0_small2048` | 41.96 | 48.87 | 45.00 | 45.28 | -5.54 | -4.26 |
| `route3_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048` | 45.26 | 51.74 | 42.80 | 46.60 | -4.22 | -2.94 |

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_*.json`
- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048/final`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048_*.yaml`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260507_131339_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260507_131840_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260507_132108_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Interpretation

E28 is a negative full-benchmark result. It improved the valid smoke subject
from the Route-3.4 level (`15.79%`) to `21.05%`, but did not transfer to the
full benchmark: MMLU-Redux `45.26`, AI2-ARC `51.74`, OpenBookQA `42.80`, mean
`46.60`.

Why it looked promising: unlike replay CE, the answer-margin signal was used as
a calibration term on top of Route-3.4 `grad_ce`, not as a replacement. The
padding-mask fix also removed a real source of false ranking pressure, and
batch-mean scope prevented the worst row-wise over-supervision.

Why it failed: the useful answer-margin rank signal is too sparse under the
current online four-candidate loop. At the end of the small loop, margin-rank
pair count is `0` and utility margin is `0.00045`, so the new term no longer
applies meaningful pressure. The model then behaves mostly like a softened
Route-3.4 variant with extra early calibration noise, which hurts OpenBookQA
and does not improve MMLU/AI2.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next credible direction should not be another local replay variant. The better
Route-3 path is confidence/coverage-conditioned learned transfer:

- Use candidate utility only when replay coverage and answer-margin separation
  are reliable.
- Add a learned no-transfer or reduced-transfer decision when all forced
  candidates have collapsed margins.
- Build an offline candidate-utility cache over broader data so the rank signal
  is not limited to a tiny online four-candidate loop.
- Keep Route-3.4 `grad_ce` as the default objective and treat utility labels as
  selective calibration, not direct supervision.

## E29 Route-3.10 Router-Quality Transfer Gate

### Goal

Route-3.10 tested the confidence/coverage-conditioned transfer idea without
adding another replay objective. It keeps the current best Route-3.4 `grad_ce`
signal and adds a deterministic transfer gate from the learned router's own
candidate distribution.

Hypothesis: if learned alignment is useful only on confident rows, reducing KV
injection on high-entropy or low-margin router rows should improve generation
benchmarks without abandoning the paper-native learned alignment path.

### Setup

- Base: paper-native Route-3 learned alignment, not v2.2 stacking
- Alignment: `learned_span_alignment`, `soft_alignment_top_k=8`
- Projector: `C2CProjector`, `learned_alignment_mode=kv_router`
- Auxiliary mode: `learned_alignment_aux_loss_mode=grad_ce`
- Base `grad_ce` auxiliary weight: `0.05`
- Span weak-prior mix: `0.10`
- Anchor prior: `learned_alignment_anchor_logit=1.0`
- Transfer gate: `router_quality`
- Gate floor / entropy threshold / margin threshold / temperature:
  `0.70` / `0.90` / `0.10` / `0.10`
- Minimum valid candidates for gating: `2`
- Train data: MMLU `auxiliary_train`, smoke `32`, small-loop `2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

Main variant:

`route3_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048`

### Implementation

- `rosetta/model/projector.py`
  - Added `learned_alignment_transfer_gate_mode=router_quality`.
  - Added gate controls for floor, entropy threshold, margin threshold,
    temperature, and minimum valid candidates.
  - Stored key/value transfer gates in `align_source_kv`.
  - Multiplied the gate into the residual KV fusion path.
- `script/train/SFT_train.py`
  - Logged key/value transfer-gate diagnostics.
- `test/test_aligner_span_overlap.py`
  - Added tests for uncertain-row suppression, confident-row preservation, and
    residual-branch multiplication.

The gate is derived from normalized router entropy and top-2 margin, then
detached from the router weights. This prevents the auxiliary diagnostic from
pushing the router into artificial low-entropy collapse.

### Diagnostics

- Unit test:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
  - Result: `49 passed`
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check rosetta/model/projector.py script/train/SFT_train.py test/test_aligner_span_overlap.py`
  - Result: passed
- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/projector.py script/train/SFT_train.py`
  - Result: passed

### Smoke Results

`grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_smoke32`:

- Train loss/final eval loss: `4.6658` / `2.2416`
- Transfer gate mean/std: `0.90688` / `0.00854`
- Router entropy/margin/top1: `0.89217` / `0.35368` / `0.55951`
- Valid rate/selected rate: `0.39678` / `1.0`
- Smoke MMLU-Redux `abstract_algebra`, limit 20: `15.79`

The smoke score matched Route-3.4 smoke (`15.79`) and did not show the
Route-3.9 smoke bump. Since Route-3.9's bump was not reliable, Route-3.10 was
still scaled once for a full-benchmark answer.

### Small-Loop 2048

`grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048`:

- Train loss/final eval loss: `0.4247` / `0.1481`
- Key/value transfer gate mean: `0.90665` / `0.90662`
- Key/value transfer gate std: `0.00885` / `0.00886`
- Key/value router entropy: `0.89258` / `0.89257`
- Key/value router margin: `0.35238` / `0.35259`
- Key/value router top1: `0.55809` / `0.55811`
- Valid rate/selected rate: `0.39856` / `1.0`

The critical diagnostic is low variance: the gate is active, but mostly acts
like a global residual shrink near `0.91`, not a selective confidence policy.

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_grad_ce_marginrankmean_c4_wgrad005_wrank002_tau010_span010_anchor1p0_small2048` | 45.26 | 51.74 | 42.80 | 46.60 | -4.22 | -2.94 |
| `route3_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048` | 41.42 | 47.30 | 42.00 | 43.58 | -7.24 | -5.96 |

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_*.json`
- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048/final`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048_*.yaml`
- Eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260508_211035_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260508_211705_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_grad_ce_qualitygate_f070_e090_m010_t010_span010_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260508_211940_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Interpretation

E29 is a clear negative result. It is below Route-3.4 on all three benchmarks
and far below Route-1 v2.2: MMLU-Redux `41.42`, AI2-ARC `47.30`, OpenBookQA
`42.00`, mean `43.58`.

Why it failed:

- The gate is not selective enough. Mean is around `0.907`, but std is below
  `0.009`, so the mechanism mostly scales every residual transfer similarly.
- The learned router remains anchor-dominated and high-entropy: top1 is around
  `0.558`, entropy around `0.893`, and valid rate around `0.399`. A confidence
  gate based on this distribution cannot create a better alignment policy by
  itself.
- v2.2's advantage comes from the full transfer policy: character-span
  candidate generation, top-k aggregation, entropy confidence, and token/head
  confidence gating. Route-3.10 only gates after the learned router has already
  produced weak candidate weights.
- The gate reduces some useful transfer while failing to sharply identify
  harmful rows.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next credible version should not be another scalar gate threshold sweep. The
better Route-3 path is a prior-preserving hybrid router:

- Use v2.2 span/top-k weights as the teacher prior.
- Parameterize learned alignment as `logits = log(v2.2_soft_weights) +
  learned_delta`, with a small delta penalty.
- Add a learned no-transfer or down-transfer head trained from broader offline
  utility labels, not from the online four-candidate loop.
- Move confidence toward token/head or token/layer/head gates, because the
  strongest v2.2 result is explicitly head-aware.

```markdown
## E? Short Name

### Goal

### Setup

- Receiver:
- Sharer:
- Strategies:
- Train data:
- Eval data:
- Seed:
- Config paths:

### Diagnostics

### Results

| Strategy | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |

### Artifacts

- Checkpoints:
- Eval summaries:
- Diagnostics:
- Logs:

### Notes
```

## E30 Route-3.11 Prior-Preserving Hybrid Router

### Goal

E30 implemented the prior-preserving Route-3 idea from the Route-3.10 notes.
Instead of replacing the symbolic alignment prior, the learned router starts
from soft span weights and learns only a bounded residual:

`router_logits = log(span_soft_weights) + bounded_learned_delta`

The hypothesis was that Route-3 could keep v2-style offset/span strength while
learning small corrections for cross-tokenizer mismatch.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Alignment: `learned_span_alignment`, `soft_alignment_top_k=4`
- Aligner prior mode: `soft_span`
- Projector prior mode: `span_log_prior`
- Delta max / delta L2 / prior CE / grad CE: `0.5` / `0.01` / `0.02` / `0.05`
- Entropy confidence alpha/floor: `0.5` / `0.5`
- Scaled variant: `hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050`
- Train data: MMLU `auxiliary_train`, smoke `32`, small-loop `2048`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

### Implementation

- `rosetta/model/projector.py`
  - Added span-log-prior routing and bounded learned-delta regularization.
  - Added prior CE diagnostics and prior-aware candidate softmax.
- `rosetta/model/aligner.py`
  - Added soft-span learned-alignment prior rows.
  - Re-enabled entropy confidence for learned-span alignment.
- `script/train/SFT_train.py`
  - Passed aligner prior mode from config and logged prior diagnostics.
- `script/evaluation/unified_evaluator.py`
  - Passed aligner prior mode during evaluation.
- `test/test_aligner_span_overlap.py`
  - Added projector prior-softmax, prior-regularization, soft-span, and
    entropy-confidence tests.

### Diagnostics

- Syntax check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m py_compile rosetta/model/aligner.py rosetta/model/projector.py script/train/SFT_train.py script/evaluation/unified_evaluator.py`
  - Result: passed
- Black check:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m black --check test/test_aligner_span_overlap.py rosetta/model/aligner.py rosetta/model/projector.py script/train/SFT_train.py script/evaluation/unified_evaluator.py`
  - Result: passed
- Unit test:
  `/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest test/test_aligner_span_overlap.py -q`
  - Result: `54 passed`

Smoke:

- Token-MLP gate variant:
  - Train/final eval loss: `4.7166` / `2.5182`
  - Smoke MMLU-Redux `abstract_algebra`, limit 20: `26.32`
  - Small-loop status: CUDA OOM at step 2, not scaled.
- Residual-scale variant:
  - Train/final eval loss: `4.9576` / `2.2630`
  - Prior selected rate / prior CE / prior top1: `0.16598` / `0.73369` / `0.48333`
  - Smoke MMLU-Redux `abstract_algebra`, limit 20: `26.32`

Small-loop:

- Train/final eval loss: `0.4955` / `0.1491`
- Key/value residual scale: `0.49986` / `0.50056`
- Key/value anchor: `0.90575` / `0.90576`
- Key/value entropy: `0.18325` / `0.18325`
- Key/value top1: `0.90584` / `0.90577`
- Prior CE / prior selected rate / prior top1: `0.72790` / `0.18325` / `0.48571`
- Valid rate: `0.29974`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050_small2048` | 40.70 | 47.83 | 40.40 | 42.97 | -7.85 | -6.57 |

### Artifacts

- Train recipes:
  `local/tmp/train_recipes/route3_learned_alignment/qwen3_0.6b_tinyllama1.1b_learned_alignment_kv_router_hybridprior_*`
- Eval configs:
  `local/tmp/eval_configs/route3_learned_alignment/route3_qwen3_tinyllama_learned_alignment_kv_router_hybridprior_*`
- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050_small2048/final`
- Summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260511_204036_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260511_204729_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_hybridprior_rscale050_nogate_top4_dmax050_l2p01_pce002_gradce005_entropy050_small2048/openbookqa/Rosetta_openbookqa_generate_20260511_205043_summary.json`
- Branch record:
  `local/final_results/route3_learned_alignment/README.md`

### Notes

E30 is a clear negative result. It correctly tested the prior-preserving
hybrid-router idea, but full benchmark performance is below both v2.2 and
Route-3.4.

Why it failed:

- The soft prior collapsed back to an anchor-heavy policy. Key/value anchor
  mass is about `0.906`, so non-anchor aggregation is mostly lost.
- The bounded learned delta did not learn transferable corrections under the
  small-loop MMLU auxiliary CE objective.
- Residual scale stayed near the initialization `0.5`, so it did not become a
  selective transfer-strength mechanism.
- Entropy confidence is token-level; v2.2's strongest result depends on
  token/head confidence gating.
- The smoke gain on `abstract_algebra` did not transfer to full benchmark.

Current best Route-3 remains Route-3.4:

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 token_mlp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 |

Next credible Route-3 should move away from local prior/gate tweaks and toward
offline candidate-utility supervision plus head-aware transfer control.


## E31 Route-3.12 Offline Candidate-Utility Supervision

### Goal

E31 tested offline candidate-utility supervision for Route-3 learned alignment.
It builds a forced-candidate utility JSONL cache with a Route-3.4 teacher/init
projector, then trains cached `grad_ce_margin_rank` instead of doing online
four-candidate replay per training step.

### Setup

- Receiver: `Qwen/Qwen3-0.6B`
- Sharer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Init/teacher checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048/final`
- Variant:
  `offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0`
- Train data: MMLU `auxiliary_train`, `2048`, train split `1843`
- Eval data: MMLU-Redux, AI2-ARC, OpenBookQA

### Diagnostics

- Unit test: `59 passed`
- Full cache coverage: `1843/1843`, cache rate `1.0`
- Cache target anchor/top1/entropy/margin: `0.2204` / `0.6447` / `0.6090` / `0.4117`
- Cache utility margin: `0.0614`
- Smoke MMLU-Redux `abstract_algebra`, limit 20: `21.05`

### Results

| Method | MMLU-Redux | AI2-ARC | OpenBookQA | Mean | Delta vs v2.2 | Delta vs Route-3.4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 | +0.00 | +1.28 |
| `route3_learned_alignment_kv_router_grad_ce_w005_span010_anchor1p0_small2048` | 46.29 | 53.13 | 49.20 | 49.54 | -1.28 | +0.00 |
| `route3_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048` | 45.38 | 52.17 | 45.80 | 47.79 | -3.03 | -1.75 |

### Artifacts

- Checkpoint:
  `local/checkpoints/route3_learned_alignment/qwen3_0.6b+tinyllama1.1b_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048/final`
- Cache summary:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048/candidate_utility_train_summary.json`
- Valid eval summaries:
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048/mmlu-redux/Rosetta_mmlu-redux_generate_20260512_235413_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048/ai2-arc/Rosetta_ai2-arc_generate_20260515_152153_summary.json`
  `local/final_results/route3_learned_alignment/qwen3_tinyllama_learned_alignment_kv_router_offlineutility_r34teacher_c4_t075_wtarget002_wrank001_anchor1p0_small2048/openbookqa/Rosetta_openbookqa_generate_20260516_103139_summary.json`

### Notes

E31 is a negative result on benchmark accuracy but validates the offline utility
cache path. It does not beat Route-3.4 or v2.2. The main observed weakness is the
small utility margin (`0.0614`), meaning many labels are weak preferences. Next
Route-3 should reuse this infrastructure with utility-margin filtering, explicit
no-transfer/down-transfer labels, and token/head transfer gates.
