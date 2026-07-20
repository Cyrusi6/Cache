# FPCT GPU R2 Numerical Protocol

Status: `PROSPECTIVE — FLOORS FROZEN BEFORE PRETRAINED OUTPUT`

## Canonical FP32 prior

For legal positive weights, compute in FP32:

`A_ij = w_ij / sum_k w_ik`

`logA_ij = log(A_ij)`

`logA_ij = logA_ij - logsumexp_k(logA_ik)`

Invalid candidates have `logA=-inf`, attention mass zero and gradient zero.
Every legal row must satisfy both:

- `abs(logsumexp(logA)) <= 2e-7`;
- `abs(sum(exp(logA)) - 1) <= 2e-7`.

`A`, legal normalization, `logA`, group logsumexp, additive mask, C_post
weighted K/V reduction, diagnostics and attention logit/softmax accumulation
remain FP32. KV may cast to BF16 only at the cache/attention input boundary.
C_post and F consume the same canonical prior tensor and SHA; per-arm
renormalization is forbidden.

## Synthetic matrix

Tests cover `m=2/3/4`, uniform and extreme priors, padding/causal/all-invalid,
GQA/MQA, FP32/BF16, permutation/refinement, 28-layer replicated accumulation
and the actual patched Qwen3 eager attention path. Pure
`fpct_eager_attention` evidence alone is insufficient.

Existing frozen output/gradient tolerances remain:

- GPU FP32 vs independent FP32 reference: `atol=2e-5`, `rtol=2e-5`;
- FP16 vs FP32: `atol=5e-3`, `rtol=5e-3`;
- BF16 vs FP32: `atol=2e-2`, `rtol=2e-2`;
- FP16/BF16 gradient relative-L2: `0.02/0.05`;
- invalid probability and gradient: exactly zero.

These are kernel/reference gates, not final 28-layer replicated-null floors.

## Metric-specific null floors

The single historical `0.0390625` floor is retired. For each metric `m`:

`tau_m = max(q99.9(depth-matched deterministic null), absolute_floor_m) * 2`

Synthetic seeds are `[1729, 7919, 104729, 20260719]`; depths are
`[1,2,4,8,14,28]`; cardinalities are `[1,2,3,4]`; cases include uniform,
`[1-eps, eps/(m-1)]`, padding and causal masks at Qwen3 head/dimension shapes.
The percentile is fixed at `99.9` and safety multiplier at `2.0`. Each metric
records units, absolute floor, null sample count and distribution SHA256.

| Metric ID | Unit | Absolute floor |
|---|---|---:|
| `delta_fact_max_abs` | logit/output value | `2e-5` |
| `delta_rep_max_abs` | logit/output value | `2e-5` |
| `delta_bypass_max_abs` | logit/output value | `2e-5` |
| `gamma_kl_prior` | nats | `1e-7` |
| `jensen_gap` | logit | `1e-7` |
| `gamma_query_variance` | probability squared | `1e-8` |
| `candidate_logit_range` | logit | `1e-6` |
| `d_k` | KV RMS | `1e-7` |
| `d_v` | KV RMS | `1e-7` |

BF16 replicated-atoms must pass per-layer frozen envelopes and the 28-layer
final-logit envelope. Report max abs, relative L2, KL and greedy equality.

## Forced-on canary

At least one non-final layer and certified panel row must exceed its own frozen
floor for `D_K`, `D_V`, candidate-logit range and either `Delta_fact` or a
query-dependent posterior metric. Replicated, bypass and `m<=1` controls must
simultaneously pass. Checkpoint-native zero gates are recorded as
`EXPECTED_NATIVE_NULL` and do not fail R2 by themselves.
