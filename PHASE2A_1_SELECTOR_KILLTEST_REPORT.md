# Phase 2A-1 Existing A-tier Selector Predictability Kill-Test

## Decision

**NO_GO**

The conjunctive gate failed. Per preregistration, Phase 2A-1 is a NO-GO: do not train a more complex or neural selector. Stop and wait for explicit authorization before any new pre-transfer cache-geometry instrumentation.

## Frozen design and execution

- Source baseline commit: `9fa1f0ac3bedefd282961a853278ab88fb376fa2`
- Implementation commit: `3deca6ae7372dca5b2964b3aa647d4d8199bc3cd`
- Development execution commit: `f7e87c3b97577dac3955cdbfaf1b39981ffa546d`
- Sealed test commit: `488adfedb30824fad08b2ae68636fe7eedfa9dc1`
- Selected candidate: `stump_cot_input_length`
- Candidate family: `single_feature_stump`
- Features: `cot_input_length`
- Frozen threshold: `{"kind": "always_fused"}`
- Calibration-selected comparator: `always_fused`
- Test execution count: one; existing attempt is terminal and no rerun path exists.

## Conjunctive GO gate

- FAIL — `primary_delta_at_least_0p5pp`
- FAIL — `primary_ci_lower_above_zero`
- FAIL — `heterogeneous_pair_sign_rule`
- FAIL — `oracle_headroom_recovery_at_least_15pct`
- FAIL — `harmful_reduction_at_least_25pct`
- PASS — `beneficial_retention_at_least_80pct`

Primary pair-balanced task-macro delta is +0.00 pp,
95% hierarchical paired-bootstrap CI
[+0.00 pp,
+0.00 pp].

## Primary and secondary results

| Estimand | Selector | Comparator | Delta | Transfer | Harm reduction | Benefit retention | Oracle recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pair-balanced task-macro | 49.32% | 49.32% | +0.00 pp | 100.00% | 0.00% | 100.00% | 0.00% |
| Pair-balanced sample-weighted | 47.15% | 47.15% | +0.00 pp | 100.00% | 0.00% | 100.00% | 0.00% |
| Same-rate random (task-macro) | 49.32% | 49.32% | +0.00 pp | 100.00% | 0.00% | 100.00% | 0.00% |

## Pair results

| Pair | Selector accuracy | Delta vs frozen comparator | 95% CI | Transfer rate |
|---|---:|---:|---:|---:|
| tinyllama | 47.98% | +0.00 pp | [+0.00 pp, +0.00 pp] | 100.00% |
| qwen25_0p5b | 47.07% | +0.00 pp | [+0.00 pp, +0.00 pp] | 100.00% |
| llama32_1b | 48.48% | +0.00 pp | [+0.00 pp, +0.00 pp] | 100.00% |
| qwen3_1p7b | 53.74% | +0.00 pp | [+0.00 pp, +0.00 pp] | 100.00% |

Strict cross-family (TinyLlama + Llama3.2) delta:
+0.00 pp. Same-tokenizer Qwen3 control delta:
+0.00 pp.

## Predictive diagnostics

- Benefit AUPRC (balanced pooled): 0.19184530333383512
- Harm AUPRC (balanced pooled): 0.1011423589992004
- Multiclass Brier: 0.4597529239066066
- 15-bin class-macro ECE: 0.02271388684116755
- Random baseline is one deterministic, outcome-blind realization. Per-stratum achieved-rate gaps are frozen in the result JSON; max absolute gap is 0.0.

## Leave-one-out sensitivity

| Fold | Selected candidate | Delta | 95% CI |
|---|---|---:|---:|
| leave_one_pair_llama32_1b | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_pair_qwen25_0p5b | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_pair_qwen3_1p7b | logreg_l2_c10 | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_pair_tinyllama | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_seed_42 | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_seed_43 | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_seed_44 | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_task_ai2-arc | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_task_mmlu-redux | stump_cot_input_length | +0.00 pp | [+0.00 pp, +0.00 pp] |
| leave_one_task_openbookqa | logreg_l2_c001 | -0.15 pp | [-0.70 pp, +0.08 pp] |

Leave-one-out rows are diagnostics only and do not feed back into the global candidate,
threshold, comparator, or GO decision.

## Interpretation boundary

- Primary features are exactly the five preregistered A-tier fields. No raw text,
  task/pair/seed metadata, IDs, labels, correctness, entropy/confidence duplicate,
  or constant fallback feature entered a model.
- Candidate selection occurred only on model-selection data after fit-only training and
  calibration-only probability/threshold selection. Test evaluated one frozen global
  candidate exactly once.
- A positive result is internal predictability evidence only. A failure terminates the
  complex-selector path under the preregistration.

## Reproducibility

The frozen split, candidate, feature, protocol, code, selection lock, model SHA files,
aggregate CSV/JSON, and result manifest live in the repository. Full per-example
predictions remain under `local/` and are intentionally not committed.
