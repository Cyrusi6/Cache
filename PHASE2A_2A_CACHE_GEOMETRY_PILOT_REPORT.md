# Phase 2A-2a Cache-Geometry Instrumentation Pilot Report

## Decision

**NO_GO.** Gate 1, exact non-perturbation against the frozen B6-native predictions, failed on the Llama3.2 → Qwen3-0.6B pair. The preregistered gates are conjunctive, so the remaining GPU evaluations and all selector fitting were stopped immediately.

This result does **not** establish whether cache geometry predicts harmful transfer. It establishes that this instrumentation run did not satisfy the prerequisite needed to interpret such a selector audit.

## Frozen provenance and isolation

- Source: `main@a320777ee3d8e2c5fbf988ad6cd840b560aab28b`.
- Isolated branch: `research/phase2a2-cache-geometry`.
- Evaluation commit: `7f57a37af18842611a3b85865de6daeb98803a5e`.
- Frozen execution manifest SHA256: `5556fbdcc3cf57f9978527256ef7b2154277d4d3b1fdae20711a3b5b88b2e042`.
- Submitted Kubernetes YAML SHA256: `50b2c84afd7e0e5da361137e288b3b3489f6abb5d219294515308d225aeb294b`.
- Phase 2A-1 split SHA256: `285b5b00cf3598bba075a97b1439b85031ef1cfffdc03b0e7e1775c6338701e0`.
- Scope: frozen `fit` content groups only, seed 42, three heterogeneous pairs. Phase 2A-1 sealed test was not read.
- No training entry point was called; all three B6 checkpoint directory SHAs were hard-validated before evaluation.
- `main` and the FPCT worktree/branch were not modified.

The final jobs were:

- `p2a2-geometry-7f57a37a-5556fbdc-24g8`: TinyLlama and Qwen2.5 on four physically idle UUIDs.
- `p2a2-geometry-7f57a37a-5556fbdc-24g4`: Llama3.2 on two physically idle UUIDs.

All selected x8 GPUs started at 1 MiB used; the selected x4 GPUs started at 396 MiB. Both jobs were deleted after Gate 1 failed, and no Phase 2A-2a Job or Pod remains.

## Instrumentation contract

The observer is disabled by default. When enabled, it records detached sample scalars after raw projected and final fused K/V are available, but before the wrapper overwrites receiver-native K/V. Prediction CSV schema and cache values are not intentionally modified.

The outcome-free layer sidecar records K/V norms, residual ratios, norm ratios, cosines, per-head residual statistics and concentration, K/V imbalance inputs, learned weight, confidence, effective gate, length ratio, and valid alignment mass/coverage. Per-layer data is diagnostic only; the frozen selector whitelist contains 177 compact aggregates.

Final local validation passed 278 repository tests. The default-off projector path preserved outputs and RNG bit-for-bit, and all 7,265 Phase 2A frozen content hashes matched the offline datasets.

## Gate 1 results

ARC and OpenBookQA finished before the fail-fast stop. MMLU had only temporary layer streams and was excluded. The planned TinyLlama/ARC instrumentation-off runtime control had not started.

| Pair | Task | Rows | Exact | Mismatch | Answer mismatch | Generation-only | Reference correct | Instrumented correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama | ARC | 351 | 351 | 0 | 0 | 0 | 196 | 196 |
| TinyLlama | OpenBookQA | 158 | 158 | 0 | 0 | 0 | 81 | 81 |
| Qwen2.5 | ARC | 351 | 351 | 0 | 0 | 0 | 192 | 192 |
| Qwen2.5 | OpenBookQA | 158 | 158 | 0 | 0 | 0 | 88 | 88 |
| Llama3.2 | ARC | 351 | 340 | 11 | 5 | 6 | 198 | 201 |
| Llama3.2 | OpenBookQA | 158 | 154 | 4 | 3 | 1 | 85 | 87 |

Across the two completed tasks, TinyLlama and Qwen2.5 were exact on all 1,018 rows. Llama3.2 differed on 15/509 rows: eight predicted-answer changes and seven generation-form/length-only changes. Its correct count increased by five, but direction is irrelevant to the non-perturbation gate: any changed generation fails exact equivalence.

The mismatch is concentrated in one model pair. This is consistent with either Llama-specific numerical/cache compatibility sensitivity or an instrumentation/timing perturbation. Because the preregistered matched-off control was TinyLlama/ARC and had not run, this pilot cannot distinguish those explanations. No adaptive-gate or selector claim can be made from these data.

## Geometry sanity and resource diagnostics

The completed cells produced 1,527 sample records and 42,756 layer records, exactly 28 layers per sample. All six cells had real within-sample K and V layer variation; 65 K and 65 V aggregate features were nonconstant across samples in every cell. This shows the observer was populated rather than constant, but it is not a formal Gate 2 result because Gate 1 already failed and MMLU was incomplete.

Instrumented evaluator wall time ranged from 83.9 to 233.8 seconds per completed cell. Peak allocated CUDA memory ranged from 3.05 to 4.54 GiB. A matched instrumentation-off run did not complete, so no overhead delta is reported; these values are debug-run resource diagnostics and are not deployment-overhead claims.

At stop time, the artifact root occupied about 295 MiB. Partial MMLU layer streams remain under `local`/`/netdisk` for audit only and are excluded from all metrics.

## Formal gate disposition

| Gate | Status |
|---|---|
| 1. Exact output equivalence | **FAIL** |
| 2. Real geometry variation | Not evaluated formally |
| 3. Pooled harm AUPRC ≥ prevalence + 0.03 | Not evaluated |
| 4. At least 2/3 held-out pairs above prevalence | Not evaluated |
| 5. Selector vs always-fused ≥ +0.5pp | Not evaluated |
| 6. Harmful reduction ≥ 15% | Not evaluated |
| 7. Beneficial retention ≥ 90% | Not evaluated |
| 8. Every pair delta ≥ −0.2pp | Not evaluated |
| 9. Brier beats cross-fitted constant prior | Not evaluated |

No geometry/outcome join was frozen, no correctness labels were opened by the selector script, and no candidate model was fit.

## Conclusion and stop rule

Phase 2A-2a is **NO_GO**. Per the preregistration:

- stop the instance-selector/adaptive-gate route;
- do not increase selector capacity;
- do not expand to three seeds, same-tokenizer control, or untouched benchmarks;
- do not start query-time transport or any subsequent prototype automatically.

A future equivalence-only Llama3.2 matched-off diagnostic would require separate authorization. It must be treated as debugging the instrumentation prerequisite, not as reopening this pilot's selector result.

## Reproducibility and artifacts

- Preregistration: `PHASE2A_2A_CACHE_GEOMETRY_PREREGISTRATION.md`.
- Schema/manifests: `recipe/eval_recipe/phase2a_2a/`.
- Aggregate: `phase2a_2a_failfast_aggregate.{json,csv}` in that directory.
- Code: `rosetta/utils/cache_geometry.py`, evaluator/wrapper integration, pilot/stats/Kubernetes scripts, and tests.
- Large outputs: `/netdisk/lijunsi/c2c-phase2a2-cache-geometry/results` (not committed).
- Frozen runtime files:
  - `/netdisk/lijunsi/c2c-phase2a2-cache-geometry/status/execution_manifest-7f57a37a.json`
  - `/netdisk/lijunsi/c2c-phase2a2-cache-geometry/status/jobs-7f57a37a-5556fbdc.yaml`

The intended full analysis commands remain implemented in `phase2a_2a_cache_geometry_stats.py`, but `freeze-join`/`analyze` were deliberately not run after Gate 1 failed.
