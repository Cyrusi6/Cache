# Phase 1.5 same-checkpoint causal diagnostics report

> Status: complete. All registered evaluations passed the final artifact contract, the pair-balanced 5,000-draw bootstrap finished, and the query-time prototype release gate failed.

## Scope and decision target

Phase 1.5 asks whether the Route-1 v2.2 gains observed in Phase 1 are caused by multiple source candidates, entropy information, learned gating, or instead by checkpoint training trajectory and sender–receiver compatibility. It is a diagnostic stage, not a method-development stage.

The frozen restrictions are:

- no query-time transport prototype before the registered release gate passes;
- no router, replay, optimal transport, RoPE correction, new gate, or new loss;
- no training for the main intervention matrix;
- no checkpoint mutation or intervention-specific checkpoint selection;
- the same full MMLU-Redux, AI2-ARC Challenge, and OpenBookQA development sets used in Phase 1;
- the same four sender→Qwen3-0.6B pairs and seeds 42, 43, and 44;
- all contrasts are paired at the example level and use the same checkpoint on both sides whenever the causal question permits it.

The scientific starting point is main commit `0d308525860d27897bde6d558798e468cf113281`. Phase 1 artifacts were produced by the audited execution revision `9b06d173eada...`; they are frozen inputs rather than rerun or rewritten. The evaluation-only intervention implementation was added in commits `8231f11` through `2b0d6a2`.

Relevant tracked sources are:

- [main intervention recipe](recipe/eval_recipe/phase1_5/route1_phase15_interventions.json)
- [Qwen2.5 seed-44 anomaly recipe](recipe/eval_recipe/phase1_5/qwen25_seed44_gate_anomaly.json)
- [intervention generator and runner](script/analysis/route1_phase15_interventions.py)
- [causal statistics](script/analysis/phase1_5_causal_diagnostics.py)
- [Kubernetes launcher](script/k8s/route1_phase15_jobs.py)
- [Phase 1 final report](ROUTE1_V22_IDENTIFIABILITY_REPORT.md)

## Registered causal questions

### Multiple source candidates

The train-k × eval-k design separates the checkpoint's training trajectory from the number of candidates used at inference:

| Checkpoint | Native evaluation | Intervention | Primary interpretation |
| --- | --- | --- | --- |
| B2, trained with k=1 | eval-k1 | eval-k4 | inference-time coverage effect on a hard-span-trained checkpoint |
| B3, trained with k=4 | eval-k4 | eval-k1 | removal of candidate coverage from a soft-span-trained checkpoint |
| B2 versus B3 at eval-k1 | k1 versus k1 | paired across checkpoints | residual train-trajectory effect when evaluation capacity is matched |
| B2 versus B3 at eval-k4 | k4 versus k4 | paired across checkpoints | residual train-trajectory effect when soft evaluation capacity is matched |

Evidence for a reusable soft-candidate mechanism requires the same-checkpoint k=4 view to beat k=1 in at least two genuinely heterogeneous pairs, the pair-balanced cross-pair 95% CI lower bound to exceed zero, and the gain to concentrate in high-ambiguity tokens or spans.

### Entropy information

Each B6 checkpoint is evaluated under three entropy views:

| View | Entropy values | Position correspondence | Learned confidence gate |
| --- | --- | --- | --- |
| native | original | preserved | preserved |
| constant | fixed at 0.93 | removed | preserved |
| shuffled | original distribution | broken within sequence using the registered seed | preserved |

`native > constant` tests whether the entropy values carry useful information. `native > shuffled` tests whether the information is tied to the correct token/span position rather than only to its marginal distribution. Constant and shuffled are evaluation-time interventions on the same B6 checkpoint; they are not new training methods.

TinyLlama constant/shuffle retraining for seeds 43 and 44 is conditional: it is allowed only if these same-checkpoint interventions first show a stable entropy effect. It is not launched pre-emptively.

### Learned gate

Each B6 checkpoint is evaluated under:

| View | Alignment-confidence component | Legacy scalar K/V component |
| --- | --- | --- |
| learned/native | checkpoint learned gate | checkpoint-native hard threshold |
| static | original static entropy scalar | checkpoint-native hard threshold |
| forced-on | forced to 1 | forced to 1 |

The forced-on intervention covers both gating components. This matters because the checkpoint also contains a legacy per-layer scalar K/V gate; treating only the token/head alignment-confidence path as “the gate” would leave a major confound intact.

If learned and forced-on are statistically indistinguishable, the adaptive-gating claim is removed. If the oracle headroom is large but the learned gate has weak predictive value, the supported next direction is calibrated null/no-transfer rather than a more elaborate adaptive gate.

## Phase 1 artifact audit

The shared authoritative root is `/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache`. The audit was completed before generating or launching Phase 1.5 evaluations.

| Artifact class | Verified inventory | Audit result |
| --- | ---: | --- |
| Phase 1 completion markers | 67 | complete |
| Phase 1 failed markers | 0 | none |
| per-example prediction CSVs | 201 | three tasks for every completed run |
| post-hoc gate diagnostics | 26 | present for the registered gate-bearing runs |
| newly trained final checkpoints | 65 | provenance-consistent |
| reused TinyLlama B6 seed-42 checkpoint | 1 | bitwise-verified directory hash |
| required non-B0 checkpoints | 66 | all load all 28 projectors |

All 66 required checkpoints passed tensor-level completeness checks: every expected projector file was loadable, all 28 receiver layers were present, and all loaded tensor values were finite. For the 65 newly trained checkpoints, the recorded run id, execution commit, training-config SHA, split hash, and dataset hash matched the corresponding manifest. The sole reused checkpoint, TinyLlama B6 seed 42, retained the verified directory SHA256:

`a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`

The shared checkpoint tree occupied approximately 61 GB, Phase 1 result CSVs approximately 401 MB, and the shared volume had approximately 923 GB free at launch audit. All three NVIDIA nodes could read the shared volume. The Kubernetes GPU request audit found all 14 NVIDIA GPUs available before Phase 1.5 submission.

These checks rule out missing artifacts, partial checkpoint serialization, obvious tensor corruption, and provenance mismatch as explanations for the planned intervention results. They do not by themselves establish any causal mechanism.

## Qwen2.5-0.5B→Qwen3-0.6B B6 seed-44 collapse

### Observed downstream failure

The full 7,265-example pooled development set contains 5,615 MMLU-Redux, 1,150 ARC, and 500 OpenBookQA examples.

| Seed | Weighted accuracy | Macro accuracy | Positive-transfer conditional rate | Negative-transfer conditional rate |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 46.194% | 50.215% | 26.671% | 19.338% |
| 43 | 45.891% | 49.498% | 24.795% | 16.863% |
| 44 | 38.789% | 41.570% | 10.263% | 10.849% |

Seed 44 therefore collapsed by roughly 7.25 percentage points relative to the mean of seeds 42 and 43. The dominant change was the loss of beneficial receiver-wrong→fused-correct transfers, not an explosion of receiver-correct→fused-wrong transfers. All 7,265 generated predictions were legal answer choices.

The failure was not specific to a single B6 evaluation artifact. Relative to the seeds 42/43 mean, seed 44 was also worse for B1 (about −4.61 pp), B3 (about −1.06 pp), and B5 (about −2.97 pp), while B2 was nearly stable (about −0.11 pp). The seed trajectory was generally unfavorable for this sender–receiver pair, and B6 amplified it.

### Excluded failure modes

Checkpoint, training, and logging inspection found no evidence of:

- NaN or Inf values in model/projector tensors;
- an incomplete or mismatched checkpoint;
- illegal predictions or evaluator parsing collapse;
- optimizer numerical failure;
- an exceptional gradient explosion relative to the other seeds;
- alignment-confidence collapse to zero.

Seed 44 ended with train loss 0.1326 and eval loss 0.1211. Its eval loss was lower than seed 42 (0.1749) and seed 43 (0.1395), which is direct evidence that this small training eval split is not a reliable selector for downstream transfer quality. The run completed normally, and the observed maximum gradient and clipping frequency were not diagnostic of a numerical failure.

### Gate state and most likely causal chain

The token/head alignment-confidence gate was essentially saturated for all three seeds: its mean was about 0.9996 and approximately 99.93% of values were in the registered high-saturation region. Because this behavior was shared by seeds 42, 43, and 44, it cannot explain the seed-specific collapse.

The checkpoint also contains a legacy scalar Gumbel K/V gate per receiver layer. At evaluation it becomes a hard decision via a zero-logit threshold. The learned logits lie near zero, so very small training-trajectory changes can flip an entire layer's K or V transfer path.

For seed 44, 14 of 28 layers had K enabled and 18 had V enabled; 9 layers had both disabled. Among the first 9 layers, only one K path and two V paths were enabled, and 7 layers had both disabled. The first 6 layers performed no transfer at all. By comparison, seeds 42 and 43 retained more early-layer transfer paths.

The strongest explanation supported before intervention is therefore:

`seed-specific training trajectory × near-zero legacy scalar hard masks × Qwen2.5/Qwen3 cache compatibility`.

This is a diagnostic hypothesis, not yet a final causal conclusion. It motivates two anomaly-only same-checkpoint interventions on Qwen2.5 B6 seed 44:

- `alignment_forced_on`: force alignment confidence to one while preserving the checkpoint-native legacy scalar K/V masks;
- `legacy_forced_on`: preserve learned alignment confidence while forcing the legacy scalar K/V masks on.

They are kept separate from the registered 72-triplet main matrix.

## Evaluation matrix

The main matrix contains four pairs, three seeds, and six non-native interventions, for 72 new three-task triplets or 216 new dataset evaluations. Thirty-six native B2/B3/B6 triplets are reused from Phase 1 as comparators.

| Pair id | Sender | Tokenizer relation used for analysis |
| --- | --- | --- |
| `tinyllama` | TinyLlama-1.1B | heterogeneous |
| `qwen3_1p7b` | Qwen3-1.7B | same tokenizer control |
| `qwen25_0p5b` | Qwen2.5-0.5B | heterogeneous |
| `llama32_1b` | Llama3.2-1B | heterogeneous |

The six new interventions are B2 eval-k4, B3 eval-k1, B6 entropy-constant, B6 entropy-shuffled, B6 gate-static, and B6 gate-forced-on. Every generated evaluator configuration records the exact selected checkpoint and intervention provenance. Required outputs per dataset are exactly one prediction CSV, one evaluator summary JSON, and `eval_intervention_provenance.json`; the provenance records that training state was not mutated.

## Ambiguity evaluation

Ambiguity buckets are fixed from a native source before comparing intervention outcomes. Top-k contrasts use the corresponding native B3 diagnostics; entropy and gate contrasts use native B6 diagnostics. This prevents an intervention from redefining the subset on which it is judged.

Two high-ambiguity definitions are reported:

- absolute ambiguity: maximum candidate count greater than one and at least one nonzero ambiguity diagnostic;
- within pair/seed/task q75: the top quartile of a composite score built from available normalized `alignment_entropy`, `one_to_many_rate`, and `boundary_mismatch` fields.

For each contrast the report includes the accuracy delta within high and low strata and the interaction `delta_high − delta_low`. The cross-pair interaction CI resamples pairs, seeds within pair, and paired examples independently within the fixed high/low strata.

## Oracle abstention headroom

For each fused method and example, the oracle selects the fused answer when fused is correct and otherwise falls back to receiver-only if the receiver is correct. It reports:

- fused accuracy and receiver accuracy;
- oracle accuracy;
- oracle headroom over fused, receiver, and the better fixed policy;
- ideal abstain count/rate, equal to receiver-correct→fused-wrong events;
- beneficial-transfer count/rate;
- the fraction of examples on which a transfer decision matters.

This oracle is not an implementable method and is not used as a performance claim. It bounds what a perfect abstention decision could recover from the already observed receiver/fused predictions.

## Statistical protocol

All ordinary intervention comparisons require exact example-key equality and the registered full task sizes. Per-task rows and a 7,265-example pooled row are produced for every pair and seed.

The primary cross-pair interval is the same pair-balanced hierarchical paired bootstrap used in Phase 1:

1. resample model pairs with replacement;
2. within each selected pair, resample seeds with replacement;
3. within each selected pair/seed cell, resample paired examples with replacement;
4. weight model pairs and seeds equally rather than allowing the largest task or pair to dominate.

The registered defaults are 5,000 bootstrap draws, 95% confidence, and deterministic bootstrap seed `20260718`. Reports also include exact McNemar p-values at the pair/seed/task level, positive-pair count, positive genuinely heterogeneous-pair count, and seed sample standard deviation (`ddof=1`). McNemar significance from a large pooled example count is not allowed to replace cross-pair stability.

## Reproducible manifest and commands

The large materialized manifests remain under `local/` and are not committed. They are regenerated from the tracked recipes and frozen Phase 1 artifacts with:

```bash
python script/analysis/route1_phase15_interventions.py generate \
  --phase1-manifest local/tmp/route1_identifiability_suite/manifest.json \
  --phase1-analysis-manifest local/tmp/route1_identifiability_suite/analysis_manifest.json \
  --phase1-artifact-root /netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache \
  --output-root local/tmp/phase1_5_causal_diagnostics \
  --results-root local/final_results/phase1_5_causal_diagnostics/rev_0d30852 \
  --recommended-shards 7 \
  --shard-results-root 2=/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/phase1_5_causal_diagnostics_x8/rev_0d30852 \
  --shard-results-root 3=/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/phase1_5_causal_diagnostics_x8/rev_0d30852 \
  --shard-results-root 4=/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/phase1_5_causal_diagnostics_x8/rev_0d30852 \
  --shard-results-root 5=/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/phase1_5_causal_diagnostics_x8/rev_0d30852

python script/analysis/route1_phase15_interventions.py generate-anomaly \
  --phase1-manifest local/tmp/route1_identifiability_suite/manifest.json \
  --phase1-analysis-manifest local/tmp/route1_identifiability_suite/analysis_manifest.json \
  --phase1-artifact-root /netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache \
  --output-root local/tmp/phase1_5_causal_diagnostics/qwen25_seed44_gate_anomaly \
  --results-root local/final_results/phase1_5_causal_diagnostics/rev_0d30852/qwen25_seed44_gate_anomaly
```

One registered two-GPU shard is executed with:

```bash
python script/analysis/route1_phase15_interventions.py run-shard \
  --manifest local/tmp/phase1_5_causal_diagnostics/manifest.json \
  --shard-index <0-6> \
  --num-shards 7
```

Cross-pod opportunistic execution is not a supported reproduction command on the current NFS deployment; use disjoint shard/run allocation.

The analysis command is:

```bash
python script/analysis/phase1_5_causal_diagnostics.py \
  --manifest local/tmp/phase1_5_causal_diagnostics/manifest.json \
  --anomaly-manifest local/tmp/phase1_5_causal_diagnostics/qwen25_seed44_gate_anomaly/manifest.json \
  --output-dir local/final_results/phase1_5_causal_diagnostics/rev_0d30852/analysis \
  --bootstrap-samples 5000 \
  --bootstrap-confidence 0.95 \
  --bootstrap-seed 20260718
```

The optional anomaly manifest adds `alignment_forced_on − native` and `legacy_forced_on − native` rows, including task/pooled paired statistics and oracle headroom, without changing the registered eight main comparisons. The command writes `paired_interventions.csv`, `hierarchical_interventions.csv`, `seed_variance.csv`, `oracle_abstention.csv`, `ambiguity_interactions.csv`, and `summary.json`. Large per-example and intermediate files remain under `local/` or `/netdisk` and are not committed.

## Kubernetes execution infrastructure

The 72 triplets are deterministically divided into seven logical two-GPU shards with run counts `[11, 11, 10, 10, 10, 10, 10]`. Within each triplet, ARC runs on the first GPU and OpenBookQA on the second concurrently, followed by MMLU-Redux on both GPUs.

The final node-pool layout uses all available memory without requiring every logical shard to request four cards:

- `4090-24gx4`: a four-GPU node job covers shards 0 and 1 sequentially or as available two-GPU groups;
- `4090-24gx8`: an eight-GPU node job covers shards 2–5 with bounded two-GPU-group parallelism;
- `4090-48gx2`: a two-GPU node job covers shard 6;
- the two Qwen2.5 seed-44 anomaly triplets are separately queued on one two-GPU group.

The launcher validates the exact Git commit and execution-manifest SHA before evaluation, pins the audited Phase 1 runtime constraints, preserves assigned CUDA UUID masks only under an explicit environment flag, checks actual visible GPU memory, and resumes only from complete per-dataset output contracts. It also pre-creates shared result trees serially and isolates the x8 result root to avoid known autofs/NFS negative-dentry and concurrent-directory-creation races.

The tracked implementation passed 236 repository tests with two pre-existing Pydantic warnings. The node-level Kubernetes manifests passed API-server dry-run before formal submission.

The x8 node later became `NodeStatusUnknown` after 84/120 shard-2-to-5 dataset outputs were complete, three MMLU outputs had provenance only, and 33 outputs had not started. Recovery of shards 2--4 on x4 and x48 was output-contract safe. A subsequent attempt to let multiple pods work-steal shard 5 exposed an important infrastructure limitation: advisory `flock` on this shared NFS path was not mutually exclusive across pods. Five datasets were therefore evaluated twice. Their predictions, correctness, ambiguity fields, gate diagnostics, and length diagnostics were identical; only latency measurements and timestamp-derived artifact names differed. The ghost bundles were moved intact to a `local/tmp` quarantine and excluded from analysis.

The remaining tail was completed with immutable, explicitly non-overlapping Kubernetes Jobs, each bound to one registered run id and a fixed physical GPU UUID pair. The old serial worker was stopped between runs and deleted only after its active MMLU output passed the complete artifact contract. Cross-pod opportunistic work stealing is consequently treated as unsupported on the current NFS deployment; reproducible recovery uses explicit run allocation plus exactly-one-output auditing.

The first final audit then found one provenance-only reproducibility defect: TinyLlama B2 eval-k4 seed 43 had been evaluated under an earlier generated manifest whose three YAML byte hashes were no longer present, although its effective intervention and checkpoint matched the final registration. That complete old bundle was quarantined and the single triplet was rerun from the final manifest. Old and new predictions, correctness, ambiguity fields, confidence/gate diagnostics, length diagnostics, bad-sample artifacts, and every CSV field except latency were identical. The rerun therefore repaired byte-level reproducibility without changing any scientific result.

The final strict audit passed all 74 runs and 222 dataset evaluations: x8 `120/120`, main `96/96`, and anomaly `6/6`. Every dataset has exactly one prediction CSV, summary, intervention provenance, gate diagnostic, and length diagnostic; registered row counts, sample-key uniqueness, JSON parsing, checkpoint/intervention identity, internal provenance hashes, and current evaluation-config hashes all pass. The authoritative main and anomaly manifest SHA256 values are respectively `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1` and `bd305268e9a8527cb75407293b49cae4e577bb10516e9643781573e861cfa5d2`.

The 5,000-draw statistics were also run independently in Kubernetes and in the local audited Conda environment with the same seed. `paired_interventions.csv`, `oracle_abstention.csv`, and `ambiguity_interactions.csv` were byte-identical. The remaining numeric differences were only floating-point serialization in sample standard deviations, with maximum absolute difference `4.337e-19` and no string or decision-field differences.

## Results

### Train-k × eval-k

| Contrast | Cross-pair delta | 95% CI | Positive heterogeneous pairs | High-ambiguity interaction | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 same checkpoint: eval-k4 − eval-k1 | −0.01 pp | [−0.13, +0.11] pp | 1/3 | q75 −0.02 pp; absolute −0.05 pp | no detectable average effect |
| B3 same checkpoint: eval-k4 − eval-k1 | +0.03 pp | [−0.06, +0.14] pp | 3/3 | q75 −0.09 pp; absolute −0.01 pp | no detectable average effect |
| B3-trained − B2-trained at eval-k1 | +1.25 pp | [−0.64, +3.56] pp | 1/3 | q75 −0.29 pp; absolute +1.17 pp | training/checkpoint contrast; inconclusive |
| B3-trained − B2-trained at eval-k4 | +1.30 pp | [−0.67, +3.63] pp | 2/3 | q75 −0.36 pp; absolute +1.20 pp | training/checkpoint contrast; inconclusive |

The same-checkpoint candidate-count interventions are effectively zero and provide no positive ambiguity-concentration evidence. By contrast, the B3-trained minus B2-trained point estimate is about +1.3 pp at either evaluation k, but its cross-pair interval crosses zero and its seed standard deviation is about 2.4 pp. The training-checkpoint difference is largest for the same-tokenizer Qwen3 pair (+3.12 pp) and TinyLlama (+2.01 to +2.02 pp), while Qwen2.5 and Llama3.2 are approximately zero. This separates inference-time candidate count from a checkpoint/training-regime effect that is strongly pair-dependent; it does not identify whether training-time k4 exposure, optimization trajectory, tokenizer identity, or another compatibility factor caused that checkpoint difference.

### Entropy interventions

| Contrast | Cross-pair delta | 95% CI | Positive heterogeneous pairs | High-ambiguity interaction | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| native − constant | +0.13 pp | [−0.13, +0.40] pp | 1/3 | q75 +0.10 pp; absolute −0.01 pp | unstable; CI crosses zero |
| native − shuffled | +0.04 pp | [−0.07, +0.22] pp | 1/3 | q75 −0.02 pp; absolute −1.19 pp | unstable; CI crosses zero |

Native entropy is not stably better than either constant confidence or position-shuffled entropy. The effects are small, change sign across pairs, and provide no reliable positive ambiguity-concentration evidence. This does not prove that entropy is uninformative in every setting; it means Phase 1.5 does not identify a general position-matched entropy benefit. The condition for additional TinyLlama constant/shuffle retraining is therefore not met.

### Gate interventions

| Contrast | Cross-pair delta | 95% CI | Positive heterogeneous pairs | Status |
| --- | ---: | ---: | ---: | --- |
| learned − static | −0.01 pp | [−0.10, +0.07] pp | 0/3 | no learned-gate advantage |
| learned − forced-on | −0.21 pp | [−0.98, +0.62] pp | 2/3 | no learned-gate advantage |

The learned token/head modulation is indistinguishable from the static-confidence view. The broader `learned − forced-on` contrast also changes checkpoint-native legacy scalar K/V masks; it has no pair-balanced advantage and reverses sign by pair, so it cannot be used as a clean token/head-gate comparison or as evidence that forced-on is universally better. This does not support an adaptive-capacity performance claim.

The ambiguity interactions are exploratory and coverage-limited. Outside TinyLlama, pooled q75 strata often separate MMLU from ARC/OpenBookQA because the latter lack varying diagnostic fields; the absolute definition is sparse for three pairs and marks all TinyLlama examples high. Consequently, the negative absolute `learned − forced-on` interaction is not treated as a universal negative ambiguity mechanism. The defensible release conclusion is narrower: neither registered top-k contrast provides reliable positive high-ambiguity concentration.

### Qwen2.5 B6 seed-44 component isolation

| View | Weighted accuracy | Positive transfer | Negative transfer | Interpretation |
| --- | ---: | ---: | ---: | --- |
| native | 38.789% | 10.263% | 10.849% | audited Phase 1 result |
| alignment forced-on, legacy native | 38.789% | 10.263% | 10.849% | null intervention at the correctness level |
| alignment learned, legacy forced-on | 40.702% | 14.036% | 12.219% | partial rescue through more positive transfer, with some added negative transfer |

Forcing only the alignment-confidence gate changed one of 7,265 predicted letters and changed zero correctness outcomes: pooled delta `+0.000 pp`, improvements/regressions `0/0`, exact McNemar `p=1.0`. This checkpoint's recorded alignment-confidence gate was already almost fully saturated, so the seed-44 collapse was not caused by that gate refusing to open.

Forcing the legacy per-layer scalar K/V masks on changed 816 predicted letters and improved pooled accuracy by `+1.913 pp` (`336` improvements versus `197` regressions; exact McNemar `p=1.85e-9`). ARC improved `+4.174 pp`, MMLU-Redux `+1.496 pp`, and OpenBookQA `+1.400 pp`. This intervention recovered about 26% of the gap between seed 44 and the mean of seeds 42/43, while also increasing negative-transfer rate. The legacy hard masks are therefore a real causal contributor to the collapse, but not a complete explanation or a robust remedy.

### Oracle abstention

| Method/view | Oracle headroom over fused | 95% CI | Ideal abstain rate | Status |
| --- | ---: | ---: | ---: | --- |
| B6 native | +8.24 pp | [+6.28, +10.19] pp | 8.244% | positive in 4/4 pairs |
| B6 entropy constant | +8.08 pp | [+6.06, +10.05] pp | 8.080% | similar to native |
| B6 entropy shuffled | +8.32 pp | [+6.27, +10.26] pp | 8.316% | similar to native |
| B6 gate static | +8.21 pp | [+6.34, +10.10] pp | 8.214% | similar to native |
| B6 gate forced-on | +10.43 pp | [+8.05, +12.34] pp | 10.430% | more recoverable negative transfer |

The B6 native oracle can recover 8.24 pp over fused, with an interval strictly above zero and positive headroom in all four pairs. This is a large no-transfer/abstention opportunity, not an achieved method result. The registered outputs do not contain a calibrated prediction score for whether transfer will help, so Phase 1.5 does not estimate gate AUC or claim that the current gate can realize this oracle. Given the null learned-gate interventions and the saturated alignment-confidence diagnostics, the supported direction is calibrated null/no-transfer rather than a stronger adaptive-gate claim.

## Final mechanism conclusion

1. **Multiple source candidates are not an identified average inference-time cause of v2.2's gain on these development tasks.** Same-checkpoint eval-k4 versus eval-k1 is essentially zero for both B2 and B3, both cross-pair intervals cross zero, and neither ambiguity definition provides reliable positive concentration. This does not rule out training-time effects of candidate exposure in other settings.
2. **Entropy confidence does not show a stable general causal signal at inference.** Native-versus-constant and native-versus-shuffled effects are +0.13 pp and +0.04 pp with intervals crossing zero and weak pair consistency. The `entropy-aware` mechanism claim should be removed, and the conditional TinyLlama seed-43/44 retraining should not be launched.
3. **The learned token/head gate does not demonstrate adaptive performance value.** Learned-versus-static is −0.01 pp with no positive cross-pair interval. Learned-versus-forced-on is broader because it also changes legacy scalar K/V masks and reverses by pair; it supplies no universal gate policy. Qwen2.5 seed 44 shows that forcing the already saturated alignment-confidence gate is a null intervention, while forcing legacy scalar K/V masks yields a real but partial +1.91 pp rescue.
4. **The remaining variation is more consistent with training-regime/checkpoint and pair-compatibility effects.** The B3/B2 checkpoint difference is strongest on the same-tokenizer Qwen3 pair and TinyLlama but absent on Qwen2.5 and Llama3.2. The experiment does not causally isolate tokenizer identity, training-time k4 exposure, or random optimization trajectory; it does show that inference-time candidate count is not the source of that difference.
5. **There is substantial abstention headroom.** B6 native has +8.24 pp oracle headroom with 95% CI [+6.28, +10.19], but Phase 1.5 does not show that the current gate predicts those events. The registered next direction is calibrated null/no-transfer.

The query-time prototype release gate **fails**. B2 has only 1/3 positive heterogeneous pairs; B3 has 3/3 but its cross-pair CI lower bound is −0.06 pp; neither contrast provides reliable positive ambiguity-concentration evidence. No query-time transport prototype is authorized from these results.
