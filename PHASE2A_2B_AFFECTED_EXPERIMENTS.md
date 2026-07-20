# Phase 2A-2b Affected Experiment Inventory

## Classification rule

- **safe**: paired arms are established to use the same prompt-relevant config and the same rendered date, or the later CPU computation operates on immutable SHA-bound stored artifacts.
- **ambiguous**: no per-example input hash exists and a date confound may enter a cross-seed/generalization interpretation, or the intended analysis never ran.
- **invalid**: paired Llama3.2 arms are confirmed to span different rendered dates; tokenizer probes establish that the corresponding rendered prompts and input IDs differ.

This inventory uses configuration, artifact filenames/timestamps, execution manifests, tokenizer templates and provenance only. It does not read correctness, labels, sealed outcomes or prediction CSV contents.

## Aggregate inventory

| Phase | Safe | Ambiguous | Invalid | Interpretation |
|---|---:|---:|---:|---|
| Phase 1 source suite | 9 | 1 | 0 | Within-seed/task component contrasts are same-date; Llama cross-seed aggregate spans 17–18 July without input hashes. |
| Phase 1.5 causal diagnostics | 24 | 0 | 30 | Same-date Llama interventions remain usable; cross-date native/intervention contrasts are invalid. |
| Phase 2A-0 opportunity audit | 1 | 1 | 0 | Stored-artifact CPU computation is reproducible; Llama cross-seed interpretation inherits date confounding. |
| Phase 2A-1 selector kill-test | 1 | 1 | 0 | Stored-artifact CPU computation is reproducible; Llama cross-seed/generalization interpretation inherits date confounding. |
| Phase 2A-2a cache geometry | 0 | 1 | 2 | ARC/OpenBookQA historical-reference Gate 1 is invalid; geometry predictivity is untested. |
| **Total audit rows** | **35** | **4** | **32** | 71 task/scope-level audit records. |

## Material impact

1. Phase 1 within-seed component identification is not invalidated by this issue: for each Llama seed/task, B1/B2/B3/B5/B6 share one date and common prompt configuration. The three-seed Llama aggregate remains ambiguous because seed 42 was rendered on 17 July while seeds 43/44 were rendered on 18 July.
2. Phase 1.5 conclusions that use Llama native and intervention arms from different dates must be withdrawn or recomputed. The exact 30 invalid task-level rows and 24 safe rows are enumerated in the machine-readable inventory.
3. Phase 2A-0 and Phase 2A-1 code/results remain reproducible as computations over immutable stored files. Claims that depend on Llama cross-seed or cross-pair generalization require qualification because prompt date is confounded with seed.
4. Phase 2A-2a Gate 1 compared 17 July historical references with 19 July runs for ARC and OpenBookQA; those two comparisons are invalid as instrumentation-equivalence evidence. Geometry predictivity did not run, so it is neither supported nor refuted.
5. Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-0.5B and TinyLlama templates contain no detected clock, timezone, locale, process-environment or randomness dependency. They are outside the Llama-specific historical drift set.

The authoritative row-level list is `recipe/eval_recipe/phase2a_2b_prompt_identity_audit/affected_experiments.csv`; its JSON companion also includes source SHAs, model fingerprints and the CPU environment matrix.
