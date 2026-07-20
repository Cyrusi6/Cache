# FPCT-3.7-R1 Sealed Certified-Support Protocol

## Dependency and scope

FPCT-3.7-R1 is authorized only after FPCT-3.5P reports exact sealed replay
equality.  It creates a new lock and new artifact root; it never resumes or
modifies the failed `b11a046...` execution.  It is a label-free CPU alignment
audit, not an accuracy, query-separability or scientific-power test.

The scientific contract is inherited from
`recipe/eval_recipe/fpct_3_7/certified_support_manifest.json`: the four pairs,
three tasks, four splits, `top_k=4`, `soft_span_overlap_v2`, uniform prior,
`certified_slot0_v1`, exact Qwen3 identity, readiness floors/ranking and resource
formulas are unchanged.  The paired protocol diff must contain zero scientific
field changes.

## Import/provenance gate

All formal modes run through `script/runtime/fpct_bootstrap.py` using the
absolute `python -I` command frozen by FPCT-3.5P.  Freeze and all twelve
pair×task shards compare the same stable closure fingerprint and record it in a
sealed shard manifest.  Every loaded `rosetta.*` module must remain under the
single current research worktree package path before and after the shard.

## Primary audit and readiness

The existing frozen audit output remains normative for raw/certified 12-cell
support, Wilson intervals, sanitizer integrity, exact identity and readiness.
Readiness uses certified `m>=2` on fit+calibration only:

- at least 30 certified-positive distinct groups in each task;
- at least 100 pooled certified-positive distinct groups.

The same-tokenizer Qwen3 control never enters ranking.  Model-selection and test
remain label-free descriptive splits and cannot alter readiness.

Qwen3 exact identity must hold for every retained eligible parent across all
splits and truncation/padding conditions: `m=1`, `i→i`, weight one,
`fallback=false`, no `m0/m2/m3/m4`, and zero extra slots.

## Additional descriptive outputs

Without changing the frozen certification decision, each pair×task reports:

1. `raw_pre_truncation → retained_after_truncation → sanitized` transitions for
   `m0..m4`, at parent, sample and distinct-content-group units;
2. certified positive-group rate, ambiguous-parent density, per-group ambiguous
   counts, ambiguous/eligible parents, `sum_i(1-Amax_i)`, and distributions of
   `Amax`, entropy and `N_eff`;
3. sanitizer-removed non-top1 prior mass;
4. receiver/source interval geometry, candidates, Unicode/byte-fallback flags,
   top-k exhaustiveness, event/cluster counts and task/split/subject strata;
5. raw and certified expansion mean/p50/p95/p99/max, attention FLOP ratio,
   K/V sidecar bytes per layer/all layers and ambiguity-density correlations;
6. identical `c_pre/c_post/f` sanitized alignment-input hashes, exact row sums,
   exact slot-0 one-hot uncertified rows, native `m0`, unchanged `m1`, and no
   illegal or truncated positive mass.

Secondary flags and the added p99/correlation descriptions are non-operative:
they cannot change certification, ranking, thresholds or GPU eligibility.

## Decisions and claim boundary

If TinyLlama misses any frozen floor, the terminal state is
`NO_GO_GPU_CURRENT_CERTIFIER`: the present conservative character-partition
certifier has insufficient experiment support.  This does not show FPCT is
mathematically invalid, that uncertified candidates are meaningless, or that
slot 0 is an optimal universal alignment.

`exact_identity` proves tokenizer-index identity only, not equality of model KV
spaces.  Cross-model `P_K/P_V` and RoPE position transport remain necessary.
The character partition is sufficient but not necessary and may discard real
byte-fallback one-to-many cases.  The headline remains confined to certified
ambiguity and, if later reached, `F-C_post`; no cross-model universal claim is
authorized.
