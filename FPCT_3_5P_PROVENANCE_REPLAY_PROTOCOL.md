# FPCT-3.5P Sealed Provenance Replay Protocol

## Status and purpose

FPCT-3.5P is a prospective, CPU-only deterministic replay of the immutable
FPCT-3.5 alignment forensic under a sealed same-process import contract.  The
historical execution at `0398d26b63e96263b813730368275ee66e313f66` remains
unchanged and is treated as `PROVISIONAL / IMPORT-ORIGIN-UNSEALED` until this
replay finishes.  No natural tokenizer or canonical sample may be opened before
the clean commit containing this protocol, its manifest, the bootstrap, the
replay target and hostile subprocess tests is pushed.

## Inherited scientific contract

The operative definitions are inherited byte-for-byte in meaning from
`FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md` and
`recipe/eval_recipe/fpct_3_5/alignment_correctness_manifest.json`:

- exact runtime identity and its hard-error/no-fallback rule;
- the message-relative clipped-character coordinate system;
- the ten sufficient conditions for certified one-to-many;
- `top_k=4` and positive-overlap exhaustiveness;
- `certified_slot0_v1`, applied identically to `c_pre`, `c_post` and `f`;
- the same pair, task, split and 7,265/7,233 canonical universes;
- the 30-per-task and 100-pooled certified support readiness floors;
- all original claim boundaries.

The machine-readable `protocol_diff.json` must report zero scientific changes.
Only import execution, provenance sealing and additional descriptive geometry
flags are new.

## Canonical sealed invocation

Every freeze, task shard and comparison is invoked from the repository root as:

```text
/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python3.10 -I \
  /home/lijunsi/projects/Cache-fpct-factorized-transport/script/runtime/fpct_bootstrap.py \
  --repo-root /home/lijunsi/projects/Cache-fpct-factorized-transport \
  --target /home/lijunsi/projects/Cache-fpct-factorized-transport/script/analysis/fpct_3_5p_provenance_replay.py \
  -- <target arguments>
```

The interpreter, repository, bootstrap and target must be absolute realpaths;
the cwd must be the repository root.  `python -I` is mandatory.  `PYTHONPATH`,
`PYTHONHOME`, user-site packages, direct-script and bare `python -m` execution
are non-operative.  The existing Conda editable distribution is recorded but
is neither removed nor used for source resolution.

Before the target runs, the bootstrap imports and attests the regular `rosetta`
package, aligner, dataset adapter, prompt evaluator, FPCT-1B audit, FPCT-3.5
audit and FPCT-3.7 audit in the same process.  It performs a real fake-tokenizer
`exact_identity` call with `apply_confidence_control=False` before any protected
natural-data path can be opened.  Each target also recomputes the closure after
execution; origin, bytes, method signature or namespace drift is a hard error.

## Replay projection and equality gate

New output is written only under
`local/final_results/fpct_factorized_transport/fpct_3_5p_provenance_replay/rev_<execution_sha>/`.
The immutable original directory is read-only.

The replay must exactly reproduce:

- 7,265/7,265 runtime-identity samples;
- 802 raw Qwen3 `m=2` parents in 104 distinct content groups;
- 410 parents and 56 groups in fit+calibration;
- 401 overlap clusters, where a cluster is a sample plus ordered candidate
  index signature;
- canonical `(sample,parent)` rows;
- ordered and multiset candidate atoms `(index, token_id, offset, weight)`;
- task/split counts and Qwen3/Qwen2.5 row equality;
- a deterministic radius-four receiver token/offset context projection.

The normalized comparison ignores execution SHA, timestamp and absolute output
path fields only.  Set or aggregate equality alone is insufficient.  Any row,
ordered atom, multiplicity or context mismatch yields
`FORENSIC_REPLAY_MISMATCH`, leaves the old sanitizer/taxonomy unchanged and
blocks FPCT-3.7-R1 and all downstream model/GPU work.

## Descriptive geometry flags

The original mutually exclusive primary category and its precedence are not
changed.  Each raw anomaly additionally receives non-exclusive flags:

`receiver_zero_length`, `receiver_overlap`, `source_zero_length`,
`source_duplicate`, `source_overlap`, `coverage_gap`, `non_monotonic`,
`topk_non_exhaustive`, `truncation_loss`, and `illegal_slot0`.

A primary-reason × secondary-flag co-occurrence matrix is reported.  In
particular, `duplicate_or_overlap_receiver_offsets` is a precedence-based stop
reason; identical Qwen3 source geometry can simultaneously contain source
overlap.  The flags are descriptive only and cannot change certification,
sanitization or readiness.

## Decisions

- Exact normalized equality: FPCT-3.5P `PROVENANCE_CONFIRMED`, proceed to the
  independently locked FPCT-3.7-R1 audit.
- Any import, attestation, identity or replay mismatch:
  `FORENSIC_REPLAY_MISMATCH` or `IMPORT_PROVENANCE_BLOCKED`; stop.
