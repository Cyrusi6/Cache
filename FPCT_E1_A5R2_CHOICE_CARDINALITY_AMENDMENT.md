# FPCT-E1 A5R2 Choice-Cardinality Recovery Amendment

> - Decision date: `2026-07-30` (`Asia/Shanghai`)
> - User decision, verbatim: `批准`
> - Approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R2_CHOICE_CARDINALITY_RECOVERY`
> - Status: `APPROVED_PROSPECTIVE_AMENDMENT / PRE-NATURAL_ONLY`
> - Protocol ID: `fpct_e1_mechanism_audit_v9_a5r2_choice_cardinality`
> - E0-design population, group/sample identities, split, operator, estimand,
>   tokenizer, alignment, and scientific threshold: `UNCHANGED`
> - E1-2, E1-3, E1-pilot, confirmatory data, model/checkpoint load, model
>   forward, GPU, Kubernetes, and training: `NOT AUTHORIZED`

## 1. Prospective decision and sole purpose

After the terminal A5R1 closure was reported, the user replied `批准` before
any successor natural-population scan. This decision authorizes only a new
versioned, label-free audit of choice representation/cardinality, the
mechanical correction paths frozen below, a pre-natural synthetic gate, and—if
and only if that audit is GO—one fresh CPU-only A5R2 input-lock attempt from
E0-design group 1.

A5R2 does not reinterpret A5R1, alter FPCT operators, or authorize mechanism
measurement. Its only question is whether the prior
`a5_materialized_row_has_fewer_than_four_choices` failure is explained by a
valid variable-cardinality ARC row or a representation-only HF materialization
difference, without changing the exact row or exact production prompt.

## 2. Immutable predecessor boundary

The operative starting commit is
`f782128774dd88081194c2873efd707044d1db2c`. The failed A5R1 execution
`37be816ad611b8b0d916bd98c840c5f31efe2b50` and root
`/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-37be816a-v1` are permanently
non-resumable and non-reusable. Its committed closure is:

`recipe/eval_recipe/fpct_e1/executions/37be816a/input_lock_failure_receipt.json`

with SHA256
`ec3ae949b557da957a1a1f295f41b91b8522420445b5479a945b1f59b5de8e8a`.

The scientific population sources are also frozen to their bytes at the
starting commit: `recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json` =
`030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d`
and `recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json` =
`25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a`.
Both must be byte-identical at the pre-natural gate and every natural replay.

The following v8 objects are immutable and must be checked byte-for-byte before
the v9 gate and again before the natural audit:

| v8 object | SHA256 |
|---|---|
| `FPCT_E1_A5R1_HASH_DOMAIN_AMENDMENT.md` | `8eafb29d3d736740730e10505ebd8217e779e5ef4d3a128cfb0db3a10a018068` |
| `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_contract.json` | `643151b67d98c84c1120b52705b0fe837fb4106664f10200ada1a692c744a0b1` |
| `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_schema.json` | `7bd2478f7ef3cfd32e752056cf161b8575b84a1f65088c84a0d2c37aec43704b` |
| `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json` | `a2e53784fa46d2f63963bdb41e327a4e17936cce2bff765802b34f3d87869a9f` |

No v7/v8 file may be overwritten. The v9 contract incorporates the v8
hash-domain and input-lock contract by exact path/SHA. A v9 artifact may not
weaken, rename, or override an inherited v8 check. A5R2 does not require and
must not fabricate a schema-v8 natural GO/BLOCKED receipt: variable-cardinality
natural artifacts use the v9 schema. Instead, every v9 natural receipt carries
an immutable v8 contract/schema/gate binding and attests that all applicable
v8 checks were replayed. If any historical v8 natural receipt is ever cited,
it must first validate independently against the immutable v8 schema; v9
validation is additional, never substitutive.

### 2.1 Immutable inherited streaming resource evidence

The v9 gate and producer must bind, not re-estimate, the following fields from
the immutable v8 gate at
`recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json`, SHA256
`a2e53784fa46d2f63963bdb41e327a4e17936cce2bff765802b34f3d87869a9f`:

- `estimated_physical_bytes_per_row = 4096`;
- `streaming_stress.threshold_bytes = 813060096` and
  `streaming_stress.bounded_peak_rss = true`;
- the complete v8 baseline and 1,000,384-row stress records, including their
  row counts, chunk counts, physical bytes, RSS, maximum row/chunk bytes, and
  semantic/replay SHA256 values;
- all eight v8 `streaming_contract_checks = true`.

These values are immutable predecessor resource evidence. A5R2 may neither
relabel them as a new v9 measurement nor combine them with a different hash or
resource domain. Any missing, changed, partially copied, or independently
re-estimated value fails the v9 pre-natural gate.

## 3. Label-free audit firewall

The A5R2 choice audit may materialize only the already-open 326 E0-design rows
and may access only:

- question text (`question` or `question_stem`);
- choice text and source option labels;
- task, pinned dataset/config/split, frozen evaluation index, and a
  non-outcome native row identifier when one is already defined;
- HF feature/container metadata needed to interpret the choice structure.

At projection entry it must discard, and thereafter neither read nor hash:
answers, answer keys, corrected answers, error type, predictions, correctness,
beneficial/harmful indicators, accuracy, loss, logits, or any model output.
`raw_full_row_sha256` is therefore forbidden in the choice audit. The audit
uses `choice_projection_sha256`, computed only from the allowlisted projection.
Source option labels are structural metadata, not gold labels.
The feature/container SHA is observed structural provenance only; it is not
treated as an independently frozen row identity. Exact row identity/order is
instead re-derived from the frozen split and E0-development manifests and
compared to the loader's task/config/split/index request plus materialized
content hash during both publication and completed verification.

The audit performs no chat-template rendering, tokenization, alignment, model
or checkpoint load, model forward, CUDA/GPU/Kubernetes call, or training. It
may call the frozen pure-string `UnifiedEvaluator.format_example` only to prove
that a canonical choice view is byte-identical to the raw production prompt.

## 4. Frozen choice view and task cardinality

The schema-aware canonical choice view accepts only explicitly supported
representations: mapping-of-sequences, sequence-of-mappings, a plain MMLU
sequence, or a feature-backed sequence conversion whose result is a finite
ordered list. Strings/bytes are never treated as a sequence of choices. Choice
order and text bytes are preserved. No missing item may be invented, no item
may be dropped from the production prompt, and no item may be reordered.

Task validity is frozen prospectively:

- `ai2-arc`: any genuine count `n >= 2` is valid. Counts 2 or 3 form the
  `GENUINE_ARC_LOW_CARDINALITY` stratum and are not errors.
- `openbookqa`: `n = 4` is required.
- `mmlu-redux`: `n = 4` is required.

ARC count 0/1 and non-four-choice OpenBookQA/MMLU-Redux rows are integrity
failures. These are task-format contracts, not data-dependent thresholds.

The frozen historical group identity remains exactly:

```text
question + first min(4,n) choices + empty padding to ten slots
```

For provenance records, `historical_choice_count = min(4,n)` and
`production_choice_count = n`. Padding is only part of the existing hash
formula; it never creates prompt choices. Source option labels are retained as
provenance. Runtime ordinal labels are the labels generated by the unchanged
E0 formatter and are recorded separately.

## 5. Orthogonal states and mutually exclusive taxonomy

Every audited row first receives three orthogonal states:

```text
identity_status       = MATCH | DESCRIPTOR_INDEX_MISMATCH
representation_status = DIRECT | HF_FEATURE_NORMALIZATION_REQUIRED | UNSUPPORTED
cardinality_status    = TASK_VALID | TASK_INVALID
```

It is then assigned exactly one root cause, with fixed precedence:

```text
DESCRIPTOR_INDEX_MISMATCH
UNSUPPORTED_FEATURE_REPRESENTATION
GENUINE_TASK_INVALID_CARDINALITY
HF_FEATURE_NORMALIZATION_ONLY
GENUINE_ARC_LOW_CARDINALITY
DIRECT_VALID
UNEXPLAINED_OTHER
```

Definitions:

- `DESCRIPTOR_INDEX_MISMATCH`: canonical extraction succeeds but the frozen
  first-four content group does not match, or task/config/split/index/native-ID
  provenance contradicts the frozen descriptor. It is never repaired in A5R2.
- `HF_FEATURE_NORMALIZATION_ONLY`: the legacy extractor undercounts or rejects
  a supported HF representation; canonical extraction is unique, matches the
  frozen group, and raw-versus-canonical E0 formatter bytes are identical.
- `GENUINE_ARC_LOW_CARDINALITY`: direct extraction yields exactly two or three
  ARC choices, the frozen group matches, and formatter bytes are unchanged.
- `GENUINE_TASK_INVALID_CARDINALITY`: the canonical count violates Section 4.
- `UNSUPPORTED_FEATURE_REPRESENTATION` and `UNEXPLAINED_OTHER` are hard stops.

Mechanical correction actions are fixed before the audit:

```text
DIRECT_VALID                       -> NONE
GENUINE_ARC_LOW_CARDINALITY        -> VARIABLE_CARDINALITY_CONTRACT_ONLY
HF_FEATURE_NORMALIZATION_ONLY      -> PROVENANCE_CANONICAL_VIEW_ONLY
all other classes                  -> BLOCK
```

`PROVENANCE_CANONICAL_VIEW_ONLY` may change only provenance/census extraction.
The raw row passed to the exact E0 production formatter remains byte-identical.
A5R2 never searches for a replacement row, remaps an index, changes a group or
sample, or rebuilds a split.

A parseable row whose canonical representation is task-invalid is still
emitted into the complete 326-row audit ledger as
`GENUINE_TASK_INVALID_CARDINALITY`; the complete lock is then BLOCKED. In
contrast, an empty, ambiguous, or otherwise unsupported representation that
cannot be interpreted without guessing must not be represented by a fabricated
row. It immediately publishes only terminal `A5R2_INPUT_LOCK_BLOCKED` with
`failure_stage=CHOICE_AUDIT`, the exact failing group ordinal, a null audit-lock
binding, and no published partial ledger/staging artifact. Audit GO always
requires a complete immutable 326-row lock.

A parseable task-invalid row follows a different terminal path: it remains in
the complete 326-row ledger and publishes an `A5R2_CHOICE_AUDIT_BLOCKED` lock,
then the input-lock receipt uses
`failure_stage=CHOICE_AUDIT_COMPLETE_MECHANICAL_BLOCK` with a non-null binding
to that complete lock. This stage must never be conflated with an unparseable
early `CHOICE_AUDIT` failure.

## 6. Audit-first transaction

The only permitted order is:

1. create the versioned amendment, strict contract/schema, implementation, and
   synthetic tests without natural-row access;
2. produce and independently verify a transactional v9 no-natural gate;
3. verify v8 immutability, JSON/schema/static checks, and `git diff --check`;
4. commit/push the complete v9 pre-natural tree and verify clean local/upstream
   equality;
5. create a fresh immutable source snapshot, UID, and initially empty root;
6. before reading audit row 1, recompute both frozen E0 data-tree hash domains,
   bind the exact 51-file/233,569-byte identity, and verify the immutable source
   snapshot receipt plus mounted-tree SHA;
7. run the complete 326-row label-free choice audit in canonical task/group
   order from ordinal 1;
8. write the row ledger, summary, and lock into one unpublished sibling staging
   directory; fsync every file, fsync the staging directory, verify all three
   artifacts there, atomically rename that directory to `choice_audit`, fsync
   the run root, and independently re-read the final artifacts. The canonical
   names are `choice_cardinality_audit_rows.jsonl`,
   `choice_cardinality_audit_summary.json`, and
   `choice_cardinality_audit_lock.json`;
9. make the mechanical GO/BLOCKED decision from the frozen taxonomy only;
10. only after audit GO, run the preimplemented v9 correction/input-lock from
   group 1 while replaying every applicable inherited v8 hard check;
11. verify and publish the v9 natural receipt; stop before E1-2.

Audit ledger rows contain no natural text: only identities, counts, enums,
booleans, and hashes. The ledger itself remains local; a compact lock/result
manifest may be committed. Any code/schema/contract change after the audit
invalidates the execution and requires another version and fresh root.
No partial staging object is a published audit artifact, and no existing final
`choice_audit` directory may be overwritten. The Torch sidecar's sole normative
audit key is exactly `choice_audit`; `choice_cardinality_audit` is forbidden as
an alias.

## 7. Audit GO and hard stops

`A5R2_CHOICE_AUDIT_GO` requires all of the following:

- exactly 326 unique frozen E0-design groups in counts ARC 128, OpenBookQA 70,
  MMLU-Redux 128 and canonical order from group 1;
- all frozen group/sample/split/data/provenance hashes unchanged;
- the dual-domain E0 data-tree binding and source-snapshot binding were
  independently recomputed and verified before audit row 1;
- every observed canonical content hash equals its expected group hash;
- zero descriptor mismatch, unsupported representation, invalid task
  cardinality, and unexplained rows;
- every row eligible for a GO taxonomy class has exact raw/canonical formatter
  byte parity (therefore all 326 rows in a complete GO audit);
- no pseudo choice, reorder, production truncation, replacement, or remap;
- zero forbidden outcome-field access;
- independently recomputed ledger, aggregate, decision, and provenance hashes.

Any violation is `A5R2_CHOICE_AUDIT_BLOCKED`, with `resume_allowed=false` and
`artifact_reuse_allowed=false`. In particular, descriptor/index mismatch must
stop; A5R2 may not search the local dataset for a matching replacement.

`A5R2_INPUT_LOCK_GO` additionally requires an unchanged audit lock, exactly
the predeclared correction mapping, all inherited v8 dual-hash/prompt/
tokenizer/alignment/streaming/atomicity checks, a full restart at group 1, and
independent completed verification. It is provenance evidence, not a
scientific result.

## 8. Fresh execution and authorization boundary

The successor must use:

```text
run_uid  = fpct-e1-a5r2-choice-cardinality-<execution_sha8>-v1
run_root = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-<execution_sha8>-v1
```

The root must be new and empty. No failed A5/A5R1 UID, root, receipt, cursor,
sidecar, or partial artifact may be reused.

Even if `A5R2_INPUT_LOCK_GO` is achieved, the following remain false:

```text
E1-2 authorization
E1-3 authorization
E1-pilot or confirmatory access
model/checkpoint load or model forward
GPU, CUDA, Kubernetes, or training
```

The terminal next state is
`HUMAN_REVIEW_REQUIRED_E1_2_NOT_AUTHORIZED`.
