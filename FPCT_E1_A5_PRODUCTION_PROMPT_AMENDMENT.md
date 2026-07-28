# FPCT-E1 A5 Actual E0 Production Runtime Prompt Amendment

> - Approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5_RUNTIME_PROMPT`
> - Approval date: 2026-07-28 (Asia/Shanghai)
> - Protocol ID: `fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt`
> - Selected contract: `ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT`
> - Rejected as primary: `STRICT_HISTORICAL_PROJECTED_FIRST4_ANCHOR`
> - Current phase: `A5 PROTOCOL / SCHEMA / PRE-NATURAL CPU QUALIFICATION`
> - E1-2 and E1-3: `NOT AUTHORIZED`
> E1-pilot: `SEALED / NOT RENDERED / NOT TOKENIZED / NOT ALIGNED / NOT RUN / NOT READ`

## 1. Prospective decision

The operative prompt and alignment anchor for the E0-design mechanism audit
shall be the prompt actually produced from the complete materialized E0 row by
the exact E0 production evaluator, prompt builder, tokenizers, chat templates,
and evaluation settings. The historical first-four projection remains
immutable, but only as the selection, content-group identity, and provenance
anchor.

This is an input-provenance contract correction from a historical projected
anchor to the actual production-runtime anchor. It is not an A4
representation-only amendment. It does not retroactively alter E0 and does not
authorize a first-four counterfactual prompt experiment.

The discovery is classified as
`HISTORICAL_PROMPT_ANCHOR_UNDERSPECIFICATION`, not operator failure, streaming
failure, dataset corruption, mechanism evidence, or accuracy invalidation. The
existing `E0_NO_GO_FOR_FURTHER_SPEND` aggregate and its claim boundary remain
unchanged. No assertion is made about accuracy under the unrun first-four
counterfactual.

## 2. Permanent disposition of A4 execution 07755a40

Execution `07755a4039e89700e59af9e141026a57142f9da0` remains:

```text
A4_INPUT_LOCK_BLOCKED
resume_allowed = false
artifact_reuse_allowed = false
scientific_result = false
```

Its source snapshot, execution identity, blocked receipt, partial progress,
temporary chunks, and every run-root artifact are historical evidence only.
They may not be copied, resumed, linked, imported, or used as the starting
state of A5. A5 must restart at the first E0-design group under a new clean
pushed commit, immutable source snapshot, UID, and empty run root.

The A4 synthetic gate and A4 instrumentation artifacts are immutable historical
evidence bound to the `07755a40` source tree. A5 code changes necessarily make
the old tracked-tree verifier inapplicable to the successor tree. A5 therefore
requires new, versioned evidence paths and must not overwrite or relabel the A4
files:

```text
recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json
recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a5.json
recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a5.json
recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a5.json
```

## 3. Dual-anchor contract

Every E0-design group has two explicitly named anchors.

### 3.1 `historical_projection_anchor`

This anchor contains the question and the first four choices in their frozen
historical order. It preserves:

- E0-design membership and task membership;
- `content_group_sha256` and `sample_key_sha256` identity;
- the historical support-selection projection;
- historical `rendered_prompt_sha256` and `alignment_sha256` as diagnostic
  provenance.

It is not the E1-2/E1-3 runtime input and cannot be substituted for the full
production prompt.

### 3.2 `production_runtime_anchor`

This anchor contains the exact materialized E0 development row, including the
complete production choice list in its original order. It is rendered by the
exact E0 evaluator/prompt-builder oracle and then passed through the exact E0
tokenizer/chat-template and frozen alignment configuration. Its prompt,
tokenization, alignment, certified-parent geometry, topology, and logical-row
universe are operative for any future E1-2/E1-3 execution.

The following remain unchanged by A5:

- all 326 E0-design group memberships: ARC 128, OpenBookQA 70,
  MMLU-Redux 128;
- content-group/sample/source-row/task identities and gold answers;
- six E0 checkpoint trees and the C_post/F operator cells;
- centered-lambda grid `{0, 0.25, 0.5, 1, 2}`;
- teacher-forced estimands, accuracy/flip definitions, group/task weights;
- the complete A4 logical-row semantics and bounded 4096-row physical
  streaming contract;
- E1-pilot and confirmatory seals.

The following must be freshly derived and frozen from the production anchor:

- rendered prompt bytes and SHA256;
- sender/receiver token IDs, offsets, spans, and alignment SHA256;
- prompt token lengths, certified-parent counts, topology records;
- logical-row counts, chunk counts, and streaming geometry receipts.

## 4. Exact E0 production renderer identity

“Use the current formatter” is not a valid attestation. A5 must bind and verify
the following before accepting any census or input-lock artifact:

The E0 provenance is a commit chain, not one interchangeable “execution
commit”:

| Layer | Frozen identity and role |
|---|---|
| Scientific operator/image source | `80fb295542ad298fae4cddb1273517b401bbcd17` |
| Initial E0 execution/config lock | `c8751b2a933484ca250b2dcf3f80e233e3809cf6` |
| Corrected dev-tree lock and replacement immutable ConfigMap basis | `ce99abc066ca55d3d6b75f1005e1d4e6188136f3` |
| Final decode-evaluation image/source | `6a51ad4ed1d66067c0ac2d3f2c8c3b5de0f5d2ba` |
| Decode-recovery execution lock | `a1bb56cf0b315fdf4b03d56eab2e5a91560c3907` |
| Final E0 result archive | `613958af38fad27e1ea933ccc0dda6d1af5cce89` |

The prompt-relevant source/config identities are:

| Object | Frozen identity |
|---|---|
| `script/evaluation/unified_evaluator.py` in final eval source | SHA256 `bba1a0d3591a00ce62586f055e92875fb01e84c708c42e922493c3aef91a0371` |
| historical whole `rosetta/utils/evaluate.py` | SHA256 `1e1d06d79769386a63436cf230cbc9c5ef09d9342b7f060b7238e4a57f94d50d` |
| exact `build_prompt` AST source segment | SHA256 `74b7ad7ebfd1df557c3eb4c36075bfe45958c915574a498622f68d287cfb4332` |
| exact `set_default_chat_template` AST source segment | SHA256 `0b5a3e0f9890daf6393f3be73c9b55fc9f93e77ed881c58994523efc9943ccb6` |
| `rosetta/model/aligner.py` in effective E0 source closure | SHA256 `fe77d72fb7103ca103fe87fa602bed545e0caf6fc75b1741b8736b41e9daf7d8` |
| `rosetta/train/dataset_adapters.py` in effective E0 source closure | SHA256 `125c22bd09b2eec0f021f75496ff2a74ef1f795d9b8c0336b70342e6eba08dc0` |
| `rosetta/utils/model_loading.py` in effective E0 source closure | SHA256 `d99b45372f1a756cd4e7bf8443c6ab838102e8dfa89f96bdc3b3910ca6532b3a` |
| `script/experiment/fpct_e0_runner.py` at result execution commit | SHA256 `d3008d18bd19bb13c8bdc5526412c23bbe64dd3ba0a7ffe799f3f928d9d08f0e` |
| rendered config index | SHA256 `555364574fb5f36b472a91ef9ea0bc9534660ff462c97566e793f7997e71b4ae` |
| rendered config bundle | SHA256 `021f09404666e05effe780d32f7b3956e145d440a565ead8c4790ac80a181391` |
| ordered 36 evaluation-config records | SHA256 `24a46571715dd0bdc5ca7953243e7c57556655e1eeed231cb306cb1f8852899a` |
| materialized E0 dev-data tree | SHA256 `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73` |
| Qwen3 tokenizer JSON | SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| TinyLlama tokenizer JSON | SHA256 `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e` |
| TinyLlama tokenizer model | SHA256 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| evaluation settings | `enable_thinking=false`, `use_cot=false`, `use_template=true` |

Additional same-mount corroboration freezes Qwen tokenizer-config
SHA256=`d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`,
bundle SHA256=`d2a315d4ca46d73b53ef973f5eda2561daf90a848e07779fac19ce9761f714be`,
and decoded chat-template `4168` bytes / SHA256=
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`;
TinyLlama tokenizer-config SHA256=
`7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a`,
bundle SHA256=`5a1a4d8005b2377b26f425fc64322ebcb22e898f8d81c603875eec5f8083809b`,
and decoded chat-template `410` bytes / SHA256=
`66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362`.
The tokenizer JSON/model hashes and revisions are `PRE_E0_LOCK`; the additional
config/bundle/template values are `SAME_MOUNT_CORROBORATION`, not retroactively
claimed as fields in the original E0 receipt. A5 must re-attest all of them on
the same frozen mount before scanning the 326 rows.

Here “asset tree” means tokenizer/config/chat-template reconstruction assets
only. It excludes `.safetensors`, `.bin`, `.pt`, checkpoint trees, and all model
weight files. This approval does not authorize reading or hashing a checkpoint
or model-weight tree.

Tokenizer-directory trees, tokenizer files not individually enumerated above,
chat-template bytes, and their SHA256 values must be resolved and frozen by the
new pre-model A5 execution receipt. Missing or ambiguous identity is a hard
failure; it must not be guessed.

The historical E0 source snapshot is the reference renderer oracle. For every
group, A5 must require exact byte equality:

```text
historical E0 renderer(materialized full row)
    == A5 production renderer(materialized full row)
```

Source-code similarity or manually inspected logical equivalence is
insufficient. The pre-natural source/config check produces
`renderer_source_identity_attested`; only exact replay over all 326 production
rows may produce `production_renderer_exactly_attested`. Neither flag may be
pre-filled from historical aggregate evidence. Any renderer, prompt-builder,
template, setting, or byte mismatch causes `A5_INPUT_LOCK_BLOCKED`.

## 5. Full-population no-model census

The census population is exactly 326 distinct E0-design content groups, in the
existing frozen order and task counts. It may perform dataset lookup, prompt
rendering, CPU tokenization/alignment, certified-topology construction, hashing,
and bounded streaming geometry only. It must not load a model or projector
checkpoint or perform a model forward.

Each record must contain at least:

```text
task
content_group_sha256
sample_key_sha256
source_row_id
historical_choice_count
production_choice_count
raw_choice_labels
gold_answer
historical_first4_question_choices_sha256
historical_rendered_prompt_sha256
historical_alignment_sha256
raw_full_row_sha256
production_rendered_prompt_sha256
production_alignment_sha256
historical_prompt_token_count
production_prompt_token_count
production_certified_parent_count
production_logical_row_count
production_physical_chunk_count
prompt_relation
choice_difference_only
```

`prompt_relation` is a closed enum:

```text
EXACT_HISTORICAL_AND_PRODUCTION_MATCH
EXTRA_CHOICES_ONLY
```

The classifier is deterministic and fail-closed:

- `EXACT_HISTORICAL_AND_PRODUCTION_MATCH` requires exact equality of the raw
  question, full choice count/order/text, formatter output bytes, and resulting
  prompt/alignment identity.
- `EXTRA_CHOICES_ONLY` requires exact equality of the question and first four
  choice labels/text/order, production choice count greater than four, and all
  prompt-input differences to be confined to the ordered suffix choices. The
  historical group/sample identities must remain exact.
- `choice_difference_only=true` if and only if the second class is proven.

Any other relation is an integrity failure, including question changes,
first-four text/order changes, instruction/template/whitespace drift, ambiguous
row mapping, gold outside A-D, missing/duplicate group, or an unclassified
difference. An extra-choice text change may leave historical identity unchanged,
but it must change and be captured by the full-row/runtime anchor; it is not a
failure when all first-four invariants still hold.

## 6. Census outputs and reductions

Before any model or checkpoint load, A5 must freeze:

- exact per-task affected and unaffected group counts;
- historical/production choice-count distributions;
- all affected content-group hashes in canonical order;
- a complete one-to-one historical-to-production anchor map;
- production prompt token-length distribution;
- production certified-parent-count distribution;
- production logical-row and physical-chunk distributions;
- exact missing, duplicate, and unexpected-relation counts;
- SHA256 and byte size of every artifact and its canonical semantic stream.

Rows are ordered by `(task canonical order, content_group_sha256,
sample_key_sha256)`, where task order is `ai2-arc`, `openbookqa`,
`mmlu-redux`. Reductions must be deterministic and partition invariant. The
A4 physical cap remains 4096 rows; changing chunk partition may not change a
semantic hash or aggregate.

For every group, mechanically require:

```text
historical_choices[0:4] == production_choices[0:4]
gold_answer in {A, B, C, D}
```

## 7. Pre-natural A5 qualification

A5 cannot cite the old `395 passed` as evidence for its changed input-lock
contract. Before natural E0-design census/input-lock access, a new gate at
`recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json`, produced by
`script/analysis/fpct_e1_a5_prompt_gate.py`, must bind the complete A5 tree and
pass at least:

- exact E0 renderer source/config identity attestation, recorded as
  `renderer_source_identity_attested=true` without consuming natural rows;
- the unchanged A4 streaming synthetic oracle under A5 sources;
- new A5 instrumentation re-attestation in versioned A5 paths, classified
  `NO_MODEL_STATIC_SOURCE_IDENTITY_PLUS_PURE_TENSOR_SYNTHETIC`;
- dual-anchor and full-population fixture tests;
- historical-vs-production classifier tests;
- streaming semantic replay and partition-equivalence tests;
- failure, corruption, atomicity, and resume tests.

Required synthetic cases are:

| Case | Required result |
|---|---|
| Four-choice row | historical and production anchors exact |
| Five-choice row, gold A | production adds E; historical identity unchanged; `EXTRA_CHOICES_ONLY` |
| Five-choice row, gold E | fail closed |
| First-four order changes | fail closed |
| E text changes | historical identity unchanged; full-row/runtime anchor changes and is detected |
| Current formatter differs from historical E0 renderer | fail closed |
| Same production semantics, different chunk partition | identical semantic aggregate |

The A5 gate must be transactional: failure publishes no usable GO receipt. It
must record all test paths/source hashes, historical A4 evidence as immutable
predecessor references, A5 instrumentation evidence hashes, test counts,
the captured test-output SHA256, fresh baseline/million-row streaming records,
estimated physical bytes per row, environment identity, and every required
boolean. Old A4 gate verification against the changed current tree is
forbidden. A5 may reference and rehash immutable A4 real-model parity evidence,
but this approval does not permit re-running an HF/Rosetta model forward. New
A5 evidence must not claim that real-model logits/loss/cache parity was
re-measured; throughout this gate `model_instantiated=false` and
`model_forward_run=false`.

## 8. A5 execution identity and order

After the new pre-natural hard gate is GO, create and push a clean successor
commit. The first execution must use identities of this form:

```text
UID  = fpct-e1-a5-runtime-prompt-<short_sha>-v1
root = /netdisk/lijunsi/fpct-e1/fpct-e1-a5-<short_sha>-v1
```

The root must be new and empty. The execution order is:

```text
new immutable source snapshot and receipt
  -> dual-anchor full-population prompt census
  -> production tokenization/alignment lock
  -> streaming geometry and raw-topology lock
  -> complete independent input-lock verification
  -> A5_INPUT_LOCK_GO receipt
```

It restarts at group 1. It must not resume at group 161 or copy any 07755a40
receipt, identity, chunk, prefix, or progress marker.

## 9. A5 input-lock hard gate

`A5_INPUT_LOCK_GO` requires every following boolean to be true:

```text
historical_projection_anchor_unchanged
e0_design_membership_unchanged
renderer_source_identity_attested
production_renderer_exactly_attested
production_data_tree_exactly_attested
all_326_groups_resolved
all_first4_choices_equal_historical
all_gold_answers_in_A_B_C_D
all_prompt_differences_classified
historical_to_runtime_mapping_complete
production_prompt_replay_equal
production_alignment_replay_equal
choice_order_preserved
raw_full_row_hashes_complete
logical_row_coverage_exact
streaming_semantic_replay_equal
bounded_peak_rss
```

It also requires:

```text
unexpected_prompt_difference_count = 0
missing_rows = 0
duplicate_rows = 0
old_execution_artifact_reused = false
whole_table_materialization_detected = false
model_or_checkpoint_loaded = false
model_forward_run = false
gpu_or_kubernetes_used = false
e1_pilot_consumed = false
confirmatory_consumed = false
```

Any failure produces `A5_INPUT_LOCK_BLOCKED`, invalidates the run root for
resume/reuse, and forbids runtime probe, checkpoint/tree lock, execution plan,
ConfigMap, E1-2, and E1-3.

The successful main JSON is not a legacy A4 or untyped manifest. It must be
validated as `schema_version=7`, protocol v7,
`artifact_type=a5_input_lock_manifest`, and
`status=A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT`. It binds the fresh execution/source
snapshot, 326-row census, production provenance, A4 streaming receipts, every
hard-gate/firewall field, and the compact sidecar by both file and semantic
SHA256. The torch compact sidecar has independent `contract_version=3`; it is
not itself a substitute for the strict JSON manifest.

## 10. Authorization and claim boundary

Authorized by this amendment:

- A5 protocol/schema and dual-anchor implementation;
- new A5 synthetic/instrumentation CPU hard gates;
- a clean pushed research-branch commit and fresh immutable snapshot/run root;
- full 326-group E0-design no-model CPU census and input lock from group 1.

Not authorized:

- resume or reuse of execution `07755a40`;
- model/projector/checkpoint loading or model forward;
- GPU, CUDA, Kubernetes, training, E1-2, or E1-3;
- rendering, tokenizing, aligning, running, or reading E1-pilot;
- confirmatory/model-selection/test access;
- a first-four counterfactual sensitivity run.

Even a complete `A5_INPUT_LOCK_GO` is input/provenance evidence only. It does
not establish mathematical validity, real-data mechanism activation, query-time
separability, task improvement, or a reason to revise the E0 aggregate.
