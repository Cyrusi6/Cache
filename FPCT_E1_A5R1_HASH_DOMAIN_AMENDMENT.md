# FPCT-E1 A5R1 Hash-Domain Recovery Amendment

> - Decision date: `2026-07-30` (`Asia/Shanghai`)
> - User decision: `可以`
> - Recorded approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`
> - Status: `APPROVED_PROSPECTIVE_AMENDMENT / PRE-NATURAL_ONLY`
> - Protocol ID: `fpct_e1_mechanism_audit_v8_a5r1_hash_domains`
> - Scientific input, population, operator, estimand, threshold, and split: `UNCHANGED`
> - E1-2, E1-3, E1-pilot, confirmatory, model forward, GPU, and training: `NOT AUTHORIZED`

## 1. Prospective decision and scope

After the A5 failure closure was presented as
`A5_INPUT_LOCK_BLOCKED -> A5R1 HUMAN APPROVAL REQUIRED`, the user replied
`可以` on 2026-07-30. This reply is prospectively recorded under the stable
decision identifier
`APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`. It authorizes only the
hash-domain repair, its no-model synthetic qualification, and a new CPU-only
input-lock attempt from E0-design group 1.

This approval does not reinterpret the blocked A5 run and does not authorize
E1-2 or any scientific model output. A successful A5R1 input lock is provenance
evidence only.

## 2. Immutable predecessor boundary

The A5 v7 protocol and the blocked execution remain immutable historical
evidence. Their bytes must not be overwritten or silently superseded in place:

| Historical object | SHA256 / identity |
|---|---|
| `FPCT_E1_A5_PRODUCTION_PROMPT_AMENDMENT.md` | `7178f2ee7c5d226b69b88726ee645aaf7d9425b150764393690136da2c62e874` |
| `recipe/eval_recipe/fpct_e1/e1_a5_prompt_contract.json` | `aba2a439ce56cd68bc7a614075f7efac1a5da36a77d7ae87b0ee5bf8c493db7f` |
| `recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json` | `1a5e139592b8afc804c730e028660b55b6df7a1dfadc864bc797eef84ca388bb` |
| `recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json` | `464e646c336d33c97a03c603d283a95b1579508d63ae0f19d7c700ccf2dcfc02` |
| `FPCT_E1_A5_DATA_HASH_RECOVERY_ADDENDUM.md` (never-operative proposal) | `4de16ef383c608c528003634e1cf0c16c9100f0bab1d32cb3868c83439ef3ce4` |
| A5 v7 protocol ID | `fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt` |
| blocked execution SHA | `9b248d2094b684f5d9e9a218919a354b7d97468e` |
| blocked native receipt SHA256 | `fe305b2c9d881b202f4a096646e9530e8d630a6fcfa095754d3101450ad7ab79` |
| committed failure closure SHA256 | `76e4dafcb1bf7673a3a4d1cbd124ba4099f8de8bca6af411cdbb9f83b32e3da4` |

The abandoned run root
`/netdisk/lijunsi/fpct-e1/fpct-e1-a5-9b248d20-v1` is permanently
non-resumable and non-reusable. No data byte, sidecar, receipt, tokenizer
object, intermediate state, or artifact from that root may enter A5R1.

The v7 synthetic gate remains valid only as historical predecessor evidence
bound to the blocked v7 source tree. It must not be re-verified as the GO gate
for the changed A5R1 tree.

## 3. Root cause and two disjoint hash domains

The blocked run compared two correct hashes that had different algorithms. The
same 51 files / 233,569 bytes produced:

- E0 declared identity:
  `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`;
- A5 generic canonical-manifest identity:
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`.

No data drift or corruption was found. A5R1 freezes the following two disjoint
domains.

### 3.1 E0 declared identity domain

- Domain ID: `e0_declared_tree_v1`.
- Algorithm ID: `relative_path_nul_file_sha256_bytes_v1`.
- Exact implementation:
  `script.experiment.fpct_e1_capture_runner._e0_declared_tree_sha256`.
- Construction: iterate files in lexicographic POSIX-relative-path order and
  update SHA256 with `UTF8(relative_path) || NUL || raw_32_byte_file_SHA256`.
- Frozen expected value:
  `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`.
- Sole normative role: equality with the E0-declared development-tree identity.

### 3.2 A5 generic asset-manifest domain

- Domain ID: `a5_generic_asset_tree_v1`.
- Algorithm ID: `canonical_json_file_manifest_v1`.
- Exact implementation:
  `script.experiment.fpct_e1_prepare_input_lock._generic_asset_tree`.
- Construction: bind every file's relative path, kind, symlink target, byte
  count, and file SHA256 in a canonical JSON file manifest, then SHA256 the
  canonical manifest.
- Previously observed same-domain value:
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`.
- Sole normative roles: A5R1 before/after input equality, same-domain
  predecessor corroboration, and tamper detection.

The two domains must be represented as separate objects carrying their domain
ID, algorithm ID, implementation symbol, and SHA256. An unqualified
`tree_sha256` is insufficient provenance. No comparison may use a SHA from one
domain as the expected value for the other domain. Cross-domain equality is
neither required nor meaningful.

## 4. A5R1 versioned protocol and artifacts

A5R1 is a new version, not an edit of v7:

- protocol ID: `fpct_e1_mechanism_audit_v8_a5r1_hash_domains`;
- contract:
  `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_contract.json`;
- schema:
  `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_schema.json`;
- future pre-natural gate:
  `recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json`;
- input-lock artifact protocol:
  `fpct_e1_e0_design_input_lock_v4_a5r1_hash_domains`.

The v8 contract incorporates the unchanged v7 prompt/population contract by
exact SHA, rather than rewriting the selected production-runtime prompt. The
v8 schema must require both named hash-domain records in the compact sidecar,
strict input-lock manifest, GO receipt, and completed-verifier evidence. A
blocked receipt must bind both complete domain records if the values were
computed or otherwise available before the failure. A failure that occurs
before hash-domain computation may omit both records; it may never publish only
one partial domain record.

## 5. Fresh pre-natural gate and SHA closure

No E0-design group may be loaded, rendered, tokenized, or aligned until a new
transactional v8 gate is GO. The gate must be published at the new A5R1 path and
must not overwrite the v7 gate. It must bind:

1. the immutable v7 predecessor hashes in Section 2;
2. this amendment, the v8 contract, and the v8 schema;
3. every code path used to compute either hash domain, prepare/verify the input
   lock, seal imports, create the source snapshot, and stream artifacts;
4. all synthetic and regression tests used for the decision;
5. the three existing A5 instrumentation artifacts as immutable no-model
   predecessor evidence;
6. a complete repo-relative `tracked_files_sha256` map;
7. `execution_tree_sha256 = SHA256(canonical_json(tracked_files_sha256))`;
8. exact environment identity, test-output SHA256, test counts, streaming
   stress evidence, and a canonical `evidence_sha256`.

The new synthetic gate must verify, without natural data:

- each algorithm against an independent synthetic reference oracle;
- deliberate divergence of the two domain hashes on a fixture without treating
  it as a failure;
- correct same-domain comparisons;
- fail-closed rejection of swapped, missing, mislabelled, or unqualified hashes;
- add/remove/rename/byte-change/symlink tamper detection;
- before/after generic-manifest equality;
- schema rejection when either domain record is absent;
- v7 four-object SHA immutability;
- blocked execution/root non-reuse;
- atomic failure with no usable GO artifact.

The v7 gate is recorded as
`historical_a5_gate_verified_against_current_tree=false`. Passing v7 tests or
rehashing the v7 gate cannot substitute for the v8 gate.

## 6. Commit, snapshot, UID, root, and group-1 order

The only permitted order is:

1. implement the approved v8 contract/code/tests without natural E0-design
   access;
2. build and independently verify the new v8 no-model synthetic gate;
3. confirm all v7 predecessor bytes and SHA256 values are unchanged;
4. run JSON/schema/static checks and `git diff --check`;
5. commit and push the complete v8 source and gate on
   `research/fpct-e1-mechanism-audit`;
6. verify clean worktree and exact local/upstream equality;
7. create a new immutable source snapshot from that exact commit and verify its
   tree and receipt;
8. create a new UID matching
   `fpct-e1-a5r1-hash-domains-<execution_sha8>-v1`;
9. create a new, initially empty root matching
   `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-<execution_sha8>-v1`;
10. under sealed CPU/offline bootstrap, compute and attest both hash domains
    before group 1, including E0-declared equality and generic same-domain
    predecessor equality;
11. restart the complete 326-group census at group 1, with no resume cursor;
12. recompute both domains after the census, require the generic before/after
    records to be exactly equal, and run an independent complete verifier;
13. publish `A5_INPUT_LOCK_GO` only after every check succeeds.

No code, test, contract, schema, threshold, input, or gate byte may change after
the v8 commit. If a change is required, the attempt becomes
`A5_INPUT_LOCK_BLOCKED`, its root is abandoned, and a new prospective version is
required.

## 7. Input-lock hard gate

In addition to all unchanged A5 v7 prompt, population, tokenizer, alignment,
streaming, atomicity, and firewall checks, A5R1 requires:

```text
e0_declared_domain_present = true
e0_declared_algorithm_exact = true
e0_declared_sha_matches_frozen_e0 = true
a5_generic_domain_present = true
a5_generic_algorithm_exact = true
a5_generic_sha_matches_same_domain_predecessor = true
a5_generic_before_equals_after = true
completed_verifier_recomputes_both_domains = true
cross_domain_comparison_detected = false
old_v7_artifact_modified = false
blocked_execution_artifact_reused = false
```

Missing, ambiguous, swapped, non-recomputable, or cross-compared domain records
are integrity failures. Any failure publishes only
`A5_INPUT_LOCK_BLOCKED`, sets `resume_allowed=false` and
`artifact_reuse_allowed=false`, and authorizes no scientific downstream stage.

## 8. Authorization and claim boundary

This amendment authorizes only:

- v8 protocol/schema/contract and implementation work;
- CPU-only, offline, no-model synthetic qualification;
- a new immutable commit/snapshot/UID/root;
- one fresh CPU tokenizer/alignment input-lock attempt beginning at group 1;
- input-lock verification and provenance reporting.

It does not authorize model or checkpoint load, HF/LLM forward, GPU, CUDA,
Kubernetes, training, E1-2, E1-3, E1-pilot, confirmatory data, or a scientific
mechanism/performance claim. `A5_INPUT_LOCK_GO` would establish only that the
frozen E0-design inputs are correctly identified and reproducibly materialized.
E1-2 remains `NOT AUTHORIZED` until a separate prospective decision is recorded.
