# FPCT-E1 A5R3 Portable Atomic Publication Amendment

> - Decision date: `2026-07-31` (`Asia/Shanghai`)
> - User decision, verbatim: `可以`
> - Approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R3_PORTABLE_PUBLICATION`
> - Protocol ID: `fpct_e1_mechanism_audit_v10_a5r3_portable_publication`
> - Base commit: `55476286c84abb6925d9c11a6cffd6ef80a6a58f`
> - Status: `APPROVED_PROSPECTIVE_AMENDMENT / PRE-NATURAL_ONLY`
> - Sole changed dimension: `CHOICE_AUDIT_DIRECTORY_PUBLICATION_PRIMITIVE`
> - E1-2 and all model/GPU/training stages: `NOT AUTHORIZED`

## 1. Prospective decision and narrow purpose

After the A5R2 terminal filesystem closure had been persisted, committed,
pushed, and reported, the user replied `可以`. No successor natural-population
scan occurred before that reply. This prospectively authorizes A5R3 to replace
only the filesystem primitive used to publish the already-frozen A5R2 choice
audit directory.

A5R2 attempted `renameat2(RENAME_NOREPLACE)` on `/netdisk`; the filesystem
returned `EINVAL` after all 326 label-free rows had been reduced in memory and
before any audit ledger, summary, or lock was published. A5R3 uses an exclusive
claim plus ordinary same-filesystem `rename(2)` protocol. It does not reinterpret
the unpublished reduction and may not reuse it.

The following remain byte-for-byte and semantically unchanged:

- the 326-group E0-design population and ordering (ARC 128, OpenBookQA 70,
  MMLU-Redux 128), starting again at group 1;
- choice parsing, variable-cardinality rules, root-cause taxonomy, correction
  actions, label-free firewall, GO/BLOCKED criteria, and thresholds;
- prompt construction, tokenizer/alignment configuration, split and hashes;
- FPCT operators, estimands, instrumentation, and scientific hypotheses.

No A5R2 natural in-memory observation may be copied into an A5R3 artifact or
used to alter a rule. This is an operational portability amendment, not a
scientific amendment.

## 2. Immutable failed execution

The A5R2 execution is permanently closed:

```text
execution_sha = e765d493733d9eec94c152506a1c57781e26fb41
run_uid        = fpct-e1-a5r2-choice-cardinality-e765d493-v1
run_root       = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-e765d493-v1
closure        = recipe/eval_recipe/fpct_e1/executions/e765d493/input_lock_failure_receipt.json
closure_sha256 = f6715f9d86b23ccc265f6ead7b24af47acaf1e1ecd9eb14552af7940d68f798c
```

Its closure is historical failure evidence only. The execution, root, UID,
source snapshot, in-memory reduction, receipt, claim, staging paths, and any
partial object are never resumable and never reusable. The A5R2 amendment,
contract, schema, and synthetic gate also remain immutable and independently
govern all unchanged v9 semantics.

The A5R1 execution `37be816ad611b8b0d916bd98c840c5f31efe2b50`
and its root remain permanently non-resumable/non-reusable as inherited by
A5R2.

## 3. Claim ownership record

Publication has exactly these direct children of one fresh run root:

```text
claim   = .choice_audit.publication.claim.json
staging = .choice_audit.staging
final   = choice_audit
receipt = choice_audit_publication_receipt.json
```

The run root must be newly and exclusively created for the A5R3 execution. The
claim is opened with `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0600`. Failure
to provide any flag is a hard error. The claim is a regular file with link
count one and contains canonical JSON. Newline serialization is an
implementation detail and is not part of this amendment's scientific
contract. Its owner fields bind the A5R3 claim protocol ID, execution SHA, run
UID, canonical run root, fixed staging basename, final basename, receipt
basename, and `choice_semantics_version=9`. It is acquired before natural row
1, so it does not and must not predict result-dependent ledger/summary/lock
hashes; those are bound later by the durable success receipt. PID, hostname,
timestamp, or other mutable process properties cannot substitute for these
ownership bindings.
`SHA256(canonical_claim_bytes)` is the sole stable owner identity; no redundant
`owner_id` field is allowed. The creation-time `(st_dev, st_ino, st_uid,
st_mode, st_nlink)` tuple is revalidated only within the publisher process and
is not persisted as a cross-process identity.

The publisher must compare the complete canonical claim bytes and the safe
regular-file properties obtained from no-follow reads after creation, before
staging verification, immediately before rename, and before success-receipt
publication. A
missing, changed, symlinked, non-regular, multiply-linked, or owner-mismatched
claim permanently blocks the root.

## 4. Frozen portable publication algorithm

The fresh run root must be a real directory owned by the executing UID and must
not be group/other writable. Claim, staging, final, and receipt remain direct
children of that root. The run root, claim, staging, and final directory must
be on the same mounted filesystem; path traversal, symlink following,
copy/delete, `shutil.move`, `os.replace`, and `renameat2` are forbidden. This
amendment does not claim that every operation is dirfd-relative; the security
boundary is the fresh owner-controlled run root plus no-follow/O_EXCL claim
and exact artifact verification.

The exact order is:

1. Before natural row 1, verify the canonical fresh run root is a real owner-controlled directory,
   is not a symlink, and contains none of claim, any staging entry, final, or
   receipt.
2. Atomically create the claim with `O_CREAT|O_EXCL|O_NOFOLLOW`; write and
   independently re-read the canonical ownership record; `fsync` the claim FD,
   close it, and `fsync` the run-root directory FD.
3. Re-verify claim ownership. Verify fixed staging, final, and receipt are
   absent with no-follow checks. Only after this durable claim gate may natural
   row 1 be read and the unchanged 326-row audit run.
4. Exclusively create the fixed fresh staging directory inside the same run
   root after the audit reduction is ready to serialize.
   Write the three unchanged canonical artifacts
   `choice_cardinality_audit_rows.jsonl`,
   `choice_cardinality_audit_summary.json`, and
   `choice_cardinality_audit_lock.json`.
5. `fsync` every artifact; `fsync` staging; independently reopen and validate
   the complete 326-row ledger, summary, lock, exact byte sizes, hashes,
   reduction, schema, and mechanical decision. Reject any additional entry.
6. Re-verify the exact claim bytes/file identity, same-device invariant,
   staging identity and contents, and final absence immediately before commit.
7. Call ordinary same-filesystem `rename(staging, final)` exactly once. The
   exclusive claim is the single-compliant-writer serialization boundary, so
   no compliant publisher can create or replace final concurrently.
8. `fsync` the run-root directory. Reopen final without following links and
   independently verify its directory identity and all three artifacts.
9. Re-verify the retained claim. Exclusively create
   `choice_audit_publication_receipt.json` with
   `O_CREAT|O_EXCL|O_NOFOLLOW`. The canonical receipt binds execution SHA, run
   UID/root, success status, retained claim SHA256, and all three final artifact
   SHA256 values. `fsync` the receipt and then the run-root directory.
10. Independently re-read and verify the exact retained claim, final directory,
    three artifacts, and success receipt. The receipt contains
    `publication_commit_point_reached=true`; this durable four-object state is
    `A5R3_CHOICE_AUDIT_PUBLICATION_GO` and may be consumed downstream even if
    the process crashes before returning to Python. A Python return is not
    success evidence.

Ordinary rename is permitted only because the exclusive claim is acquired
before staging exists and every authorized writer must obey this protocol.
Uncooperative mutation of the run root is outside the single-writer premise,
but every observable mutation is a tamper failure; the root must therefore be
access-controlled to the authorized publisher during the transaction.

The final directory is valid input to a read-only downstream consumer only when
the exact retained claim and exact success receipt are also present and all
cross-bindings validate. It is never permission to re-enter or resume the
publisher. A publication invocation that begins with claim, any staging entry,
final, or receipt already present must not mutate any of them. A process that
loses `O_EXCL` claim acquisition is a non-owner: it exits read-only and must not
publish a competing BLOCKED receipt or contaminate the winner. A separate
read-only completed verifier may later classify the durable state.

## 5. Failure and crash semantics

Every owner-side exception, failed check, failed write/fsync/rename/receipt
publication, unexpected entry, device mismatch, or owner mismatch is terminal
for that UID/root. An `O_EXCL` claim loser is not the owner and therefore exits
read-only; it does not classify or mutate the winner's root. The publisher must
not clean up claim, staging, or final after a
transaction begins; retained objects are tombstones and forensic evidence.
No failed root may be retried, resumed, repaired in place, or used as an input
to another run.

Crash injection is frozen at least at these boundaries:

- claim created but not fully written;
- claim written but before claim/root fsync;
- after claim/root fsync;
- after staging creation and after each staged artifact;
- after staged fsync and verification;
- immediately before and immediately after rename;
- after first root fsync;
- immediately before and immediately after receipt creation;
- after receipt fsync but before root fsync;
- before the final independent four-object re-read.

For every injected state, a new publisher invocation on that root must refuse
to proceed and must not overwrite any byte. Final plus retained claim without
the exact success receipt is a terminal crash/BLOCKED state and is not
retroactively accepted as GO. Receipt without an exact retained claim or exact
final artifacts is also BLOCKED. Only the exact final + retained claim +
success receipt state committed through Step 10 yields GO.

## 6. Mandatory synthetic and tamper gate

Before any new natural row is read, a v10 synthetic gate must validate:

- exact schema/contract and immutable A5R2 artifact/closure SHA bindings;
- ordinary rename success on a synthetic scratch directory on the same target
  mounted filesystem class, with `renameat2`, replace, and copy/delete absent;
- exclusive-claim concurrency: exactly one of concurrent compliant publishers
  acquires the claim and no loser mutates staging/final, writes a competing
  receipt, or publishes BLOCKED;
- refusal for pre-existing regular or symlink claim, staging, final, or receipt;
- rejection of wrong owner bytes, changed claim inode, hard link, non-regular
  claim, owner mutation at every verification boundary, and directory/path
  substitution;
- rejection of staged or final artifact tamper, additional entries, schema,
  hash, byte-count, row-count, reduction, and decision mismatch;
- same-device enforcement and injected rename/fsync/receipt failures;
- every crash point in Section 5, followed by fail-closed re-entry;
- a completed publication's final bytes equal the staged verified bytes, its
  permanent claim/receipt bindings validate, and publisher re-entry is
  read-only and refuses the existing state;
- zero natural-row, tokenizer, alignment, model/checkpoint, forward, CUDA/GPU,
  Kubernetes, training, E1-pilot, confirmatory, E1-2, or E1-3 access.

Synthetic scratch roots are explicitly non-natural test fixtures. They may be
removed only by the test harness after their expected state and hashes have
been recorded; production A5R3 roots are never cleaned or reused.

Gate failure is `A5R3_PRE_NATURAL_GATE_BLOCKED`; it authorizes no natural
access and may not be bypassed by weakening tests or changing filesystem
semantics after observing natural data.

## 7. Fresh execution and natural-access boundary

Only after the complete A5R3 amendment, contract, schema, implementation,
tests, and synthetic gate are committed and pushed, and local/upstream SHA are
equal and clean, may the following fresh identity be created:

```text
run_uid  = fpct-e1-a5r2-choice-cardinality-<execution_sha8>-v1
run_root = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-<execution_sha8>-v1
```

A5R3 v10 is an operational overlay/gate only. Natural ledger, summary, lock,
input-lock artifacts, protocol IDs, schemas, and UID/root formats remain the
unchanged A5R2 v9 contract. A fresh immutable source snapshot and receipt are
required. The run root must
be new and empty. A5R3 must recompute the frozen source/data bindings before
row 1, acquire and durably verify the claim before row 1, and rerun the full
326-row label-free audit from group 1. No artifact or
observation from `37be816a` or `e765d493` may be used. If publication and the
unchanged A5R2 mechanical audit are GO, the already-frozen conditional CPU
input lock may run from group 1. Any failure produces a terminal blocked
receipt and stops.

Even after input-lock GO, A5R3 authorizes only provenance reporting and human
review. E1-2, E1-3, E1-pilot/confirmatory access, model/checkpoint loading,
model forward, GPU/CUDA/Kubernetes, training, and any scientific claim remain
unauthorized.
