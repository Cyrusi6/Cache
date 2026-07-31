# FPCT-E1 A5R4 Inherited-Setgid Directory-Mode Amendment

> - Decision date: `2026-07-31` (`Asia/Shanghai`)
> - User decision, verbatim: `可以`
> - Approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R4_SETGID_MODE_PREDICATE`
> - Protocol ID: `fpct_e1_mechanism_audit_v11_a5r4_setgid_mode_predicate`
> - Base commit: `5f2e41148d9b923e1264e83d76106389c155a596`
> - Status: `APPROVED_PROSPECTIVE_AMENDMENT / PRE-SUCCESSOR-NATURAL-ACCESS`
> - Sole changed dimension: `OWNER_CREATED_STAGING_DIRECTORY_INHERITED_SETGID_MODE_PREDICATE`
> - E1-2 and all model/GPU/training stages: `NOT AUTHORIZED`

## 1. Prospective decision and narrow purpose

After the A5R3 execution `3263531ed7137efd241de3c951bf7c5e9d37e669`
had been terminally closed, committed, pushed, and reported, the user replied
`可以`. No successor natural row was read before that reply. This decision
prospectively authorizes exactly one operational change: an owner-created
choice-audit staging directory may retain filesystem-inherited setgid when all
ordinary group and other permission bits remain zero.

The failed A5R3 execution established that `/netdisk` created the fixed staging
directory as mode `02700` even though the process requested `0700`. The
directory was owned by the executing UID, had group/other permission bits
`000`, and was on the intended filesystem. The exact-`0700` implementation
therefore failed closed after the complete 326-row label-free reduction and
before final-directory or success-receipt publication. A5R4 changes only that
directory-mode predicate. It neither accepts nor reuses the unpublished
reduction.

The following remain unchanged:

- A5R2 v9 population, ordering, parser, cardinality rules, taxonomy,
  correction actions, thresholds, payload schemas, and mechanical decision;
- prompt, tokenizer, alignment, split, and data/source bindings;
- A5R3 v10 claim and receipt envelopes, basenames, exclusive-claim ownership,
  single ordinary same-filesystem rename, fsync order, verification, crash,
  no-resume, no-reuse, and durable commit-point semantics;
- FPCT operators, endpoints, estimands, hypotheses, and claim boundaries.

A5R4 is a v11 operational overlay over unchanged v9 scientific semantics and
the unchanged v10 A5R3 publication algorithm. It is not a scientific amendment.

## 2. Immutable failed execution and predecessor bindings

The immediately preceding execution is permanently closed:

```text
execution_sha = 3263531ed7137efd241de3c951bf7c5e9d37e669
run_uid        = fpct-e1-a5r2-choice-cardinality-3263531e-v1
run_root       = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-3263531e-v1
closure        = recipe/eval_recipe/fpct_e1/executions/3263531e/input_lock_failure_receipt.json
closure_sha256 = 8ca3537914edb880b3436012841ec47b8d416345e2646fce22597c60d3bf4d17
```

The closure records 326 natural E0-design rows reduced only in memory,
`unpublished_natural_choice_statistics_human_or_reviewer_accessed=false`, a
durable claim, staging mode `02700`, no final directory, and no publication
receipt. That UID, root, source snapshot, claim, staging directory, in-memory
state, and any other artifact are forensic evidence only. They must not be
cleaned, resumed, repaired, read for unpublished statistics, or reused.

The following A5R3 normative objects are immutable predecessors:

| Object | SHA256 |
|---|---|
| `FPCT_E1_A5R3_PORTABLE_PUBLICATION_AMENDMENT.md` | `ccfb73bdfd396d5941e975b09143450189e75fffd4884482ce613cf03feb6cd4` |
| `recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_contract.json` | `2ba72cfbccb5a1b7c9f46c02510f4763a2917210d3884c7b5d8223580db9e272` |
| `recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_schema.json` | `b24d197481c01c917ad9d4063441f96ac23979bdb4b3db24855e551c7fb91460` |
| `recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_synthetic_gate.json` | `004a8feb272af78c439f970b1ae637400e641a4fd93a412f29832f288871a4c2` |

The last gate additionally binds evidence SHA256
`907f3b6a631ed8a1c6bb13cf559097fc96fc9b5a2b2148ebf0c8eabcaeca63c9`,
execution-tree SHA256
`54f108bb07f3d10aa759e4eb93ee011915e5b93e91b864ae32da6923a06b1369`,
and `43 passed / 0 failed` synthetic tests. A5R4 must verify these exact bytes
before accepting their inherited evidence.

## 3. Frozen directory-mode predicate

The process continues to request mode `0700` when exclusively creating the
fixed staging directory. Immediately after creation, the staging entry is
accepted if and only if all of the following hold:

1. it is a real directory and not a symlink;
2. its `st_uid` equals the effective UID of the executing publisher;
3. its owner permission bits are exactly `0700`;
4. every ordinary group and other permission bit is zero, equivalently
   `S_IMODE(st_mode) & 0077 == 0`;
5. neither setuid nor sticky is present;
6. the only permitted optional special bit is setgid, so the complete accepted
   permission modes are exactly `00700` and `02700`;
7. it is on the same device as the fresh run root.

Mode `02700` has the additional inherited-setgid requirements:

- before child creation, the real non-symlink parent/run root is owned by the
  executing UID and itself has setgid;
- the child `st_gid` equals the parent's `st_gid`;
- parent and child are on the same device;
- the observation is made immediately after `mkdir(mode=0700)`, before any
  child artifact is written.

No `chmod`, mode normalization, retry, alternate staging name, fallback mode,
or relaxation after an observed natural result is allowed. Mode `0700` remains
valid. Modes with setuid, sticky, missing owner permissions, any group/other
permission, wrong owner, wrong inherited group, symlink/non-directory identity,
or device mismatch remain terminal failures.

The predicate is rechecked before staged writes, immediately before rename,
and after rename on the resulting final-directory inode. The rename must
preserve the accepted inode, device, UID, GID, and mode. Read-only completed
verification must apply the same predicate. Claim and publication-receipt files
remain exact owner-only regular files of mode `0600`; A5R4 does not broaden any
file permission rule. The A5R3 run-root rule also remains unchanged: it must be
a real directory owned by the executing UID and may not be group/other
writable.

## 4. A5R3 publication algorithm remains normative

A5R4 does not create a new publication envelope. Claim and receipt continue to
use:

```text
schema_version = 10
protocol_id    = fpct_e1_mechanism_audit_v10_a5r3_portable_publication
success_status = A5R3_CHOICE_AUDIT_PUBLICATION_GO
```

The exact A5R3 claim, fixed staging, final, receipt, three canonical artifact
names, operation order, exclusive single-compliant-writer premise, exactly-one
ordinary rename, no-overwrite property, fsync boundaries, tamper checks,
read-only loser/re-entry behavior, and terminal crash states remain operative.
Only exact claim + final three artifacts + durable receipt is GO. A Python
return remains non-evidence.

All A5R3 publication functions outside the complete directory-mode enforcement
closure must retain their predecessor AST projections. The v11 gate must bind
the new predicate, directory-token helper, staging creation, staged/final
verification, rename path, completed verifier, their single audit call site,
and the six required active-gate/source-binding plumbing functions as one
explicit changed AST closure. The sealed-prepare source/module closure must
name both the immutable A5R3 verifier and active A5R4 verifier. This prevents an
unrelated publication or scientific change from being hidden inside the
amendment while allowing the same predicate and inode token to be enforced at
every frozen boundary.

## 5. Mandatory CPU/offline synthetic gate

Before any successor natural row is read, the v11 gate must validate at least:

- strict A5R4 contract/schema and exact A5R3 four-object, gate-evidence,
  gate-tree, and `3263531` closure SHA bindings;
- exact acceptance of `00700` and properly inherited `02700`;
- rejection of setgid without a setgid parent, parent/child GID mismatch,
  setuid, sticky, incomplete owner permissions, every group/other permission,
  wrong UID, symlink, non-directory, and device mismatch;
- a synthetic `0700` directory path and an inherited-`02700` directory path;
- a scratch regression on `/netdisk/lijunsi/fpct-e1` that records the real
  parent, staging, and renamed-final UID/GID/device/inode/mode, verifies
  setgid inheritance and byte-preserving ordinary rename, and removes only its
  synthetic scratch fixture after evidence is recorded;
- mode/inode preservation across rename and the same checks in completed
  read-only verification;
- unchanged A5R3 claim/receipt v10 schema and status;
- all inherited A5R3 concurrency, tamper, crash injection, loser, re-entry,
  no-overwrite, fsync, and receipt cross-binding tests;
- static absence of `renameat2`, `os.replace`, copy/delete, cleanup of failed
  production roots, and any mode-normalizing `chmod` in the A5R4 transaction;
- zero natural-row, tokenizer, alignment, model/checkpoint, forward, CUDA/GPU,
  Kubernetes, training, E1-pilot, confirmatory, E1-2, or E1-3 access.

The target-filesystem probe is synthetic infrastructure evidence, not a natural
audit. Gate failure is `A5R4_PRE_NATURAL_GATE_BLOCKED`; it authorizes no
successor natural access. Gate GO is
`GO_PRE_NATURAL_A5R4_SETGID_MODE_HARD_GATE` and authorizes only the fresh CPU
execution described below.

## 6. Fresh execution and stopping boundary

Only after the complete A5R4 amendment, contract, schema, implementation,
tests, and gate have been committed and pushed, and the worktree is clean with
local/upstream SHA equality, may a completely fresh immutable snapshot, UID,
and empty root be created:

```text
run_uid  = fpct-e1-a5r2-choice-cardinality-<execution_sha8>-v1
run_root = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-<execution_sha8>-v1
```

The successor must recompute all frozen source/data bindings, acquire and
durably verify a fresh claim before natural row 1, and rerun the complete
326-row label-free audit from group 1. It may not use the `3263531` claim,
staging directory, cursor, snapshot, reduction, or observation. Only if both
the unchanged v9 mechanical audit and unchanged A5R3 publication are GO may
the already-frozen conditional CPU input lock run from group 1.

Any failure is terminal for that new UID/root. Even after input-lock GO, A5R4
authorizes only provenance reporting and human review. E1-2, E1-3, E1-pilot,
confirmatory access, model/checkpoint loading, model forward, GPU/CUDA,
Kubernetes, training, and scientific claims remain unauthorized.
