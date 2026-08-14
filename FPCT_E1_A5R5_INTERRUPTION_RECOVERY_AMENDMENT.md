# FPCT-E1 A5R5 Fresh-Root Interruption-Recovery Amendment

> - Decision date: `2026-08-14` (`Asia/Shanghai`)
> - User authorization: `PLEASE IMPLEMENT THIS PLAN`
> - Approval ID: `APPROVED_FPCT_E1_ROOT_CAUSE_AND_SINGLE_FACTOR_RECOVERY`
> - Protocol ID: `fpct_e1_mechanism_audit_v12_a5r5_fresh_root_recovery`
> - Base commit: `a47d52f8adf5ada79919d39ac42d7b9fe655595f`
> - Status: `APPROVED_PROSPECTIVE_AMENDMENT / PRE-SUCCESSOR-NATURAL-ACCESS`

## 1. Purpose and scientific boundary

A5R5 closes an externally interrupted A5R4 input-lock execution and authorizes
one entirely fresh execution. It changes process supervision only. It does not
change the E0-design population, prompts, tokenizer/alignment, candidate
sanitizer, `A`, top-k, thresholds, operators, checkpoints, endpoints or claims.

The successor must begin at group 1 from a clean pushed commit and a new Git
archive, UID and empty root. The old root is forensic evidence only. No file,
cursor, choice-audit result, geometry row, template chunk or observation from
that root may be consumed by the successor.

## 2. Immutable interrupted execution

```text
execution_sha = a47d52f8adf5ada79919d39ac42d7b9fe655595f
run_uid        = fpct-e1-a5r2-choice-cardinality-a47d52f8-v1
run_root       = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-a47d52f8-v1
tree_sha256    = 302762365c964e93a3f3fe3ad063447998b51207f6f9ab57b0594c4132184187
file_count     = 708
total_bytes    = 39278548
inventory      = 812 entries / c480ecfa6b508fa1cad028ae54dc18b64f18295e0ce4239a598386b36131ebdc
```

The retained A5R3 publication is a valid completed choice-audit publication,
but the downstream input lock is incomplete: 23 row-template files and the
geometry triplet exist, while the canonical sidecar, top-level manifest,
`A5R2_INPUT_LOCK_GO.json` and `A5R2_INPUT_LOCK_BLOCKED.json` do not. No process
is running. The external termination cause itself is not verified. This is not
a scientific result and does not authorize reading or
reusing the partial downstream state.

The entire root, source snapshot and controller history are permanently
`NO_RESUME / NO_REPAIR / NO_REUSE / NO_CLEANUP`.

## 3. Detached one-shot controller

The A5R5 controller is a supervision wrapper around the unchanged sealed
`fpct_e1_prepare_input_lock.py` command. It must:

- verify branch, clean worktree and local/upstream SHA equality;
- verify the exact old-root hash-only forensic closure;
- derive a new UID/root from the new execution SHA and require absence;
- materialize a new Git archive snapshot and independently verify its receipt;
- create a controller-state directory outside the scientific run root;
- launch exactly one detached worker with CPU/offline environment variables;
- publish a durable no-overwrite materialization claim before creating the run
  root, so even a crash during snapshot creation consumes that identity;
- acquire a durable `O_EXCL` launch claim before creating the worker;
- reject direct/repeated worker entry with an independently bound `O_EXCL`
  worker-start claim;
- record immutable command, PID, timestamps and hashes in the dedicated hidden
  sibling controller root, never in the scientific run root;
- bind liveness to Linux process start-time plus exact command-line hash, so a
  recycled PID cannot be mistaken for the worker;
- never relaunch, resume or repair a root after a worker has been launched;
- classify a missing terminal result after process death as terminal
  interruption requiring another prospective commit/root;
- recognize GO only after process exit 0 plus a second, snapshot-executed deep
  replay of the complete input-lock producer/consumer graph (identity, assets,
  choice audit, census, geometry, every template chunk, global index/receipt,
  sidecar, manifest and GO receipt). A root containing temp or unbound entries
  fails before the inherited verifier is invoked.

Logs and controller state are operational evidence. They may not be placed in
the scientific input-lock directory or used as scientific inputs.

## 4. Ordered authorization

1. Freeze this amendment, manifest, controller and tests before successor
   natural access.
2. Run CPU/offline synthetic and forensic checks.
3. Commit and push; require clean local/upstream equality.
4. Create a fresh snapshot/UID/root.
5. Launch the one-shot CPU input lock from group 1.
6. If and only if input-lock GO and independent provenance verification pass,
   freeze the separate E1-2/E1-3 diagnostic/intervention execution lock before
   any pretrained output.

The user's 2026-08-14 authorization conditionally permits E1-2/E1-3, selection
of exactly one factor, its implementation and the matched E1-pilot described
in the approved plan. It does not permit model-selection/test, confirmatory
runs, native null, selector, new gate, cross-model expansion or 36-run formal
confirmation.

## 5. Hard stops

Any source/upstream mismatch, old-root fingerprint change, existing successor
root, snapshot mismatch, nonzero worker exit, missing terminal artifact,
identity mismatch, unexpected root entry, integrity failure, model access
before the diagnostic lock, or access to E1-pilot before operator freeze is a
hard stop. A failed successor root is never retried.

## 6. Claim boundary

A5R5 GO proves only that the pre-output recovery protocol and one-shot
supervision are complete. Input-lock GO is provenance evidence, not mechanism
or performance evidence. The negative E0 result remains frozen and unchanged.
