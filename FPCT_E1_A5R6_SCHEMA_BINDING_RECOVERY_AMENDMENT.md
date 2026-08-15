# FPCT-E1 A5R6 Schema-Binding Recovery Amendment

> - Decision date: `2026-08-15` (`Asia/Shanghai`)
> - User authorization: explicit A5R6 prospective amendment and fresh rerun
> - Protocol ID: `fpct_e1_mechanism_audit_v13_a5r6_schema_binding_recovery`
> - Base commit: `ab4052cd0f97c248e674ed4f7b2e0a4fa23d87f5`
> - Status: `APPROVED / PRE-NATURAL / COMMIT+PUSH PENDING`

## 1. Sole purpose

A5R6 repairs one deterministic producer/schema integration defect discovered
after the A5R5 CPU input lock completed all 326 E0-design groups. It also makes
the failed-root forensic inventory portable across NFS/NSS ID-mapping
namespaces. It changes no population, prompt, tokenizer, alignment, candidate,
prior, operator, checkpoint, endpoint, threshold or claim.

The successor must use a new clean pushed commit, source snapshot, UID, root and
controller root and must restart from group 1. No A5R5 choice-audit file,
geometry, chunk, sidecar, cursor or result may be imported.

## 2. Immutable A5R5 failure

```text
execution_sha = ab4052cd0f97c248e674ed4f7b2e0a4fa23d87f5
run_root       = /netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-ab4052cd-v1
status         = A5R2_INPUT_LOCK_BLOCKED / WORKER_TERMINAL_FAILURE
failure        = V9_INPUT_LOCK / VALUEERROR / oneOf matched 0 branches
```

The natural CPU preprocessing completed 326/326 groups and
30,370,816/30,370,816 logical rows with zero missing/duplicate rows. Geometry
and streaming receipts were GO. The failure occurred before publication of the
canonical input-lock manifest and GO receipt. It is an engineering integration
failure, not a mechanism, accuracy or performance result. The root is
permanently `NO_RESUME / NO_REPAIR / NO_REUSE / NO_CLEANUP`.

## 3. Frozen schema-binding fix

The immutable v9 schema remains byte-identical:

```text
recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json
SHA256 = 9cb387628e4b8fcf6c978e082b8c3380dbc62406c6aa460892c679872d656d7f
```

The immutable A5R3 publication schema also remains byte-identical:

```text
recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_schema.json
SHA256 = b24d197481c01c917ad9d4063441f96ac23979bdb4b3db24855e551c7fb91460
```

`_choice_audit_binding()` must always return exactly the v9 five-field
projection:

- `lock`;
- `ledger`;
- `status`;
- `audit_semantic_sha256`;
- `correction_actions_selected`.

The A5R3 publication claim/receipt remains present in the verified audit object
and is independently and strictly verified before that projection. It is not
embedded into the v9 binding. Missing, tampered, aliased or cross-bound A5R3
publication evidence remains a hard failure.

## 4. Portable failed-root identity

Cross-namespace forensic content uses algorithm
`kind_path_mode_size_content_v2_no_numeric_owner`. The digest binds entry type,
relative path, mode, size and regular-file content SHA, including empty
directories. Raw numeric UID and GID are excluded.

Ownership and mode are checked separately in the current execution namespace:

- every entry is owned by the effective user;
- every entry uses the root-local group and that group is authorized for the
  current process, without freezing its numeric value;
- only directories and regular files are allowed;
- world-writable, setuid, sticky and setgid regular-file entries are forbidden;
- setgid directories, group-writable data and world-readable data remain
  permitted where inherited from the retained root.

This is a logical forensic seal, not a WORM claim.

## 5. Pre-natural hard gate

Before any successor natural access:

1. freeze this amendment, machine manifest, failure observation, controller,
   gate and tests;
2. prove the v9 and A5R3 schema SHA values above are unchanged;
3. generate a real synthetic A5R3 publication and prove its independently
   verified audit projects to exactly five v9 fields;
4. validate a complete v9 input-lock manifest with that projection;
5. prove reinserting `publication` reproduces schema rejection;
6. prove uniform namespace GID remapping cannot change the portable digest but
   unsafe owner/group/type/mode states fail locally;
7. run inherited A5R2/A5R3/A5R4/A5R5 producer/controller regressions;
8. replay the retained A5R5 `worker_result.json` and `worker.log` bytes against
   their frozen SHA256 values;
9. bind the exact gate SHA, evidence SHA, test contract and tracked-source map
   into both the materialization claim and controller lock;
10. commit and push, then require clean local/upstream equality.

The controller must verify the current committed gate before it creates the
successor state root. The source snapshot must contain the identical verified
gate. Launch, worker and deep verification revalidate this binding; a later
clean pushed commit without a newly generated gate therefore cannot obtain
natural access.

## 6. One-shot execution and downstream boundary

The new execution derives the unchanged scientific UID/root pattern from the
new commit and uses an A5R6-specific hidden controller root. Materialization,
launch and worker-start claims are no-overwrite and the worker remains bound to
the launched PID/start-time/command line. Any failed successor consumes that
identity permanently.

E1-2/E1-3 remains prohibited until worker exit 0, canonical GO with no BLOCKED
receipt, snapshot-executed deep replay of the full producer/consumer graph and
publication of a no-overwrite `deep_verifier_receipt.json` bound to the gate,
controller lock, manifest, sidecar and GO receipt.
Even then, a separate diagnostic/intervention pre-output lock is required
before any of the six E0 checkpoints may be loaded. E1-pilot, model-selection,
test and confirmatory data remain sealed.
