# FPCT-CFM-HARNESS-H1 Protocol

## Scope and authority

H1 is a config-only producer-consumer contract audit. It may read source,
historical locks, manifests, Kubernetes templates, controller code, and the
sealed R2m image. It may create audit code, schemas, deterministic compiler
outputs, mutation fixtures, and server-side dry-run requests. It must not load a
model or dataset, expose CUDA, create an optimizer/checkpoint, read performance
outputs, retry R2m, create R2n, or authorize execution.

Every H1 artifact must contain:

- `scientific_output=false`
- `training_authorized=false`

R2m remains immutable: scientific commit `80fb295542ad298fae4cddb1273517b401bbcd17`,
image `sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851`,
UID `fpct-r2m-80fb295-v1`, lock SHA256 `db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff`,
and immutable-final SHA256 `6c1ce4322cf7773e57abb6a0e1604c75947fb3de2927ffd4f8420aea60a9b306`.

## Historical red fixtures

The audit must reproduce, rather than manually assert, both permanent failures:

1. R2l lock is rejected because `/resource_geometry` is absent.
2. R2m lock is rejected by the real confirmatory `load_lock` because
   `/manifest_sha256` is absent.

Static discovery from `load_lock` must recover exactly `/run_uid`,
`/scientific_code_commit`, `/image`, and `/manifest_sha256`. The actual
confirmatory manifest SHA256 is
`5e04fe7ffa2f5df6ed0159b67be88a1c9547f0cd11886d461166e1ce7ba455e4`.
The last training-compatible R2 lock is
`recipe/eval_recipe/fpct_gpu_r2/fpct_gpu_r2j_run_lock.json`, whose top-level
`manifest_sha256` has that exact value.

## Audit method

The H1 registries are protocol seeds, not a substitute for discovery. The audit
must parse real Python AST, argparse definitions, shell/YAML command bodies,
placeholders, inline Python, JSON-pointer annotations, and alias flow. Dynamic
access that cannot be resolved must have an explicit annotation; otherwise H1
fails.

For every consumer, the equality test is:

`discovered_required_inputs == declared_required_inputs`

The audit must additionally compare runner `ARM_ORDER`, seeds, examples,
optimizer steps, process count, batch/accumulation, scheduler, learning rate,
weight decay, warmup, max length, precision, checkpoint rule, retry rule, and
release ordering against the confirmatory manifest.

## Schemas and compiler

Strict versioned schemas are required for the execution lock, prerequisite
receipt, smoke result, training config, arm result, triplet manifest, controller
state, formal completion, release receipt, and terminal result. Critical objects
use `additionalProperties=false`; strict loaders reject duplicate keys,
NaN/Inf, null where prohibited, bool-as-number, empty/malformed hashes, stale
identity, and cross-field inconsistencies.

One deterministic compiler is the sole producer for candidate execution lock,
ConfigMap, critical K8s projection, and authorization receipt. Hash domains and
the non-self-referential dependency DAG are frozen in `hash_dag.json`.

## Mutation coverage

Every discovered required input receives delete/null/empty/wrong-type and
identity/hash mutations where applicable. Cross-object mutations cover stale
UID/root/commit/image, manifest disagreement, asset mismatch, seed/arm/order,
world-size/GPU count, prerequisite artifact/classification/SHA, unresolved
placeholders, premature state transitions, selective-arm retry, duplicate
release, duplicate JSON keys, and extra fields. All mutations must fail before
model, dataset, optimizer, subprocess, or CUDA access.

## Exact-image dry-run

The sealed R2m image is used with no GPU request, no model/data mount, network
disabled, and `CUDA_VISIBLE_DEVICES=""`. Tripwires forbid model/dataset loading,
optimizer creation, training subprocess launch, checkpoint writes, and CUDA
access. The dry-run exercises the real legacy `load_lock`, three smoke configs,
36 formal configs, controller DAG/negative transitions, server-side dry-run of
smoke/formal Jobs, and Git/ConfigMap/mounted/container byte identity.

## Decision rule

Only these terminal classifications are valid:

- `H1_AUDIT_GO_NO_EXECUTION_AUTHORITY`
- `H1_AUDIT_BLOCKED`
- `H1_REQUIRES_NEW_IMAGE_QUALIFICATION`

GO requires complete bidirectional closure, full mutation coverage, exact-byte
proof, complete no-model dry-run, and no required image-internal code change. If
the audit proves that a runner/controller/finalizer inside the R2m image must be
changed, classification is `H1_REQUIRES_NEW_IMAGE_QUALIFICATION`. Every outcome
stops for separate human review and grants no execution authority.
