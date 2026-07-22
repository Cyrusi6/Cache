# FPCT-CFM-HARNESS-H1 Audit Report

## 1. Terminal result

- Classification: `H1_REQUIRES_NEW_IMAGE_QUALIFICATION`
- Audit status: `COMPLETE`
- Scientific output: `false`
- Training authorized: `false`
- Recovery execution authorized: `false`

H1 completed the config-only audit, but the current R2m image cannot be reused
for confirmatory recovery. Its finalizer, matched-smoke producer, formal-triplet
producer, and controller evidence/state contract do not all bind the required
run UID, execution-lock SHA, image digest, and prerequisite-artifact SHA. The
necessary fixes touch Python files embedded in the sealed image, so a later
recovery revision must build and separately qualify a new image. H1 does not
create that revision and does not authorize any execution.

## 2. Isolation and permanent state

The audit started from clean local/upstream commit
`cf27780ca48e435cb0f3655daf3adcd9e25ece54` on
`research/fpct-factorized-transport`. The prospective H1 protocol was committed
and pushed before implementation or exact-image probing as commit `0fd69e0`
(`chore(fpct): lock confirmatory harness H1 audit protocol`).

R2m remains immutable:

- Engineering classification: `R2M_IMMUTABLE_GO`
- Scientific commit: `80fb295542ad298fae4cddb1273517b401bbcd17`
- Image: `sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851`
- UID: `fpct-r2m-80fb295-v1`
- Operative lock raw SHA256: `db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff`
- Immutable final SHA256: `6c1ce4322cf7773e57abb6a0e1604c75947fb3de2927ffd4f8420aea60a9b306`
- Campaign: `PAUSED_HARNESS_AUDIT_REQUIRED`
- Matched smoke: `BLOCKED_BEFORE_STEP_1`
- Optimizer steps/checkpoints: `0/0`

The H1 candidate uses the non-execution sentinels
`h1-config-only-not-an-execution-uid` and
`H1_CONFIG_ONLY_NO_EXECUTION_ROOT`. It is not an R2m patch, retry, replacement
run, or R2n execution identity.

## 3. Historical red fixtures

All frozen historical facts were machine reproduced:

1. The real confirmatory `load_lock` rejects the R2m operative lock because
   `/manifest_sha256` is absent.
2. Static AST discovery of `load_lock` yields exactly `/run_uid`,
   `/scientific_code_commit`, `/image`, and `/manifest_sha256`.
3. The historical consumer manifest does not declare that complete set.
4. R2l missing `/resource_geometry` and R2m missing `/manifest_sha256` remain
   permanent negative fixtures.
5. The actual confirmatory manifest raw SHA256 is
   `5e04fe7ffa2f5df6ed0159b67be88a1c9547f0cd11886d461166e1ce7ba455e4`.
6. The last training-compatible R2 lock has the same top-level manifest hash.

No historical artifact was supplemented or reinterpreted.

## 4. Producer-consumer and stage-graph audit

The source-derived graph contains 11 nodes and 10 edges and is acyclic:

`manifest -> execution lock -> config preflight -> immutable final -> matched smoke -> formal training -> formal completion -> model-selection -> held-out -> statistics -> terminal result`

The committed discovered graph records, for every stage, producer path and blob
SHA, input artifacts, discovered pointers, output schema, prerequisites,
permitted/forbidden side effects, next-stage authorization, argparse contract,
and—where applicable—the K8s command, environment, image, resources, volumes,
node, UID, and root identity.

Static discovery covers constant subscripts, chained `.get()`, membership,
`required.issubset`, JSON-pointer literals, alias propagation, argparse,
Kubernetes inline Python guards, placeholders, paths, volumes, environment, and
resource requests. There are zero unresolved, unannotated dynamic accesses.

Current prospective declarations and discovered requirements compare as:

| Consumer | Declared = discovered | Finding |
| --- | --- | --- |
| `legacy_load_lock` | yes | Exact four-pointer closure |
| `r2_controller` | yes | Current direct state/evidence reads close on `/run_uid` and `/status` |
| `train_config_builder` | no | Declaration is stale and omits `/run_uid` while declaring unused receiver/sender roots |
| `matched_smoke` | no | Transitive `load_lock` requirements are missing from the declaration |
| `formal_triplet` | no | Transitive `load_lock` requirements are missing from the declaration |
| `r2_k8s_renderer` | no | Source reads parent objects as well as declared leaf fields |
| `r2m_finalizer` | no | No prospective consumer declaration exists for eight discovered input fields |
| `trainer` | no | No declaration exists for 25 discovered model/training config keys |
| `statistics` | no | No declaration exists for the five required cell columns |

These mismatches are a fail-closed audit finding; they are not silently repaired
inside the already-pushed protocol seed.

## 5. Output identity gaps requiring a new image

The required semantic binding set is: run UID, lock SHA, image digest, and
prerequisite-artifact SHA.

- R2m finalizer: binds none of the four identities in its output.
- Matched-smoke result: binds `run_uid`, but not lock SHA, image digest, or
  immutable-final/prerequisite SHA.
- Formal-triplet result: binds `run_uid`, but not lock SHA, image digest, or
  matched-smoke/prerequisite SHA.
- Controller state: binds `run_uid` and the lock SHA under
  `run_lock_sha256`, but not image digest or prerequisite-artifact SHA. Its
  evidence validator checks status rather than the full upstream identity.
- Historical matched-smoke and seed-triplet K8s guards check classification,
  authorization, or status, but do not bind the complete upstream identity.

The first three producers and the controller are implemented in
`fpct_gpu_r2m_finalize.py`, `fpct_confirmatory_runner.py`, and
`fpct_gpu_r2_controller.py`, which are inside the sealed R2m image. Therefore
the correct recovery boundary is new-image qualification, not R2m reuse.

## 6. Schemas, compiler, and hash DAG

Ten strict versioned schemas cover execution lock, prerequisite receipt, smoke
result, training config, arm result, triplet manifest, controller state, formal
completion, release receipt, and terminal result. Every schema closes its top
level with `additionalProperties=false`. Strict loading and validation reject
duplicate keys, NaN/Inf, prohibited nulls, bool-as-number, malformed or empty
hashes, wrong types, stale identities, and cross-object inconsistencies.

One deterministic compiler produces the H1 candidate lock projection,
immutable ConfigMap, smoke/formal critical Job projections, and no-authority
receipt. The top-level manifest SHA, nested manifest SHA, and actual manifest
raw SHA are identical. The runner `ARM_ORDER` exactly matches the frozen
confirmatory manifest.

- Candidate lock projection SHA256:
  `7c0bfbb15600b62d91f54e85e576a8d8fea455f2f6cdfff43f2b0f29bfef2683`
- Candidate ConfigMap SHA256:
  `2252cd4f6799f8f385d0a9c65016c5517603a5d290023cf8d046d7a8aef498d0`
- Candidate smoke Job SHA256:
  `210e114aba2689b7a2787cedc75f1c12d86aa94879ac5bdd05465fe692d47e5a`
- Candidate formal Job SHA256:
  `ddfd35dc2ba949464068f68f5cdfcab5810bca209c9f660e6b83369d6d7ed4fc`
- No-authority receipt SHA256:
  `6258a226082e04d6a7319c2dd7f0893c707aa8e792eb9b26f2c0ed8ce3b23195`

## 7. Mutation and regression coverage

The final matrix contains 389 mutations; all fail closed before any model,
dataset, optimizer, subprocess, CUDA, or checkpoint action. Coverage includes
per-field delete/null/empty/wrong-type mutations, malformed/wrong hashes,
single-byte mutations, stale UID/root/commit/image, extra and duplicate keys,
nested/top-level manifest disagreement, asset mismatch, seed/arm/order,
world-size/GPU count, prerequisite classification/SHA, template placeholders,
premature transitions, selective-arm omission/retry, and duplicate releases.

The targeted CPU-only suite passes: `14 passed`.

## 8. Exact-image and K8s dry-run evidence

The exact R2m image was run with network disabled, no GPU request, no model/data
mount, and empty `CUDA_VISIBLE_DEVICES`. Tripwires recorded zero model, dataset,
optimizer, subprocess, checkpoint, and CUDA accesses.

- Real legacy R2m probe: exit 1 with the expected incomplete-lock error.
- Candidate probe: exit 0 and `SEALED`.
- Training configs: all 3 smoke arms plus 36 formal arm configs validated
  (`39` total).
- Controller simulation: reached `HELD_OUT_RELEASED` only along the full DAG.
- Premature formal completion, duplicate model-selection release, and duplicate
  held-out release were all rejected.
- Git/candidate, ConfigMap data, mounted bytes, and main-container bytes all
  equal the candidate lock SHA shown above.
- No training output directory, optimizer state, or checkpoint was created.

Only Kubernetes server-side dry-run was used. The immutable ConfigMap, smoke
Job, and formal Job were accepted as dry-run objects; all three carry explicit
`scientific-output=false` and `training-authorized=false` annotations, and the
formal projection retains the frozen two-GPU request. A read-only cluster check
confirmed that no ConfigMap or Job from H1 was created.

## 9. Artifacts and hashes

- Audit result SHA256:
  `35ac8d723ab27a153ee5875519ee909395d066b9d8459d2bd74922dd57945998`
- Mutation matrix SHA256:
  `e33e9162da7e0de15128681099e90adbb84348c60e49fd9d3f1480cde55c5e23`
- Discovered consumer registry SHA256:
  `07a55f7da523fb7b45afcb5a35a05745d453ca99f9b8d75fca8665c27008806a`
- Discovered stage graph SHA256:
  `3316aec41ce19d84a63b0aa466dc9533a46a9c399dea62f98c7d7c5565160f35`
- Exact-image dry-run summary SHA256:
  `fb7e2dab94a570b3b17d08ed74715662eef520dc62bdfef6b87dbecbb3d3ad21`
- K8s server dry-run summary SHA256:
  `4f56971c50e11caa0ba4fbdc25da5047b819398547095b077cd8f8263f5b2762`

Detailed config-only evidence is committed under
`recipe/eval_recipe/fpct_cfm_harness_h1/audit_artifacts/`. The local reproducibility
root is
`/home/lijunsi/projects/Cache-fpct-factorized-transport/local/final_results/fpct_cfm_harness_h1/rev_0fd69e0/`.

## 10. Reuse-versus-requalification decision

Decision: **requalification required**.

R2m's engineering GO remains valid and immutable, but its confirmatory harness
cannot be recovered by adding a field to the old lock, reusing its UID/root, or
retrying its smoke. A future, separately authorized revision must prospectively
close the declarations, add full producer artifact identity binding and
consumer guards, build a new image, and repeat the applicable no-model and
image qualification gates. H1 grants no authority to do so.

## 11. Hard-stop confirmation

H1 did not load a model or dataset, execute a CUDA/model forward, create an
optimizer or checkpoint, read accuracy/correctness/model-selection/held-out,
retry R2m smoke, modify or reuse the R2m UID/root/lock/ConfigMap/results, create
R2n, or create a real Kubernetes resource. The task stops at human review.
