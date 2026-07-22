# FPCT GPU R2m Config-Closure Protocol

## Status and scope

R2m is a prospective configuration/provenance-only recovery. R2l remains
permanently `GPU_ENGINEERING_BLOCKED_R2L`; its UID, root, image, lock, outputs,
and 21/23 result are immutable and cannot be patched, resumed, retried, or
reinterpreted.

No production scientific implementation may change in R2m. In particular,
`fpct_attention.py`, `wrapper.py`, projector, aligner, prior, mask, candidate
certifier, flat atom kernel, parent-logit reuse, original R2 runner, training,
data, seeds, thresholds, and statistics are byte-frozen to the R2l blobs.

## Canonical certified geometry

The only recovered scientific input is the pre-existing label-free certified
geometry object at JSON pointer
`/resource_geometry/tinyllama_all_splits` in the R2k immutable run lock. It is
mechanically extracted and canonically serialized; it is not recomputed and no
task label, correctness, model-selection, or held-out output is read.

The canonical source must have:

- source SHA256 `6d5f6221be15db5af030f9c1d7702b6d1e814bfe4e423e065353f3167c914284`;
- exactly `ai2-arc`, `mmlu-redux`, and `openbookqa`;
- exactly three rows;
- finite numeric `mean`, `p95`, and `max` values;
- canonical projection SHA256
  `221c5164c60ec4d27abe714a68cec2f6a1a630d031195e0f0e63818133a5c6a2`.

The canonical source file is
`recipe/eval_recipe/fpct_gpu_r2k/immutable_v1_run_lock.json`, file SHA256
`d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`,
Git blob `e993f8dc79207a439f5d029feb478307c2be34bb`.

## Closure contract

The R2m lock must pass all of the following before any scientific output:

1. strict JSON parsing that rejects NaN and Infinity;
2. the tracked JSON schema;
3. exact equality with the canonical geometry projection;
4. exact task set and row count;
5. declared schema, consumer-manifest, production-blob, and geometry hashes;
6. consumer JSON-pointer type/requiredness checks;
7. consumer source/blob and aggregate-geometry AST fingerprints;
8. expected image, commit, UID, and run-root identity;
9. byte equality among Git lock, immutable ConfigMap data, mounted Pod lock,
   and main-container lock;
10. the original aggregate consumer fixture returning the exact canonical
    geometry and `expansion_mean=true`, `expansion_p95=true`.

The lock cannot contain the SHA256 of its own raw bytes without a circular
fixed-point dependency. Therefore raw-byte identity is bound by the tracked
preflight binding manifest and immutable ConfigMap metadata, while the lock
itself declares a canonical parsed-payload SHA computed with only that single
field omitted. Every non-self-referential closure hash remains inside the lock.

## Fail-closed negatives

Before execution, tests must reject missing geometry, missing
`tinyllama_all_splits`, empty/extra/missing tasks, wrong source SHA, changed or
non-finite numbers, wrong projection hash, stale image/UID/root/commit,
ConfigMap/Git byte mismatch, mounted/Git byte mismatch, and consumer manifest
or runner blob mismatch. The historical R2l lock must be rejected with the
explicit missing pointer.

## Output-free exact-image preflight

`CONFIG_PREFLIGHT_ONLY` may import the final candidate image and run the
validator, but it must not load model weights or datasets, initialize CUDA,
execute a model forward, produce a task metric, or authorize training. Its
output is limited to hashes, schema status, task names, row count,
consumer-closure status, `scientific_output=false`, and
`training_authorized=false`.

Any preflight failure may be repaired only before scientific output. A lock
byte change invalidates the old preflight and requires a new lock SHA and new
preflight record.

## Immutable gate and stopping rule

After preflight GO, a new R2m image, UID, root, immutable ConfigMap, and run
lock execute the frozen sequence: complete synthetic 8/8; original 16
conditions and P2--P6; original 23/23; six R2l semantic checks; actual Qwen3
28-layer FP32/BF16 prefill plus decode4 bitwise check; eight-block balanced
checkpoint-native/forced-on canary; sealed finalizer.

From the first scientific output, the revision is one-shot. Any correctness,
semantic, activation, resource, HBM, no-sync, or provenance failure is terminal
`GPU_ENGINEERING_BLOCKED_R2M`. A second run-lock/provenance omission pauses the
entire execution campaign for harness audit; R2n must not be started
automatically.

Training remains mechanically blocked until the finalizer writes
`R2M_IMMUTABLE_GO`. Matched smoke and formal training retain the exact frozen
seed, arm, data, optimizer, checkpoint, retry, and statistical contracts.
