# FPCT-GPU-R2m Execution Report

Date: 2026-07-22

## Outcome

The immutable engineering gate reached `R2M_IMMUTABLE_GO`. Synthetic numerical
checks, the original 23 compatibility checks, all six semantic checks, actual
Qwen3 FP32/BF16 decode4 bitwise controls, and both balanced canaries passed.

The subsequent matched training smoke stopped before optimizer step 1 with:

`GateError: incomplete confirmatory run lock`

The frozen confirmatory runner requires top-level keys `run_uid`,
`scientific_code_commit`, `image`, and `manifest_sha256`. The operative R2m lock
contains the first three but not `manifest_sha256`. The error occurs in
`load_lock` before the diagnostic output directory is created, so the run
performed zero optimizer steps and created no checkpoint.

This is the second run-lock/provenance omission in the R2 campaign. Per the
prospective R2m rule, the execution campaign is paused for a full harness audit.
The smoke was not retried, no formal seed job was submitted, and no automatic
R2n revision was created.

## Frozen provenance

- Starting SHA: `aafbe536dea0dfb385b4314cff54c4bf0ad7c198`
- R2m protocol commit: `66e4cdd197370edfb988bf5ae98d6987a9e43993`
- Clean execution/image commit: `80fb295542ad298fae4cddb1273517b401bbcd17`
- Operative replacement-lock commit: `80f69df8e2a37ab475eab80e70a75e137989efb3`
- Branch: `research/fpct-factorized-transport`
- Run UID: `fpct-r2m-80fb295-v1`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-80fb295-v1`
- Run-lock raw SHA256: `db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff`
- Canonical parsed SHA256: `f731660f73bb4e8daa9163cce533648c972758d5cbc18e93bbafdf040a7cc072`
- Image digest: `sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851`
- Image config/tree SHA256: `667e9151970a62bed3d9832c03812189ab955728baa90eca77f90080bd3b8d4d` / `b2090b58362312384c330865b499080750eaf380eb071a2e44917e4ca28f6b51`
- Image tar SHA256: `e203812574cd2abd4fa024cee48e1aaab480ddfb550d00e6e951843f94a76d78`
- Training sidecar SHA256: `48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`

The earlier `fpct-r2m-66e4cdd-v1` candidate preflight was superseded before any
scientific output because its image embedded the protocol HEAD. It did not
authorize or produce the immutable gate reported here.

## Canonical certified geometry

The geometry was mechanically extracted from
`recipe/eval_recipe/fpct_gpu_r2k/immutable_v1_run_lock.json`, file SHA256
`d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`,
Git blob `e993f8dc79207a439f5d029feb478307c2be34bb`, at
`/resource_geometry/tinyllama_all_splits`. Source SHA256 is
`6d5f6221be15db5af030f9c1d7702b6d1e814bfe4e423e065353f3167c914284`
and projection SHA256 is
`221c5164c60ec4d27abe714a68cec2f6a1a630d031195e0f0e63818133a5c6a2`.

| Task | Mean | P95 | Max |
| --- | ---: | ---: | ---: |
| ai2-arc | 1.2374040678425364 | 1.2973320158102766 | 1.4024390243902438 |
| mmlu-redux | 1.2259415516723047 | 1.3 | 1.4183673469387754 |
| openbookqa | 1.2390702194565386 | 1.2818298368298369 | 1.3142857142857143 |

## Config closure and exact-byte proof

- Config validator/finalizer targeted suite: `22 passed`.
- R2l lock was rejected for missing `/resource_geometry`; R2k canonical and R2m
  candidate locks passed.
- Negative tests failed closed for missing/empty/extra geometry, wrong source or
  projection SHA, changed/nonfinite/wrong-type numbers, stale image/UID/root/SHA,
  ConfigMap byte mismatch, and consumer blob mismatch.
- Git file, immutable ConfigMap data, init-container mounted lock, and main
  container startup lock all had raw SHA256 `db67428e...`.
- ConfigMap server object SHA256: `a2b3763ee98fbbe95b4b9e2b899693175e7cdd5fcbca49c497f357218fdcdb74`.
- Preflight result SHA256: `bc89fb484d73220a1862fecb15ef58cd23497b26500bcabde5dee83cf5b1689f`.
- Preflight recorded `scientific_output=false` and `training_authorized=false`.
- Ten frozen production scientific blobs matched the R2l allowlist byte for byte.

## Immutable GPU gate

- Synthetic GPU numerical gate: `8/8 GO`; complete lower-level checks all true.
- Original operator matrix: `16/16` conditions, `5/5` P2--P6 profiles.
- Original compatibility result: `23/23 GO`.
- Latency median/p95 ratios: `0.6749175501` / `0.6514860209`.
- Peak HBM: `4.2174715996 GiB`.
- Checkpoint-native FP32/BF16 factorization delta: `0 / 0`.
- Replicated and m<=1 controls: exact.
- Forced-on FP32/BF16 factorization delta: `0.2450048923 / 0.96875`.
- Hot-path host synchronization: `0`.

All required semantic checks passed:

- `native_parent_map_complete`
- `unknown_sidecar_fails_closed`
- `mixed_memory_exact_null_bitwise`
- `mixed_batch_exact_active_isolation`
- `actual_qwen_decode4_exact_null`
- `active_route_not_bypassed`

Actual Qwen3 used 28 layers, FP32 and BF16, prefill plus four decode steps. Cache,
84 per-step layer endpoints, and final logits were bitwise equal in every exact
control; forced-on samples remained active.

Balanced canary results:

| Mode | CUDA balanced median ratio | One-sided 95% block-bootstrap UCB | Qualified |
| --- | ---: | ---: | --- |
| checkpoint-native | 1.0556907947 | 1.0713349947 | yes |
| forced-on | 1.0626994163 | 1.0753476011 | yes |

The sealed finalizer artifact has SHA256
`6c1ce4322cf7773e57abb6a0e1604c75947fb3de2927ffd4f8420aea60a9b306`
and records `classification=R2M_IMMUTABLE_GO`, `training_authorized=true`.

## Kubernetes execution

All resources were scoped by run UID and ran on `4090-48gx2`. Image loader,
preflight, synthetic GPU gate, pretrained matrix, semantic gate, and balanced
canary completed `1/1`, with zero container restarts. The matched-smoke pod used
two GPUs, failed `0/1`, and also had zero restarts. No resource outside this run
UID was deleted or modified.

## Training and claim boundary

- Matched smoke: `BLOCKED_BEFORE_STEP_1` due missing top-level
  `manifest_sha256` in the immutable lock.
- Optimizer steps: `0`.
- Checkpoints: `0`.
- Formal triplets/runs submitted: `0/12`, `0/36`.
- Accuracy/correctness, model-selection, and held-out accessed: `false`.

The result supports an engineering statement only: the frozen R2m operator and
runtime satisfy the preregistered numerical, semantic, bitwise, resource, and
latency gates on the TinyLlama-to-Qwen3 setup. It provides no matched-training,
accuracy, task-performance, query-time mechanism, or cross-model scientific
claim. Formal confirmatory execution remains paused until a separately reviewed
full harness audit is completed.
