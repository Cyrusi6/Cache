# FPCT GPU R2l Final Report

## Terminal decision

R2l is terminal `GPU_ENGINEERING_BLOCKED_R2L`.

The complete synthetic GPU gate passed 8/8. The original pretrained
compatibility matrix then completed all 16 operator conditions and all five
P2--P6 profiles, but its sealed aggregate passed 21/23 checks. The two failed
checks are `expansion_mean` and `expansion_p95`. Per the prospective one-shot
contract, no semantic Job, balanced active canary, matched smoke, or formal
training was submitted after this result.

## Identity and provenance

- Operator repair: `d71d21b1e315787e9af1cefb324abd310fd335f7`
- Immutable checker/image source: `43b825b34204326029590da7b9d51b67d7916208`
- Pre-output lock commit: `577c32ccdbb8dc06d29a4a179e3080d0fb8dca7e`
- Image: `docker.io/library/fpct-gpu-r2l:43b825b@sha256:e805c714f4a77be82fe89e36a100750ba25ad815b5af004d6f9ae4233f37492e`
- Embedded source tree: `2381e0aa14d25ac7d72a964d03a2a784f5b95e66eaad0b0ad0a0d7fd241af5ca`
- Run UID: `fpct-r2l-43b825b-v1`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2l-43b825b-v1`
- Run-lock SHA256: `ed95ac14219c1c8cefe83f93a32667ddcf318f2a2df85b0725ec5217bdceea96`

Image loader and synthetic GPU Jobs completed 1/1. The pretrained Job exited
through the sealed scientific target with code 1 and zero Pod restarts; this
is not an eviction, preemption, storage, network, or node failure.

## What passed

The R2l semantic-map repair achieved the original checkpoint-native endpoint:

- FP32 `Delta_fact=0`;
- BF16 `Delta_fact=0`;
- `expected_native_null=true`;
- collapse bypass, replicated atoms, local replicated numerical control, and
  m=1 control all pass;
- C_post/F pre-collapse candidate identity passes;
- forced-on D_K, D_V, logit range, and query activation all pass;
- forced-on FP32/BF16 deltas are `0.245005/0.96875`;
- latency median/p95 ratios are `0.943316/0.957141`;
- peak HBM is `4.21747 GiB`;
- hot-path and scientific sync checks pass;
- finite, prior, mask, panel, projector-state, and eager-runtime checks pass.

Thus the mixed-memory map repair recovered the bitwise checkpoint-native
parent computation without disabling the active flat route. This remains an
engineering statement, not a task-performance result.

## Why the immutable gate failed

The frozen R2l run lock omitted the required
`resource_geometry.tinyllama_all_splits` object. The unchanged original runner
loads geometry with:

`lock.get("resource_geometry", {}).get("tinyllama_all_splits", {})`.

It consequently recorded `certified_geometry={}`. Both expansion checks are
defined to fail closed unless geometry rows exist, so `expansion_mean=false`
and `expansion_p95=false` even though no measured expansion overrun occurred.
The previous frozen geometry values would have satisfied the limits, but they
were not bound into this immutable run lock and cannot be inserted after
scientific output.

This is an immutable provenance/configuration integrity failure, not evidence
that the repaired operator exceeded the expansion thresholds. It is still
terminal under R2l because the protocol forbids modifying the lock or rerunning
the same revision after scientific output. Any recovery requires a new
prospective R2m-or-later protocol, image, UID, root, and run lock.

## Stopped stages and firewall

- Six-check R2l semantic gate: not run.
- Eight-block balanced checkpoint-native/forced-on canary: not run.
- Matched smoke seed 104729: not run.
- Formal seeds 45--56: 0/12 triplets, 0/36 runs.
- Optimizer steps: 0.
- Checkpoints: none.
- Accuracy/correctness, model-selection, and held-out: not accessed.

R2k remains permanently `GPU_ENGINEERING_BLOCKED_R2K`; R2l neither retries nor
reinterprets it. R2l itself must also remain immutable and must not be resumed.

## Claim boundary

R2l supports only the engineering claim that the corrected mixed-memory
semantic map can restore checkpoint-native bitwise equality while retaining
the forced-on active route. Because the immutable resource/provenance gate did
not complete, R2l provides no matched-training evidence, no task-accuracy
claim, no formal query-time factorization mechanism claim, and no cross-model
claim.
