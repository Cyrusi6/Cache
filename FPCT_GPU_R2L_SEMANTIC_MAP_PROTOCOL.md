# FPCT GPU R2l Mixed-Memory Semantic-Map Protocol

## Status and predecessor boundary

This document is the prospective pre-output lock for R2l. It is frozen before
any new R2l pretrained or GPU forward.

R2k remains permanently `GPU_ENGINEERING_BLOCKED_R2K` with scientific commit
`458b0260fc5475c9ae578eb68b8dff2b2699e2f4`, image digest
`sha256:bc19b894c18eea266596011b748a4a22c73b80788e0fdd4a08b5f33059bf51ca`,
run UID `fpct-r2k-458b026-v1`, run-lock SHA256
`d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`,
and checkpoint-native FP32/BF16 deltas `3.7909e-5/0.625`. Its passing resource
ratios `0.710337/0.706328` do not override the failed exact-null invariant.
Nothing in R2l retries, supplements, overwrites, or reinterprets R2k.

## Frozen root cause

R2k created each layer's pre-bound semantic map from an all-false `[B,S]`
tensor and filled only sidecar spans. Ordinary native parents outside sidecar
coverage therefore remained false. In mixed memory,
`parent_equivalent.all(dim=-1)` could not select the already computed exact
parent output even when every covered sidecar parent was certified equivalent.
The checkpoint-native path instead used the mathematically equivalent flat
packed reduction, exposing a different numerical call order that accumulated
in BF16.

R2l tests this root cause prospectively. If the repair does not restore bitwise
checkpoint-native identity, R2l may continue diagnostic repair only before the
scientific freeze. Once immutable output is produced, the revision is one-shot.

## Normative semantic formula

Let `U[b,i]` be sidecar coverage and `E_sc[l,b,i]` be discrete certified
sidecar equivalence. The full parent map is

`E[l,b,i] = (not U[b,i]) or (U[b,i] and E_sc[l,b,i])`.

The implementation contract is:

1. Initialize the complete map to boolean true on the layout device.
2. For every sidecar span, first set the entire span to false.
3. If `parent_equivalent` exists, copy it into that span.
4. Otherwise, if `parent_force_native` exists, copy it into that span.
5. Otherwise leave the covered span false: missing metadata fails closed.
6. Native gaps, leading parents, trailing parents and generated native tail
   remain true.
7. Invalid candidate atoms cannot alter native-parent semantics.
8. No float tolerance, `D_K`, `D_V`, approximate equality, host scalar, Python
   branch on CUDA data, or per-query/head route may certify equivalence.

The existing sample-level `all_parent_equivalent =
parent_equivalent.all(dim=-1)` and tensor-only final `torch.where` selection
remain normative. R2l changes the truth map, not the attention mathematics.

## Implementation boundary

The only existing scientific function authorized to change is
`bind_fpct_layout_layer_semantics` in `rosetta/model/fpct_attention.py`.
New R2l-only validation, analysis, runner, test, manifest, report and K8s files
are allowed. Existing status/report ledgers may be appended.

The flat-atom kernel, parent eager adapter, parent-logit reuse, candidate
projection/fusion, projector, gate, initialization, prior, mask, certifier,
alignment, timing method, thresholds, data, seeds, training recipe and
statistics are frozen. The protocol verifier removes the authorized function
from baseline and working copies and requires every remaining byte of
`fpct_attention.py` to be identical. It also pins the SHA256 of all other
forbidden scientific files.

## Red-green contract

Before repair, a regression must demonstrate that a source length of nine with
an equivalent sidecar only on `[2:5)` incorrectly leaves native positions
false. After repair, the same test must pass. Tests must also cover:

- one false sidecar parent with native leading/trailing positions true;
- missing sidecar metadata failing closed;
- multiple discontinuous spans and native gaps;
- a mixed batch where the exact sample selects parent output and the active
  sample remains on the flat path;
- the pre-bound semantic-map path rather than only group-equality fallback;
- FP32/BF16, GQA/MQA, padding, causal prefill and at least decode4;
- actual patched Qwen3 eager across 28 layers.

Checkpoint-native exact null requires direct
`F_native == F_replicated_native == C_post_native` using `torch.equal`, tensor
byte SHA, `max_abs=0`, and `ULP=0` at selected pre-o-proj, post-o-proj,
residual/cache and final-logit endpoints. The historical numerical tolerances
remain only for packed/replicated numerical invariance, never as sufficient
evidence for exact null.

## Active-path preservation

Forced-on evidence must contain at least one covered parent with equivalence
false, make `all_parent_equivalent` false for the corresponding sample, and
execute the unchanged flat-atom path. `D_K`, `D_V`, candidate-logit range and
factorized delta must exceed the already frozen metric-specific null floors.
Exact and active samples in one batch must not contaminate each other.
C_post/F pre-collapse candidate tensors and hashes remain identical.

Replicated-atoms, collapse bypass and m<=1 remain distinct controls. Training
mode must retain fixed-RNG dropout behavior, finite nonzero candidate-sensitive
gradients, and no F-only parameter.

## Two-stage execution

### Focused diagnostic

Before science freeze, R2l may iterate only targeted CPU/HF tests, synthetic
GPU mixed-memory exact-null, one label-free pretrained mixed-memory canary, and
focused checkpoint-native/forced-on latency/no-sync. It may not rerun natural
alignment/certifier audits, history, Phase2A, geometry, accuracy, model
selection or held-out evaluation.

Freeze qualification requires bitwise FP32/BF16 final identity, no 28-layer
first divergence, complete truth-table coverage, active forced-on behavior,
unchanged resource thresholds, no hot-path synchronization, and a clean
machine-verifiable allowlist diff.

### Immutable confirmatory gate

After qualification, a new scientific commit, image, run UID/root and run-lock
are frozen. The one-shot gate requires the complete synthetic numerical gate,
16/16 operator conditions, P2--P6, the original 23/23 checks, the six new
semantic checks, and then the balanced checkpoint-native/forced-on canary.

New checks are `native_parent_map_complete`, `unknown_sidecar_fails_closed`,
`mixed_memory_exact_null_bitwise`, `mixed_batch_exact_active_isolation`,
`actual_qwen_decode4_exact_null`, and `active_route_not_bypassed`.

Original resource limits remain median ratio `<=1.50`, p95 ratio `<=1.75`,
mean/p95 expansion `<=1.35/1.50`, peak HBM `<90%`, and zero hot-path host sync.
The balanced canary keeps the R2k `1.35` median and `1.50` one-sided UCB limits
for both checkpoint-native and forced-on paths.

Any correctness, active, resource or no-sync failure is terminal
`GPU_ENGINEERING_BLOCKED_R2L`; any repair requires R2m. Only a prospective
infrastructure failure before scientific output may be
`INFRASTRUCTURE_INCONCLUSIVE_R2L`.

## Conditional training and claim boundary

Training is unauthorized until every immutable gate component is GO. If GO,
the unchanged seed-104729 three-arm four-step smoke runs first; only a complete
smoke GO authorizes the unchanged 12 matched triplets and 36 formal runs.
Performance remains sealed until 36/36, and held-out remains sealed until the
model-selection gate.

R2l can establish only semantic-map correctness, bitwise checkpoint-native
recovery, active flat-path preservation and engineering readiness. It cannot by
itself establish task improvement, the formal query-time mechanism, or
cross-model generality.
