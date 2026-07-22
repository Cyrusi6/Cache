# FPCT GPU R2k Equivalent-Kernel Latency Protocol

Status: `PRE-DIAGNOSTIC LOCK`

Parent worktree HEAD and upstream at protocol drafting:
`4e06f0b970057ee374dd6ff9d02d1ca6cb133aa6` on
`research/fpct-factorized-transport`.

## Immutable predecessor

R2j remains permanently classified as `GPU_ENGINEERING_BLOCKED_R2`.
Its scientific commit is
`efa02fba98adff2a891445c4908a8dc9ac8c7fff`, image digest is
`sha256:8eac5693511a3172547d80e4348b72bb4c57cbe0ad15ed47d607d19e0d5ccccf`,
run-lock SHA256 is
`51ce0a5e62906bb6388f9e0dbfd539fac120db3e70ef3259838e5f7289887c32`,
and its frozen median latency ratio is `1.6885509119773743 > 1.50`.
Nothing in R2k retries, overwrites, supplements, or reinterprets R2j.

No task correctness, model-selection output, or held-out output may be read
before the existing confirmatory release conditions are satisfied.

## Two disjoint phases

### A. `DIAGNOSTIC_ONLY`

This phase may iterate over profiling and mathematically equivalent kernel
implementations. Every diagnostic revision uses a new run UID and output root.
It cannot produce an R2k GO, alter R2j, authorize training, or access task
correctness. Diagnostic artifacts are labeled `DIAGNOSTIC_ONLY` and are never
pooled with the immutable gate.

The exact eight fresh-process block order is:

1. `C_POST -> F`
2. `F -> C_POST`
3. `F -> C_POST`
4. `C_POST -> F`
5. `C_POST -> F`
6. `F -> C_POST`
7. `F -> C_POST`
8. `C_POST -> F`

The process seeds are respectively `104729` through `104736`. Each arm uses
20 unmeasured warmups followed by 50 measured forwards. Each measured forward
records both CUDA-event device time and synchronized wall time. Latency runs
have profiler, NVTX, `record_function`, tracing and mechanism instrumentation
disabled. A separate trace run enables scopes and is excluded from latency.
All samples are retained.

The label-free panel contains the frozen canonical R2 row plus shape-only rows
covering sequence-length p50/p95, certified expansion p50/p95, m=1/2/3/4,
GQA/MQA, prefill and decode4. Both checkpoint-native and parameter-free
forced-on C_post/F canaries are measured. Forced-on results are engineering
diagnostics only and can never replace checkpoint-native operator evidence.

For each block and condition, the arm statistic is the median of its 50 CUDA
samples. The paired block ratio is `median(F) / median(C_post)`. The primary
balanced diagnostic estimate is the median of the eight paired ratios. A
one-sided 95% block-bootstrap upper confidence bound uses 50,000 resamples of
the eight blocks with replacement and seed `20260722`. Synchronized wall-time
versions are mandatory secondary results.

Freeze qualification, for both checkpoint-native and forced-on paths, is:

- balanced diagnostic CUDA median ratio `<= 1.35`;
- one-sided block-bootstrap 95% UCB `<= 1.50`;
- no numerical, provenance, host-sync, or resource integrity failure.

These are engineering qualifications only. They do not replace the immutable
resource gate.

### B. `IMMUTABLE_CONFIRMATORY_GATE`

After qualification, scientific code is committed and pushed, a new immutable
image is built, and a new run UID, root, run-lock and per-rank sealed
attestation are created. The complete R2k gate is executed once. Once any
formal latency ratio from that gate is read, the same scientific revision may
not be edited and rerun. A subsequent optimization requires a prospective R2l.

The gate must pass the full synthetic numerical suite, all 16 operator
conditions, P2--P6, all original 23 checks, and the new balanced/forced-on
resource canary. The original compatibility gate remains exactly:

- one warmup plus seven measurements;
- ratio of medians `<= 1.50`;
- seven-sample max (legacy p95) ratio `<= 1.75`;
- certified mean expansion `<= 1.35`;
- certified p95 expansion `<= 1.50`;
- peak HBM below 90% of device memory.

The balanced active gate and the legacy gate must both pass. A slow formal
result is `GPU_ENGINEERING_BLOCKED_R2K`; it is not an infrastructure retry.
Only a preregistered infrastructure event observed before reading ratios can
make the gate inconclusive.

## Equivalent flat-atom kernel

For every legal atom `(i,j)`, define

```
u_tij = q_t K_ij^T / sqrt(d) + log A_ij + mask_ti.
```

Let `gamma_tij` be the within-parent softmax and `beta_ti` the softmax over
parent log-sum-exp scores. Direct substitution gives

```
beta_ti gamma_tij
= exp(u_tij) / sum_(i',j') exp(u_ti'j'),
```

so one FP32 softmax over the flattened legal atom axis is exactly the same
global distribution. R2k may change only the computational realization of
this identity.

The production design is frozen as follows:

- the parent eager adapter runs once and its FP32 parent logits are reused;
- an equivalent parent occupies one real active slot containing the final
  parent K/V with log prior zero; other equivalent atoms do not enter matmul;
- non-equivalent atoms enter one FP32 flat softmax and one probability-times-V
  reduction;
- the prior enters exactly once through `log A`;
- no `[B,H,Q,S,D]` grouped-value tensor is allocated;
- beta/gamma are reconstructed only in instrumentation-on diagnostics;
- the all-parent-equivalent result is the exact parent adapter output without
  CUDA scalar extraction or a host branch;
- active forced-on F must use this same path and satisfy the resource gates.

The reusable `FPCTPackedLayout` owns structural parent/candidate indices,
safe indices, active maps, log priors, row offsets and semantic parent metadata.
It is created once after alignment, moved to the target device, reused across
layers, and never caches autograd tensors. A single segment does not use
`torch.cat`; redundant dtype/device casts are prohibited. Timing scopes are
truly disabled in production latency measurement and enabled only in separate
trace runs. C_post must not be intentionally slowed.

## Scientific invariants

The following cannot change in R2k: operator mathematics; candidates, top-k
or order; canonical FP32 A/logA; masks; `certified_slot0_v1`; parent-equivalence
definition; eager backend; trainable parameters; thresholds; data, panels,
seeds, training or statistics. Alignment/certification and natural-data
sidecars are not rerun.

Required tests include FP64 grouped-versus-flat identity; FP32/BF16 output and
gradient comparisons; m=0..4; extreme priors, padding and invalid atoms;
GQA/MQA; mixed equivalent/non-equivalent batches; fixed-RNG dropout; actual
patched Qwen3 eager; 28-layer replicated-null; collapse bypass; m<=1 identity;
checkpoint-native exact zero; forced activation; absence of F-only parameters;
and absence of hot-path host synchronization. C_post/F pre-collapse candidate
tensors and hashes remain identical.

## Historical timing audit boundary

R2h, R2i-v1, R2i-v2 and R2j P2/P3 manifests and traces are read-only
descriptive evidence. The three scientific hot-path blobs are identical
between R2i scientific commit `8d21c72` and R2j scientific commit `efa02fb`.
Historical profiles did not capture GPU UUID, clocks, temperature, power,
P-state, throttle reasons, foreign processes or CPU scheduling telemetry.
Those fields therefore remain `not_captured`; no infrastructure attribution
is permitted. R2k diagnostics must capture them prospectively.

## Conditional resumption

Only a full R2k GO authorizes the unchanged seed-104729 matched four-step
smoke, followed by the unchanged 12-seed by three-arm confirmatory design.
All 36 runs must complete before performance release; model-selection must
precede the one-time held-out release. Forced-on evidence never substitutes
for checkpoint-native O. The T/O/I/N definitions and statistical protocol are
unchanged.

## Decision tree

1. Protocol lock missing or provenance mismatch: stop before GPU.
2. Diagnostic qualification fails: continue only with a new diagnostic code
   revision and new diagnostic UID; no GO and no training.
3. Diagnostic qualification passes: freeze and attest a new scientific
   commit/image/run-lock.
4. Immutable gate infrastructure failure before ratio access: `INCONCLUSIVE`.
5. Any immutable numerical, control, latency, HBM, expansion or sync failure:
   `GPU_ENGINEERING_BLOCKED_R2K`; no same-revision retry and no training.
6. All immutable checks pass: `R2K_GPU_GATE_GO`, then matched smoke.
7. Smoke integrity failure: stop before formal training.
8. Smoke GO: resume the already frozen confirmatory state machine.

## Claim boundary

R2k can establish numerical equivalence and engineering readiness of a faster
F implementation. It cannot by itself establish query-time mechanism
activation, accuracy improvement, or a cross-model claim.
