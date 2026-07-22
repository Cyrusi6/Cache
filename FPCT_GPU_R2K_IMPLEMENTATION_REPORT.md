# FPCT GPU R2k Equivalent-Kernel Implementation Report

Status: `CPU/HF READY; GPU DIAGNOSTIC NOT STARTED`

## Scientific boundary

The implementation changes only the computational realization of F attention
and profiling control in the three R2i/R2j hot-path files. It does not change
the candidate set/order, top-k, FP32 prior/log prior, masks,
`certified_slot0_v1`, eager backend, trainable parameters, thresholds, data,
seeds, training recipe or confirmatory statistics. The sealed natural
alignment/certified-support audit was not rerun.

## Kernel changes

- The shared Qwen eager parent adapter executes once and optionally returns its
  FP32 masked logits. F reuses these logits for parent-equivalent slots.
- F now applies one FP32 softmax over the flat legal atom axis and one
  probability-times-value matmul. This is algebraically identical to
  `beta(i) * gamma(j|i)`.
- The old `[B,H,Q,S,D] group_value` allocation, grouped max/sum, gamma scatter
  and beta-times-group-value reduction were removed from the production path.
- Exact parent-equivalence is computed from the final candidate/collapsed K/V
  once during candidate projection. Per-layer semantic maps are bound into the
  reusable layout once before attention hooks are installed.
- Equivalent groups retain one active slot with exact final parent K/V and
  logA zero. The atom QK input for that slot is zeroed, then the already
  computed parent logit is inserted. Remaining equivalent atoms are inactive.
- `safe_parent`, safe candidate indices and slot indices are stored in the
  reusable layout. A single sidecar segment avoids `torch.cat`; no-op
  device/dtype casts are skipped.
- `record_function` scopes are guarded by `fpct_profile_scopes`. Legacy P2/P3
  wall timing disables scopes; a separate profiler trace enables them.

## Tests

- New R2k equivalent-kernel tests cover FP64 grouped/flat identity; FP32/BF16
  output and gradients; m=0..4; GQA; mixed equivalent/non-equivalent batches;
  parent-logit call count/reuse; zeroed equivalent atom QK; layer semantic map
  reuse; instrumentation-off scopes; and static absence of host-sync APIs and
  `group_value`.
- Existing actual Qwen3 eager/DynamicCache/GQA/MQA integration, production,
  reference, numerical, runner, replicated-collapse, m<=1, forced activation,
  state/config and pre-collapse identity tests remain green.
- FPCT-targeted suite: `190 passed, 3 warnings`.
- CPU-safe full suite: `429 passed, 2 warnings`.
- No tolerance was changed or relaxed.

## Diagnostic runner

`script/experiment/fpct_gpu_r2k_latency.py` implements isolated fresh-process
paired blocks, raw CUDA/wall samples, prospective hardware/process telemetry,
shape-only geometry rows, a separate profiler trace and the frozen block
bootstrap aggregator. Its aggregate classification is always
`DIAGNOSTIC_ONLY`; it cannot emit R2k GO.

## Current authorization

No new GPU, Kubernetes, pretrained forward, training, checkpoint, accuracy,
model-selection or held-out output has been produced by this implementation
revision. The next step is to commit/push this diagnostic code, bind it to a
new diagnostic image/UID/root, and run the preregistered diagnostic only.
