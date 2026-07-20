# FPCT GPU Hardening Report — CPU and Frozen GPU Gates

Status: `GPU_ENGINEERING_BLOCKED`

## Production changes

- Added a reusable `FPCTPackedLayout` containing only structural parent,
  candidate, destination, row-offset, valid and log-prior tensors.
- The layout is built once before layer attention and reused across layers. A
  generated native tail causes one new structural build per decode step, not per
  layer.
- Per-layer packing now uses tensor gather/scatter/indexing only. The
  `pack_fpct_memory` hot path contains no `.tolist()`, `.item()`, `.cpu()`,
  `.numpy()` or parent loop.
- Only certified `m>=2` parents use candidate atoms. `m=1` remains one base slot;
  `m=0` remains native. Candidate children inherit the parent mask and bias.
- Added parameter-free replicated-collapse inference. It expands by the frozen
  prior but repeats the collapsed parent KV, and is mathematically equivalent
  to `C_post`.
- Added off-by-default tensor-only aggregate diagnostics: gamma/prior KL,
  gamma query variance, candidate-logit variance/range, Jensen gap, `D_K`,
  `D_V`, output delta and expansion. No raw KV ledger is retained.
- FP16/BF16 attention dot products, masks, softmax and diagnostic reductions use
  FP32 accumulation.
- Training/evaluation loaders now propagate replicated-collapse and
  instrumentation flags. No trainable parameter was added.

The single scalar synchronization needed to size a padded batch layout occurs
only in the once-per-structure builder, outside per-layer attention. The formal
GPU profiler gate must still verify no packing-caused GPU→CPU synchronization in
the layer hot path.

## Stochasticity and regression gates

- `C_post` and `F` candidate-specific fused KV are bitwise equal after resetting
  the same RNG with projector dropout 0.1 and identical call order.
- `K=1` training-mode `C_post/F` parent outputs are bitwise equal.
- Parent nuisance remains computed/sampled once and broadcast to candidates.
- Default and `c_pre` state dictionaries remain unchanged; replicated-collapse
  and instrumentation are parameter-free and off by default.

## Real Qwen3 CPU integration

Transformers' actual `Qwen3ForCausalLM`, eager attention and `DynamicCache` were
instantiated from random tiny configs without downloading or loading pretrained
weights. Tests covered:

- prefill plus multi-step decode and cache append;
- batch size two with unequal left padding and causal masks;
- GQA (`Hkv=2`) and MQA (`Hkv=1`);
- `c_pre/c_post/f` runtime switches and config roundtrip;
- `m0/m1/m>=2`, exact `m1`, replicated collapse and activation;
- forward/backward gradients to candidate fused KV;
- finite outputs and no new parameters;
- off-by-default mechanism instrumentation.

## Tests

- Reference/production/Qwen/sanitizer targeted: `64 passed`.
- Complete CPU-safe repository suite: `360 passed, 2 warnings`.
- Earlier sealed import hostile suite: `21 passed`.

The statements above describe the CPU/HF hardening checkpoint before the
frozen K8s execution. The later GPU results below supersede only the execution
status; they do not change the CPU implementation evidence.

## Frozen GPU execution

- Run UID: `fpct-cfm-371e72f1-20260720`.
- Scientific code: `371e72f14da41f5509eafa21553c7a7418c9a53e`.
- Run-lock SHA256:
  `2a4db8f26def997c95b590a34718916b772f686f5c00eabb2f2b69f0dfe5e5ec`.
- Image: `docker.io/library/fpct-confirmatory:371e72f1@sha256:c851056733f3b7affc85ae5dabd870043f3ae7d3010d245705f5b9ded8dc36ab`.
- Node/device: `4090-48gx2`, NVIDIA GeForce RTX 4090.
- Actual init and main container image IDs matched the frozen digest.

The final synthetic numerical gate was `GO`. FP32 versus the independent FP32
reference had maximum absolute error `2.384185791015625e-07` and gradient
relative-L2 error `1.5795092167536495e-07`. FP16/BF16 output, gradient and row
sum checks passed; invalid probability/gradient was exactly zero; `m<=1` and
replicated-collapse greedy controls passed. The synthetic activation-null floor
was frozen at `0.0390625` before the first pretrained output.

## Pretrained smoke hard stop

The first real TinyLlama-to-Qwen3 pretrained smoke produced no accuracy or
correctness output, but failed three preregistered engineering/integrity gates:

- activation failed: `output_delta_l2=0`, candidate-logit range/variance and
  query variance were all zero, below the frozen `0.0390625` null floor;
- replicated-collapse failed with maximum output delta `0.71875`;
- CUDA profiling reported `cudaDeviceSynchronize` and
  `cudaStreamSynchronize`.

The controls/resource checks that passed were: finite outputs, `m<=1` exact
control, peak HBM (`4.1550` GiB C_post; `4.1710` GiB F), median latency ratio
`1.2769`, p95 latency ratio `1.2839`, and expanded-slot ratio `1.1458`.

The controller therefore entered terminal `GPU_ENGINEERING_BLOCKED`. This was
not an infrastructure failure, so no retry, matched-training smoke, formal
12-seed training, model-selection, held-out evaluation or checkpoint creation
was permitted.
