# FPCT GPU Hardening Report — CPU Gates

Status: `CPU/HF HARDENING GO; GPU NOT YET STARTED`

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

No pretrained model forward, CUDA/GPU, Kubernetes, training, checkpoint or
accuracy evaluation has occurred. GPU numerical/resource/profiler gates remain
pending the separate frozen formal execution manifest.
