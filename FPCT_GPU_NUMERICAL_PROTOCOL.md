# FPCT GPU numerical and activation protocol

This protocol is prospective. It may run only from a committed and pushed run lock whose scientific-code SHA, image digest, runtime attestation, model/tokenizer/data assets, certified sidecars, split IDs, thresholds, and analysis code all verify.

## Synthetic numerical gate

The synthetic gate covers forward and backward reference comparisons, global-denominator equivalence, prior single-use, padding/causal/zero-support masks, invalid probability and gradient, replicated collapse, `m<=1`, GQA/MQA, and no NaN/Inf.

Fixed tolerances:

- GPU FP32 vs FP32 reference: `atol=2e-5`, `rtol=2e-5`.
- FP16 vs FP32: `atol=5e-3`, `rtol=5e-3`.
- BF16 vs FP32: `atol=2e-2`, `rtol=2e-2`.
- FP16 probability row-sum error `<=2e-3`.
- BF16 probability row-sum error `<=5e-3`.
- FP16 gradient relative-L2 error `<=0.02`.
- BF16 gradient relative-L2 error `<=0.05`.
- Masked/invalid probability and gradient are exactly zero.

Failure cannot be repaired by relaxing a tolerance. Greedy tokens must be identical for replicated-collapse vs C_post and for `m<=1` C_post vs F.

## Resource and kernel gate

- certified mean expansion `<=1.35`;
- p95 expansion `<=1.50`;
- peak HBM below 90% of physical memory and `<=22.5GiB` on a 24GB GPU;
- F/C_post median latency ratio `<=1.50`;
- F/C_post p95 latency ratio `<=1.75`;
- no OOM;
- no FPCT-caused GPU-to-CPU synchronization in the per-layer hot path.

The packed layout is built once per structural source length and reused across layers. The layer hot path must contain no `.tolist()`, `.item()`, `.cpu()`, `.numpy()`, or Python parent loop. Only certified `m>=2` rows expand; `m=1` is one slot and `m=0` uses native fallback.

## Pretrained unlabeled smoke

Only the frozen TinyLlama to Qwen3 pair is permitted. Before accuracy is visible, run fixed unlabeled prefill/decode, batch/padding, GQA/cache, C_post/F switching, replicated collapse, `m<=1`, certified `m>=2`, and save/reload checks. Freeze an activation floor from synthetic null controls before the first pretrained output. On certified ambiguity, real F-C_post activation and at least one of gamma/KL/Jensen/query dependence must exceed the floor; replicated collapse and `m<=1` must return to C_post.

Then run diagnostic seed `104729`, 128 fixed training examples, each arm for four optimizer steps. The gate checks identical step-0 trainable tensors, keys, data order, optimizer/scheduler state, C_post/F pre-collapse fused candidates, finite loss/gradient, checkpoint reload, latency/HBM/expansion, and mask integrity. Loss magnitude is not a progression criterion.
