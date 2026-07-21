# FPCT-GPU-R2h Hard-Gate-Zero Canonicalization Addendum

## Prospective boundary

This addendum is frozen after terminal R2g-v2 and before any R2h GPU,
pretrained, training, checkpoint, accuracy or correctness output. R2g-v2
remains immutable with result SHA256
`8c510e8fb6fffa76b3f3a4b39a5cab64fcf0a6c7ffaea90dd4dc0b3b27f39375`.
R2h requires a new scientific SHA, image, run-lock, run UID and root.

No threshold, operator equation, panel, prior, model, tokenizer, data, training
recipe, seed or statistical rule changes.

## Frozen evidence and hypothesis

Moving the parent eager adapter ahead of hierarchical kernels did not change
the FP32 checkpoint-native delta: R2g-v2 reproduced the exact R2f value
`4.291534423828125e-5`, above the frozen `4.0e-5` floor. Thus parent call order
is falsified as the cause.

All 504 FP32 checkpoint-native panel-layer trace cells nevertheless record
nonzero fused-versus-native RMS (K maximum `4.8113e-7`, V maximum
`1.8114e-7`) while the hard legacy key/value gates are zero. H1 already defines
a zero checkpoint-native gate as the exact native path. R2h tests whether
enforcing that mathematical identity at the shared candidate sidecar boundary
removes the false factorized signal.

## Single scientific intervention

After the shared candidate-specific fuser and parent nuisance have been
computed, broadcast the existing parent hard gate to candidates and apply:

- if key gate equals zero, every candidate key is set exactly to the native
  parent key;
- if value gate equals zero, every candidate value is set exactly to the
  native parent value;
- otherwise the candidate-specific fused tensor is unchanged.

The selection must be tensor-only `torch.where`; it may not call `.item()`,
`.cpu()`, `.numpy()` or create a host branch. It applies identically to C_post
and F before collapse/preservation, adds no parameter, and does not alter
forced-on or nonzero/training gates.

Trace tensors must be cloned at capture time so later cache mutation cannot
change the diagnostic snapshot.

## Required validation

- a deliberately perturbed fake fuser with hard gate zero must produce exact
  native candidate and collapsed tensors;
- the same fuser with nonzero gate must retain candidate differences;
- C_post/F pre-collapse identity, K=1, replicated, global/hierarchical,
  gradients, GQA/MQA, real Qwen CPU integration and legacy-default regressions
  must pass;
- CPU-safe full suite and hot-path static checks must pass.

The new execution must repeat the complete synthetic GPU gate and all 16
pretrained conditions plus P2--P6. The checkpoint-native floor remains
`4.0e-5`; failure blocks matched smoke and training without threshold changes.

## Claim boundary

This intervention only enforces the already frozen native semantics of a hard
zero gate. It is not an F-only advantage and cannot be performance evidence.
Formal query-time mechanism claims remain conditional on matched training and
the checkpoint-native O contrast.
