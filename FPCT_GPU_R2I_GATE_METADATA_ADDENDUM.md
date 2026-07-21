# FPCT-GPU-R2i Hard-Gate Parent-Equivalence Metadata Addendum

## Prospective boundary

This addendum is frozen after terminal R2h and before any R2i GPU,
pretrained, training, checkpoint, accuracy or correctness output. R2h remains
immutable with result SHA256
`967ebe70421923390a47543d78f6e5f3dca8100addfffeb473fb434e5448067b`.
R2i requires a new scientific SHA, image, run-lock, run UID and artifact root.

No threshold, operator equation, panel, prior, model, tokenizer, data, training
recipe, seed or statistical rule changes.

## Frozen evidence and hypothesis

R2h reproduced the frozen FP32 checkpoint-native delta
`4.291534423828125e-5 > 4.0e-5`, while BF16 was zero. In the FP32 OP02 trace,
all 504 panel-layer cells had zero hard key/value gates, and candidate plus
collapsed K/V were elementwise identical to the trace-time native parent.
Therefore candidate canonicalization succeeded, but the packed-attention path
did not retain the semantic fact that a hard-zero parent must use the exact
native branch against the final parent cache.

R2i prospectively tests one engineering hypothesis: explicit parameter-free
hard-gate parent-equivalence metadata must travel with each sidecar segment and
participate in the packed parent-equivalence mask.

## Single scientific intervention

At the shared C_post/F candidate boundary, compute a boolean parent field:

`parent_force_native[b,i] = all_h(key_gate[b,h,i] == 0) AND all_h(value_gate[b,h,i] == 0)`.

The field is stored in `FPCTSidecarSegment`. During tensor-only packing, it is
ORed with the existing exact K/V comparison when constructing
`parent_equivalent`. A forced-native parent uses the current final parent
cache/logit/value in the hierarchical beta/gamma adapter; an all-forced-native
sample returns the already computed shared C_post parent adapter output.

The intervention:

- applies identically to C_post and F candidate construction;
- does not change candidate tensors, priors, masks or trainable parameters;
- is false for nonzero/training gates and therefore preserves factorized
  candidates;
- introduces no host scalar branch, `.item()`, `.cpu()` or `.numpy()` in the
  attention hot path;
- does not change replicated, bypass, m<=1 or forced-on conditions.

## Required validation

- hard-zero projection must emit `parent_force_native=true` and nonzero gates
  must emit false;
- deliberately residual candidate tensors with true metadata must return the
  parent adapter bitwise, while false metadata must remain factorized;
- metadata shape, dtype and device contracts must hard fail;
- C_post/F pre-collapse identity, K=1, replicated, global/hierarchical,
  gradients, GQA/MQA, actual Qwen CPU integration and legacy-default
  regressions must pass;
- CPU-safe full suite and hot-path static checks must pass.

The new execution repeats the complete synthetic GPU gate and all 16
pretrained conditions plus P2--P6. The checkpoint-native floor remains
`4.0e-5`; any failure blocks matched smoke and training without threshold
changes.

## Claim boundary

This metadata only carries the already frozen semantics of an exact hard-zero
legacy gate. It is not an F-only mechanism, does not constitute task evidence,
and cannot justify a performance claim. Formal mechanism evidence remains
conditional on matched training and the checkpoint-native O contrast.
