# FPCT GPU R2e Prospective Recovery Addendum

Status: `PROSPECTIVE_PRE_OUTPUT`

This addendum is frozen after terminal R2d and before any R2e pretrained
forward. R2 through R2d remain immutable and non-resumable. The changes remain
inside the preregistered shared-adapter, refinement-invariance and scoped
host-synchronization gates.

## Frozen repair

1. When no sidecar exists yet, both C_post and F patch every receiver layer with
   the same FP32 eager adapter. The initial section therefore cannot diverge by
   backend before candidate sidecars are constructed.
2. Normal F still builds the certified packed layout. Within that layout, a
   parent group whose legal candidate K and V are all exactly equal to the
   collapsed parent is dynamically represented by one active atom with zero
   log prior. This is a tensor-only algebraic refinement collapse; no CUDA
   scalar is read by Python. Distinct candidate groups remain fully expanded.
3. The replicated diagnostic constructs the fully expanded identical atoms and
   log priors, computes the expanded global FP32 probability, groups that mass
   back to each parent and compares it with parent attention probability. The
   frozen FP32/BF16 tolerances remain `2e-5`/`2e-2`. Its returned output remains
   the analytic parent output so numerical-null perturbations are not fed into
   later layers.
4. Static residual-scale constants use device-native tensor creation rather
   than `torch.tensor(..., device=cuda)`. No parameter or state-dict key is
   added.

## Unchanged boundaries

- Canonical prior, operator definitions, diagnostic panel, models, data, seeds,
  resource gates, training recipe and accuracy firewall are unchanged.
- Forced-on remains an engineering canary only.
- Any R2e execution requires a new scientific SHA, image, run-lock, run UID and
  complete restart from the synthetic GPU gate.
- Any failed hard check terminates R2e and forbids training.

At freeze time there was no R2e pretrained output, GPU execution, training,
checkpoint, accuracy or correctness result.
