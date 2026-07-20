# FPCT GPU R2f Hierarchical Attention Addendum

Status: `PROSPECTIVE_PRE_OUTPUT`

This addendum is frozen after terminal R2e-v2 and before any R2f pretrained
forward. It implements the global-equivalent hierarchical beta/gamma form that
was already allowed by the original operator contract.

## Frozen implementation

For every query/head, atom logits remain

`s_ij + log A_ij + parent_mask`.

Within each parent, R2f computes gamma by a stable FP32 grouped logsumexp. The
group logit is then passed through one FP32 beta softmax across all parents.
The output is beta-weighted query-specific group values. The returned atom
probability is `beta_i * gamma_ij`, so it has the same global denominator as the
flat formulation.

Parents whose active atoms are exactly equivalent to the collapsed parent use
the parent logit and value exactly. If every parent in a sample is equivalent,
the output is selected from the same parent eager adapter as C_post. Distinct
candidate parents remain fully factorized.

Inactive atoms have `log_prior=-inf` and are excluded from D_K, D_V, gamma and
candidate-count diagnostics. This restores the synthetic null definition; no
threshold is changed.

## Boundaries

- FP32/BF16 tolerances, panel, models, data, seeds, resource gates, training
  recipe and accuracy firewall are unchanged.
- No F-only parameter or route is added.
- Dropout/candidate fusion order and parent nuisance remain unchanged.
- R2f requires a new SHA/image/run-lock/run UID and full GPU-gate restart.

At freeze time no R2f GPU/pretrained/training/checkpoint/accuracy output existed.
