# FPCT-GPU-R2g Parent-First Exact-Null Addendum

## Status and prospective boundary

This addendum is frozen after terminal R2f and before any R2g GPU numerical,
pretrained, training, checkpoint, accuracy or correctness output. R2f remains
immutable at run UID `fpct-r2f-d08b22b-v1` with terminal result SHA256
`73af6279b5d30c41271b17810a747884d05f83b1e2afe338e048399baa2d35df`.
R2g must use a new scientific SHA, image, run-lock, run UID and artifact root.

No threshold, metric floor, panel, model, tokenizer, data, operator, seed,
training recipe or statistical rule is changed.

## Frozen diagnostic fact

R2f completed 16 isolated conditions and five profiles. All hard checks passed
except `expected_native_null`. In FP32, checkpoint-native `Delta_fact` was
`4.291534423828125e-5`, while its pre-output threshold was `4.0e-5`; BF16 was
exactly zero.

The post-output tensor-only audit established:

- all 504 FP32 panel-by-layer fused candidate K/V tensors equal native parent
  K/V elementwise;
- all collapsed K/V tensors equal native parent K/V elementwise;
- reconstructed packed records are all parent-equivalent;
- C_post, replicated-atoms and collapse-bypass logits are exactly equal;
- the FP32 deviation appears only after entering the hierarchical function and
  first crosses `2e-5` at deep layers.

## Prospective R2g hypothesis

R2f computes the shared parent eager attention only after atom logits, grouped
reductions, beta/gamma and hierarchical value reduction. Even though the final
tensor selection chooses the parent result for an all-equivalent sample, this
changes the GPU numerical call order relative to C_post. R2g tests whether
computing the same parent eager adapter before all hierarchical atom/group
kernels restores the frozen checkpoint-native null without changing the
factorized result for non-equivalent parents.

This is a numerical engineering hypothesis, not a performance hypothesis.

## Single authorized scientific change

`fpct_qwen_hierarchical_attention_forward` must compute the existing shared
`fpct_qwen_eager_attention_forward` parent result at function entry, before any
atom/group matmul, logsumexp, scatter, beta/gamma or value reduction. The
function must then compute the unchanged hierarchical result and use the same
tensor-only mixed-batch selection:

`output = where(all_parent_equivalent, parent_output, hierarchical_output)`.

The change must not:

- add a host scalar branch or synchronization;
- alter beta/gamma equations, priors, masks or diagnostics;
- skip the hierarchical path for mixed batches;
- introduce trainable parameters;
- change C_pre, C_post, fuser, gate, selector, position or null contracts;
- change any frozen numerical tolerance.

## Validation and execution gate

Before building an image:

1. exact parent-equivalent output remains bitwise equal to the parent adapter;
2. a call-order regression proves the parent adapter is entered before grouped
   matmuls;
3. flat/global versus hierarchical, gradients, replicated, m<=1, GQA/MQA,
   CPU Qwen integration and legacy-default regressions pass;
4. the CPU-safe full suite passes;
5. hot-path static checks remain clean.

R2g must then repeat the complete synthetic GPU gate and the complete 16-cell
pretrained matrix plus P2--P6 profiles. `expected_native_null` keeps the same
metric-specific floor. Any failure again terminates the new controller and
blocks matched smoke and formal training.

## Claim boundary

Passing R2g would establish numerical recovery and engineering readiness only.
Forced-on activation remains a canary, not performance evidence. Matched smoke,
formal training, model-selection and held-out evaluation remain conditionally
blocked behind the full R2g pretrained GO.
