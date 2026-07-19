# FPCT-1C Pure-Tensor Operator Contract

## Scope and status

FPCT-1C defines an independent CPU-only mathematical oracle for `C_pre`, `C_post`, `F`, and the replicated-collapse diagnostic. It does not instantiate a Hugging Face model, load model weights/checkpoints, or modify the first-round production path.

Status: `GO` after all float64/float32, forward, gradient, mask, degeneration, refinement, permutation, GQA/MQA and future-`g=0` tests passed at the prospectively approved tolerances.

The normative scientific contrast remains `F-C_post`. Replicated-collapse is a diagnostic for excluding slot-expansion/kernel-shape explanations; it is not a fourth training arm.

## Tensor contract

| Symbol | Shape | Meaning |
|---|---|---|
| `Q` | `[B,Hq,T,D]` | receiver queries |
| `K^R,V^R` | `[B,Hkv,N,D]` | native/base parent slots and zero-support fallback |
| `K^S,V^S` | `[B,Hs,N,K,Ds]` | source candidates |
| `K^f,V^f` | `[B,Hkv,N,K,D]` | candidate-specific fused KV |
| `A,valid` | `[B,N,K]` | raw alignment prior and candidate legality |
| `b,m` | broadcastable to `[B,Hq,T,N]` | parent bias and causal/padding mask |

The reference accepts a shared fuser callable

`Phi(K^R,V^R,K^S,V^S) -> (K^f,V^f)`.

For candidate-specific execution, native/base inputs are broadcast across `K`; the same callable and parameters are used by `C_post` and `F`. `Hq` must be divisible by `Hkv`; GQA/MQA expansion repeats KV heads only for attention computation.

Candidate legality is:

`legal = valid AND finite(A) AND A>0`.

Illegal mass is set to exact zero before row normalization. A zero-support parent retains its receiver-native slot as the common fallback. A candidate-bearing parent does not also receive a native sibling in the first-round `g=1` contract.

## Operators

### C_pre

First normalize legal `A`, then collapse source KV:

`Kbar_i = sum_j A_ij K^S_ij`, `Vbar_i = sum_j A_ij V^S_ij`.

Call the shared fuser once per parent:

`(Kpre_i,Vpre_i)=Phi(K^R_i,V^R_i,Kbar_i,Vbar_i)`.

Each candidate-bearing parent contributes one transported slot. Zero-support parents contribute their original native fallback. All visible slots use one global softmax denominator.

### C_post

Call the same fuser independently for each legal candidate:

`(Kf_ij,Vf_ij)=Phi(K^R_i,V^R_i,K^S_ij,V^S_ij)`.

Collapse before attention:

`Kpost_i=sum_j A_ij Kf_ij`, `Vpost_i=sum_j A_ij Vf_ij`.

The result is one transported slot per candidate-bearing parent, using the same ordinary global attention as `C_pre`.

### F

Retain the exact same `Kf_ij,Vf_ij`. For each legal child:

`s_tij = Q_t dot Kf_ij / sqrt(D) + b_ti + m_ti + log A_ij`.

Zero-support native fallback slots and all legal children share one flat global softmax denominator. `A` enters exactly once through `log A`; values are not multiplied by `A` after softmax.

The hierarchical oracle is equivalent:

- `gamma_t(j|i) = softmax_j(log A_ij + q_t Kf_ij/sqrt(D))`;
- parent transport score is `logsumexp_j(log A_ij + q_t Kf_ij/sqrt(D)) + b_ti + m_ti`;
- parent/native probabilities use one global softmax;
- child probability is parent probability times `gamma`.

### Replicated-collapse diagnostic

Compute `Kpost_i,Vpost_i`, replicate that identical pair across each legal child, and expand using the original `A`. Because child logits and values are identical within a parent and `sum_j A_ij=1`, the expanded result must equal `C_post`.

## Frozen invariants

The following all passed:

1. flat global `F` equals hierarchical beta/gamma;
2. if every parent has `m<=1`, `C_pre=C_post=F`;
3. an affine fuser gives `C_pre=C_post`;
4. replicated-collapse equals `C_post`;
5. splitting one candidate into identical duplicates while splitting its prior mass is invariant;
6. candidate permutation with the same permutation of `A/valid` is invariant;
7. every child inherits its parent's causal/bias mask;
8. invalid/padding/zero-support child probability and invalid-candidate gradients are exactly zero;
9. native fallback atoms and source children share one denominator;
10. flat/hierarchical forward values and gradients agree; PyTorch gradcheck passes;
11. the Jensen gap is nonnegative and is strictly positive for distinct candidate logits;
12. the future general reference recovers the exact receiver-native path at `g=0`.

The future `g=0` test is not part of the first-round production operator. It exists only as a correctness contract for a separately authorized future native-sibling extension.

## Numerical rules

| Precision/check | Frozen rule | Result |
|---|---|---|
| float64 | `atol=1e-10`, `rtol=1e-8` | pass |
| float32 | `atol=2e-5`, `rtol=2e-5` | pass |
| gradcheck | `eps=1e-6`, `atol=1e-5`, `rtol=1e-3` | pass |
| invalid probability/gradient | exactly zero | pass |
| fp16/bfloat16 | deferred until a separate pre-GPU lock | not evaluated |

No tolerance was relaxed after observing a failure.

## Implementation and tests

- Reference: `script/analysis/fpct_reference_operator.py`
- Tests: `test/test_fpct_reference_operator.py`
- Targeted result: `19 passed`

This GO establishes numerical correctness of the reference equations only. It does not show that the mechanism is activated in real model states and does not establish task-accuracy improvement.
