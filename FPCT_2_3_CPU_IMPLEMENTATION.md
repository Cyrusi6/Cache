# FPCT-2/3 CPU Production Path

## Status and claim boundary

FPCT-2 and FPCT-3 are `GO` for the CPU implementation gate. This status means that the factorized operator has an isolated production seam and that the pure-tensor/reference, masking, degeneration, nuisance-sharing, gradient and legacy-default tests pass. It does not establish real-model activation, task-accuracy improvement, GPU numerical behavior or training stability.

The only headline scientific contrast remains `F-C_post`. `C_post-C_pre` isolates candidate-specific nonlinear fusion. `F-C_pre` is an overall system contrast and is not sufficient for mechanism attribution.

## Frozen production behavior

- `fpct_operator` accepts only `c_pre`, `c_post` or `f`; an unset flag retains the legacy path.
- Unset/`c_pre` uses the existing gather-average-project-write path.
- `c_post` and `f` call the same candidate-specific projector helper and produce elementwise-identical fused candidates under matched parameters, state and randomness.
- Parent entropy, confidence, legacy scalar gate and Gumbel sample are computed once and reused for every candidate. The nonlinear fuser core remains candidate-specific.
- Raw alignment `A` is masked and L1-normalized. Learned alignment, weight calibration, injection/transfer gates and Route3 routing are rejected by the FPCT path.
- `c_post` collapses candidate-specific fused KV with `A` before ordinary attention.
- `f` stores `[B,Hkv,N,K,D]` fused KV plus `A`, validity and parent location in a non-parameter sidecar. At attention, only `m>=2` parents replace their placeholder with legal child atoms.
- `m=1` occupies one slot. `m=0` retains the original receiver-native fallback. No receiver-native sibling/null is added.
- Every child inherits its parent attention-mask/bias column and adds `log A` once. Values are not multiplied by `A` after softmax.
- Generated/native slots and legal children share one global softmax denominator.
- Children are not inserted into `DynamicCache` as sequence positions and receive no independent position IDs. First-round position mode remains `legacy`, with `a=1` and `g=1`.
- GQA/MQA uses the existing `Hq/Hkv` repetition rule at attention computation.
- No operator-specific trainable parameter is added; default, `c_pre` and `f` state-dict keys and initialized tensors are identical under a matched seed.

## Implementation surface

- Sidecar and packed/global attention: `rosetta/model/fpct_attention.py`
- Candidate projection and attention hook: `rosetta/model/wrapper.py`
- Parent nuisance capture/reuse: `rosetta/model/projector.py`
- Configuration plumbing: `rosetta/train/model_utils.py`, `rosetta/utils/evaluate.py`, `script/train/SFT_train.py`
- Reference oracle: `script/analysis/fpct_reference_operator.py`
- CPU production tests: `test/test_fpct_production_path.py`
- Reference tests: `test/test_fpct_reference_operator.py`

## Verification

Targeted command:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/lijunsi/projects/Cache-fpct-factorized-transport \
  /home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python -m pytest -q --no-cov \
  --basetemp=local/tmp/pytest-fpct-targeted \
  test/test_fpct_reference_operator.py \
  test/test_fpct_production_path.py \
  test/test_fpct_1b_structural_support_audit.py
```

Result: `52 passed`.

The existing CPU-safe suite was run separately with the Phase2A outcome audit excluded by the research-line firewall. Result: `288 passed, 2 warnings`.

The tests cover reference/production equality, dense/ambiguous-only packing, `K=1` in evaluation and training modes, shared parent Gumbel nuisance, flat/hierarchical equivalence, replicated collapse, refinement and permutation invariance, causal/padding/zero-support behavior, all-invalid stability, GQA/MQA shapes, exact invalid gradients, state-dict/default regression, configuration serialization, finite outputs and single use of the candidate prior.

## Deferred GPU boundary

No Hugging Face/LLM forward, model weight, checkpoint, CUDA kernel, GPU, Kubernetes job, training or formal accuracy evaluation was used. Float16/bfloat16 tolerances, GPU kernel equivalence and training parameters remain subject to a separate prospective human lock.
