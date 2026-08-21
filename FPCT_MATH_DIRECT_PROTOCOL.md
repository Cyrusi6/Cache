# FPCT-MATH-DIRECT — exact math.md operator matched pilot

Status: PRE-OUTPUT LOCK
Classification: exploratory, single pair, single seed; not confirmatory.

## Scientific question

On TinyLlama-1.1B sender → Qwen3-0.6B receiver, does matched training
with the first-round operator specified by `math.md` improve accuracy when the
candidate axis is preserved until receiver-query attention?

The headline contrast remains `F - C_post`. Both arms use exactly the same
candidate-specific projector/fuser, frozen alignment prior, position transport,
data, initialization, optimizer, budget and eager attention backend. Their only
operator difference is collapse before attention (`C_post`) versus global
query-time attention over legal candidate atoms with one `log A_ij` term (`F`).

## Frozen operator

`fpct_position_mode=math` means:

1. inverse-RoPE sender candidate K at its actual sender position;
2. inverse-RoPE receiver parent K at its actual receiver position;
3. shared candidate-specific nonlinear fusion in content space;
4. re-RoPE every fused candidate K at the receiver parent position;
5. `C_post` averages fused candidates with normalized `A` before attention;
6. `F` retains the candidate axis and applies one global softmax over all legal
   child atoms with `log A` included exactly once.

The first round fixes `a=1`, `g=1`, `include_response=false`, eager attention,
top-k 4 uniform frozen alignment, and `certified_slot0_v1`. It introduces no
native-null candidate, selector, new gate, F-only parameter, old-projector
initialization, or extra loss.

## Matched training

- seed/projector-init seed: `2026082101`;
- fresh C_post and F projector states;
- 2,048 MMLU auxiliary_train examples from the frozen E0 training membership;
- 1 epoch, 64 optimizer steps;
- two processes / two GPUs, per-device batch 1, accumulation 16, global batch 32;
- AdamW, LR `1e-4`, weight decay `0.01`, linear schedule, warmup `0.10`;
- BF16, max length 1024, max grad norm 1.0, dropout 0.1;
- sender and receiver frozen;
- arms run serially on the same two-GPU node.

Both formal integrity records must agree on trainable keys, step-0 tensor hash,
data-order hash and initial optimizer/scheduler state. Any mismatch makes the
experiment INCONCLUSIVE.

## Evaluation and decision

Only the already-open exploratory E0-design set is used: ARC 128, OpenBookQA 70,
MMLU-Redux 128 distinct content groups. E1-pilot, model-selection and test remain
sealed.

The four frozen cells are:

- `Y_CC`: C_post-trained checkpoint + C_post inference;
- `Y_CF`: C_post-trained checkpoint + F inference;
- `Y_FC`: F-trained checkpoint + C_post inference;
- `Y_FF`: F-trained checkpoint + F inference.

Each Y is equal-weight accuracy across groups within task, then equal weight
across the three tasks. Report:

`T = Y_FF - Y_CC`

`D_C = Y_CF - Y_CC`

`D_F = Y_FF - Y_FC`

`O = (D_C + D_F) / 2`

`T` is the matched system effect. `O` is the same-checkpoint query-time
factorization effect. One seed is positive only if both `T>0` and `O>0`.
Only that condition permits a separate prospective decision to add two matched
seeds. No significance, cross-pair or universal claim is permitted.

## Provenance and stopping

The implementation commit is
`5903dcb4892e400cbbee194357c057e21957ad7f`. The run uses a clean immutable
Git snapshot of the subsequent pre-output lock commit. Alignment sidecar SHA256
is `48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`;
E0-design materialized tree SHA256 is
`f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`.

Stop without interpreting performance on provenance mismatch, missing checkpoint,
nonfinite loss/gradient, mask/prior invariant failure, unequal matched integrity,
or incomplete evaluation cell. Negative and null results are retained.
