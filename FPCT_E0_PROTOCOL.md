# FPCT-E0-FAST exploratory protocol

## Status and isolation

This protocol was frozen before any FPCT-E0 accuracy output was produced or read. FPCT-E0 is an independent exploratory run on `research/fpct-e0-exploratory`, based on `77f03c6b068fffd9bbec210aeb71701e6b8d6bd7`. It does not alter or reinterpret R2m, H1, main, Phase2A, `math.md`, or any confirmatory artifact. Its outputs are not eligible for formal statistics.

Run UID: `fpct-e0-20260722-v1`

Result root: `/netdisk/lijunsi/fpct-e0/fpct-e0-20260722-v1`

Pair: TinyLlama-1.1B-Chat-v1.0 sender to Qwen3-0.6B receiver

Headline contrast: `F - C_post`

## Operator contract

The existing projector/fuser produces already residual-enriched candidate K/V:

\[
(\Delta K_{ij},\Delta V_{ij})=F_\theta(K_i^r,V_i^r,K_j^s,V_j^s),\quad
K^F_{ij}=K_i^r+\Delta K_{ij},\quad V^F_{ij}=V_i^r+\Delta V_{ij}.
\]

The production `C2CProjector.forward()` return must therefore not receive another native residual.

For C_post:

\[
K_i^{post}=\sum_j A_{ij}K^F_{ij},\qquad
V_i^{post}=\sum_j A_{ij}V^F_{ij}.
\]

For F:

\[
z_{tij}=\langle Q_t,K^F_{ij}\rangle/\sqrt d+\log A_{ij}+M_{ti},
\quad p_{tij}=\frac{e^{z_{tij}}}{\sum_{i',j'}e^{z_{ti'j'}}},
\quad o_t=\sum_{i,j}p_{tij}V^F_{ij}.
\]

Each legal candidate uses the same candidate-specific fuser. K/V stay paired. `A` enters once, as a logit prior, and is not multiplied into V after softmax. Children inherit the parent causal/padding mask. The denominator is global across all active parent/candidate atoms. Non-ambiguous parents use one C_post-equivalent atom; invalid candidates have exactly zero probability, contribution, and gradient.

Fixed nuisance conditions are `a=1`, `g=1`, `position_mode=legacy`, no native null, no selector, no new gate/router/loss, no de-RoPE/re-RoPE, `include_response=false`, eager attention, top-k 4, uniform alignment prior, and `certified_slot0_v1`. Sender and receiver are frozen. F adds no trainable parameter. Existing legacy nuisance remains matched and unchanged across arms.

Training loss is response-only next-token cross entropy: prompt labels are `-100`; only the existing projector/fuser is trainable.

## Pre-output gates

The executable oracle in `script/analysis/fpct_e0_formula_oracle.py` must pass in the exact image before training. It checks FP32 oracle/production output and probability parity, BF16 finite forward/backward, GQA ratios 1/2/4, exact-null and identical-candidate collapse, permutation and refinement invariance, invalid probability/gradient zero, inherited causal masking, one global softmax, prior-once use, pre-collapse C_post/F candidate identity, and identical trainable parameter names/shapes. FP32 tolerance is `atol=rtol=1e-6`; BF16 tolerance is `atol=rtol=5e-3`. Failure is terminal `E0_ENGINEERING_BLOCKED` and is not scientific evidence against FPCT.

## Matched training

Seeds and within-pod arm order are frozen:

- 2026072201: C_post, F
- 2026072202: F, C_post
- 2026072203: C_post, F

Each arm uses exactly 2,048 frozen MMLU auxiliary-training sidecar examples, one epoch, 64 optimizer steps, two processes/two GPUs, per-device batch 1, gradient accumulation 16, effective batch 32, learning rate `1e-4`, weight decay `0.01`, linear scheduler, warmup `0.10`, max grad norm `1.0`, max length 1,024, BF16, eager attention, projector dropout `0.1`, and the step-64 checkpoint. No old B2/B3/B6 checkpoint initializes a run.

Within each seed, step-0 trainable tensors, trainable keys, RNG-before-training, data membership/order, rendered training bytes, alignment sidecar, optimizer/scheduler recipe, image, node, and GPU pair must match. Each arm is a fresh Python process. Both must complete step 64 and pass checkpoint save/reload equality.

## Development evaluation

Evaluation uses only the prospectively selected, label-free certified calibration groups in `exploratory_dev_manifest.json`: ARC 128, OpenBookQA 70, MMLU-Redux 128. Selection is SHA-ascending within task and never uses logits, attention, gamma, or correctness. Confirmatory model-selection/test correctness and seeds 45--56 remain sealed.

Each seed evaluates four cells at step 64 only:

- `Y_CC`: C_post-trained, C_post inference
- `Y_CF`: C_post-trained, F inference
- `Y_FC`: F-trained, C_post inference
- `Y_FF`: F-trained, F inference

Evaluation is deterministic greedy zero-shot Non-CoT generation, `max_new_tokens=64`, eager attention, with identical prompt bytes, group IDs, extraction, and generation configuration. Accuracy is first equal-weighted within distinct content group, then within task, then equally over the three tasks.

The frozen estimands are:

\[
T_s=Y_{FF}-Y_{CC},\quad D_{C,s}=Y_{CF}-Y_{CC},\quad
D_{F,s}=Y_{FF}-Y_{FC},\quad O_s=(D_{C,s}+D_{F,s})/2,\quad
I_s=D_{F,s}-D_{C,s}.
\]

`T` is the matched training-plus-inference system effect. `O` is the average immediate query-time operator effect. `I` is training/operator interaction. With `n=3`, no significance claim is permitted.

## Execution, retry, and stopping

The exact image is `docker.io/library/fpct-gpu-r2m:80fb295@sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851`. One two-GPU Job is used per seed on `4090-48gx2`; the two arms and all four evaluation cells execute sequentially in that pod. The immutable E0 ConfigMap is mounted at `/opt/fpct-e0`; the image's `/opt/fpct` is never replaced by a host worktree.

Only one whole-seed retry is allowed, and only for eviction/preemption, node loss, or transient Kubernetes/volume failure. Both arms rerun with byte-identical inputs in a new attempt directory; the old attempt remains quarantined. OOM, NaN/Inf, parity failure, a single-arm failure, integrity mismatch, scientific exception, loss, or accuracy never authorizes retry or recipe changes.

## Frozen decision rule and claim boundary

`E0_GO_EXPLORATORY_SIGNAL` requires all three seeds/four cells, mean `T >= +1.00 pp`, at least 2/3 positive `T`, no task with three-seed mean `T < -2.00 pp`, all numerical/identity/mask/prior-once controls, and nonzero real-m>=2 mechanism activation. Mechanism signal is `positive` only if mean `O>0` and at least 2/3 `O` values are positive; otherwise it is `unresolved`.

If the complete run misses that gate, the result is `E0_NO_GO_FOR_FURTHER_SPEND`: this means only that the current pair, 2,048-example/64-step recipe, and development set did not justify a 36-run confirmatory campaign. It does not prove FPCT ineffective or its true effect zero. Engineering/data/integrity failures are `E0_ENGINEERING_OR_EXECUTION_BLOCKED`; failure to finish within 24 hours solely from GPU capacity is `E0_INCOMPLETE_CAPACITY`.

## Pre-accuracy execution amendment

Commit `c8751b2a933484ca250b2dcf3f80e233e3809cf6` transcribed the materialized development-tree SHA without its final two hexadecimal characters (`73`). Attempt 1 detected this mismatch at the runner's startup input gate, before model loading, training, evaluation, or accuracy output. The materialized data, development-group manifest, rendered configs, operator code, oracle, image, seeds, arm order, budget, analysis, and decision rules were unchanged. Under the protocol's prospective 30-minute allowance for E0 oracle/runner/config wiring repair, the manifest value was corrected to `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`. The original immutable ConfigMap, Jobs, logs, and attempt records remain preserved; execution resumes from a new commit and immutable ConfigMap without reading accuracy.
