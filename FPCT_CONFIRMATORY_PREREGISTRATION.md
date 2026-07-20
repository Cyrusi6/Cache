# FPCT single-pair confirmatory preregistration

Status: prospective and output-sealed. This document becomes operative only together with `recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json` and a committed `confirmatory_run_lock.json` that binds an immutable scientific-code commit to an immutable container digest. No pretrained output, training loss, accuracy, or held-out result may be inspected before that run lock is committed and pushed.

## Scientific question and arms

The confirmatory population is limited to TinyLlama-1.1B-Chat-v1.0 sender to Qwen3-0.6B receiver, the frozen three-task suite, the frozen distinct-content-group splits, 2,048 MMLU `auxiliary_train` training examples, and the specified 64-step projector-training budget. This is a single-pair confirmation and cannot establish cross-model universal validity.

All arms use `a=1`, `g=1`, `position_mode=legacy`, `include_response=false`, eager attention, frozen sender/receiver weights, `certified_slot0_v1`, and the same candidate-specific projector/fuser family. No native null, selector, new gate, F-only trainable parameter, de-RoPE/re-RoPE, or learned candidate router is permitted.

- `C_pre`: collapse source candidates under normalized `A_ij`, then invoke the shared nonlinear fuser once.
- `C_post`: invoke the shared candidate-specific fuser for every legal candidate, collapse fused K/V under `A_ij` before attention, and expose one slot per parent.
- `F`: use the same pre-collapse fused candidates as `C_post`, preserve certified candidate factorization until the receiver query, add `log A_ij` exactly once, and normalize native slots plus all legal child atoms in one global softmax denominator.

The unique headline internal operator contrast is `F-C_post`. `C_post-C_pre` identifies candidate-specific nonlinear fusion placement. `F-C_pre` is an overall system difference and cannot by itself identify mechanism.

## Frozen matched design

Fresh formal seeds are `45..56`. For every seed the three arms originate from the same fresh step-0 projector initialization, reset RNG, data membership/order, optimizer/scheduler recipe, two-GPU hardware pair, image, model assets, and certified alignment sidecars. Old B2/B3/B6 checkpoints are excluded from initialization, selection, and formal statistics.

The within-pod order is balanced:

- 45, 51: `c_pre,c_post,f`
- 46, 52: `c_pre,f,c_post`
- 47, 53: `c_post,c_pre,f`
- 48, 54: `c_post,f,c_pre`
- 49, 55: `f,c_pre,c_post`
- 50, 56: `f,c_post,c_pre`

Each arm uses 2,048 fixed MMLU auxiliary-training examples, one frozen seed-specific order, one epoch, exactly 64 optimizer steps, two processes, per-device batch size 1, gradient accumulation 16, effective global batch 32, learning rate `1e-4`, weight decay `0.01`, linear scheduling, warmup ratio `0.10`, max grad norm `1.0`, max length 1,024, projector dropout `0.1`, BF16, and FP32 prior normalization/softmax/logsumexp/loss reduction. Step 0 and step 32 may be retained only for checksum and diagnostic integrity. The only formal checkpoint is completed step 64.

All 36 formal runs must complete as 12 matched triplets before performance evaluation is released. One infrastructure-only retry of an entire seed triplet is allowed for eviction, preemption, transient storage/network failure, or node failure. Numerical failure, OOM, bad results, or a single-arm failure may not trigger a selective retry or recipe change. Fewer than 12 complete triplets gives `INCONCLUSIVE`; a subset is not a formal analysis.

## Evaluation firewall and cells

The frozen group counts are:

| split | ARC | OpenBookQA | MMLU-Redux |
|---|---:|---:|---:|
| model-selection | 186 | 77 | 815 |
| held-out test | 453 | 195 | 2,296 |

Correctness is first computed within distinct content group, then group-equal within task, then equal-weight across the three tasks. The primary population contains at least one certified `m>=2` parent. Overall, sample-weighted, per-task, and `m<=1` controls are secondary.

For each seed:

- `Y_P`: C_pre-trained, C_pre inference.
- `Y_CC`: C_post-trained, C_post inference.
- `Y_CF`: C_post-trained, F inference.
- `Y_FC`: F-trained, C_post inference.
- `Y_FF`: F-trained, F inference.

The preregistered estimands are:

- `T_s = Y_FF,s - Y_CC,s`, matched total FPCT system effect.
- `D_C,s = Y_CF,s - Y_CC,s`, direct operator effect at the C_post-trained checkpoint.
- `D_F,s = Y_FF,s - Y_FC,s`, direct operator effect at the F-trained checkpoint.
- `O_s = (D_C,s + D_F,s)/2`, query-time operator main effect.
- `I_s = D_F,s - D_C,s`, training-adaptation interaction.
- `N_s = Y_CC,s - Y_P,s`, candidate-specific nonlinear-fusion placement effect.

`T>0` alone is not a query-time mechanism claim. If `T>0` but `O` is unresolved, the result is limited to matched training/system improvement.

## Frozen inference and statistics

The hierarchical paired bootstrap uses 50,000 replicates and seed `20260719`: resample training seeds first, then distinct content groups within each task, using the same group draw jointly for all cells; tasks remain equally weighted. Percentile 95% intervals and every seed-level value are reported.

The exact 4,096 sign-flip test is not an assumption-free exact test of the composite mean null `E[T]<=0`. It tests paired deltas under the sharp/symmetric null with sign-exchangeability. The mean is the target estimand; `mean(T)>=+1.00pp` remains a separate practical-effect gate. The report must include the preregistered one-sided and two-sided exact sign-flip results, hierarchical bootstrap interval, one-sided paired t-test sensitivity (independent seed deltas, approximate normality of their mean), and exact sign-test sensitivity (exchangeable independent signs, continuous/no-tie null). No test is selected after seeing results.

Model-selection releases held-out test mechanically unless one of these holds:

1. one-sided 90% UCB of `T` is below `+0.50pp`;
2. `mean(T)<=-1.00pp` and at least 9/12 seed deltas are negative;
3. any numerical, control, resource, provenance, split, or run-integrity failure.

Held-out primary testing is performed once. `PERFORMANCE_GO` requires all of:

1. `mean(T)>=+1.00pp`;
2. one-sided exact sign-flip `p<=0.05`;
3. hierarchical 95% CI lower bound above zero;
4. at least 9/12 `T_s>0`;
5. no task has mean decline greater than 1pp with at least 8/12 seeds declining in that task.

Only after `PERFORMANCE_GO`, fixed-sequence gatekeeping tests `O` with the same one-sided exact sign-flip alpha `0.05`; no additional Bonferroni correction is applied. `MECHANISM_SUPPORTED_GO` additionally requires the `O` test and interval, `mean(D_F)>0`, at least 9/12 positive `O_s`, mechanism diagnostics above their pre-output synthetic numerical-null floor, replicated collapse returning to C_post, `m<=1` returning to C_post, and exact-identity control remaining K=1.

Final classifications are `MECHANISM_SUPPORTED_GO`, `PERFORMANCE_GO_MECHANISM_UNRESOLVED`, `FUTILITY_NO_GO`, `HARM_NO_GO`, or `INCONCLUSIVE`, using the frozen manifest logic. The design has approximate 83% power for SD=2pp, true effect=1.5pp, and one-sided alpha .05 under a normal approximation; it cannot exclude every real effect below 0.5pp.

## Claim boundary

Exact tokenizer identity means index alignment is identity, not that model KV spaces are equal; P_K/P_V and existing cross-model positional transport remain necessary. `certified_slot0_v1` is a conservative causal-isolation device, not a universal aligner. Character-interval partition is sufficient but not necessary, so real byte-fallback one-to-many cases may be discarded. The headline concerns certified ambiguity only. Sanitized C_pre is not guaranteed to equal historical unsanitized v2.2, and the three-arm experiment cannot alone claim superiority to original C2C/v2.2.
