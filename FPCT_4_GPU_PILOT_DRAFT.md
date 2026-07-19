# FPCT-4 GPU Pilot Draft — Non-Operative

## Status

`REVIEW REQUIRED`. This document is a non-operative design draft produced after FPCT-1B `SINGLE_PAIR_PILOT_READY`, FPCT-1C `GO`, and FPCT-2/3 CPU `GO`. It does not authorize GPU use, Kubernetes, model/checkpoint loading, training, accuracy evaluation or creation of a job manifest.

TinyLlama is the only eligible pilot pair. This supports a single-pair pilot only, not a cross-pair confirmatory claim. `F-C_post` remains the unique headline contrast.

## Proposed evidence sequence

### 1. Fixed-checkpoint activation smoke

Use an explicitly approved diagnostic checkpoint only to verify that the operator changes candidate posterior quantities on `m>=2` parents without destabilizing the exact `m<=1` control. This is diagnostic evidence and cannot be reported as a formal FPCT performance result.

Required controls:

- `m<=1` exact `F=C_post` control;
- replicated-collapse inference control;
- candidate permutation/refinement controls;
- no NaN/Inf, mask violations or duplicate use of `A`;
- measured expanded-slot ratio versus the CPU estimate.

### 2. Matched training arms

Train exactly three matched arms with identical initialization, data order, seed, optimizer schedule, checkpoint selection rule and training budget:

- `C_pre`
- `C_post`
- `F`

`C_post-C_pre` identifies candidate-specific nonlinear fusion. `F-C_post` is the headline query-time factorization-preservation effect. `F-C_pre` is an overall system contrast only.

Old B2/B3/B6 checkpoints may be used only for approved initialization or diagnostics. They are not formal FPCT results, and the historical `+8.24pp` is not FPCT headroom.

### 3. C_post/F 2×2 inference intervention

For the matched `C_post`-trained and `F`-trained checkpoints, evaluate both inference operators:

- `Cpost-trained + Cpost inference = Y_CC`
- `Cpost-trained + F inference = Y_CF`
- `F-trained + Cpost inference = Y_FC`
- `F-trained + F inference = Y_FF`

The predeclared interaction is:

`(Y_FF - Y_FC) - (Y_CF - Y_CC)`.

This separates training adaptation from inference-time factorization use. It does not replace the matched-arm `F-C_post` headline comparison.

## Required mechanism diagnostics

Report these on newly generated FPCT checkpoints and re-audit beneficial/harmful events rather than reusing old event labels:

- `gamma(j|i)`, including entropy and departure from the frozen prior;
- candidate-logit variance and range;
- `KL(gamma || A)`;
- Jensen gap;
- fused candidate distances `D_K` and `D_V`;
- exact `m<=1` control and `m=2/3/4` strata;
- replicated-collapse inference control;
- latency, peak HBM and expanded-slot ratio;
- task/pair/sample aggregation without selecting a favorable view.

These distinguish mathematical correctness, real-data mechanism activation and task-accuracy improvement. Candidate count alone proves only structural opportunity.

## Resource expectation for the selected pair

The FPCT-1B distribution implies ambiguous-only mean expansion ratios of 1.2376 on ARC, 1.2392 on OpenBookQA and 1.2271 on MMLU-Redux; p95 is 1.2975, 1.2821 and 1.3015. Dense top-k4 would be about 3.69×–3.76× on average. The GPU pilot must measure rather than assume these ratios and must report latency/HBM alongside task outcomes.

## Human approval required before operation

The following remain deliberately unset and must be locked prospectively before any GPU job is authored or launched:

1. exact diagnostic checkpoint provenance and permitted initialization rule;
2. seed count and seed identities;
3. matched training budget, optimizer schedule and checkpoint-selection rule;
4. formal primary effect threshold and multiplicity/decision rule;
5. paired effect/power analysis using observed discordance and matched-retraining seed variance;
6. fp16/bfloat16 numerical tolerances and GPU reference-kernel checks;
7. exact evaluation split release, sample exclusions and event-audit schema;
8. latency/HBM measurement protocol and resource ceiling;
9. stopping rules for smoke, training and evaluation failures.

No parameter in this list may be chosen after viewing the corresponding natural GPU result.

## Prohibited interpretation

This draft is not a Kubernetes job, training recipe or execution authorization. FPCT-4 remains `REVIEW REQUIRED / GPU NOT AUTHORIZED`; FPCT-5 and later stages remain `NOT AUTHORIZED`.
