# FPCT math-direct three-seed replication results

## Terminal decision

The frozen mechanical classification is `STOP_CURRENT_OPERATOR`.

The original single-seed system gain did not replicate. Across seeds
`2026082101`, `2026082102`, and `2026082103`, mean matched system effect was
`T=-0.667163 pp`. Although two seeds had `T>0`, the frozen `+1.00 pp` mean floor
failed and the ARC task mean was `-2.083333 pp`, below the frozen `-2.00 pp`
task-harm boundary. All integrity controls were GO, so this is a scientific
negative result rather than an infrastructure failure.

The query-time mechanism gate also failed: mean `O=-0.050843 pp`, with only one
of three seeds positive. The complete-gold-response teacher-forced diagnostic
agreed on mechanism: mean `O_logp=-0.000007980`, effectively zero, with only one
of three seeds positive.

## Accuracy results

All values below are equal-weight task-macro percentages or percentage-point
differences on the already-open E0-design population.

| Seed | Y_CC | Y_CF | Y_FC | Y_FF | T | D_C | D_F | O |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026082101 | 41.153274 | 40.892857 | 42.581845 | 42.581845 | +1.428571 | -0.260417 | 0.000000 | -0.130208 |
| 2026082102 | 42.500000 | 42.239583 | 43.363095 | 43.102679 | +0.602679 | -0.260417 | -0.260417 | -0.260417 |
| 2026082103 | 44.620536 | 45.357143 | 40.848214 | 40.587798 | -4.032738 | +0.736607 | -0.260417 | +0.238095 |
| Mean | — | — | — | — | **-0.667163** | +0.071925 | -0.173611 | **-0.050843** |

Task-mean `T` was ARC `-2.083333 pp`, OpenBookQA `+1.904762 pp`, and
MMLU-Redux `-1.822917 pp`.

The frozen gates were evaluated without amendment:

- mean `T >= +1.00 pp`: FAIL;
- at least two of three `T>0`: PASS;
- every task mean `T >= -2.00 pp`: FAIL (ARC);
- all integrity controls GO: PASS;
- mean `O>0` and at least two of three `O>0`: FAIL.

## Continuous teacher-forced diagnostic

Each content group used the complete eight-token gold response. Values are mean
answer-token natural log probabilities, with the same four cells and equal-weight
task macro as accuracy.

| Seed | T_logp | D_C_logp | D_F_logp | O_logp |
|---:|---:|---:|---:|---:|
| 2026082101 | +0.008631279 | -0.000261391 | +0.000189246 | -0.000036072 |
| 2026082102 | +0.020499519 | +0.000204398 | -0.000481739 | -0.000138670 |
| 2026082103 | +0.026488111 | +0.000556448 | -0.000254843 | +0.000150802 |
| Mean | **+0.018539636** | +0.000166485 | -0.000182445 | **-0.000007980** |

`T_logp` was positive for all three seeds and all three task means. The related
collapsed-inference training contrast `Y_FC-Y_CC` had mean `+0.018722082` in
logp, but `-0.493552 pp` in accuracy. Therefore factorized training changed the
trajectory and improved average gold-response likelihood, but did not improve
the discriminative task outcome reliably. Because every response contains the
same eight-token answer format, this aggregate logp diagnostic cannot by itself
show that the correct-choice margin improved.

## Step-0 gradient diagnostic

The frozen rank-0 minibatch diagnostic performed no optimizer update.

- global C_post/F gradient cosine: `0.9992233935`;
- C_post gradient norm: `266.9571280`;
- F gradient norm: `253.4720762`;
- F/C_post norm ratio: `0.9494861`;
- missing-gradient parameter count: `0/0`;
- loss difference, F minus C_post: `+0.00222683`.

The gradients are almost parallel but not identical, with an approximately five
percent smaller global norm under F. Over 64 updates that small operator-induced
difference is sufficient to produce divergent projector trajectories.

## Mechanistic diagnosis

The observed negative and unstable accuracy is dominated by training trajectory,
not by a useful query-time factorization effect.

- At the F-trained checkpoints, direct F versus C_post accuracy effects were
  `0.000000`, `-0.260417`, and `-0.260417 pp`; none was positive.
- For seed `2026082103`, switching from C_post-trained/C_post inference to
  F-trained/C_post inference produced 12 more losses than gains across the three
  task populations. Enabling F inference added only one further net loss. Thus
  nearly all of its `T=-4.032738 pp` came from the learned trajectory.
- The continuous query-time effect was near zero (`O_logp=-7.98e-6`), while the
  training-cell logp contrast was consistently positive. This separates a real
  training adaptation from the absent headline query-time mechanism.
- The earlier fixed-checkpoint audit found extreme source-to-fused contraction and
  very small transported-parent attention mass. The present replication is
  consistent with that diagnosis: the candidate axis perturbs gradients but does
  not provide a stable inference-time signal.

This evidence does not support deploying F at query time. It also does not meet
the frozen condition for recommending F training with C_post deployment, because
that route's accuracy was itself unstable and negative on average.

## Research decision and claim boundary

The exact current `math.md` operator should stop. Additional seeds, longer training,
or confirmatory evaluation are not justified for this operator.

The broader FPCT question remains researchable, but the next experiment must be a
new prospectively frozen mechanism rather than a continuation of this arm. The
highest-priority route is receiver-native null/harm avoidance, because the main
failure is large seed/task-specific harmful transfer without measurable query-time
benefit. A second route is partition-aware span composition, which should treat
compositional subtokens as parts to combine rather than alternatives competing in
one softmax. Neither route is authorized or implemented by this result.

This is an exploratory single-pair result on TinyLlama-1.1B to Qwen3-0.6B, the
64-step budget, and the E0-design groups. It is not a cross-model, model-selection,
test, or confirmatory claim. E1-pilot, model-selection, test, and confirmatory data
remained sealed.

## Provenance

- Execution commit: `d234546d6596628a55e28d6586930019d8c8ce3f`.
- Full local result: `/netdisk/lijunsi/fpct-math-direct-r2/math-direct-replication-20260824-v1/fpct_math_direct_replication_result.json`.
- Full result SHA256: `1ee5dbb6e5012d8cb4839bf53949db3a226c3364c5b20c6e02c9aa363efc023c`.
- Step-0 diagnostic SHA256: `ddee436d885ef7afeed28e9ae6eec23162662c8dcd311bd2870d911288369651`.
- Training jobs: `fpct-math-r2-s2102-d234546` and
  `fpct-math-r2-s2103-d234546`; both Completed with zero restarts on
  `4090-48gx2`.
- Old-seed diagnostic/reducer job: `fpct-math-r2-s2101-teacher-d234546`;
  Completed with zero restarts on `4090-48gx2`.
