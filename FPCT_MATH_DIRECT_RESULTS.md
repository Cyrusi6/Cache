# FPCT-MATH-DIRECT single-seed result

## Outcome

The exact first-round `math.md` implementation completed matched C_post and F
training and all four inference cells on the already-open 326-group E0-design set.
Matched-integrity status is GO.

| Estimand | Value (pp) | Interpretation |
|---|---:|---|
| `T = Y_FF - Y_CC` | +1.428571 | Positive one-seed matched system/training result |
| `D_C = Y_CF - Y_CC` | -0.260417 | Switching a C_post-trained checkpoint to F is harmful |
| `D_F = Y_FF - Y_FC` | 0.000000 | Switching an F-trained checkpoint between C_post/F has zero task-macro effect |
| `O = (D_C + D_F)/2` | -0.130208 | Headline query-time operator effect is not positive |
| `I = D_F - D_C` | +0.260417 | Descriptive training-adaptation interaction |

The preregistered additional-seed condition required both `T>0` and `O>0`.
It is false. Classification: `SINGLE_SEED_NOT_POSITIVE`; no additional seeds are
authorized by this experiment.

## Four-cell task-macro accuracy

| Cell | Accuracy (%) |
|---|---:|
| `Y_CC` | 41.153274 |
| `Y_CF` | 40.892857 |
| `Y_FC` | 42.581845 |
| `Y_FF` | 42.581845 |

## Per-task accuracy

| Cell | ARC (%) | OpenBookQA (%) | MMLU-Redux (%) |
|---|---:|---:|---:|
| `Y_CC` | 43.750000 | 41.428571 | 38.281250 |
| `Y_CF` | 42.968750 | 41.428571 | 38.281250 |
| `Y_FC` | 44.531250 | 45.714286 | 37.500000 |
| `Y_FF` | 45.312500 | 45.714286 | 36.718750 |

The matched system delta `Y_FF-Y_CC` is +1.5625 pp on ARC, +4.2857 pp on
OpenBookQA, and -1.5625 pp on MMLU-Redux. This heterogeneity and the single seed make
the system-level positive result exploratory only.

## Integrity and artifacts

- Seed: `2026082101`.
- C_post/F training: 64/64 optimizer steps each, 2,048 identical examples.
- Equal across arms: step-0 trainable tensors, trainable keys, data order, pre-training
  RNG, optimizer structure, and scheduler initial state.
- C_post checkpoint SHA256:
  `85ab34828aec2d3014c4c125deafbf92e067ce686f9ba154a7da7af9933359de`.
- F checkpoint SHA256:
  `27f6793c3ef00896457fa5fc5859f8e3ced93053ca45db67fe0b0cffe0f03ca7`.
- Full local result:
  `/netdisk/lijunsi/fpct-math-direct-r5/math-direct-20260821-v1/fpct_math_direct_result.json`.
- Full result SHA256:
  `7a3902e7a386e553de8228491f4b22e409c7c4ae3e78083ed042d2d97a36e9dd`.
- Compact tracked result:
  `recipe/eval_recipe/fpct_math_direct/versioned_outputs/single_seed_2026082101_result.json`.

## Claim boundary

This experiment supports only the statement that, for one matched seed on the
TinyLlama-1.1B to Qwen3-0.6B pair and the open 326-group exploratory set, F training
produced a higher system score but query-time factorization itself did not produce a
positive effect. It is not a significance result, not confirmatory, not cross-pair,
and not evidence of a universal improvement.

