# FPCT-E0 Exploratory Results

## Outcome

The frozen result is `E0_NO_GO_FOR_FURTHER_SPEND`. All three seeds completed with matched integrity `GO`, zero evaluation skips, and nonzero candidate-factorization activation. The engineering execution succeeded; the exploratory performance gate did not.

| Seed | T (pp) | O (pp) | Integrity | Mechanism activation |
|---|---:|---:|---|---|
| 2026072201 | +1.2128 | -0.1302 | GO | nonzero |
| 2026072202 | -3.3780 | -1.0826 | GO | nonzero |
| 2026072203 | +1.3467 | -0.2158 | GO | nonzero |
| Mean | -0.2728 | -0.4762 | all GO | all nonzero |

Mean task-level matched system effects were ARC `-1.5625 pp`, MMLU-Redux `+3.1250 pp`, and OpenBookQA `-2.3810 pp`.

The frozen GO rule failed because mean `T` was below `+1.00 pp` and OpenBookQA crossed the `-2.00 pp` task floor. Although two of three `T` values were positive, all three direct query-time effects `O` were negative. Therefore the two positive system effects cannot be attributed to query-time factorization preservation; they are consistent with training-trajectory differences.

This exploratory result does not prove that FPCT is generally ineffective. It says only that TinyLlama to Qwen3 with 2,048 training examples, 64 optimizer steps, these three development tasks, and this operator implementation does not justify the planned 36-run confirmatory campaign.

## Versioned artifacts

The exact final aggregate, per-seed effects, mechanism diagnostics, matched-integrity records, compact CSV tables, and SHA256 manifest are in `recipe/eval_recipe/fpct_e0/versioned_outputs/`.

The full 55GB run remains at `/netdisk/lijunsi/fpct-e0/fpct-e0-20260722-v1`. Checkpoints, per-sample predictions/CoT, W&B logs, and runtime caches are intentionally not committed to Git.
