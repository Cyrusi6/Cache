# FPCT-E0 decode-mask recovery addendum

## Failure and evidence

Seed `2026072201` attempt 4 completed both C_post and F arms at 64 optimizer steps. Both formal integrity records passed checkpoint reload, and `matched_integrity.json` is `GO`. The training artifacts are therefore retained read-only.

All first-pass evaluation cells were invalid: the evaluator skipped samples with `FPCT sidecar segment exceeds cache source length`, and the mechanism probe terminated on the same exception. Empty 0% summaries are quarantined and must not enter the E0 effect report.

A one-prompt, no-correctness diagnostic captured:

- prefill: `source_length=129`, sidecar parent range `[3,120)`;
- first decode: `source_length=1`, with the same sidecar range and `source_length_hint=129`.

The root cause is in `RosettaModel.forward`. It slices each section's attention mask as `attention_mask[:, :end]`. During cached decode, `end` is the number of current tokens, not the total source length, so the full prompt mask is incorrectly reduced to one token. The correct slice boundary is `initial_past_length + end`.

## Prospective repair and evidence boundary

The production repair changes only that mask boundary and adds an actual Qwen3/DynamicCache regression through the public wrapper `forward` path. It does not change alignment, candidate geometry, C_post/F operators, priors, trainable parameters, checkpoints, training, scoring, or thresholds.

Because the production source changes, evaluation recovery must use a newly built immutable image whose Git SHA and source-tree SHA are recorded before output. The prior image remains the training image for seed 2026072201 attempt 4; its checkpoints are loaded read-only by the corrected evaluation image.

Evaluation recovery creates a clean attempt directory so the invalid empty CSV/summary files are never mixed with recovered artifacts. It may symlink the completed C_post/F checkpoint directories and matched-integrity record from attempt 4, but it may not modify or retrain them. Seeds 2026072202 and 2026072203 run from step 0 using the corrected image; the decode repair is inactive during ordinary no-past training forwards.

The corrected image is `docker.io/library/fpct-e0-decode@sha256:19c7a81568b4701fa60f11c682423d9eb812fe6e8f95bcfc36aa21eb98e82683`, built from commit `6a51ad4ed1d66067c0ac2d3f2c8c3b5de0f5d2ba`. Its embedded source-tree SHA is `1534f7fe2010ebc51b160b3bcb74e58d9eb44b3fd6cb8aad7674356dc91f4b7c`. Before recovery, a one-prompt smoke must show prefill source length 129, decode source length 130, no sidecar-bound violation, and no correctness access.
