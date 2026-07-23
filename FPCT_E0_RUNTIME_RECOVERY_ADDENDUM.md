# FPCT-E0 runtime-write recovery addendum

## Failure classification

Seed 2026072201 attempt 2 completed all 64 C_post optimizer steps and wrote its step-64 checkpoint, but the sealed bootstrap rejected the process during post-target attestation with `immutable image source tree hash mismatch`. Rank 0 was then terminated before `fpct_formal_integrity.json` was sealed, so attempt 2 is incomplete and cannot supply an official C_post arm or any evaluation checkpoint.

The failure occurred before F training or any E0 accuracy evaluation. It is an orchestration/runtime-write failure, not a numerical, model, operator, mask, prior, OOM, or task-performance result. Offline W&B was allowed to use the process working directory `/opt/fpct`, which polluted the immutable image source tree before post-attestation.

## Authorized recovery

The user explicitly requested that a failed run be made to continue to completion. Recovery changes only writable runtime locations:

- `WANDB_DIR`, cache/config/data directories, Hugging Face caches, XDG cache, Torch extensions and temporary files are redirected below `/fpct-e0`;
- `PYTHONDONTWRITEBYTECODE=1` prevents Python bytecode writes into `/opt/fpct`;
- an exact-image CPU preflight must demonstrate that representative offline W&B initialization and imports leave the sealed source-tree SHA unchanged;
- attempt 2 remains preserved and quarantined from formal evidence;
- seed 2026072201 restarts from step 0 as attempt 3, followed in the same two-GPU Pod by seed 2026072202 attempt 2 and seed 2026072203 attempt 2.

No checkpoint is reused. The exact image, immutable scientific ConfigMap, seeds, arm order, data order, initialization, 2,048 examples, 64 steps, optimizer, scheduler, precision, evaluation cells, thresholds and claim boundary remain unchanged. The three-seed continuation is not conditional on any training loss or accuracy result.
