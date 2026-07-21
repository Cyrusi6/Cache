# FPCT-GPU-R2j Training-Integrity Recovery Addendum

## Prospective boundary

This addendum is frozen after terminal R2i-v2 and before any R2j GPU,
pretrained, training, checkpoint, accuracy or correctness output. R2i-v1 and
R2i-v2 remain immutable and may not be resumed. R2j requires a new scientific
SHA, image, run-lock, run UID and root.

The FPCT operator, parent-equivalence metadata, thresholds, panels, models,
data, seeds, training recipe and statistical rules do not change.

## Frozen evidence

R2i-v2 repeated the complete GPU and pretrained gates and again passed all
23 checks. Its matched smoke failed before any optimizer-step or checkpoint
artifact. W&B output was correctly isolated under `/fpct-run`; two integrity
tool defects remained:

1. `_fpct_tensor_state_sha` called `.view(torch.uint8)` directly on a 0-D BF16
   trainable parameter and raised `self.dim() cannot be 0` during step-0 state
   hashing on rank 1.
2. DDP initialization can append a second temporary Torch
   `_remote_module_non_scriptable.py` directory with byte-identical source.
   The bootstrap validates both directories but previously retained duplicate
   identical markers in the stable `sys.path`, creating a rank-0 post-target
   fingerprint mismatch.

## Frozen repairs

The trainer hashes exact parameter bytes only after
`contiguous().cpu().reshape(-1).view(torch.uint8)`. Dtype, shape and name remain
separate hash inputs; no numerical tensor conversion is introduced.

The bootstrap continues to hard-fail any unexpected temporary directory,
foreign Rosetta candidate, unexpected cache entry or distinct generated source.
After validating the exact Torch remote-module structure and source hash, it
deduplicates only byte-identical stable markers. Raw `sys.path` remains in the
full attestation.

W&B output-path isolation from R2i-v2 remains operative for matched and formal
Jobs.

## Required validation and execution

- scalar BF16, vector and changed-value state-hash regressions;
- duplicate identical Torch remote-module marker deduplication while existing
  sealed-import negative controls continue to pass;
- targeted confirmatory/sealed-import tests and CPU-safe full suite;
- new immutable image/run-lock/root;
- complete synthetic GPU gate and complete pretrained matrix from scratch;
- only after both GO, a new three-arm four-step matched smoke.

Any failure is terminal for that run. Tolerances, recipes and scientific
thresholds may not be changed.

## Claim boundary

These repairs establish deterministic integrity measurement and sealing only.
They are not operator changes and provide no performance evidence. Formal FPCT
claims remain conditional on a complete matched smoke and the preregistered
12-seed checkpoint-native evaluation.
