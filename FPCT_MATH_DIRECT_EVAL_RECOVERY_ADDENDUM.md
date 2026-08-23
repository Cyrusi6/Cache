# FPCT-MATH-DIRECT evaluation-only recovery addendum

## Scope

This prospective operational amendment was made after both matched training arms had
completed, but before the twelve-cell result reducer had completed. It changes no
model, operator, checkpoint, sample, label, threshold, aggregation, or scientific
decision rule.

The frozen MMLU-Redux development tree contains only the 49 subjects represented by
the preregistered 128 E0-design groups. The math-direct MMLU YAMLs accidentally
omitted the explicit 49-subject allowlist already present in the frozen E0 rendered
YAMLs. `unified_evaluator.py` consequently enumerated all dataset subjects and failed
on a subject that was intentionally absent from the materialized development tree.

## Recovery contract

- Reuse the completed C_post and F step-64 checkpoints; do not retrain.
- Keep the original 12 YAMLs and the pre-output manifest immutable.
- For MMLU only, derive a recovery YAML by inserting the exact subject list from
  `recipe/eval_recipe/fpct_e0/rendered/eval_2026072201_Y_CC_mmlu-redux.yaml`.
- Require the allowlist to contain exactly 49 subjects and equal the directory set in
  the frozen development tree.
- Reuse a cell only when it has exactly one completed `*_cot.csv`; never overwrite a
  completed cell.
- Run the original frozen evaluator and original frozen result reducer from Git tree
  `8a27c52415b20a90a1f167b2ec011bfaeb0bfe13`.
- Keep E1-pilot, model-selection, test, additional seeds, and confirmatory data sealed.

## Execution record

- r6 stopped before model loading because the training snapshot had acquired runtime
  `__pycache__` directories and therefore failed the exact snapshot-universe check.
- r7 used a newly materialized immutable snapshot, completed Y_CC/MMLU, then stopped
  before Y_CF model loading because the controller checked a not-yet-created output
  directory under shell `pipefail`.
- r8 used a second fresh immutable snapshot, created missing output directories before
  checking them, reused three completed Y_CC cells, completed the remaining nine
  cells, and ran the frozen reducer.

The r8 job was `fpct-math-direct-eval-recovery-r8` on `4090-48gx2`; it completed with
zero pod restarts. The mounted snapshot receipt SHA256 was
`71862dab24dd3049c4ac0b88848d4837898a1d4e86fcd15a2f21e150e229e75f`.

