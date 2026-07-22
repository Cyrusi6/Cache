# FPCT GPU R2l Semantic-Map Implementation Report

## Authorized repair

The only modified existing scientific function is
`bind_fpct_layout_layer_semantics` in `rosetta/model/fpct_attention.py`.
Its layer map now starts as boolean true for the complete native source axis.
Each sidecar span is then set to false before discrete metadata is considered.
`parent_equivalent` has first priority, `parent_force_native` is the fallback,
and missing metadata leaves the covered span false.

This implements the frozen truth formula
`E = (not U) or (U and E_sc)` without changing the flat-atom kernel, parent
eager adapter, parent-logit reuse, packing mathematics, candidates, prior,
mask, projector, gate, certifier, timing, resource thresholds or training
recipe.

## Red-green evidence

Before the repair, the new mixed-memory regression suite failed all five initial
cases. For a length-nine source with an equivalent sidecar on `[2:5)`, the old
map was

`[false, false, true, true, true, false, false, false, false]`.

The failures also reproduced the poisoned native edges for one false sidecar
parent, missing metadata, discontinuous spans and mixed exact/active batches.

After the three-line semantic repair, all cases pass. Missing metadata fails
closed only within the sidecar span, while leading, trailing and intermediate
native gaps remain true.

## Exact-null and active tests

The targeted tests include:

- source-length-nine and discontinuous truth-table cases;
- the pre-bound layer semantic path;
- mixed exact/active samples in one batch;
- FP32/BF16 and GQA/MQA tensor paths;
- fixed-RNG training dropout and finite nonzero active-candidate gradients;
- actual Qwen3 eager with 28 layers, partial sidecar coverage, unequal padding,
  causal prefill and four decode steps;
- direct `torch.equal`, tensor-byte SHA, `max_abs=0` and `ULP=0` checks for
  per-layer pre-o-proj, post-o-proj, decoder hidden state, cache and final
  logits;
- an actual Qwen3 mixed batch in which the exact sample is bitwise equal and
  the active sample remains different;
- unchanged state-dict keys and existing replicated/bypass/m<=1 controls.

## CPU/HF validation

- New R2l focused tests: `14 passed` before the larger suites.
- All FPCT tests: `207 passed, 3 warnings`.
- CPU-safe full repository suite: `446 passed, 2 warnings` when run with the
  repository-required `local/tmp` basetemp.
- The initial full-suite attempt used `/tmp` and produced two unrelated Route1
  path-contract failures; rerunning without modifying code or assertions under
  `local/tmp` passed all 446 tests.
- Machine-verifiable protocol diff: `GO`; after masking the authorized function,
  the remainder of `fpct_attention.py` is byte-identical to baseline
  `2091c109...`, and every forbidden file SHA remains frozen.

No R2l GPU/pretrained output, training, checkpoint, accuracy/correctness,
model-selection or held-out evaluation had been run when this implementation
candidate report was written.
