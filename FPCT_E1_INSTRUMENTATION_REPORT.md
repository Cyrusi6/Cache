# FPCT-E1 Instrumentation Hard-Gate Report

## Decision

`E1-1 = GO`.

The new explicit capture path passes the synthetic formula gate and is observationally inert on an actual `Qwen3ForCausalLM` random small configuration with eager attention and `DynamicCache`. No pretrained weights, natural prompt, E1-pilot row, confirmatory outcome, GPU, or training operation was used for this gate.

## E0 measurement diagnosis

The E0 mechanism probe cannot support a claim that the query posterior was invariant. E0 generated only two tokens per task sample, read metrics after generation, and the wrapper cleared `_fpct_mechanism_metrics` at every `forward()`. The surviving decode forward normally had `q_length=1`, while the legacy metric computed `gamma.var(dim=query)`, making the recorded zero a measurement-definition artifact.

E0 accuracy and its frozen `E0_NO_GO_FOR_FURTHER_SPEND` classification remain unchanged. E1 repairs only the interpretation and future collection of mechanism measurements.

## Implemented capture contract

- `begin_fpct_capture(mode, metadata, query_mask)` and `end_fpct_capture()` define an explicit lifecycle.
- Capture modes include `prefill`, `teacher_forced_response`, and `greedy_decode`.
- Teacher-forced eligibility uses the causal shift: query position `t` is eligible exactly when `labels[:, t+1] != -100`; prompt and final non-predicting positions are excluded.
- Welford state accumulates across forwards/decode steps and is not replaced by the last forward.
- Candidate visibility is query-specific and inherits the actual attention mask. Invalid/invisible atoms do not enter posterior moments.
- Reports contain global, layer, head, query, parent, candidate-count, per-candidate gamma moments, KL, TV, logit-range/variance, Jensen-gap, and top-1-change summaries.
- The capture path stores no raw K/V tensor (`stores_raw_kv=false`).
- Legacy behavior without an explicit capture remains available for historical diagnostics; E1 conclusions must use the explicit lifecycle.

## Numerical evidence

The frozen parity artifact is `recipe/eval_recipe/fpct_e1/e1_instrumentation_parity.json` (SHA256 `00fb8403f1376eb0a3d4ccf498a6c7230a313aec7259f661553ca0e164a2d1fc`).

- teacher-forced logits: bitwise equal;
- teacher-forced loss: bitwise equal;
- teacher-forced `DynamicCache`: bitwise equal;
- three greedy decode steps: logits and cache bitwise equal at every step;
- generated token sequence: identical (`[4,13,58]`);
- answer-query mask: two eligible positions, prompt positions excluded;
- parent summaries present and JSON serializable.

The frozen synthetic artifact is `recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance.json` (SHA256 `f652ee99f8bce508b3f35953b10ad11ad4046ab0f868bf93fc08a82bb6ef14b2`).

- query-changing posterior variance: `0.19730721414089203`;
- posterior top-1 change: true, rate `1.0`;
- two forwards and two eligible queries were accumulated;
- identical-candidate KL, TV, and Jensen gap: exactly `0`;
- raw K/V retained: false.

## Test evidence

The independently rerun targeted gate completed `82 passed` across:

- `test/test_fpct_instrumentation.py`;
- `test/test_fpct_e1_mechanism_audit.py`;
- `test/test_fpct_qwen_cpu_integration.py`;
- `test/test_fpct_reference_operator.py`;
- `test/test_fpct_production_path.py`.

The centered-lambda oracle includes native atoms and transported children in one global denominator. It validates `lambda=0 == C_post`, `lambda=1 == F`, permutation/refinement invariance, causal label shift, and invalid probability/gradient exact zero. Topology now includes `taxonomy_unresolved`; an unexplained non-partition is not silently called a competing overlap.

An all-FPCT historical test invocation reaches the old R2l repository-diff verifier, which intentionally rejects every file added after its frozen historical allowlist (including H1, E0, and E1 artifacts). This is not a functional regression and its historical allowlist was not rewritten. The current E1 targeted production/reference/Qwen gates all pass.

## Remaining boundary before E1-2

E1-1 validates measurement lifecycle and posterior statistics. It does not yet establish real-data activation or task benefit. Before the natural E0-design audit, the execution adapter must additionally bind the parent summaries to source/fused dispersion, parent attention mass, topology, and teacher-forced gold log-prob rows, then be committed and hash-locked. Missing capabilities must fail closed.

E1-pilot remains `SEALED / NOT RUN / NOT READ`; confirmatory data remain sealed.
