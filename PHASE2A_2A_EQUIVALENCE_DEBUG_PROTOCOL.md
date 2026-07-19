# Phase 2A-2a Llama3.2 Gate-1 Equivalence Debug Protocol

## Scope

This branch diagnoses only the Llama3.2 seed-42 Gate-1 failure observed in Phase 2A-2a. It does not evaluate cache-geometry predictability, train a selector, run MMLU, read the Phase 2A-1 sealed test, or modify any checkpoint.

The frozen sample scope is the Phase 2A fit split for AI2-ARC (351 rows) and OpenBookQA (158 rows), for 509 rows total. Comparison fields are sample ID, `pred`, `cot_pred`, complete `cot_output`, and `cot_gen_length`; correctness and labels are neither read nor used by the diagnostic comparator.

## Controlled execution

One Kubernetes Pod requests exactly one GPU on `4090-24gx4`. All evaluator processes run serially inside that Pod, inherit the same CUDA-visible physical GPU, use the same immutable runtime identity, B6 checkpoint, model paths, task configs, generation parameters, and fit-only filter.

The fixed order is:

1. OFF-A: ARC, then OpenBookQA.
2. OFF-B: ARC, then OpenBookQA.
3. If OFF-A and OFF-B differ, stop immediately.
4. ON-A: ARC, then OpenBookQA.
5. ON-B: ARC, then OpenBookQA.
6. If ON is unstable or differs from OFF, run NOOP-A and NOOP-B, each ARC then OpenBookQA.

OFF omits cache-geometry instrumentation. ON performs the original detached geometry reductions and scalar capture. NOOP enters the enabled instrumentation control flow, but `capture_projector_cache_geometry` returns before batch inspection, CUDA reductions, CPU scalar synchronization, or layer-record streaming.

## Frozen classification

- OFF-A != OFF-B: baseline/runtime numerical nondeterminism; immediate stop.
- OFF stable, NOOP unstable or different from OFF: instrumentation control-flow or runtime perturbation.
- OFF and NOOP stable/exact, ON unstable or different: geometry-reduction/synchronization observer effect.
- OFF stable/current ON exact but frozen reference different: historical environment/reference drift.
- Only an exact OFF/NOOP/ON result could justify proposing a revised non-perturbing observer; no selector experiment may start automatically.

Every run records the exact config and checkpoint SHA, command, wall time, prediction artifact SHA, visible GPU UUID before and after, container/runtime identity, PyTorch/CUDA/cuDNN versions, driver/GPU provenance, and determinism checks. Full per-example generations remain under `/netdisk/lijunsi/c2c-phase2a2-equivalence-debug/results` and are not committed.
