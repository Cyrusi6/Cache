# FPCT-E0 operational amendment: serial completion without a hard 24-hour stop

## Human authorization

On 2026-07-23, before any FPCT-E0 accuracy result was read, the user explicitly authorized completing all three seeds serially on the same `4090-48gx2` node. The 24-hour target is changed from a hard stopping rule to an estimated completion time only.

This amendment changes execution scheduling only. It does not change the scientific image, model pair, operators, alignment, sanitizer, training examples, optimizer steps, seeds, arm order, initialization, data order, evaluation cells, aggregation, thresholds, or claim boundary frozen by `FPCT_E0_PROTOCOL.md` and `e0_manifest.json`. Completion of seeds 2026072202 and 2026072203 is not conditional on seed 2026072201 accuracy.

## Serial execution contract

1. The active seed 2026072201 attempt 2 remains untouched and runs to its normal terminal state.
2. A continuation Job for seeds 2026072202 and 2026072203 is created in suspended state, requests the same two 48GB GPUs, uses the same exact image and immutable ConfigMap, and is pinned to `4090-48gx2`.
3. A CPU-only Kubernetes release-controller Job reads only the seed-completion marker and Kubernetes phase. It does not read predictions, correctness, accuracy, training loss, or mechanism metrics. Its service account is restricted to `get` on the source Job and `get/patch` on the suspended continuation Job.
4. Only after seed 2026072201 reports `COMPLETE` does the controller unsuspend the continuation Job.
5. The continuation Pod runs seed 2026072202 attempt 2 and then seed 2026072203 attempt 2 as separate Python processes. If seed 2026072202 fails, seed 2026072203 is not started.
6. No 24-hour capacity kill is applied. Reported times are estimates, not scientific gates.

The failed attempt-1 Jobs and both immutable ConfigMaps remain preserved. The additional 24GB nodes are not used for formal training because their measured 24,564 MiB capacity is below the observed 26,533 MiB lower bound of the active training run.
