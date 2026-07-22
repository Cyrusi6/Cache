# FPCT GPU R2l Focused Diagnostic Report

## Decision

The sealed focused diagnostic is `DIAGNOSTIC_QUALIFIED` and remains
`DIAGNOSTIC_ONLY`. It qualifies the semantic-map repair for a distinct,
one-shot immutable R2l compatibility gate. It does not produce R2l GO,
authorize training, rescue R2k, or support any task-performance claim.

R2k remains permanently terminal as `GPU_ENGINEERING_BLOCKED_R2K` at scientific
SHA `458b0260fc5475c9ae578eb68b8dff2b2699e2f4`, image digest
`sha256:bc19b894c18eea266596011b748a4a22c73b80788e0fdd4a08b5f33059bf51ca`,
run UID `fpct-r2k-458b026-v1`, and run-lock SHA256
`d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`.
Its exact-null and resource results are neither retried nor reinterpreted.

## Frozen focused execution

- Scientific repair commit: `d71d21b1e315787e9af1cefb324abd310fd335f7`
- Execution-lock commit: `87277f5757ea2b0c0d7f2419d62595cc27860fc4`
- Diagnostic image: `docker.io/library/fpct-gpu-r2l-diag:d71d21b@sha256:ef04b033a2038125166e10bd71699bf7222ce7e1b8217cb4e965b752eff194a7`
- Embedded source-tree SHA256: `9fabb08e0eda02461a85fdfb276860f9d4c3ad57f642d2a49ad3c265fa5eed97`
- Run UID: `fpct-r2l-diag-d71d21b-v1`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2l-diag-d71d21b-v1`
- Run-lock SHA256: `93513df3113a074d6e0542bf918864c5c4b274eee6f3872e769a1532aa67c31a`
- Focused aggregate SHA256: `275e24f031ae9f627b07eb91babe8847e192c9fa9891d102d8672d73a89d24ce`
- Aggregate sealed-attestation SHA256: `f56bd691cd868aa7e968ff4dbaac4751669f0ae22f704415efc3b25809a24c09`

The image-loader and focused diagnostic Jobs both completed `1/1` on
`4090-48gx2` with zero Pod restarts. All fourteen execution attestations have
`target_exit_code=0`.

## Exact-null and semantic-map evidence

The synthetic mixed-memory cases cover FP32/BF16 and GQA/MQA. The full parent
map passes `E = NOT U OR (U AND E_sc)`, and missing sidecar metadata remains
fail-closed only inside the covered span. In mixed batches, the exact sample
selects the parent result while the active sample keeps the flat atom route.

On the fixed label-free pretrained row, both FP32 and BF16 satisfy all of the
following direct comparisons:

- `F_native == C_post_native`;
- `F_replicated_native == F_native`;
- final logits have identical tensor-byte SHA256;
- `max_abs=0` and `ULP=0`;
- all 84 per-dtype layer endpoint comparisons are equal;
- no first-divergence layer exists.

FP32 final-logit SHA256 is
`9b5ff4b032942f3b26882c6a11239c213972ab6fe627bc2aa977c108dda71325`;
BF16 final-logit SHA256 is
`852272cf1ae970b80d11ef2d7b731a515095e61f18b4440c805777b9911f49c5`.
The actual Qwen3 eager 28-layer CPU/HF integration regression separately
covers padding, causal prefill, four decode steps, cache equality, GQA/MQA,
FP32/BF16, and final-logit byte equality.

## Active-route preservation

Forced-on remains active in both dtypes and C_post/F pre-collapse candidates
remain identical. The focused activation values are:

| Dtype | Delta_fact max abs | candidate-logit range | D_K | D_V |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.0550814 | 0.258056 | 2.42133 | 1.70623 |
| BF16 | 0.46875 | 0.245254 | 4.37443 | 1.62952 |

These values exceed the frozen metric-specific null floors. They demonstrate
that the repair did not turn the active F route into a bypass. They do not
establish task usefulness.

## Focused resource and profiler evidence

The single prospective block used 20 warmups and 50 measurements per arm for
each canary. It is a focused pre-freeze diagnostic, not the immutable resource
decision.

| Canary | CUDA median F/C_post | CUDA p95 F/C_post | Result |
| --- | ---: | ---: | --- |
| checkpoint-native | 1.06931 | 1.03412 | pass |
| forced-on | 1.04409 | 1.08302 | pass |

Peak HBM is `4.76606 GiB`; mean and p95 expansion are `1.22689`. The separate
C_post and F traces contain no host synchronization event inside the frozen
FPCT hot scopes. Trace SHA256 values are
`01d749f41135cef2c393816141041d965dadd41e60fdbcf64617e38980744c4f`
and `1eafadb16e0c02b15808b39331e02316cd87d91fb43eb0efc7cb6b355045f241`.

## Verification and next boundary

The implementation diff remains confined to
`bind_fpct_layout_layer_semantics`; the flat atom kernel, parent eager/logit
reuse, wrapper, projector, aligner, prior, masks, thresholds, data, seeds,
training, and statistics remain frozen. Targeted FPCT tests are `207 passed`,
and the CPU-safe full suite is `446 passed, 2 warnings`.

No accuracy/correctness, model-selection, held-out result, training loss,
optimizer step, or checkpoint was read or produced. The only permitted next
scientific action is a new immutable image/run UID/run-lock executing the
original 23 checks plus the six R2l semantic checks and the retained balanced
checkpoint-native/forced-on canary. Any immutable failure is terminal
`GPU_ENGINEERING_BLOCKED_R2L` and cannot be repaired under the same revision.
