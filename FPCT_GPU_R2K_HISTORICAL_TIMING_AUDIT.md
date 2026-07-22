# FPCT GPU R2k Historical Timing Audit

Status: `DESCRIPTIVE_ONLY`

This report reads only the frozen P2/P3 profile manifests and Chrome traces
from R2h, R2i-v1, R2i-v2 and R2j. It does not access task labels,
correctness, model-selection or held-out outputs, and it does not alter R2j's
permanent `GPU_ENGINEERING_BLOCKED_R2` classification.

## Wall-time summary

| Run | C_post median (s) | F median (s) | F/C_post median | max ratio | C_post MAD/CV | F MAD/CV |
|---|---:|---:|---:|---:|---:|---:|
| R2h | 0.655635 | 0.707643 | 1.079325 | 1.061494 | 0.026687 / 0.03285 | 0.012403 / 0.01583 |
| R2i-v1 | 0.639624 | 0.738131 | 1.154008 | 1.136974 | 0.007588 / 0.01697 | 0.010777 / 0.01626 |
| R2i-v2 | 0.686937 | 0.742478 | 1.080853 | 1.071523 | 0.005402 / 0.03829 | 0.010701 / 0.02439 |
| R2j | 0.421690 | 0.712045 | 1.688551 | 1.345503 | 0.007083 / 0.10743 | 0.028104 / 0.05490 |

R2j's ratio jump is driven primarily by a much lower C_post median, not by a
corresponding F slowdown relative to the earlier runs. Its first C_post sample
was 1.3234 times the seven-sample median and the fitted iteration slope was
`-0.01322 s/iteration`, while F's slope was `+0.00543 s/iteration`. This is
evidence that one warmup plus a fixed C_post-then-F order is fragile; it is not
evidence that R2j should be retried or reclassified.

## Profiler and kernel summary

| Run | C_post profiled scope (s) | F profiled scope (s) | scope ratio | C_post summed kernels (s) | F summed kernels (s) | kernel ratio |
|---|---:|---:|---:|---:|---:|---:|
| R2h | 1.420443 | 1.632412 | 1.149227 | 0.066327 | 0.072542 | 1.093697 |
| R2i-v1 | 1.453776 | 1.654801 | 1.138278 | 0.066669 | 0.073109 | 1.096593 |
| R2i-v2 | 1.494304 | 1.666366 | 1.115146 | 0.066824 | 0.073101 | 1.093942 |
| R2j | 0.944368 | 1.637428 | 1.733888 | 0.066551 | 0.072898 | 1.095371 |

Summed device-kernel ratios are stable near 1.094--1.097 across all four
runs. In contrast, the profiled host scope reproduces R2j's large ratio. The
dominant common kernels are receiver GEMMs and common elementwise/reduction
kernels; the additional F work appears mainly as extra launches and FP32
packing/grouping operations rather than a large change in the main GEMMs.
This supports prospective removal of the second parent QK calculation,
grouped scatter/reduction and five-dimensional `group_value`, but remains a
descriptive engineering diagnosis.

## Provenance and order

The profile loop order was fixed as P2 C_post, P3 F, P4 replicated, P5
instrumented F and P6 decode4. P2/P3 each used one warmup and seven measured
forwards. The three scientific hot-path Git blobs are identical between R2i
commit `8d21c72` and R2j commit `efa02fb...`:

- `rosetta/model/fpct_attention.py`: `be395083...`;
- `rosetta/model/wrapper.py`: `5890df52...`;
- `script/experiment/fpct_gpu_r2_runner.py`: `41dc22dd...`.

All traces identify an NVIDIA GeForce RTX 4090 with 48 GB-class total memory,
128 SMs and compute capability 8.9. They do not record GPU UUID, SM/memory
clock, temperature, power, P-state, throttle reasons, foreign GPU processes,
node pod inventory, CPU affinity or scheduler state. Therefore no claim about
thermal throttling, interference or node scheduling is identifiable from the
historical artifacts. R2k records those fields prospectively.

The full read-only aggregate, raw seven-sample arrays, trace hashes and top
kernel totals are stored in
`recipe/eval_recipe/fpct_gpu_r2k/historical_timing_audit.json`.

## Conclusion

The historical evidence motivates a balanced fresh-process benchmark and an
equivalent flat-atom kernel. It does not rescue R2j, does not establish an
infrastructure failure, and contains no accuracy or mechanism-performance
evidence.
