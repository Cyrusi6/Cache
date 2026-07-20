# FPCT GPU R2 Profiling Protocol

Status: `PROSPECTIVE`

## Required scopes

- `fpct.layout_prepare`
- `fpct.project_candidates`
- `fpct.pack`
- `fpct.attention`
- `fpct.diagnostics`
- `receiver_attention`
- `scientific_forward`
- `harness_sync`
- `profiler_lifecycle`

## Frozen profiles

| ID | Workload |
|---|---|
| `P0_GPU_NEGATIVE` | pure GPU operations, no explicit scalar extraction |
| `P1_ITEM_POSITIVE` | explicit `x.sum().item()`; parser must detect sync |
| `P2_CPOST_OFF` | C_post, instrumentation off |
| `P3_F_OFF` | F real candidates, instrumentation off |
| `P4_F_REPLICATED` | F replicated atoms |
| `P5_F_ON` | F with diagnostics capture |
| `P6_DECODE4` | prefill plus at least four decode steps |

Store raw events, Chrome trace, counts, timestamps, CPU ancestry/stack, D2H
direction, enclosing scope and artifact SHA. A parser that misses P1 is invalid.

## Host-sync contract

- Layout width and row maps come from CPU frozen alignment metadata or an
  explicit `max_slots_hint`, followed by one H2D transfer. GPU
  `expanded_slots.max().cpu()` is forbidden.
- GPU forward never branches on `torch.any(cuda_tensor)`; sidecar legality is
  CPU-certified and hash-bound.
- CUDA `kv_cache_index.item()` is forbidden; scalar metadata remains on CPU.
- Diagnostics-off performs no CPU materialization. Diagnostics-on retains only
  detached device tensors during forward and materializes once after
  `scientific_forward` in an explicit diagnostics scope.
- Measurement synchronization is allowed only in `harness_sync`. Profiler
  teardown synchronization is allowed only outside scientific ranges and only
  when the same event appears in P0.

Hard gate inside `fpct.pack` and `fpct.attention`: zero `aten::item`,
`_local_scalar_dense`, D2H copy, `cudaStreamSynchronize`,
`cudaDeviceSynchronize` and `cudaEventSynchronize`. `fpct.project_candidates`
must have no per-candidate D2H. P3/P4 may not introduce scientific-scope sync
relative to P2.

## Resource gates

The existing prospective ceilings remain unchanged: certified mean expansion
`<=1.35`, p95 `<=1.50`, peak HBM `<90%`, F/C_post median latency ratio
`<=1.50` and p95 ratio `<=1.75`.
