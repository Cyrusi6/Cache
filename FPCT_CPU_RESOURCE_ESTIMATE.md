# FPCT CPU Resource Estimate

## Scope

This is a static CPU estimate based on the frozen FPCT-1B parent-support rows and the locally cached Qwen3-0.6B receiver `config.json`. It loads no model weights and executes no model forward. The local detailed JSON is not committed.

Definitions:

- `extra_slots = sum_i max(m_i-1,0)`
- `expanded_slots = native_slots + extra_slots`
- ambiguous-only sidecar atoms `= sum_i m_i * 1[m_i>=2]`
- dense top-k4 slots `= native_slots + 3 * eligible_parent_count`
- attention-score FLOP ratio is approximated by expanded/native memory-slot ratio for fixed query/head dimension.

The receiver has 28 layers, 16 query heads, 8 KV heads and head dimension 128. At bfloat16, one K/V atom costs 4,096 bytes per layer.

## Pair/task estimate

`Ambig sidecar/layer` is the mean complete K/V storage for ambiguous children. `Dense sidecar/layer` is the corresponding allocation if every eligible parent were materialized at top-k4. `Cache increment/all layers` is the mean incremental K/V storage relative to one native slot per parent. `Dense ratio` is the mean dense-top-k4 attention-score ratio.

| Pair | Task | Expansion mean | p50 | p90 | p95 | max | Ambig sidecar/layer | Dense sidecar/layer | Cache increment/all layers | Dense ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tinyllama | ai2-arc | 1.2376 | 1.2381 | 1.2846 | 1.2975 | 1.4024 | 236.42 KiB | 2.2078 MiB | 3.63 MiB | 3.7387 |
| tinyllama | openbookqa | 1.2392 | 1.2391 | 1.2727 | 1.2821 | 1.3143 | 200.78 KiB | 1.8426 MiB | 3.07 MiB | 3.6921 |
| tinyllama | mmlu-redux | 1.2271 | 1.2273 | 1.2842 | 1.3015 | 1.4184 | 265.32 KiB | 2.6496 MiB | 4.04 MiB | 3.7621 |
| qwen25_0p5b | ai2-arc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 2.2078 MiB | 0.00 MiB | 3.7387 |
| qwen25_0p5b | openbookqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 1.8426 MiB | 0.00 MiB | 3.6921 |
| qwen25_0p5b | mmlu-redux | 1.0008 | 1.0000 | 1.0000 | 1.0000 | 1.1341 | 1.14 KiB | 2.6496 MiB | 0.02 MiB | 3.7621 |
| llama32_1b | ai2-arc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 2.2078 MiB | 0.00 MiB | 3.7387 |
| llama32_1b | openbookqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 1.8426 MiB | 0.00 MiB | 3.6921 |
| llama32_1b | mmlu-redux | 1.0006 | 1.0000 | 1.0000 | 1.0000 | 1.1333 | 0.89 KiB | 2.6496 MiB | 0.01 MiB | 3.7621 |
| qwen3_1p7b control | ai2-arc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 2.2078 MiB | 0.00 MiB | 3.7387 |
| qwen3_1p7b control | openbookqa | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 KiB | 1.8426 MiB | 0.00 MiB | 3.6921 |
| qwen3_1p7b control | mmlu-redux | 1.0008 | 1.0000 | 1.0000 | 1.0000 | 1.1341 | 1.14 KiB | 2.6496 MiB | 0.02 MiB | 3.7621 |

## Interpretation

For the selected TinyLlama pilot, ambiguous-only packing raises mean attention-score work by about 22.7%–23.9%, with p95 expansion about 28.2%–30.1%. Dense top-k4 expansion would instead be about 3.69×–3.76× on average. The ambiguous-only design therefore removes most of the structurally unnecessary dense expansion.

The other heterogeneous pairs have no expansion on ARC/OpenBookQA and only sparse MMLU expansion, matching the FPCT-1B readiness result. These are resource/structural facts only; they do not establish query-time logit separability or accuracy improvement.

## Provenance

- FPCT-1B execution SHA: `7f8af71968a39bc6cba2e4e34de762b291cda834`
- Frozen parent-support SHA256: `bfacda50295a63c186369a3168c60cf72939cdfc940e16d32d7644154daad76b`
- Receiver config SHA256: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`
- Local resource JSON: `local/final_results/fpct_factorized_transport/fpct_1b_ambiguity_support/rev_7f8af71968a39bc6cba2e4e34de762b291cda834/resource_estimate.json`
- Local resource JSON SHA256: `51532857ebc6f3984ad0670ef8adf13ce5cb2f8b36757fd4575985ac088c5edc`
