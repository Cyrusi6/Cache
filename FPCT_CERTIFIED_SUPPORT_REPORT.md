# FPCT-3.7-R1 Certified Support Report

Execution SHA: `7aecf2370df8a544b553baa6a7a58b24191e02ef`

Stable closure fingerprint: `5be64db77a54fb68ce44e2aabf7968c370bc4eff501fefe4a0fcb03ce571c472`

Status: `SINGLE_PAIR_PILOT_READY`

Selected pair: `tinyllama`

All 12 pair×task shards completed under the sealed import contract. Independent
verification recomputed 60 pair/task/split rows, 29,060 sample rows and 28,932
distinct-content-group rows. Qwen3 exact identity held across every split and
retained padding/truncation condition: every eligible parent was `m=1`, mapped
`i→i` with weight one, had `fallback=false`, and created zero extra slots.

## Fit+calibration readiness

| Pair | ARC groups | OpenBookQA groups | MMLU groups | Pooled | Ready |
|---|---:|---:|---:|---:|---|
| TinyLlama→Qwen3 | 511 | 228 | 2,495 | 3,234 | yes |
| Qwen2.5→Qwen3 | 0 | 0 | 0 | 0 | no |
| Llama3.2→Qwen3 | 0 | 0 | 1 | 1 | no |
| Qwen3 identity control | 0 | 0 | 0 | 0 | control only |

TinyLlama passes the frozen floor of at least 30 positive distinct groups in
each task and at least 100 pooled. It is the only ready heterogeneous pair, so
the authorized claim remains single-pair only.

Qwen2.5 and the Qwen3 raw diagnostic each contained the historical 56 MMLU
groups/410 raw m2 parents, all removed by the conservative certifier. Llama3.2
retained one fit+calibration certified group. These outcomes do not alter the
frozen TinyLlama selection rule.

## Factorization exposure and resources

Across all splits TinyLlama has certified-positive support in every content
group. Certified ambiguous-parent density is 0.2003 for ARC, 0.2088 for
OpenBookQA and 0.1855 for MMLU. The packed expansion distribution is:

| Task | Mean | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| ARC | 1.2374 | 1.2380 | 1.2973 | 1.3296 | 1.4024 |
| OpenBookQA | 1.2391 | 1.2389 | 1.2818 | 1.2981 | 1.3143 |
| MMLU | 1.2259 | 1.2261 | 1.3000 | 1.3333 | 1.4184 |

Thus the prospective resource support gate (`mean<=1.35`, `p95<=1.50`) is met
on all three tasks before any GPU execution. Detailed raw-pre-truncation →
retained → sanitized transitions, exposure, geometry, sanitizer removal mass,
sidecar/cache bytes and correlations remain in the local per-shard descriptive
artifacts.

Sanitizer checks established identical three-arm alignment-input hashes, exact
row normalization, exact uncertified slot-0 one-hot behavior, native `m0`,
unchanged `m1`, no illegal/truncated positive mass and no Qwen identity extra
slot.

## Provenance

- Freeze attestation SHA256:
  `cbe85babd12622b8ae03a5e166e22c96cca8f96d7851770537dd62be48901f67`.
- Pre-audit lock SHA256:
  `8127d66743f0656faeb6e3cfb754502489cc97e36f9dce751c90594e5efc121a`.
- Certified summary SHA256:
  `a0409299f639c4f41709e2845fb1ab65e8542f98cee5c014dfd36b21d552e863`.
- R1 result SHA256:
  `0f45c2d2c06c22ed6f4d0c6eda9ed75524e7010dba83f421209dd29f36d28cc8`.

This report establishes structural opportunity and resource support only. It
does not establish query-time separability, mechanism activation or accuracy.
