# FPCT-1B Structural Support Audit

## Decision

FPCT-1B completed without an integrity failure. The frozen engineering-readiness result is:

- global status: `SINGLE_PAIR_PILOT_READY`;
- ready/rank-1/selected heterogeneous pair: `tinyllama`;
- `qwen25_0p5b` and `llama32_1b`: support exists only in MMLU-Redux and does not pass the per-task or pooled engineering floor;
- `qwen3_1p7b`: same-tokenizer control, permanently excluded from readiness and ranking.

This is a label-free structural-opportunity result. It is not evidence for FPCT accuracy improvement, query-time separability, mathematical validity, or scientific power.

## Frozen execution identity

| Item | Value |
|---|---|
| Branch | `research/fpct-factorized-transport` |
| Execution commit (Commit A) | `7f8af71968a39bc6cba2e4e34de762b291cda834` |
| Operative v2 manifest | `f7c8bd7fbc456484d1a40ca88d32dc8da3104c422a5addd89f7d033b12c82511` |
| Hash-only split manifest | `aa40b696aa91cebb5c0c77774db170d4450a8d6d712087731d9b28cf23557050` |
| Pre-audit lock | `7e882a9b0253f0ae8bc613a54c3ad088540bd229eed45be8a746c78eac3ff80c` |
| Pilot-selection lock | `153b6544376bd9f6c43fb098358db9cfd85eb746cb05cb27d81db12b0f088e7c` |
| Verification record | `d8137fb6ba3ac769e3807d60a0e3471f604490a9b67c861fb433c99b4d5defa0` |

Prepare independently reproduced the frozen 7,265 canonical sample rows, 7,233 distinct content groups, all three task content hashes, and the aggregate dataset hash. It resolved tokenizer provenance only from local cache evidence. For Qwen3-1.7B, no immutable source-repository commit was present locally; the audit therefore makes no such claim and records only exact tokenizer-byte identity to the immutable Qwen3-0.6B snapshot.

The first selection launch exited before entering `align_chat_messages_soft` or writing any shard because Python resolved an older installed `rosetta` package. No natural result was emitted or overwritten. The unchanged frozen code was relaunched with `PYTHONPATH` pinned to the hashed worktree production sources; all completed shards and final provenance refer to those local files.

## Selection support: fit + calibration

Primary support is `primary_structural_m2: m>=2`. Wilson intervals below are ordinary two-sided 95% intervals. The Bonferroni-9 LCB is retained only in the machine-readable sensitivity output and did not affect eligibility or ranking.

| Pair | Task | Positive groups / total | Support rate | Wilson 95% | Eligible parents: m0 / m1 / m2 / m3 / m4 |
|---|---|---:|---:|---:|---:|
| tinyllama | ai2-arc | 511 / 511 | 100% | [99.2539%, 100%] | 0 / 52,839 / 9,640 / 3,187 / 241 |
| tinyllama | openbookqa | 228 / 228 | 100% | [98.3431%, 100%] | 0 / 19,108 / 3,705 / 1,353 / 7 |
| tinyllama | mmlu-redux | 2,495 / 2,495 | 100% | [99.8463%, 100%] | 0 / 320,152 / 55,677 / 16,788 / 1,196 |
| qwen25_0p5b | ai2-arc | 0 / 511 | 0% | [0%, 0.7461%] | 0 / 65,907 / 0 / 0 / 0 |
| qwen25_0p5b | openbookqa | 0 / 228 | 0% | [0%, 1.6569%] | 0 / 24,173 / 0 / 0 / 0 |
| qwen25_0p5b | mmlu-redux | 56 / 2,495 | 2.2445% | [1.7325%, 2.9033%] | 0 / 393,403 / 410 / 0 / 0 |
| llama32_1b | ai2-arc | 0 / 511 | 0% | [0%, 0.7461%] | 0 / 65,907 / 0 / 0 / 0 |
| llama32_1b | openbookqa | 0 / 228 | 0% | [0%, 1.6569%] | 0 / 24,173 / 0 / 0 / 0 |
| llama32_1b | mmlu-redux | 50 / 2,495 | 2.0040% | [1.5234%, 2.6322%] | 0 / 393,507 / 306 / 0 / 0 |
| qwen3_1p7b control | ai2-arc | 0 / 511 | 0% | [0%, 0.7461%] | 0 / 65,907 / 0 / 0 / 0 |
| qwen3_1p7b control | openbookqa | 0 / 228 | 0% | [0%, 1.6569%] | 0 / 24,173 / 0 / 0 / 0 |
| qwen3_1p7b control | mmlu-redux | 56 / 2,495 | 2.2445% | [1.7325%, 2.9033%] | 0 / 393,403 / 410 / 0 / 0 |

The absence of `m0` among eligible parents is an observed property of this frozen input/alignment configuration; the zero-support contract remains operative and no pseudo candidate was created.

## Readiness and pilot lock

| Pair | Minimum task positive groups | Pooled positive groups | Task-macro support | Ready | Rank |
|---|---:|---:|---:|---:|---:|
| tinyllama | 228 | 3,234 | 100% | yes | 1 |
| qwen25_0p5b | 0 | 56 | 0.7482% | no | — |
| llama32_1b | 0 | 50 | 0.6680% | no | — |
| qwen3_1p7b control | 0 | 56 | 0.7482% | excluded | — |

Only TinyLlama passes both frozen conditions: at least 30 positive distinct content groups in every task and at least 100 pooled. Therefore only a single-pair pilot claim is supported at this gate; no cross-pair confirmatory claim is available.

## Integrity and verification

The independent verifier passed all locked checks:

- exact schema and column order;
- exactly 60 pair×task×split aggregate rows;
- ordinary Wilson 95% and sensitivity-only Bonferroni-9 LCB;
- no NaN/Inf;
- no unresolved invalid/negative/nonfinite mass, duplicate legal source index, normalization/uniform failure, `m>4`, or content-group inconsistency;
- m3/m4 did not affect selection;
- Qwen3 did not enter readiness or ranking;
- model-selection/test reporting did not alter the pilot lock;
- deterministic reread/reduction and all provenance hashes.

The seven detailed artifacts remain under:

`local/final_results/fpct_factorized_transport/fpct_1b_ambiguity_support/rev_7f8af71968a39bc6cba2e4e34de762b291cda834/`

The 4,345,744-row parent file and other per-parent/sample/group artifacts are intentionally untracked. Their hashes, row counts, and byte sizes are recorded in `recipe/eval_recipe/fpct_1b/fpct_1b_result_manifest.json`.

No Hugging Face/LLM model was instantiated or run. No model weights or checkpoints were loaded, read, or modified. No GPU, CUDA computation, Kubernetes, training, accuracy evaluation, Phase2A outcome, selector output, or beneficial/harmful event was used.
