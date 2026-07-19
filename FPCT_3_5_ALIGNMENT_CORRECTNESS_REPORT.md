# FPCT-3.5 Alignment Correctness Report

## Decision

FPCT-3.5 identity forensic is `GO TO CONDITIONAL CORRECTION`.

- All 7,265 canonical samples passed exact Qwen3-0.6B receiver versus Qwen3-1.7B sender runtime identity.
- All 802 raw Qwen3 `m>=2` parents across all splits were classified as `duplicate_or_overlap_receiver_offsets`; none was certified one-to-many.
- The frozen fit+calibration consistency check exactly reproduced 56 positive content groups and 410 `m2` parents.
- No rendered-text, token-ID, offset-array, content-span, message-range, tokenizer-path or behavior-fingerprint mismatch occurred.
- No row was `unexplained_other`.

This GO authorizes implementation of explicit `exact_identity` and the common certified/slot-0 sanitizer. It does not authorize GPU, pretrained-model forward, training or accuracy evaluation.

## Frozen execution

| Item | Value |
|---|---|
| Pre-data execution SHA | `0398d26b63e96263b813730368275ee66e313f66` |
| Pre-data lock SHA256 | `ebf184ad578ae4e864a63c315a95fc07060b0ca4aebd58015f1b938d8f1287e2` |
| Forensic summary SHA256 | `1dc030adcc2e6d45a822c7f01899f442fc49cde654868df600d6e5fb28848584` |
| Receiver/Qwen3 behavior fingerprint | `ccf72f82e8b6bdc2f0b2e198d7bab79d9c1579b5c194e757b4a1159bba603207` |
| Qwen2.5 behavior fingerprint | `8fe3f6d6363e6935676082b7c4e9ce1b749e74566c9b98f5dfcd96ded57ce887` |

The first MMLU computation reached 5,615/5,615 but was denied permission when creating the sibling-worktree local artifact directory. It produced no partial artifact or manifest. The exact frozen command was rerun with the required write permission; scientific code, inputs and parameters were unchanged.

## Runtime identity

The receiver and Qwen3 control matched on:

- tokenizer-relevant files and behavior fingerprint;
- backend tokenizer JSON, vocab, added tokens and special tokens;
- normalizer, pre-tokenizer, decoder and post-processor;
- chat template;
- rendered text, IDs, offsets, content spans and message ranges for every sample;
- exact recovery of each original message content from its rendered content span.

`name_or_path` and model configuration were audited separately and were not used as behavior identity fields. The receiver and sender directories remained distinct and correct.

## Root cause of the 410 fit+calibration parents

Every raw anomaly is a paired ByteLevel-offset alias. A common pattern is:

- one token represents a leading-space-plus-symbol string, such as `" Ω"`, with offset `[x,x+2]`;
- the adjacent token represents the symbol alone, such as `"Ω"`, with offset `[x+1,x+2]`;
- both receiver tokens therefore have overlapping positive-length offsets;
- raw soft-span alignment gives both parents the same two candidates with weights `[0.5,0.5]`;
- each row includes its identity index, but neither row is a geometrically valid receiver-one/source-many partition.

All 802 rows have `m=2`, uniform weights and an identity candidate. They form 401 paired overlap events; fit+calibration contains 410 parents, or 205 paired events. Exact identity correctly maps each parent only to itself.

## Taxonomy and distribution

| Split | Raw parents | Positive groups |
|---|---:|---:|
| fit | 280 | 35 |
| calibration | 130 | 21 |
| model-selection | 112 | 13 |
| test | 280 | 35 |
| all | 802 | 104 |

Primary taxonomy:

- `duplicate_or_overlap_receiver_offsets`: 802
- every other registered category: 0

Parent-level Unicode-category incidence:

- mathematical symbol `Sm`: 692 parents;
- space separator `Zs`: 396;
- open punctuation `Ps`: 86;
- lowercase letter `Ll`: 50;
- uppercase letter `Lu`: 34;
- control `Cc`: 16.

Dominant code points were SPACE (396 occurrences), SUPERSET OF `⊃` (268), LOGICAL OR `∨` (162), IDENTICAL TO `≡` (82), THERE EXISTS `∃` (80), LEFT PARENTHESIS (76) and FOR ALL `∀` (72).

MMLU subject counts were dominated by `formal_logic` (666/802), followed by high-school statistics and physics (18 each), college computer science (18), nutrition (14), professional law (12), and smaller STEM/business/social-science cells. These are descriptive offset-distribution facts, not task outcomes.

## Qwen3 versus Qwen2.5 row-level comparison

The sender directories, runtime `name_or_path` values and fingerprints are distinct, ruling out a path/object alias:

| Sender | Directory | Runtime `name_or_path` | Runtime fingerprint |
|---|---|---|---|
| Qwen3-1.7B control | `/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen3-1.7B` | `/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen3-1.7B` | `ccf72f82e8b6bdc2f0b2e198d7bab79d9c1579b5c194e757b4a1159bba603207` |
| Qwen2.5-0.5B | `/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen2.5-0.5B-Instruct` | `/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen2.5-0.5B-Instruct` | `8fe3f6d6363e6935676082b7c4e9ce1b749e74566c9b98f5dfcd96ded57ce887` |

Nevertheless, the affected row sets are exactly equal:

| Set | Qwen3 | Qwen2.5 | Intersection | Differences | Jaccard |
|---|---:|---:|---:|---:|---:|
| Positive content groups | 104 | 104 | 104 | 0/0 | 1.0 |
| `(sample,parent)` rows | 802 | 802 | 802 | 0/0 | 1.0 |
| Relative offset signatures | 802 | 802 | 802 | 0/0 | 1.0 |
| Candidate token-ID signatures | 802 | 802 | 802 | 0/0 | 1.0 |

The exact equality is therefore explained by shared offset/token behavior on these Unicode forms, not by aggregate coincidence and not by loading the wrong tokenizer path.

## Local artifacts

Root:

`/home/lijunsi/projects/Cache-fpct-factorized-transport/local/final_results/fpct_factorized_transport/fpct_3_5_alignment_correctness/rev_0398d26b63e96263b813730368275ee66e313f66/`

- MMLU forensic ledger: 802 rows, SHA256 `9dc0fe6e5120aae2cd3b39bfe5154b327cd4647565430f48c3bdd3dcd2d024bf`.
- MMLU Qwen3/Qwen2.5 comparison rows: 802 rows, SHA256 `aab208a8f1ec98b669ebf3c0ce3776e5a4ddfd85d192652ad91cda8ab8410767`.
- ARC summary SHA256: `c41efb514c67e4ca614434d505cf33625c00c55b7907622fa8717f8d4d4f2712`; its schema-only ledger/comparison contain zero rows with SHA256 `756114bc...` and `f9fdbe40...`.
- OpenBookQA summary SHA256: `10efccbb9907d9d063ad22521e220e0d4c7f5007c103784c2509a0e5dfb27d74`; its schema-only ledger/comparison contain zero rows with the same respective SHA256 values.
- MMLU summary SHA256: `56bcef073d5924eb069aa9e4e9c72bf46ad298986e6a97bfdf2359ed695b88c6`.
- Detailed runtime tokenizer payloads and per-parent rows remain local-only and are not committed.

## Claim boundary

This result proves exact runtime identity for the Qwen3 control and explains the raw same-tokenizer soft-span anomalies as offset aliases. It does not prove that TinyLlama heterogeneous rows are certified, query-separable or accuracy-beneficial. Those questions require the corrected certified-support audit and later operator/model gates.
