# FPCT-3.5 Alignment Correctness Protocol

## 1. Status, timing and claim boundary

This protocol is frozen before any new per-parent natural anomaly ledger is generated. The historical FPCT-1B soft-span artifacts remain immutable and are retained as a raw structural-support audit. FPCT-3.5 asks whether apparent one-to-many rows are tokenizer-induced, exact-identity artifacts, or offset aliases before any GPU, pretrained-model forward, training or accuracy output is released.

Throughout the existing implementation, `slm` denotes the receiver tokenizer/sequence and `llm` denotes the sender tokenizer/sequence. FPCT-3.5 freezes that direction explicitly.

The single confirmatory pair remains TinyLlama-1.1B sender to Qwen3-0.6B receiver. The same-tokenizer Qwen3-1.7B sender is an exact-identity control and can never enter pilot ranking. `F-C_post` remains the unique headline contrast. Candidate count alone is not a mechanism or accuracy claim.

FPCT-3.5 is `GO` only if all Qwen3 runtime identity invariants hold, every raw Qwen3 `m>=2` row receives a frozen offset-taxonomy explanation, no tokenizer/pair path mix-up exists, and the corrected certified-support audit passes its integrity checks. Any failure makes FPCT-3.5 `INCONCLUSIVE` and blocks all GPU/Kubernetes/training stages.

## 2. Immutable inputs

- Starting research commit: `d296a18be9cc3b0dce3c07f4c2d7244145f2e3ac`.
- FPCT-1B execution commit: `7f8af71968a39bc6cba2e4e34de762b291cda834`.
- Operative FPCT-1A v2 manifest SHA256: `f7c8bd7fbc456484d1a40ca88d32dc8da3104c422a5addd89f7d033b12c82511`.
- Canonical input SHA256: `0366be7e5b129710024543bc065774bef165b6b6bca92541a3d641aea2918114`.
- Split manifest SHA256: `aa40b696aa91cebb5c0c77774db170d4450a8d6d712087731d9b28cf23557050`.
- Canonical sample count: 7,265; distinct content groups: 7,233.
- Historical raw artifacts are read-only and are never overwritten.

Before natural forensic execution, the protocol, manifest, diagnostic code and synthetic tests must be committed and pushed. The clean commit containing them is the pre-data execution SHA. A local-only lock records that SHA and all tracked dependency hashes. Any later change to those frozen files invalidates the forensic run.

## 3. Exact runtime identity

`exact_identity` is permitted only when every item below is exactly equal between receiver and sender tokenizers for the canonical call:

1. rendered chat text;
2. input IDs;
3. offset mappings;
4. content spans;
5. message token ranges;
6. complete vocabulary including added tokens;
7. special-token definitions;
8. backend normalizer, pre-tokenizer, decoder and post-processor;
9. chat template;
10. tokenizer-relevant file set and file hashes;
11. deterministic runtime tokenizer fingerprint.

Each computed content span must also slice the rendered text back to the exact original non-empty message content; equal-but-wrong span heuristics are a hard error.

The tokenizer-relevant file fingerprint includes tokenizer JSON/model/config, vocab/merges, added-token and special-token files. Model architecture config and generation config are not tokenizer files and do not enter this fingerprint.

If any invariant fails, `exact_identity` raises a hard error. It must not fall back to soft span alignment.

For every eligible receiver parent `i`, exact identity produces:

`source_indices[i] = [i,-1,-1,-1]`

`source_weights[i] = [1,0,0,0]`.

Therefore every eligible parent has `m=1`, `fallback=false`, and no FPCT extra slot. Non-eligible/template positions remain inactive.

## 4. Common relative-coordinate system

Certification is performed in message-relative character coordinates, not by comparing raw offsets from potentially different rendered templates. For a receiver token interval and source candidate interval, clip each raw offset to its tokenizer-specific message content span and subtract that span's start. The resulting intervals refer to the same message content string.

Zero-length clipped intervals are never certified. A raw `m>=2` parent is certified one-to-many if and only if:

1. the receiver interval has positive length;
2. it has no positive-length overlap with another eligible receiver interval in the same message;
3. legal source candidate indices are unique;
4. every candidate source interval has positive length;
5. every candidate interval intersects the receiver interval;
6. candidate intersections with the receiver interval are pairwise disjoint;
7. their union covers the receiver interval exactly, without gaps or over-coverage;
8. candidate source indices are monotonic in source-span order;
9. the retained candidate set exhausts all positive-overlap source tokens before top-k truncation (`positive_overlap_counts == legal candidate count`);
10. there is no zero-length, duplicate-offset or overlap alias.

Only these rows contribute to certified `m>=2` support. All other raw `m>=2` rows are `offset_uncertified`.

## 5. Mutually exclusive anomaly taxonomy

Every raw anomaly receives exactly one primary category using this frozen precedence:

1. `tokenizer_or_pair_path_mixup`;
2. `rendered_text_difference`;
3. `token_id_difference`;
4. `offset_difference`;
5. `zero_length_offset`;
6. `duplicate_or_overlap_receiver_offsets`;
7. `exact_duplicate_source_offsets`;
8. `partial_overlap_source_offsets`;
9. `candidate_missing_identity_index`;
10. `unexplained_other`.

For heterogeneous pairs, normal rendered/ID/offset differences are recorded as comparison flags but do not by themselves make a structurally valid row uncertified. The primary category for a raw `m>=2` row is assigned from its local interval geometry unless a path/fingerprint integrity failure exists. For the Qwen3 exact control, any rendered/ID/offset mismatch is a hard identity failure.

## 6. Local-only forensic ledger

The per-parent ledger is stored only under:

`local/final_results/fpct_factorized_transport/fpct_3_5_alignment_correctness/rev_<execution_sha>/`.

It contains no label, answer, correctness, prediction or Phase2A field. Required columns include:

- pair, task, split, sample hash and content-group hash;
- subject for distribution summaries;
- receiver/source token indices and token IDs;
- escaped token text;
- raw offsets and clipped message-relative intervals;
- Unicode code point, name and category for the receiver interval text;
- candidate indices, IDs, offsets, clipped intervals and weights;
- whether the identity index is present;
- anomaly category, certified flag and certification failure reason;
- sender directory, `name_or_path`, tokenizer file fingerprint and runtime fingerprint.

The ledger and detailed comparisons are never committed. Tracked reports contain only aggregate counts and hashes.

## 7. Qwen3 identity forensic

All 7,265 canonical samples are checked. The audit must establish sample-by-sample equality of rendered text, IDs, offsets, content spans and message ranges, in addition to the once-per-runtime tokenizer fingerprint equality.

Every raw Qwen3 `m>=2` row is enumerated. The previously published FPCT-1B fit+calibration facts—56 positive content groups and 410 `m2` parents—are prior consistency checks, not adjustable thresholds. The audit reports category counts, Unicode categories and MMLU subject distribution.

Any identity mismatch, unexplained row or tokenizer/path mix-up stops the study before production correction or GPU use.

## 8. Qwen3 versus Qwen2.5 row-level comparison

Without using labels or outcomes, compare:

- positive content-group sets;
- positive `(sample_hash,parent_index)` sets;
- offset-signature sets;
- candidate-token-ID signatures;
- sender directory, `name_or_path`, tokenizer file fingerprint and runtime fingerprint.

For each set report equality, intersection size, left/right differences, union size and Jaccard. Aggregate-count equality is never accepted as a root-cause explanation.

## 9. Conditional correction

Only after the Qwen3 identity gate passes:

1. add an explicit production `exact_identity` alignment mode;
2. force the Qwen3 exact control to use it;
3. add a common heterogeneous sanitizer used identically by `C_pre`, `C_post` and `F`:
   - certified rows retain their original candidates and normalized `A`;
   - offset-uncertified raw `m>=2` rows become deterministic slot-0 one-hot rows;
   - `m=0` and `m=1` retain the frozen contracts;
   - certification/sanitization is revalidated after max-length/source truncation against the actually retained source sequence;
   - the non-FPCT legacy/default path remains unchanged.

Raw slot 0 must itself be a legal positive-mass candidate. If it is not, sanitization hard-fails rather than choosing another anchor. Parent entropy is recomputed from the sanitized row once and shared by all three arms.

No F-only sanitizer, parameter, fallback or gate is allowed.

## 10. Certified support re-audit

The corrected execution SHA runs all pair/task/split cells and reports both raw and corrected quantities:

- raw and certified `m0/m1/m2/m3/m4`;
- offset-uncertified parent/sample/group counts;
- raw-minus-certified support;
- ordinary two-sided Wilson 95% intervals;
- raw and certified expansion, K/V memory and attention-score FLOP estimates.

Readiness and ranking use certified `m>=2` only. Qwen3 is reported twice: historical raw soft-span diagnostic and exact-identity control.

The Qwen3 exact control hard expectation is `m0=0`, all eligible parents `m1`, `m2=m3=m4=0`, and zero extra slots.

TinyLlama remains single-pair pilot-ready only if each task has at least 30 certified-positive distinct content groups and the pooled count is at least 100. Otherwise all GPU work stops.

## 11. Synthetic tests frozen before natural data

The pre-data suite must cover:

- identical tokenizer with duplicate raw offsets still yields exact-identity `m=1`;
- identical tokenizer with ordinary offsets yields exact-identity `m=1`;
- genuine receiver-one/source-many disjoint partition is certified;
- duplicate source offsets are uncertified;
- overlapping receiver offsets are uncertified;
- text/ID/offset mismatch hard-fails exact identity;
- sanitizer makes uncertified rows one-hot for all three arms;
- identity rows create zero extra FPCT slots.

## 12. Decisions and stopping rules

FPCT-3.5 reports one of:

- `GO`: runtime identity passes, all Qwen3 anomalies explained, no mix-up, correction tests pass and certified audit is complete;
- `INCONCLUSIVE`: any identity/taxonomy/provenance/integrity failure;
- `NO_GO_GPU`: correctness passes but TinyLlama certified readiness fails.

An `INCONCLUSIVE` or `NO_GO_GPU` result blocks pretrained-model forward, GPU numerical tests, Kubernetes, training and accuracy evaluation. Thresholds and taxonomy cannot be relaxed after observing natural anomalies.
