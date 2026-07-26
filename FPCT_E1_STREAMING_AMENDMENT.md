# FPCT-E1 A4 Representation-Preserving Streaming Amendment

> Approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A4_STREAMING`
> Approval date: 2026-07-26 (Asia/Shanghai)
> Protocol ID: `fpct_e1_mechanism_audit_v6_representation_preserving_streaming`
> Current phase: `PRE-NATURAL-DATA IMPLEMENTATION AND SYNTHETIC GATES`
> Prospective authorization: successor input lock after synthetic GO; runtime,
> checkpoint/plan, and E1-2 only after input-lock GO; E1-3 only after finalized
> E1-2
> Population currently authorized after the gates: E0-design only
> E1-pilot: `SEALED / NOT RENDERED / NOT TOKENIZED / NOT ALIGNED / NOT RUN / NOT READ`

## 1. Decision and prospective scope

The user prospectively approves replacement of the cumulative per-sample
logical-row materialization with deterministic bounded streaming. This is an
engineering-only, representation-preserving amendment. It changes only how
logical rows are generated, chunked, written, verified, resumed, and read.

The amendment preserves without modification:

- the complete logical Cartesian row universe;
- every existing logical row-key field and value;
- candidate identity, prior `A`, mask, and topology;
- every row's statistical weight;
- the `C_post` and `F` endpoints;
- teacher-forced `Delta log p(y*)`, end-task accuracy, flips, and every other
  frozen estimand;
- the E1-2 checkpoint/operator cells and E1-3 centered-lambda grid;
- the E0-design population and split membership;
- the sealed status of E1-pilot, model-selection, test, and confirmatory data.

The approved change does not modify E0, does not make E1 confirmatory, and does
not authorize training or operator selection. Retaining every layer, head,
answer query, and certified parent is mandatory; engineering pressure may not
be resolved by changing the scientific population.

## 2. Historical attempts remain abandoned

Execution `612697dfc44ab46699728b8d2de0a6fce980a889` remains:

```text
INCONCLUSIVE_RESOURCE_CEILING / ABANDONED
resume_allowed = false
artifact_reuse_allowed = false
scientific_result = false
```

This approval is prospective and cannot be applied retroactively. The observed
`616448` rows are recorded only as the trigger for review. They may not be used
to select a replacement ceiling. The old `262144` limit remains the historical
engineering memory-integrity guard for that abandoned cumulative
representation; it was not a scientific threshold.

No artifact or partial state from `744a943...`, `d1698177...`, or
`612697df...` may enter the A4 successor execution. A successor requires a new
clean pushed commit, source archive and receipt, immutable mounted-tree receipt,
run UID, run root, execution plan, and receipts, and it must restart the CPU
E0-design input lock from its first sample.

## 3. Explicitly not approved

The following remain forbidden:

- raising `262144` to a value chosen from the observed `616448`;
- reinterpreting or resuming the old execution in place;
- dropping or sampling samples, parents, queries, layers, heads, candidates, or
  rows;
- truncation, early stopping, topology filtering, or a sample-specific escape;
- changing alignment, sanitizer, candidate order, prior, masks, operators,
  thresholds, splits, checkpoints, lambda grid, endpoints, or estimands;
- loading models/checkpoints or starting an E1-2 shard before the successor
  input lock is complete and has a `GO` receipt;
- rendering, tokenizing, aligning, running, or reading E1-pilot;
- accessing model-selection, test, confirmatory, or Phase2A outcomes.

## 4. Logical-row contract

For sample `s`, freeze:

```text
L   = 28 receiver layers
Hq  = 16 receiver query heads
Q_s = len(answer_queries_s)
P_s = len(certified_parents_s)
N_s = L * Hq * Q_s * P_s
```

`answer_queries_s` and `certified_parents_s` retain the exact sequence frozen
by the compact input geometry. Neither may be sorted, filtered, or regenerated
differently during Pass B.

The sample-local ordinal is:

```text
row_ordinal = (((layer * Hq + query_head) * Q_s + query_idx) * P_s
               + parent_idx)

0 <= layer       < 28
0 <= query_head  < 16
0 <= query_idx   < Q_s
0 <= parent_idx  < P_s
0 <= row_ordinal < N_s
```

Its required inverse is:

```text
parent_idx = row_ordinal % P_s
x = row_ordinal // P_s
query_idx = x % Q_s
x //= Q_s
query_head = x % 16
layer = x // 16
```

The forward and inverse maps must be bijective for every sample. The ordinal is
only a physical locator. It does not replace or enter the existing logical row
key.

### 4.1 Existing logical row key

The existing row-key array remains in this exact field order:

```text
sample_sha256
content_group_sha256
input_sha256
alignment_sha256
labels_sha256
gold_response_sha256
layer
query_head
kv_head
query_position
target_position
target_token_id
parent_position
candidate_count
topology
```

`kv_head` retains the frozen GQA map:

```text
kv_head = floor(query_head / (Hq / Hkv))
```

`row_ordinal` is added beside that key for physical localization. It is not a
component of the key and must not change any legacy join.

### 4.2 Stable identities

Canonical JSON means UTF-8 JSON with sorted object keys, compact separators,
finite values only, and no trailing line feed. `endpoint_id` is the canonical
JSON array `[seed, checkpoint_arm, inference_operator, cell, task,
lambda_value]` in that exact order; it therefore distinguishes every frozen
seed/checkpoint/operator/task/lambda endpoint. Freeze:

```text
logical_row_id = SHA256(
    UTF8("fpct-e1-a4-logical-row-id-v1") || NUL ||
    canonical_json(existing_row_key_array)
)

endpoint_row_id = SHA256(
    UTF8(endpoint_id) || NUL || ASCII(logical_row_id)
)
```

The logical ID is independent of endpoint, seed/checkpoint, lambda, chunk size,
file path, Parquet encoding, and run UID. The endpoint ID mechanically binds a
logical row to one frozen endpoint. IDs are lowercase 64-character hexadecimal
SHA256 values.

## 5. Deterministic physical chunks

Production physical chunks contain at most:

```text
physical_chunk_rows = 4096
```

Chunk `c` for sample `s` covers the half-open range:

```text
[c * 4096, min((c + 1) * 4096, N_s))
```

The final chunk may be smaller. A production chunk may never be larger than
4096 rows. Smaller sizes are allowed only in synthetic equivalence tests.

Each chunk manifest row must record at least:

```text
sample_sha256
chunk_index
first_row_ordinal
end_row_ordinal_exclusive
row_count
first_logical_row_id
last_logical_row_id
canonical_row_stream_sha256
physical_file_sha256
physical_file_bytes
schema_sha256
```

It also records a relative immutable file path and protocol identity. Coverage
is proven by ordered ranges, not by retaining a set of all row IDs:

- the first range starts at zero;
- adjacent ranges are exactly contiguous;
- no range overlaps another range;
- the final range ends at `N_s`;
- every range is nonempty and no larger than 4096;
- the sum of `row_count` equals `N_s`;
- every emitted ordinal and logical row ID matches the frozen map.

## 6. Semantic and physical provenance are different objects

Parquet bytes may differ when chunk or row-group boundaries differ. Therefore,
physical-file SHA equality is not a requirement for chunk-partition
equivalence. Physical SHA256, byte size, schema SHA, and relative path are still
mandatory provenance.

Semantic equivalence requires exact equality of:

- decoded logical row sequence in frozen ordinal order;
- logical row count and IDs;
- row-key values, metrics, weights, topology, and endpoint identity;
- canonical semantic stream SHA256;
- every final aggregate and estimand.

For each row, canonical semantic-row bytes contain every semantic field but no
physical path, chunk index, row-group metadata, file SHA, or run-local value.
The sample semantic stream hash is incrementally computed in ordinal order as:

```text
SHA256(canonical_semantic_row_0 || LF || ... ||
       canonical_semantic_row_(N_s-1) || LF)
```

This hash must be invariant when the same synthetic stream is written with
chunk sizes `1`, `7`, `257`, `4096`, or `8192`. The `8192` case is a synthetic
partition-equivalence test only; production remains capped at 4096.

## 7. Deterministic reductions

Formal analysis always reads samples in ascending `sample_sha256` order and
rows within a sample in ascending `row_ordinal` order. If generation is
parallel, range manifests are merged into this order before any formal
accumulator is updated. Shard or chunk completion order must not affect a
floating-point result.

Content-group, task, and cross-task weights remain exactly as preregistered.
Chunk-local summaries are integrity diagnostics only; they cannot replace the
canonical row-level analyzer or become formal estimands.

## 8. Two bounded passes

### 8.1 Pass A: compact geometry lock

Pass A stores, per sample, only:

- task and sample/content-group/provenance hashes;
- the frozen `answer_queries` and `certified_parents` sequences;
- `Q_s`, `P_s`, and exact integer `N_s`;
- answer-query and parent-sequence SHA256 values;
- compact raw-topology metadata and its hash;
- expected chunk count.

It must not expand logical rows. Its outputs are:

```text
input_geometry_manifest.json
input_geometry_samples.parquet
input_geometry_receipt.json
```

Pass A freezes exact E0-design count/sum/min/p50/p95/max/argmax row volume and
estimated physical rows, disk bytes, file count, and inode need. A disk and
inode preflight must pass before Pass B.

### 8.2 Pass B: streaming attestation/materialization

Pass B derives rows from the compact geometry in ordinal order, holding no more
than 4096 expanded rows at once. Its required asymptotic memory is:

```text
O(Q_s + P_s + physical_chunk_rows)
```

and never `O(28 * 16 * Q_s * P_s)`. Compact sidecars should be one immutable
sample artifact plus a deterministic index. Any monolithic sidecar is permitted
only if it contains no expanded logical rows and bounded-memory evidence proves
that peak RSS does not grow linearly with `N_s`.

## 9. Pre-natural synthetic qualification

No new natural E0-design row may be read until all tests below pass on CPU and
their receipts are committed and pushed.

### 9.1 Reference-set equivalence

The historical materialized row generator is retained as test-only reference
for small synthetic cases. The matrix covers:

```text
Q in {1,2,3}
P in {1,2,4}
L in {1,2,28}
Hq/Hkv ratio in {1,2,4}
```

Reference and streaming implementations must have identical row-key multisets,
row values, weights, topology, `kv_head` mapping, endpoint identities, semantic
stream hashes after canonical ordinal replay, and final aggregates.

### 9.2 Chunk-partition invariance

Logical partition sizes `1`, `7`, `257`, `4096`, and `8192` must produce
identical semantic streams, logical IDs, counts, metrics, and formal aggregates.
For the `8192` synthetic logical partition, the Parquet writer still emits at
most 4096 rows per physical batch/file; `8192` never relaxes that bound.
Physical Parquet SHA equality is neither expected nor tested.

### 9.3 Million-row stress and prospective RSS lock

A synthetic population with at least 1,000,000 logical rows must complete
without a logical-row ceiling, a million-row Python list, or whole-table read.
It must emit exactly the ordinal interval `[0,N)`, reproduce the semantic SHA,
and remain bounded.

The numerical peak-RSS threshold is intentionally not inferred from the
observed natural value `616448`. It is prospectively derived using two fresh
CPU processes with the same complete synthetic mechanism-row schema and the
production 4096-row Parquet writer:

```text
baseline_rows = the smallest 28 * 16 * Q * P that spans at least two chunks
stress_rows   = the smallest 28 * 16 * Q * P that is at least 1,000,000
chunk_uncompressed_bound = 4096 * max_canonical_row_bytes
allocator_and_buffer_allowance = 16 * chunk_uncompressed_bound
threshold_bytes = baseline_peak_rss_bytes + allocator_and_buffer_allowance
bounded_peak_rss = stress_peak_rss_bytes <= threshold_bytes
```

The multiplier 16 is frozen before the stress observation. It conservatively
allows simultaneous bounded Python-row, Arrow, Parquet writer, verifier, and
allocator copies while still rejecting memory that grows with the million-row
logical universe. The physical disk preflight estimate is the smallest power
of two not smaller than `max_canonical_row_bytes`; it is based on uncompressed
canonical fixed-schema rows rather than favorable synthetic Parquet
compression.

Before any successor natural input lock, the fixed environment, schema hashes,
4096-row production chunk, measurement method, both synthetic cases, measured
peaks, the formula above, threshold bytes, per-row disk estimate, and evidence
SHA must be written into the tracked pre-natural synthetic gate. Until that
objective synthetic-only record exists, `bounded_peak_rss` is false and natural
execution is blocked. A threshold selected after reading successor natural
rows is invalid.

`evidence_sha256` is not self-referential: it is SHA256 over the canonical,
LF-framed `evidence` object only. The containing gate file has its ordinary
external file SHA recorded by later source/run receipts.

### 9.4 Corruption rejection

The verifier must reject a missing or duplicate chunk, overlapping range,
ordinal gap, reordered or mutated row, wrong sample hash, endpoint, schema,
expected `N_s`, chunk SHA, or semantic stream SHA.

### 9.5 Crash/resume equivalence

Tests interrupt execution during a partial temporary chunk, before Parquet
close, before artifact rename, before manifest write, and before manifest
rename. Temporary files never count as complete. Completed immutable chunks are
not rewritten; an incomplete range restarts at its range beginning. The final
coverage and semantic SHA must equal a clean execution exactly.

### 9.6 Whole-table materialization prohibition

Production paths may not call or depend on:

```text
list(all_logical_rows)
pyarrow.read_table(full_artifact)
to_pylist(full_artifact)
read_text(large_jsonl)
pandas.read_parquet(full_artifact)
```

Synthetic tests monkeypatch these entry points to fail, in addition to static
review. Expanded logical rows may exist only in the current bounded chunk.

## 10. Successor input-lock hard gate

Every field below must be true:

```text
protocol_and_schema_versioned
old_attempt_artifact_reused == false
e1_pilot_consumed == false
model_or_checkpoint_loaded == false
gpu_or_kubernetes_used == false

expected_logical_rows_eq_emitted
ordinal_first_eq_zero
ordinal_last_eq_expected_minus_one
ordinal_ranges_contiguous
missing_rows == 0
duplicate_rows == 0
overlapping_chunks == 0

row_key_reference_equivalence
weights_reference_equivalence
topology_reference_equivalence
semantic_stream_replay_equal
chunk_partition_semantic_equivalence
aggregate_partition_equivalence

bounded_peak_rss
whole_table_materialization_detected == false
atomic_no_overwrite
crash_resume_equivalence

raw_topology_complete
task_counts_exact_128_70_128
input_assets_unchanged_during_lock
```

Any failure is `A4_INPUT_LOCK_BLOCKED`. It forbids creation of a runtime probe,
execution plan, ConfigMap, or checkpoint Job.

## 11. Execution order after implementation

1. Add only this protocol, versioned schema/contract, streaming implementation,
   verifier, synthetic tests, and the renderer/plan closure required by those
   changes.
2. Pass and freeze every synthetic/CPU gate without reading new natural
   E0-design rows or loading models/checkpoints.
3. Create and push a clean A4 commit containing no operator, alignment, split,
   threshold, lambda, checkpoint, or estimand change.
4. Create a new source archive, immutable receipts, empty run root, UID, and
   execution plan. Never copy an old run root.
5. Restart the CPU E0-design compact geometry lock and streaming input lock from
   the beginning.
6. Only after a complete input-lock `GO` may the immutable-image probe,
   six-checkpoint/tree lock, 108-shard plan, ConfigMap verification, E1-2
   `C_post`, and E1-2 `F` stages be created or executed.
7. E1-3 remains dependent on a complete and deeply verified immutable E1-2
   finalized receipt. The old DAG is historical and non-operative; A4 must
   freeze a new DAG and new receipts.

Throughout E1-2 and E1-3, E1-pilot remains:

```text
SEALED / NOT RENDERED / NOT TOKENIZED /
NOT ALIGNED / NOT RUN / NOT READ
```

This amendment first authorizes recovery of the engineering input-lock path and
prospectively condition-authorizes the downstream A4 DAG exactly as stated
above. It does not itself assert mechanism activation, headroom, task
improvement, or confirmatory evidence, and it does not authorize training or
operator selection.
