# FPCT-E1 A5-R1 E0 Data-Tree Hash-Domain Recovery Addendum (Draft)

> - Status: `HUMAN APPROVAL REQUIRED / NOT OPERATIVE`
> - Requested approval ID: `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`
> - Classification: `PROSPECTIVE_IMPLEMENTATION_PROVENANCE_CORRECTION`
> - Scientific input, operator, estimand, threshold, population: `UNCHANGED`
> - Blocked execution: `9b248d2094b684f5d9e9a218919a354b7d97468e`
> - Blocked execution reuse/resume: `FORBIDDEN`
> - E1-2, E1-3, E1-pilot, confirmatory: `NOT AUTHORIZED`

## 1. Pre-group-1 failure

The clean A5 execution `9b248d20` passed source-snapshot provenance and resolved,
enumerated, and loaded the local tokenizer objects. It then failed before the
later strict tokenizer-bundle/template attestation completed and before the
first E0-design group was loaded or rendered.
The immutable receipt is `A5_INPUT_LOCK_BLOCKED`, with failed check
`a5_materialized_e0_development_tree_sha_changed`. No census row, alignment row,
sidecar, geometry, model output, checkpoint, GPU, Kubernetes job, or scientific
result was produced.

The run root is permanently abandoned:

`/netdisk/lijunsi/fpct-e1/fpct-e1-a5-9b248d20-v1`

Immutable evidence SHA256:

- initial A5 gate: `464e646c336d33c97a03c603d283a95b1579508d63ae0f19d7c700ccf2dcfc02`;
- source-snapshot receipt file: `dc4396d50594f0db120fcba9252e85c92e343a014230379ebe8c51e645082981`;
- execution identity: `2247ccf274ea8c55b25b55fd5ac3fb2a191d6796f5ea06a9b18b4a9d8e6b7a0b`;
- blocked receipt: `fe305b2c9d881b202f4a096646e9530e8d630a6fcfa095754d3101450ad7ab79`.

## 2. Root cause: two valid hashes from different domains

The data bytes did not drift. The producer compared hashes from two distinct
algorithms:

- frozen E0 declared identity, algorithm
  `relative_path_nul_file_sha256_bytes_v1`:
  `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`;
- A5 rich generic before/after manifest, algorithm
  `canonical_json_file_manifest_v1`:
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`.

Both were recomputed over the same 51 files / 233,569 bytes. The existing exact
E0 reference implementation recomputed the frozen E0 value byte-for-byte. The
generic value is not an alternative value in the frozen E0 declared-hash domain
and must never be cross-compared with the E0 declared value.

## 3. Proposed prospective repair (not yet authorized)

The successor stores both hashes with explicit algorithm domains:

1. `e0_declared_tree_sha256` is computed by the existing
   `script.experiment.fpct_e1_capture_runner._e0_declared_tree_sha256`, which
   reproduces `fpct_e0_runner.tree_sha256` exactly. Only this field is compared
   with the frozen E0 identity.
2. `tree_sha256` remains the richer generic canonical-manifest hash. It is used
   only for A5 before/after input-asset equality and tamper detection.
3. The compact sidecar and strict main manifest bind both domains, their
   algorithm IDs, and the aggregate before/after asset SHA.

This correction does not change any file, group, answer, prompt, tokenizer,
alignment parameter, candidate, topology, streaming row, operator, checkpoint,
lambda, endpoint, or decision rule.

## 4. Successor execution requirements if approved

Before any new E0-design row access, the changed contract/code/tests must pass a
fresh A5 prompt gate and be committed and pushed on the research branch. The
successor must use a new commit, source snapshot, UID, and empty run root and must
restart from group 1. No byte or receipt from `9b248d20` may be reused.

Until the requested approval is explicit, no recovery gate, successor commit,
snapshot, run root, natural census, or input lock may be launched.
