# FPCT-3.5P Sealed Provenance Replay Report

Execution SHA: `7aecf2370df8a544b553baa6a7a58b24191e02ef`

Stable closure fingerprint: `609e3f055d452b554f7d9619a36342503b7894b1dfbb72d93d5c13374f7a590b`

Status: `PROVENANCE_CONFIRMED`

The replay used the absolute `python3.10 -I` bootstrap and the same-process
module closure. All three task shards and finalization reproduced the freeze
fingerprint before and after target execution. The old Conda editable metadata
was recorded, but every loaded `rosetta.*` origin and the formal audit targets
resolved inside the current FPCT worktree.

## Original versus replay

| Quantity | Historical | Sealed replay | Equal |
|---|---:|---:|---|
| canonical samples | 7,265 | 7,265 | yes |
| runtime-identity samples | 7,265 | 7,265 | yes |
| raw Qwen3 m2 parents | 802 | 802 | yes |
| distinct positive groups | 104 | 104 | yes |
| fit+calibration m2 parents | 410 | 410 | yes |
| fit+calibration positive groups | 56 | 56 | yes |
| overlap clusters | 401 | 401 | yes |
| Qwen3/Qwen2.5 paired-equal rows | 802 | 802 | yes |

The ordered and multiset normalized ledger projections, ordered and multiset
`(candidate index, token ID, offset, weight)` atoms, radius-four context token
windows, task/split counts and Qwen3/Qwen2.5 row projections were all exactly
equal. There were no mismatches.

- Normalized projection SHA256:
  `736547b97bd7d175ea2324a08ec864e4d10e8aac1cd51a2b0029a46f367fe4d3`.
- Context projection SHA256:
  `40928be6c677a0c408bd13b79a2dde0e1a5bb0a033370a02025f56921c15121f`.

## Geometry co-occurrence

The frozen primary stop reason remains
`duplicate_or_overlap_receiver_offsets` for all 802 rows. Non-exclusive flags
show that all 802 rows also have source overlap and 10 rows have exact duplicate
source intervals. Therefore the result is not described as a receiver-only
offset problem.

| Primary reason | Secondary flag | Count |
|---|---|---:|
| duplicate_or_overlap_receiver_offsets | receiver_overlap | 802 |
| duplicate_or_overlap_receiver_offsets | source_overlap | 802 |
| duplicate_or_overlap_receiver_offsets | source_duplicate | 10 |

## Provenance

- Freeze attestation SHA256:
  `50510c1e7c6222c0206cbd896863c3b130cf112efd0dcf56460f3f76eff924ba`.
- Pre-data lock SHA256:
  `29dcd9652d58f32ebf3f23a4cab1cfc4d98aeff7ea2fde35a0f5cff34491157b`.
- Replay comparison SHA256:
  `d25832c21259bf2becc9951a9f3200108dcbc87e929866ccd336efcad0219534`.

This is an import-provenance and alignment-forensic result only. Exact tokenizer
identity does not imply equal model KV spaces.
