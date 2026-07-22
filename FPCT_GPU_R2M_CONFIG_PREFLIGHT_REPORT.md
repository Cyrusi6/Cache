# FPCT-GPU-R2m Config Preflight Report

Date: 2026-07-22

## Scope

This report records the operative output-free `CONFIG_PREFLIGHT_ONLY` execution
for run UID `fpct-r2m-80fb295-v1`. The preflight did not load a model or dataset, did not use
CUDA, did not produce a task metric, and did not authorize training.

## Frozen identity

- Protocol commit: `66e4cdd197370edfb988bf5ae98d6987a9e43993`
- Clean execution commit/upstream: `80fb295542ad298fae4cddb1273517b401bbcd17`
- Image: `docker.io/library/fpct-gpu-r2m:80fb295@sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851`
- Image source tree SHA256: `b2090b58362312384c330865b499080750eaf380eb071a2e44917e4ca28f6b51`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-80fb295-v1`
- Run-lock raw SHA256: `db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff`
- Canonical parsed SHA256: `f731660f73bb4e8daa9163cce533648c972758d5cbc18e93bbafdf040a7cc072`
- Geometry projection SHA256: `221c5164c60ec4d27abe714a68cec2f6a1a630d031195e0f0e63818133a5c6a2`

## Exact-byte closure

The Git lock file, immutable ConfigMap data, init-container mounted file, and
main-container startup copy all had raw SHA256
`db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff`.
The Kubernetes ConfigMap was created with `immutable=true` from the mechanically
generated JSON object; no lock field was copied by hand into a Job template.

## Result

- Schema: `GO`
- Consumer closure: `GO`
- Task set: `ai2-arc`, `mmlu-redux`, `openbookqa`
- Geometry rows: `3`
- Classification: `CONFIG_PREFLIGHT_GO`
- `scientific_output=false`
- `training_authorized=false`

The detailed result is stored locally at
`/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-80fb295-v1/preflight/config_preflight_result.json`
with SHA256 `bc89fb484d73220a1862fecb15ef58cd23497b26500bcabde5dee83cf5b1689f`.
The sealed bootstrap attestation has SHA256
`2cc2c2b45c40fc26745ed0e154f3388430f58be8758e1902a1be3cac594a54f8`.

An earlier output-free candidate preflight used run UID
`fpct-r2m-66e4cdd-v1`, image digest `sha256:a374ca4c...`, and lock SHA
`3e85670c...`. It was superseded before any scientific output because its image
embedded the protocol HEAD rather than the subsequently pushed clean execution
commit. It cannot authorize a GPU gate or training and is retained only as
pre-science provenance history.

This GO permits submission of the separately frozen immutable GPU gate. It is
not scientific evidence and does not by itself authorize matched smoke or
formal confirmatory training.
