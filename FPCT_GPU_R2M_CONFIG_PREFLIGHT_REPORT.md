# FPCT-GPU-R2m Config Preflight Report

Date: 2026-07-22

## Scope

This report records the output-free `CONFIG_PREFLIGHT_ONLY` execution for run UID
`fpct-r2m-66e4cdd-v1`. The preflight did not load a model or dataset, did not use
CUDA, did not produce a task metric, and did not authorize training.

## Frozen identity

- Protocol commit: `66e4cdd197370edfb988bf5ae98d6987a9e43993`
- Image: `docker.io/library/fpct-gpu-r2m:66e4cdd@sha256:a374ca4c24bd25730d71b45346f5d9f64e2507f85e2bc4a7e86ffa520abfba1a`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-66e4cdd-v1`
- Run-lock raw SHA256: `3e85670cda5e3895373b23a62b38b201e9191c259c1b7981ab8547e0280d1be9`
- Canonical parsed SHA256: `3665d462d72b2aa586fcad4d1fb01bcc2dde5f00cab79340841ae58fc1bc9ca1`
- Geometry projection SHA256: `221c5164c60ec4d27abe714a68cec2f6a1a630d031195e0f0e63818133a5c6a2`

## Exact-byte closure

The Git lock file, immutable ConfigMap data, init-container mounted file, and
main-container startup copy all had raw SHA256
`3e85670cda5e3895373b23a62b38b201e9191c259c1b7981ab8547e0280d1be9`.
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
`/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-66e4cdd-v1/preflight/config_preflight_result.json`
with SHA256 `d8e75a0bcf985f17b4e5c4ec760264fc6b3636b26a7c1b3dc94eb2b5fdea29c9`.
The sealed bootstrap attestation has SHA256
`723008ac8b7393b89a796232aa586707256077c10128a3ac5c579fe78e1d6fda`.

This GO permits submission of the separately frozen immutable GPU gate. It is
not scientific evidence and does not by itself authorize matched smoke or
formal confirmatory training.
