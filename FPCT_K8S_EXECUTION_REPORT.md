# FPCT K8s Execution Report

Status: `TERMINAL — GPU_ENGINEERING_BLOCKED`

## Frozen execution identity

- Run UID: `fpct-cfm-371e72f1-20260720`.
- Namespace: `c2c-research`.
- Scientific code: `371e72f14da41f5509eafa21553c7a7418c9a53e`.
- Branch: `research/fpct-factorized-transport`.
- Run-lock SHA256:
  `2a4db8f26def997c95b590a34718916b772f686f5c00eabb2f2b69f0dfe5e5ec`.
- Image digest:
  `sha256:c851056733f3b7affc85ae5dabd870043f3ae7d3010d245705f5b9ded8dc36ab`.
- Frozen node pool: `4090-48gx2` only; maximum formal seed parallelism `1`.
- Shared run root:
  `/netdisk/lijunsi/fpct-confirmatory/fpct-cfm-371e72f1-20260720`.

The image was imported once by a run-UID-scoped loader and bound locally by
exact digest. Every scientific Job used a sealed init probe, the frozen
ConfigMap `fpct-lock-371e72f1`, read-only model storage and the same run root.

## Jobs

| Job | GPU | Node | Result | Retry |
|---|---:|---|---|---:|
| `fpct-load-371e72f1-48x2` | 0 | `4090-48gx2` | complete; exact image digest installed | 0 |
| `fpct-gpu-gate-371e72f1` | 1 | `4090-48gx2` | `GO` | 0 |
| `fpct-pretrained-smoke-371e72f1` | 1 | `4090-48gx2` | exit 1; `GPU_ENGINEERING_BLOCKED` | 0 |

The pretrained Job failure was a scientific/engineering gate failure, not pod
eviction, preemption, node, network or storage failure. The preregistered
infrastructure-only retry rule therefore did not apply.

No matched-smoke Job and no seed `45..56` triplet Job was submitted. No K8s
resource outside this run UID was deleted, retried or modified.

## Controller terminal state

The recoverable controller advanced sequentially:

`RUN_LOCKED -> GPU_NUMERICAL_GO -> TERMINAL(GPU_ENGINEERING_BLOCKED)`.

The controller recorded zero completed triplets, zero model-selection releases
and zero held-out releases. The failure evidence is the local-only pretrained
smoke result with SHA256
`7b7471aa62880521b0ac4b185af3c6ece1e4f21096f36c8bea9eb6e82e8be90b`.

## Reproduction and artifacts

The immutable submitted YAML and detailed artifacts are stored below the shared
run root in `jobs/`, `results/` and `state/`. Their absolute paths, sizes and
SHA256 values are frozen in
`recipe/eval_recipe/fpct_confirmatory/fpct_gpu_hard_stop_manifest.json`.
