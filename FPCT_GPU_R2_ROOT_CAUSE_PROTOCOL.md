# FPCT GPU R2 Root-Cause Protocol

Status: `PROSPECTIVE — NO NEW PRETRAINED OUTPUT`

This protocol governs a new execution revision. The prior scientific image
`371e72f...`, image digest `c851056...`, run-lock SHA256 `2a4db8...`, Jobs and
artifact root remain immutable and terminal `GPU_ENGINEERING_BLOCKED`.

## Frozen hypotheses

- `H1_GATE_NATIVE_NULL`: a fresh projector has no checkpoint and initializes
  every key/value scalar gate logit to zero. Checkpoint-native eval applies
  `(logit > 0)`, so both gates are exactly zero and candidate-fused KV reduces
  to receiver-native KV. Zero activation in this state is
  `EXPECTED_NATIVE_NULL`, not an engineering failure.
- `H2_BF16_PRIOR_DRIFT`: casting `A`, `log A` or the packed additive mask to
  BF16 before normalization/global attention breaks exact refinement mass and
  may accumulate across 28 receiver layers.
- `H3_BACKEND_AUTODISPATCH`: the old smoke configuration did not explicitly
  set `attn_implementation`. It therefore has no runtime proof of eager and,
  under the current Transformers dispatch rules, may have selected SDPA.
- `H4_PROFILER_SCOPE_CONFOUND`: the old profiler rejected any synchronize in
  the entire forward/profiler lifecycle, mixing harness, layout, diagnostics,
  legacy code and profiler teardown with the FPCT per-layer hot path.

No hypothesis may be added after new pretrained output and then used to resume
formal science. An unregistered cause may be debugged, but any repair requires
a new scientific commit, image and run-lock from the GPU gate.

## Zero-output provenance

Before tokenizer use, natural-prompt access or model forward, record and hash:

- projector checkpoint directory and fresh/loaded classification;
- every mapped layer's key/value scalar gate logits;
- legacy scalar and alignment-confidence eval modes;
- receiver/sender `_attn_implementation`, attention classes and backends;
- source, canonical prior, log-prior, packed-mask and KV dtypes;
- Torch/Transformers/CUDA versions, scientific SHA, image digest and sealed
  module closure.

R2 loads sender and receiver with `attn_implementation="eager"`. After load,
on every rank and after every reload, both configs and every attention layer
must attest eager or execution stops before forward.

## Operator conditions

Each condition runs in an independent process with a newly constructed wrapper,
identical model hashes, frozen step-0 projector state/hash, panel, sidecar, RNG
and eager backend. Mutating operator flags on a previously-forwarded wrapper is
forbidden.

| ID | Operator/path | Gate mode | Purpose |
|---|---|---|---|
| `OP01_CPOST_NATIVE` | C_post | checkpoint-native | parent-collapse baseline |
| `OP02_F_NATIVE` | F real candidates | checkpoint-native | native scientific condition |
| `OP03_F_REP_NATIVE` | F replicated atoms | checkpoint-native | refinement control |
| `OP04_F_FORCED` | F real candidates | forced-on diagnostic | engineering canary |
| `OP05_F_REP_FORCED` | F replicated atoms | forced-on diagnostic | active-gate refinement control |
| `OP06_F_BYPASS` | sidecar validated, attention collapses to parent | matched | exact routing/cache control |
| `OP07_M1_CPOST` | C_post, all eligible `m<=1` | matched | exact one-slot control |
| `OP08_M1_F` | F, all eligible `m<=1` | matched | exact one-slot control |

Forced-on uses only the existing parameter-free legacy scalar and alignment
confidence interventions. It cannot change checkpoint tensors, enter training
or accuracy cells, or support a performance claim.

## Delta definitions

- `Delta_fact = F_real_candidates - F_replicated_atoms`, compared on the same
  packed/global-attention path.
- `Delta_rep = F_replicated_atoms - C_post`, a numerical refinement control.
- `Delta_bypass = F_collapse_to_parent - C_post`, an exact wiring/cache control.

`F-C_post` is not an activation metric in R2. `Delta_fact` alone measures
candidate-specific query-time activation. Bypass and `m<=1` must recover
C_post within frozen dtype/depth tolerances.

## Label-free diagnostic panel

The panel is selected only from the sealed TinyLlama fit/calibration geometry
artifacts. For each task and certified cardinality `m=2,3,4`, sort distinct
content-group SHA256 ascending and take the first two groups. Within a selected
group choose the lexicographically first `(sample_sha,parent_index)` row of the
requested cardinality. This yields 18 fixed rows without labels, correctness or
model output. IDs and source hashes are frozen in
`recipe/eval_recipe/fpct_gpu_r2/diagnostic_panel_manifest.json`.

## First-divergence decision tree

For FP32 and BF16, capture layer-indexed aggregates/hashes for source/native and
fused KV, gates, prior invariants, packed maps, parent mass, candidate logits,
gamma/KL/Jensen/query variance, pre/post projection output, residual and final
logits. Full KV is never tracked in Git.

1. Bypass first differs at layer 0: wiring/state/cache bug.
2. Replicated FP32 first differs: packing/mask/global-denominator bug.
3. FP32 passes and BF16 grows with depth: precision bug.
4. Forced-on `D_K/D_V` remain null: gather/fuser/wiring bug.
5. Candidate internals activate but logits remain small: weak propagation, not
   operator non-execution.

The metric sink is layer-indexed or uses online sum/max/count; last-layer
overwrite is forbidden.

## Progression

R2 pretrained GO requires eager attestation, FP32 prior invariants,
`EXPECTED_NATIVE_NULL` consistency, forced-on canary activation, identical
C_post/F pre-collapse candidates, bypass/replicated/`m<=1` controls, finite
masked output, scoped no-host-sync and existing expansion/HBM/latency gates.
Failure is terminal `GPU_ENGINEERING_BLOCKED_R2`; training is forbidden.
