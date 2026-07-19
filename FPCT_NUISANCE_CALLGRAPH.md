# FPCT Frozen Nuisance Callgraph

## Decision

The current `C_pre` path has an unambiguous implementation seam: source candidates are reduced in `RosettaModel._weighted_source_kv_from_indices` before the shared `C2CProjector.forward`, and the resulting one-slot-per-parent cache is consumed by the receiver attention through `DynamicCache`. `C_post/F` can therefore share one candidate-specific projection helper and differ only at the final pre-attention collapse/packing step.

FPCT-2 may proceed only with the boundaries below. Route3 learned routing, position rewrites, native sibling/null, selector and new trainable gates remain rejected.

## Existing C_pre sequence

1. `rosetta/train/dataset_adapters.py`
   - materializes `source_indices`, raw `source_weights`, `source_confidence`, `source_entropy`, and override flags;
   - masks invalid indices and L1-renormalizes weights;
   - computes entropy/confidence once per receiver parent.
2. `rosetta/model/wrapper.py:622-630`
   - activates the soft-alignment path only when a section carries `soft_alignment`.
3. `rosetta/model/wrapper.py:846-910`
   - reads one parent-level alignment record;
   - the frozen first-round config has weight calibration disabled; `c_post/f` reject a non-`none` calibration mode so raw `A` remains the prior;
   - `C_pre` calls `_weighted_source_kv_from_indices`, which gathers `[B,Hs,N,K,Ds]` and reduces `K` before projection.
4. `rosetta/model/wrapper.py:912-944`
   - calls the shared projector once per parent slot and passes the already-computed parent confidence/entropy/weights;
   - projectors without internal confidence use the existing external residual-confidence scaling.
5. `rosetta/model/projector.py:4456-4640` (`C2CProjector.forward`)
   - candidate/source and receiver-native features enter the common MLP/fuser core;
   - `key_scalar/value_scalar` are part of the nonlinear fuser core;
   - legacy scalar Gumbel/hard gates are sampled/applied at parent/head granularity;
   - alignment confidence/entropy gates are computed from parent-level alignment features and projector hidden state;
   - global residual scales are applied once;
   - output is receiver-native KV plus a gated projected residual.
6. `rosetta/model/wrapper.py:951-985`
   - writes the projected single slot back into the receiver `DynamicCache`;
   - no extra token position is created.
7. Receiver attention
   - consumes the ordinary cache with its existing RoPE-space key, value, causal/padding mask, attention implementation and global softmax denominator.

## Frozen nuisance ownership

| Object | Existing owner | FPCT-2/3 rule |
|---|---|---|
| alignment `A` | aligner/dataset adapter | frozen prior; mask then L1 normalize; `F` adds `log A` once |
| entropy/confidence | adapter + projector | compute once per parent; broadcast unchanged to candidates |
| legacy scalar Gumbel/hard gate | `C2CProjector.forward` | sample once per parent/head; reuse for every candidate |
| residual scale | projector parameter path | unchanged and shared |
| nonlinear projected feature/scalar | projector fuser core | candidate-specific in `C_post/F` |
| position/RoPE | source and receiver cache-space KV | `legacy`; no de-RoPE/re-RoPE and no child position IDs |
| causal/padding bias | receiver attention mask | every child inherits the parent column exactly |
| native fallback | receiver cache | retain only for `m=0`; no new native sibling/null |
| ordinary generated/template native slots | receiver cache | remain in the same global denominator |

## Authorized implementation seam

- `c_pre` and an unset flag execute the existing gather-average-project-write path byte-for-byte.
- `c_post` and `f` call one shared helper that:
  1. gathers legal candidate KV;
  2. computes/captures parent nuisance once from the same `C_pre` averaged source;
  3. projects candidates with the captured confidence and legacy gate broadcast;
  4. collapses candidates with `A` into the ordinary cache placeholder.
- `c_post` stops there.
- `f` additionally stores a non-parameter candidate sidecar. At attention time only parents with `m>=2` replace their placeholder by legal child atoms; `m=1` remains one slot and `m=0` remains native fallback.
- Packed child atoms inherit the parent attention-mask column and receive `log A`; they do not enter `DynamicCache` as new sequence positions.

## Stop conditions

Implementation must stop if any of the following is required:

- enabling `learned_alignment_mode`, injection gate, transfer gate, or Route3 router;
- changing position IDs/RoPE or extending `DynamicCache` positions for children;
- recomputing entropy/confidence/legacy random gates per candidate;
- adding an F-only parameter or value-side multiplication by `A` after attention softmax;
- changing the default/unset state dict, output, cache, or config behavior.
