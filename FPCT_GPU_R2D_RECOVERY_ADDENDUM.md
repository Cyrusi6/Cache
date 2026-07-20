# FPCT GPU R2d Prospective Recovery Addendum

Status: `PROSPECTIVE_PRE_OUTPUT`

This addendum is frozen after the terminal label-free R2c gate and before any
R2d pretrained forward. It does not modify or reinterpret the immutable R2,
R2b or R2c executions. It is limited to the already preregistered H2 numerical
refinement and H4 profiling hypotheses plus the preregistered requirement that
C_post and F share one eager adapter.

## Frozen repair

1. C_post and F both retain the candidate sidecars and build the same packed
   layout. C_post uses the exact collapsed-parent branch of that shared adapter;
   F alone selects expanded candidates when certified extra slots exist.
2. A batch with no certified extra slot uses the exact parent branch for both
   C_post and F. Therefore the natural m<=1 control does not allocate a
   different attention shape.
3. Replicated-atoms remains an expanded identical-atom calculation with the
   frozen log prior. Each layer records the expanded-versus-parent numerical
   delta. The returned diagnostic output is the analytic parent result so that
   the 28-layer control tests the local refinement error without feeding a
   numerical-null perturbation back through later layers. This remains a
   parameter-free diagnostic, not a fourth arm.
4. The expanded replicated calculation must independently satisfy the existing
   FP32 `2e-5` and BF16 `2e-2` per-layer tolerances. The end-to-end replicated
   output, bypass and m<=1 outputs must be exactly the shared parent path.
5. Static alignment layer scales are non-persistent module buffers. This removes
   per-call CPU scalar construction under `fpct.project_candidates` without
   adding state-dict keys or trainable parameters.

## Unchanged boundaries

- Canonical A/logA/mask and attention accumulation remain FP32.
- Operator, panel, model/data hashes, seeds, training recipe, accuracy firewall,
  resource ceilings and all numerical tolerances are unchanged.
- Forced-on remains label-free engineering evidence only.
- R2d requires a new scientific SHA, immutable image, run-lock, run UID and
  complete restart from the synthetic GPU gate.
- Any failed R2d hard check terminates that run; it cannot be patched or resumed.

At freeze time no R2d model forward, GPU execution, training, checkpoint,
accuracy or correctness output existed.
