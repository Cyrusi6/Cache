# Phase 2A-2b Prompt Identity Protocol Recommendations

## Required changes before any new GPU evaluation

1. Mark every paper-facing config with `eval.formal_experiment: true` and enable the prompt identity contract. Formal configs without the contract must fail during evaluator construction.
2. Dynamic templates must receive an explicit immutable `date_string` in `DD Mon YYYY` form. A template that uses the system clock but lacks an explicit override is prohibited.
3. Run a CPU-only freeze pass after dataset scope, content-group split, prompt format, sender/receiver tokenizer paths and generation/logits mode are final. The local manifest must contain canonical messages plus sender/receiver rendered prompts and input IDs. Because it contains raw prompt material, keep it out of Git; commit only its SHA, row count and compact provenance.
4. Bind the manifest path, SHA256 and expected row count in verify mode. Normal evaluator startup must regenerate every row and compare scope, sample keys, exact frozen material, hashes, tokenizer assets/revision and template SHA before CUDA visibility, CUDA queries, worker spawn, checkpoint load or inference.
5. Recheck the compact identity record immediately after runtime input preparation and before each model call. Save per-example canonical-message SHA, rendered-prompt SHA, input-ID SHA, token count, template SHA and tokenizer revision beside the normal results.
6. Treat any mismatch as a run failure. Do not continue a partial task, substitute a new manifest, or regenerate the freeze after looking at results.

## Minimal configuration pattern

```yaml
eval:
  formal_experiment: true

prompt_identity:
  enabled: true
  formal_experiment: true
  mode: verify
  date_string: 17 Jul 2026
  timezone: UTC
  locale: C
  expected_rows: <frozen-row-count>
  manifest: local/prompt_identity/<run-id>.json
  manifest_sha256: <sha256>
```

Freeze on CPU before the config is changed to verify mode:

```bash
python script/evaluation/unified_evaluator.py \
  --config <freeze-config.yaml> \
  --freeze-prompt-identity-manifest local/prompt_identity/<run-id>.json
```

The normal evaluator automatically performs the full verify preflight. A CPU-only release check is also available:

```bash
python script/evaluation/unified_evaluator.py \
  --config <verify-config.yaml> \
  --verify-prompt-identity-only
```

## Phase 2A-2b-specific preregistration changes

- Preserve the original Phase 2A-2a STOPPED/NO_GO record and cite the separate status addendum.
- Create a new preregistration and a new frozen prompt manifest; do not reuse or overwrite the historical Gate-1 reference.
- First rerun only OFF/OFF and OFF/ON exact equivalence on the previously authorized fit scope. Do not join outcomes or test geometry predictivity until exact identity and exact generation both pass.
- Bind all arms to one prompt-manifest SHA and one code commit. The runner should refuse arms with a different `date_string`, prompt scope, template/tokenizer fingerprint or expected sample set.
- Keep Phase 2A-1 sealed outcomes inaccessible during freeze, equivalence and manifest review. Any later geometry predictivity confirmation requires separate authorization and a still-untouched test benchmark.
- Do not infer geometry failure from the historical Gate-1 mismatch and do not infer geometry success from current OFF/ON equivalence. The causal question remains untested.

Two-stage evaluator paths are intentionally unsupported by the new protocol and fail closed until their multi-prompt identity can be specified and tested.
