# Phase 2A-2b Prompt Identity Audit Report

## Scope and decision

This CPU-only audit started from `f1059dee343969661bb9492f0231d9bb58261706` on the isolated branch `research/phase2a2-prompt-identity-audit`. No GPU/Kubernetes job, model checkpoint, selector, correctness field, label, sealed outcome or prediction CSV content was used.

The root cause is confirmed: among the five formal model tokenizers, only Llama3.2 has a dynamic chat template. It calls `strftime_now` to inject `Today Date` and supports an explicit `date_string` override. Unpinned runs therefore change rendered prompts and token IDs across dates. The evaluator now has a fail-closed prompt identity protocol that freezes exact input material and verifies it before any GPU/model activity.

Historical decision: Phase 2A-2a remains **STOPPED/NO_GO under its preregistration**, but the earlier Gate-1 inference is superseded by historical input drift. Geometry predictivity remains untested.

## Formal tokenizer/template audit

| Tokenizer | Chat-template SHA256 | Tokenizer revision/fingerprint | Dynamic dependency |
|---|---|---|---|
| Llama-3.2-1B-Instruct | `5816fce10444...` | `local-assets-sha256:7d3b404aac3e...` | `system_clock` via `strftime_now`; fixed `date_string` required |
| Qwen2.5-0.5B-Instruct | `cd8e9439f057...` | `local-assets-sha256:db19fe18b051...` | none detected |
| Qwen3-0.6B | `a55ee1b16601...` | `local-assets-sha256:68bc2ba35de2...` | none detected |
| Qwen3-1.7B | `a55ee1b16601...` | `local-assets-sha256:68bc2ba35de2...` | none detected |
| TinyLlama-1.1B-Chat-v1.0 | `66291cf0045c...` | `local-assets-sha256:9e53cd419a47...` | none detected |

The scanner checks system clock/date functions, timezone access, locale access, process environment variables and randomness. A formal dynamic template is accepted only when the sole dependency is the system clock and the template exposes an explicit `date_string` override; timezone/locale/environment/random dependencies fail closed.

## Implemented fail-closed protocol

- One shared renderer now supplies the same explicit template variables to evaluator input preparation and `TokenAligner`, including both sender and receiver tokenizers.
- The CPU freeze pass stores canonical messages once and exact rendered prompt/input IDs per tokenizer role, together with their SHA256 values and token counts.
- Tokenizer provenance includes chat-template SHA, tokenizer class/vocabulary size, local asset-file hashes and either upstream revision or a deterministic local-assets revision.
- Verify mode requires the manifest path, exact manifest SHA and expected row count. It regenerates and compares the full scope and every sample before CUDA visibility setup, `torch.cuda` queries, worker spawn, checkpoint load or model forward.
- Runtime preparation checks each sample again before inference. Result sidecars store compact hashes and provenance, not duplicated raw prompt material.
- Formal two-stage paths and unsupported dynamic templates abort instead of falling back.

## CPU invariance verification

With `date_string="17 Jul 2026"`, Llama3.2 produced identical record SHA, rendered-prompt SHA, input-ID SHA and token count under:

| Ambient date | Timezone | Locale | Input-ID SHA prefix | Tokens |
|---|---|---|---|---:|
| 2026-07-17 | UTC | C | `a999a085e527...` | 39 |
| 2031-12-31 | Asia/Shanghai | zh_CN.utf8 | `a999a085e527...` | 39 |
| 2042-02-03 | America/New_York | en_US.utf8 | `a999a085e527...` | 39 |

An end-to-end two-row CPU freeze repeated under default and Asia/Shanghai/zh_CN environments produced the identical full manifest SHA `e3e23dc807d4eb41828f26f977158e5d01f7808bed207765950af11983f90884`. A third CPU-only verify under America/New_York/en_US completed with status `verified_before_gpu_start`; the CUDA visibility/query path was not reached.

## Historical audit result

The machine-readable audit contains 71 task/scope rows: 35 safe, 4 ambiguous and 32 invalid.

- Phase 1: 9 same-date within-seed/task Llama component contrasts are safe; the cross-seed Llama aggregate is ambiguous because seed 42 and seeds 43/44 span different dates without input hashes.
- Phase 1.5: 24 same-date Llama task/intervention contrasts are safe; 30 cross-date native/intervention contrasts are invalid. Those comparisons cannot identify top-k, entropy or gate effects because prompt identity changed simultaneously.
- Phase 2A-0 and Phase 2A-1: their CPU computations over immutable, SHA-bound artifacts are safe to reproduce. Llama cross-seed/generalization interpretations remain ambiguous because the inputs inherited the date/seed confound.
- Phase 2A-2a: the ARC and OpenBookQA frozen-reference Gate-1 comparisons are invalid because 17 July references were compared with 19 July runs and different input IDs are confirmed. The geometry predictivity stage never ran.

This audit does not recalculate any accuracy or selector metric. It changes only which historical comparisons have valid input identity.

## Reproduction

```bash
PYTHONPATH=. python script/analysis/phase2a_2b_prompt_identity_audit.py \
  --output-json recipe/eval_recipe/phase2a_2b_prompt_identity_audit/affected_experiments.json \
  --output-csv recipe/eval_recipe/phase2a_2b_prompt_identity_audit/affected_experiments.csv

python -m pytest -q --no-cov \
  --basetemp=local/tmp/pytest-prompt-identity \
  test/test_prompt_identity.py \
  test/test_unified_evaluator_prompt_identity.py \
  test/test_phase2a_2b_prompt_identity_audit.py
```

See `PHASE2A_2B_AFFECTED_EXPERIMENTS.md` for interpretation, `PHASE2A_2B_PROTOCOL_RECOMMENDATIONS.md` for the successor protocol, and `PHASE2A_2A_STATUS_ADDENDUM_PROMPT_IDENTITY.md` for the non-overwriting status statement.
