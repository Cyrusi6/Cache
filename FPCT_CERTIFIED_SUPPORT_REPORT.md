# FPCT Certified Support Report

## Decision

FPCT-3.7 is `INCONCLUSIVE` because the first natural-audit invocation failed an execution-provenance gate before producing any alignment row or shard artifact.

No certified-support, readiness, expansion, memory or FLOP result is available from this execution revision. FPCT-3.8, FPCT-3.9, Hugging Face model forward, GPU, Kubernetes, training and accuracy evaluation were not entered.

## Frozen execution

| Item | Value |
|---|---|
| Corrected execution SHA | `b11a046597b2466c1c6ba95c4d3693e76523c3b3` |
| Branch | `research/fpct-factorized-transport` |
| Local/upstream before freeze | identical at `b11a046...` |
| FPCT-3.7 pre-audit lock SHA256 | `311ddf36bc0ab598ec52eae5236ad14f007a4645373200d58a301c9fcfd9cdb5` |
| Frozen targeted tests | `102 passed, 2 warnings` |
| Published shards | `0 / 12` |
| Natural alignment rows | `0` |

## Failure

TinyLlama/ARC and Llama3.2/ARC independently failed on their first call with:

`TypeError: TokenAligner.align_chat_messages_soft() got an unexpected keyword argument 'apply_confidence_control'`

The frozen research source is:

- path: `/home/lijunsi/projects/Cache-fpct-factorized-transport/rosetta/model/aligner.py`;
- SHA256: `fe77d72fb7103ca103fe87fa602bed545e0caf6fc75b1741b8736b41e9daf7d8`;
- its method signature includes `apply_confidence_control`.

The Conda environment contains an editable `rosetta` mapping to `/home/lijunsi/projects/KVcache/C2C/rosetta`. Under script-mode `sys.path`, that mapping resolves `rosetta.model.aligner` to:

- `/home/lijunsi/projects/KVcache/C2C/rosetta/model/aligner.py`;
- SHA256 `1d68fe69aecd03382fb9bf1385ee110f5f218902c0f5566c3827e0d8afa0922a`;
- a method signature without `apply_confidence_control`.

Thus the natural command did not execute the production source whose SHA was frozen. This is an import-path provenance mismatch, not a scientific support result.

Qwen2.5 did not launch because its delegated write request was rejected before execution. Qwen3 was not launched after the hard stop. Neither case produced an artifact.

## Stop-rule application

Adding `PYTHONPATH`, changing the invocation mode, or patching the audit to prepend the repository root would be a post-freeze execution-contract correction. The frozen rule requires this execution revision to remain `INCONCLUSIVE`; no in-place patch or rerun is allowed.

A future attempt requires a new prospective execution revision that freezes and tests the exact imported module path and SHA before any natural rendering/tokenization/alignment. This report does not authorize that amendment or the next attempt.

## Local evidence

Root:

`/home/lijunsi/projects/Cache-fpct-factorized-transport/local/final_results/fpct_factorized_transport/fpct_3_7_certified_support/rev_b11a046597b2466c1c6ba95c4d3693e76523c3b3/`

- `failure.json`: 2,374 bytes; SHA256 `051a425b83ff38d401323d8708da8032723adbe0a3fcb72c72ecb48f4706af2b`.
- `controller_state.json`: 267 bytes; SHA256 `aa46ba058f5ad623723cfa5cb866ede74ba2cf923f650a529066991a9aebc617`.
- `pre_audit_lock.json`: SHA256 `311ddf36bc0ab598ec52eae5236ad14f007a4645373200d58a301c9fcfd9cdb5`.
- No `shards/` or `.incomplete/` directory exists.

## Claim boundary

This failure says nothing about whether heterogeneous tokenizer factorization support exists, whether `F` differs from `C_post` in a real model, or whether task accuracy would improve. It only establishes that execution provenance was not sufficiently sealed in revision `b11a046...`.
