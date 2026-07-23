# FPCT-E0 TMPDIR closure recovery addendum

## Scope and observed failure

Seed `2026072201` attempt 3 completed all 64 C_post optimizer steps, wrote the formal integrity record, and then failed the sealed post-target attestation with `sealed module closure changed while the target was running`. F, evaluation, and the later seeds did not start. Attempt 3 remains quarantined and none of its checkpoints may be reused by the recovery run.

The source tree stayed byte-identical. The pre/post Rosetta module lists were also identical. The unstable field was the PyTorch distributed generated-module entry in `sys.path`: the runtime-write recovery had changed `TMPDIR` from `/tmp` to `/fpct-e0/tmp`, while the immutable bootstrap intentionally canonicalizes only `/tmp/tmp*` directories containing `_remote_module_non_scriptable.py` into a content SHA marker.

## Prospective operational repair

The recovery keeps all W&B, Hugging Face, XDG, and Torch extension caches outside `/opt/fpct`, under `/fpct-e0`. It changes only `TMPDIR` back to the container-local `/tmp`, restoring the exact bootstrap normalization contract. No model, tokenizer, alignment, operator, seed, initialization, data order, optimizer, scheduler, precision, checkpoint, evaluation, threshold, or claim field changes.

Before another GPU run, an exact-image CPU-only preflight must:

1. construct the same sealed GPU module closure and formal SFT target identity;
2. initialize and finish W&B in the frozen offline mode;
3. prove the pre/post stable fingerprint is identical;
4. prove any PyTorch remote-module path is represented by the canonical `/tmp/<torch-remote-module-sha256=...>` marker rather than a raw random directory;
5. prove the immutable image source-tree SHA remains unchanged; and
6. attest that no model forward, training, checkpoint, or accuracy operation occurred.

Only a `GO` preflight authorizes seed `2026072201` attempt 4. The complete seed restarts from step 0, then the same Pod serially runs seed `2026072202` attempt 2 and seed `2026072203` attempt 2. Accuracy remains unread until normal evaluation completion.
