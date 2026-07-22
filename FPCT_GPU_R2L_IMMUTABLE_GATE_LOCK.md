# FPCT GPU R2l Immutable Gate Lock

## Frozen identity

- Classification: `IMMUTABLE_CONFIRMATORY_GATE`
- Operator repair commit: `d71d21b1e315787e9af1cefb324abd310fd335f7`
- Immutable checker/image commit and upstream: `43b825b34204326029590da7b9d51b67d7916208`
- Image: `docker.io/library/fpct-gpu-r2l:43b825b@sha256:e805c714f4a77be82fe89e36a100750ba25ad815b5af004d6f9ae4233f37492e`
- Embedded source tree: `2381e0aa14d25ac7d72a964d03a2a784f5b95e66eaad0b0ad0a0d7fd241af5ca`
- Run UID: `fpct-r2l-43b825b-v1`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2l-43b825b-v1`
- Image tar SHA256: `37855f1c209fd519f8c7942501d822e1dcd27c0ec9f7d6149c51170577f02c62`

The image is distinct from the focused diagnostic image. The only code added
after the qualified operator repair is immutable-check execution and testing;
the production attention blob, flat atom kernel, projector, wrapper, prior,
mask, data, thresholds, training, and statistics remain frozen.

## One-shot sequence

1. Import the exact image on `4090-48gx2` and validate the digest.
2. Run the complete synthetic GPU numerical sequence.
3. Run all 16 pretrained operator conditions and the original P2--P6 profiles.
4. Require the original gate to report exactly 23/23 checks GO.
5. Run fresh R2l synthetic mixed-memory checks and the actual Qwen3 eager
   28-layer FP32/BF16 prefill-plus-decode4 checker.
6. Recompute direct bitwise `C_post_native = F_native = F_replicated_native`
   evidence from the immutable condition artifacts and require all six R2l
   semantic checks.
7. Only then run the retained eight-block balanced checkpoint-native and
   forced-on canary.
8. A sealed finalizer writes `training_authorized=true` only if every preceding
   component passes.

The original compatibility resource limits remain median ratio at most 1.50,
p95 ratio at most 1.75, mean expansion at most 1.35, p95 expansion at most
1.50, peak HBM strictly below 90%, and hot-path sync count zero. The balanced
canary retains 20 warmups plus 50 measurements per arm in eight blocks and
requires both canaries' balanced median at most 1.35 and one-sided 95% block
bootstrap UCB at most 1.50.

## Failure and conditional training boundary

Once immutable scientific output exists, the revision is one-shot. Any
correctness, semantic, active-route, resource, no-sync, or provenance failure
is terminal `GPU_ENGINEERING_BLOCKED_R2L`; it cannot be fixed or retried as
R2l. Only a preregistered infrastructure failure before scientific output may
be classified `INFRASTRUCTURE_INCONCLUSIVE_R2L`.

Matched smoke uses seed 104729, 128 examples, three matched arms, and four
optimizer steps. It is mechanically blocked unless the immutable final result
is GO. Formal seeds 45--56 are mechanically blocked unless matched smoke is
GO; all original arm ordering, 2048-example membership, 64-step recipe,
checkpoint rule, retry rule, and T/O/I/N statistics remain unchanged.

At this lock point no R2l immutable Job has been submitted, no new immutable
pretrained output exists, no optimizer step or checkpoint has been created,
and no accuracy/correctness, model-selection, or held-out output has been read.
R2k remains permanently `GPU_ENGINEERING_BLOCKED_R2K` and is not retried or
reinterpreted.
