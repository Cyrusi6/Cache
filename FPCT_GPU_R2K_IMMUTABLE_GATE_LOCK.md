# FPCT GPU R2k Immutable Gate Lock

## Frozen identity

- Classification: `IMMUTABLE_CONFIRMATORY_GATE`
- Scientific commit/upstream: `458b0260fc5475c9ae578eb68b8dff2b2699e2f4`
- Image: `docker.io/library/fpct-gpu-r2k:458b026@sha256:bc19b894c18eea266596011b748a4a22c73b80788e0fdd4a08b5f33059bf51ca`
- Embedded source tree: `50861304a574b29c86c13d9d1006338a5b948f849717ac396ac883ac1c97a034`
- Run UID: `fpct-r2k-458b026-v1`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2k-458b026-v1`
- Run-lock SHA256: `d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`
- Image tar SHA256: `333424aabef63e4001b7dd3a41915939ee94040dd7f519a7c084f19ca0d934ac`

The image was built from a detached archive of the frozen scientific commit,
not from the later documentation/lock commits. It is distinct from the
diagnostic image while embedding the same scientific source tree.

## Gate sequence

1. Import the exact image to the target node and verify sealed runtime/backend
   provenance.
2. Run the complete synthetic GPU numerical gate.
3. Run all 16 pretrained operator conditions and the original P2--P6 profiles.
4. Require all original 23 checks, including the unchanged 1+7 compatibility
   resource gate.
5. Only after the original gate is GO, run the separately frozen eight-block
   checkpoint-native/forced-on balanced canary.
6. Authorize seed-104729 matched smoke only if both gate components pass.

The original compatibility limits remain ratio-of-medians at most 1.50,
max/legacy-p95 ratio at most 1.75, mean expansion at most 1.35, p95 expansion
at most 1.50, and peak HBM strictly below 90%. The balanced canary requires
both checkpoint-native and forced-on CUDA median ratios at most 1.35 and their
one-sided 95% block-bootstrap UCBs at most 1.50.

## One-shot and firewall rules

Only a preregistered infrastructure condition occurring before the formal
ratio is read may produce an infrastructure-only replacement. After the
formal ratio is read, this scientific revision has no retry. Any numerical,
operator, resource, or active-canary failure produces
`GPU_ENGINEERING_BLOCKED_R2K`; the next optimization would require a distinct
R2l protocol and scientific revision.

No accuracy/correctness, training, checkpoint, model-selection, or held-out
output existed when this lock was created. R2j remains permanently
`GPU_ENGINEERING_BLOCKED_R2` and is neither retried nor reinterpreted.
