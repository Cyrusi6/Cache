# FPCT GPU R2k Diagnostic Report

## Decision

The sealed R2k v2 diagnostic is `DIAGNOSTIC_QUALIFIED`. This is an
engineering freeze qualification only. It cannot produce R2k GO, alter the
terminal R2j result, authorize training, or release task correctness.

The immutable confirmatory gate must use a new image, run UID, root, and run
lock. It must retain the original 23-check compatibility gate and add the
balanced checkpoint-native and forced-on canaries. Once the formal immutable
ratio is read, the same scientific revision cannot be changed and retried.

## Frozen execution

- Scientific code: `458b0260fc5475c9ae578eb68b8dff2b2699e2f4`
- Image: `sha256:3a4240bc26610b43737861f46cbee88de5d1552002d03de9f0f9407d71da32e5`
- Run UID: `fpct-r2k-diag-458b026-v2`
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2k-diag-458b026-v2`
- Run-lock SHA256: `3a283885bdb1ff2639f13d3cbe3875b2ab673c47d28a88741144b7dfafcb56f6`
- Aggregate SHA256: `d1c8dd03b4312f3231db2191a92091612927303219dfe3ba54512c3d31913066`
- Independent verified summary SHA256: `8935481c428ed5ec262d1b5876f9a6894c73838d597d1be7a0896a2a32c1afa1`

V1 remains an infrastructure-only terminal caused by a missing attestation
parent. Its unsealed block was not opened for ratios and was not reused. V2's
first trace attempt stopped before model setup because an empty trace directory
had been precreated. The same-UID resume skipped every completed sealed latency
block and geometry artifact and produced only the missing trace and aggregate.

## Balanced diagnostic results

Each canary used eight fresh processes, the frozen ABBA ordering, 20 warmups,
50 CUDA-event measurements and 50 synchronized-wall measurements per arm. The
independent verifier recomputed all medians and the 50,000-replicate one-sided
block-bootstrap UCB directly from raw samples.

| Canary | CUDA median F/C_post | one-sided 95% UCB | wall median | Qualification |
| --- | ---: | ---: | ---: | --- |
| checkpoint-native | 1.058329 | 1.077335 | 1.058631 | pass |
| forced-on | 1.074972 | 1.077734 | 1.074968 | pass |

Both satisfy the prospective diagnostic thresholds: balanced median at most
1.35 and one-sided 95% UCB at most 1.50. The forced-on block-0 ratio of about
0.69 is retained rather than excluded; the raw C_post absolute time is higher
in that block, and the prospective telemetry contains no registered
infrastructure condition that permits removal.

## Trace and resource evidence

- GPU: NVIDIA GeForce RTX 4090,
  `GPU-e1fc63b0-33de-dee9-c690-530e9123b970`; all samples P0.
- SM clock range: 2520--2745 MHz; memory clock: 10501 MHz.
- Temperature: 34--53 C; power: 65.84--217.73 W.
- No active throttle reason and no foreign compute process.
- Peak HBM: 4.8666 GiB.
- Mean and p95 expanded-slot ratio: 1.22689.
- No host synchronization event occurred inside the frozen FPCT hot scopes.

The trace scientific-scope ratio is 1.09406 and the summed-kernel ratio is
1.04985. The FPCT attention scope itself is 0.08770 s for C_post and 0.14219 s
for F; F also spends 0.04420 s in packing. The remaining differential is a
collection of small eager where/scatter/gather/copy kernels, not a changed
operator or an omitted prior.

The pure geometry panel reports much larger micro-kernel ratios (about
3.29--3.79). It is descriptive shape stress evidence only: it is not the
pretrained freeze endpoint and cannot override either canary qualification or
the forthcoming immutable compatibility gate.

## Verification and claim boundary

The independent verifier checked the 8 x 2 x 2 x 50 raw-sample contract,
sealed attestations, process seeds, arm order, medians, bootstrap UCB,
telemetry, trace hashes, lack of hot-path synchronization, provenance, and the
accuracy firewall. No task accuracy/correctness, training loss, checkpoint,
model-selection, or held-out result was read or generated.

R2j remains `GPU_ENGINEERING_BLOCKED_R2` at ratio 1.6885509. This diagnostic
does not rescue or reinterpret it. The only permitted next scientific action
is a distinct, one-shot immutable R2k gate built from the frozen scientific
commit.
