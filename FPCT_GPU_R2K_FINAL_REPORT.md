# FPCT GPU R2k Final Report

## Terminal decision

R2k terminates as `GPU_ENGINEERING_BLOCKED_R2K`.

The equivalent flat-atom implementation recovered latency well beyond the
frozen resource requirement, but the one-shot immutable gate failed the
`expected_native_null` correctness invariant. The active canary, matched smoke,
training, model-selection, and held-out evaluation were therefore not run.

## Provenance

- Start HEAD required by the task: `4e06f0b970057ee374dd6ff9d02d1ca6cb133aa6`
- Frozen scientific code: `458b0260fc5475c9ae578eb68b8dff2b2699e2f4`
- Immutable image: `sha256:bc19b894c18eea266596011b748a4a22c73b80788e0fdd4a08b5f33059bf51ca`
- Embedded source tree: `50861304a574b29c86c13d9d1006338a5b948f849717ac396ac883ac1c97a034`
- Immutable run-lock: `d5d55ad310be50cf2a5dfb0a8cf6747a9dc98510fff668a542cf62b19efb1f62`
- Run UID/root: `fpct-r2k-458b026-v1` /
  `/netdisk/lijunsi/fpct-confirmatory/fpct-r2k-458b026-v1`
- Node/imageID: `4090-48gx2` /
  `docker.io/library/fpct-gpu-r2k:458b026@sha256:bc19b894...`
- Sealed fingerprint: `1fd1fbed17e6aa0ad6bc3d1f8f301465187dd2f0dd3d19c66b353b08ebaaeecc`

The image-loader and synthetic jobs completed. The pretrained Job exited 1
only after writing the complete aggregate and raising the frozen scientific
gate failure. It had zero restarts and no infrastructure retry.

## Historical timing audit and diagnostic qualification

The read-only R2h/R2i-v1/R2i-v2/R2j wall median ratios were
`1.0793/1.1540/1.0809/1.6886`; their trace summed-kernel ratios were
`1.0937/1.0966/1.0939/1.0954`. R2i and R2j had identical hot-path Git blobs.
The historical artifacts lacked sufficient prospective telemetry to reinterpret
R2j, which permanently remains `GPU_ENGINEERING_BLOCKED_R2`.

The R2k diagnostic used eight fresh-process ABBA blocks, 20 warmups and 50
measurements per arm with CUDA-event and synchronized-wall timing. The
checkpoint-native median/UCB were `1.058329/1.077335`; forced-on were
`1.074972/1.077734`. Both passed the diagnostic-only `1.35/1.50` qualification.
No hot-scope synchronization was present. Trace scientific and summed-kernel
ratios were `1.09406` and `1.04985`; F attention/packing scopes were 0.14219 s
and 0.04420 s. The pure shape panel's larger ratios remain descriptive only.

## Equivalent-kernel implementation and tests

The frozen science uses one parent eager call and reuses its FP32 parent logits;
non-equivalent atoms use one flat FP32 softmax and one probability-value
reduction. The prior grouped beta/gamma reductions and the five-dimensional
`[B,H,Q,S,D]` group-value allocation were removed. Reusable layout indices and
semantic maps are bound once, and production timing disables record-function
scopes while separate traces enable them.

The flat identity, FP32/BF16 forward and gradient checks, m=0--4,
padding/invalid/extreme-prior, GQA/MQA, mixed equivalence, dropout RNG, actual
Qwen3 eager, 28-layer controls, bypass, replicated, m<=1, no-new-parameter and
host-sync tests passed. FPCT targeted tests were `190 passed`; the CPU-safe full
suite was `429 passed` before freeze. The immutable synthetic GPU gate passed
all eight checks and was sealed at result SHA
`85641e2766f3a91caafeab6e60386f3e51b1790d29f80e7900443b76c402a57b`.

## Immutable 23-check result

All 16 operator conditions and all five P2--P6 profiles completed. Twenty-two
of 23 checks passed. The sole failure was `expected_native_null`.

Resource recovery itself passed:

| Endpoint | C_post | F | Ratio | Limit |
| --- | ---: | ---: | ---: | ---: |
| median wall seconds | 0.618157 | 0.439100 | 0.710337 | 1.50 |
| p95 wall seconds | 0.643173 | 0.454291 | 0.706328 | 1.75 |

Peak HBM was 4.2175 GiB; mean/p95 expansion passed the frozen 1.35/1.50
ceilings; eager runtime, prior, finite/mask, pre-collapse identity, bypass,
replicated, m=1, forced-on activation, and no-hot-sync checks all passed.

The native-null failure is decisive. All 28 key and value legacy gate logits
were exactly zero and C_post/F pre-collapse candidates were identical, yet real
F differed from replicated/C_post by `3.7909e-5` in FP32 and `0.625` in BF16,
exceeding the frozen `2e-5/2e-2` tolerances. Replicated, bypass and m=1 controls
remained exact.

## Read-only root-cause inference

The strongest code-path explanation is that the new all-parent-equivalent
fast-return predicate is false for mixed memory. The frozen semantic map starts
ordinary non-sidecar native parents at false and only fills sidecar ranges;
`parent_equivalent.all()` therefore prevents selection of the already-computed
exact parent output. The checkpoint-native path falls through to the flat-atom
reduction, where a mathematically equivalent but numerically different packed
call order creates a small FP32 discrepancy that compounds strongly in BF16.

This is an inference from the frozen code and sealed controls, not a repaired
or rerun result. No scientific file was modified after reading the formal
ratio. Any fix requires a separately preregistered R2l revision.

## Stopped stages and claim boundary

The immutable active-canary Job was not submitted because the original gate
failed first. Its diagnostic-only forced-on latency remains engineering
context, not a substitute for the immutable gate. Seed-104729 matched smoke,
the 12 x 3 formal runs, checkpoints, accuracy/correctness, model-selection,
held-out release, and T/O/I/N statistics are all absent.

R2k supports only this statement: an equivalent-kernel implementation greatly
reduced the measured latency ratio but did not preserve the exact
checkpoint-native null invariant under the immutable pretrained gate. It
provides no accuracy improvement, matched-training effect, or query-time
mechanism evidence.
