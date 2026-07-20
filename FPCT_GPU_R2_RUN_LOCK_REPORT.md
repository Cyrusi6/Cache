# FPCT GPU R2 Run-Lock Report

Status: `PRE_OUTPUT_LOCKED_R2`

- Scientific SHA: `9f2ffcd9ff21e4575f8fe870167eb04a7c86edb5`.
- Branch/upstream at image build: `research/fpct-factorized-transport` / same SHA.
- Run UID: `fpct-r2-9f2ffcd9-v1`.
- Image: `docker.io/library/fpct-gpu-r2:9f2ffcd9@sha256:d04455bf67177792548c3add74214f23ce097a004131481624886631725817ef`.
- Image source-tree SHA256: `b3278fc7e950221177c7d575a4ecc4269cf0473dfa4b3b46147962a2182b65fe`.
- Image tar SHA256: `9167434a3bfa1fb4a078a20cf49ee72945d20b8228da4d3953ecea20c8558e93` (`3,520,229,376` bytes).
- Operative run-lock SHA256: `c4b0ca20bea54f2dbbb9eaabf5bdbb0dc5b74835a3284e5d846b46f1e6a2a331`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2-9f2ffcd9-v1`.
- Node pool: `4090-48gx2`; formal seed parallelism is one dual-GPU pod.

The image records Python 3.11.11, Torch 2.6.0+cu124, CUDA 12.4,
Transformers 4.52.4, datasets 4.0.0, accelerate 1.9.0 and SciPy 1.15.3.
Its embedded head, branch and upstream all match the scientific SHA.

The 2048-example training sidecar was copied into the new run root and verified
byte-identical at SHA256
`48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`.
No result, Job, checkpoint or mutable execution state is reused from the old
terminal run root.

All normative documents, execution files, the image-loader plus four execution
K8s templates, image tar, sidecar
and certified resource geometry were independently re-hashed against the lock.
The certified expansion gates remain below mean 1.35 and p95 1.50 for all
three tasks.

At lock time there had been no new R2 pretrained output, GPU execution,
training, checkpoint, accuracy or correctness access. The first authorized
execution is image import followed by the complete synthetic GPU numerical
sequence. Natural pretrained diagnostics remain blocked until that sequence is
GO.

The first gate Pod never started because kubelet resolved `repository@digest`
while the initial loader had created only a `tag@digest` alias. The pending Job
was deleted before container start. The loader and lock were prospectively
amended to create both aliases; this is an infrastructure-only pre-output
amendment, not a scientific retry.

## Terminal v1 execution outcome

The amended image alias allowed the complete GPU numerical sequence to run and
pass. The first natural label-free condition then failed before its first model
forward while hashing fresh projector state: a scalar gate tensor was passed to
`view(torch.uint8)`. The v1 controller is terminal
`GPU_ENGINEERING_BLOCKED_R2`; its run root will not be patched or resumed.
Tokenizers, weights and the frozen label-free panel had been loaded, but there
were zero condition output files, zero model forwards, zero accuracy reads,
zero training and zero checkpoints.

## R2b replacement lock

- Scientific SHA: `7ceae185512b100b4b7d7f6970710a4637c568b0`.
- Run UID: `fpct-r2b-7ceae185-v1`.
- Image: `docker.io/library/fpct-gpu-r2b:7ceae185@sha256:d035cb31abe71640258aeb9cf48b9c7b7d39ff71346f0ac44bf0a2c5408ff463`.
- Image source tree: `db84f6546539fd2eed3b2afcd0e1a9f3f84d94fba1fbeb6bd2d22835fa610d49`.
- Image tar SHA256: `29c13fbb5be88004df367c0454d7f09e748c57b4943e4f828929c6f1b43d3caa`.
- R2b run-lock SHA256: `99dcb8114d60f55604f69b4c721e8348fc6f2f14e4eea17976bfe8df49a3f913`.

R2b changes only scalar tensor hashing and its regression test. It does not
reuse v1 numerical outputs or condition directories and must restart from the
GPU numerical sequence.

R2b passed that numerical sequence, then stopped in the first C_post trace
forward because `packed` was undefined when no sidecar was present. Execution
entered the first receiver attention call but produced no complete model output
and no condition artifact or accuracy. R2b is terminal and non-resumable.

## R2c replacement lock

- Scientific SHA: `e1133549c8d5efda7c09b06632e55964d94cad4d`.
- Run UID: `fpct-r2c-e1133549-v1`.
- Image: `docker.io/library/fpct-gpu-r2c:e1133549@sha256:94437d56d3a496935eb6486b83cebbf556a39c468d06e820514315be2adf0550`.
- Image config/source-tree SHA256: `195e2225...` / `742b1458...`.
- Image tar SHA256: `aeda9aab8ea64df4f3b2f927bf403ec13085d0e98f168953763eba7b8e314e58` (`3,520,236,544` bytes).
- R2c run-lock SHA256: `3ea3c3ea99e4e7d3e7a814082aaa038bc264779b49ce2206cd64b74515ca9c61`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2c-e1133549-v1`.

R2c changes only the C_post trace-path initialization and adds an actual Qwen3
no-sidecar trace regression. The operator, prior, diagnostic panel, numerical
floors, training recipe and release gates are unchanged. R2 and R2b remain
terminal, immutable and non-resumable. R2c has a new image, run UID, run root,
sidecar copy and image-loader; no numerical or natural condition artifact is
reused. At this lock boundary R2c has produced no pretrained output, training,
checkpoint, accuracy or correctness result. It must restart from the complete
GPU numerical sequence.

## Terminal R2c execution outcome

R2c passed the complete synthetic GPU sequence and completed all 16 isolated
pretrained operator conditions plus P2--P6 profiles. Eager runtime, canonical
FP32 prior, finite/mask controls, fresh-gate `EXPECTED_NATIVE_NULL`, forced-on
activation and resource/latency gates passed. It nevertheless failed the hard
pre-collapse, bypass, replicated-atoms, m<=1 and hot-path synchronization
checks. The controller is terminal `GPU_ENGINEERING_BLOCKED_R2`; the run root
will not be patched or resumed. No matched smoke, training, checkpoint,
accuracy or correctness evaluation was started.

The label-free trace identifies two prospective repair targets. First, C_post
did not retain the same sidecar/layout allocation path as F, so BF16 controls
were identical at layer 0 but diverged after different numerical paths. Second,
all P2--P6 profiles contained 336 scalar H2D synchronizations from
`Projector._current_alignment_layer_scales` inside `fpct.project_candidates`.
Any repair requires a new scientific SHA, image, run-lock, run UID and complete
restart from the synthetic GPU gate.

## R2d replacement lock

- Scientific SHA: `71ba96d2cad1cbf6894cff4e4ad08ef5a915d0e6`.
- Run UID: `fpct-r2d-71ba96d-v1`.
- Image: `docker.io/library/fpct-gpu-r2d:71ba96d@sha256:04b7b6428bd3bb4f31bb4968f8bfff68c6fb09f477f402211f1631984c0ff6cb`.
- Image config/source-tree SHA256: `1cae6486...` / `4d1c8eb6...`.
- Image tar SHA256: `86282ff639d1fb50e8ebdd233ff30ae40e0e91a56b0f91db78aa407db1e2b717` (`3,520,287,232` bytes).
- R2d run-lock SHA256: `2e1c998ff7e61438db2808ed82c694bf0afbae164a576064e5860b09f3126a4e`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2d-71ba96d-v1`.

The R2d addendum, source files, tests, image provenance, tar and sidecar copy
have been independently hashed. R2d is a new execution and does not reuse R2c
conditions, profiles or numerical artifacts. Before this lock there was no R2d
pretrained forward, GPU execution, training, checkpoint, accuracy or
correctness result. The first authorized step is the complete synthetic GPU
gate.

## Terminal R2d execution outcome

R2d passed its complete synthetic GPU gate and completed all 16 conditions and
five profiles. FP32 replicated, bypass and m<=1 end-to-end deltas became exactly
zero, and the FP32 expanded local canary was `9.54e-6`. BF16 still failed
because the first section used different adapters before sidecars existed; its
controls accumulated to `0.9375`. The BF16 expanded output canary also included
one-ULP output quantization (`0.0625 > 0.02`). The scalar buffer fix reduced
hot-path syncs from 336 to 280, with the remaining events isolated to
`_current_alignment_residual_scales`.

R2d is terminal `GPU_ENGINEERING_BLOCKED_R2` and will not be patched or
resumed. No matched smoke, training, checkpoint, accuracy or correctness result
was produced.

## R2e replacement lock

- Scientific SHA: `26539300de50d89a1be5a7871b6e78d9b715f535`.
- Run UID: `fpct-r2e-2653930-v1`.
- Image: `docker.io/library/fpct-gpu-r2e:2653930@sha256:50b89faa7148a73e32caf7be62220fca874d5a5d084de09d33278eed2e1374ba`.
- Image config/source-tree SHA256: `d1fd8700...` / `63a3e128...`.
- Image tar SHA256: `f47b7f80c0251fa0dfb81cd1594878e390b172eb2ea3bd071a6ac7eb7d7d2fdc` (`3,520,293,376` bytes).
- R2e run-lock SHA256: `e4d4392f4b3039541ab940acb16e4091a428af6da0f0eb2f6fa40d18c2b1cc5b`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2e-2653930-v1`.

R2e is fully isolated from R2d numerical, condition and profile artifacts. Its
source, addendum, tests, image provenance, tar and sidecar copy are frozen by
hash. Before this lock there was no R2e pretrained/GPU output, training,
checkpoint, accuracy or correctness result.

The first R2e v1 loader render was rejected by Kubernetes client-side decoding
before a Job existed because the pure-numeric short SHA was parsed as a numeric
label. No Pod, container, GPU or scientific output existed. All R2 execution
templates now quote `git_sha`; the abandoned v1 controller/config map is not
used. The operative infrastructure-only replacement is:

- Run UID: `fpct-r2e-2653930-v2`.
- Run-lock SHA256: `05c100a74e8737317f78977177b18ccfae9aed17c20444ac22c49fa250f6746e`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2e-2653930-v2`.
- ConfigMap: `fpct-r2e-v2-lock-2653930`.

The scientific image and all scientific thresholds are unchanged.

## Terminal R2e-v2 execution outcome

R2e-v2 passed the complete GPU gate and all exact structural controls:
pre-collapse identity, bypass, replicated, grouped-probability canary, m<=1 and
hot-path no-sync. P2--P5 each recorded zero scientific/hot synchronization.
It still failed because flat attention retained expanded kernel width even when
only one atom per parent was active, yielding native-null deltas of `4.12e-5`
(FP32) and `0.625` (BF16). In addition, inactive collapsed atoms remained in
the diagnostic prior mask and inflated D_K/D_V synthetic floors, making the
forced-on check non-identifying.

The controller is terminal `GPU_ENGINEERING_BLOCKED_R2`; no matched smoke,
training, checkpoint, accuracy or correctness evaluation ran.

## R2f hierarchical replacement lock

- Scientific SHA: `d08b22b339698ad81c0a3651891185294d8307e6`.
- Run UID: `fpct-r2f-d08b22b-v1`.
- Image: `docker.io/library/fpct-gpu-r2f:d08b22b@sha256:cb91ec54576d55885891d0f4dc07d81b8876866bfe196c088f67c3a7dee1ede9`.
- Image config/source-tree SHA256: `af92705d...` / `df0246f1...`.
- Image tar SHA256: `8cb0983f8c9d4fda21e1a4bdb071c6b07ae2f880cfe77fd5fdde93a3f3c88760` (`3,520,301,056` bytes).
- R2f run-lock SHA256: `1990589f1c3eb07e08b56b1bf9e0c90e16c6ac0c92cbe5d067ce95f2294d683a`.
- Run root: `/netdisk/lijunsi/fpct-confirmatory/fpct-r2f-d08b22b-v1`.
- ConfigMap: `fpct-r2f-lock-d08b22b`.

R2f is a new immutable execution. It does not patch or resume R2e-v2, and it
does not reuse R2e numerical, condition, profile or result artifacts. The
operative change is the prospectively frozen global-equivalent hierarchical
beta/gamma attention adapter with an exact parent-equivalent branch and
active-only diagnostics. Source, tests, image, tar, certified sidecar and all
normative inputs are bound by the run-lock.

Before this lock there was no R2f GPU numerical output, pretrained output,
training, checkpoint, accuracy or correctness result. The first authorized
operation is the complete synthetic GPU numerical gate. Pretrained conditions
remain blocked until that gate is recorded as GO; matched smoke and formal
training remain conditionally blocked behind the subsequent R2 pretrained GO.

## Terminal R2f execution outcome

R2f passed the complete synthetic GPU gate and completed all 16 isolated
pretrained conditions plus P2--P6 profiles. It passed eager runtime,
canonical FP32 prior, finite/mask checks, forced-on activation, pre-collapse
identity, collapse bypass, replicated-atoms, m<=1, hot-path no-sync and all
latency/HBM/expansion gates.

The sole failure was the frozen checkpoint-native numerical-null check. FP32
`Delta_fact` was `4.291534423828125e-5`, exceeding the pre-output
`4.0e-5` floor by `2.91534423828125e-6`; BF16 was exactly zero. Offline
tensor-only review found all fused candidate K/V and collapsed K/V exactly
equal to native K/V in all 504 FP32 panel-layer records, and reconstructed all
records as parent-equivalent. C_post, replicated-atoms and bypass logits were
exactly equal. The residual difference therefore occurs only after entering
the hierarchical execution order; the first above-`2e-5` pre-o-proj,
post-o-proj and residual differences occurred at layers 23, 26 and 22.

The threshold is unchanged and the run will not be patched or resumed. The
controller is terminal `GPU_ENGINEERING_BLOCKED_R2`. No matched smoke,
training, checkpoint, accuracy or correctness evaluation ran. Any parent-first
call-order test requires a new prospective scientific SHA, image, run-lock,
run UID and complete restart from the synthetic GPU gate.
