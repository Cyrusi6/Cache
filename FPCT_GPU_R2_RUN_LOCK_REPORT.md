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
