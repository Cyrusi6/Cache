# FPCT GPU R2 Run-Lock Report

Status: `PRE_OUTPUT_LOCKED_R2`

- Scientific SHA: `9f2ffcd9ff21e4575f8fe870167eb04a7c86edb5`.
- Branch/upstream at image build: `research/fpct-factorized-transport` / same SHA.
- Run UID: `fpct-r2-9f2ffcd9-v1`.
- Image: `docker.io/library/fpct-gpu-r2:9f2ffcd9@sha256:d04455bf67177792548c3add74214f23ce097a004131481624886631725817ef`.
- Image source-tree SHA256: `b3278fc7e950221177c7d575a4ecc4269cf0473dfa4b3b46147962a2182b65fe`.
- Image tar SHA256: `9167434a3bfa1fb4a078a20cf49ee72945d20b8228da4d3953ecea20c8558e93` (`3,520,229,376` bytes).
- Run-lock SHA256: `b9d371ccc4e1fac18d60b1f674e3dbe3cb930bd0f99179d7ec6beb0ebb2564c2`.
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

All normative documents, execution files, K8s templates, image tar, sidecar
and certified resource geometry were independently re-hashed against the lock.
The certified expansion gates remain below mean 1.35 and p95 1.50 for all
three tasks.

At lock time there had been no new R2 pretrained output, GPU execution,
training, checkpoint, accuracy or correctness access. The first authorized
execution is image import followed by the complete synthetic GPU numerical
sequence. Natural pretrained diagnostics remain blocked until that sequence is
GO.
