# FPCT confirmatory pre-output run lock

Operative scientific code is `0a549c95e9a8c90d6402d58b7eccead0b9d039cd`. The immutable image is `docker.io/library/fpct-confirmatory:0a549c95@sha256:8c2441dd459c7d845e693f5b084ec40c241c2c2c60201b92a37329d87fe374ef` and contains no mounted host source, Conda environment, site-packages, or other worktree. The tag is included only to match the imported containerd reference; the digest remains authoritative.

The single 2,048-example `certified_slot0_v1` training sidecar has SHA256 `48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`. It was generated under sealed CPU execution at commit `4703ca0...`; the only tracked changes through the operative SHA are K8s templates, status documents, and their manifest test. Diagnostic 128 and all formal arms read the same frozen prefix/cache.

The shared K8s root is `/netdisk/lijunsi/fpct-confirmatory/fpct-cfm-0a549c95-20260720`. Model assets are read-only from the existing shared identifiability model cache; the run root is exclusive to this `run_uid`. There were no Running pods in `c2c-research` at lock inspection. Pre-output storage probes showed that `/netdisk` is available only on `4090-48gx2`, while `/home/lijunsi` does not expose the frozen assets on any worker. To avoid post-lock asset replication or divergent local copies, the operative hardware pool is therefore one 48GB two-GPU worker and maximum seed-pod parallelism is one.

No pretrained model forward, GPU kernel, training, checkpoint evaluation, model-selection accuracy, or held-out result occurred before this lock. The next authorized action is the sealed container probe followed by the synthetic GPU numerical gate.

The final no-model container probe completed with stable closure fingerprint `fd5aa9eec0e002c7ca0f94b76245df1c6e305b9736bfba08c2a9887070013799`. The node-distribution tar SHA256 is `5357a8a9e91f0a238a3fead7c1877abe261c22b5fad1f1e3552ef974cf673620` (3,520,156,672 bytes). Earlier candidate images stopped before model loading on missing evaluator dependencies and a Python symlink; neither image was used for GPU execution.
