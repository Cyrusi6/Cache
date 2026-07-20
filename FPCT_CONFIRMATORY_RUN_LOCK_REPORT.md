# FPCT confirmatory pre-output run lock

Operative scientific code is `7e1aafbecf7561ea5dc416ebcdf8ea0fac76996b`. The immutable image is `fpct-confirmatory@sha256:37b40279bd48cff05abe323002837712f27abc1184914d2ab296f4e55c597dba` and contains no mounted host source, Conda environment, site-packages, or other worktree.

The single 2,048-example `certified_slot0_v1` training sidecar has SHA256 `48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`. It was generated under sealed CPU execution at commit `4703ca0...`; the only tracked changes through the operative SHA are K8s templates, status documents, and their manifest test. Diagnostic 128 and all formal arms read the same frozen prefix/cache.

The shared K8s root is `/netdisk/lijunsi/fpct-confirmatory/fpct-cfm-7e1aafbe-20260720`. Model assets are read-only from the existing shared identifiability model cache; the run root is exclusive to this `run_uid`. There were no Running pods in `c2c-research` at lock inspection. Eligible capacity was 4 + 8 + 2 NVIDIA GPUs across the three amd64 workers; actual parallelism remains bounded by live idle capacity and seven two-GPU seed pods.

No pretrained model forward, GPU kernel, training, checkpoint evaluation, model-selection accuracy, or held-out result occurred before this lock. The next authorized action is the sealed container probe followed by the synthetic GPU numerical gate.

The final no-model container probe completed with stable closure fingerprint `302264e57a7c9e77444fe1928fb14279e964059232aeede8536ec5d23dd4142e`. Earlier candidate images stopped before model loading on missing evaluator dependencies and a Python symlink; neither image was used for GPU execution.
