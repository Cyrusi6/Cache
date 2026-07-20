# FPCT Kubernetes execution protocol

The operative image is built from the committed scientific-code SHA and is addressed as `image@sha256`. No host Conda, site-packages, source worktree, or other worktree is mounted. Model weights, tokenizers, certified sidecars, frozen splits, checkpoints, and results use explicitly declared read-only/read-write storage contracts. An init container runs the sealed probe; every distributed rank reruns same-process attestation. The controller verifies the actual runtime image ID.

Every resource has labels `project=fpct`, `study=confirmatory`, `git_sha`, `seed`, `arm`, and a unique `run_uid`. The controller may inspect cluster-wide capacity but may delete or retry only resources with its exact `run_uid`. It never preempts another workload.

Weights and immutable assets are prefetched once and reused read-only. One formal seed uses one two-GPU pod, within which three independent arm subprocesses execute in the frozen balanced order. Parallelism is `min(7, currently idle and authorized two-GPU worker pairs)`. The same seed triplet uses the same node, GPU pair, image, and environment.

Checkpoint and result publication is atomic and SHA-addressed. A triplet receives at most one infrastructure-only whole-triplet retry. Eviction, preemption, transient network/storage failure, or node failure qualify. Numerical failure, OOM, scientific integrity failure, or an unfavorable result do not. An arm is never selectively rerun.

Performance evaluation remains sealed until all 36 formal step-64 checkpoints and matched-triplet integrity records exist. Model-selection is released once. Held-out is released at most once and only by the frozen futility rule. The controller persists an append-only event ledger and atomic state after every transition so restart is idempotent.

Any post-lock change to model, operator, alignment, training, evaluation, or statistical code invalidates the image/run lock and requires a new prospective revision beginning again at the GPU numerical gate.
