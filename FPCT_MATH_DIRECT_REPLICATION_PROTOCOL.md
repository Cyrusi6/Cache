# FPCT-MATH-DIRECT-R2 replication and mechanism protocol

## Status and scope

This protocol is frozen before any seed `2026082102` or `2026082103` pretrained
forward, gradient, training, or evaluation output. It prospectively extends the
completed single-seed math-direct experiment without changing its implementation,
data, budget, metric, or decision threshold.

The existing seed `2026082101`, its checkpoints, twelve evaluation cells, and result
SHA256 remain immutable. The new work remains exploratory and does not access
E1-pilot, model-selection, test, or confirmatory data.

## Exact replication

- Pair: TinyLlama-1.1B sender to Qwen3-0.6B receiver.
- Operators: C_post and F with `fpct_position_mode=math`.
- Seeds: `2026082102`, `2026082103`.
- Within each seed: fresh C_post then fresh F, serially on the same two-GPU node.
- Across seeds: execution may be parallel on separate nodes.
- Each arm: 2,048 MMLU auxiliary-train examples, one epoch, 64 optimizer steps,
  per-device batch 1, gradient accumulation 16, effective global batch 32, LR
  `1e-4`, weight decay `0.01`, linear schedule, warmup `0.10`, dropout `0.1`, BF16,
  eager attention, and frozen sender/receiver.
- Each seed uses its own seed for projector initialization and distributed data order;
  its C_post/F arms must match exactly on initialization, membership/order, pre-train
  RNG, optimizer, scheduler, and hardware.
- The frozen certified training sidecar and the 326-group E0-design development set
  are reused by exact SHA. No old projector checkpoint initializes a new arm.

## Step-0 gradient diagnostic

Before interpreting training results, run one no-update diagnostic on the first
rank-0 training microbatch selected by the frozen seed-`2026082102` distributed
sampler. Construct one fresh projector state, reset identical CPU/CUDA/Python/NumPy
RNG before C_post and F, and compute both backward passes without an optimizer step.

Report exact gradient norm and cosine for every receiver layer and for K path, V path,
and parent nuisance/gate/confidence parameters. Also report loss, fixed batch hash,
parameter keys, nonfinite count, missing-gradient count, and global gradient cosine.
The diagnostic is explanatory only and cannot select a seed or alter training.

## Teacher-forced diagnostic

For every seed, checkpoint arm, and task, evaluate the same frozen E0-design groups
under C_post and F inference with the complete gold response teacher-forced. Primary
diagnostic value is mean answer-token `log p(y*)` per distinct content group, task
equal within the three-task macro. It produces the same four cells as accuracy:
`Y_CC`, `Y_CF`, `Y_FC`, and `Y_FF`, and reports log-probability analogues of
`T`, `D_C`, `D_F`, `O`, and `I`.

Teacher-forced results are diagnostic and do not replace the frozen accuracy gate.
No raw KV tensors are saved.

## Frozen accuracy decisions

System replication passes if and only if all conditions hold across the three math
seeds:

1. mean `T >= +1.00 pp`;
2. at least two of three seeds have `T > 0`;
3. no task has mean `T < -2.00 pp`;
4. every matched-integrity and evaluation provenance check is GO.

Query-time mechanism continuation additionally requires mean `O > 0` and at least
two of three seeds with `O > 0`.

The mechanical terminal classification is:

- `QUERY_TIME_FPCT_CONTINUE` when both system and mechanism gates pass;
- `FACTORIZED_TRAINING_COLLAPSED_INFERENCE` when the system gate passes and the
  mechanism gate fails;
- `STOP_CURRENT_OPERATOR` when the system gate fails.

No threshold may be changed after output. If the second classification is reached,
the allowed claim is training adaptation/regularization and deployment uses C_post;
it is not a query-time factorization claim. If the third is reached, current F stops
and later native-null or partition-aware work requires a separate prospective lock.

## Output and firewall

Large checkpoints, JSONL rows, predictions, and logs remain under
`/netdisk/lijunsi/fpct-math-direct-r2/`. Git receives only protocol, code, tests,
configurations, manifests, compact aggregates, and artifact SHA256 records.

