# Phase 2A-0: Calibrated Null/No-Transfer Opportunity Audit

> 状态：完成。基于 Cache `main` commit `29a96947fc5d5e0a8f457f75edf7f2e745932bce`，仅复用冻结的 receiver-only 与 B6-native 逐例结果完成数据审计和 CPU 统计。未启动 GPU/Kubernetes、未训练任何 adapter/router、未修改 checkpoint，也未进入 instrumentation rerun 或 selector 训练。

## 1. 结论摘要

Phase 2A-0 得出三个直接结论：

1. 当前确实存在较大的 no-transfer 机会。pair-balanced sample-weighted 的 oracle 相对真实 best fixed policy 的 headroom 为 `+8.2439 pp`，best-fixed-aware hierarchical paired bootstrap 95% CI 为 `[+6.2318, +10.2019] pp`。
2. Phase 1.5 报告的 `+8.24 pp` 点估计不只是相对 fused 的空间。B6-native 在全部 `36/36` 个 pair × seed × task 单元中都优于 receiver-only，因此 retrospective `max(receiver,fused)` 在每个单元和总体上都是 fused；故该点估计也严格等于 oracle-over-best-fixed。
3. 这仍然只是 label-aware oracle 上界。当前产物没有部署时可用的 receiver/fused uncertainty、margin 或 calibrated score；现有 pre-transfer alignment 字段又高度冗余，因此本审计不能声称已有 selector 能实现该收益。

本阶段只确认“机会存在”，不评价后续方法是否达到 GO 条件。后续比较与放行规则已冻结在 `PHASE2A_PREREGISTRATION.md`，本阶段到此停止。

## 2. 数据来源与完整性

统计复用 Phase 1 的冻结产物：

- Phase 1 artifact commit：`9b06d173eada148343ddfb71a31721c7ae5f7ad5`。
- Phase 1 analysis manifest SHA256：`3d2fe098d1db27442cdb6f99690912634bc2f983115ea34dddfbe20313130bab`。
- 三任务 receiver input-set fingerprint：`0366be7e5b129710024543bc065774bef165b6b6bca92541a3d641aea2918114`。
- receiver-only：3 个任务文件，共 7,265 行。
- B6-native：4 pairs × 3 seeds × 3 tasks = 36 个文件，共 87,180 行。
- 总输入文件：39；全部具有相同的 45 列 schema。

严格审计通过：

- ARC、OpenBookQA、MMLU-Redux 每个单元分别为 1,150、500、5,615 行。
- 所有文件均使用 `(task, subject, question_id)` 做 join；未使用行号。
- 所有 pair/seed 的输入文本、选项、标签和 sample-key 集合均与 receiver-only 完全一致。
- MMLU 在不同运行中的行顺序不稳定，因此任何 row-index split/join 都不合法。
- 7,265 条 benchmark rows 对应 7,233 个唯一 normalized-content groups；MMLU 内有 32 个重复内容组、共 64 行，跨任务重复为 0。后续 split 必须把同内容行绑定在同一 split。

87,180 是重复观测数，不是独立样本数：receiver-only 只运行一次并跨 4 pairs × 3 seeds 复用，实际唯一 benchmark rows 为 7,265。

## 3. 四类事件与恒等式

对每个 canonical sample，令 `R` 表示 receiver 是否正确，`F` 表示 B6-native fused 是否正确：

| 事件 | 定义 | utility |
|---|---|---:|
| Receiver correct, Fused correct | `R=1, F=1` | 0 |
| Beneficial transfer | `R=0, F=1` | +1 |
| Harmful transfer | `R=1, F=0` | -1 |
| Receiver wrong, Fused wrong | `R=0, F=0` | 0 |

由此有：

- `receiver accuracy = both-correct + harmful`；
- `fused accuracy = both-correct + beneficial`；
- `oracle accuracy = 1 - both-wrong`；
- `oracle - fused = harmful rate`；
- `oracle - receiver = beneficial rate`；
- `oracle - max(receiver,fused) = min(beneficial, harmful)`；
- `mean transfer utility = beneficial - harmful = fused - receiver`。

## 4. Pair-balanced 总体结果

主表同时报告 task-macro 和 sample-weighted。task-macro 对三个任务等权；sample-weighted 在每个 pair/seed 内按 7,265 个样本合并，然后对 seed、pair 等权。

| 指标 | Task-macro | 95% CI | Sample-weighted | 95% CI |
|---|---:|---:|---:|---:|
| Receiver correct, Fused correct | 29.7332% | [27.1638, 32.2966] | 27.9158% | [25.8293, 30.0826] |
| Beneficial transfer | 20.4199% | [16.1364, 23.5139] | 19.0158% | [14.8255, 21.8400] |
| Harmful transfer | 8.5352% | [6.3438, 10.7561] | 8.2439% | [6.2318, 10.2019] |
| Receiver wrong, Fused wrong | 41.3117% | [38.0703, 45.6683] | 44.8245% | [41.8984, 49.0711] |
| Receiver accuracy | 38.2684% | [36.5082, 40.0749] | 36.1597% | [35.0585, 37.2746] |
| Fused accuracy | 50.1531% | [46.9583, 53.3265] | 46.9316% | [44.1535, 49.4666] |
| Oracle accuracy | 58.6883% | [54.3317, 61.9297] | 55.1755% | [50.9289, 58.1016] |
| Oracle over fused | +8.5352 pp | [+6.3438, +10.7561] | +8.2439 pp | [+6.2318, +10.2019] |
| Oracle over receiver | +20.4199 pp | [+16.1364, +23.5139] | +19.0158 pp | [+14.8255, +21.8400] |
| Oracle over best fixed | **+8.5352 pp** | **[+6.3438, +10.7561]** | **+8.2439 pp** | **[+6.2318, +10.2019]** |
| Fused over receiver | +11.8847 pp | [+8.6204, +15.1380] | +10.7720 pp | [+7.9640, +13.3231] |

四事件比例在每个输出行均严格加和为 1，accuracy/headroom 恒等式也由脚本逐行验证。

## 5. Pair、seed 与 task 分解

### 5.1 Pair 分解

以下为 across-seed sample-weighted 结果；receiver-only 在所有 pair 中相同，因为同一份 baseline 被复用。

| Pair | Receiver | Fused | Oracle | Oracle over best fixed |
|---|---:|---:|---:|---:|
| TinyLlama-1.1B → Qwen3-0.6B | 36.1597% | 47.1347% | 56.8892% | +9.7545 pp |
| Qwen3-1.7B → Qwen3-0.6B | 36.1597% | 50.2730% | 57.6141% | +7.3411 pp |
| Qwen2.5-0.5B → Qwen3-0.6B | 36.1597% | 43.6247% | 49.2957% | +5.6710 pp |
| Llama3.2-1B → Qwen3-0.6B | 36.1597% | 46.6942% | 56.9030% | +10.2088 pp |

三个预注册异构 pair 的 sample-weighted aggregate headroom 为 `+8.5448 pp`，95% CI `[+5.8116, +10.7012] pp`。Qwen3-1.7B 同 tokenizer pair 作为 control，不计入异构 pair 数判定。

### 5.2 Seed 分解

以下为四 pair 等权、sample-weighted 结果：

| Seed | Receiver | Fused | Oracle | Oracle over best fixed |
|---:|---:|---:|---:|---:|
| 42 | 36.1597% | 47.9279% | 56.3589% | +8.4310 pp |
| 43 | 36.1597% | 47.3017% | 55.2100% | +7.9083 pp |
| 44 | 36.1597% | 45.5637% | 53.9567% | +8.3930 pp |

Qwen2.5 seed 44 的 fused accuracy 崩塌降低了该 seed 的总体 fused/oracle accuracy，但并未消除 oracle no-transfer 空间。

### 5.3 Task 分解

以下为 pair、seed 等权的 task-specific 结果：

| Task | Receiver | Fused | Oracle | Oracle over best fixed |
|---|---:|---:|---:|---:|
| ARC | 40.1739% | 54.9783% | 62.9203% | +7.9420 pp |
| OpenBookQA | 39.6000% | 50.5167% | 59.9833% | +9.4667 pp |
| MMLU-Redux | 35.0312% | 44.9635% | 53.1606% | +8.1971 pp |

全部 36 个 pair × seed × task 单元及其逐项 CI 位于 `PHASE2A_0_OPPORTUNITY_AGGREGATES.csv`；JSON 同时保留来源 SHA、schema、字段覆盖和 feature redundancy 审计。

## 6. `+8.24 pp` 是否真的是相对 best fixed

Phase 1.5 的 across-pair `oracle_abstention.csv` 行技术上只填充了 `oracle_headroom_over_fused`；aggregate `oracle_headroom_over_best_fixed` 为空。因此原报告的 `[+6.28,+10.19] pp` 区间形式上是 oracle-minus-fused 的区间。

正式复现命令会自动核验 Phase 1 suite manifest、Phase 1.5 execution manifest 和旧 oracle CSV 的 SHA256，并解析唯一的 `b6_native/__all__/across_pairs` 行；若旧点估计与本轮 best-fixed point 不一致则直接失败。该结论不依赖手工抄表。

本轮重新检查后：

- fused 在 `36/36` 个 pair × seed × task 单元中严格优于 receiver；最小差值仍为 `+2.2796 pp`；
- fused 在 `12/12` 个 pair × seed pooled 单元中也严格优于 receiver；
- pair-balanced sample-weighted fused 为 46.9316%，receiver 为 36.1597%。

因此 retrospective best fixed policy 在所有当前审计层级都是 fused，`oracle-over-fused` 与 `oracle-over-max(receiver,fused)` 的点估计完全相同。新脚本进一步在每个 bootstrap draw 内重新计算非线性的 `max(receiver,fused)`；得到真正 best-fixed-aware CI `[+6.2318,+10.2019] pp`，与旧 fused-only CI 接近但统计定义更严格。

这不等于后续 selector 可以只与 fused 比较。正式主比较必须是 calibration split 上选出的 global best fixed policy，随后冻结并应用到 model-selection/test。

## 7. Selector 字段审计

### A. 真正 pre-transfer、部署时可用

- `cot_input_length`：可在 receiver tokenizer 后、模型 forward 前得到。
- `alignment_bucket`、`candidate_count`、`candidate_count_max`、`one_to_many_rate`、`alignment_entropy`、`boundary_mismatch`、`confidence`、`fallback_rate`：由 input preparation 中的 tokenizer/alignment 结果生成，早于 fused forward。
- `question` 与 `A`–`D` 以及 task/subject/pair metadata 在部署时可知，但 raw text 和 identity-like metadata 不进入 confirmatory primary selector；否则会转化为 benchmark memorization 或 task/pair base-rate router。

现有 A 类字段的有效维度很低：

- 全部 87,180 行满足 `alignment_entropy == one_to_many_rate`，最大误差 `2.78e-17`；
- `confidence == 1 - 0.5 × alignment_entropy`，最大误差 `1.11e-16`；
- `fallback_rate` 全部为 0；
- TinyLlama 的 7,265 个样本全部为 one-to-many，1,240 个样本 boundary mismatch 非零；
- Qwen3-1.7B 与 Qwen2.5 各只有 104/7,265 个 one-to-many；Llama3.2 为 97/7,265；三者 boundary mismatch 全为 0。

因此当前 A 类特征强烈 pair-coded，不能把列数误读为独立预测信息量。

### B. 需要 fused forward 后才能得到

- B6 `pred`、`cot_pred`、`cot_gen_length`、`cot_output`；
- compact gate 字段，包括 `gate`、K/V gate mean/std/saturation、record/token counts；
- fused latency 和生成后 extraction 字段。

这些字段适合诊断，但用于 selector 就已经支付 transfer/fused forward 成本，不能支持真正 no-transfer 决策。`answer_latency_ms` 还受到节点、并发和运行顺序混杂，禁止作为 confirmatory feature。

### C. Receiver-only dual-pass 上界

receiver CSV 中的 `pred`、生成长度、输出文本、latency 和 extraction 字段均需要完整 receiver-only forward。它们只能用于 dual-pass 上界；当前产物没有 receiver logits、option margin、entropy、sequence log-prob 或 calibration score。

当前 schema 还有明确的覆盖缺口：B6 的 `gate_token_count` 和四个 gate saturation 字段在 87,180 行中全部为空，`E`–`J` 与三个 extraction 字段也全部为空；dataset-level 和 posthoc gate JSON 是聚合统计，不能按 dev sample join。后续若需要这些信息，必须单独预注册 instrumentation rerun，不能在本阶段补跑。

### D. 标签或泄漏字段，禁止使用

- `true_answer`、`is_correct`、`ground_truth_normalized`；
- joined receiver/fused correctness、四事件标签、utility、oracle choice；
- `question_id`、row index、filename/timestamp、checkpoint/run id、seed/method 作为 selector feature；
- 任何由 test label 选择出的特征、阈值或模型。

`question_id` 只允许 join/split；seed/method 只允许 provenance/stratification。

## 8. Bootstrap 定义

正式统计使用 10,000 draws、95% percentile CI、seed `20260719`：

1. 7,265 个 canonical row-key samples 在 task 内成对重采样；receiver/fused 不独立抽样；
2. 同一个 sample draw 同步应用于所有 pair/seed，保留复用 receiver 和样本难度的相关性；
3. aggregate 行重采样 pair，再在选中 pair 内重采样 seed；
4. task-macro 固定三个任务并等权，sample-weighted 固定原任务样本权重；
5. 每个 draw 内先聚合 receiver/fused/oracle，再计算 `oracle - max(receiver,fused)`。

该实现比把 87,180 行视为独立样本更保守，也避免用 cell-wise hindsight policy 冒充一个 global fixed policy。

本轮机会 CI 延续 Phase 1 的 benchmark-row estimand，没有把 32 个重复内容组作为 bootstrap cluster；这些重复只占 64/7,265 rows。后续 selector split 和 confirmatory bootstrap 则按预注册 content group 绑定/聚类，以杜绝同题跨 split 泄漏。

## 9. 可复现命令与交付物

```bash
/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python \
  script/analysis/phase2a_0_opportunity_audit.py \
  --manifest recipe/eval_recipe/phase2a_0/opportunity_audit_manifest.json \
  --output-csv PHASE2A_0_OPPORTUNITY_AGGREGATES.csv \
  --output-json PHASE2A_0_OPPORTUNITY_AGGREGATES.json
```

交付物：

- `PHASE2A_0_OPPORTUNITY_AUDIT.md`
- `PHASE2A_PREREGISTRATION.md`
- `PHASE2A_0_OPPORTUNITY_AGGREGATES.csv`
- `PHASE2A_0_OPPORTUNITY_AGGREGATES.json`
- `recipe/eval_recipe/phase2a_0/opportunity_audit_manifest.json`
- `script/analysis/phase2a_0_opportunity_audit.py`
- `test/test_phase2a_0_opportunity_audit.py`

大体积逐例数据继续只保留在 `/netdisk/.../local/`，没有复制或提交到 Git。

## 10. 阶段判定

Opportunity audit 通过：存在稳定、非零的 oracle-over-best-fixed 空间，并且该空间出现在全部 pairs、seeds 和 tasks。

Method GO 尚未评估：没有 selector prediction、calibration、threshold、held-out test 或 selective-risk 结果。按任务要求，Phase 2A-0 完成后停止，等待审查；不得自动进入 instrumentation rerun 或 selector 训练。
