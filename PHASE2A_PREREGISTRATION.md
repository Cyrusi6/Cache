# Phase 2A Calibrated Null/No-Transfer Preregistration

> 本文件只冻结后续 selector 的数据、比较、统计和放行协议，不授权在 Phase 2A-0 中训练 selector、启动 GPU、补 instrumentation 或修改 B6 checkpoint。任何后续执行必须由新的明确命令启动。

## 1. 研究问题

给定部署时、transfer 前可获得的信息，selector 是否能对每个 sample 在以下两种 fixed actions 中做出可泛化选择：

- `a=0`：不 transfer，执行 receiver-only；
- `a=1`：执行 B6-native fused transfer。

令 `y_R,y_F∈{0,1}` 为 receiver/fused correctness，`u=y_F-y_R∈{-1,0,+1}`，则 selector 的逐例正确性为：

`y_selector = y_R + a × u`。

本阶段目标是 calibrated null/no-transfer，不开发新 transport、router、gate、loss、OT、RoPE correction 或 replay。

## 2. Primary estimand 与 aggregation

Primary estimand：held-out test 上

`Δ_primary = Accuracy(selector) - Accuracy(calibration-selected best fixed policy)`。

Primary aggregation 固定为 pair-balanced task-macro：

1. 每个 task 内按 sample 求 accuracy；
2. 三个 tasks 等权；
3. seeds 等权；
4. pairs 等权。

Sample-weighted pair-balanced 结果为 mandatory secondary，不得因为结果更有利而切换 primary。另报告三异构 pair aggregate 与 same-tokenizer Qwen3 control。

异构 pair 集合预注册为：

- TinyLlama-1.1B → Qwen3-0.6B；
- Qwen2.5-0.5B → Qwen3-0.6B；
- Llama3.2-1B → Qwen3-0.6B。

Qwen3-1.7B → Qwen3-0.6B 是 same-tokenizer control。考虑到审稿人可能质疑 Qwen2.5/Qwen3 是否属于“真正异构”，额外报告 TinyLlama + Llama3.2 strict cross-family sensitivity，但不得事后改变正式三异构集合。

## 3. Fixed-policy comparator

Primary comparator 是一个 global fixed policy `p*∈{always receiver, always fused}`，只在 calibration split 上选择：

1. 使用 primary pair-balanced task-macro 计算两种 fixed policy；
2. 选择 calibration accuracy 更高者；
3. 若完全相同，选择 receiver-only，以降低 transfer 成本和风险；
4. `p*` 在 model-selection/test 打开前冻结；
5. 禁止按 test task、pair、seed 或 sample 重新选择 fixed policy。

Test 上的 retrospective `max(receiver,fused)` 只作为诊断性的 oracle opportunity benchmark，不能用于选择 comparator 或阈值。

所有主表必须首先报告 selector vs `p*`，不能只报告 selector vs fused。

## 4. Canonical sample 与绑定规则

### 4.1 Join key

现有产物使用 `(task, subject, question_id)` 做完整性 join，并用 normalized question+choices SHA256 验证内容一致。禁止使用 row order。

### 4.2 Split group key

后续 split 使用 normalized content group，而不是 question id：

```text
content_hash = SHA256(normalized_question + normalized_choices_A_to_J)
group_hash   = SHA256("cache-phase2a-v1-29a96947" + dataset_content_sha256 + content_hash)
```

同一 content group 的所有行必须进入同一 split。当前 7,265 rows 中有 7,233 content groups，MMLU 内 32 个重复内容组必须绑定。所有 pair、seed、method、intervention 对同一 group 继承同一 split。

本轮冻结的 `dataset_content_sha256` 为 `0366be7e5b129710024543bc065774bef165b6b6bca92541a3d641aea2918114`。任何输入内容变化都必须产生新的 split version，禁止沿用旧 test。

Split key 禁止包含 label、prediction、correctness、event、pair、seed、method 或 checkpoint。

## 5. 四分区数据协议

为严格分离拟合、校准、选择和最终检验，使用四个不相交分区：

| Split | Hash interval | 用途 |
|---|---:|---|
| fit | `[0.00, 0.30)` | 拟合 selector 参数；不得选择最终阈值或报告 GO |
| calibration | `[0.30, 0.45)` | 校准 score、选择 action threshold、选择 global fixed comparator `p*` |
| model-selection | `[0.45, 0.60)` | 在完全预定义的 selector candidates 中选择一个；不得再调 threshold |
| test | `[0.60, 1.00)` | 最终一次性评测；不得选择字段、模型、阈值或 comparator |

`group_hash` 的前 64 bits 映射到 `[0,1)`。同一 salt、dataset content fingerprint、normalization code 和字段 whitelist 必须在 test 打开前写入 manifest 并提交。

按上述冻结 hash 对当前 7,265 rows 的预期分配为：

| Task | Fit | Calibration | Model-selection | Test |
|---|---:|---:|---:|---:|
| ARC | 351 | 160 | 186 | 453 |
| OpenBookQA | 158 | 70 | 77 | 195 |
| MMLU-Redux | 1,658 | 846 | 815 | 2,296 |
| Total | 2,167 | 1,076 | 1,078 | 2,944 |

这些计数只由 input content hash 决定，不读取 label、prediction 或 event。正式 selector 执行前仍须生成逐 group split manifest 并核对其 SHA。

禁止在 model-selection 后用 fit+calibration+model-selection refit；否则 calibration/model-selection 将不再严格分开。若未来确需 refit，必须在查看任何 test outcome 前另行预注册 nested cross-fitting 方案。

## 6. Feature tier 与 confirmatory whitelist

### 6.1 Primary deployable feature pool

Confirmatory selector 只能使用审计为 A 类且不包含 raw identity/text 的结构特征：

- `cot_input_length`；
- `candidate_count`；
- `candidate_count_max`；
- `one_to_many_rate`；
- `boundary_mismatch`。

当前 `alignment_entropy == one_to_many_rate`，`confidence == 1-0.5×entropy`，`fallback_rate` 恒为 0，因此它们不作为额外独立 primary features。若后续 instrumentation 改变这些关系，必须重新预注册，不能直接加入现有 test。

Raw question/options、subject/task/pair metadata 不进入 primary selector。它们可在明确标记的 secondary analysis 中研究，但不得替代 primary GO 结果。

### 6.2 B/C/D tier

- B 类 fused output/gate/latency 仅用于 post-fused diagnostic；不能称为 no-transfer selector。
- C 类 receiver full-forward 输出仅用于 dual-pass upper bound；不能称为零额外计算 selector。
- D 类 label/correctness/event/utility/oracle/identity 字段永远禁止作为 feature。

Test split 不得用于决定保留哪个 A 类字段、如何标准化、缺失值处理、交互项、阈值或 selector candidate。

## 7. Candidate selection 与 calibration 顺序

后续实现前，所有 selector candidates、正则、标准化、缺失值规则和 random seeds 必须写入单独 manifest。执行顺序固定：

1. 在 fit split 拟合每个预定义 candidate；
2. 在 calibration split 拟合 calibration mapping，并为每个 candidate 冻结 action threshold；
3. 同一 calibration split 选择 global fixed comparator `p*`；
4. 在 model-selection split 选择唯一 candidate；
5. 不再修改参数、feature、calibration 或 threshold；
6. 一次性运行 test 并生成 locked report。

若 candidate 数量、feature family 或阈值搜索空间未在 test 前冻结，则 confirmatory test 无效。

## 8. Leave-one-out 泛化协议

### 8.1 Leave-one-seed-out

- held-out seed 的 test groups 作为外层 test；
- 其他 seeds 只能使用各自 fit/calibration/model-selection groups；
- 所有 seeds 中属于 test split 的同内容 observations 均不得进入开发；
- comparator 和 threshold 在每个 outer fold 的 calibration 数据上重新选择并冻结。

### 8.2 Leave-one-task-out

- 整个 held-out task 不进入 fit/calibration/model-selection；
- held-out task 全量作为 outer test；
- selector、calibration、threshold 和 comparator 只由其余 tasks 决定。

### 8.3 Leave-one-pair-out

- held-out pair 的 test groups 作为外层 test；
- 其他 pairs 只能使用 fit/calibration/model-selection groups；
- 同一 test content 在其他 pairs 的复制观测也不得进入开发；
- 每 fold 重新在非 held-out pairs 的 calibration 上选择 `p*`。

所有 outer-fold 结果都必须完整报告，不能只选方向有利的 folds。内部 dev/test 仍属于同一组三个开发 benchmark，不得包装成外部 benchmark 泛化。

## 9. 统计协议

Primary CI 使用 10,000-draw pair-balanced hierarchical paired bootstrap：

1. 重采样 pair；
2. 在选中 pair 内重采样 seed；
3. task 内重采样 canonical content groups；重复内容组作为一个 cluster 被抽中，并保留该组全部 member rows 与原 row weights；
4. 同一个 sample draw 同步应用于 receiver、fused、selector 和所有重复 observations；
5. 每个 draw 内重新应用冻结的 fixed comparator，并计算 nonlinear headroom/recovery。

报告：

- primary accuracy delta 与 95% CI；
- task-macro 与 sample-weighted；
- each pair/seed/task point estimate；
- positive pair 数与三异构 pair 方向；
- seed mean、sample std (`ddof=1`)；
- harmful accepted、beneficial accepted；
- leave-one-seed/task/pair-out；
- test-selected fixed diagnostic 只能标为 retrospective sensitivity。

不能用 pooled McNemar 或 87,180 repeated rows 的普通 IID CI 替代跨 pair 稳定性。

## 10. Headroom recovery 与 transfer preservation

定义：

- `H_all = P(u=-1)`：always-fused 的全部 harmful transfers；
- `H_selector = P(a=1 and u=-1)`：selector 仍接受的 harmful transfers；
- `B_all = P(u=+1)`：always-fused 的全部 beneficial transfers；
- `B_selector = P(a=1 and u=+1)`：selector 保留的 beneficial transfers。

指标：

- harmful reduction = `1 - H_selector/H_all`；
- beneficial retention = `B_selector/B_all`；
- conservative true-headroom recovery =
  `(Accuracy(selector)-Accuracy(test retrospective best fixed)) /
   (Accuracy(oracle)-Accuracy(test retrospective best fixed))`。

所有 harmful/beneficial 指标先按 primary pair-balanced task-macro 权重分别聚合 numerator 与 denominator，再取 ratio；禁止先算 cell-wise ratio 后平均。Bootstrap 在每个 draw 内重算 ratio。恢复比例同样先聚合 accuracy，再取 ratio。

恢复比例使用 test retrospective best fixed 仅作为严格分母/机会基准，不用于训练或选择。若 selector 低于该 fixed policy，recovery 可为负，不截断为 0。任何分母为 0 或未定义时，GO 自动失败。

## 11. GO / NO-GO 条件

以下六项必须在 held-out primary test、pair-balanced task-macro 上同时成立：

1. selector 相对 calibration-selected `p*` 至少 `+0.5 pp`；
2. pair-balanced 95% CI 下界严格大于 0；
3. 三个预注册异构 pairs 中至少 `2/3` 的 held-out point estimate 为正，其余 pair 的 point estimate 不低于 `-0.2 pp`；
4. conservative oracle-over-best-fixed headroom recovery 至少 15%；
5. harmful transfer 相对 always fused 至少减少 25%；
6. beneficial transfer 至少保留 80%。

所有条件为 conjunctive。任一失败即 NO-GO，不进入更复杂 query-time transport/router。Sample-weighted 和 leave-one-out 结果必须同时报告，但不得用于替换 primary 以获得 GO。

## 12. 结果解释锁定

- headroom 大但 A 类 selector 失败：说明现有部署时特征缺乏预测能力；优先考虑是否需要一次独立预注册的 instrumentation audit，而不是直接扩大模型容量。
- 只有 B 类特征有效：只能说明 post-fused diagnosis 有信息，不能支持 no-transfer compute saving。
- 只有 C 类特征有效：说明 dual-pass uncertainty 有上界价值，但不是零额外成本方案。
- 只有 pair/task metadata 有效：解释为 compatibility/base-rate routing，不宣称 sample-level calibrated transfer。
- 异构 pair 不稳定而 same-tokenizer control 稳定：转向 sender–receiver compatibility，不进入通用 selector 主张。

## 13. Test 保密与停止规则

在正式 test 运行前必须提交：split manifest hash、feature whitelist、candidate manifest、normalization/calibration/threshold code hash、bootstrap seed 和 fixed-comparator rule。

Test 运行后禁止：

- 改 feature、阈值、candidate 或 comparator；
- 按 test 结果删除 pair/task/seed；
- 把 model-selection 或 test 合并回训练；
- 只报告相对 fused 的有利比较。

Phase 2A-0 不执行这些后续步骤。本文件提交后停止并等待审查。
