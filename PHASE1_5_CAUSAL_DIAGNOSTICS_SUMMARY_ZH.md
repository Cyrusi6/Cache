# Phase 1.5 同 checkpoint 因果诊断摘要

> 当前状态：全部完成。所有注册评测通过最终产物审计，pair-balanced 5,000 次 bootstrap 已完成，query-time prototype 放行失败。

## 这一步要回答什么

Phase 1 只能说明不同训练方法最终 checkpoint 的差异，仍混合了模块作用、训练轨迹和模型对兼容性。Phase 1.5 固定 checkpoint，只在推理期改变一个因素，回答：

1. 多个 source candidates 本身是否有因果贡献；
2. entropy 的数值和位置对应关系是否真的提供信息；
3. learned gate 是否优于 static 或 always-on；
4. Qwen2.5 seed 44 的崩塌是否来自 gate 硬开关和训练轨迹；
5. receiver-only 与 fused 之间是否存在可被 abstention 利用的 headroom。

本阶段禁止 query-time transport、router、replay、OT、RoPE、新 gate 和新 loss。主矩阵不训练、不修改权重，也不为单个干预选择不同 checkpoint。

## 已完成的可信性审计

- Phase 1 的 67 个 completion markers、201 个逐例 prediction CSV、26 组 post-hoc gate diagnostics 均存在，failed marker 为 0。
- 65 个新训练 checkpoint 加 1 个 bitwise-verified 复用 checkpoint，共 66 个必需 checkpoint，均能加载 28 层 projector，tensor 全部有限。
- 65 个本地 checkpoint 的 run id、执行 commit、训练配置 SHA、split hash 和数据 hash 均与 provenance 一致。
- TinyLlama B6 seed 42 复用权重的目录 SHA256 为 `a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`。
- 三个 NVIDIA 节点均可访问 `/netdisk`；启动审计时共享盘约剩余 923 GB，14 张 NVIDIA GPU 无活动实验 request。

因此，后续异常不能简单归因于缺 checkpoint、权重损坏、逐例文件缺失或版本来源不明。

## Qwen2.5 seed 44 已经能够确定的事实

| Seed | 三任务 weighted accuracy | macro accuracy | 正迁移条件率 | 负迁移条件率 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 46.194% | 50.215% | 26.671% | 19.338% |
| 43 | 45.891% | 49.498% | 24.795% | 16.863% |
| 44 | 38.789% | 41.570% | 10.263% | 10.849% |

seed 44 相对 seeds 42/43 均值下降约 7.25pp，主要原因是 receiver-wrong→fused-correct 从约 25% 降到 10.3%，不是负迁移爆炸。7,265 个预测全部是合法选项。

它也不是数值训练崩溃：checkpoint 完整，无 NaN/Inf；训练正常退出；梯度与 clipping 没有异常证据；seed 44 的 train loss 为 0.1326、eval loss 为 0.1211，后者甚至低于 seed 42 的 0.1749 和 seed 43 的 0.1395。这再次说明小训练 eval split 的 loss 不能替代下游迁移评测。

alignment-confidence gate 在三个 seeds 上都约 99.93% 高饱和，不能解释 seed 特异失败。更关键的是 checkpoint 内仍有逐层 legacy scalar K/V gate，其 eval 决策是对接近 0 的 logit 做硬阈值。seed 44 的 28 层中有 9 层 K/V 同时关闭；前 9 层有 7 层同时关闭，前 6 层完全没有 transfer。当前最符合证据的解释是：

`seed 训练轨迹 × legacy scalar 硬开关 × Qwen2.5/Qwen3 cache compatibility`。

为验证这一点，另跑两个不进入主矩阵的 seed-44 干预：只强制 alignment-confidence 开启，以及只强制 legacy scalar K/V 开启。

这两个反事实已经完成。只强制 alignment-confidence 开启后，7,265 个样本中只有 1 个预测字母变化，而且该样本前后都错；pooled accuracy、正迁移和负迁移分别保持 `38.789%`、`10.263%`、`10.849%`，McNemar `p=1.0`。因此 seed 44 的异常不是 learned alignment-confidence/token-head gate 没有打开造成的，该 gate 在这个 checkpoint 上实际上已经饱和。

只强制 legacy scalar K/V masks 开启后，pooled accuracy 提升到 `40.702%`，相对 native 为 `+1.913 pp`；逐例有 336 个改善、197 个回退，McNemar `p=1.85e-9`。正迁移条件率升至 `14.036%`，负迁移也升至 `12.219%`。它大约恢复了 seed 44 相对 seeds 42/43 均值缺口的 26%，说明 near-zero logit 经硬阈值后关闭过多层确实是一个因果来源，但仍不能解释剩余大部分崩塌，也不是稳健的全开解法。

## 同 checkpoint 主矩阵

主矩阵为 4 个模型对 × 3 seeds × 6 个新干预，共 72 个三任务 triplets；Phase 1 的 36 个 native B2/B3/B6 triplets 直接作为 comparator。

| 问题 | 同 checkpoint 对照 |
| --- | --- |
| candidates | B2 eval-k1/k4；B3 eval-k1/k4，形成 train-k × eval-k 2×2 |
| entropy 数值 | B6 native − constant-0.93 |
| entropy 位置 | B6 native − sequence-internal shuffled |
| learned gate | B6 learned − static |
| adaptive gate 必要性 | B6 learned − forced-on；forced-on 同时覆盖 alignment gate 和 legacy scalar K/V gate |

模型对中 `qwen3_1p7b` 是同 tokenizer 控制；TinyLlama、Qwen2.5 和 Llama3.2 记为真正异构模型对。

## ambiguity、oracle 与统计协议

- top-k 对照固定使用 native B3 diagnostics 分桶；entropy/gate 对照固定使用 native B6 diagnostics，避免干预本身改变分桶定义。
- 同时报告 absolute high ambiguity 和 pair/seed/task 内 composite ambiguity score 的 top quartile；composite 使用可用的 entropy、one-to-many rate 与 boundary mismatch。
- oracle 在每个样本上选择 fused 正确答案，否则在 receiver 正确时 abstain 回 receiver。它只给出理论上限，不是新方法。
- 主统计与 Phase 1 一致：pair→seed→paired example 的层级 bootstrap，模型对和 seed 等权；默认 5,000 次、95% CI、bootstrap seed `20260718`。
- 同时输出逐 pair/seed/task accuracy delta、McNemar、正向 pair 数、真正异构正向 pair 数和 seed sample std。大样本聚合 McNemar 不能替代跨 pair CI。

统计入口为 [phase1_5_causal_diagnostics.py](script/analysis/phase1_5_causal_diagnostics.py)；传入可选的 `--anomaly-manifest` 后，会在不改变主 8 项对照的前提下加入 Qwen2.5 seed 44 的 `alignment_forced_on − native`、`legacy_forced_on − native` 逐任务/pooled 配对统计与 oracle。输出包括 `paired_interventions.csv`、`hierarchical_interventions.csv`、`seed_variance.csv`、`oracle_abstention.csv`、`ambiguity_interactions.csv` 和 `summary.json`。大体积逐例及中间文件保留在 `local/` 或 `/netdisk`，不提交 Git。

## 运行基础设施

- 72 个 triplets 固定分成七个双卡逻辑 shard，数量为 `[11,11,10,10,10,10,10]`。
- 每个 triplet 先由 GPU0 跑 ARC、GPU1 跑 OpenBookQA，再由两卡跑 MMLU-Redux。
- x4 节点覆盖 shards 0/1，x8 节点覆盖 shards 2–5，x48 节点覆盖 shard 6；seed-44 两个额外干预单独排队。
- launcher 核验精确 Git commit、execution manifest SHA、固定 runtime constraints、可见 GPU UUID 和实际显存；完成输出才允许 resume。
- 已针对共享 NFS 目录创建竞态、autofs negative dentry 和 x8 结果根隔离做了防护。
- 当前实现通过 `236 passed`，保留 2 个既有 Pydantic warnings；节点级 Job 均通过 Kubernetes API server dry-run。

x8 节点随后在 shards 2--5 完成 84/120 个 dataset outputs、留下 3 个 MMLU provenance-only partial 和 33 个未启动输出时进入 `NodeStatusUnknown`。shards 2--4 的跨节点恢复按输出契约完成；但 shard 5 的多 Pod work stealing 实测证明当前共享 NFS 上的 advisory `flock` 不能提供跨 Pod 互斥，导致 5 个 dataset 各产生两套结果。逐例审计确认两套的预测、正确性、ambiguity、gate 和 length 完全一致，仅 latency 与时间戳文件名不同；ghost bundles 已完整移入 `local/tmp` 隔离区，不参与统计。

最后尾部改用 immutable、显式不重叠的 Kubernetes Jobs：每个 Job 固定一个 run id 和一对物理 GPU UUID，旧串行 worker 在当前 MMLU 完整落盘后才删除。当前部署不再把跨 Pod opportunistic work stealing 视为安全恢复方式；可复现恢复合同是显式 run 分配加 exactly-one-output 审计。

第一次最终审计又发现一个纯 provenance 可复现性缺口：TinyLlama B2 eval-k4 seed 43 的三任务由较早生成的 manifest 执行，当前最终 YAML 的字节 SHA 与产物记录不一致，虽然 checkpoint、有效 top-k 和干预语义一致。旧 bundle 已完整隔离，单独按最终 manifest 重跑。新旧逐例预测、正确性、ambiguity、confidence/gate、length、bad samples 以及除 latency 外的全部 CSV 科学字段完全一致，因此修复了字节级可复现性但没有改变实验结果。

最终严格审计现已全 PASS：74/74 runs、222/222 dataset evaluations；x8 `120/120`、main `96/96`、anomaly `6/6`。每个 dataset 都恰好有一份 CSV、summary、provenance、gate diagnostics 和 length diagnostics，行数、sample key、JSON、checkpoint/intervention、内部 provenance SHA 与当前 config SHA 全部通过。主 manifest SHA 为 `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1`，anomaly manifest SHA 为 `bd305268e9a8527cb75407293b49cae4e577bb10516e9643781573e861cfa5d2`。

5,000 次统计分别在 Kubernetes 与本地审计 Conda 环境独立运行，随机 seed 相同。paired、oracle、ambiguity 三份 CSV 字节级一致；其余差异只来自 sample std 的浮点序列化，最大绝对差 `4.337e-19`，没有字符串、结论或放行字段差异。

## 正式结果

| 机制问题 | 关键对照 | 聚合 delta / 95% CI | ambiguity interaction | 当前判断 |
| --- | --- | --- | --- | --- |
| 多 candidates | B2/B3 same-checkpoint eval-k4 − eval-k1 | B2 −0.01 pp [−0.13,+0.11]；B3 +0.03 pp [−0.06,+0.14] | q75 −0.02/−0.09 pp；absolute −0.05/−0.01 pp | 未识别到平均推理期贡献 |
| entropy 数值 | native − constant | +0.13 pp [−0.13,+0.40] | q75 +0.10 pp；absolute −0.01 pp | 不稳定，删除主张 |
| entropy 位置 | native − shuffled | +0.04 pp [−0.07,+0.22] | q75 −0.02 pp；absolute −1.19 pp | 不稳定，删除主张 |
| learned gate | learned − static | −0.01 pp [−0.10,+0.07] | q75 −0.06 pp；absolute −0.30 pp | 无 learned 优势 |
| adaptive gate 必要性 | learned − forced-on | −0.21 pp [−0.98,+0.62] | q75 −0.08 pp；absolute −3.97 pp，CI [−7.30,−0.92] | pair 方向翻转；无统一优势 |
| Qwen2.5 seed44 | alignment-only/legacy-only forced-on | +0.00 pp / +1.91 pp [1.31,2.53] | 不适用 | alignment gate 非因；legacy hard mask 是部分原因 |
| abstention | B6 native oracle − fused | +8.24 pp [6.28,10.19] | 4/4 pairs 为正 | 转向 calibrated null/no-transfer |

同 checkpoint 的 candidate 数量变化几乎没有影响，也没有提供可靠的高 ambiguity 正向集中证据。B3-trained 相对 B2-trained 在 eval-k1/k4 下仍有约 +1.25/+1.30 pp 的点估计，但跨 pair CI 均跨 0、seed std 约 2.4 pp；模型对上主要来自同 tokenizer 的 Qwen3（+3.12 pp）和 TinyLlama（约 +2.02 pp），Qwen2.5 与 Llama3.2 接近 0。这说明差异编码在训练 regime/checkpoint 中并且强烈依赖模型对，不能归因于推理期 soft candidate coverage；本实验也不能进一步区分训练期 k4 暴露、优化轨迹、tokenizer 身份或其他 compatibility 因素。

entropy 的数值和位置对应都没有跨 pair 稳定贡献，因而不再补 TinyLlama constant/shuffle seeds 43、44；这并不等于证明 entropy 在所有设置中都无信息。learned token/head modulation 相对 static 近乎完全相同。`learned − forced-on` 还同时改变 legacy scalar K/V masks，并且不同模型对方向翻转，因此既不是纯 token/head 对照，也不能说明 forced-on 普遍更优。现有证据不能支持 content-conditioned adaptive capacity 的性能主张。

ambiguity interaction 只能作为探索性敏感性分析：TinyLlama 的 absolute 定义全部为 high，其他模型对的 absolute high 很稀疏；pooled q75 在多个 pair 上又近似把 MMLU 与 ARC/OpenBookQA 分开，混入 task 差异。因此不能把负 interaction 写成普适反机制；能够用于放行的结论只是：两个 same-checkpoint top-k 对照都没有可靠的正向 ambiguity concentration。

B6 native 的 oracle headroom 为 +8.24 pp，95% CI [6.28,10.19]，说明 receiver-correct→fused-wrong 仍有很大的可恢复空间。但这六份注册统计没有 calibrated gate score，不能从 oracle 反推现有 gate 的预测能力；结合 learned gate 的 null 结果，下一方向应是 calibrated null/no-transfer，而不是继续强化 adaptive gate。

## 放行规则

只有满足以下三项，才允许进入小型 query-time prototype：

1. 同 checkpoint 的 top-k4 在至少两个真正异构模型对上优于 top-k1；
2. pair-balanced 跨 pair 95% CI 下界大于 0；
3. 收益集中在高 ambiguity token/span。

否则按预注册规则收缩主张：learned≈forced-on 时删除 adaptive gate 主张；entropy 无稳定作用时删除 entropy-confidence 主张；oracle headroom 大但 gate 无预测能力时转向 calibrated null/no-transfer；收益只在同 tokenizer pair 稳定时转向 sender–receiver cache compatibility。

最终结论：**放行失败，不进入 query-time transport prototype。** B2 只有 1/3 个真正异构 pair 为正；B3 虽为 3/3，但跨 pair CI 下界为 −0.06 pp；两者都没有可靠的正向 ambiguity 集中。第一阶段应删除 `entropy-aware` 与 `adaptive token/head gate` 的已验证机制主张，也不能把 v2.2 的开发集收益解释为已识别的推理期多 candidate 因果效应。剩余变化更符合 training-regime/checkpoint、legacy scalar hard mask 与 sender–receiver compatibility 的组合，但本实验不单独识别其中任一因素；后续只保留 calibrated null/no-transfer 与 compatibility 方向。
