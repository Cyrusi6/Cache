# FPCT-E1-FAST-RCA 结果报告

## 结论

FPCT-E1-FAST-RCA 的前瞻性判定为：

`NO_EXPLOITABLE_FIXED_CHECKPOINT_HEADROOM`

六个 E0 checkpoint、326 个冻结 E0-design content groups 和 18 个
checkpoint×task shards 全部完成并通过 SHA/row-count verifier。四个可选择干预均未同时
满足预注册的 95% LCB、5/6 checkpoint arms 正向、三任务均非负和 exact controls 条件。
因此没有冻结 production winner，`root_cause=null`、`selected_variant=null`、
`training_authorized=false`；320-step 单 seed 和条件式另外两 seeds 均未启动。

这不是“FPCT 数学无效”的结论。它表示现有 E0 checkpoints 上没有一个已冻结的单因素
干预显示出足够稳定、可利用的 fixed-checkpoint teacher-forced headroom。

## 执行与完整性

- Execution SHA：`07b2a3f5f1437d382ce1b548661200ac0fb2ea61`。
- K8s Job：`fpct-e1-fast-formal-07b2a3f5-20260820-233130-701513`，node=
  `4090-24gx4`，restart=`0`，terminal=`Succeeded`。
- UTC 时间：`2026-08-20T15:31:30Z` 至 `2026-08-20T18:33:18Z`。
- Population：ARC/OpenBookQA/MMLU-Redux=`128/70/128` distinct groups。
- Coverage：18/18 shards，17,604 compact rows，1,956 matched checkpoint-group cells；
  expanded long-form rows=`0`。
- Final verifier：`GO_ALL_SHARDS_COMPLETE`。
- `lambda_0` exact control：maximum absolute logp delta=`0.0`。
- E1-pilot、model-selection、test、confirmatory 均未读取。

同一 pod 最多使用六个互斥 worker lanes。各 lane 只处理不同 shard；已完成 shard 通过
receipt 只读复用，没有覆盖或只补跑某个 arm。GPU 0/3 的既有负载未被修改；本任务只在
GPU 1/2 上执行 frozen-checkpoint forward。

## 前瞻性 winner 判定

下表均为相对原 F 的 group-level answer-token mean `delta logp(y*)`。CI 是冻结的
20,000-replicate hierarchical paired bootstrap 95% percentile interval，顶层单位是三个
training seeds，两个 checkpoint arms 在 seed 内配对。

| 干预 | task-macro mean | bootstrap 95% CI | 正向 checkpoint arms | 三任务均非负 | Eligible |
|---|---:|---:|---:|---:|---:|
| centered lambda=0.25 | `+0.00025335` | `[-0.00031074, +0.00085394]` | `5/6` | 否（ARC `<0`） | 否 |
| centered lambda=0.5 | `+0.00007732` | `[-0.00046502, +0.00070885]` | `3/6` | 否（ARC `<0`） | 否 |
| parent-mass preserving | `+0.00009137` | `[-0.00056641, +0.00067193]` | `4/6` | 否（ARC/MMLU `<0`） | 否 |
| partition composition | `-0.00010404` | `[-0.00098185, +0.00076642]` | `2/6` | 否 | 否 |

`lambda=2` 和 K-only/V-collapse 按预注册仅作诊断，不可被选择。Exact RoPE correction
在当前 frozen nonlinear projector 上没有可分离 `P_K`，因此在自然输出前已经标记为
`STRUCTURALLY_UNAVAILABLE_AT_FIXED_CHECKPOINT`；没有用近似旋转冒充 math.md operator。

## 原 F 的同 checkpoint效应

原 F 相对 C_post 的 teacher-forced task-macro `delta logp(y*)` 为
`-0.00011850`。分任务为：

- ARC：`-0.00010409`；
- OpenBookQA：`-0.00038335`；
- MMLU-Redux：`+0.00013196`。

正向 group-cell 比例为 `0.49847`，接近随机对半；六个 checkpoint 的 task-macro 方向为
三正三负。该结果与 E0 的即时 operator 负信号相容，但它是 E0-design 的机制诊断，不能
替代 accuracy 或 confirmatory claim。

## 机制定位

冻结 instrumentation 证明 query-time candidate mechanism 确实被激活，但没有形成稳定的
task-aligned signal：

- source 相对 dispersion：K=`0.07504`、V=`0.28004`；
- fused 相对 dispersion：K=`0.00002266`、V=`0.00000476`；
- fused/source 保留率：K=`0.00030218`、V=`0.00001701`；
- candidate logit range mean=`0.04904`；
- `KL(gamma||A)` mean=`0.001238`，`TV(gamma,A)` mean=`0.01120`；
- answer-query gamma variance mean=`0.00009435`；
- 每个 group-cell 都至少观察到一次 posterior top-1 改变，平均 change rate=`0.15052`；
- parent attention mass mean=`0.002056`；
- factorized-vs-collapse output delta L2 mean=`0.0006855`；
- Jensen gap mean=`0.001263`。

这些 mechanism metrics 与 group-cell `F-C_post delta logp` 的 Pearson 相关绝对值均不超过
`0.055`。因此最强的描述性定位是：原始 source candidates 有明显差异，但现有
projector/fuser 把相对 K/V 差异分别压缩到约 `0.030%` 和 `0.0017%`；剩余 posterior
变化虽可测，却只作用于平均约 `0.206%` 的 parent mass，且没有与 gold-token收益对齐。

这是一条描述性 localization，不是预注册的唯一根因标签。因为没有 intervention 通过
winner gate，正式 `root_cause` 必须保持 `null`。

## 决策与下一步边界

- 不实现 lambda=0.25：它是最接近的候选，但 CI 跨 0 且 ARC 为负。
- 不实现 parent-mass 或 partition hybrid：两者未通过方向/任务 gate。
- 不启动任何 320-step 训练或补 seeds。
- 如果继续研究，建议另开前瞻性协议测试“candidate-distinction-preserving fuser”：在
  nonlinear fusion 后保留受控的 source residual，或把 source projection 改为显式可分离、
  可做 exact de-RoPE/re-RoPE 的路径。该方向会改变 operator/fuser，不能在本轮结果后
  直接实现，也不能追认为本轮 winner。
- native null、selector、新 gate、跨模型和正式 confirmatory 继续 sealed。

## 产物

- Persistent root：
  `/netdisk/lijunsi/fpct-e1/fpct-e1-fast-rca-07b2a3f5-v1`。
- Formal shard inventory SHA256：
  `f12d8b7da233f437916b98a9e796c68c30ca9531d5698ce3eed281d0ee1eb275`。
- Root-cause result SHA256：
  `4467bf4b8c6966c838f66e2f4548b4e4650717a8f2e41bae9380281a40789c1f`。
- Descriptive mechanism summary SHA256：
  `a7a869ae239dc29c4d9a961428942ba9d95299b951947a0af7ccf4ee340b09c2`。

大体积 sample/layer-head rows 保存在 `/netdisk`，不提交 Git。Git 只保存本报告、compact
result manifest 和状态更新。
