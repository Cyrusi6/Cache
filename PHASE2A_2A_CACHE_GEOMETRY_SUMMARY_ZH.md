# Phase 2A-2a Cache Geometry Pilot 中文总结

## 最终判定

**NO_GO。** 第一项“instrumentation 开启后必须与冻结的原 B6-native 逐例完全一致”失败，因此按预注册立即停止剩余 GPU 评测和全部 selector 拟合。

本轮不能回答“cache geometry 是否能预测 harmful transfer”。能够确定的是：当前 instrumentation run 没有通过解释 selector 结果所必需的无扰动前置门控。

## 已完成结果

ARC 与 OpenBookQA 的 6 个 cell 在停止前完整结束：

- TinyLlama：509/509 样本 exact；
- Qwen2.5：509/509 样本 exact；
- Llama3.2：494/509 exact，15 个不一致；其中 8 个答案变化，7 个仅生成文本形式或长度变化。

Llama3.2 的不一致分布为：

- ARC：11/351，5 个答案变化、6 个 generation-only；
- OpenBookQA：4/158，3 个答案变化、1 个 generation-only。

这些变化让 instrumented 版本比 reference 多答对 5 题，但方向不影响判定：无扰动门控要求逐例 generation/prediction 完全一致，任何变化都失败。

## Geometry 记录是否正常

正常。六个完整 cell 共得到：

- 1,527 条 sample records；
- 42,756 条 layer records；
- 每样本严格 28 层；
- 每个 cell 都有 65 个非恒定 K 特征和 65 个非恒定 V 特征；
- 所有样本均有真实的层间 K/V variation。

因此 observer 不是空值或 pair 常数。但 Gate 1 已失败、MMLU 未完成，所以这些只能说明 instrumentation 有数据，不能算正式 Gate 2，更不能进入 harm AUPRC 或 selector accuracy 分析。

## 为什么不能继续训练 selector

九项 GO gate 是 conjunctive：第一项失败后，其余八项不再评估。继续跑 MMLU 或拟合 184 个候选只会消耗算力，并产生不可解释的结果。

本轮已按规则做到：

- 删除全部 Phase 2A-2a Kubernetes Job/Pod；
- 未运行 geometry/outcome join；
- 未让统计脚本读取 correctness；
- 未拟合 stump、logistic regression 或 tree；
- 未进入三 seeds、same-tokenizer control、新 benchmark 或 query-time prototype；
- 未修改 main 或 FPCT worktree/branch。

## 可以与不可以得出的结论

可以说：

- exact mismatch 集中在 Llama3.2 pair，TinyLlama 和 Qwen2.5 在两个完整任务上全部一致；
- 这提示 Llama sender–receiver 组合存在更高的数值/cache compatibility 敏感性；
- 当前 instance selector/adaptive gate 路线应停止。

不可以说：

- geometry 能或不能预测 harmful transfer；
- mismatch 一定由 instrumentation 引起；
- mismatch 一定只是底层数值非确定性。

因为匹配的 Llama3.2 instrumentation-off 诊断没有预注册且没有运行，当前无法区分 observer timing perturbation 与 Llama-specific baseline sensitivity。若以后要做，只能在新授权下进行最小的 equivalence-only debug，不能自动重开 selector pilot。

## 资源与产物

- 正式执行 commit：`7f57a37af18842611a3b85865de6daeb98803a5e`。
- Execution manifest SHA：`5556fbdcc3cf57f9978527256ef7b2154277d4d3b1fdae20711a3b5b88b2e042`。
- Job YAML SHA：`50b2c84afd7e0e5da361137e288b3b3489f6abb5d219294515308d225aeb294b`。
- 完整报告：`PHASE2A_2A_CACHE_GEOMETRY_PILOT_REPORT.md`。
- 小型 aggregate：`recipe/eval_recipe/phase2a_2a/phase2a_2a_failfast_aggregate.{json,csv}`。
- 大体积逐层结果：`/netdisk/lijunsi/c2c-phase2a2-cache-geometry/results`，约 295 MiB，不提交 Git。

Instrumentation-off 匹配对照尚未完成，因此不报告 overhead delta；现有 wall time/显存仅为 debug 资源诊断，不包装为部署开销。
