# Phase 2A-1 Selector Kill-Test 中文摘要

## 结论

**NO_GO**

六项联合门槛至少一项失败，因此 Phase 2A-1 为 NO-GO。按预注册停止复杂/神经 selector，不自动进入下一阶段；如要继续，只能等待是否授权一次新的 pre-transfer cache-geometry instrumentation。

冻结后的唯一候选是 `stump_cot_input_length`，比较对象是在
calibration split 选定的 `always_fused`。主指标
（pair-balanced task-macro）差值为 +0.00 pp，
95% 分层 paired bootstrap CI 为
[+0.00 pp,
+0.00 pp]。

- transfer rate：100.00%
- harmful transfer reduction：0.00%
- beneficial transfer retention：100.00%
- oracle-over-best-fixed headroom recovery：0.00%
- 未通过门槛：primary_delta_at_least_0p5pp, primary_ci_lower_above_zero, heterogeneous_pair_sign_rule, oracle_headroom_recovery_at_least_15pct, harmful_reduction_at_least_25pct

本轮严格只用了五项 A-tier 特征，未使用题目文本、task/pair/seed、ID、标签、
正确性字段、entropy/confidence 冗余字段或恒零 fallback。Test 在候选、阈值、
comparator、代码和模型 SHA 全部冻结后只执行了一次。逐例文件保留在 `local/`，
仓库只提交小型聚合结果与审计 manifest。
