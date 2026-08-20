# FPCT-E1-FAST-RCA 前瞻性修订

> 状态：`PRE-OUTPUT LOCK`
> 日期：2026-08-20（Asia/Shanghai）
> 研究线：`research/fpct-e1-mechanism-audit`

## 决策与历史边界

本修订停止 A5R8/A5R7 的 30,370,816-row 预物化路线。旧 root
`/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-1c64b606-v1` 永久保留为工程记录；其
producer 和 inner verifier 已完成，但 outer receipt 因双 JSON framing 失败，因此不
追认为 deep-verifier GO，也不把其自然结果用于科学结论。

唯一允许复用的是 SHA256 固定的 25.5MB compact E0-design input cache：

```text
/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-1c64b606-v1/input_lock/e0_design_input_lock.pt
SHA256=d843512b7e446efd229fe3097030407e59acb20b4de591a84b88176bd8b0eec9
```

不读取、复制或重放 `input_row_templates/`，不物化逐
`answer-query × parent × layer × head` rows。Compact cache 只提供冻结的 326 个
E0-design groups、runtime feature、answer query、certified parent/topology 与 provenance；
它不把 A5R7 的状态转化为科学 GO。

## 科学问题与 estimand

固定六个 E0 step-64/final checkpoint（三 seed × C_post-trained/F-trained）和 ARC
128、OpenBookQA 70、MMLU-Redux 128 个 distinct content groups。对完整 gold response
做 teacher forcing。每个 group 先对 answer tokens 等权计算
`mean log p(y*)`，task 内 groups 等权，三任务等权。干预主诊断是相对原 F 的
`Δlogp(y*)`；accuracy 不用于当前选择。

机制统计只在线保留 compact moments：KL/TV、跨 answer-query gamma variance、top-1
change、candidate-logit range/variance、Jensen gap、parent mass、source/fused K/V
dispersion/energy、projector contraction 与 factorized-collapse output delta。禁止保存 raw
KV 或展开 30M rows。

## 前瞻冻结的固定-checkpoint干预

1. `centered λ∈{0,.25,.5,1,2}`：`λ=0` 仅为 C_post exact control；`λ=2`
   仅为外推诊断；可选择值只有 `.25/.5`。
2. `K-only/V-collapse`：保留 candidate K，所有 candidate V 复制 collapsed V；仅用于
   区分 Jensen/parent evidence 与 V routing，不可成为 production winner。
3. `parent-mass preservation`：先用 C_post 得到 parent probability，再用 F 的
   `gamma(j|i,q)` 只在 parent 内分配，所有 parent 仍共用原 C_post denominator。
4. `partition composition`：certified partition rows 用真实 intersection length 归一化，
   在 nonlinear fuser 前 composition；严格 competing rows 才保留 F。当前 compact
   population 的 certified m≥2 rows 均为 partition，必须透明报告 competing branch 为空。
5. `RoPE frame correction`：固定 checkpoint 下结构性不可执行且不可选择。现有
   `C2CProjector` 在第一个 nonlinear map 前已拼接 source/receiver K，未暴露独立
   `P_K`；sender/receiver head dimension 也不同。因此不能精确计算
   `R_r P_K R_s^{-1}`。本轮禁止用 pre-project rotation 或其他不等价近似冒充公式。

所有实际干预使用相同 checkpoint、input、mask、candidate identity、fuser 参数和
teacher-forced query。除 partition composition 明示改变 composition weights 外，不改变
A；不训练，不读取 E1-pilot。

## 唯一因素选择

Selectable intervention 必须同时满足：

- 相对原 F 的 task-macro `Δlogp(y*)` hierarchical paired bootstrap 95% LCB > 0；
- 六个 checkpoint arms 至少 5 个方向为正；bootstrap 顶层只有三个 training seeds，
  seed 内 C-trained/F-trained checkpoint 成对携带，不把六 checkpoint 当六个独立 seed；
- ARC、OpenBookQA、MMLU-Redux 各 task mean 均不为负；
- `λ=0=C_post`、invalid/mask、provenance、complete coverage controls 全部 GO。

合格项按 task-macro mean 最大者选择；差值在 `1e-6` 内按
`λ=.25 → λ=.5 → parent-mass → partition-composition`。胜出标签分别对应 candidate
deviation scale/contraction、Jensen parent-evidence inflation 或
partition-as-competing-candidates。若无合格项，状态固定为
`NO_EXPLOITABLE_FIXED_CHECKPOINT_HEADROOM`，停止训练。

## 条件训练

只在一个因素获选并通过新增 production oracle 后，才可运行 seed `2026081401` 的
C_post、原 F、新 operator 三臂 320-step matched pilot。只有 `T_new>0`、`O_new>0`、
teacher-forced `Δlogp>0`、机制超过 numerical null 且 integrity controls GO，才补
`2026081402/2026081403`。E1-pilot 始终 exploratory；confirmatory、model-selection、
test、native null、跨模型与 36-run 保持 sealed。
