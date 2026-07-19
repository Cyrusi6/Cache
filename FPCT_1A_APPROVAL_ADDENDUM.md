# FPCT-1A-R Human Decision Lock and Prospective Amendment

> Stage：FPCT-1A-R
> 状态：`GO`
> FPCT-1B：`NOT AUTHORIZED`
> Source commit：`7207aafffc7f72976473815bc11102f8b06dddc1`
> Branch：`research/fpct-factorized-transport`
> 日期：2026-07-19（Asia/Shanghai）

## 1. 修订权限、时间边界与 operative protocol

本文件记录用户在任何 FPCT 自然 ambiguity audit、逐样本 alignment 统计、operator output 或 accuracy result 出现前作出的人工决定。它是前瞻性修订，不是基于自然数据的事后改写。

原文件保持为 never-executed v1 历史记录：

- `FPCT_1A_AMBIGUITY_PROTOCOL.md`；
- `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest.json`。

v1 在任何自然数据执行前被 supersede。当前 operative protocol 由以下三部分共同组成：

1. v1 中未被本 addendum 明确替换的 input、pair、split、alignment、candidate legality、zero-support、provenance 和 serialization contracts；
2. 本 `FPCT_1A_APPROVAL_ADDENDUM.md`；
3. `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest_v2.json`。

冲突优先级为：v2 manifest 与本 addendum共同覆盖 v1；`FPCT_PREREGISTRATION.md` 作为总体 claim boundary 继续有效。`math.md` 仍是 non-normative reference，本阶段不得修改。

## 2. 保持不变的 FPCT 核心边界

- `F-C_post` 仍是 query-time factorization preservation 的唯一 headline contrast；
- `C_post-C_pre` 只识别 candidate-specific nonlinear fusion；
- `F-C_pre` 只表示整体系统差异；
- 第一轮继续固定 `a(x)=1`、`g=1`、`position_mode=legacy`；
- receiver-native null、de-RoPE/re-RoPE、Phase 2A selector、新 gate 和新 entropy/confidence 路径继续排除；
- sender/receiver 冻结；正式 task evidence 仍要求 matched retraining；
- FPCT-1B 只允许 label-free structural-support audit，不是 accuracy test，也不是 scientific power kill-test。

## 3. 人工批准的 structural-support definitions

对 eligible receiver parent `i`，先按 v1 legality contract mask invalid candidates、对剩余有限正质量重新 L1 normalization，并定义：

\[
m_i=\sum_j\mathbf 1[A_{ij}>0],
\qquad
N_{\mathrm{eff}}(i)=\frac{1}{\sum_jA_{ij}^2},
\qquad
S_i=1-\max_jA_{ij}.
\]

### 3.1 Primary structural support

唯一 primary structural-support threshold 冻结为：

`primary_structural_m2: m_i>=2`。

当前 frozen alignment 为 `uniform + top_k=4`，因此精确等价于：

\[
S_i\ge\frac12,
\qquad
N_{\mathrm{eff}}(i)\ge2.
\]

整数 `m_i` 是当前 uniform contract 下的分类依据；float `S/N_eff` 只用于 identity/integrity 检查。

### 3.2 Secondary sensitivity/enrichment

- `high_cardinality_m3: m_i>=3`；uniform 下等价于 `S_i>=2/3`、`N_eff>=3`；
- `strict_m4: m_i=4`；uniform 下等价于 `S_i=3/4`、`N_eff=4`。

两者只能用于 sensitivity、enrichment 和后续机制分层：

- 不得否决 pair；
- 不得改变 pilot ranking；
- 不得替代 headline direct structural-support ceiling；
- 不得在结果出现后升级为 primary。

## 4. Exact control、headline indicator 与互斥 strata

Sample-level headline structural-support indicator：

\[
D_s
=
\mathbf 1[\exists\text{ eligible parent }i:m_i\ge2].
\]

Pair/task/split 的 direct structural-support ceiling：

\[
C^{struct}_{ptr}
=
\frac{\sum_{s\in\mathcal S_{ptr}}D_s}
{|\mathcal S_{ptr}|}.
\]

`high_cardinality_support_ceiling` 单独定义为样本中至少存在一个 `m>=3` parent 的比例；它是 secondary ceiling，不能称为 headline direct-effect ceiling。

Exact natural control 冻结为：样本中所有 eligible parents 均满足 `m<=1`。只有该 stratum 才能在相同权重、mask、position 和共同 fallback 下要求 fixed-state `F=C_post`。

`m=2` 是 mechanism-positive low-cardinality stratum，不是 negative control，也不得使用 `below-gate negative control` 等命名。

Parent-level 必须输出五个互斥 strata：

- `m0_zero_support`：`m=0`；
- `m1_one_to_one`：`m=1`；
- `m2_low_cardinality`：`m=2`；
- `m3_high_cardinality`：`m=3`；
- `m4_strict`：`m=4`。

每个 eligible parent 恰好进入一个 stratum。Top-k 固定为 4；出现 `m>4` 是 config/integrity failure。

Zero-support contract 完全保留：独立报告，`A_max/S/N_eff=null`，不填 epsilon、伪 candidate、nearest neighbor 或 slot 0，不归入 one-to-one，不用于 mechanism claim，并保留在 eligible-parent/sample denominator。

## 5. FPCT-1B 的统计用途边界

FPCT-1B 的目标仅是确认冻结输入中是否存在结构性一对多机会以及其工程规模。Candidate count 不能证明 query-time separability、实际 posterior activation、task benefit 或 FPCT 数学正确性。

Selection/description unit 保持为 distinct content group。Pair eligibility 和 pilot ranking 只使用 v1 冻结的 `fit+calibration` label-free groups。三任务等权 task-macro 只用于描述和 ranking，不提供 power guarantee。

对每个 pair×task 至少报告：

- `observed_positive_group_count`：至少一个 parent 满足 `primary_structural_m2` 的 group 数；
- `observed_total_group_count`；
- `observed_support_rate`；
- ordinary two-sided 95% Wilson interval `[wilson95_low,wilson95_high]`。

普通 Wilson interval 使用 `z=z_0.975`：

\[
\frac{
\hat p+\frac{z^2}{2n}
\pm z\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}}
}{1+\frac{z^2}{n}}.
\]

9-cell Bonferroni simultaneous Wilson LCB 保留为 mandatory sensitivity output：familywise confidence 固定为 95%，每个 one-sided cell 使用 error `0.05/9`。但该 LCB：

- 不得参与 readiness status；
- 不得参与 pair eligibility；
- 不得参与 pilot ranking；
- 不得解释为 power guarantee。

## 6. Formal effect/power analysis deferred

FPCT-1A-R 不锁定、也不使用以下量：

- `delta_pos`；
- `delta_direct_all`；
- `n_req`；
- paired-discordance power gate；
- matched-training seed-variance gate。

原因是当前尚无 FPCT operator 的 paired discordance、effect estimate 或 matched-retraining seed variance。上述分析只有在后续阶段获得相应证据、且在正式 accuracy evaluation 前另行前瞻批准后才能建立。

因此 v1 中基于 delta、harmonic effective sample size、projected count 或 `n_req` 的 eligibility/kill gate 全部不再 operative。

## 7. Engineering readiness states

下列阈值是人工批准的工程 readiness heuristic，不是 scientific power guarantee。

对 heterogeneous pair `p`，定义：

- 每个 task 的 primary-positive distinct group count 均至少 30；
- 三任务 pooled primary-positive distinct group count 至少 100。

同时满足两项则 `pair_pilot_ready=true`。

全局 readiness 状态互斥且优先级固定：

1. `NO_SUPPORT`：所有 heterogeneous pairs 的 pooled `m>=2` positive group 总数为 0；
2. `DIAGNOSTIC_ONLY`：存在至少一个 positive group，但 0 个 heterogeneous pair 达到工程门槛；
3. `SINGLE_PAIR_PILOT_READY`：恰有 1 个 heterogeneous pair 达到工程门槛；
4. `CROSS_PAIR_PILOT_READY`：至少 2 个 heterogeneous pairs 达到工程门槛。

这些状态只描述现有数据的工程可用性。它们不检验 accuracy effect，不证明 query-time separability，也不授权 FPCT-1B、operator 实现、GPU 或训练。

## 8. Pilot ranking

只对 `pair_pilot_ready=true` 的 heterogeneous pairs 排名，按以下未舍入值依次比较：

1. 最弱任务 `observed_positive_group_count` descending；
2. 三任务等权 `task_macro_support_rate` descending；
3. `pooled_positive_group_count` descending；
4. canonical pair ID lexicographic ascending。

Same-tokenizer control 永不参与 ranking，不能替代 heterogeneous pilot。`m>=3`、`m=4`、Wilson interval、Bonferroni LCB、accuracy、correctness、beneficial/harmful、Phase 2A outcome、token-micro rate 或 posterior/gate statistic 均不得用于 ranking/tie-break。

## 9. 后续 mechanism diagnostics

以下指标预注册为后续 operator pilot 的 mechanism diagnostics，但 FPCT-1A-R/1B 不计算：

- `D_K`：candidate key dispersion；
- `D_V`：candidate value dispersion；
- candidate-logit variance；
- Jensen gap `logsumexp_A(u)-E_A[u]`。

它们用于区分“存在多个 candidate”与“query 确实可分离 candidate”。未经 operator 实现和相应阶段授权，不得提前计算或用于当前 readiness selection。

## 10. 状态、授权与历史 commit 边界

- FPCT-1A-R：`GO`；只表示人工决定与前瞻性协议修订已锁定；
- FPCT-1B：`NOT AUTHORIZED`；
- FPCT-1C、FPCT-2 及以后：`NOT AUTHORIZED`。

v1 中 `commit=false`、`push=false` 描述的是当时 FPCT-1A 执行边界，不是历史错误。随后用户单独授权 commit/push，形成并推送了 source commit `7207aafffc7f72976473815bc11102f8b06dddc1` 到独立 branch。该授权不追溯扩大 v1 的 audit/model/GPU/training 权限。

本修订若验证通过，只授权在 `research/fpct-factorized-transport` 上 commit 并 push 到同名远端 branch；不创建 PR、不合并 `main`、不 rebase。
