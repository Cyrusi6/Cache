# FPCT 预注册：Query-time Factorization-Preserving Cache Transport

> 阶段：FPCT-0——研究线隔离与预注册
> 预注册日期：2026-07-19（Asia/Shanghai）
> 固定基线：`9fa1f0ac3bedefd282961a853278ab88fb376fa2`
> 分支：`research/fpct-factorized-transport`
> 独立 worktree：`/home/lijunsi/projects/Cache-fpct-factorized-transport`
> 时间边界：本研究线在 Phase 2A-1 结果揭晓前启动；FPCT-0 未查看或使用 Phase 2A-1 未公开结果。

## 1. 预注册状态与不可回改边界

本文件冻结 FPCT 的核心科学问题、三个 operator、因果对比、第一轮方法边界、证据层级、阶段门槛和 claim boundary。以下内容不得根据 Phase 2A-1 后续结果或 FPCT 自然数据结果回头修改：

- primary hypothesis；
- `F - C_post` 是 headline 主效应；
- `C_post - C_pre` 与 `F - C_pre` 的因果解释；
- 第一轮 `a(x)=1`、`g=1`、`position_mode=legacy` 等固定条件；
- fixed-checkpoint 诊断不能替代 matched-training 证据；
- 数学性质、机制激活和 task accuracy 改善必须分别判定。

如后续发现定义缺口，只能在查看对应自然结果前新增带日期、理由和 commit 的前瞻性 addendum；不得覆盖原文，不得借 addendum 改写 primary hypothesis。缺乏客观依据的数值阈值统一标记为“需人工批准后才能锁定”，且必须在查看相关自然数据结果前批准。

FPCT-1A-R 于 source commit `7207aafffc7f72976473815bc11102f8b06dddc1` 之后、任何自然 structural-support audit 之前完成用户人工批准。当前 FPCT-1A operative protocol 由 never-executed v1、`FPCT_1A_APPROVAL_ADDENDUM.md` 和 `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest_v2.json` 共同组成；冲突时 addendum/v2 覆盖 v1，但不改变本文件的 primary hypothesis、三个 operator 或 `F-C_post` headline contrast。

根目录 `math.md` 是用户提供的相关公式讨论稿，作为 non-normative reference 保留，本任务不修改。它描述了包含 de-RoPE/re-RoPE、receiver-native null、sidecar 和后续 `g` 扩展的更一般构想；与本文件第一轮冻结项或 FPCT-0 至 FPCT-11 gate 冲突时，一律以本预注册为准。该讨论稿本身不构成 FPCT-1 授权。

## 2. 核心科学问题与可证伪假设

### 2.1 核心问题

当 tokenizer mismatch 使 receiver memory slot `i` 对应多个 source memory candidates `j` 时，是否应保留 candidate factorization，直到 receiver query `q_t` 出现后再完成归一化，而不是在 attention 前把候选平均成单一 KV slot？

### 2.2 Primary hypothesis

在 sender、receiver、alignment、top-k、fuser 容量、训练数据、初始化、数据顺序、seed 和训练预算全部 matched 的条件下，`F` 相对 `C_post` 会在预先定义的真实一对多 ambiguity population 上改善 primary task endpoint；该改善来自 query-conditioned candidate posterior，而不是 candidate-specific nonlinear fusion 本身。

这是可证伪假设。以下任一情况都不能支持 headline hypothesis：

- 完整 matched-training 证据中，`F - C_post` 未达到预先批准的 primary decision rule；
- `F` 的差异只出现在 duplicate/identical-candidate 数值负对照容差内；
- task accuracy 有变化，但没有证据表明 query posterior 在真实一对多数据中被激活；
- 只观察到 `C_post - C_pre`，但 `F - C_post` 不成立；
- 只用 fixed-checkpoint、旧 B2/B3/B6 checkpoint 或 unmatched checkpoint 得到差异；
- 改善依赖第一轮明确禁止的 outer selector、receiver-native null 或 position rewrite。

### 2.3 Secondary hypothesis

`C_post - C_pre` 测量 candidate-specific nonlinear fusion 的作用。它可以独立成立或不成立，但不能代替 `F - C_post` 支持 query-time factorization preservation。

### 2.4 Claim boundary

- `F - C_post`：唯一可用于归因 query-time factorization preservation 的 headline 主效应。
- `C_post - C_pre`：只归因于“先逐 candidate 共享非线性融合、再平均”相对“先平均、再融合”的差异。
- `F - C_pre`：整体系统差异，混合了 nonlinear fusion placement 与 query-time factorization，不能单独用于机制归因。
- 同 checkpoint 数值性质成立，不等价于机制在真实数据中被激活。
- 机制被激活，不等价于 task accuracy 改善。
- task accuracy 改善但缺少 `F - C_post` 和机制证据时，只能报告经验系统差异，不能报告 FPCT 机制成立。
- 本预注册不主张效率、延迟、跨任务普适性或跨模型普适性；这些需要各自的预注册和证据。

## 3. 记号与共同条件

以下公式按 layer/head 独立成立；为简洁省略 layer/head 下标。

- `t`：receiver query 时刻，query 为 `q_t ∈ R^d`。
- `i`：receiver memory parent slot。
- `j ∈ J_i`：与 parent `i` 对齐的 source candidate。
- `A_ij ≥ 0`：固定 alignment mass。对每个至少有一个合法 candidate 的 `i`，`Σ_{j∈J_i} A_ij = 1`。
- `K^S_j, V^S_j`：source candidate KV。
- `K^R_i, V^R_i`：冻结的 receiver-native/base KV。
- `z_i`：parent `i` 的、对三个 operator 完全相同且不依赖当前 query 的其他 shared conditioning。
- `Φ_θ = (Φ^K_θ, Φ^V_θ)`：共享 fuser。`C_post` 与 `F` 必须逐 candidate 使用完全相同的 `Φ_θ`；三臂参数容量必须 matched，`F` 不得增加 candidate router。
- `L_t`：query `t` 可见、且具有合法正质量 source support 的 transported parent 集合。
- `N_t`：query `t` 可见、但不与 candidate-bearing transported parent 并列的普通 receiver-native slots，例如按冻结基线路径保留的 generated/template/no-support positions；三臂完全相同。
- `m_ti`：共同 parent mask；合法时为 `0`，非法时为 `-∞`。
- `b_ti`：三个 operator 共享且保持不变的其他 attention logit 项。
- `d`：attention head dimension。

所有 invalid candidate 均视为 `A_ij=0`，并约定 `log 0=-∞`；实现时必须先 mask 再进入安全的 log-softmax 路径，不能对未经保护的零值直接取 log。每行 `A_i` 的归一化是 operator contract，避免 candidate 数量意外改变 parent prior。第一轮沿用 `position_mode=legacy`；entropy、confidence、legacy gate 及其位置均保持不变，因此不在下面的对比公式中重复展开。

第一轮 `g=1/no null` 的含义是：对每个 candidate-bearing parent `i∈L_t`，不再并列一个与 candidates 竞争的 receiver-native sibling atom；它不删除 `N_t` 中原本就存在的普通 native slots。若固定基线允许零-support parent，其既有 native fallback 必须在三臂中保持完全相同并在 FPCT-1C tensor contract 中显式冻结，不能把 fallback 改造成 candidate-bearing parent 的新 null atom。

对任意 `u∈N_t`，共同 native score 定义为：

\[
s^R_{tu}
=
\frac{q_t^\top K^R_u}{\sqrt d}+b^R_{tu}+m^R_{tu}.
\]

其中 `b^R_tu` 与 `m^R_tu` 是冻结 receiver 路径原有的 bias 与 causal/padding mask，三臂逐元素相同。

### 3.1 Frozen nuisance placement

固定基线 commit 中 `C_pre` 的调用图是 nuisance placement 的唯一参照：

- `position_mode=legacy`、entropy、confidence、legacy gate 的输入、输出、参数、应用顺序以及相对 fuser/residual/attention 的位置全部冻结。
- 任何在 `C_pre` 中按 parent 计算一次的量，在 `C_post`/`F` 中也只能按 parent 计算一次并原样 broadcast 到 candidates；不得逐 candidate 重算、重归一化或增加 candidate-specific gate。
- candidate expansion 只改变本文件明确写出的 fusion/marginalization 位置，不得顺带改变 receiver-native residual、legacy mask 或其他 attention bias。
- 若 FPCT-1C/2 的 tensor-contract 审计无法把固定基线调用图无歧义地映射到三个 operator，则停止，不得用新 nuisance placement 补齐。

## 4. 三个冻结 operator

### 4.1 `C_pre`：先平均 source KV，再进入 fuser

先在 candidate 轴上平均 source KV：

\[
\bar K^S_i = \sum_{j\in J_i} A_{ij}K^S_j,
\qquad
\bar V^S_i = \sum_{j\in J_i} A_{ij}V^S_j.
\]

再通过单次 fuser：

\[
(K^{pre}_i,V^{pre}_i)
=
\Phi_\theta(\bar K^S_i,\bar V^S_i;K^R_i,V^R_i,z_i).
\]

形成单 transported slot，并与共同 native-only slots 做同一次 global normalization：

\[
s^{pre}_{ti}
=
\frac{q_t^\top K^{pre}_i}{\sqrt d}+b_{ti}+m_{ti},
\qquad
\alpha^{pre}_{ti}
=
\frac{\exp s^{pre}_{ti}}
{D^{pre}_t},
\qquad
D^{pre}_t
=
\sum_{u\in N_t}e^{s^R_{tu}}
+\sum_{r\in L_t}e^{s^{pre}_{tr}},
\]

\[
o^{pre}_t
=
\sum_{u\in N_t}\frac{e^{s^R_{tu}}}{D^{pre}_t}V^R_u
+\sum_{i\in L_t}\alpha^{pre}_{ti}V^{pre}_i.
\]

这就是当前 v2.2 路径的抽象定义：candidate factorization 在 fuser 前消失。

### 4.2 `C_post`：逐 candidate 共享融合，再在 attention 前平均

每个 source candidate 独立通过同一个共享 fuser：

\[
(K^f_{ij},V^f_{ij})
=
\Phi_\theta(K^S_j,V^S_j;K^R_i,V^R_i,z_i)
\qquad \forall j\in J_i.
\]

然后仍在 attention 前按 `A_ij` 压成单 slot：

\[
K^{post}_i
=
\sum_{j\in J_i}A_{ij}K^f_{ij},
\qquad
V^{post}_i
=
\sum_{j\in J_i}A_{ij}V^f_{ij}.
\]

单 transported slot attention 与共同 native-only slots 的 global normalization 为：

\[
s^{post}_{ti}
=
\frac{q_t^\top K^{post}_i}{\sqrt d}+b_{ti}+m_{ti},
\qquad
\alpha^{post}_{ti}
=
\frac{\exp s^{post}_{ti}}
{D^{post}_t},
\qquad
D^{post}_t
=
\sum_{u\in N_t}e^{s^R_{tu}}
+\sum_{r\in L_t}e^{s^{post}_{tr}},
\]

\[
o^{post}_t
=
\sum_{u\in N_t}\frac{e^{s^R_{tu}}}{D^{post}_t}V^R_u
+\sum_{i\in L_t}\alpha^{post}_{ti}V^{post}_i.
\]

`C_post` 保留 candidate-specific nonlinear fusion，但在 query attention 前仍消除 candidate 轴。

### 4.3 `F`：保留 factorization 到 receiver query 后

`F` 使用与 `C_post` 完全相同的 candidate-specific fuser 输出：

\[
(K^f_{ij},V^f_{ij})
=
\Phi_\theta(K^S_j,V^S_j;K^R_i,V^R_i,z_i).
\]

不在 attention 前平均。对全部合法 `(i,j)` 加入 alignment prior `log A_ij`，执行一次 global attention softmax：

\[
s^F_{tij}
=
\frac{q_t^\top K^f_{ij}}{\sqrt d}
+b_{ti}+m_{ti}+\log A_{ij},
\]

\[
\alpha_{tij}
=
\frac{\exp s^F_{tij}}
{D^F_t},
\qquad
D^F_t
=
\sum_{u\in N_t}e^{s^R_{tu}}
+\sum_{r\in L_t}\sum_{\ell\in J_r}\exp s^F_{tr\ell},
\]

\[
\alpha^R_{tu}=\frac{e^{s^R_{tu}}}{D^F_t},
\qquad
o^F_t
=
\sum_{u\in N_t}\alpha^R_{tu}V^R_u
+\sum_{i\in L_t}\sum_{j\in J_i}\alpha_{tij}V^f_{ij}.
\]

`α_tij` 是全部合法 `(i,j)` 上的 global attention mass。对合法 parent `i∈L_t`，定义 query-conditioned source-candidate posterior：

\[
\gamma_t(j\mid i)
=
\frac{\alpha_{tij}}
{\sum_{\ell\in J_i}\alpha_{ti\ell}}.
\]

`A_ij` 是 query 出现前的 alignment prior，`γ_t(j|i)` 是 query 出现后的 source-candidate posterior；两者不是同一对象。`A_ij` 只通过 `log A_ij` 进入一次，不能在 softmax 后再次乘到 value 上。

## 5. 三个因果对比

| 对比 | 冻结的设计轴/归因目标 | 不允许的解释 |
|---|---|---|
| `C_post - C_pre` | candidate-specific nonlinear fusion 的作用 | query-time factorization preservation |
| `F - C_post` | query-time factorization preservation 的 headline 主效应 | outer selection、position rewrite 或额外 null path 的作用 |
| `F - C_pre` | 整体系统差异 | 单独用于任何一个机制归因 |

所有正式性能对比都必须来自 matched retraining。若三臂没有形成同一 matched group，差异只能标记为 diagnostic。

若 `Φ_θ` 对 source KV 为仿射、其他输入固定且数值路径相同，则 `C_post` 与 `C_pre` 应等价；因此 `C_post-C_pre` 的可解释来源正是 candidate-specific nonlinear placement，而不是 candidate axis 本身。

必须区分两种 estimand：

- matched retraining 的 task contrast 估计“采用该 operator family 并按同一训练协议适配”后的总效应；不同 arm 的最终 `θ` 与优化轨迹可以不同。
- 同 `θ`、同 query/state 的 fixed-state operator intervention 才隔离即时计算位置效应，但只属于 mechanistic diagnostic，不能代替正式 task evidence。

因此 headline 归因要求 matched-trained `F-C_post`、同状态 operator diagnostic、真实 `γ_t(j|i)` activation 和 controls 共同闭环；不能仅凭任一单项声称纯计算机制。

## 6. 第一轮方法固定项

- `a(x)=1`：不使用 Phase 2A outer selector，不做 event/example-level 路由。
- `g=1`：transport family 获得全部 attention prior，不加入 receiver-native null。
- `position_mode=legacy`：不加入 de-RoPE/re-RoPE 或其他 position rewrite。
- 不改变 entropy、confidence 和 legacy gate 的定义、参数或调用位置。
- top-k 固定；alignment `A_ij` 的生成、归一化和版本固定。
- sender 与 receiver 权重冻结。
- fuser 可以按 matched recipe 训练；三臂必须使用 matched retraining。
- 旧 B2/B3/B6 checkpoint 只允许用于 diagnostic 或 initialization，不得进入正式 matched-training aggregate。
- 旧 B6 的 `+8.24pp` 不能被解释为 FPCT headroom、effect-size prior、power estimate 或成功阈值。

### 6.1 `a(x)`、`g`、`ρ`、`γ` 与 `α` 的区别

- `a(x)`：attention 外部的样本/事件级 selector，决定是否启用 transport。第一轮固定为 `1`。
- `g_i^h`：attention 内部、query posterior 形成前的 transport-family vs receiver-native prior mass。第一轮固定为 `1`，且不实现 receiver-native null。
- `ρ_ti^h`：未来引入 receiver-native component 后，给定 query 的 transport-family vs native posterior；它不是 Phase 2A selector，也不是 alignment prior。
- `γ_t(j|i)`：给定 query、parent 且已在 source family 内时的 source-candidate posterior；这是第一轮 factorization activation 的直接对象。
- `α_tij`：第一轮在全部合法 `(i,j)` 上 global softmax 后的最终 attention mass。

未来 general form 中，令 `u^R_ti` 为 native score，`Z^S_ti=Σ_j A_ij exp(u_tij)`，则仅在显式 native component 已获授权时定义：

\[
\rho_{ti}
=
\frac{g_i Z^S_{ti}}
{(1-g_i)e^{u^R_{ti}}+g_iZ^S_{ti}}.
\]

第一轮 `g=1` 时该 branch posterior 对合法 source parent 退化为 `ρ=1`，所以 headline activation 不能用退化的 `ρ` 衡量，而必须用 `γ_t(j|i)` 及 global `α_tij`。端点 `g∈{0,1}` 必须用 mask 实现，不能直接计算 `logit(g)` 或无保护的 `log 0`。

若未来再组合 outer selector 和 support indicator，必须显式写成 `g_eff(x,i,h)=a(x) r_i g_i^h`，其中 `r_i=1` 表示 candidate-bearing parent。第一轮对 `i∈L_t` 固定 `a=r=g=1`；`N_t` 中普通 native slots 不使用这个 branch prior。

不得用 `a(x)` 代替 `g`，不得把 `A_ij` 或 `g` 称为 query posterior，不得把 `ρ` 解释成 Phase 2A selector，也不得把 source-candidate posterior `γ` 与 native/source posterior `ρ` 混用。

## 7. Endpoints

### 7.1 Primary endpoint

Primary estimand 是 matched-training 后，在预先冻结的 ambiguity-positive evaluation population 上，canonical end-task accuracy 的 paired difference：

\[
\Delta_{primary}
=
Y^{ambiguity}_{task}(F)
-Y^{ambiguity}_{task}(C_{post}).
\]

Ambiguity-positive eligibility 必须在 operator 输出和 task outcome 之前，由冻结的 input、causal mask 与 alignment manifest 决定：预声明的 evaluation query-parent 对中，至少一个 query-visible parent `i` 的合法正质量 candidate 数满足 `|{j:A_ij>0}| ≥ 2`。不得用实际 attention mass、posterior `γ/ρ`、“是否被模型使用”或 correctness 来选择 primary population。具体 task suite、样本聚合单位、多个 eligible parent 的样本归类规则和统计 decision rule 尚无客观依据，必须在查看 FPCT 自然结果前由人工批准并写入 addendum/manifest。

FPCT-1A-R 已人工批准 `primary_structural_m2: m>=2` 作为 label-free structural support 与正式 matched-training ambiguity population 的统一结构 eligibility。`high_cardinality_m3` 与 `strict_m4` 只作 sensitivity/enrichment，不得否决 pair、改变 pilot ranking 或替代 headline structural-support ceiling。

只有 `F - C_post` 达到预先批准的 primary rule，且数值正确性、真实机制激活与控制项均通过时，才允许支持 headline hypothesis。

### 7.2 Secondary endpoints

- 全 evaluation population 的 canonical task accuracy：`F - C_post`、`C_post - C_pre`、`F - C_pre`。
- 按 task、model pair、seed、candidate count 和 alignment ambiguity 分层的 paired effect。
- one-to-one 与 same-tokenizer control 上的 paired effect。
- 预先冻结的 token-level NLL、answer log-probability 或 margin；具体选项需人工批准后锁定。
- 训练稳定性、峰值显存、推理延迟和 cache size，只作为 secondary engineering endpoint；不能反向改变 primary claim。

### 7.3 Mechanistic endpoints

- Candidate posterior activation：`γ_t(j|i)=α_tij/Σ_l α_til` 相对 prior `A_ij` 的 query-conditioned 改变；不得把 parent-level global mass 改变误报为 within-parent posterior activation。
- Query sensitivity：同一 parent/candidate set 面对不同合法 query 时，`γ_t(j|i)` 是否发生系统性改变。
- Jensen/non-commutativity diagnostic：

\[
J_{ti}
=
\log\sum_j A_{ij}e^{u_{tij}}
-\sum_jA_{ij}u_{tij}
\ge 0,
\]

  只证明 candidate scores 不可交换地参与 softmax；`J_ti>0` 不保证 task accuracy 改善。
- Factorization effect：`o^F_t - o^{post}_t` 及其与 ambiguity/candidate count 的关系。
- Nonlinear fusion residual：

\[
R_i
=
\sum_j A_{ij}\Phi_\theta(K^S_j,V^S_j;K^R_i,V^R_i,z_i)
-\Phi_\theta\!\left(\sum_jA_{ij}K^S_j,\sum_jA_{ij}V^S_j;K^R_i,V^R_i,z_i\right),
\]

  用于解释 `C_post - C_pre`，不得用于解释 `F - C_post`。
- 新 FPCT checkpoint 的 beneficial/harmful event audit：必须重新统计 receiver-native vs `C_pre`、receiver-native vs `C_post`、receiver-native vs `F`，以及 headline `F` vs `C_post` 的 paired correctness/margin transitions；不能沿用旧 B6 event audit，也不能看结果后挑 comparator。
- Causal parent-mask audit：posterior mass 不得流向 query 不可见的 parent 或 invalid candidate。

具体 activation effect threshold、分层 cutoff 和 event 判定 margin 若不能由 duplicate/identical control客观确定，必须人工批准后才能锁定。

## 8. Controls

### 8.1 Ambiguity control

- 主分析至少报告 candidate count、alignment mass distribution 和 alignment entropy 的完整分层或连续关系。
- 不得只挑选 `F` 获益的 ambiguity 子集。
- 除 `|{j:A_ij>0}| ≥ 2` 的结构定义外，任何 entropy/bin cutoff 均需在自然结果前人工批准。

### 8.2 One-to-one control

- 所有合法 parents（或被隔离测试的完整 attention segment）均满足 `K=1` 且 `A_i1=1`。
- 在相同参数、mask、position 和数值路径下，`C_pre=C_post=F` 应退化等价。
- 超出数值负对照容差的系统差异是 correctness failure，不是性能发现。
- 混合序列中只有某一个 parent 为 `K=1` 时，其他 ambiguous parents 仍可改变全局 attention normalization；不得误称该单个 parent 足以保证整段输出相同。

### 8.3 Same-tokenizer control

- 精确负对照固定为 sender/receiver tokenizer 相同、identity alignment `A_ii=1`，且每个 parent `K=1`。
- 该 control 用于检查 FPCT 是否在没有 tokenizer-induced factorization need 时制造差异。
- same-tokenizer 上的异常获益不能单独支持 cross-tokenizer FPCT；异常退化必须单独报告。
- 其他 same-tokenizer “known alignment” 只能作为普通 secondary control，不能替代 identity/K=1 exact control。

### 8.4 Duplicate/identical-candidate numerical negative control

- 把一个 candidate 精确复制为多个相同 `(K,V,z,b,mask)` candidate，并把原 mass `A` 分割为若干非负 mass，其和仍为 `A`。
- 三个 operator 的输出都应与未分割版本保持 refinement invariance；`F` 的 posterior 只在 replicas 间按 prior 比例拆分。
- 该 control 的实际误差 envelope 是优先的数值等价基准。若它不足以确定 tolerance，不得自行添加 multiplier；必须转为人工批准项。

## 9. Matched initialization、data order、seed 与 training budget

每个正式 matched group 必须同时包含 `c_pre`、`c_post`、`f`：

- 同一 base checkpoint 和相同可训练参数初始化；参数映射及 hash 必须记录。
- 相同 model pair、tokenizer、dataset snapshot、split、preprocessing、batch construction 和 example order。
- 相同 seed 列表；seed 必须 paired，不得按 arm 替换失败 seed。
- 相同 optimizer、scheduler、precision、gradient accumulation、global batch、更新步数或训练 token 数。
- 相同 validation cadence 和 checkpoint selection rule。若使用 early stopping，规则必须预先固定且三臂相同；否则使用相同固定预算。
- 相同 top-k、`A_ij`、position mode、legacy gate、entropy 和 confidence 设置。
- 相同资源等级；训练预算按 optimizer steps/训练 token/effective batch 匹配，不能用 wall-clock 代替。不得给某一 arm 额外调参或补训。
- 只聚合完整 matched triplet。缺失、失败或污染的 triplet 整体标记为 incomplete，不得只删除不利 arm。

下列项目尚需人工批准后锁定：seed 数与具体 seed、训练 budget、task suite、统计模型、置信区间/显著性或 practical-effect rule、失败重跑规则。它们必须在查看对应自然结果前写入不可变 manifest。

## 10. Fixed-checkpoint 诊断与 matched-training 证据

### 10.1 Fixed-checkpoint 只允许回答

- operator 是否按公式执行；
- invariants、mask、shape、precision 和 stability 是否正确；
- `γ` 是否可能随 query 改变；未来 native-null 阶段再检查 `ρ`；
- 一对多 candidate 是否在实现中真实存在；
- implementation 是否值得进入 matched-training。

Fixed-checkpoint 结果必须标记 `evidence_class=diagnostic`，不能进入正式 task accuracy aggregate。

### 10.2 Formal evidence 必须满足

- 新 FPCT matched checkpoints；
- 完整 matched group；
- 冻结后的 recipe、data order、seed、budget 和 evaluation manifest；
- `F - C_post` 作为 primary contrast；
- controls 与 mechanistic audit 同时报告。

旧 B2/B3/B6 只可标记 `diagnostic` 或 `initialization_only`。旧 B6 的 `+8.24pp` 不能作为 FPCT headroom。新 FPCT checkpoint 完成后，必须从头重新审计 beneficial/harmful events。

## 11. 数值正确性不变量

### 11.1 Flat / hierarchical softmax 等价

令不含 `log A_ij` 的 candidate score 为：

\[
u_{tij}=\frac{q_t^\top K^f_{ij}}{\sqrt d}+b_{ti}+m_{ti}.
\]

Flat posterior 为：

\[
\mathcal D_t
=
\sum_{u\in N_t}e^{s^R_{tu}}
+\sum_r\sum_l A_{rl}e^{u_{trl}},
\qquad
\alpha^{flat}_{tij}
=
\frac{A_{ij}e^{u_{tij}}}
{\mathcal D_t}.
\]

只对合法 parent `i∈L_t` 定义 hierarchical conditional posterior：

\[
Z_{ti}=\sum_jA_{ij}e^{u_{tij}},
\qquad
\pi_{ti}=\frac{Z_{ti}}{\mathcal D_t},
\qquad
\gamma_t(j\mid i)=\frac{A_{ij}e^{u_{tij}}}{Z_{ti}}.
\]

必须满足：

\[
\alpha^{flat}_{tij}=\pi_{ti}\gamma_t(j\mid i).
\]

同时对 `u∈N_t` 有 `α^R_tu=e^{s^R_tu}/\mathcal D_t`。非法 parent 不计算 conditional posterior，直接令其全部 `α_tij=0`，避免 `0/0`。实现需用稳定 `logsumexp`；两种实现的差异不得超过预先批准的数值容差。

### 11.2 `K=1` 退化

若全部合法 parent 只有一个 candidate 且 `A_i1=1`，则：

\[
C_{pre}=C_{post}=F
\]

在相同 fuser 参数、mask、position 和 precision 下应成立。

### 11.3 Refinement invariance

若 candidate `j` 被复制成一组完全相同的 replicas `r`，且 `Σ_r A_ijr=A_ij`，则三个 operator 的 output 和 downstream logits均应与复制前等价；`F` 的 posterior 只允许在 replicas 间按 prior 比例重新分配。

### 11.4 Causal parent mask

若 parent `i∉L_t`，则其全部 child candidates 必须有 `α_tij=0`。child 不得绕过 parent causal mask。

### 11.5 Invalid candidate 无质量

padding、越界、被 top-k/mask 排除或 alignment mass 为零的 candidate 必须满足 `α_tij=0`，且不能通过 NaN、renormalization 或 fallback 获得质量；改变其 KV 不得影响输出，其梯度也应为零。

### 11.6 后续 `g=0` exact receiver recovery

当未来引入 receiver-native null/path 时，transport family logits 可加 `log g`，receiver-native path 加 `log(1-g)`。必须保证：

\[
g=0 \Longrightarrow \text{transport mass}=0
\Longrightarrow \text{output exactly equals frozen receiver-native path}.
\]

该性质是后续扩展的 correctness gate，不授权第一轮加入 receiver-native null；第一轮仍固定 `g=1`。

## 12. 三层证据必须分开

| 证据层 | 最多允许的结论 | 不能推出 |
|---|---|---|
| 数学/数值性质成立 | operator 实现与预注册公式一致，invariants 通过 | 真实数据会使用 factorization；task accuracy 会提升 |
| 机制在真实数据中被激活 | 第一轮 `γ` 随 query 和 ambiguity 系统变化，且 mask/controls 正确；未来 native-null 阶段另审计 `ρ` | task accuracy 会提升 |
| Task accuracy 改善 | 在 matched-training 与冻结评测中达到 decision rule | 若缺 `F-C_post` 与 mechanism evidence，不能归因 FPCT |

Headline claim 需要三层证据形成闭环；任何一层缺失都必须缩小 claim。

## 13. 阈值政策

- 不凭经验随意填写 effect size、accuracy delta、posterior shift、entropy cutoff、seed 数或训练预算。
- 数值等价优先以 duplicate/identical-candidate control 的同 dtype、同 device、同 operation-order 误差 envelope 为基准。
- 若 duplicate envelope 不能覆盖所需 test matrix，新增 tolerance 必须人工批准，且批准发生在自然数据结果之前。
- task accuracy 的统计 rule、practical-effect threshold、置信水平、multiplicity handling 和 futility rule 均为“需人工批准后才能锁定”。
- 任何未提前锁定的阈值都不能在看到自然数据后补写；对应结果只能判为 `INCONCLUSIVE`，不能判 `GO`。

## 14. 阶段、依赖、资源等级与门槛

### 14.1 资源等级

| 等级 | 含义 |
|---|---|
| `R0` | 环境/版本只读检查、worktree 隔离、文档和静态路径检查；禁止 model forward |
| `R1` | CPU-only 实现、tiny-tensor oracle、单元测试、静态/数值检查 |
| `R2` | 单 GPU fixed-checkpoint diagnostic；不产生正式性能证据 |
| `R3` | 有限多 GPU matched pipeline smoke/pilot；只按预注册用途解释 |
| `R4` | 多 GPU 正式 matched training 与冻结评测 |
| `R5` | 独立 replication、机制审计和最终证据收口；按需要使用多 GPU |

### 14.2 FPCT-0 至 FPCT-11

| 阶段 | 内容 | 依赖 | 资源 | GO | INCONCLUSIVE | NO-GO / 停止 |
|---|---|---|---|---|---|---|
| FPCT-0 | 研究线隔离、预注册、claim boundary | 固定 base SHA | R0 | 隔离安全、文档与检查完成 | 人工决策项未列全或路径状态不明 | branch/path 冲突、影响 Phase 2A-1、无法冻结核心问题 |
| FPCT-1A | v1 ambiguity protocol locking（never executed） | FPCT-0 GO | R0 | 历史协议已冻结 | 人工决定当时尚未锁定 | 已由 FPCT-1A-R 在自然数据前 supersede |
| FPCT-1A-R | Human decision lock 与 prospective amendment | FPCT-1A v1 + source commit `7207aaff...` | R0 | addendum/v2/schema/hash 完整，人工决定锁定 | provenance 或字段仍不一致 | 接触自然结果、回改 primary hypothesis 或污染其他 worktree |
| FPCT-1B | Label-free structural-support audit | FPCT-1A-R GO + 显式授权 | R1 CPU；禁止 forward | audit 完整并输出唯一 engineering readiness state | required cell/hash/schema 不完整 | correctness/integrity failure；`NO_SUPPORT` 是 readiness state，不是数学否证 |
| FPCT-1C | Reference equations、tiny-tensor oracle、synthetic negative-control envelope 与数值 acceptance rule | FPCT-1B decision + 人工授权 | R1 | oracle 覆盖六项不变量，数值 rule 在实现验收前锁定 | tolerance/语义仍待批准 | 公式自相矛盾、无法定义 headline contrast 或无法形成可审计数值规则 |
| FPCT-2 | 隔离实现 `C_post`/`F` 与 manifest plumbing | FPCT-1B `SINGLE_PAIR_PILOT_READY` 或 `CROSS_PAIR_PILOT_READY` + FPCT-1C GO + 人工授权 | R1 | 默认路径不变、operator 与 manifest 可选择 | 实现不完整但无反证 | 必须修改/污染 Phase 2A 路径或无法保持三臂容量 matched |
| FPCT-3 | CPU 数值、mask、gradient 与退化测试 | FPCT-2 GO + FPCT-1C 锁定的数值 rule | R1 | 全部 invariant 在预先批准容差内通过 | nondeterminism 或批准 rule 无法执行 | 持续违反 mask、invalid mass、K=1 或 refinement invariance |
| FPCT-4 | 冻结 task/control/seed/budget/statistical manifest | FPCT-3 GO | R0/R1 | task/statistical rules 在任何自然数据 diagnostic 前批准并 hash；不得回看 FPCT-3 误差重选数值 tolerance | 仍有未批准决策 | 设计无法识别 `F-C_post` 或无法形成 matched triplet |
| FPCT-5 | Fixed-checkpoint diagnostic 与 posterior activation smoke | FPCT-4 GO | R2 | 实现可运行、controls 正确、无 correctness failure | 旧 checkpoint distribution shift 导致 activation 不明 | 数值/mask failure；task 表现本身不构成 NO-GO |
| FPCT-6 | Matched-training pipeline smoke | FPCT-5 GO | R3 | 三臂从 matched init/data order 完成可复现 smoke | 基础设施失败、未形成完整 triplet | 公平 matching 无法实现或某 arm 需特殊预算 |
| FPCT-7 | 与 confirmatory split 隔离的 development-only pilot/feasibility | FPCT-6 GO | R3 | 只检查预声明的 OOM、throughput、determinism、训练稳定性；不得用 pilot accuracy 改 endpoint/threshold/operator recipe | 运行稳定性不足，且只需预批准的基础设施修复 | 必须依据 pilot outcome 后验改阈值、任务、arm recipe 或假设才能继续 |
| FPCT-8 | 正式多 seed matched retraining | FPCT-7 GO | R4 | 全部预注册 matched groups 完成 | 预先允许的重试后仍有 incomplete group | 系统性训练不稳定、预算不可行或 matching 被破坏 |
| FPCT-9 | 执行 FPCT-4 已冻结的 primary/secondary/control evaluation | FPCT-8 GO | R4 | `F-C_post` 达到预批准 primary rule 且 controls 通过 | 精度不足、数据缺失或发现 rule 未锁定 | 完整证据触发预注册 failure/futility rule |
| FPCT-10 | Mechanistic audit、beneficial/harmful re-audit、replication | FPCT-9 GO 或明确授权的机制追查 | R5 | 机制激活、task gain 与 controls 一致，并完成新 checkpoint event audit | task gain 与机制证据不闭环 | 泄漏/causal violation，或机制方向与 headline 归因冲突 |
| FPCT-11 | 证据综合、claim freeze、负/正结果归档 | FPCT-10 决策完成 | R0/R5 | 形成不越界的最终 claim 与可复现 manifest | 仅能形成 mixed/inconclusive report | headline 被否证；归档负结果并停止扩大 claim |

### 14.3 通用停止规则

- 任一阶段 `NO-GO`：停止自动晋级；只有新的、前瞻性人工授权才能开展限定的诊断，不得改写原假设。
- 任一阶段 `INCONCLUSIVE`：不得按 `GO` 解释，不得扩大资源；只有预先允许或人工批准的补充证据才能继续。
- invariant、causal mask、invalid mass、数据泄漏或 matched fairness 失败属于硬停止问题。
- 不得因 GPU 可用而自动进入下一阶段。
- 不得以 Phase 2A-1 结果改变 FPCT stage gate、primary hypothesis 或 headline contrast。

## 15. Artifact、命名、manifest、commit 与 checkpoint 隔离

### 15.1 Canonical slug 与 operator ID

- 研究线 slug：`fpct_factorized_transport`。
- operator ID：`c_pre`、`c_post`、`f`；文档展示名为 `C_pre`、`C_post`、`F`。
- run ID：`<model-pair>__<operator>__seed_<seed>`。
- Kubernetes job（未来若获授权）：必须以 `fpct-` 开头，且不得复用 Phase 2A job name。

### 15.2 Tracked paths

未来只有在对应阶段获授权后才可创建：

- `recipe/train_recipe/fpct_factorized_transport/`
- `recipe/eval_recipe/fpct_<stage>/`（protocol manifests；当前为 `fpct_1a/`）
- `recipe/eval_recipe/fpct_factorized_transport/`
- `script/analysis/fpct_*`
- `test/test_fpct_*`
- 根目录 `FPCT_*.md`

禁止写入或修改任何 `PHASE2A_*`、`phase2a_*` 或 `recipe/**/phase2a_*` 路径。

### 15.3 Untracked artifacts

- 临时产物：`local/tmp/fpct_factorized_transport/`
- checkpoint：`local/checkpoints/fpct_factorized_transport/rev_<execution-sha>/<matched-group>/<model-pair>/<operator>/seed_<seed>/final/`
- 正式结果：`local/final_results/fpct_factorized_transport/rev_<execution-sha>/<matched-group>/<model-pair>/<operator>/seed_<seed>/<dataset>/`

`local/` 不得提交。任何 artifact 目录都不得覆盖既有目录；重跑必须使用新 execution SHA/run ID。

### 15.4 Manifest minimum fields

未来 manifest 至少记录：

- `schema_version`、`suite`、`stage`、`run_id`、`operator`、`evidence_class`；
- `base_commit`、`execution_commit`、`preregistration_sha256`；
- `matched_group_id`、model/tokenizer identifiers 与 revisions；
- `alignment/top_k/position_mode/a/g/legacy_gate/entropy/confidence`；
- initialization、dataset snapshot、split、preprocessing、data-order 和 config hashes；
- seed、training budget、optimizer/scheduler/precision；
- checkpoint path/hash、evaluation manifest hash、output/provenance path；
- `diagnostic`、`initialization_only` 或 `formal_matched_training` 证据分类。

### 15.5 Commit/checkpoint rules

- FPCT 工作只在独立 worktree/branch 中进行；不得在 Phase 2A-1 当前工作树切分支或写 FPCT 文件。
- commit 只包含本次 FPCT 文件；不得带入 Phase 2A-1 未跟踪文件或任何 `local/` 产物。
- FPCT-0 与 FPCT-1A 的 `commit=false/push=false` 描述各自当时的执行边界；随后用户单独授权并推送 source commit `7207aafffc7f72976473815bc11102f8b06dddc1`。该后续授权不是历史错误，也不追溯授权 audit、model forward、GPU 或训练。
- checkpoint 必须绑定 execution commit、preregistration hash 和 matched group；不得原地覆盖。
- 旧 B2/B3/B6 checkpoint 必须显式标记 `diagnostic` 或 `initialization_only`，不得伪装为新 FPCT checkpoint。
- 本研究线 primary hypothesis 的冻结时间早于 Phase 2A-1 结果揭晓；后续 commit 不得根据 Phase 2A-1 结果回改。

## 16. FPCT-0 明确禁止项

FPCT-0 不授权：模型运行代码修改、attention/sidecar 实现、model forward、GPU/Kubernetes、训练、checkpoint 修改、Phase 2A-1 未公开结果访问、FPCT-1、commit 或 push。

FPCT-0 的唯一输出是安全隔离、预注册/状态文档及文档/路径一致性检查。

## 17. FPCT-1A v1 prospective addendum（historical, never executed）

> Addendum date：2026-07-19
> 时间边界：在任何新的自然 ambiguity audit、逐样本 alignment 统计或 FPCT model result 之前

本节记录 source commit `7207aafffc7f72976473815bc11102f8b06dddc1` 中的 v1 协议历史。v1 未运行自然 audit，并已由 FPCT-1A-R 在自然数据前 supersede。未冲突的 pair/input/split/alignment/legality/zero-support/provenance contracts 继续有效。

| Substage | 内容 | 资源 | 当前授权 |
|---|---|---|---|
| FPCT-1A | v1 ambiguity protocol locking | R0，文档/manifest only | historical / superseded before natural data |
| FPCT-1A-R | human decision lock and prospective amendment | R0，文档/manifest only | `GO` |
| FPCT-1B | label-free tokenizer/alignment support audit | R1，CPU audit；禁止 model forward | `NOT AUTHORIZED` |
| FPCT-1C | 原 FPCT-1 reference equations、tiny-tensor oracle 与数值 acceptance rule | R1 | `NOT AUTHORIZED` |

FPCT-2 的依赖前移为 `FPCT-1B SINGLE_PAIR_PILOT_READY 或 CROSS_PAIR_PILOT_READY + FPCT-1C GO + 人工授权`。

### 17.1 Frozen pair/input universe

- Heterogeneous：`tinyllama`、`qwen25_0p5b`、`llama32_1b`；
- Same-tokenizer control：`qwen3_1p7b`，永不成为 primary pilot；
- Tasks：`ai2-arc`、`openbookqa`、`mmlu-redux` canonical evaluation inputs；
- Engineering readiness 与 pilot selection 只使用 `fit+calibration` label-free support；
- model-selection/test、accuracy、correctness、beneficial/harmful 和 Phase 2A outcome 不得参与 pair selection。

### 17.2 Frozen candidate/support definitions

Eligible parent 固定为 `message_mask=true` 且非 receiver padding。Candidate 必须同时满足 valid、source index in range、sender non-padding、finite raw `A`、`A>0`；先 mask 后 L1 renormalize。

- `m=0`：zero-support，独立报告，metrics 为 null，不填伪 candidate；
- `m=1`：one-to-one；
- `m=2`：mechanism-positive low-cardinality；
- `m=3`：high-cardinality sensitivity；
- `m=4`：strict sensitivity。

`A` 只取 raw alignment `source_weights`，不得乘 confidence、entropy、gate 或 selector。

只有所有 eligible parent 均为 `m<=1` 的 sample 才是 exact no-factorization control。`m=2` 是 primary structural-support population 的一部分，不能称为 negative control。

### 17.3 Approved structural thresholds

固定配置 `uniform + top_k=4` 使合法行满足：

\[
A=1/m,\qquad S=1-1/m,\qquad N_{\mathrm{eff}}=m.
\]

人工批准并冻结：

- primary：`primary_structural_m2: m>=2`，uniform 下等价 `S>=1/2`、`N_eff>=2`；
- sensitivity：`high_cardinality_m3: m>=3`；
- sensitivity：`strict_m4: m=4`。

`m>=3` 和 `m=4` 不得否决 pair、改变 pilot ranking 或替代 headline direct structural-support ceiling。Headline indicator 为 `D_s=1[样本存在 m>=2 parent]`。

Distinct content group 是统计单位；三任务等权 task-macro 只用于描述和 ranking。普通 95% Wilson interval 是 pair×task 描述输出；9-cell Bonferroni simultaneous Wilson LCB 只作 sensitivity。`delta_pos`、`delta_direct_all`、`n_req` 和 formal power gate 全部 deferred。

因此 FPCT-1A-R 为 `GO`，FPCT-1B 仍为 `NOT AUTHORIZED`。

### 17.4 FPCT-1B engineering readiness boundary

任一 required pair/task/input/hash/schema 不完整或存在未解决 integrity failure 时为 `INCONCLUSIVE`。完整 audit 只输出以下工程 readiness 状态：

- `NO_SUPPORT`：所有 heterogeneous pairs 的 pooled `m>=2` positive group 总数为 0；
- `DIAGNOSTIC_ONLY`：存在支持，但无 pair 同时达到每 task 30、pooled 100；
- `SINGLE_PAIR_PILOT_READY`：恰有一个 pair 达到工程门槛；
- `CROSS_PAIR_PILOT_READY`：至少两个 pairs 达到工程门槛。

这些状态不是 accuracy test、power guarantee 或 FPCT 数学有效性判定，也不自动授权下一阶段。

完整批准决定、pilot tie-break、mechanism diagnostic deferment、output schema 与排序合同见 `FPCT_1A_APPROVAL_ADDENDUM.md` 和 `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest_v2.json`。

## 18. FPCT-1A-R operative amendment

### 18.1 Operative protocol composition

- v1 protocol SHA256：`6dced3da6b8f82228666eb64250f51dd6db53e78203249904d47c56e26988ea4`；
- v1 manifest SHA256：`8cb562a6dc915c59275b652bc99deb83e3d81c291185c931a9ed8325f1cb27f4`；
- approval addendum：`FPCT_1A_APPROVAL_ADDENDUM.md`；
- operative manifest：`recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest_v2.json`，`schema_version=2`。

v1 保留为 never-executed history；addendum/v2 只在自然数据前替换 threshold、ceiling、readiness、ranking、statistics-use 和 output-field provisions。

### 18.2 Pilot ranking and mechanism boundary

只对达到工程门槛的 heterogeneous pairs 排名：最弱 task positive-group count、task-macro observed support rate、pooled positive-group count、canonical pair ID。Same-tokenizer control 永不排名。

Candidate count 只能证明 structural opportunity。`D_K`、`D_V`、candidate-logit variance 和 Jensen gap 预注册为后续 operator pilot diagnostics；FPCT-1A-R/1B 不计算。

### 18.3 Commit history and authorization

v1 的 `commit=false/push=false` 是当时边界。用户随后单独授权并推送 commit `7207aafffc7f72976473815bc11102f8b06dddc1`；该事实不追溯扩大 v1 权限。

当前状态：FPCT-1A-R `GO`；FPCT-1B、FPCT-1C、FPCT-2 及以后均 `NOT AUTHORIZED`。
