# FPCT-1A Ambiguity Support Audit Protocol

> Stage：FPCT-1A——Ambiguity Support Audit Protocol Locking
> 状态：`REVIEW REQUIRED`
> 下一阶段：FPCT-1B `NOT AUTHORIZED`
> 固定基线：`9fa1f0ac3bedefd282961a853278ab88fb376fa2`
> Branch：`research/fpct-factorized-transport`
> 日期：2026-07-19（Asia/Shanghai）

## 1. 目的、证据边界与时间防火墙

本阶段只冻结 FPCT-1B 的 label-free ambiguity support audit 协议，不执行 audit。FPCT-1A 不回答任何 pair 的自然 ambiguity 有多少，也不回答 FPCT 是否有效。

本协议在查看新的自然 ambiguity 分布、逐样本 alignment 输出、accuracy、correctness、beneficial/harmful event 或 Phase 2A selector outcome 前冻结：

- 输入、pair、task 和 split universe；
- tokenizer/alignment 的 byte/config provenance；
- parent/candidate 合法性与 zero-support contract；
- nominal/effective ambiguity 指标与 threshold grid；
- pair eligibility、support floor、pilot ranking 与 tie-break；
- FPCT-1B 输出 schema、aggregation、排序与 GO rules。

任何未获批准的数值不得在 FPCT-1B 结果出现后补写。`math.md` 仍是 non-normative reference；与本协议或 `FPCT_PREREGISTRATION.md` 冲突时，以两份规范文档为准。

## 2. FPCT-0 一致性门

FPCT-1A 开始前已确认：

- `F-C_post` 仍是 query-time factorization preservation 的唯一 headline contrast；
- `C_post-C_pre` 只识别 candidate-specific nonlinear fusion；
- `F-C_pre` 只表示整体系统差异；
- 第一轮继续固定 `a(x)=1`、`g=1`、`position_mode=legacy`；
- 不加入 de-RoPE/re-RoPE、receiver-native null、Phase 2A selector、新 entropy/confidence 路径或新 gate；
- baseline entropy、confidence、legacy gate 的定义、参数、调用位置保持冻结；
- FPCT worktree 中没有 `PHASE2A_*`/`phase2a_*` dirty path。

若任一条件后续失效，FPCT-1A/1B 立即 `BLOCKED`。

## 3. Model-pair universe

Receiver 固定为 `Qwen/Qwen3-0.6B`。

| Canonical pair ID | Sender | 类型 | Pilot eligibility |
|---|---|---|---|
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | heterogeneous | 可按 support rule 竞争 |
| `qwen25_0p5b` | `Qwen/Qwen2.5-0.5B-Instruct` | heterogeneous | 可按 support rule 竞争 |
| `llama32_1b` | `meta-llama/Llama-3.2-1B-Instruct` | heterogeneous | 可按 support rule 竞争 |
| `qwen3_1p7b` | `Qwen/Qwen3-1.7B` | same-tokenizer control | 永不成为 primary pilot |

规则：

- pair universe 不得根据 FPCT-1B support、accuracy 或 Phase 2A outcome增删；
- same-tokenizer pair 必须完整报告，但不得计入 heterogeneous GO 数量；
- Qwen3-1.7B 与 Qwen3-0.6B 的 `tokenizer_config.json` 和 `tokenizer.json` SHA256 完全相同，是 byte-level same-tokenizer control；
- exact tokenizer file SHA、model/directory SHA 和已知 revision 冻结在 protocol manifest；
- Qwen3-1.7B 与 Llama-3.2-1B 的 tracked config 未提供 resolved immutable HF revision；FPCT-1B 必须校验冻结 tokenizer byte SHA，不得用可变 `main` ref 代替 byte identity。

## 4. Task/input universe

FPCT-1B 只允许使用稳定三任务 canonical evaluation inputs：

| Task ID | Dataset/config/split | Frozen rows | Distinct content groups | Dataset revision |
|---|---|---:|---:|---|
| `ai2-arc` | `allenai/ai2_arc` / `ARC-Challenge` / `test` | 1,150 | 1,150 | `210d026faf9955653af8916fad021475a3f00453` |
| `openbookqa` | `openbookqa` / `main` / `test` | 500 | 500 | `388097ea7776314e93a529163e0fea805b8a6454` |
| `mmlu-redux` | `edinburgh-dawg/mmlu-redux-2.0` / 57 subjects / `test` | 5,615 | 5,583 | `372ea425445d51e1ba1188c56e5e893f8138621f` |

合计 7,265 canonical rows、7,233 distinct content groups。该规模与数据 fingerprint 已公开于 Phase 2A-0，只作为 input provenance，不用于选择 ambiguity threshold。

Prompt/input rendering 固定为：

- `answer_method=generate`；
- `use_cot=false`；
- `use_template=true`；
- `sample_interval=1`；
- 单一 user message；
- `add_generation_prompt=true`；
- `enable_thinking=false`；
- sender/receiver 各自使用自己的 chat template；
- chat-rendered text 使用 `add_special_tokens=false`、`return_offsets_mapping=true`；
- 不执行 truncation；
- 只读取 question/choices/prompt 构造字段，不读取 label、prediction、correctness 或 event outcome。

不得用 raw-question-only、不同 chat template、不同 special-token policy 或截断输入替换该 universe。

## 5. Split contract

复用已前瞻冻结的 content-group split：

| Split | Hash interval | 允许用途 |
|---|---|---|
| `fit` | `[0.00, 0.30)` | label-free pair eligibility |
| `calibration` | `[0.30, 0.45)` | label-free pair eligibility 与 pilot ranking |
| `model-selection` | `[0.45, 0.60)` | 后续机制开发检查；不得改 pair/threshold/pilot |
| `test` | `[0.60, 1.00)` | sealed reporting；不得用于选择 |

Split version 固定为 `cache-phase2a-v1-29a96947`。Group key 是 normalized question+choices content SHA256；重复 content 必须进入同一 split。

执行顺序必须是：

1. 先 resolve tokenizer revisions，并 materialize/hash 全部 content-group split assignment；这一步不运行 tokenizer/alignment；
2. 在任何 support audit 前冻结并 hash analysis code、protocol/manifest、threshold、support-floor parameters、tokenizer/config/input/split provenance；
3. 只运行一次 `fit+calibration` label-free support audit；
4. 机械应用已冻结 eligibility/ranking，写出并 hash `pilot_selection_lock.json`；不得根据 support 输出修改 code/config/threshold/floor；
5. lock 完成后才允许 materialize `model-selection/test` support reporting；
6. 后两者不得回写 threshold、support floor、pair eligibility 或 pilot。

FPCT-1A 不生成逐 group split manifest；该文件必须在 FPCT-1B 获授权后、任何 audit 前按冻结算法 materialize 并记录 SHA256。

## 6. Frozen alignment contract

| Field | Frozen value |
|---|---|
| `alignment_strategy` | `soft_span_overlap_v2` |
| `soft_alignment_top_k` | `4` |
| `soft_alignment_score_mode` | `uniform` |
| `soft_alignment_boundary_bonus` | `0.5` |
| `soft_alignment_boundary_tolerance` | `1` character |
| `soft_alignment_min_weight` | `0.0` |
| `soft_alignment_candidate_window` | `0` |
| `soft_alignment_reweight_mode` | `none` |
| `soft_alignment_reweight_strength` | `1.0`，inactive |
| `soft_alignment_reweight_power` | `2.0`，inactive |
| normalization | selected legal positive scores 做 row-wise L1 normalization |

Candidate selection order 固定为：

1. weight score descending；
2. overlap length descending；
3. source token length descending；
4. source index ascending；
5. 截断到 top-4。

`A_ij` 只取 raw `source_weights`。不得乘 source confidence、entropy、legacy gate、selector 或任何 accuracy-derived weight。

## 7. Parent universe 与合法 candidate

Audit parent universe 固定为：

`eligible parent = message_mask == true AND receiver padding mask == false`。

Chat-template/native-only rows不进入 zero-support denominator。它们属于 FPCT 预注册中的普通 native-only slots。

对 eligible receiver parent `i`，candidate slot `j` 合法当且仅当：

1. 若实现提供 standalone valid mask，则该 mask 为真；
2. `0 <= source_index_ij < sender_token_count`；
3. source index 不指向 sender padding；
4. raw `A_ij` 有限；
5. raw `A_ij > 0`。

当前稳定输出没有独立 valid mask，FPCT-1B 必须从 index range、sender padding、finite weight 和 positive weight 派生；不得把 `positive_overlap_counts` 当成 post-top-k legal count。

必须区分 expected inactive slot 与 malformed invalid slot：

- standalone mask 为假、sentinel/out-of-range index 或 in-range zero-mass slot，且 raw weight 是有限的精确 `0`：expected inactive，先 mask，单独计数，不是 integrity failure；
- mask 为假/越界/sender-padding slot 却带正质量：`invalid_positive_mass`，integrity failure；
- 任意非有限或负 raw weight：integrity failure；
- duplicate legal source index、合法行归一化失败或 uniform-grid identity 失败：integrity failure。

因此 top-k tensor 中正常的 `(-1, 0)` padding 不得被误算成自然 zero-support 错误，也不得导致 pair eligibility 自动失败。

非法 candidate 必须先 mask。设合法集合为 `V_i`，若非空：

\[
A_{ij}
=
\frac{A^{raw}_{ij}}
{\sum_{k\in V_i}A^{raw}_{ik}}
\qquad j\in V_i.
\]

同一 parent 内重复合法 source index 是 integrity failure，不得把 duplicate slot 当成自然 ambiguity。

## 8. Support strata 与 zero-support contract

令 `m_i=|V_i|`：

- zero-support：`m_i=0`；
- one-to-one：`m_i=1`；
- nominal ambiguous：`m_i>=2`；
- effective ambiguous：`m_i>=2` 且通过预注册 mass threshold。

Zero-support contract：

- 不除以零；
- `A_max`、`S`、`N_eff` 输出为 null/空字段，不写 0 或 NaN；
- 不静默填充 epsilon、伪 candidate、nearest neighbor 或 slot 0；
- 不归入 one-to-one；
- 保留在 eligible receiver-parent 和 sample 分母中；
- 作为独立 stratum 报告；
- 排除在 FPCT mechanism population 外；
- 后续三臂只能共同沿用已冻结的 receiver-native fallback，不得借此引入 candidate-bearing parent 的 native null。

若 aligner 已输出合法、质量为 1 的 fallback candidate，则按合法规则属于 one-to-one，同时另报 `fallback_flag=true`；不得事后改称 zero-support。

## 9. Ambiguity metrics

对 `m_i>0`：

\[
N_{\mathrm{eff}}(i)
=
\frac{1}{\sum_jA_{ij}^2},
\qquad
A_{\max}(i)=\max_jA_{ij},
\qquad
S_i=1-A_{\max}(i).
\]

每个 sample 必须报告：

- eligible receiver parent count；
- zero-support、one-to-one、nominal ambiguous counts/rates；
- 每个 threshold 的 effective counts/rates；
- 是否包含 nominal ambiguous parent；
- 是否至少包含一个 effective parent；
- 是否属于 exact no-factorization control（所有 eligible parent 均为 `m<=1`）；
- 是否属于 below-gate-primary nominal control（存在 `m>=2`，但没有 gate-primary-effective parent）；
- `N_eff` 与 `S` 的 mean、max、p25、p50、p75、p90、p95；
- expected inactive、invalid-positive-mass、negative、nonfinite、duplicate-index integrity counts；
- fallback parent count/rate。

每个 pair/task/split 必须报告：

- token-level support；
- sample-level support；
- direct-effect ceiling；
- `N_eff` 与 `S` 的 count、mean、min、p25、p50、p75、p90、p95、p99、max。

Quantile 固定使用线性插值。空集合在 JSON 写 null、CSV 写空字段。

## 10. Effective ambiguity threshold grid

静态代码合同给出：

\[
A_{ij}=\frac1{m_i},
\qquad
S_i=1-\frac1{m_i},
\qquad
N_{\mathrm{eff}}(i)=m_i
\]

因为当前配置是 `uniform + top_k=4 + min_weight=0 + reweight=none`。因此自然 support grid 只能是：

| `m_i` | `S_i` | `N_eff` |
|---:|---:|---:|
| 1 | 0 | 1 |
| 2 | 1/2 | 2 |
| 3 | 2/3 | 3 |
| 4 | 3/4 | 4 |

候选 threshold：

| ID | 定义 | 数学含义 | 当前 uniform 等价 |
|---|---|---|---|
| `lenient_s50` | `m>=2 AND S>=1/2` | 非 top-1 mass 至少 50% | `m>=2` |
| `primary_s67`（推荐） | `m>=2 AND S>=2/3` | 非 top-1 mass 至少为 top-1 的 2 倍 | `m>=3` |
| `strict_s75` | `m>=2 AND S>=3/4` | 非 top-1 mass 至少为 top-1 的 3 倍 | `m=4` |

推荐把 `primary_s67` 作为 FPCT-1B support gate 的 gate-primary threshold，因为它是在固定离散 grid 上不退化为 nominal `m>=2` 的最小 threshold。`lenient_s50` 与 `strict_s75` 固定为 sensitivity，只报告，不得重排 pilot。

该 gate-primary 不回改 FPCT-0 已冻结的正式 matched-training ambiguity-positive population：未来正式 performance primary 仍以“至少一个合法 `m>=2` parent”为结构 eligibility。`primary_s67` 只决定本阶段的 strong-support eligibility、pilot ranking 和预注册分层；两种“primary”不得混用。

但“m>=3 是否是唯一正确的 practical ambiguity boundary”没有独立客观依据，因此 gate-primary threshold 仍标记 `HUMAN APPROVAL REQUIRED`。未批准前 FPCT-1A 为 `REVIEW REQUIRED`，FPCT-1B 不得启动。

当前 uniform contract 下，threshold classification 使用整数 `m` 的精确等价关系：`m>=2`、`m>=3`、`m=4`。观测 float64 `A/S/N_eff` 只用于验证 uniform identity，不直接承担 `2/3` 浮点边界分类；若 identity 失败，视为 alignment/config integrity failure，不通过额外 epsilon 修补。该整数规则不得自动外推到未来 non-uniform alignment。

## 11. Sample-level direct-effect ceiling

对 pair `p`、task `t`、split `r`，同时冻结 nominal direct support 与 gate-primary effective support：

\[
D_s
=
\mathbf 1[\text{sample }s\text{ 至少有一个 }m>=2\text{ parent}],
\]

\[
E_s
=
\mathbf 1[\text{sample }s\text{ 至少有一个 gate-primary-effective parent}],
\]

\[
\mathrm{NominalDirectCeiling}_{ptr}
=
\frac{\sum_sD_s}{|\mathcal S_{ptr}|},
\qquad
\mathrm{Ceiling}_{ptr}
=
\frac{\sum_sE_s}{|\mathcal S_{ptr}|}.
\]

按本阶段约定，字段 `direct_effect_ceiling` / `Ceiling_pair` 仍指相应 sample population 中 `E_s=1` 的 gate-primary 操作性 ceiling；`nominal_direct_support_ceiling` 必须同时报告，不能省略。

解释边界：

- 同一权重、`g=1`、只切换 `C_post` 与 `F` 时，所有 eligible parent 均为 `m<=1` 的样本才是 exact one-to-one/no-support control；在共同 mask/native fallback 下，fixed-state `F-C_post` 应精确退化为 0；
- identical/duplicate-candidate 数值负对照也是 exact control，但 duplicate legal source index 在自然 audit 中仍属于 integrity failure，不能冒充自然 ambiguity；
- 没有 gate-primary-effective parent、但含 `m=2` nominal ambiguity 的样本是 below-gate-primary negative-control stratum，不是数学恒等 control；`F-C_post` 在这些样本上仍可能非零；
- 因此这里的 `direct-effect ceiling` 是 gate-primary mechanism estimand 的操作性识别 ceiling，不是所有可能 fixed-state operator difference 的严格数学上界；below-gate-primary stratum 的差异必须单独报告，且不得驱动 gate-primary FPCT mechanism claim；
- matched retraining 后，checkpoint adaptation 仍可能产生系统总效应，因此该 ceiling 只适用于 gate-primary fixed-state estimand，不是最终 accuracy effect 的数学上界；
- ceiling 只表示现有输入是否提供识别机会，不证明 FPCT 数学或性能有效。

## 12. Pair eligibility support floor

Pair eligibility 只使用 `fit+calibration` 的 label-free support。Selection/power-planning unit 固定为 distinct content group；canonical sample rows 仍单独报告，这个选择不预先决定 FPCT-4 的正式 accuracy aggregation。

对每个 pair/task 合并 `fit+calibration` 后，令：

- `k_pt`：至少一个 gate-primary-effective parent 的 distinct content groups 数；
- `n_pt`：distinct content groups 总数；
- `C_hat_pt=k_pt/n_pt`；
- `L_pt`：预注册 simultaneous one-sided Wilson lower confidence bound。

令 familywise lower-bound error 为 `eta`。Eligibility 共涉及 3 个 heterogeneous pairs × 3 tasks，即 9 个 selection cells；使用 Bonferroni `eta_cell=eta/9`。对 `n>0`、`p_hat=k/n`、`z=z_(1-eta_cell)`：

\[
L
=
\frac{
\widehat p+\frac{z^2}{2n}
-z\sqrt{\frac{\widehat p(1-\widehat p)}n+\frac{z^2}{4n^2}}
}{1+\frac{z^2}{n}}.
\]

任一 selection cell `n=0` 为 incomplete。Hash split 被视为对冻结 content groups 的 exchangeable partition；若 split 或 group provenance 失败，不得解释 Wilson projection。

Pilot label-free support score：

\[
\mathrm{Score}_p
=
\frac13\sum_{t\in\mathcal T}L_{pt}.
\]

每个 task 等权，每个 content group 最多贡献 1，不按 receiver-token 数或 prompt length 加权。在 9 个 cell 同时覆盖事件成立时，`Score_p` 是 pair 的 task-macro support lower bound；它不是 token-micro support score。

Support floor 由 minimum practical effect 推导。人工需锁定：

- `delta_pos`：gate-primary-positive population 上的 minimum practical paired direct effect；
- `delta_direct_all`：全 population 上的 minimum practical direct effect；
- `alpha`：two-sided test level；
- `power=1-beta`；
- familywise Wilson LCB confidence `1-eta`；
- support-floor estimand：推荐 `task_macro_cluster`。

参数必须满足 `0<delta_direct_all<=delta_pos<=1`。这些值只规划 fixed-state/direct mechanism reach；matched retraining 可能影响 gate-negative samples，因此不得把本 floor 称为最终 accuracy futility bound。

采用最保守 paired-discordance bound `q_star=1`：

\[
n_{\mathrm{req}}
=
\left\lceil
\frac{q_\star
\left(z_{1-\alpha/2}+z_{1-\beta}\right)^2}
{\delta_{\mathrm{pos}}^2}
\right\rceil.
\]

令 `N_t^grp` 为冻结 task input 中 distinct content-group 数；它由同一 content normalization/split algorithm 在任何 alignment audit 前先 materialize 并 hash，不得用 raw row count 或 receiver-token count 替代。用 `N_t^grp` 与 `L_pt` 投影 gate-primary-positive group 数：

\[
\widetilde n_{pt}=\lfloor N_t^{grp}L_{pt}\rfloor,
\qquad
\widetilde n_p=\sum_t\widetilde n_{pt},
\]

\[
n^{macro}_{p,eff}
=
\begin{cases}
\dfrac{9}{\sum_{t=1}^{3}1/\widetilde n_{pt}}, & \min_t\widetilde n_{pt}>0,\\
0, & \text{otherwise},
\end{cases}
\qquad
C^{macro}_{p,L}=\mathrm{Score}_p.
\]

Heterogeneous pair eligible 当且仅当：

1. `n_macro_p,eff >= n_req`；
2. `C_macro_p,L * delta_pos >= delta_direct_all`；
3. fit/calibration input、split、tokenizer、config、threshold、support parameters 与 analysis code hash 均在 audit 前冻结；
4. 每个 content group 只计一次，且无 integrity failure。

推荐统计参数为 `alpha=0.05`、`power=0.80`、familywise one-sided Wilson 95% LCB（9-cell Bonferroni）与 `task_macro_cluster` estimand。该 harmonic effective size 要求三个 task 都有正 support，并惩罚 task imbalance，替代任意的“至少两 task/每 task 固定数量”阈值。这些选择以及 `delta_pos/delta_direct_all` 均需人工批准。尤其 practical-effect 值没有仓库内客观唯一依据，不得自动填写。

因此 support floor 当前是完整的 parametric rule，但还不能产生 pass/fail；FPCT-1A 保持 `REVIEW REQUIRED`。

## 13. Pilot selection rule

只对通过 gate-primary support floor 的 heterogeneous pairs 排名：

1. `Score_p` descending；
2. minimum task-level `L_pt` descending；
3. `n_macro_p,eff` descending；
4. `widetilde n_p` descending；
5. canonical pair ID lexicographic ascending。

禁止用下列信息排名或打破平局：

- accuracy/correctness；
- beneficial/harmful events；
- Phase 2A selector outcome；
- test support；
- token-micro rate；
- posterior/gate statistics；
- sensitivity threshold 下的排名。

Same-tokenizer control 永远 unranked，不能替代 heterogeneous pilot。

## 14. FPCT-1B output contract

所有 outputs 位于 `local/final_results/fpct_factorized_transport/fpct_1b_ambiguity_support/rev_<execution-sha>/`，不得提交。

| File | Row unit | 用途 |
|---|---|---|
| `parent_support.csv` | pair/task/split/canonical-sample/eligible-parent | candidate legality、stratum 与 integrity |
| `sample_support.csv` | pair/task/split/canonical-sample | sample support 与 ceiling；带 content-group hash |
| `content_group_support.csv` | pair/task/split/distinct-content-group | selection unit；group 内不一致为 integrity failure |
| `aggregate_support.csv` | threshold/aggregation/pair/task/split | 全 aggregation |
| `pair_eligibility.csv` | pair | support floor、rank、failure reason |
| `pilot_selection_lock.json` | one object | fit+cal selection seal |
| `audit_summary.json` | one object | provenance、aggregates、decision |
| `provenance.json` | one object | commit/config/input/tokenizer/split hashes |

`content_group_support.csv` 必须记录 `group_member_count`、lexicographic-min `representative_sample_key_sha256` 和 `member_support_consistent`。同一 normalized question+choices group 的 canonical sample members 若在同一 pair/task 下产生不同 support strata，audit 为 `INCONCLUSIVE`，不得择一代表。

### 14.1 Required aggregate fields

至少包括：

- pair、pair type、task、split；
- raw sample count、distinct content-group count；
- receiver parent count；
- zero-support count/rate；
- one-to-one count/rate；
- nominal ambiguity count/rate；
- nominal-positive sample count/rate；
- effective ambiguity count/rate for all threshold IDs；
- `N_eff` 与 secondary mass distributions；
- effective-positive sample count/rate；
- direct-effect ceiling；
- nominal direct-support ceiling；
- exact no-factorization control sample count/rate；
- below-gate-primary nominal-control sample count/rate；
- support score；
- pilot eligibility、rank、selected flag；
- eligibility failure reason。

### 14.2 Aggregations

必须同时输出：

- `token_micro`：pool eligible receiver parents；
- `sample_weighted`：每个 canonical sample row 等权；
- `content_group_weighted`：每个 distinct content group 等权，供 eligibility/support audit；
- `task_macro`：三个 task 的 sample-weighted metric 等权平均；
- pair-level decomposition：上述 weighting scheme 必须分别按每个 pair 展开；它是 mandatory grouping view，不是第五种 weighting scheme。

缺失 task 是 audit incomplete，不得从 task-macro 分母静默删除。不得只报告最有利 aggregation。

Task-macro row 使用 `task=__TASK_MACRO__`；不可相加的 raw count 字段写 null，只填写等权 macro rate/distribution。Selection-only 字段的适用范围固定为 `threshold=primary_s67`、`split=fit_calibration`、`aggregation=content_group_weighted`：task row 填 `support_lcb/projected_positive_group_count`，pair task-macro row 填 `support_score/macro_effective_group_count/pilot_*`；其他行写 null，不得选择性省略列。

### 14.3 Sorting

- pair order：`tinyllama`、`qwen25_0p5b`、`llama32_1b`、`qwen3_1p7b`；
- task order：`ai2-arc`、`openbookqa`、`mmlu-redux`；
- split order：`fit`、`calibration`、`fit_calibration`、`model-selection`、`test`；
- threshold order：`lenient_s50`、`primary_s67`、`strict_s75`；
- aggregation order：`token_micro`、`sample_weighted`、`content_group_weighted`、`task_macro`；
- sample rows 再按 content-group hash、sample hash、parent index ascending。

Failure reason enum 冻结为：

- `control_only`；
- `missing_input_or_hash`；
- `integrity_failure`；
- `insufficient_macro_effective_groups`；
- `insufficient_direct_reach`；
- `eligible`。

单值 `eligibility_failure_reason` 按上列顺序取第一个适用值；所有布尔 conjunct 与原始数值仍必须逐列输出。排序、ranking 和 threshold 比较使用未四舍五入的 float64/rational 值，展示舍入不得参与 tie-break。

### 14.4 Serialization contract

- CSV 使用 UTF-8、LF、manifest 中的精确 header 顺序；不得省列或追加未登记列；
- boolean 写小写 `true/false`，integer 写十进制，float 写可 round-trip 的 17 位有效数字；null 在 CSV 为空字段；
- JSON 必须是 RFC 8259-compatible UTF-8，key 按 lexicographic order 稳定序列化，indent 2；禁止 NaN/Infinity；
- schema 变更必须提升 `schema_version` 并在运行前重新人工审查，不能在看到 support 后静默改列。

## 15. FPCT-1B decision rules

判定优先级固定为：

1. 任一 required pair/task/input/hash/schema 不完整，或存在未解决 integrity failure：`INCONCLUSIVE`；不得用已完成 pair 的通过数提前判 GO；
2. 全部 required cells 完整后，按通过 support floor 的 heterogeneous pair 数判定：
   - 0：`NO-GO`；
   - 1：`LIMITED GO`，只允许进入 single-pair pilot；未来若取得机制证据，claim scope 仍限于该 pair；
   - 至少 2：`CROSS-PAIR GO`；
3. umbrella `GO` 仅指 `LIMITED GO` 或 `CROSS-PAIR GO`；same-tokenizer control 不计入通过数量。

任何判定只说明现有数据是否有识别能力，不能解释为 FPCT 数学有效或无效，也不能解释为 task accuracy 会提升或不会提升。

## 16. FPCT-1A gate 与人工决策

FPCT-1A `GO` 需要：

- 本协议与 manifest 完整；
- gate-primary threshold 获人工批准；
- support-floor parameters 获人工批准；
- 没有运行自然 ambiguity audit；
- 没有读取新逐样本统计；
- 没有修改模型代码；
- JSON/path/diff checks 通过。

当前仍需人工批准：

1. gate-primary `primary_s67` 及 sensitivity grid；
2. support-floor estimand `task_macro_cluster`；
3. `delta_pos`、`delta_direct_all`、`alpha`、`power` 与 familywise Wilson LCB confidence。

因此当前状态为 `REVIEW REQUIRED`，不是 `GO`；FPCT-1B 继续 `NOT AUTHORIZED`。

## 17. 禁止事项

FPCT-1A 禁止：

- 编写或运行 ambiguity audit 脚本；
- 对完整数据调用 tokenizer/alignment；
- 打开新的逐样本 alignment CSV/JSON；
- model forward；
- 修改 model/aligner/wrapper/projector；
- GPU/Kubernetes；
- 训练或 checkpoint mutation；
- 查看 Phase 2A-1 未公开结果；
- commit/push；
- 自动进入 FPCT-1B。
