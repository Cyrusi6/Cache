# FPCT-E1 机制审计预注册

> 研究线：`research/fpct-e1-mechanism-audit`
> 基线提交：`613958af38fad27e1ea933ccc0dda6d1af5cce89`
> 预注册日期：2026-07-26（Asia/Shanghai）
> 当前资源边界：只允许阶段 0、instrumentation hard gate、E0-design mechanism audit 与 E0-design centered-λ sweep；不训练，不运行或读取 E1-pilot，不释放 confirmatory 数据。

## 1. 科学动机与不可回改边界

FPCT-E0 已冻结为 `E0_NO_GO_FOR_FURTHER_SPEND`。三次 matched run 的工程完整性均为 `GO`，candidate factorization 机制均有非零激活，但系统效应均值为 `T=-0.2728 pp`，即时 query-time operator 效应均值为 `O=-0.4762 pp`，且 `0/3` 个 seed 的 `O` 为正。E1 不重新解释、不覆盖、也不扩大 E0；尤其不得把后续改进追溯为 E0 成功。

E1 的当前问题是：在不增加 seed、不训练新 operator、也不读取新 pilot outcome 的条件下，先确认机制测量是否有效，再定位 candidate 差异在哪一层收缩，最后用同 checkpoint 的连续干预判断当前 factorized state 是否存在可利用 headroom。

在以下三步完成之前，不得冻结或选择新的训练 operator：

1. instrumentation hard gate；
2. E0-design full mechanism/topology audit；
3. E0-design centered-λ sweep。

在上述步骤完成、审计结果被冻结、且下一 operator 通过新的前瞻性 amendment 明确锁定之前，E1-pilot 必须保持 `SEALED / NOT RUN / NOT READ`。Confirmatory model-selection、test、正式 seeds 与任何 outcome 始终保持 sealed。

## 2. E0 只读冻结

以下 tracked 文件是 E1 的只读输入锚点：

| 路径 | SHA256 | 合同 |
|---|---|---|
| `FPCT_E0_RESULTS.md` | `632af006b37f1edfb26356d47e615def7052158fa1cbe8d944f063f8e99c1c50` | 不修改、不重写分类或 claim boundary |
| `recipe/eval_recipe/fpct_e0/versioned_outputs/artifact_manifest.json` | `a988061a1353fb8ce6a944b6359f3a0117857013495246f009612492a1ceb42d` | 作为整个 compact E0 result bundle 的 hash anchor；每个子产物必须继续匹配其中记录的 SHA256 |
| `recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json` | `25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a` | E0-design group identity 的只读来源 |

完整运行仍位于 `/netdisk/lijunsi/fpct-e0/fpct-e0-20260722-v1`。E1 只允许读取预注册诊断所需的 E0 checkpoint、输入与非 confirmatory outcome；不得修改 checkpoint、覆盖 E0 artifact、补跑 E0 seed 或改变 E0 aggregate。

## 3. 已批准的 prospective amendment

用户于 2026-07-26 在 E1-pilot outcome 被运行或读取之前明确批准：

```text
APPROVED_PROSPECTIVE_AMENDMENT:
E1-pilot source = support-fit-only certified groups
counts = ARC 128 / OpenBookQA 70 / MMLU-Redux 128
selection = frozen domain-separated SHA ordering
role = exploratory mechanism pilot
confirmatory eligibility = no
```

该修订解决了 remaining-calibration 不足的问题，不改变 E0 outcome，也不赋予 E1-pilot confirmatory 资格。

### 3.1 冻结选择规则

E1-pilot eligibility 必须同时满足：

- pair 为 E0 的 TinyLlama→Qwen3 heterogeneous pair；
- source partition 为 frozen support audit 的 `fit`；
- distinct content group 具有 certified `m>=2` structural support；
- content-group members 一致；
- 不属于 E0-design；
- 不属于 model-selection 或 test。

对每个 task 独立计算：

```text
rank_hash = SHA256(
    hex("667063742d65312d70696c6f742d763100")
    || UTF8(canonical_task_id)
    || NUL
    || UTF8(content_group_sha256)
)
```

按 `(rank_hash, content_group_sha256)` 升序取前 `128/70/128` 个 unique groups。选择不得读取或依赖 label、answer、logit、prediction、correctness、beneficial/harmful、E0 effect 或 Phase2A outcome。

冻结数据 manifest 为 `recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json`，共 652 rows，SHA256 为 `030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d`。E0-design 与 E1-pilot 各为 ARC 128、OpenBookQA 70、MMLU-Redux 128；两者 distinct content-group intersection 为 0。

## 4. 三层数据 firewall

| 层 | 来源与数量 | 当前用途 | 当前状态 |
|---|---|---|---|
| E0-design | 已使用的 calibration groups：ARC 128、OpenBookQA 70、MMLU-Redux 128 | instrumentation 验证后的 full mechanism audit、topology audit、centered-λ sweep 与路线诊断 | `OPEN FOR E1 DIAGNOSTICS`；已是历史 development outcome |
| E1-pilot | support-fit-only certified groups：ARC 128、OpenBookQA 70、MMLU-Redux 128 | 未来单一已冻结 operator 的 exploratory mechanism pilot | `SEALED / NOT RUN / NOT READ`；不得用于当前实现、阈值、路线或 operator 选择 |
| Confirmatory | 原 model-selection、test 与正式 seed universe | 未来独立正式确认 | `SEALED / NOT AUTHORIZED` |

E1-pilot 的 manifest/hash 可以在阶段 0 物化和校验，但不得 render model input、tokenize/run alignment、运行 forward、读取 answer/correctness、计算任何模型统计或查看任何 outcome。E1-pilot 只有在阶段 1–3 完成、结果被冻结并通过新的 prospective operator-freeze amendment 后，才可能获得单独授权。

## 5. 固定执行顺序

```text
E1-0 protocol + split lock
  -> E1-1 instrumentation hard gate
  -> E1-2 E0-design mechanism/topology audit
  -> E1-3 E0-design centered-lambda sweep
  -> prospective root-cause decision and single-operator freeze
  -> only then may a separate authorization release E1-pilot
```

任何 hard gate 失败都停止后续步骤。不得跳过 instrumentation 直接解释旧 probe，也不得先查看 E1-pilot 再选择 operator。

## 6. E1-1：mechanism instrumentation hard gate

### 6.1 Capture lifecycle

Instrumentation 必须改成显式生命周期，例如：

```python
model.begin_fpct_capture(mode="teacher_forced_response")
outputs = model(...)
metrics = model.end_fpct_capture()
```

至少区分 `prefill`、`teacher_forced_response` 与 `greedy_decode`。普通 `forward()` 不得清空未结束 capture；一次 capture 的多次 forward/decode step 必须累计，且 `end_fpct_capture()` 后才封口。禁止依赖“最后一次 forward 留下的全局字典”。

### 6.2 在线统计与 query variance

使用 Welford 在线累计，不保存大规模 raw KV。统计键至少覆盖：seed、checkpoint arm、task、sample、layer、head、response query position、parent position 与 candidate count `m`。

对 answer query 集合 `T_answer`，query variation 定义为：

\[
\operatorname{Var}_{t\in T_{answer}}\left[\gamma_{tij}\right],
\]

而不是在单个 `q_length=1` decode forward 内求方差。还必须累计：

\[
\operatorname{KL}(\gamma_{ti}\Vert A_i)
=\sum_j\gamma_{tij}\log\frac{\gamma_{tij}}{A_{ij}},
\qquad
\operatorname{TV}(\gamma_{ti},A_i)
=\frac12\sum_j|\gamma_{tij}-A_{ij}|,
\]

以及 candidate posterior top-1 是否随 answer query 改变。Invalid candidate 不进入归一化，且其 probability 与 gradient 必须严格为 0。

### 6.3 Instrumentation parity

在同一 checkpoint、输入、operator、RNG state 与 backend 下比较 instrumentation OFF/ON：

- logits；
- loss；
- generated token；
- cache tensors、length 与 mask semantics。

目标是 bitwise equal。若底层 reduction 使 bitwise equality 客观不可能，必须在任何自然 instrumentation output 前单独冻结严格 dtype-specific tolerance；不得看到结果后放宽。

### 6.4 Synthetic oracle

| Case | 必须满足 |
|---|---|
| 所有 candidates 完全相同 | KL、TV、Jensen gap 与 output delta 均为 0 |
| 两个 answer queries 偏好不同 candidates | cross-query `gamma_query_variance > 0`，top-1 change 被正确累计 |
| candidate permutation + prior 同步置换 | 输出与统计不变 |
| candidate refinement/duplicate with split prior | 输出与统计在冻结 tolerance 内不变 |
| invalid candidate | probability 与 gradient 精确为 0 |
| centered `lambda=0` | 与 C_post/replicated-collapse 相同 |
| 多 decode step | 累计 count 等于所有合法 step 之和，不被最后一步覆盖 |

### 6.5 Hard gate

只有同时满足下列条件，E1-1 才为 `GO`：

- 所有 formula oracle 通过；
- instrumentation ON/OFF 等价；
- synthetic query-changing case 的 cross-query `gamma_query_variance > 0`；
- 多步累计不被最后一次 forward 覆盖；
- capture mode/source counts 与 invalid masks 可审计；
- no NaN/Inf、no output mutation、no leaked raw KV。

任一条件失败则 E1-1=`BLOCKED`，不得进入自然 E0-design mechanism audit。

预注册输出：

- `FPCT_E1_INSTRUMENTATION_REPORT.md`；
- `e1_instrumentation_parity.json`；
- `e1_synthetic_query_variance.json`。

## 7. E1-2：E0-design full mechanism/topology audit

本阶段不训练，只读复用 E0 的六个 checkpoint：三个 seeds 各自的 C_post-trained 与 F-trained step-64 checkpoint。对每个 checkpoint 在相同输入/state 下运行 `C_post` 与 `F` inference，保持原四-cell 结构：`Y_CC`、`Y_CF`、`Y_FC`、`Y_FF`。

### 7.1 Teacher-forced answer endpoint

对完整 gold response 做 teacher forcing，只统计 label `!= -100` 的 answer query。Primary diagnostic 为：

\[
\Delta\log p(y^*)
=\log p_F(y^*)-\log p_{C_{post}}(y^*).
\]

End-task accuracy 与 correctness flip 仅作 secondary。该 fixed-checkpoint diagnostic 不能替代 matched training，也不能成为 confirmatory claim。

### 7.2 Candidate 差异的三层分解

每个 certified `m>=2` parent 至少记录：

1. raw source candidate dispersion：`D_K^source`、`D_V^source`；
2. projected/fused candidate dispersion：`D_K^fused`、`D_V^fused`；
3. attention functional effect：candidate-logit range/variance、KL、TV、Jensen gap、parent attention mass、factorized-vs-collapse output delta 与 gold-token log-prob delta。

对 `X in {K,V}`，绝对与归一化 dispersion 定义为：

\[
D_X=\sum_j A_{ij}\lVert X_{ij}-\bar X_i\rVert_2^2,
\qquad
\bar X_i=\sum_jA_{ij}X_{ij},
\]

\[
\widetilde D_X=
\frac{D_X}{\sum_jA_{ij}\lVert X_{ij}\rVert_2^2+\epsilon}.
\]

Projector/fuser contraction ratio 为：

\[
r_K=\frac{D_K^{fused}}{D_K^{source}+\epsilon},
\qquad
r_V=\frac{D_V^{fused}}{D_V^{source}+\epsilon}.
\]

其中 `r << 1` 表示强收缩。报告必须同时给出 numerator、denominator 与 `epsilon`，不能只给 ratio；`epsilon` 的 dtype-specific 值须在自然审计前由 oracle/config 冻结。

### 7.3 Topology strata

每个 `m>=2` parent 必须归入一个互斥 stratum：

| Stratum | 定义 | 当前解释边界 |
|---|---|---|
| `partition_compositional` | 相邻 source subtokens 的不重叠 intersection 合在一起覆盖 receiver span | 可能应 composition，不一定应作为互斥 candidates |
| `competing_overlap` | 多个 source spans 各自对 receiver span 提供重叠解释，而非 disjoint partition | 最接近 marginalization hypothesis |
| `boundary_fallback` | template、special token、offset boundary 或 sanitizer/fallback 产生 | collapse/fallback/exclusion 候选，不能当 headline support |
| `neighbor_expansion` | candidate window 引入、且不属于直接 overlap cover 的邻居 | 必须单独验证语义，不得并入 competing overlap |

每类报告 group/sample/parent 占比、`m`、source/post-fuser dispersion、KL/TV、parent mass、answer log-prob effect 与 accuracy flip。无法唯一分类的 row 必须作为 `taxonomy_unresolved` integrity stratum 报告，不得静默分到最有利类别。

### 7.4 解释矩阵

- source 小、fused 小：当前 tokenizer pair 的 functional opportunity 小；
- source 大、fused 小：projector/fuser contraction；
- source 大、fused 大、posterior 接近 prior：query discrimination、position/prior/temperature 候选根因；
- source 大、fused 大、posterior 变化明显但 output/log-prob 影响小：parent attention mass 低或 candidate V 差异缺乏任务效用。

本阶段不设性能 GO，也不允许选择新 operator。预注册输出：

- `e1_mechanism_rows.parquet`；
- `e1_mechanism_summary.json`；
- `e1_layer_head_summary.csv`；
- `e1_projector_contraction.csv`；
- `e1_candidate_topology.csv`；
- `FPCT_E1_MECHANISM_AUDIT.md`。

逐 query/parent 大文件保存在 `local/` 或共享存储，不提交 Git；tracked result manifest 必须记录路径、bytes、row count 与 SHA256。

## 8. E1-3：E0-design centered-λ sweep

本阶段仍不训练。它只在 E0-design 与相同六个 E0 checkpoint 上做 same-state operator intervention，检验当前 candidate directions 是否存在连续、可利用的 fixed-state headroom。

对合法 candidates 的 fused state 定义 prior-weighted center：

\[
\bar K_i=\sum_jA_{ij}\widetilde K_{ij},
\qquad
\bar V_i=\sum_jA_{ij}\widetilde V_{ij}.
\]

Centered-λ candidates 定义为：

\[
K^{(\lambda)}_{ij}
=\bar K_i+\lambda(\widetilde K_{ij}-\bar K_i),
\qquad
V^{(\lambda)}_{ij}
=\bar V_i+\lambda(\widetilde V_{ij}-\bar V_i).
\]

之后使用与 `F` 相同的合法 mask、parent bias 和单一 global denominator：

\[
z^{(\lambda)}_{tij}
=\frac{q_t^\top K^{(\lambda)}_{ij}}{\sqrt d}
+b_{ti}+m_{ti}+\log A_{ij},
\]

\[
p^{(\lambda)}_{tij}
=\frac{\exp z^{(\lambda)}_{tij}}
{\sum_{u\in N_t}\exp s^R_{tu}
+\sum_{r}\sum_k\exp z^{(\lambda)}_{trk}},
\qquad
o_t^{(\lambda)}
=\sum_{u\in N_t}p^R_{tu}V^R_u
+\sum_i\sum_jp^{(\lambda)}_{tij}V^{(\lambda)}_{ij}.
\]

冻结 grid 为：

```text
lambda in {0, 0.25, 0.5, 1, 2}
```

合同含义：

- `lambda=0`：所有 children 等于 prior-weighted parent center；因为 `sum_j A_ij=1`，必须精确退化到 replicated-collapse/C_post；
- `lambda=1`：精确等于当前 F operator；
- `lambda in {0.25,0.5}`：只收缩 candidate-specific deviation，不改变 center、prior、mask、native slots 或参数；
- `lambda=2`：只作 extrapolative diagnostic，不是已批准训练 operator；
- `m<=1`：所有 λ 必须与 C_post 相同；
- `A` 只通过 `log A` 加入一次，softmax 后不得再次乘 V；
- native atoms 与 source child atoms 必须共享同一个 global denominator。

每个 λ 都报告 teacher-forced `Delta logp(y*)`、accuracy、flip、KL/TV、query variance/top-1 change、parent mass、Jensen gap、output delta，以及按 seed/task/checkpoint-arm/topology/layer/head 的分解。排序、聚合和所有 λ 必须同时报告，不得只保留表现最好的 λ。

本 sweep 是 fixed-checkpoint response-surface diagnostic，不是训练结果。阶段 3 结束后才允许基于 E0-design 证据形成 root-cause record；任何新 operator 只能通过新的 prospective amendment 选择一个因素并冻结。该 amendment 必须早于 E1-pilot forward/outcome。

## 9. 当前禁止事项

在 E1-pilot 单独获批前，禁止：

- 扩大 E0 seed 数、启动 36-run 或任何新训练；
- 运行、读取、聚合或选择 E1-pilot outcome；
- 读取 confirmatory model-selection/test outcome；
- 根据 E1-pilot、confirmatory 或 Phase2A outcome 改机制定义、λ grid、topology 或 operator；
- 同时修 fuser、RoPE、prior、temperature、composition 或 native null；
- 引入 native null/`g`、selector、新 gate、Route3 router 或 F-only 参数；
- 修改 E0 tracked result、E0 checkpoint 或 E0 aggregate；
- 把 fixed-checkpoint headroom 解释为 task improvement 或 confirmatory evidence。

## 10. 当前 claim boundary

- Instrumentation oracle 成立，只说明测量正确，不说明真实机制被利用。
- Candidate 差异存在，只说明 structural/representation opportunity，不说明 query 可利用。
- Query posterior 随 answer query 改变，只说明机制被激活，不说明 gold-token log-prob 或 accuracy 改善。
- Centered-λ 出现正的 fixed-checkpoint `Delta logp`，只说明当前 checkpoint 可能存在干预 headroom，不等价于 matched-training improvement。
- E1-pilot 永不具有 confirmatory eligibility。
- 任何 NO-GO 只针对当前 TinyLlama→Qwen3 pair、E0 checkpoint、数据层与 operator family；不得写成 FPCT 普遍无效。
