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

使用 Welford 在线累计，不保存大规模 raw KV。统计键至少覆盖：seed、checkpoint arm、task、sample、layer、head、response query position、parent position 与 candidate count `m`。每个 sample capture 的 long-form row ceiling 前瞻冻结为 `262,144`；它跨 layer/forward 累计，并必须在 CPU materialization 前 fail closed。该 ceiling 是内存完整性上限，不是根据自然数据选择的科学阈值。

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
- `e1_synthetic_query_variance.json`；
- `recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate.json`，汇总六项布尔 hard gate 与逐文件 SHA256 evidence。

冻结数值合同沿用 reference operator 的预数据规则：float64 `atol=1e-10, rtol=1e-8`，float32 `atol=2e-5, rtol=2e-5`；序列化的 centered-`lambda=0` 行按 float32 tolerance 验证，invalid probability/gradient 必须精确为 0。真实 random-small Qwen3 eager + `DynamicCache` 集成 oracle 中，`F(lambda=0)` 与 C_post、默认 `F` 与显式 `F(lambda=1)` 均要求 bitwise identical。

## 7. 自然输出前的 execution/provenance lock

任何 E0-design tokenizer/alignment、pretrained model forward 或 correctness output 之前，必须先完成并 push successor Commit A2。Commit A2 冻结本文、schema、manifest、consolidated gate、科学代码、executor、K8s render-only templates 与 tests；原 `744a943...` Commit A 不满足本节 v3 closure，禁止作为 execution SHA。之后的唯一允许顺序是：

2026-07-26 的前瞻性 operational-closure 复核发生在任何 dataset row lookup、自然 tokenization/alignment、runtime probe 或模型输出之前。原 execution SHA `744a943ea804dfebe4e6d3cba756b6a89763002f` 因 runtime-probe renderer、mounted ConfigMap receipt 消费与只读路径闭包尚不完整而被标记为 `ABANDONED_BEFORE_NATURAL_DATA`；它不是科学 NO-GO，禁止 resume 或复用任何 artifact。只有包含本节新增硬门的后继 clean/pushed execution commit 才能成为实际 source snapshot。

```text
clean pushed successor Commit A2
  -> exact git-archive source snapshot + portable Git receipt
  -> CPU-only E0-design input lock + raw-topology export/verify
  -> model-output-free immutable-image runtime probe
  -> six checkpoint/data/model/config tree lock + 108-shard execution plan
  -> immutable ConfigMap mounted-byte verification
  -> Commit B records receipts/status only
  -> natural E1-2 execution from the successor Commit A2 read-only snapshot
```

CPU input lock 只允许 326 个已开放的 E0-design distinct groups；E1-pilot 不得 render、tokenize、align 或进入 sidecar。逐样本 long-form 行数在 model load 前精确冻结为：

```text
answer_query_count * certified_parent_count * 28 receiver layers * 16 query heads
```

每个样本硬上限为 `262,144`，每 task 预先记录 `count/sum/min/p50/p95/max/argmax`；分位数固定为 nearest-rank `sorted[ceil(p*n)-1]`。Raw-to-runtime topology ledger 必须在任何 model output 前从 pre-sanitizer alignment 生成、逐行验证并纳入 plan。

Source snapshot 必须由 clean、已 push 的 execution commit 通过 `git archive` 产生。Portable receipt 同时绑定 commit SHA、Git tree OID、`ls-tree` mode/blob/path、archive bytes、mounted-tree SHA 与无额外路径；K8s 只读挂载该 snapshot，禁止挂 live worktree。Immutable image probe 只记录 Python/package/CUDA/hardware metadata，明确 model/tokenizer/checkpoint/output 均未加载。

Runtime probe 必须由 snapshot 内 exact renderer 对 snapshot 内 exact template 确定性渲染，producer/template/replacements/rendered YAML 均记录 bytes 与 SHA256。Source、renderer-output 与唯一 runtime-output root 在 lexical path 和 resolved physical path 上必须互不重叠，任一 symlink alias hard error。Probe Pod 在 import torch/CUDA 之前必须对 `/opt/fpct/.fpct_e1_source_snapshot_receipt.json` 执行 `verify-mounted`，同时验证 canonical raw receipt bytes/SHA/size 与 mounted-tree SHA；仅回显调用者提供的 tree SHA 不构成 provenance。

Plan 必须完整锁定 sender/receiver 全 runtime asset trees、E0 dev-data tree、全部 rendered configs，以及六个 immutable `final` projector trees；每个 `final` projector set 必须机械证明与同 attempt 的 `checkpoint-64` projector set byte-identical，且 final/step-64 两棵树均只读哈希。Runtime 强制 `projector_load_mode=strict_attested`，missing/unexpected keys 必须为空；legacy 默认仍保持 historical non-strict 行为。

Capture artifact 只写 Parquet，固定 `4096` 行 batch/row-group；verify、baseline join、stage merge 与 analyzer 均须 bounded streaming，禁止整表 `read_table/read_text/to_pylist` materialization。每个 shard 使用 recoverable exclusive claim lease、atomic no-overwrite artifact 和 deterministic resume。Initial plan/gate/runtime receipt 与 E1-2 finalized receipt 使用 `immutable: true` ConfigMap，必须通过 `<1 MiB` 门及 API mounted-byte verification。

`verify`/`verify-finalized` 必须生成不可省略的 mounted-byte receipt；baseline/F renderer 必须消费 initial receipt，E1-3 renderer 还必须消费 finalized receipt。Rendered Job 固定携带 external expected plan SHA；`run-shard` 在 backend/model load 前验证当前 mounted plan self-hash，并机械重算 `claim_id=SHA256(plan_sha256 || ':' || shard_id)`。这关闭 immutable ConfigMap 被同名删除重建为另一份 self-consistent plan 的窗口。

Capture 与 probe 容器使用 read-only root filesystem，所有 HOME/XDG/HF/Torch/CUDA/W&B cache 只指向隔离的 `/tmp` emptyDir。唯一持久可写 hostPath 是当前 execution 的 capture/runtime output root。Capture output 必须与 source/E0/models/input/raw 等只读 host roots 在 physical path 上不相同、不嵌套且无 symlink alias；container mount paths 亦须 lexical disjoint。E1-3 finalized E1-2 tree 只允许作为 output root 内经 receipt 锁定的更具体 read-only nested mount，任何 shard output 不得落入该子树。

任何 successor Commit A2 中的 operator、alignment、training/evaluation、threshold、input、schema 或 analysis code 变更都会使 execution plan 失效；不得在看到自然输出后原地修补继续。Commit B 只能记录执行 receipts 和状态，K8s 始终运行 successor Commit A2 snapshot。

## 8. E1-2：E0-design full mechanism/topology audit

本阶段不训练，只读复用 E0 的六个 checkpoint：三个 seeds 各自的 C_post-trained 与 F-trained step-64 checkpoint。对每个 checkpoint 在相同输入/state 下运行 `C_post` 与 `F` inference，保持原四-cell 结构：`Y_CC`、`Y_CF`、`Y_FC`、`Y_FF`。

### 8.1 Teacher-forced answer endpoint

对完整 gold response 做 teacher forcing；query position `t` 当且仅当 `labels[t+1] != -100` 时进入 answer-query population。Primary diagnostic 为：

\[
\Delta\log p(y^*)
=\log p_F(y^*)-\log p_{C_{post}}(y^*).
\]

End-task accuracy 与 correctness flip 仅作 secondary。该 fixed-checkpoint diagnostic 不能替代 matched training，也不能成为 confirmatory claim。

在任何自然 E1 mechanism output 前，gold response 的文本合同冻结为 canonical assistant content
`The correct answer is {A|B|C|D}.`，其中字母由原 E0-design 数据的 canonical answer 决定。使用与 E0 相同的 prompt formatter、receiver/source chat template、`enable_thinking=false` 和 `include_response=false`；labels 由完整两轮 chat 的最后一个 assistant section 产生，prompt 及 padding 均为 `-100`。逐 token 使用 `logits[t] -> labels[t+1]`，先在每个完整 response 内求和，再在同一 distinct content group 内对 member samples 等权平均，最后按 distinct content group 等权汇总。不得把 teacher-forced next-token argmax accuracy 称为 end-task accuracy。

Secondary end-task accuracy 固定为当前 frozen code、同一 checkpoint/operator/λ 下的 E0-compatible deterministic greedy generation/parser correctness，并在每个 distinct content group 中只计一次。C_post baseline 只运行一次；每个 F λ（包括真实执行的 `λ=0` 和 endpoint `λ=1`）均独立生成自己的 current-operator correctness。Executor 只能从不可变 C_post artifact 机械 join baseline correctness，不得允许 runtime backend 自报 `cpost_*`。逐 query row 只重复携带该 group outcome 供拓扑 join，aggregate 时不得按 query、parent、layer 或 head 重复加权。

C_post capture 中的 replicated-collapse 只用于 parameter-free mechanism diagnostic expansion，以证明 slot shape 不是解释；它不是第四个 arm，也不替代实际 C_post endpoint/current-operator correctness。F backend 永远不得自行提供 C_post baseline 字段。

### 8.2 Candidate 差异的三层分解

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

本轮 mechanism sufficient statistics 统一以 FP32 accumulation 计算，冻结 `epsilon=1e-12`。Source tap 位于合法 source candidate gather 后、projector/fuser 前；fused tap 位于 candidate-specific projector/fuser 与 legacy confidence/gate 后、A-collapse 和 centered-λ 变换前。两者的 `D/E` 均对各自完整 `head x feature` tensor 求 Frobenius 标量；sender heads 与 receiver heads 不作伪一一对应。Attention row 中 `query_head` 表示 `Hq`，`kv_head=floor(query_head/(Hq/Hkv))`；完整 source/fused Frobenius scalar 可重复 join 到 query-head rows，但 projector contraction 不得被描述为 source-head 对 receiver-head 的配对比较。

### 8.3 Topology strata

每个 `m>=2` parent 必须归入一个互斥 stratum：

| Stratum | 定义 | 当前解释边界 |
|---|---|---|
| `partition_compositional` | 相邻 source subtokens 的不重叠 intersection 合在一起覆盖 receiver span | 可能应 composition，不一定应作为互斥 candidates |
| `competing_overlap` | 每个 candidate 都独立完整覆盖 receiver span，source index/token ID 唯一、span 互异，且不存在 boundary/window/zero-length/duplicate/partial-overlap/receiver-overlap alias | 最接近 marginalization hypothesis；普通 overlap 不足以进入本类 |
| `boundary_fallback` | template、special token、offset boundary 或 sanitizer/fallback 产生 | collapse/fallback/exclusion 候选，不能当 headline support |
| `neighbor_expansion` | candidate window 引入、且不属于直接 overlap cover 的邻居 | 必须单独验证语义，不得并入 competing overlap |

每类报告 group/sample/parent 占比、`m`、source/post-fuser dispersion、KL/TV、parent mass、answer log-prob effect 与 accuracy flip。无法唯一分类的 row 必须作为 `taxonomy_unresolved` integrity stratum 报告，不得静默分到最有利类别。

当前 operative sanitizer=`certified_slot0_v1` 且 candidate window=`0`。因此真正进入 C_post/F runtime 的 certified `m>=2` parent，按 certification 定义必然是 disjoint ordered complete cover，并在 functional mechanism rows 中如实标为 `partition_compositional`。`competing_overlap`、`boundary_fallback`、`neighbor_expansion` 与 `taxonomy_unresolved` 只能出现在单独的 raw-to-runtime topology ledger；已经共同退化为 slot-0 的 uncertified row 不得伪装成 F 实际使用的 factorized parent。Functional table 与 raw topology ledger 必须同时报告，且不能把 raw stratum 当作 runtime mechanism activation。

Taxonomy precedence 固定为 `boundary_fallback -> neighbor_expansion -> partition_compositional -> strict competing_overlap -> taxonomy_unresolved`。Raw-to-runtime ledger 必须在任何 model output 前由 CPU input lock 从 pre-sanitizer alignment 生成，并绑定 receiver/source token IDs、absolute/relative spans、intersections、candidate origin、raw/runtime indices/weights/`m`、certification reason、alias、functional eligibility 与 span-geometry SHA。由于 candidate window 固定为 0，它不得生成 `neighbor_expansion`；没有显式独立竞争证据时也不得生成 `competing_overlap`。该 ledger 只定位 alignment/topology，不保存或推断 accuracy outcome。

### 8.4 解释矩阵

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
- `e1_centered_lambda_summary.csv`；
- `FPCT_E1_MECHANISM_AUDIT.md`。

Model-output-free raw topology 同时生成 JSONL、Parquet、aggregate CSV、aggregate JSON 与 hash manifest；stage finalization 另生成 stage artifact/result manifest。逐 query/parent 大文件保存在 `local/` 或共享存储，不提交 Git。

逐 query/parent 大文件保存在 `local/` 或共享存储，不提交 Git；tracked result manifest 必须记录路径、bytes、row count 与 SHA256。

## 9. E1-3：E0-design centered-λ sweep

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

执行图冻结为依赖安全的四步：先运行 18 个 C_post baseline shards 并写 completion marker；再运行 18 个 F endpoint shards；36 个 endpoint 全部 closure 后必须完成 bounded analyzer、stage manifest 与 immutable `FINALIZED_E1_2` receipt；只有该 receipt 在 render 时经过 deep verification 并以只读 ConfigMap/hostPath 挂载，才运行 `3 × 2 × 4 additional F lambdas × 3 = 72` 个 E1-3 shards。`λ=1` 复用 E1-2 F endpoint；`λ=0` 必须作为真实 F runtime control 执行，不能用 C_post artifact 代替。最终 closure 包含全部 108 shards。阶段内可并行，禁止跨阶段并行；自然输出后如需修改 successor Commit A2 代码，本 execution 直接失效，不得原地 patch/rerun。

本 sweep 是 fixed-checkpoint response-surface diagnostic，不是训练结果。阶段 3 结束后才允许基于 E0-design 证据形成 root-cause record；任何新 operator 只能通过新的 prospective amendment 选择一个因素并冻结。该 amendment 必须早于 E1-pilot forward/outcome。

## 10. 当前禁止事项

在 E1-pilot 单独获批前，禁止：

- 扩大 E0 seed 数、启动 36-run 或任何新训练；
- 运行、读取、聚合或选择 E1-pilot outcome；
- 读取 confirmatory model-selection/test outcome；
- 根据 E1-pilot、confirmatory 或 Phase2A outcome 改机制定义、λ grid、topology 或 operator；
- 同时修 fuser、RoPE、prior、temperature、composition 或 native null；
- 引入 native null/`g`、selector、新 gate、Route3 router 或 F-only 参数；
- 修改 E0 tracked result、E0 checkpoint 或 E0 aggregate；
- 把 fixed-checkpoint headroom 解释为 task improvement 或 confirmatory evidence。

## 11. 当前 claim boundary

- Instrumentation oracle 成立，只说明测量正确，不说明真实机制被利用。
- Candidate 差异存在，只说明 structural/representation opportunity，不说明 query 可利用。
- Query posterior 随 answer query 改变，只说明机制被激活，不说明 gold-token log-prob 或 accuracy 改善。
- Centered-λ 出现正的 fixed-checkpoint `Delta logp`，只说明当前 checkpoint 可能存在干预 headroom，不等价于 matched-training improvement。
- E1-pilot 永不具有 confirmatory eligibility。
- 任何 NO-GO 只针对当前 TinyLlama→Qwen3 pair、E0 checkpoint、数据层与 operator family；不得写成 FPCT 普遍无效。
