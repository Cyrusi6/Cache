# FPCT 状态

> 当前阶段：FPCT-1A-R——Human Decision Lock and Prospective Amendment
> 当前判定：`GO`
> 下一阶段：FPCT-1B `NOT AUTHORIZED`
> 更新时间：2026-07-19（Asia/Shanghai）

## 1. 隔离身份

| 项目 | 冻结值 |
|---|---|
| 原 Phase 2A-1 工作树 | `/home/lijunsi/projects/Cache` |
| 原工作树 branch | `main`；未切换 |
| 原工作树检查时 HEAD | `9fa1f0ac3bedefd282961a853278ab88fb376fa2` |
| FPCT worktree | `/home/lijunsi/projects/Cache-fpct-factorized-transport` |
| FPCT branch | `research/fpct-factorized-transport` |
| FPCT base SHA | `9fa1f0ac3bedefd282961a853278ab88fb376fa2` |
| FPCT-1A-R source commit | `7207aafffc7f72976473815bc11102f8b06dddc1` |
| 研究线启动时间边界 | Phase 2A-1 结果揭晓前 |
| Phase 2A-1 未公开结果 | 未查看、未使用 |
| FPCT canonical slug | `fpct_factorized_transport` |

操作前，目标 branch 的 local ref、remote-tracking ref 和远端同名 branch 均未发现；目标 sibling 路径不存在；已登记 worktree 只有原 `/home/lijunsi/projects/Cache`。因此创建时没有覆盖已有 branch、worktree 或路径。

原 Phase 2A-1 工作树检查时只有以下未跟踪路径；FPCT-0 未读取其结果内容、未修改、未暂存、未删除：

- `recipe/eval_recipe/phase2a_1/candidate_manifest.json`
- `recipe/eval_recipe/phase2a_1/feature_whitelist.json`
- `recipe/eval_recipe/phase2a_1/protocol_manifest.json`
- `script/analysis/phase2a_1_selector_killtest.py`

最终一致性复查时，原工作树又出现 `test/test_phase2a_1_selector_killtest.py`。这说明 Phase 2A-1 仍在并行推进；FPCT-0 仅记录路径变化，未打开或修改该文件。

交付前最后复查时，原工作树仍保持 `main` 和固定 HEAD，但并行活动已继续产生 tracked/untracked 状态变化：tracked 为 `EXPERIMENT.md`、`FRAMEWORK_UPDATE.md`；另新增 `recipe/eval_recipe/phase2a_1/content_group_split_manifest.{json,sha256}` 与 `test/test_phase2a_1_selector_sealing.py`。这些均属于原工作树的并行研究活动，本任务未读取其内容、未修改、未暂存，也未停止相关工作。

用户已确认 `math.md` 是其自有的相关公式讨论稿。本任务未创建或修改该文件，并将其视为 non-normative reference；它原不在默认提交范围内，用户随后显式授权将原样文件纳入独立 `research/fpct-factorized-transport` branch 的本次 commit。与冻结预注册冲突时以 `FPCT_PREREGISTRATION.md` 为准。核心符号映射为：讨论稿 `p_tij` 对应 global `α_tij`，`γ_tij` 对应 source-candidate posterior `γ_t(j|i)`，`ρ_ti` 对应未来 native/source posterior。其更一般构想包含 de-RoPE/re-RoPE、显式 receiver-native null/sidecar、不同 legacy-gate 建议、缺少正式 `C_post` arm 的 2×2 设计，以及 Q0/Q2/Q3/Q4 候选顺序，均不覆盖第一轮 legacy/`g=1`/三臂 matched 规则，也不构成 FPCT-1 授权。

## 2. 当前范围与授权资源

当前唯一授权资源等级仍为 `R0`：

- repo/branch/worktree/path 的只读环境检查；
- 读取代码、配置和已有 FPCT 文档；
- 编写/修改 FPCT protocol 文档与 manifest；
- Markdown、git diff、路径与禁止项一致性检查。
- 验证通过后，只在 `research/fpct-factorized-transport` commit 并 push 到同名远端 branch。

当前明确未授权：

- 修改任何模型运行代码；
- 实现 attention、sidecar、operator 或 manifest plumbing；
- 运行 model forward；
- 使用 GPU 或提交/查询 Kubernetes FPCT job；
- 训练、评测模型或修改 checkpoint；
- 查看/使用 Phase 2A-1 未公开结果；
- 运行自然 ambiguity audit 或完整数据 tokenizer/alignment；
- 打开新的逐样本 alignment CSV/JSON；
- 自动进入 FPCT-1B；
- 创建 PR、合并 `main` 或 rebase；
- 修改 `/home/lijunsi/projects/Cache`、其他 worktree 或 `math.md`。

## 3. 阶段状态表

资源等级和完整 GO/INCONCLUSIVE/NO-GO 规则见 `FPCT_PREREGISTRATION.md`。

| 阶段 | 当前状态 | 依赖 | 资源等级 | 当前授权 | 决策摘要 |
|---|---|---|---|---|---|
| FPCT-0 | `GO` | 固定 base SHA | R0 | 已完成 | GO only to protocol locking |
| FPCT-1A | `SUPERSEDED BEFORE NATURAL DATA` | FPCT-0 GO | R0 | 历史完成 | never-executed v1 保留 |
| FPCT-1A-R | `GO` | FPCT-1A v1 + source commit `7207aaff...` | R0 | 是，仅 prospective amendment | human decisions locked；v2 operative |
| FPCT-1B | `NOT AUTHORIZED` | FPCT-1A-R GO + 显式授权 | R1 CPU | 否 | label-free structural-support audit |
| FPCT-1C | `NOT AUTHORIZED` | FPCT-1B decision + 人工授权 | R1 | 否 | reference equations/oracle/numerical rule |
| FPCT-2 | `NOT AUTHORIZED` | FPCT-1B `SINGLE_PAIR_PILOT_READY`/`CROSS_PAIR_PILOT_READY` + FPCT-1C GO + 人工授权 | R1 | 否 | 实现 `C_post`/`F` 与 manifest plumbing |
| FPCT-3 | `NOT AUTHORIZED` | FPCT-2 GO | R1 | 否 | CPU 数值、mask、gradient、退化测试 |
| FPCT-4 | `NOT AUTHORIZED` | FPCT-3 GO | R0/R1 | 否 | 冻结 task/control/seed/budget/statistical manifest |
| FPCT-5 | `NOT AUTHORIZED` | FPCT-4 GO | R2 | 否 | fixed-checkpoint diagnostic |
| FPCT-6 | `NOT AUTHORIZED` | FPCT-5 GO | R3 | 否 | matched-training pipeline smoke |
| FPCT-7 | `NOT AUTHORIZED` | FPCT-6 GO | R3 | 否 | confirmatory-disjoint development-only pilot |
| FPCT-8 | `NOT AUTHORIZED` | FPCT-7 GO | R4 | 否 | 正式多 seed matched retraining |
| FPCT-9 | `NOT AUTHORIZED` | FPCT-8 GO | R4 | 否 | 执行冻结 evaluation |
| FPCT-10 | `NOT AUTHORIZED` | FPCT-9 decision | R5 | 否 | mechanism/event re-audit/replication |
| FPCT-11 | `NOT AUTHORIZED` | FPCT-10 decision | R0/R5 | 否 | claim freeze 与归档 |

FPCT-1A-R `GO` 只表示 human decisions 与 v2 protocol 已锁定，不构成 FPCT-1B 授权；所有后续阶段仍需显式人工授权。

## 4. FPCT-0 完成检查表

- [x] 报告原 repo HEAD、branch、status 和 worktrees。
- [x] 识别原 Phase 2A-1 正在使用的工作树与未跟踪相关路径。
- [x] 未停止、切换或修改 Phase 2A-1。
- [x] 验证固定 base SHA 是有效 commit。
- [x] 创建 sibling worktree `/home/lijunsi/projects/Cache-fpct-factorized-transport`。
- [x] 创建独立 branch `research/fpct-factorized-transport`。
- [x] 冻结 `C_pre`、`C_post`、`F` 完整公式。
- [x] 冻结三个 causal contrast 与 headline claim boundary。
- [x] 冻结第一轮方法边界。
- [x] 定义 endpoints、controls、matched-training 和 evidence class。
- [x] 定义 FPCT-0 至 FPCT-11 的依赖、资源等级和 gates。
- [x] 定义六项数值正确性不变量。
- [x] 记录研究线先于 Phase 2A-1 结果揭晓启动。
- [x] 记录新 FPCT checkpoint 必须重新审计 beneficial/harmful events。
- [x] 记录旧 B6 `+8.24pp` 不能作为 FPCT headroom。
- [x] 定义 artifact、manifest、commit 和 checkpoint 隔离规则。
- [x] 用户确认 `math.md` provenance；文件作为 non-normative reference 保留且未修改，并显式授权原样纳入独立 research branch commit。
- [x] 人工审查并批准 FPCT-1A-R structural-support/readiness decisions。
- [ ] 人工显式授权 FPCT-1B；当前不授权。

### 4.1 FPCT-1A protocol checklist

- [x] FPCT-0 headline 与第一轮冻结项一致性通过。
- [x] 冻结三个 heterogeneous pairs 与 Qwen3 same-tokenizer control。
- [x] 冻结三任务 input/prompt/split universe 与 provenance。
- [x] 冻结合格 parent/candidate、mask-first renormalization 与 zero-support contract。
- [x] 冻结 `m`、`N_eff`、`A_max`、`S`、sample/pair support 与 ceiling。
- [x] 冻结 `primary_structural_m2` 与 `high_cardinality_m3`/`strict_m4` sensitivities。
- [x] 冻结 label-free engineering readiness、ordinary Wilson 描述与 pilot tie-break。
- [x] 冻结 FPCT-1B CSV/JSON schema、aggregation 与排序。
- [x] 未运行自然 ambiguity audit、tokenizer/alignment 或 model forward。
- [x] 明确 formal effect/power analysis deferred；无 operative delta/n_req gate。
- [x] 冻结 v1 superseded-before-natural-data 与 v2 operative provenance。

## 5. 决策记录

| ID | 日期 | 决策 | 依据与不可回改边界 |
|---|---|---|---|
| FPCT-D000 | 2026-07-19 | 在 Phase 2A-1 结果揭晓前启动独立 FPCT 线 | 后续不得用 Phase 2A-1 结果回改 FPCT primary hypothesis |
| FPCT-D001 | 2026-07-19 | 固定 base `9fa1f0ac3bedefd282961a853278ab88fb376fa2` | 与并行研究线隔离，保证 provenance |
| FPCT-D002 | 2026-07-19 | 冻结 `C_pre`、`C_post`、`F` | `C_post` 与 `F` 使用相同 candidate-specific fuser |
| FPCT-D003 | 2026-07-19 | `F-C_post` 为 headline 主效应 | `C_post-C_pre` 只测 nonlinear fusion；`F-C_pre` 不单独归因 |
| FPCT-D004 | 2026-07-19 | 第一轮固定 `a(x)=1`、`g=1`、legacy position | 排除 Phase 2A selector、native null 和 position rewrite 混杂 |
| FPCT-D005 | 2026-07-19 | 正式性能证据必须 matched retraining | fixed checkpoint 与旧 B2/B3/B6 仅诊断/初始化 |
| FPCT-D006 | 2026-07-19 | 不编造未知阈值 | 优先 duplicate control；不足则人工批准且必须先于自然结果 |
| FPCT-D007 | 2026-07-19 | FPCT-0 不 commit、不 push、不进入 FPCT-1 | 等待人工审查 |
| FPCT-D008 | 2026-07-19 | 用户确认 `math.md` 为自有公式讨论稿；作为 non-normative reference 保留 | 冲突时以预注册为准；不纳入 FPCT-0 交付/commit，不授权后续实现 |
| FPCT-D009 | 2026-07-19 | 将 FPCT-1 前瞻拆为 1A protocol、1B support audit、1C oracle | 在自然 audit 前增加 data-identifiability gate，不改 primary hypothesis |
| FPCT-D010 | 2026-07-19 | Pair universe 固定为 3 hetero + Qwen3 same-tokenizer control | same-tokenizer 永不作为 primary pilot |
| FPCT-D011 | 2026-07-19 | v1 historical：推荐 gate-primary `S>=2/3` | never executed；由 FPCT-D016/D017 在自然数据前 supersede |
| FPCT-D012 | 2026-07-19 | v1 historical：selection proposal 使用 simultaneous support LCB | never executed；由 FPCT-D019/D021 与 v2 ranking 在自然数据前 supersede |
| FPCT-D013 | 2026-07-19 | Zero-support 独立 stratum，禁止伪 candidate | metrics null，排除 mechanism population，共享既有 native fallback；正常 `(-1,0)` padding 非 integrity failure |
| FPCT-D014 | 2026-07-19 | v1 historical：区分 exact control 与 high-cardinality gate | exact-control 部分保留；`m=2` 当前由 FPCT-D018 明确为 mechanism-positive |
| FPCT-D015 | 2026-07-19 | 用户显式授权将原样 `math.md` 纳入本次 research branch commit | 文件仍为 non-normative；不覆盖 preregistration，不授权实现或 FPCT-1B |
| FPCT-D016 | 2026-07-19 | `primary_structural_m2: m>=2` 成为唯一 primary structural support | uniform 下等价 `S>=1/2`、`N_eff>=2`；headline indicator `D_s` 基于此定义 |
| FPCT-D017 | 2026-07-19 | `high_cardinality_m3` 与 `strict_m4` 仅 sensitivity/enrichment | 不否决 pair、不改变 ranking、不替代 headline ceiling |
| FPCT-D018 | 2026-07-19 | `m=2` 是 mechanism-positive low-cardinality stratum | 只有 sample 内所有 eligible parents `m<=1` 才是 exact `F=C_post` control |
| FPCT-D019 | 2026-07-19 | FPCT-1B 不进行 accuracy effect 或 scientific power kill-test | distinct content group；ordinary Wilson 描述；Bonferroni LCB sensitivity-only |
| FPCT-D020 | 2026-07-19 | Formal effect/power analysis deferred | `delta_pos`、`delta_direct_all`、`n_req` 无 operative gate；等待 operator paired/seed evidence |
| FPCT-D021 | 2026-07-19 | 工程 readiness 门槛为每 task 30、pooled 100 positive groups | 只形成 `NO_SUPPORT`/`DIAGNOSTIC_ONLY`/`SINGLE_PAIR_PILOT_READY`/`CROSS_PAIR_PILOT_READY`，不是 power guarantee |
| FPCT-D022 | 2026-07-19 | v1 never executed，addendum + v2 前瞻 supersede | source commit `7207aaff...`；v1 文件原样保留 |
| FPCT-D023 | 2026-07-19 | v1 的 commit/push 禁令仅描述当时边界；用户随后单独授权并推送 `7207aaff...` | 不视为历史错误；不追溯授权 audit、forward、GPU、训练或 FPCT-1B |

## 6. 已锁定决定与 deferred items

已锁定：

1. primary `primary_structural_m2`；sensitivities `high_cardinality_m3`、`strict_m4`；
2. exact control、`D_s` headline ceiling、五个互斥 parent strata；
3. distinct content-group descriptive audit、ordinary 95% Wilson、sensitivity-only Bonferroni LCB；
4. 工程 readiness 30/task + 100 pooled 与 label-free ranking。

Deferred：`delta_pos`、`delta_direct_all`、`n_req`、paired-discordance power 和 matched-training seed-variance rule。Deferred 不等于待在 FPCT-1B 结果后补写；必须等 operator evidence 存在后，以单独前瞻性人工批准锁定。

## 7. Artifact/path contract

- Tracked prefix：`FPCT_*`、本次显式批准的 non-normative `math.md`、protocol manifest `recipe/eval_recipe/fpct_<stage>/`、未来正式 config `recipe/**/fpct_factorized_transport/`、`script/analysis/fpct_*`、`test/test_fpct_*`。
- 当前另按根目录协作规范更新 `FRAMEWORK_UPDATE.md`，分别记录 FPCT-0、FPCT-1A v1 与 FPCT-1A-R amendment；这些修改只属于 FPCT 研究线。
- Untracked root：`local/{tmp,checkpoints,final_results}/fpct_factorized_transport/`。
- Operator ID：`c_pre|c_post|f`。
- 每个 formal run 必须带 `matched_group_id`、execution commit、preregistration hash、config/data-order/init hashes、seed/budget 和 evidence class。
- 禁止写入 `PHASE2A_*`、`phase2a_*` 或既有 checkpoint/result 目录。
- FPCT-0 不创建任何 `local/` artifact、recipe、代码、checkpoint 或结果目录；FPCT-1A v1 manifest 保留，FPCT-1A-R 新增 `FPCT_1A_APPROVAL_ADDENDUM.md` 与 `ambiguity_protocol_manifest_v2.json`，不创建 audit/result artifact。

## 8. 当前判定

### FPCT-0：GO only to protocol locking

FPCT-0 只授权了当前 FPCT-1A 的 protocol/manifest 工作，不授权 audit 或实现。

### FPCT-1A-R：GO

Human decisions、approval addendum、v2 manifest、operative provenance 与 schema 已锁定。该 GO 只完成 protocol revision。

### FPCT-1B：NOT AUTHORIZED

不得运行 tokenizer/alignment audit，不得 materialize 新逐样本 support，不得自动进入。

### FPCT-1C：NOT AUTHORIZED

不得运行 reference oracle、tiny-tensor numerical work 或实现前 correctness work。

### FPCT-2 及以后：NOT AUTHORIZED

没有模型代码、forward、GPU、训练或 checkpoint 权限。
