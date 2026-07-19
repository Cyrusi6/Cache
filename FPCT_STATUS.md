# FPCT 状态

> 当前阶段：FPCT-1A——Ambiguity Support Audit Protocol Locking
> 当前判定：`REVIEW REQUIRED`
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
- commit 或 push。

## 3. 阶段状态表

资源等级和完整 GO/INCONCLUSIVE/NO-GO 规则见 `FPCT_PREREGISTRATION.md`。

| 阶段 | 当前状态 | 依赖 | 资源等级 | 当前授权 | 决策摘要 |
|---|---|---|---|---|---|
| FPCT-0 | `GO` | 固定 base SHA | R0 | 已完成 | GO only to protocol locking |
| FPCT-1A | `REVIEW REQUIRED` | FPCT-0 GO | R0 | 是，仅 protocol/manifest | gate threshold、support estimand 与参数待人工批准 |
| FPCT-1B | `NOT AUTHORIZED` | FPCT-1A GO + 显式授权 | R1 CPU | 否 | label-free ambiguity support audit |
| FPCT-1C | `NOT AUTHORIZED` | FPCT-1B decision + 人工授权 | R1 | 否 | reference equations/oracle/numerical rule |
| FPCT-2 | `NOT AUTHORIZED` | FPCT-1B `LIMITED GO`/`CROSS-PAIR GO` + FPCT-1C GO + 人工授权 | R1 | 否 | 实现 `C_post`/`F` 与 manifest plumbing |
| FPCT-3 | `NOT AUTHORIZED` | FPCT-2 GO | R1 | 否 | CPU 数值、mask、gradient、退化测试 |
| FPCT-4 | `NOT AUTHORIZED` | FPCT-3 GO | R0/R1 | 否 | 冻结 task/control/seed/budget/statistical manifest |
| FPCT-5 | `NOT AUTHORIZED` | FPCT-4 GO | R2 | 否 | fixed-checkpoint diagnostic |
| FPCT-6 | `NOT AUTHORIZED` | FPCT-5 GO | R3 | 否 | matched-training pipeline smoke |
| FPCT-7 | `NOT AUTHORIZED` | FPCT-6 GO | R3 | 否 | confirmatory-disjoint development-only pilot |
| FPCT-8 | `NOT AUTHORIZED` | FPCT-7 GO | R4 | 否 | 正式多 seed matched retraining |
| FPCT-9 | `NOT AUTHORIZED` | FPCT-8 GO | R4 | 否 | 执行冻结 evaluation |
| FPCT-10 | `NOT AUTHORIZED` | FPCT-9 decision | R5 | 否 | mechanism/event re-audit/replication |
| FPCT-11 | `NOT AUTHORIZED` | FPCT-10 decision | R0/R5 | 否 | claim freeze 与归档 |

FPCT-1A 即使获 `GO`，也不构成 FPCT-1B 授权；所有后续阶段都需要显式人工授权。

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
- [ ] 人工审查并批准尚未锁定的设计决策。
- [ ] 人工显式授权 FPCT-1B；当前不授权。

### 4.1 FPCT-1A protocol checklist

- [x] FPCT-0 headline 与第一轮冻结项一致性通过。
- [x] 冻结三个 heterogeneous pairs 与 Qwen3 same-tokenizer control。
- [x] 冻结三任务 input/prompt/split universe 与 provenance。
- [x] 冻结合格 parent/candidate、mask-first renormalization 与 zero-support contract。
- [x] 冻结 `m`、`N_eff`、`A_max`、`S`、sample/pair support 与 ceiling。
- [x] 提出 gate-primary/sensitivity threshold grid，并披露 uniform degeneracy。
- [x] 冻结 parametric support-floor formula、label-free score、tie-break 与 cross-pair gates。
- [x] 冻结 FPCT-1B CSV/JSON schema、aggregation 与排序。
- [x] 未运行自然 ambiguity audit、tokenizer/alignment 或 model forward。
- [ ] 人工批准 recommended gate-primary threshold。
- [ ] 人工批准 support-floor estimand 与 practical/statistical parameters。

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
| FPCT-D011 | 2026-07-19 | 推荐 gate-primary `S>=2/3`；sensitivities `S>=1/2`、`S>=3/4` | uniform/top-k4 下分别对应 m>=3、m>=2、m=4；不回改 formal `m>=2` population；需人工批准 |
| FPCT-D012 | 2026-07-19 | Pair selection 只用 fit+cal label-free simultaneous support LCB | distinct content-group unit；3×3 Bonferroni；禁止 accuracy/test/Phase2A outcome；固定 tie-break |
| FPCT-D013 | 2026-07-19 | Zero-support 独立 stratum，禁止伪 candidate | metrics null，排除 mechanism population，共享既有 native fallback；正常 `(-1,0)` padding 非 integrity failure |
| FPCT-D014 | 2026-07-19 | Exact direct support 与 gate-effective support 分开 | 只有全体 parent `m<=1` 才保证 fixed-state `F=C_post`；含 `m=2` 的 below-threshold stratum 非数学恒等 control |
| FPCT-D015 | 2026-07-19 | 用户显式授权将原样 `math.md` 纳入本次 research branch commit | 文件仍为 non-normative；不覆盖 preregistration，不授权实现或 FPCT-1B |

## 6. 尚未锁定、需人工决定

FPCT-1A 只剩以下人工批准项：

1. 是否批准 recommended gate-primary `primary_s67: S>=2/3` 及两个 sensitivity definitions；
2. 是否批准 support-floor estimand `task_macro_cluster`；
3. support floor 的 `delta_pos`、`delta_direct_all`、alpha、power 与 familywise Wilson LCB confidence。

批准必须发生在 FPCT-1B 任何自然统计之前。后续训练/accuracy 阶段的 seed、budget、task accuracy rule 等仍未授权，不属于本次批准范围。

## 7. Artifact/path contract

- Tracked prefix：`FPCT_*`、本次显式批准的 non-normative `math.md`、protocol manifest `recipe/eval_recipe/fpct_<stage>/`、未来正式 config `recipe/**/fpct_factorized_transport/`、`script/analysis/fpct_*`、`test/test_fpct_*`。
- 当前另按根目录协作规范更新 `FRAMEWORK_UPDATE.md`，分别记录 FPCT-0 隔离/预注册与 FPCT-1A protocol locking；这些修改只属于 FPCT 研究线。
- Untracked root：`local/{tmp,checkpoints,final_results}/fpct_factorized_transport/`。
- Operator ID：`c_pre|c_post|f`。
- 每个 formal run 必须带 `matched_group_id`、execution commit、preregistration hash、config/data-order/init hashes、seed/budget 和 evidence class。
- 禁止写入 `PHASE2A_*`、`phase2a_*` 或既有 checkpoint/result 目录。
- FPCT-0 不创建任何 `local/` artifact、recipe、代码、checkpoint 或结果目录；FPCT-1A 只创建 required protocol manifest `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest.json`，不创建 audit/result artifact。

## 8. 当前判定

### FPCT-0：GO only to protocol locking

FPCT-0 只授权了当前 FPCT-1A 的 protocol/manifest 工作，不授权 audit 或实现。

### FPCT-1A：REVIEW REQUIRED

Protocol、候选 threshold grid、support-floor derivation、split/pilot/zero-support/output contracts 已完整；但 gate-primary threshold、support-floor estimand 和 practical/statistical parameters 尚未人工批准，因此不能判 `GO`。

### FPCT-1B：NOT AUTHORIZED

不得运行 tokenizer/alignment audit，不得 materialize 新逐样本 support，不得自动进入。

### FPCT-2 及以后：NOT AUTHORIZED

没有模型代码、forward、GPU、训练或 checkpoint 权限。
