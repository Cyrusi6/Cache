# FPCT 状态

> 当前阶段：FPCT-3.8/3.9 production hardening and real-Qwen CPU integration
> 当前判定：`CPU/HF HARDENING GO`；FPCT-3.5P provenance-confirmed，FPCT-3.7-R1 single-pair ready
> 下一阶段：先 commit/push hardening，再冻结 confirmatory/GPU/K8s manifest；GPU 尚未启动
> 更新时间：2026-07-20（Asia/Shanghai）

## 1. 隔离身份

| 项目 | 冻结值 |
|---|---|
| 原 Phase 2A-1 工作树 | `/home/lijunsi/projects/Cache` |
| 原工作树 branch | `main`；未切换 |
| 原工作树检查时 HEAD | `9fa1f0ac3bedefd282961a853278ab88fb376fa2` |
| FPCT worktree | `/home/lijunsi/projects/Cache-fpct-factorized-transport` |
| FPCT branch | `research/fpct-factorized-transport` |
| FPCT base SHA | `9fa1f0ac3bedefd282961a853278ab88fb376fa2` |
| FPCT-OVERNIGHT starting SHA | `d296a18be9cc3b0dce3c07f4c2d7244145f2e3ac` |
| FPCT-3.5 pre-data execution SHA | `0398d26b63e96263b813730368275ee66e313f66` |
| FPCT-3.7 corrected execution SHA | `b11a046597b2466c1c6ba95c4d3693e76523c3b3` |
| FPCT-3.7 pre-audit lock SHA256 | `311ddf36bc0ab598ec52eae5236ad14f007a4645373200d58a301c9fcfd9cdb5` |
| FPCT-3.7 freeze-time local/upstream | `b11a046...` / `b11a046...`；后续 failure-report commit 不修改 frozen code/config/tests |
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

本次 `FPCT-CPU-GATE` 已将当前授权提升到 CPU-only `R1`：

- 执行 FPCT-1B 本地 tokenizer、canonical prompt 和 production alignment audit；
- 执行纯 tiny-tensor FPCT-1C reference forward/autograd；
- 在 FPCT-1B READY 且 FPCT-1C GO 时，条件式执行 FPCT-2/3 CPU production path 与 tests；
- 基于冻结的真实 `m` 分布进行 CPU 静态资源估算；
- 在 FPCT-2/3 GO 后生成 FPCT-4 non-operative GPU pilot draft；
- 只在当前 `research/fpct-factorized-transport` commit 并 push。

本次 `FPCT-OVERNIGHT` 进一步提供条件式 R1–R4 授权，但严格受 correctness gate 约束。Corrected execution commit `b11a046...` 已在 certified-support audit 前推送并冻结；首次自然 invocation 随即暴露 runtime import provenance mismatch，因此 FPCT-3.7 按硬规则判定 `INCONCLUSIVE`，该 execution revision 的全部下游条件授权失效。

本次 `FPCT-OVERNIGHT-R1` 已人工授权一个全新的 prospective revision。旧 `b11a046...` execution、`e394321...` failure record、旧 lock 与 local artifacts 保持不可变；新 revision 不补 `PYTHONPATH`、不卸载旧 editable、不复用旧 shard。当前只完成 sealed bootstrap、regular package、protocol diff、replay/audit targets 与结果前 hostile subprocess tests；自然 replay/audit 必须等待 clean commit push 后开始。

首个 prospective commit `9e501d7...` 成功生成 pre-data lock，但三个 shard 均在自然路径读取前因 stable fingerprint mismatch hard-fail。差异仅来自 Torch 每进程生成的 `/tmp/tmpXXXX/_remote_module_non_scriptable.py` 目录名；0 natural rows、0 shard artifact。该 execution 不继续使用。新 prospective amendment 记录完整随机路径，并只在生成源码 SHA、cache 结构和无 foreign `rosetta` 候选全部验证后规范化目录名；hostile suite 增至 21/21。

下列操作在本 execution revision 中已被硬停止：

- 运行任何 Hugging Face/LLM model forward；
- 使用 GPU 或提交 Kubernetes FPCT job；
- 训练、评测模型或修改 checkpoint；
- 加载模型权重或 checkpoint；

全程仍明确禁止：

- 查看/使用 Phase 2A-1 未公开结果；
- 创建 PR、合并 `main` 或 rebase；
- 修改 `/home/lijunsi/projects/Cache`、其他 worktree 或 `math.md`。

## 3. 阶段状态表

资源等级和完整 GO/INCONCLUSIVE/NO-GO 规则见 `FPCT_PREREGISTRATION.md`。

| 阶段 | 当前状态 | 依赖 | 资源等级 | 当前授权 | 决策摘要 |
|---|---|---|---|---|---|
| FPCT-0 | `GO` | 固定 base SHA | R0 | 已完成 | GO only to protocol locking |
| FPCT-1A | `SUPERSEDED BEFORE NATURAL DATA` | FPCT-0 GO | R0 | 历史完成 | never-executed v1 保留 |
| FPCT-1A-R | `GO` | FPCT-1A v1 + source commit `7207aaff...` | R0 | 是，仅 prospective amendment | human decisions locked；v2 operative |
| FPCT-1B | `SINGLE_PAIR_PILOT_READY` | FPCT-1A-R GO + 本次显式授权 | R1 CPU | 已完成 | TinyLlama 唯一 ready/rank-1；无 integrity failure |
| FPCT-1C | `GO` | FPCT-1B complete，无 integrity failure | R1 CPU | 已完成 | reference equations/oracle；19 tests pass |
| FPCT-2 | `GO` | FPCT-1B READY + FPCT-1C GO | R1 CPU | 已完成 | shared candidate fuser + sidecar/packed global attention |
| FPCT-3 | `GO` | FPCT-2 GO | R1 CPU | 已完成 | targeted 52 pass；CPU-safe full suite 288 pass |
| FPCT-3.5 | `GO` | pre-data SHA `0398d26...` | R1 CPU | 已完成 | 7,265/7,265 identity；802 raw anomalies 全为 receiver-offset overlap alias |
| FPCT-3.6 | `GO` | FPCT-3.5 identity forensic GO | R1 CPU | 已完成 | commit `b11a046...`；exact_identity + common sanitizer；102 tests pass |
| FPCT-3.7 | `INCONCLUSIVE` | pushed/frozen `b11a046...` | R1 CPU | 已停止 | wrong editable `rosetta` resolved under script mode；0 shards/0 rows；不原地重跑 |
| FPCT-3.5P | `PROVENANCE_CONFIRMED` | clean/pushed `7aecf23...` | R1 CPU | 已完成 | 7,265/802/104/410/56/401 与 ordered/context projection 精确复现 |
| FPCT-3.7-R1 | `SINGLE_PAIR_PILOT_READY` | FPCT-3.5P exact replay | R1 CPU | 已完成 | 12 shards verified；TinyLlama 511/228/2495，Qwen exact identity |
| FPCT-3.8 | `GO` | certified TinyLlama readiness | R1 CPU | 已完成 | reusable vectorized layout；hot path 无 host-sync API/parent loop |
| FPCT-3.9 | `GO` | FPCT-3.8 GO | R1 CPU | 已完成 | actual Qwen3 eager/DynamicCache random-config integration；360 full-suite pass |
| FPCT-4 | `PRE-OUTPUT RUN LOCK READY` | FPCT-3.9 GO | R2+ | 条件授权 | scientific SHA、image digest、sidecar/model/data hashes 与 sealed container probe 已冻结；GPU 尚未启动 |
| FPCT-5 | `BLOCKED / NOT ENTERED` | FPCT-4 GO | R2 | 否 | 未运行 fixed-checkpoint diagnostic |
| FPCT-6 | `BLOCKED / NOT ENTERED` | FPCT-5 GO | R3 | 否 | 未运行 matched-training smoke |
| FPCT-7 | `BLOCKED / NOT ENTERED` | FPCT-6 GO | R3 | 否 | 未冻结 confirmatory execution |
| FPCT-8 | `BLOCKED / NOT ENTERED` | FPCT-7 GO | R4 | 否 | 未启动 12-seed training |
| FPCT-9 | `BLOCKED / NOT ENTERED` | FPCT-8 GO | R4 | 否 | 未运行 model-selection/held-out evaluation |
| FPCT-10 | `BLOCKED / NOT ENTERED` | FPCT-9 decision | R5 | 否 | 未运行 mechanism/event re-audit |
| FPCT-11 | `BLOCKED / NOT ENTERED` | FPCT-10 decision | R0/R5 | 否 | 仅记录当前 claim boundary |

本次 prompt 已提供上述条件授权；任何条件不满足时立即停止，不自动扩大范围。

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
- [x] 人工显式授权并完整执行 FPCT-1B。
- [x] Commit A `7f8af719...` 在自然 audit 前推送并作为 execution SHA。
- [x] Fit+calibration selection、pilot lock、reporting 与独立 verify 完成。
- [x] FPCT-1C reference oracle 的 forward/gradient/degeneration/invariance tests 全部通过。
- [x] FPCT-2 nuisance callgraph 与 production seam 无歧义。
- [x] FPCT-2/3 candidate sidecar、packed global attention、config plumbing 与 CPU tests 完成。
- [x] Legacy-default/state-dict regression 通过；未新增 F-only 参数。
- [x] 基于 FPCT-1B 真实 `m` 分布完成 CPU 静态 expansion/cache/FLOP 估算。
- [x] 仅生成 FPCT-4 non-operative draft；未生成 Kubernetes Job，未启动 GPU。

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
| FPCT-D024 | 2026-07-19 | Commit A `7f8af71968a39bc6cba2e4e34de762b291cda834` 定义为 FPCT-1B execution SHA | prepare 只生成 hash-only split/provenance；Commit A 推送并确认 local/upstream 相同后才 freeze |
| FPCT-D025 | 2026-07-19 | FPCT-1B=`SINGLE_PAIR_PILOT_READY`，唯一 ready/rank-1/selected pair 为 `tinyllama` | selection 只用 fit+calibration distinct groups；TinyLlama 三任务 positive groups 511/228/2495；其他 heterogeneous pairs 不过门槛 |
| FPCT-D026 | 2026-07-19 | Reporting split 不改变 pilot lock，same-tokenizer control 不参与 ranking | verifier 对 60 rows、Wilson、provenance、deterministic reduction 全部通过 |
| FPCT-D027 | 2026-07-19 | FPCT-1C=`GO` | 19 个 pure-tensor oracle tests 在预批准 float64/float32/gradcheck tolerance 下通过；未放宽 tolerance |
| FPCT-D028 | 2026-07-19 | FPCT-2 production seam 采用 candidate sidecar + ambiguous-only packed global attention | children 继承 parent mask/bias；`A` 只以 `log A` 使用一次；不伪造 DynamicCache position |
| FPCT-D029 | 2026-07-19 | Parent nuisance 在 `C_post/F` 中只计算/采样一次并广播 | entropy/confidence/legacy gate/Gumbel 保持 parent-level；candidate-specific nonlinear core 共享 |
| FPCT-D030 | 2026-07-19 | FPCT-2/3=`GO` | targeted 52 pass；CPU-safe full suite 288 pass；default/c_pre state_dict 与 legacy gather regression 通过 |
| FPCT-D031 | 2026-07-19 | FPCT-4 只形成 non-operative draft | seed、budget、effect/power、fp16/bfloat16 tolerance 与 GPU resource ceiling 必须另行人工前瞻批准 |
| FPCT-D032 | 2026-07-20 | FPCT-3.5 在任何新自然 anomaly ledger 前冻结 exact runtime identity、offset taxonomy 与 certified-support oracle | same-tokenizer raw soft-span 只作 historical diagnostic；exact identity 必须 hard K=1，失败即阻断 GPU |
| FPCT-D033 | 2026-07-20 | Tokenizer behavior fingerprint 排除 model config/generation config，但保留其独立 provenance | exact identity 比较 backend/vocab/added/special/chat template/tokenizer files；`name_or_path` 只用于路径审计 |
| FPCT-D034 | 2026-07-20 | Certified one-to-many 必须穷尽全部 positive-overlap source tokens | `positive_overlap_counts` 与 retained legal candidates 不相等时不得 certify；source truncation 后必须重验 |
| FPCT-D035 | 2026-07-20 | Qwen3 runtime identity 通过，raw soft-span 802 parents 全部是 `duplicate_or_overlap_receiver_offsets` | fit+cal 精确复现 56 groups/410 m2；无 path mix-up 或 unexplained row |
| FPCT-D036 | 2026-07-20 | Qwen3 与 Qwen2.5 raw anomaly sets 逐行完全相等，但 behavior fingerprint 与路径不同 | groups/parents/offset/candidate-ID Jaccard 均 1；解释为共享 offset/token behavior，不是错误 tokenizer object |
| FPCT-D037 | 2026-07-20 | `exact_identity` 成为 same-tokenizer hard K=1 production strategy | rendered/IDs/offset/content/range/fingerprint 任一不等即 hard error；不 fallback |
| FPCT-D038 | 2026-07-20 | `certified_slot0_v1` 只在显式 FPCT recipe 中共同作用于三臂 | uncertified m>=2 使用 raw slot0 one-hot；truncate 后重验并重算 parent entropy；legacy default 不变 |
| FPCT-D039 | 2026-07-20 | FPCT-3.7 execution `b11a046...` 判定 `INCONCLUSIVE` | script-mode import 命中 `/home/lijunsi/projects/KVcache/C2C/rosetta` 而非冻结 worktree；0 shard/0 row；禁止补 `PYTHONPATH` 原地重跑 |
| FPCT-D040 | 2026-07-20 | R1 使用 regular `rosetta` package + absolute `python -I` bootstrap，而非临时 `PYTHONPATH` | 不修改全局 editable；同进程冻结完整 module closure、origin/SHA/signature 与 pre/post target fingerprint |
| FPCT-D041 | 2026-07-20 | FPCT-3.5P/3.7-R1 scientific fields 零变化 | exact identity、certifier、`certified_slot0_v1`、top-k、readiness/ranking/resource formulas 全部继承；新增 geometry/exposure 仅描述性 |
| FPCT-D042 | 2026-07-20 | `9e501d7...` pre-data lock 后的三个 shard 在 0-row 时因随机 Torch remote-module temp path 停止 | 不放宽 closure；完整路径继续记录，stable identity 改为验证后的 generated-source SHA；旧 execution 不复用 |
| FPCT-D043 | 2026-07-20 | FPCT-3.5P=`PROVENANCE_CONFIRMED` | historical 与 sealed replay 的 ordered/multiset atoms、context、802 parents、104 groups、401 clusters 全等 |
| FPCT-D044 | 2026-07-20 | FPCT-3.7-R1=`SINGLE_PAIR_PILOT_READY`，selected=`tinyllama` | fit+cal groups 511/228/2495；Qwen exact identity 全 split hard K=1；12 shards independent verify |
| FPCT-D045 | 2026-07-20 | FPCT-3.8/3.9 CPU/HF hardening=`GO` | reusable vectorized packed layout、replicated collapse、dropout identity、actual Qwen3 DynamicCache/GQA/MQA；full 360 pass |
| FPCT-D046 | 2026-07-20 | 正式 performance 设计锁定为 TinyLlama→Qwen3 单 pair、12 seeds、三臂 64-step matched triplets | `F-C_post` 唯一 headline；36 个 step-64 checkpoint 全部完成前不释放 performance |
| FPCT-D047 | 2026-07-20 | exact sign-flip 的假设边界按 R1 前瞻修正 | 检验 sharp/symmetric sign-exchangeability；mean estimand 与 +1pp practical gate 独立；固定报告 paired-t 与 exact-sign sensitivities |
| FPCT-D048 | 2026-07-20 | 正式执行采用两层锁 | 先提交 scientific code/protocol，再以该 commit 构建 digest-addressed image 并提交 run lock；任一 scientific-code 变更使 image/run lock 失效 |
| FPCT-D049 | 2026-07-20 | Operative scientific SHA=`850b9d1a...`，image digest=`sha256:447ab481...` | 独立 FP32 oracle + exact-null floor；sealed no-model fingerprint `a56ee395...`；sidecar SHA `48caee80...` |
| FPCT-D050 | 2026-07-20 | 正式 K8s hardware pool 收窄到 `4090-48gx2`，seed parallelism=1 | pre-output probes 证明 `/netdisk` 仅该节点可见，`/home/lijunsi` 不含冻结资产；不在 lock 后复制模型/sidecar 到其他节点 |
| FPCT-D051 | 2026-07-20 | 首次 synthetic GPU gate artifact 作废并从 gate 重启 | 数值 tolerance 全通过，但 activation floor 错由 BF16 probability row-sum error 推导；在任何 pretrained output 前改为 independent FP32 reference + replicated/m≤1 output-null，禁止使用旧 floor `0.024414` |
| FPCT-D052 | 2026-07-20 | Pretrained smoke resource gate 改为 7-repeat median/p95 + CUDA profiler | 单次 latency 占位在任何 pretrained forward 前删除；硬门保持 median≤1.50、p95≤1.75、无 profiled host sync |

## 6. 已锁定决定与 deferred items

已锁定：

1. primary `primary_structural_m2`；sensitivities `high_cardinality_m3`、`strict_m4`；
2. exact control、`D_s` headline ceiling、五个互斥 parent strata；
3. distinct content-group descriptive audit、ordinary 95% Wilson、sensitivity-only Bonferroni LCB；
4. 工程 readiness 30/task + 100 pooled 与 label-free ranking。

历史 FPCT-1A-R 中 deferred 的 formal effect/power gate 仍不回填到 structural-support audit。本次 `FPCT-OVERNIGHT-R1` 已另行前瞻锁定 confirmatory seeds `45–56`、64-step matched budget、final-step-64 checkpoint rule、BF16/FP16 numerical tolerances、resource ceilings、futility gate、held-out thresholds、sign-flip assumption boundary 和 stopping/retry rules；这些不能根据后续输出修改。尚待 execution run lock 冻结的是 container image digest、实际 node/imageID、model file hashes 和由 synthetic null 机械生成的 activation floor，均必须发生在对应 pretrained 输出之前。

## 7. Artifact/path contract

- Tracked prefix：`FPCT_*`、本次显式批准的 non-normative `math.md`、protocol manifest `recipe/eval_recipe/fpct_<stage>/`、未来正式 config `recipe/**/fpct_factorized_transport/`、`script/analysis/fpct_*`、`test/test_fpct_*`。
- 当前另按根目录协作规范更新 `FRAMEWORK_UPDATE.md`，分别记录 FPCT-0、FPCT-1A v1 与 FPCT-1A-R amendment；这些修改只属于 FPCT 研究线。
- Untracked root：`local/{tmp,checkpoints,final_results}/fpct_factorized_transport/`。
- Operator ID：`c_pre|c_post|f`。
- 每个 formal run 必须带 `matched_group_id`、execution commit、preregistration hash、config/data-order/init hashes、seed/budget 和 evidence class。
- 禁止写入 `PHASE2A_*`、`phase2a_*` 或既有 checkpoint/result 目录。
- FPCT-1B 详细 artifact 位于 `local/final_results/fpct_factorized_transport/fpct_1b_ambiguity_support/rev_7f8af71968a39bc6cba2e4e34de762b291cda834/`，不提交；tracked result manifest 只记录 SHA、row count 与 byte size。
- 同目录下 `resource_estimate.json` 为不提交的 CPU 静态资源明细；tracked 摘要为 `FPCT_CPU_RESOURCE_ESTIMATE.md`。

## 8. 当前判定

### FPCT-0：GO only to protocol locking

FPCT-0 只授权了当前 FPCT-1A 的 protocol/manifest 工作，不授权 audit 或实现。

### FPCT-1A-R：GO

Human decisions、approval addendum、v2 manifest、operative provenance 与 schema 已锁定。该 GO 只完成 protocol revision。

### FPCT-1B：SINGLE_PAIR_PILOT_READY

完整 CPU structural-support audit 与独立 verify 通过。TinyLlama 是唯一 ready pair，因此只允许 single-pair pilot，不允许 cross-pair confirmatory claim。

### FPCT-1C：GO

Reference equations、flat/hierarchical、退化、refinement/permutation、mask、global denominator、forward/gradient、Jensen 与 future-general `g=0` recovery 全部通过预批准 tolerance。该 GO 仅表示数学/数值 reference 正确。

### FPCT-2/3：GO

Production seam、shared candidate fuser、parent nuisance broadcast、ambiguous-only packed global attention、GQA/MQA、gradient、default/state-dict/config regression 均通过 CPU tests。没有运行 HF/LLM forward，因此该 GO 不构成 real-model activation 或 accuracy 证据。

### FPCT-4：SCIENTIFIC CODE LOCK IN REVIEW / GPU NOT STARTED

旧 CPU-GATE 生成的 non-operative draft作为历史记录保留。当前已冻结 single-pair confirmatory preregistration、GPU numerical protocol、K8s protocol、50,000-replicate hierarchical statistics、4096 sign-flip/paired-t/sign-test code、一次性 split firewall controller、matched triplet runner、immutable image 和 K8s templates。首次 scientific lock `92724ec...` 后的镜像复核发现每臂重复 alignment 不满足 frozen-sidecar 复用合同；后续 candidate image 又在模型加载前分别因 evaluator closure dependency 和 Python realpath 硬停。首个 synthetic gate 虽通过 tolerance，但因 activation floor 数量使用错误在 pretrained output 前作废。最终 scientific SHA 为 `850b9d1a...`，image digest 为 `sha256:447ab481...`，sealed probe fingerprint 为 `a56ee395...`；2048 sidecar SHA 为 `48caee80...`。所有 K8s stages 均有 init probe 并固定到唯一可见冻结资产的双 GPU worker。当前没有 pretrained model output、训练或 accuracy；下一步是重跑 corrected synthetic GPU numerical gate。

### FPCT-3.5：GO

Pre-data commit `0398d26...` 已在自然结果前推送。全量 forensic 证明 Qwen3 runtime identity；802/802 raw m2 rows 均由重叠 receiver offsets 产生，fit+cal 为 410 parents/56 groups。Qwen3/Qwen2.5 四类 row sets 全等且路径/fingerprint distinct，无 tokenizer mix-up。

### FPCT-3.6：GO

生产 aligner 已增加 hard `exact_identity`；dataset/evaluator 已增加 opt-in `certified_slot0_v1` sanitizer。Synthetic、production aligner、audit finalize/independent-verify round trip 与既有回归合计 `102 passed, 2 warnings`。这些已由 corrected execution commit `b11a046...` 推送并冻结。

### FPCT-3.7：INCONCLUSIVE

Pre-audit lock 正常生成，但 TinyLlama/ARC 与 Llama3.2/ARC 的首次调用都在进入 alignment method 前失败。Conda editable mapping 将 script-mode `rosetta` 解析到旧 `/home/lijunsi/projects/KVcache/C2C/rosetta`，其方法签名缺少冻结调用参数。没有发布 shard、row 或任何 support/resource 统计。根据 freeze 后不修科学执行合同的规则，本 revision 不增加 `PYTHONPATH`、不 patch、不重跑；FPCT-3.8 及以后全部未进入。

## 9. FPCT-CPU-GATE 最终隔离复查

- `/home/lijunsi/projects/Cache` 保持 `main`，HEAD `a320777ee3d8e2c5fbf988ad6cd840b560aab28b`，复查时 clean。
- `/home/lijunsi/projects/Cache-phase2a2-cache-geometry` 保持 `research/phase2a2-cache-geometry`，复查时 HEAD `00db4c7eeffc57a852c67fd1aedad9fd823ca528` 且 clean。该 worktree 在本任务期间由并行活动从操作前记录的 `b1748bd...` 前进；FPCT 未在其中执行写操作、切换分支或读取结果文件。
- `/home/lijunsi/projects/Cache-phase2a2-equivalence-debug` 保持 `research/phase2a2-equivalence-debug`，复查时 HEAD `f1059dee343969661bb9492f0231d9bb58261706` 且 clean；FPCT 未在其中执行写操作、切换分支或读取结果文件。
- 当前 diff 不包含 `PHASE2A_*`/`phase2a_*` 文件；v1 protocol、v1/v2 manifest、approval addendum 与 `math.md` 均无 diff。
- `math.md` SHA256 仍为 `98d1b61f84d046548d5ba0070d6858c7080cb14fdef9169b08ad167461b809ad`；operative v2 manifest SHA256 仍为 `f7c8bd7fbc456484d1a40ca88d32dc8da3104c422a5addd89f7d033b12c82511`。
