# FPCT-E1 状态

> 当前阶段：E1-FAST-RCA prospective pre-output lock
> 当前状态：`PRE-OUTPUT TARGETED GATE GO / COMMIT+PUSH PENDING / MODEL OUTPUT=0`
> 下一步：commit/push 当前 lightweight executor；随后创建 18-shard fixed-checkpoint plan
> 更新时间：2026-08-20（Asia/Shanghai）

## 2026-08-20 E1-FAST-RCA 路线切换

- A5R7/A5R8 的 30,370,816-row input-lock 路线停止；已有 root 保留为工程记录，
  不追认为 deep-verifier GO，不继续 resume/rebuild。
- 前瞻性 operative addendum 为 `FPCT_E1_FAST_RCA_AMENDMENT.md`，machine manifest 为
  `recipe/eval_recipe/fpct_e1_fast_rca/fast_rca_manifest.json`。
- 只按精确 SHA 复用 25.5MB compact E0-design input cache；不读取或物化 30M row
  templates。Population 固定为 ARC/OpenBookQA/MMLU-Redux=`128/70/128`。
- 六个 E0 checkpoints × 三 tasks 构成 18 个可并行 shard；每个 shard 内按冻结顺序
  比较 C_post、F、centered λ、K-only、parent-mass 和 partition composition。
- Exact RoPE correction 在固定 checkpoint 的不可分离 nonlinear C2CProjector 上结构性
  不可执行，已在自然输出前标记为不可选择；禁止以近似旋转冒充 math.md 公式。
- 当前没有运行新 model forward/GPU/K8s/training；E1-pilot、model-selection、test、
  confirmatory、native null、跨模型和 36-run 仍 sealed/not authorized。
- Targeted CPU/reference/random-small-Qwen/runtime suite=`99 passed / 0 failed`。
  CPU-safe full suite=`1082 passed / 42 historical guards failed`；42 项均来自 A4/A5R1–R7
  frozen tracked-tree/live-source gates、已记录旧 fixture API drift 或 R2l/R2m historical
  production allowlist。它们在合法 successor source 上按设计 fail-closed，不是本修订的
  operator/runtime regression，也未通过修改旧 gate 来绕过。
- 首个 pushed pre-output SHA `1c74a522...` 的 `prepare` 在 root 创建和任何 model
  output 前因 direct-file CLI 缺少 repo-root `sys.path` 而停止；目标 root 未创建。
  仅允许后继 commit 修复 CLI bootstrap，并以新 SHA/new root 重启。
- `86b4b240...` 成功生成 18-shard plan（SHA=`30f779d9...`），但在任何 model load
  前决定增加独立 one-sample/non-scientific smoke namespace；该 plan 因 executor
  successor 而封存，未运行 shard。新 plan 必须绑定后继 commit/new root。
- `70f64c78...` 的前四次 smoke 均未到 model load：r1/r2 为 node-local hostPath
  不存在，r3 为无关依赖 bootstrap backtracking，r4/r5 为容器无 Git/宿主 Git GLIBC
  不兼容。唯一后继修复是允许 manifest 注入严格 full SHA，并要求它与 plan execution
  SHA 完全一致；科学干预、input、checkpoint、endpoint 不变。
- `2c1f79a6...` r6 完成真实 model/checkpoint load 后、首个 forward 前因浅层 batch
  device mover 遇到 `messages:list[dict]` 停止；sample metrics 文件为空。后继只复用仓库
  已有递归 tensor mover 语义，旧 root/temp 不进入分析，仍需 new commit/root。

## 隔离身份

| 项目 | 冻结值 |
|---|---|
| Worktree | `/home/lijunsi/projects/Cache-fpct-e1-mechanism-audit` |
| Branch | `research/fpct-e1-mechanism-audit` |
| Base SHA | `613958af38fad27e1ea933ccc0dda6d1af5cce89` |
| E0 result | `E0_NO_GO_FOR_FURTHER_SPEND`；只读且不重新解释 |
| E0 result doc SHA256 | `632af006b37f1edfb26356d47e615def7052158fa1cbe8d944f063f8e99c1c50` |
| E0 compact artifact anchor SHA256 | `a988061a1353fb8ce6a944b6359f3a0117857013495246f009612492a1ceb42d` |
| E1 split manifest SHA256 | `030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d` |
| Superseded execution SHA | `744a943ea804dfebe4e6d3cba756b6a89763002f`；`ABANDONED_BEFORE_NATURAL_DATA`；禁止 resume/reuse |
| Failed execution SHA | `d1698177e60455a42731165b88a66374b8826718`；自然 alignment 后、artifact 前 integrity failure；禁止 resume/reuse |
| Resource-ceiling execution SHA | `612697dfc44ab46699728b8d2de0a6fce980a889`；自然 multi-task alignment 后、artifact 前 fail-closed；禁止 resume/reuse |
| Successor execution SHA | `9b248d2094b684f5d9e9a218919a354b7d97468e`；`A5_INPUT_LOCK_BLOCKED`；永久禁止 resume/reuse |
| A4 successor execution SHA | `07755a4039e89700e59af9e141026a57142f9da0`；`A4_INPUT_LOCK_BLOCKED`；永久禁止 resume/reuse |
| A5 protocol ID | `fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt` |
| A5 selected prompt contract | `ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT` |
| A5 successor execution SHA | `9b248d2094b684f5d9e9a218919a354b7d97468e`；pre-group-1 provenance failure；永久禁止 resume/reuse |
| A5R1 execution SHA | `37be816ad611b8b0d916bd98c840c5f31efe2b50`；choice-cardinality failure；永久禁止 resume/reuse |
| A5R3 terminal execution SHA | `3263531ed7137efd241de3c951bf7c5e9d37e669`；complete in-memory reduction 后、publication 前 mode failure；永久禁止 resume/reuse |
| Consolidated gate SHA256 | `d17b4b7f35e2384abfbd300cf109484d6aacc91dd84cd3f2573a8d5dd36d4b17` |

## 人工决策锁

2026-07-26 用户在任何 E1-pilot forward/outcome 之前批准：

- E1-pilot source：support-fit-only certified groups；
- counts：ARC 128 / OpenBookQA 70 / MMLU-Redux 128；
- selection：frozen domain-separated SHA ordering；
- role：exploratory mechanism pilot；
- confirmatory eligibility：no。

已物化 652-row split manifest。E0-design 与 E1-pilot 各为 `128/70/128`，distinct content-group intersection=`0`。E1-pilot 仍未 render、tokenize、align、forward 或读取 outcome。

## 阶段表

> 下表保留 A3 resource-ceiling failure 当时的历史状态；当前 operative 状态见文末 A4 阶段表。

| 阶段 | 状态 | 依赖 | 允许资源 | 决策/边界 |
|---|---|---|---|---|
| E1-0 protocol + split lock | `GO` | E0 result commit `613958a...` + human amendment | 文档、hash-only split lock | E0 readonly；E1-pilot/confirmatory sealed |
| E1-1 instrumentation hard gate | `GO` | E1-0 GO | synthetic tensor + random-small Qwen CPU | pre-data suite `169 passed`；ON/OFF bitwise；cross-query variance>0 |
| Commit A2 execution lock | `INCONCLUSIVE / ABANDONED` | E1-1 GO + operational closure | CPU input lock | d169 在自然 alignment 中因 slot-order taxonomy bug fail-closed；0 artifact/model/GPU |
| Commit A3 correctness lock | `GO` | d169 receipt + synthetic permutation regression | 文档、单一 classifier 修复、tests only | clean/pushed execution=`612697df...`；只排序派生 intersections；未改 candidate/A/operator/threshold/split |
| E1-2 input/topology/provenance lock | `INCONCLUSIVE_RESOURCE_CEILING / ABANDONED` | clean pushed A3 | CPU tokenizer/alignment only | observed logical rows `616448 > 262144`；0 artifact/model/GPU；禁止 resume/reuse |
| E1-2 C_post baselines | `CONDITIONALLY AUTHORIZED / NOT STARTED` | A5R5 input-lock GO + independent verifier + pre-output lock | 最多 6 个 GPU workers；具体资源须在新锁中冻结 | 未创建 runtime probe/plan/ConfigMap/Job，未运行 checkpoint inference |
| E1-2 F endpoints | `CONDITIONALLY AUTHORIZED / NOT STARTED` | 18 C_post closure + same pre-output lock | 同上 | 只识别同 checkpoint F-C_post mechanism；无性能 GO |
| E1-2 analyzer/finalized receipt | `CONDITIONALLY AUTHORIZED / NOT STARTED` | C_post+F endpoints complete | CPU analysis | 必须产生 deep-verified immutable `FINALIZED_E1_2` |
| E1-3 frozen interventions | `CONDITIONALLY AUTHORIZED / NOT STARTED` | complete baseline/F closure + pre-output intervention DAG | frozen fixed-checkpoint inference only | centered λ、RoPE、K/V、parent-mass、partition hybrid；无训练 |
| E1 root-cause/operator freeze | `CONDITIONALLY AUTHORIZED / NOT STARTED` | E1-3 complete + mechanical eligibility/ranking | protocol + one production factor | 无合格 intervention 时机械停止训练 |
| E1-pilot | `SEALED / NOT RUN / NOT READ` | unique operator freeze + implementation/oracle/CPU/HF/GPU/training lock GO | 条件式 matched training | exploratory only；永无 confirmatory eligibility |
| Confirmatory | `SEALED / NOT AUTHORIZED` | 后续完整阶段链 | 未授权 | model-selection/test/formal seeds 不得读取 |

## Instrumentation hard gate

Consolidated gate 固定六项且全部为 true：formula oracles、instrumentation ON/OFF equivalence、synthetic positive query variance、multi-step accumulation、invalid probability/gradient exact zero、no NaN/Inf。冻结证据包括：

- `FPCT_E1_INSTRUMENTATION_REPORT.md`；
- `recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate.json`；
- parity SHA256=`00fb8403f1376eb0a3d4ccf498a6c7230a313aec7259f661553ca0e164a2d1fc`；
- synthetic query-variance SHA256=`f652ee99f8bce508b3f35953b10ad11ad4046ab0f868bf93fc08a82bb6ef14b2`。

最终 15-file pre-data suite 为 `169 passed`。全 FPCT CPU suite 为 `356 passed`；项目 CPU-safe full suite 为 `595 passed, 2 warnings`。另有两个历史 immutable verifier（R2l repository allowlist、R2m production hash lock）按设计拒绝后续 E1 scientific changes；未修改历史锁。

在 744a943 被前瞻 supersede 后，新增 runtime renderer/mounted receipt/expected-plan/path-disjoint/image-digest/fresh-subprocess closure；同一 15-file pre-data suite 重新执行为 `211 passed, 0 failed`，all-FPCT CPU suite=`400 passed`，项目 CPU-safe full suite=`639 passed`。该结果只证明前输出工程与 provenance 闭包，不是自然机制激活或性能证据。

## Execution lock

> 本节保留 A2/A3 当时的 execution contract 和 review 边界；A4 的后续人工批准及新执行顺序见文末，不回写历史原文。

- History：`744a943...` 只生成 exact git archive、外置 source receipt 与空 input-lock directory；0 dataset lookup/tokenization/alignment/model/GPU。它作为 `ABANDONED_BEFORE_NATURAL_DATA` 保留，不是科学 NO-GO。
- Failure：`d1698177...` 从 valid snapshot 启动 CPU input lock，已访问 MMLU-Redux/high_school_geography 的自然 row、tokenizer 与 alignment；raw-topology classifier 将合法的 top-k slot permutation 误当 span coverage failure。0 sidecar/manifest/raw artifact/model/checkpoint/GPU；状态=`INCONCLUSIVE_INTEGRITY_FAILURE`，禁止 resume/reuse。
- Correction：certified taxonomy 仅对派生 intersection 做 span sort 后检查无缝覆盖；candidate records、indices、weights、A 与 slot-0 语义保持原顺序。两/三 candidate permutation 与 ledger 端到端回归通过；未改 sanitizer、alignment、operator、threshold 或 split。A3 15-file=`213 passed`、all-FPCT=`402 passed`、项目 CPU-safe full suite=`641 passed`。
- A3 resource failure：clean/pushed `612697df...` 的 immutable source snapshot/receipt 均通过；CPU input lock 在 E0-design 多 task 的自然 lookup/tokenization/alignment 后，对至少一个样本计算出精确 logical rows=`616448`，超过预注册累计每样本上限 `262144`，于任何 input/raw/runtime/model artifact 前 fail-closed。`616448=28×16×1376`，不是 duplicate emission。精确已处理 sample/row/hash 与失败 group hash未物化，不猜测。
- Contract interpretation：`262144` 是前瞻冻结的 engineering memory-integrity guard，不是科学 effect threshold；但 operative v4 将其实现为累计 logical-row ceiling，而真正 Parquet physical batch/row-group 是 `4096`。因此不得在同一 execution 中把 `262144` 静默重解释为 chunk size，也不得按观测值提高 ceiling、截断 rows、丢弃样本或复用 partial state。
- Review gate：当前唯一推荐的继续路线是经人工明确批准的新版本 representation-preserving streaming 合同：logical row universe、row keys、统计权重与 endpoints 全部不变，physical chunks 确定性流式写出，并由新 commit/snapshot/run root 从头执行。该路线尚未获批；successor SHA 未分配。
- Source（未来、尚未授权）：human-approved new protocol → new clean pushed commit/run UID/root → exact git archive → snapshot root 内 canonical Git receipt；不得挂 live worktree或复用 744/d169/612 artifact。
- Input：仅 E0-design；逐样本 rows=`answer queries × certified parents × 28 × 16`，hard ceiling=`262144`，task-level `sum/min/p50/p95/max/argmax` 预先冻结。
- Topology：pre-sanitizer raw-to-runtime ledger 在任何 model output 前生成；uncertified rows 共同 slot-0 collapse，不携带 functional metric。
- Checkpoints：实际加载六个 immutable `final` projector trees；同时 hash `checkpoint-64`，并要求 projector set byte-identical；strict-attested load 的 missing/unexpected keys 为空。
- Runtime：exact snapshot renderer/template；host 与 Pod 内两次 mounted receipt/tree/raw-byte verification；之后才读取 immutable image Python/package/CUDA metadata。Sender/receiver full asset trees、dev-data/config/checkpoint trees均在 model load 前验证。
- Storage：Parquet-only、4096-row bounded streaming、atomic no-overwrite、recoverable exclusive claim lease。
- K8s（historical v4 contract；当前未授权）：曾要求 initial/finalized ConfigMaps immutable、mounted-byte receipt、expected plan SHA、read-only rootfs 与固定 `4090-48gx2`。A3 未创建 runtime probe、plan、ConfigMap 或 Job；未来 resource contract 待新人工 amendment。
- DAG（historical/non-operative）：`18 C_post -> marker -> 18 F -> 36 closure -> analyzer/finalized receipt -> 72 lambda additions -> 108 closure`。当前禁止启动；未来 DAG 必须重新前瞻冻结。

## Centered-λ 锁

```text
K(lambda)_ij = Kbar_i + lambda * (Ktilde_ij - Kbar_i)
V(lambda)_ij = Vbar_i + lambda * (Vtilde_ij - Vbar_i)
lambda        = {0, 0.25, 0.5, 1, 2}
```

`lambda=0` 必须等于 C_post/replicated-collapse；`lambda=1` 必须等于 F；`lambda=2` 仅作诊断。所有 λ 共用原 A、mask、native atoms、global denominator 和 checkpoint，不训练、不调参、不只报告最优 λ。

## 当前执行确认

- E0 tracked result/checkpoint 未修改；
- E1-pilot 未运行、未读取；confirmatory model-selection/test 未释放；
- d169 failed attempt 与 612697df resource-ceiling attempt 均读取过 E0-design 自然输入并运行 tokenizer/alignment；两者精确 processed count/hash 均未物化，均未生成可用 audit artifact，均禁止 resume/reuse。
- 未运行 pretrained model forward，未加载模型权重或 checkpoint；
- 未运行 GPU/Kubernetes，未训练、未新增 seed、未启动 36-run；
- E1-pilot 仍未 render/tokenize/align/run/read；confirmatory仍 sealed。
- 当前结论只说明现有 cumulative-row representation 的工程上限不足，不是机制、数学或性能结论。`744a943...`、`d1698177...` 与 `612697df...` 均不在可继续执行链上。

## 2026-07-26 A4 representation-preserving streaming 前瞻修订

### 人工授权与不变量

用户已在任何新自然 input-lock 、pretrained output 或 E1-pilot outcome 之前明确批准 `APPROVED_PROSPECTIVE_AMENDMENT_E1_A4_STREAMING`。A4 只改变物理存储和 reduction 调度，不改变：

- 完整 logical-row universe 与原 15-field row key；
- 每个 metric、weight、topology、prior、mask、estimand 和 endpoint；
- operator、checkpoint、centered-λ grid、E0-design membership 和科学 claim boundary；
- production physical chunk size=`4096`，禁止截断 logical rows、删除大样本或将累计 ceiling 事后改名为 chunk size。

历史 execution `744a943...`、`d1698177...`、`612697df...` 均永久 `ABANDONED / NON-REUSABLE`；A4 不 resume 也不复用任何 partial state。上方 A3 记录保留为当时的真实执行边界，不是需要回写的错误。

### A4 operative 阶段表

| 阶段 | 当前状态 | 依赖 | 授权资源 | 决策边界 |
|---|---|---|---|---|
| A4 v6 protocol/schema/implementation | `GO — PRE-NATURAL` | human prospective approval | 文档、schema、CPU synthetic code/tests | amendment/contract/schema 与实现已由 gate hash 闭包；不等于自然 input-lock GO |
| A4 synthetic hard gate | `GO` | v6 implementation complete | CPU synthetic only | 10/10 冻结 checks 成立；`395 passed` |
| Clean pushed Commit A4 | `GO` | synthetic hard gate GO | 当前 research branch commit/push | commit/upstream=`07755a4039e89700e59af9e141026a57142f9da0` |
| A4 source snapshot + run root | `GO / NOW ABANDONED` | clean local/upstream A4 identity | immutable snapshot/run UID/root | UID=`fpct-e1-a4-streaming-07755a40-v1`；root 永久 no-resume/no-reuse |
| Streaming CPU input lock | `BLOCKED / NO USABLE ARTIFACT` | 新 snapshot/run root | E0-design CPU tokenizer/alignment | 前 160/326 groups exact；第 161 个 ARC group prompt SHA fail-closed |
| E1-2 C_post/F mechanism audit | `BLOCKED / NOT AUTHORIZED` | complete successor input-lock GO | 无 | 0 runtime/plan/model/forward；需新的人工前瞻决策 |
| E1-3 centered-λ sweep | `BLOCKED / NOT AUTHORIZED` | immutable finalized E1-2 | 无 | 未进入 |
| E1-pilot | `SEALED / NOT RUN / NOT READ` | operator 最终冻结及独立后续授权 | 当前无 | exploratory only；confirmatory eligibility=`no` |

### v6 实现和已锁定 synthetic 证据

- Normative amendment：`FPCT_E1_STREAMING_AMENDMENT.md`。
- Machine-readable contract/schema：`recipe/eval_recipe/fpct_e1/e1_streaming_contract.json` 与 `e1_streaming_schema.json`。
- Streaming identity/storage verifier：`script/analysis/fpct_e1_streaming_verify.py`；synthetic gate runner：`script/analysis/fpct_e1_streaming_synthetic_gate.py`。
- Input-lock、runtime primitive spool/capture、analyzer 和 K8s execution-plan consumer 已改为 4096-row bounded streaming；aggregate-only capture 不保留 cumulative long-form list。
- Tests 覆盖原 row-key/reference equivalence、ordinal bijection、chunk-partition semantic equivalence、百万级 logical rows、bounded RSS、atomic no-overwrite、crash/resume、corruption/tamper fail-closed 与 analyzer deterministic reduction。
- Normative SHA256：amendment=`2fd6412a5533ecc4085e52a81e514425ae5fe6ffa64ec47ebbd7bc9669a20c74`；contract=`919d7c9955c749d2d3603a2356701c65e5b552d22b8ab8cc23c91ea7246e6f22`；schema=`5d389e81f87a18e02889204f055e61c9a7fed8957e8d990f9642558700a39618`。
- Final synthetic gate=`GO`：artifact SHA256=`42b6c98fd5f27a49e258bae4b79ff3c4673c9144465a3c43f9a672d228c82df4`，evidence SHA256=`d22bb1997234dd0c895f1b819e9cbf3bb1102633502f482cbb7b2d4d4ecc8e43`，`395 passed in 4020.78s`。
- Full-schema baseline=`4480 rows / 2 chunks / 147111936 B peak RSS`；stress=`1000384 rows / 245 chunks / 192376832 B peak RSS`；冻结 threshold=`283295744 B`，其中 canonical row bound=`2078 B`、allocator/buffer allowance=`136183808 B`。Stress 在阈值内，且 whole-table materialization 未被检测到。
- 十项 machine checks 全部成立：aggregate/chunk partition/reference row-key/weights/topology/semantic replay equivalence、bounded RSS、atomic no-overwrite、crash/resume；`whole_table_materialization_detected=false`。
- A4 instrumentation re-attestation=`GO`：`110 passed`，synthetic gamma query variance=`0.19730721414089203`；parity/synthetic/hard-gate SHA256 分别为 `d7e78e5e8b54aa2ae21e105b53ad5cbdc8b52d78b6e422c36bdeed919d40c2f3`、`7524fb2ce873dde3450ad12ec26e39d6497f0f99fd9fe755edcfafda44c2ab40`、`0cd401328c5942f51a21f98a3418eb3a18c08a489c6a5cc8c99ef6775b1d6b3e`。Historical v1 evidence 保持原样，未覆盖。
- Project CPU-safe complement（排除 gate 已覆盖的 14 个文件）=`449 passed, 2 deselected`；两项 deselected 均为绑定旧 R2l/R2m tree identity 的历史 guard，在 A4 后继分支上按设计 fail-closed（旧 diff allowlist 与旧 `fpct_attention.py` SHA），不属于 A4 runtime regression。与 gate 的 `395 passed` 合计为 `844 passed`。
- Clean pushed A4 commit / execution SHA=`07755a4039e89700e59af9e141026a57142f9da0`；该 execution 已在 CPU input-lock 中 fail-closed，不再是可继续 successor。

### A4 当前 firewall

- A4 execution 已对 E0-design 运行 CPU-only dataset lookup/chat rendering/tokenization/alignment：前 160 个 group 与历史 anchor 完全相等，第 161 个 group fail-closed；没有产生可用 input-lock artifact；
- 尚未加载 E0 pretrained model 权重或 projector checkpoint，未运行 model forward；
- 尚未运行 GPU、Kubernetes 或 training；
- E1-pilot 仍为 `SEALED / NOT RUN / NOT READ`，confirmatory 仍 sealed；
- `07755a40` run root 永久 abandoned/non-reusable；只有新的人工前瞻修订、新 clean commit/snapshot/run root 且新 input-lock=`GO` 后，才可能条件式进入 E1-2，E1-3 仍必须等待 finalized E1-2。

## 2026-07-27 A4 input-lock prompt-anchor failure closure

- Execution=`07755a4039e89700e59af9e141026a57142f9da0`；UID=`fpct-e1-a4-streaming-07755a40-v1`；root=`/netdisk/lijunsi/fpct-e1/fpct-e1-a4-07755a40-v1`。
- Generic blocked receipt SHA256=`bc9002daebd1e8921d5fd5ca0705ab59efd115b350a785e0c368d322161cdec0`；execution identity SHA256=`bdafa8df8d609b9bbf4c3468a7bb1452ec459c6e1adb5201a85ea772aa3f40df`；source receipt file SHA256=`b792803b23e22d082743d7f39aefcebd7178c5cd6e10108756f6693d7a778a8d`，其内部 `receipt_sha256`=`a5b5e87a816377fcaa4cb2080031da22532239a08c65ef5b07265ca4759b0afa`。Producer 原生失败字段仅为 `input_lock_rendered_prompt_sha_mismatch`。
- Read-only post-block diagnostic（不得伪装为 generic blocked receipt 原生字段）：总 population=`326` groups；前 `160` 个 exact match；第 `161` 个为 ARC group=`2ac15877caa468bf7ee3f2c16bcb9fd7b6e122bc2081c7eaaf2ecd0f64af42e4`、sample=`55c885c900afb3b5f7a4541c68797000852c31971527148f5c29b6ccf783b4d8`、source row=`836`、eval qid=`32`。
- Historical/actual rendered SHA256 分别为 `2b933c569545f2e26944f1702c1e878c20cb9554cd7d679cd48395b9da6e2828` / `ad7828e4c67fad1515e4bb768624114be20091fd6bbb86960a3931d439a04a9d`；alignment SHA256 分别为 `1440a0c16db39ecb915318fd836dbad959c1ca5fb275b145343e8011cd79657f` / `206b4ddc9b24874ea7b2d892e7c770901d18ccb5eb30e7cf3ac5d22fea5f03ed`。
- Root cause：旧 E0 dev-anchor 从 label-free support projection 的 `choices[:4]` 渲染 A-D；materialized ARC row 保留额外 E 选项，E0/A4 production `UnifiedEvaluator` 遍历完整 choices。Membership hash 只看归一化后的前四项，因而先通过；随后 exact rendered/alignment anchor fail-closed。这不是 streaming、operator、mechanism 或 accuracy 结果。
- 0 usable sidecar/manifest/templates/raw/runtime/plan；0 model/checkpoint load、forward、GPU、Kubernetes、training。E1-pilot 继续 `SEALED / NOT RENDERED / NOT TOKENIZED / NOT ALIGNED / NOT RUN / NOT READ`。
- 机器可读闭环：`recipe/eval_recipe/fpct_e1/executions/07755a40/input_lock_failure_receipt.json`。当前 root 永久 no-resume/no-reuse。
- `HUMAN REVIEW REQUIRED`，尚未批准也不自动推荐两项之一：`STRICT_HISTORICAL_PROJECTED_FIRST4_ANCHOR` 或 `ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT`。任一选择都必须先形成新的前瞻 amendment，再使用新 commit/snapshot/run UID/root；不得原地放宽、重跑或复用本 execution。

## 2026-07-28 A5 actual E0 production-runtime prompt 前瞻修订

> 上一节最后一项是 A4 failure closure 当时的历史状态。用户随后在任何 A5
> natural census、model/checkpoint load、E1-2 或 E1-pilot access 之前作出独立
> 人工批准；不是回写或删除旧失败记录。

### 人工决定与双锚点

- Approval ID：`APPROVED_PROSPECTIVE_AMENDMENT_E1_A5_RUNTIME_PROMPT`。
- Selected：`ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT`。
- Rejected as primary：`STRICT_HISTORICAL_PROJECTED_FIRST4_ANCHOR`。
- `historical_projection_anchor` 继续冻结 question + first four choices，只承担
  support selection、content-group/sample identity 与 historical provenance。
- `production_runtime_anchor` 使用 exact materialized E0 row、完整 production
  choice list、exact historical E0 evaluator/prompt builder、tokenizer 和 chat
  template；这是未来 E1-2/E1-3 的唯一 operative prompt/alignment anchor。
- 事件分类为 `HISTORICAL_PROMPT_ANCHOR_UNDERSPECIFICATION`。不修改或重算 E0
  aggregate，不推断 first-four counterfactual accuracy，也不改变 operator、
  checkpoint、lambda grid、estimand、weight 或 group membership。

Normative sources：

- `FPCT_E1_A5_PRODUCTION_PROMPT_AMENDMENT.md`；
- `recipe/eval_recipe/fpct_e1/e1_a5_prompt_contract.json`；
- `recipe/eval_recipe/fpct_e1/e1_a5_prompt_schema.json`。

### A5 当前阶段表

| 阶段 | 当前状态 | 依赖 | 当前授权 | 边界 |
|---|---|---|---|---|
| A5 v7 protocol/contract/schema | `GO — HUMAN DECISION LOCKED` | A4 failure closure + prospective approval | 文档/schema/manifest | 输入 provenance 修订，不是 mechanism/performance evidence |
| Dual-anchor implementation | `GO / PRE-NATURAL` | v7 contract | code + CPU synthetic tests | 不得读取 natural census、model/checkpoint 或 E1-pilot |
| A5 prompt-specific synthetic gate | `GO` | implementation complete | CPU synthetic only | path=`e1_a5_prompt_synthetic_gate.json`；SHA256=`464e646c...`；旧 A4 gate 仅作 07755a40 historical evidence |
| Clean pushed A5 commit/snapshot/root | `GO / EXECUTION NOW ABANDONED` | 新 gate GO | research branch commit/push + immutable snapshot | commit=`9b248d20...`；fresh UID/root 已建立，未复用 A4 artifacts |
| 326-group CPU census/input lock | `A5_INPUT_LOCK_BLOCKED / NO USABLE CENSUS` | gate GO + clean push + fresh root | 仅执行了 pre-group-1 provenance attestation | 在第 1 个 group lookup/render/tokenize/align 前因 data-tree hash-domain mismatch fail-closed |
| A5R1 hash-domain recovery | `APPROVED / PRE-NATURAL GATE GO` | A5 failure closure + 2026-07-30 prospective approval | CPU-only protocol/code/tests/gate | gate SHA256=`a2e53784...`；在 gate 冻结时尚未 commit/snapshot/natural census |
| A5R1 clean commit/snapshot/root | `GO / EXECUTION NOW BLOCKED` | v8 gate GO | research commit/push + immutable snapshot | commit=`37be816a...`；649-entry snapshot 与 fresh UID/root 已验证 |
| A5R1 326-group CPU input lock | `A5_INPUT_LOCK_BLOCKED / NO USABLE CENSUS` | clean local/upstream + new snapshot/root | CPU tokenizer/alignment only | choice-cardinality hard check 失败；0 persisted census rows；禁止 resume/reuse |
| Runtime/checkpoint/plan/E1-2 | `NOT AUTHORIZED` | 另行授权，即使 input-lock GO 也不自动进入 | 无 | 禁止 model/checkpoint load 和 forward |
| E1-3 | `NOT AUTHORIZED` | finalized E1-2 + 另行授权 | 无 | 未进入 |
| E1-pilot | `SEALED / NOT RUN / NOT READ` | operator freeze 后独立授权 | 无 | 不得 render/tokenize/align/run/read |
| Confirmatory | `SEALED / NOT AUTHORIZED` | 后续完整阶段链 | 无 | model-selection/test/formal seeds 不得访问 |

### A5 hard gate 与 execution firewall

- Census population 固定为 326 distinct E0-design groups：ARC 128、OpenBookQA
  70、MMLU-Redux 128。
- `prompt_relation` 只允许
  `EXACT_HISTORICAL_AND_PRODUCTION_MATCH` 或 `EXTRA_CHOICES_ONLY`；question、
  first-four text/order、instruction/template/whitespace、gold A-D 或唯一 row
  mapping 任一异常均 fail-closed。
- 每一 group 必须满足 historical first four 等于 production first four，且
  gold answer 属于 A-D；完整记录 dual hashes、choice counts、raw row hash、
  production token/alignment/topology/logical-row geometry。
- 新 pre-natural gate 必须使用
  `recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json` 和 A5 versioned
  instrumentation artifacts；不得用改变后的 tree 去通过旧 A4
  `verify_tracked_gate`，也不得覆盖旧 gate。
- A5 gate 已原子发布并独立自验：`248 passed / 0 failed`，artifact/evidence
  SHA256=`464e646c336d33c97a03c603d283a95b1579508d63ae0f19d7c700ccf2dcfc02`
  / `2fc5c70da231cfd0aba54656c5a34ff142594208f4a006348f73607559af4a47`。
  Million-row stress=`1,000,384` logical/emitted rows、245 chunks、semantic replay
  exact；peak RSS=`676,200,448 B`，低于冻结 threshold=`812,384,256 B`。
- A5 instrumentation parity/synthetic/hard-gate SHA256 分别为
  `220c8da80e2b5409a82cff386048e3afb8814efb23a6cf4bf84216d4e7066333`、
  `6f294b7c68eeee3e8f488e19c55f081dfcd23b04ae241f5367222a1bb0f10f16`、
  `29a3018d900231be3eca10b6e3762872bf588458390a1d595fe4f2012861ac58`。
- `07755a40` 继续永久 abandoned：resume=false、artifact reuse=false、
  scientific result=false。该 A5 前置条件随后由 `9b248d20` 的 clean commit、
  snapshot、UID/root 满足，但该 execution 已在 group 1 前 fail-closed；任何新的
  group-1 重启都必须先获得独立 A5R1 人工批准。
- 本次 A5 人工批准不授权 model/projector/checkpoint load、model forward、GPU、
  CUDA、Kubernetes、training、E1-2、E1-3、E1-pilot 或 confirmatory access。

### A5 clean execution 与 pre-group-1 failure closure

- Clean pushed execution/source SHA=`9b248d2094b684f5d9e9a218919a354b7d97468e`；
  UID=`fpct-e1-a5-runtime-prompt-9b248d20-v1`；root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5-9b248d20-v1`。该 root 永久
  abandoned，resume/reuse=`false`。
- Immutable source snapshot：641 entries；Git tree=
  `ec8254302e831d83a169971964e848550e659f75`；mounted-tree SHA256=
  `ee824b0d51d015a615f0db836ab1d0e2b555e36bcc55a5a284245dcc98f1cb65`；
  source receipt file/internal SHA256=
  `dc4396d50594f0db120fcba9252e85c92e343a014230379ebe8c51e645082981` /
  `9d051cca1290e4d90560587e5d95ed8f81c8e4a156090baf323614a0a07ec95e`。
- Sealed bootstrap 与 snapshot provenance 通过；本地 Qwen3/TinyLlama
  tokenizer 路径完成解析/文件枚举且 tokenizer objects 被加载，但后续 strict
  tokenizer-bundle/template attestation 尚未完成。Producer 在任何 E0-design
  group 被 lookup、render、tokenize 或 align 前，因
  `a5_materialized_e0_development_tree_sha_changed` 终止。
- 原生 terminal receipt=`A5_INPUT_LOCK_BLOCKED`；blocked receipt / execution
  identity SHA256=
  `fe305b2c9d881b202f4a096646e9530e8d630a6fcfa095754d3101450ad7ab79` /
  `2247ccf274ea8c55b25b55fd5ac3fb2a191d6796f5ea06a9b18b4a9d8e6b7a0b`。
  `natural_e0_design_group_count=0`；未生成 census row、alignment row、sidecar、
  geometry、runtime、plan 或 scientific result。
- Read-only root-cause verification 对同一 E0 development tree 的 51 files /
  233,569 bytes 重算：冻结 E0 算法
  `relative_path_nul_file_sha256_bytes_v1` 得
  `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`；
  A5 rich generic 算法 `canonical_json_file_manifest_v1` 得
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`。
  前者与 frozen E0 manifest/runtime provenance exact match，因此分类为
  `HASH_ALGORITHM_DOMAIN_MISMATCH_DATA_BYTES_UNCHANGED`，不是 data drift/corruption。
- Machine-readable failure closure：
  `recipe/eval_recipe/fpct_e1/executions/9b248d20/input_lock_failure_receipt.json`。
  A5R1 proposal=`FPCT_E1_A5_DATA_HASH_RECOVERY_ADDENDUM.md`；在本 failure
  closure 冻结时它仍仅为 draft。用户随后于 2026-07-30 前瞻批准，operative
  successor 见下一节；本段不被追溯改写为当时已经授权。
- 未加载 model/checkpoint，未运行 model forward、GPU、CUDA、Kubernetes 或
  training；E1-2/E1-3 未进入；E1-pilot 与 confirmatory 继续 sealed。

### 2026-07-30 A5R1 human decision 与 v8 pre-natural gate

- 用户回复 `可以` 已前瞻记录为
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`。授权仅覆盖双
  hash-domain 修复、CPU/offline/no-model gate、全新 commit/snapshot/UID/root
  与一次从 group 1 开始的 CPU input lock；不授权 E1-2/E1-3 或模型输出。
- Operative protocol=`fpct_e1_mechanism_audit_v8_a5r1_hash_domains`；新增
  amendment/contract/schema SHA256=
  `8eafb29d3d736740730e10505ebd8217e779e5ef4d3a128cfb0db3a10a018068` /
  `643151b67d98c84c1120b52705b0fe837fb4106664f10200ada1a692c744a0b1` /
  `7bd2478f7ef3cfd32e752056cf161b8575b84a1f65088c84a0d2c37aec43704b`。
- Declared domain 固定
  `relative_path_nul_file_sha256_bytes_v1` / `f3dcf2c7...`，只与 frozen E0
  identity 比较；generic domain 固定 `canonical_json_file_manifest_v1` /
  `12f537ca...`，只用于同域 predecessor、before/after 和 tamper。Manifest、
  sidecar、GO/blocked receipt 与 completed verifier 都显式绑定两域。
- v8 synthetic gate=`GO_PRE_NATURAL_A5R1_HASH_DOMAIN_HARD_GATE`；artifact /
  evidence / tracked-tree SHA256=
  `a2e53784fa46d2f63963bdb41e327a4e17936cce2bff765802b34f3d87869a9f` /
  `14225f02046f0fe626b3fab1cd8d07d5e45e873ac54f3d2e0ba78d78a6fdd003` /
  `8449418e9518847ef7b6c0a7e9bd638d83d3bb098f5253a8564d4f4cd7e1e77d`。
- Tests=`291 passed / 0 failed`，output SHA256=
  `652e62bfb95a6acc1cfe156aa67b7c8010856105f85ef329f5dfa8847fe96250`。
  Stress=`1,000,384` logical/emitted rows、245 chunks、semantic replay exact；
  peak RSS=`676,876,288 B` < frozen threshold=`813,060,096 B`。独立 gate verify
  与 v7 四件套 SHA 检查通过。
- 一次 direct-file 调用在 import 前失败、一次完整计算在最终发布时被 sandbox
  read-only 拒绝；两者均未生成 gate/partial artifact。随后生成的 precommit
  gate `681532e9...` 在 schema/terminal-atomicity 复核中被主动隔离为 local-only
  invalid evidence，未 commit、未作为 operative gate、未访问 natural data。
  修复并冻结回归后从头运行，才产生上述唯一 operative gate。
- 在 gate 冻结时尚未创建 successor commit/snapshot/UID/root，也未访问任何
  E0-design natural group；随后获授权执行的 `37be816a...` 结果见下一节。
  `9b248d20` 继续永久 no-resume/no-reuse；E1-2/E1-3、model/checkpoint/
  forward、GPU/K8s/training、E1-pilot/confirmatory 均未授权。

### 2026-07-30 A5R1 CPU input-lock terminal failure

- Clean commit/push=`37be816ad611b8b0d916bd98c840c5f31efe2b50`；UID=
  `fpct-e1-a5r1-hash-domains-37be816a-v1`；root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-37be816a-v1`。649-entry immutable
  snapshot 的 Git tree=`7aea57815c6abb0005739f77d9ea753f22e26d85`，mounted
  tree SHA256=`1c2f3a1db2036104d13e1f1f07225d5eeaf8e8c4276c827c1b59340354875508`；
  source receipt file/internal SHA256=
  `9e0ac64a3653b58a7c518650f5795ebd10ecc24ffe702def754e663ba480339b` /
  `1bbb0c4e38b36138b8d7c3592f8bbe10d91c1b174860610f947feec18086b6d4`。
- Sealed bootstrap、snapshot 与两个 hash domains 均通过；CPU/offline producer
  加载本地 tokenizer 与 ARC dataset 后，在冻结的 choice-cardinality contract
  触发 `a5_materialized_row_has_fewer_than_four_choices`。Terminal receipt=
  `A5_INPUT_LOCK_BLOCKED`；blocked/identity file SHA256=
  `46143877891c15fab1b5ebd3d359b80f3d9aa7464ceb961a0e8f353b0873bee2` /
  `c0a1e1b1b0de7700e4a7d7ce79c3317af30c183b1f321187b2cd505c592b27e5`。
- Producer 没有在失败前持久化 group ordinal，因此不事后扫描自然 population
  猜测失败行；仅对已授权的 frozen ordinal-1 ARC row 做单行 replay，确认其为
  canonical 四选项，但不能定位后续失败。失败 group/已在内存处理的 group 数记为
  unknown。Persisted census rows=`0`，未生成 sidecar、manifest、geometry 或
  streaming artifact，不能作为 mechanism/scientific evidence。
- Machine-readable closure：
  `recipe/eval_recipe/fpct_e1/executions/37be816a/input_lock_failure_receipt.json`；
  SHA256=`ec3ae949b557da957a1a1f295f41b91b8522420445b5479a945b1f59b5de8e8a`。
  Root cause 当前仅冻结为
  `MATERIALIZED_CHOICE_CARDINALITY_CONTRACT_FAILURE_UNLOCATED`；任何 parser/
  population 修订与新执行都需要新的前瞻人工批准。
- 当前 execution/root 永久 resume/reuse=`false`。未加载 model/checkpoint，未
  运行 model forward、GPU、CUDA、Kubernetes 或 training；E1-2/E1-3 未进入，
  E1-pilot 与 confirmatory 继续 sealed。

## 2026-07-31 A5R2 choice-cardinality 前瞻锁

### 人工批准与修订边界

- 用户已在新的自然 population audit、tokenization/alignment、模型输出或
  E1-pilot outcome 之前批准
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R2_CHOICE_CARDINALITY_RECOVERY`。
- A5R2 只修复 materialized choice cardinality 的输入/provenance 合同：ARC
  接受实际 `n>=2`，其中 2/3-choice row 是真实 low-cardinality；OpenBookQA 与
  MMLU-Redux 仍要求 exact four choices。Production runtime 使用完整、未 padding、
  未 truncation 的 choice list；historical projection 继续使用 `min(4,n)`。
- 失败 execution=`37be816ad611b8b0d916bd98c840c5f31efe2b50` 及其 root
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-37be816a-v1` 永久 no-resume/no-reuse；
  不读取或复用其 partial state。
- Label firewall 保持不变：audit 只允许 question、choices 与冻结的定位/provenance
  metadata；不得读取 correctness、accuracy、beneficial/harmful、model prediction
  或任何 selector outcome。

Normative sources 与 SHA256：

- `FPCT_E1_A5R2_CHOICE_CARDINALITY_AMENDMENT.md` =
  `89753bcbdec66d07c36bcfc3a5c636e66704cddce5546ac0e321d7a9ea053384`；
- `recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_contract.json` =
  `a4bdf4a229d26b367fb8ea7c90adf39d72e94c56daf4d095346bf673c5c2eb1b`；
- `recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json` =
  `9cb387628e4b8fcf6c978e082b8c3380dbc62406c6aa460892c679872d656d7f`。

### Pre-natural synthetic gate 与当前授权

| 阶段 | 状态 | 证据/依赖 | 授权边界 |
|---|---|---|---|
| A5R2 protocol/contract/schema | `GO — HUMAN DECISION LOCKED` | 上述三项 SHA256 | 仅输入/provenance 合同修订 |
| A5R2 pre-natural synthetic gate | `GO` | `325 passed / 0 failed`；tracked tree=`60946d56e4c11ebcff8ac44b94b8bcdedbd6a6c5aea71ed168f854d6dce996ca`；evidence=`8788694b7715676a84e453f80b3bb596c5688a7660fa916964ab15ddaf9213c4`；artifact=`cfc8c7da3bb074b0a4fcb23276af8d8404516b656337abb5c3034724272d2cd9` | CPU/offline/no-natural synthetic evidence only |
| A5R2 clean commit/snapshot/UID/root | `GO / CONSUMED BY TERMINAL BLOCK` | commit=`e765d493...`；658-entry immutable snapshot | `e765d493` root 禁止 resume/reuse |
| Label-free 326-row choice audit | `BLOCKED AT ATOMIC PUBLICATION` | 326 rows read/reduced in memory；`renameat2(RENAME_NOREPLACE)`=`EINVAL` | 无 final ledger/summary/lock；不得推断自然 decision |
| CPU input-lock | `BLOCKED / NOT RUN` | 没有可用 audit GO lock | 不得 tokenizer/alignment 或进入 E1-2 |
| E1-2 / E1-3 | `NOT AUTHORIZED` | 独立后续授权 | 禁止 model/checkpoint load 或 forward |
| E1-pilot / confirmatory | `SEALED / NOT RUN / NOT READ` | operator freeze 后独立授权 | exploratory pilot 仍无 confirmatory eligibility |

Gate 只证明 variable-cardinality parser、allowlisted projection、ordinal/summary/
lock、dual data-tree/source-snapshot binding、atomic publish 与 fail-closed taxonomy
在 synthetic fixtures 上满足合同；它不是自然 choice distribution、input-lock、
mechanism activation 或 task improvement 的证据。在 gate artifact 冻结时，新的
326-row 自然 audit 尚未运行；其后获授权的单次执行结果见下一节。未加载
model/checkpoint，未运行 model forward、GPU、CUDA、Kubernetes 或 training，且未
进入 E1-2/E1-3。

## 2026-07-31 A5R2 CPU choice-audit terminal failure

- Clean commit/push=`e765d493733d9eec94c152506a1c57781e26fb41`；UID=
  `fpct-e1-a5r2-choice-cardinality-e765d493-v1`；fresh root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-e765d493-v1`。Snapshot 共 658 entries，
  Git tree=`5ed0ec9da90838d4b05fc7ff22ee6b22d8140f37`，mounted-tree SHA256=
  `ebaa80303447e931dd0695ba49d76d21d1f630f87e94577828b53d15239e2c5f`；
  source receipt file/internal SHA256=
  `e6a2930130cdc8e966596195015a12a4020455a9eb3a9f4f75fa59223748d1f5` /
  `4c681ab9ec5032841b4055c2bdfdaf9d05e02f39fb09f54ea60bd6320a55657a`。
- 首次 launcher 因 Conda `python` symlink 在 target import 前被 sealed bootstrap
  拒绝；未创建任何自然 artifact。改用解释器 realpath `python3.10` 后，正式
  CPU/offline audit 从 group 1 读取并在内存中完成 326-row label-free reduction。
- 在 final `choice_audit` 目录发布时，当前 `/netdisk` 文件系统对
  `renameat2(RENAME_NOREPLACE)` 返回 `EINVAL`。Staging 按 fail-closed finally
  清理；没有 final ledger、summary 或 audit lock，因而不得从执行位置推断
  natural mechanical GO/BLOCK。Tokenizer/chat-template/alignment 以及后续 sidecar/
  input manifest 均未执行或持久化。
- Native terminal receipt=`A5R2_INPUT_LOCK_BLOCKED`，stage/code=
  `PRECOMPUTATION / ERRNO_22_INVALID_ARGUMENT`，blocked/identity SHA256=
  `498a3571a8933d34c3bdf7e4f2641b8da5f863bff7a5f87124488794dcc71fd2` /
  `a2c8786c290be420dad6a89a7615fb1db49c7092c0ce600c5d2add8f0534df8f`；
  strict v9 schema validation=`GO`。
- Machine-readable closure=
  `recipe/eval_recipe/fpct_e1/executions/e765d493/input_lock_failure_receipt.json`，
  SHA256=`f6715f9d86b23ccc265f6ead7b24af47acaf1e1ecd9eb14552af7940d68f798c`。
  本轮未做 post-failure natural scan，未修改 code/threshold；execution/root 永久
  resume/reuse=`false`。任何 publication 修复必须先有新的 prospective amendment、
  commit、snapshot、UID/root；不得原地重跑。
- 0 model/checkpoint load、0 model forward、0 GPU/CUDA/Kubernetes、0 training；
  E1-2/E1-3 未进入，E1-pilot/confirmatory 继续 sealed。

## 2026-07-31 A5R3 portable publication pre-natural lock

- 用户在 A5R2 terminal closure 已提交、推送并报告后，以原文 `可以` 做出
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R3_PORTABLE_PUBLICATION`。批准发生在
  successor natural row、tokenizer/alignment、模型输出和 E1-pilot outcome 前。
  唯一获准变化是 choice-audit directory publication primitive；326-group
  population/order、choice parser/cardinality、taxonomy/actions、threshold、prompt、
  tokenizer/alignment、operator 和 estimand 保持 A5R2 v9 不变。
- A5R3 是 schema-v10 operational overlay，`choice_semantics_version=9`。发布顺序
  冻结为：natural row 1 前 O_EXCL/no-follow 持久 claim；完整 reduction 后创建固定
  staging；ordinary same-filesystem rename 一次；永久保留 claim；以 O_EXCL+fsync
  `choice_audit_publication_receipt.json` 作为 durable commit point。只有 exact
  claim + final 三 artifacts + receipt 的交叉绑定可供 downstream 只读消费；任何
  crash/tamper/intermediate state 永久 BLOCKED 且不得 resume/reuse。
- Normative amendment/contract/schema SHA256=
  `ccfb73bdfd396d5941e975b09143450189e75fffd4884482ce613cf03feb6cd4` /
  `2ba72cfbccb5a1b7c9f46c02510f4763a2917210d3884c7b5d8223580db9e272` /
  `b24d197481c01c917ad9d4063441f96ac23979bdb4b3db24855e551c7fb91460`。
  A5R2 四个 normative objects 与 `e765d493` closure 均按 frozen SHA 复验且未修改；
  `e765d493` root/in-memory reduction/artifacts 永久不可复用。
- CPU/offline synthetic hard gate=`GO_PRE_NATURAL_A5R3_PORTABLE_PUBLICATION_HARD_GATE`：
  `42 passed`；target `/netdisk` scratch ordinary-rename/same-device/fsync probe=`GO`，
  probe evidence=`28b8adda2e833fe1f298425f79ba70b9f059a929770ae1a62c34b996cbc697fc`；
  tracked tree=`659151ff0f3d2d45b6368a36783d4525f082415211fe50dc5e85de82b8c60a37`；
  gate evidence=`b4c090fab3404020fd649262d198ac4f45237b47f8c2823b43ae64b5609e6399`；
  gate artifact SHA256=`92c63e3b922405b8c0dcb42a642eeef7b528f1115ea7fc3ba4ab325c61c87bac`。
  Inherited CPU closure 共 353 项：首轮 `348 passed / 5 failed` 中四项为 synthetic
  fixture 尚未指向新 active gate、一项为 contract-required `/tmp/tmp*` domain 被
  非规范 TMPDIR 改写；修正 fixture 后四项通过，并在规范 `/tmp` domain 下单独验证
  sealed-import 项通过。没有放宽 invariant 或 tolerance。
- 本记录冻结时新的 natural choice audit、CPU tokenizer/alignment input-lock 均未
  运行；clean execution commit/snapshot/UID/root 尚未生成。`math.md` SHA256 仍为
  `98d1b61f84d046548d5ba0070d6858c7080cb14fdef9169b08ad167461b809ad`。
  0 model/checkpoint load、0 forward、0 GPU/CUDA/Kubernetes、0 training；E1-2/E1-3
  未授权，E1-pilot/confirmatory 继续 sealed。

## 2026-07-31 A5R3 b6109443 pre-natural verifier closure

- 首个 clean/pushed A5R3 execution SHA=`b6109443e1b4c35eef73c322f30e9aec37194ce7`，
  UID=`fpct-e1-a5r2-choice-cardinality-b6109443-v1`，fresh root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-b6109443-v1`。665-entry snapshot 的 Git
  tree=`edebc2a25e045baa52e0d2d559b8837eb6a86d08`，mounted-tree SHA256=
  `28c63507264962c4fe5424d6429763b067ec23c8678a21767da7f4d360946f29`。
- Sealed bootstrap 在 natural row 1、publication claim、tokenizer/alignment 前，因
  A5R3 loader 调用了 A5R2 gate 的 live tracked-tree verifier 而 fail-closed。A5R2 gate
  是 byte-immutable predecessor evidence；其历史 live tree 必然不同于 A5R3 prepare，
  因此只能按 frozen SHA+strict schema 消费，不能再次要求 live A5R3 tree 等于 A5R2。
- Terminal receipt status/stage/code=
  `A5R2_INPUT_LOCK_BLOCKED / PRECOMPUTATION / A5R2_TRACKED_SOURCE_TEST_MAP_IS_STALE`；
  blocked/identity SHA256=
  `85f078749a1b36fca8c4101e62985e303c7e29a56b9bc7f53c1bcb0d10c369fb` /
  `bce6e712ce2fc23bda9a78a300ee33794b72135801ec42b0c68ef4749a16c78a`。
  Git closure=`recipe/eval_recipe/fpct_e1/executions/b6109443/input_lock_failure_receipt.json`，
  SHA256=`efbecc09a9b5bb7933beb0c9bf87dc2dedf4eb17314eec3aa263f963de56ee56`。
  Root/snapshot/identity/receipt 永久 no-resume/no-reuse。
- 修复白名单只有 immutable predecessor gate consumption：保留 A5R2 gate file SHA 和
  strict v9 schema/status 检查，移除其对 successor live source map 的错误要求。没有
  修改 population/parser/taxonomy/threshold/publication algorithm/operator。Replacement
  A5R3 synthetic gate=`43 passed`；tracked tree=`54f108bb07f3d10aa759e4eb93ee011915e5b93e91b864ae32da6923a06b1369`；
  evidence=`907f3b6a631ed8a1c6bb13cf559097fc96fc9b5a2b2148ebf0c8eabcaeca63c9`；
  artifact SHA256=`004a8feb272af78c439f970b1ae637400e641a4fd93a412f29832f288871a4c2`。
- 自然 rows read=`0`，claim/staging/final/publication receipt=`0`，tokenizer/alignment=
  `0`，model/checkpoint/forward/GPU/CUDA/K8s/training=`0`。下一执行必须新 commit、
  snapshot、UID/root 并从 group 1 重启；E1-2/E1-3 仍未授权。

## 2026-07-31 A5R3 3263531e post-reduction publication-mode closure

- Clean/pushed execution SHA=`3263531ed7137efd241de3c951bf7c5e9d37e669`；fresh
  root=`/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-3263531e-v1`。666-entry snapshot
  Git tree=`8293de1da48eb7e9109d4e858923f5ebfd4f99c4`，mounted tree=
  `5c43edbf7ff916929d75876ae156390f722eded86b07741eb64d29fb65e51e74`，
  receipt=`ed9df4ddfacbcc9c3d1ccfe37d19d07e7b8862ac434d9d050ee7107fdaef06d0`。
- Sealed CPU execution 按 v10 在 natural row 1 前创建并持久化 claim。完整 326-row
  label-free reduction 留在内存后，fixed staging 由目标文件系统继承 setgid，实际
  mode=`02700`；实现冻结为 exact `00700`，因此在任何 final/receipt 生成前 fail-closed，
  code=`A5R3_FIXED_STAGING_IDENTITY_OR_MODE_IS_UNSAFE`。
- Claim SHA256=`91834557...`；blocked/identity SHA256=`c7065a61...` /
  `4243e8ac...`。Final choice audit、publication receipt、tokenizer/alignment input lock
  均未生成；程序执行了 label-free reduction，但 human/agent/reviewer 未打开、汇总或
  将 unpublished natural choice statistics 用于 recovery。
- Git closure=`recipe/eval_recipe/fpct_e1/executions/3263531e/input_lock_failure_receipt.json`，
  SHA256=`8ca3537914edb880b3436012841ec47b8d416345e2646fce22597c60d3bf4d17`。
  Claim/staging/root 永久保留作 forensic tombstone；禁止 cleanup/resume/reuse。
- 当前状态=`HUMAN REVIEW REQUIRED`。候选 A5R4 只能前瞻性改变 mode safety predicate：
  owner 必须匹配、group/other permission bits 必须全为 0，允许目标文件系统继承的
  setgid bit；并必须新增真实 target-FS inheritance regression、新 gate、新 clean
  pushed SHA/snapshot/UID/root，从 group 1 重启。尚未获批，未改实现。
- 0 model/checkpoint load、0 forward、0 GPU/CUDA/K8s/training；E1-2/E1-3 未进入，
  E1-pilot/confirmatory 继续 sealed。

## 2026-07-31 A5R4 inherited-setgid mode 前瞻锁

- 用户在 `3263531e` terminal closure 已提交、推送且报告后，以原文 `可以` 前瞻批准
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R4_SETGID_MODE_PREDICATE`。本次唯一变化是
  owner-created staging/final directory mode predicate；choice semantics/payload 仍为
  v9，A5R3 claim/receipt/rename/fsync/crash/no-resume envelope 仍为 v10。
- Predicate 只接受 `00700` 或合法继承的 `02700`：real directory、owner=euid、
  owner bits=`0700`、group/other bits=`0`、无 setuid/sticky、same device；`02700`
  额外要求 parent 预先 setgid、parent owner=euid、child GID=parent GID。Staging 创建后
  冻结 `(dev,ino,uid,gid,mode)`，写入前、rename 前后及 completed verifier 均重验。
- Amendment/contract/schema SHA256=`803884c0...` / `af4cf2af...` / `cb90cec6...`。
  CPU/offline gate=`GO_PRE_NATURAL_A5R4_SETGID_MODE_HARD_GATE`，`78 passed`；tracked
  tree=`753f8af50f409e199a943cb6b73b7f9f5553b5d2ab5344c77725384158a6e388`，
  evidence=`7e3ee42b93325edc8a65e2cc7f25e013fe952ddd6bb3d4cda3622a8b7ad981c9`，
  artifact SHA256=`0f3477575b8a75fb4b6028c0eeda19aa56885aca2fdb0addce0171a5e02c7ea8`。
- 真实 `/netdisk/lijunsi/fpct-e1` synthetic scratch 观察 parent/staging/final mode
  均=`02700`、UID/GID/device 一致、rename inode 保持、bytes/fsync 通过；probe evidence=
  `ad7bfe38965840f93e5f27fb6706915a80380250edc4e269859dc509dc5fe69c`，scratch 已清理。
- 补充 inherited suites：A=`155 passed / 2 failed`，B=`254 passed / 2 failed`。四项
  failure 均在 immutable predecessor snapshot 原样复现：两项 A5R1 oracle 仍调用已移除
  的 `e0_data_hash_domains` 参数，两项 A5 prompt 临时 repo fixture 未复制冻结 population
  sources；不是 A5R4 regression，未修改这些历史 fixture，也未把它们计入 v11 GO。
- 最终 sealed-import/prepare/source-snapshot targeted suite=`94 passed`；formal prepare
  closure 同时绑定 immutable A5R3 verifier 与 active A5R4 verifier 的模块 origin/SHA。
- 本锁未读取任何 successor natural row，未运行 tokenizer/alignment、model/checkpoint/
  forward、GPU/CUDA/K8s/training。Gate GO 只授权下一 clean pushed SHA 用 fresh snapshot/
  UID/root 从 group 1 运行 label-free audit，并在机械 GO 时运行 CPU input lock；E1-2/
  E1-3/E1-pilot/confirmatory 仍未授权。

## 2026-08-14 A5R5 fresh-root interruption recovery 前瞻锁

- `a47d52f8...` A5R4 execution 的 choice audit 已 durable GO，但 downstream input
  lock 在首个 sample 的 21 个 row-template chunks 后被外部中断；终止原因本身未验证。
  它没有 canonical sidecar/main manifest/A5R2 GO/A5R2 BLOCKED，不能补造 producer
  failure receipt，也不能解释为 mechanism/performance result。
- Hash-only forensic closure：708 files、39,278,548 bytes、tree SHA256=
  `302762365c964e93a3f3fe3ad063447998b51207f6f9ab57b0594c4132184187`；observation
  SHA256=`fa2a28107d6dfd84b99152ca99dc312cf509059b6376851ad6323c871fe6f10e`。
  旧 root/snapshot/partial artifacts 永久 no-cleanup/no-resume/no-repair/no-reuse。
- 用户以 `PLEASE IMPLEMENT THIS PLAN` 批准 A5R5 fresh-root recovery、后续条件式
  E1-2/E1-3、单因素修复及单 seed→条件三 seed；仍不授权 confirmatory、native null、
  selector、新 gate、跨模型或 36-run。
- A5R5 controller 使用 hidden sibling control root、run-root 前 O_EXCL materialization
  claim、worker 前 O_EXCL launch/worker-start claims、CPU/offline allowlist environment 与
  `start_new_session=True`。Worker 必须等到 launch receipt 并以 PID/starttime/cmdline SHA
  证明自己是 Popen child；任何中断或 PID reuse 均 terminal，不自动 retry/resume。
  Scientific root 不含 log/PID/controller state；GO 还需 snapshot 内完整 producer verifier
  独立重放。
- Pre-natural gate=`GO_PRE_NATURAL_A5R5_FRESH_ROOT_RECOVERY`；`147 passed`，test
  output SHA256=`a4c9e7fb841d924c29b109a7cdc19742a0017f33f26d6a8ec07438e810eac500`，
  evidence=`f2da57e24b13e1ac2f712b764c5cecf6c6b44e27c0a2eda37c1960ce6fa395e9`，
  gate artifact SHA256=`6190db2d26ad55c8a467c880ff0050c1e3c7bd0cff2bec83d8a320bc9afc21f9`。
- 本锁冻结时 successor natural/tokenizer/alignment/model/checkpoint/forward/GPU/CUDA/
  K8s/training/E1-pilot/confirmatory 均为 0。只有 clean pushed SHA 与全新 root 可执行。

## 2026-08-15 A5R6 schema-binding recovery 前瞻锁

- A5R5 execution `ab4052cd...` 完成 326/326 groups、30,370,816 logical rows 和
  7,564 chunks；geometry/streaming GO、missing/duplicate=`0/0`。最终 v9 manifest
  validation 因 `choice_audit.publication` 不在 strict five-field schema 中而 fail-closed；
  worker exit=`1`，canonical BLOCKED receipt 已发布，无 manifest/GO。该 root 永久
  no-resume/no-repair/no-reuse/no-cleanup，且不是科学结果。
- A5R6 不修改 immutable v9 schema（SHA=`9cb38762...`）或 A5R3 publication schema
  （SHA=`b24d1974...`）。`_choice_audit_binding()` 固定五字段；publication claim/receipt
  仍由 A5R3 verifier 独立严格验证并保留在 verified audit 中。
- Full integration regression 真实生成 326-row audit+publication；五字段完整 v9
  manifest 通过，重新注入 publication 必须复现 `oneOf=0`。
- Failed-root portable inventory 使用 `kind/path/mode/size/content`，排除 numeric UID/GID：
  9,343 entries、SHA=`67c572aa...`；ownership/group/mode 改为当前 namespace 内安全检查。
- Operative aggregate suite=`318 passed, 4 deselected`。四项 deselected 是已记录的
  historical fixture/API drift；另三项 A5R5 gate 在后继 prepare SHA 上按设计 fail-closed，
  不计入 A5R6 operative suite，也未修改历史 gate。
- A5R6 gate=`GO_PRE_NATURAL_A5R6_SCHEMA_BINDING_RECOVERY`；test output SHA=
  `92e4a68c0480d41f1c92b012c8f5b3a3b8ae59f0d8bedf7428abb24253333a32`；evidence=
  `c6402176b1d6b8ec0e2a95569a8356d2c6081c0d0be1018a48c6689490be35f6`；gate artifact=
  `9f4787a63edfe4c0d80079fb7ff9165fd5dc24ee878e2753f20fb28caea56d4a`。
- Gate 的 exact tracked/immutable universe、required test nodes、四个历史 deselection、
  A5R5 worker result/log bytes 与 portable failed-root closure 已冻结；controller 在创建
  successor state root 前强制验证并将 gate 绑定到 claim/lock。Deep GO 必须额外发布
  可幂等严格复核的 no-overwrite `deep_verifier_receipt.json`。
- 本锁生成时 successor natural/model/checkpoint/forward/GPU/K8s/training=`0`；只有新
  clean pushed commit/snapshot/UID/root 可从 group 1 重建。Deep verifier GO 前禁止
  E1-2/E1-3；GO 后仍必须先冻结独立 diagnostic/intervention pre-output lock。

## 2026-08-15 A5R7 active-gate recovery 前瞻锁

- A5R6 execution `b2e34999...` 在任何 natural row 前终止：
  `PRECOMPUTATION / A5R4_TRACKED_SOURCE_TEST_MAP_IS_STALE`。Input root 仅有
  execution identity 与 canonical BLOCKED；无 choice audit、geometry、sidecar、
  manifest 或 GO。Root/UID/controller 永久 no-resume/no-relaunch/no-reuse。
- 根因是 production loader 对 immutable A5R4 gate 调用了 historical full live-tree
  verifier；successor prepare 已合法变化，因此 tracked map 必然 stale。该失败与自然
  ambiguity、FPCT mechanism 或 accuracy 无关。
- 用户已前瞻批准 A5R7。历史 A5R4 gate 固定 SHA=`0f347757...`，只按 canonical
  file、whole-file SHA、strict v11 schema、identity/status/evidence 消费；当前源码改由
  独立 v14 A5R7 gate 验证。
- Current-source continuity 已机械验证：A5R3 publication AST `10/10` 不变；A5R4
  mode/publication AST 除 loader 外 `13/13` 不变；新 loader AST 单独冻结；mode static
  checks 全部 GO。
- Targeted production-preflight/controller/gate/A5R4 suite=`103 passed`；最终冻结 suite=
  `326 passed, 4 deselected`，test output SHA256=
  `cfac8e707d5f561cf13f0ce0cf54b2f0eec8a7aba4f7b9210f3120f39c5b7b3a`。
  Final gate status=`GO_PRE_NATURAL_A5R7_ACTIVE_GATE_RECOVERY`，evidence SHA256=
  `728cfb0f07be59270d364d877357e586135e416942df07dade04cd420b7c8543`，
  artifact SHA256=`15da7e33db7b967873c4a9d6f1e23f507435756343d58341a126475450f10eb4`。
  独立 verifier 与真实 production loader 均已无 monkeypatch 重放通过；successor natural
  access 仍为零。
- `E1-2/E1-3=NOT AUTHORIZED`，直到新 A5R7 root canonical GO、无 BLOCKED、strict
  durable deep receipt 与独立 pre-output lock 全部成立。
