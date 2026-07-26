# FPCT-E1 状态

> 当前阶段：Commit A3 E0-design CPU input lock
> 当前状态：`INCONCLUSIVE_RESOURCE_CEILING / REVIEW REQUIRED`
> 下一步：等待人工决定是否停止，或以前瞻性新版协议实现 representation-preserving streaming；不得自动重跑
> 更新时间：2026-07-26（Asia/Shanghai）

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
| Successor execution SHA | `HUMAN_REVIEW_REQUIRED_NOT_ASSIGNED` |
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

| 阶段 | 状态 | 依赖 | 允许资源 | 决策/边界 |
|---|---|---|---|---|
| E1-0 protocol + split lock | `GO` | E0 result commit `613958a...` + human amendment | 文档、hash-only split lock | E0 readonly；E1-pilot/confirmatory sealed |
| E1-1 instrumentation hard gate | `GO` | E1-0 GO | synthetic tensor + random-small Qwen CPU | pre-data suite `169 passed`；ON/OFF bitwise；cross-query variance>0 |
| Commit A2 execution lock | `INCONCLUSIVE / ABANDONED` | E1-1 GO + operational closure | CPU input lock | d169 在自然 alignment 中因 slot-order taxonomy bug fail-closed；0 artifact/model/GPU |
| Commit A3 correctness lock | `GO` | d169 receipt + synthetic permutation regression | 文档、单一 classifier 修复、tests only | clean/pushed execution=`612697df...`；只排序派生 intersections；未改 candidate/A/operator/threshold/split |
| E1-2 input/topology/provenance lock | `INCONCLUSIVE_RESOURCE_CEILING / ABANDONED` | clean pushed A3 | CPU tokenizer/alignment only | observed logical rows `616448 > 262144`；0 artifact/model/GPU；禁止 resume/reuse |
| E1-2 C_post baselines | `BLOCKED / NOT STARTED` | human-approved successor protocol + new complete input lock | 目前无授权资源 | 未创建 runtime probe/plan/ConfigMap/Job，未运行 checkpoint inference |
| E1-2 F endpoints | `BLOCKED / NOT STARTED` | 18 C_post closure | 目前无授权资源 | 只识别同 checkpoint F-C_post mechanism；无性能 GO |
| E1-2 analyzer/finalized receipt | `BLOCKED / NOT STARTED` | C_post+F endpoints complete | 目前无授权资源 | 必须产生 deep-verified immutable `FINALIZED_E1_2` |
| E1-3 centered-lambda sweep | `BLOCKED / NOT STARTED` | immutable finalized receipt | 目前无授权资源 | grid `{0,0.25,0.5,1,2}`；最终 closure=108；无训练 |
| E1 root-cause/operator freeze | `NOT AUTHORIZED` | E1-3 complete + frozen report | protocol only | 只允许新的前瞻性 amendment 选择一个因素 |
| E1-pilot | `SEALED / NOT RUN / NOT READ` | 新 operator amendment + 单独授权 | 未授权 | exploratory only；永无 confirmatory eligibility |
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
