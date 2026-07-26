# FPCT-E1 状态

> 当前阶段：Commit A pre-data execution lock
> 当前状态：`PRE-DATA LOCK GO / NATURAL EXECUTION NOT STARTED`
> 下一步：commit/push Commit A；随后只对 E0-design 生成 CPU input/topology lock 与 model-output-free runtime probe
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
| Commit A execution SHA | `PENDING_COMMIT` |
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
| Commit A execution lock | `GO / COMMIT PENDING` | E1-1 GO | 文档、代码、tests only | 尚无自然 tokenization/alignment/model output |
| E1-2 input/topology/provenance lock | `AUTHORIZED AFTER COMMIT A` | clean pushed Commit A | CPU tokenizer/alignment；model-output-free K8s runtime probe | 只允许 326 E0-design groups；生成 immutable plan/receipts |
| E1-2 C_post baselines | `NOT STARTED` | input/topology/provenance lock GO | 18 K8s shards；1 GPU/shard；最多双卡并行 | actual checkpoint inference；无训练；不得启动 F endpoint 竞态 |
| E1-2 F endpoints | `BLOCKED BY BASELINE MARKER` | 18 C_post closure | 18 K8s shards；1 GPU/shard | 只识别同 checkpoint F-C_post mechanism；无性能 GO |
| E1-2 analyzer/finalized receipt | `BLOCKED BY 36-SHARD CLOSURE` | C_post+F endpoints complete | bounded CPU analysis | 必须产生 deep-verified immutable `FINALIZED_E1_2` |
| E1-3 centered-lambda sweep | `BLOCKED BY FINALIZED_E1_2` | immutable finalized receipt | 72 additional K8s shards；1 GPU/shard | grid `{0,0.25,0.5,1,2}`；最终 closure=108；无训练 |
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

## Execution lock

- Source：clean pushed Commit A → exact git archive → Git tree/archive/mounted-byte receipt → K8s read-only mount；不得挂 live worktree。
- Input：仅 E0-design；逐样本 rows=`answer queries × certified parents × 28 × 16`，hard ceiling=`262144`，task-level `sum/min/p50/p95/max/argmax` 预先冻结。
- Topology：pre-sanitizer raw-to-runtime ledger 在任何 model output 前生成；uncertified rows 共同 slot-0 collapse，不携带 functional metric。
- Checkpoints：实际加载六个 immutable `final` projector trees；同时 hash `checkpoint-64`，并要求 projector set byte-identical；strict-attested load 的 missing/unexpected keys 为空。
- Runtime：immutable image probe、sender/receiver full asset trees、dev-data/config/checkpoint trees均在 load 前验证。
- Storage：Parquet-only、4096-row bounded streaming、atomic no-overwrite、recoverable exclusive claim lease。
- K8s：initial/finalized ConfigMaps immutable 且 `<1 MiB`；input/raw/finalized/source mounts read-only；只给当前 run 的 output root 可写；固定 `4090-48gx2`，每 shard 1 GPU，最多两个 shard 并行。
- DAG：`18 C_post -> marker -> 18 F -> 36 closure -> analyzer/finalized receipt -> 72 lambda additions -> 108 closure`。

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
- 未运行自然 E0-design tokenizer/alignment audit，未运行 pretrained model forward；
- 未运行 GPU/Kubernetes，未训练、未新增 seed、未启动 36-run；
- 当前 GO 只到 Commit A 与后续 CPU/model-output-free execution lock，不是 E1-2 scientific GO。
