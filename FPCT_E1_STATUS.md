# FPCT-E1 状态

> 当前阶段：E1-0 protocol/data firewall locking
> 当前状态：`GO`
> 下一步：E1-1 instrumentation hard gate
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

## 人工决策锁

2026-07-26 用户批准 `APPROVED_PROSPECTIVE_AMENDMENT`：

- E1-pilot source：support-fit-only certified groups；
- counts：ARC 128 / OpenBookQA 70 / MMLU-Redux 128；
- selection：frozen domain-separated SHA ordering；
- role：exploratory mechanism pilot；
- confirmatory eligibility：no。

已物化 652-row split manifest。E0-design 与 E1-pilot 各为 `128/70/128`，distinct content-group intersection=`0`。Selection 未访问 label、answer、prediction、correctness 或 model outcome。

## 阶段表

| 阶段 | 状态 | 依赖 | 允许资源 | 决策/边界 |
|---|---|---|---|---|
| E1-0 protocol + split lock | `GO` | E0 result commit `613958a...` + human amendment | 文档、hash-only data lock | E0 readonly；E1-pilot sealed；confirmatory sealed |
| E1-1 instrumentation hard gate | `AUTHORIZED / NOT STARTED` | E1-0 GO | synthetic tensor + parity tests；按主任务授权执行 | 必须通过 oracle、ON/OFF parity、cross-query variance 与 multi-step accumulation |
| E1-2 E0-design mechanism/topology audit | `CONDITIONAL / NOT STARTED` | E1-1 GO | E0 checkpoints；无训练 | 仅 E0-design；full teacher-forced answer audit；无性能 GO |
| E1-3 E0-design centered-λ sweep | `CONDITIONAL / NOT STARTED` | E1-2 complete | E0 checkpoints；无训练 | grid `{0,0.25,0.5,1,2}`；same-state diagnostic |
| E1 root-cause/operator freeze | `NOT AUTHORIZED` | E1-3 complete + frozen report | protocol only | 只允许前瞻选择一个因素；必须早于 E1-pilot outcome |
| E1-pilot | `SEALED / NOT RUN / NOT READ` | 新 operator amendment + 单独执行授权 | 未授权 | exploratory only；永无 confirmatory eligibility |
| Confirmatory | `SEALED / NOT AUTHORIZED` | 后续完整阶段链 | 未授权 | model-selection/test/formal seeds 不得读取 |

## Instrumentation hard gate

进入 E1-2 前必须全部满足：

- formula/synthetic oracles 全部通过；
- instrumentation ON/OFF 对 logits、loss、generation 与 cache 等价；
- query-changing synthetic case 的 cross-answer-query `gamma_query_variance > 0`；
- 多 decode step 累计不被最后一个 forward 覆盖；
- invalid probability/gradient 精确为 0；
- no NaN/Inf，capture 不改变 model output。

失败则 E1-1=`BLOCKED`，E1-2/3 不得运行。

## Centered-λ 锁

对 `Kbar_i=sum_j A_ij Ktilde_ij`、`Vbar_i=sum_j A_ij Vtilde_ij`：

```text
K(lambda)_ij = Kbar_i + lambda * (Ktilde_ij - Kbar_i)
V(lambda)_ij = Vbar_i + lambda * (Vtilde_ij - Vbar_i)
lambda        = {0, 0.25, 0.5, 1, 2}
```

`lambda=0` 必须等于 C_post/replicated-collapse；`lambda=1` 必须等于 F；`lambda=2` 仅作诊断。所有 λ 共用原 A、mask、native atoms、global denominator 和 checkpoint，不训练、不调参、不只报告最优 λ。

## 当前执行确认

- E0 文件未修改；
- E1-pilot 未运行、未读取；
- confirmatory model-selection/test 未释放；
- 未训练、未新增 seed、未启动 36-run；
- 当前只授权按 `instrumentation hard gate -> E0-design audit -> E0-design centered-lambda sweep` 顺序推进。
