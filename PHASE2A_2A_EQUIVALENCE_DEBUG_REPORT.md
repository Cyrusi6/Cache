# Phase 2A-2a Llama3.2 Gate-1 Equivalence Debug Report

## 结论

正式分类为 **`historical_environment_or_reference_drift`**，不是 baseline/runtime numerical nondeterminism，也不是 geometry instrumentation observer effect。

在同一 Pod、同一物理 RTX 4090、同一 checkpoint/config/runtime 下：

| 比较 | ARC | OpenBookQA | 合计 | 结论 |
| --- | ---: | ---: | ---: | --- |
| frozen reference vs OFF-A | 11/351 mismatch | 4/158 mismatch | 15/509 | 历史 reference 漂移 |
| frozen reference vs OFF-B | 11/351 mismatch | 4/158 mismatch | 15/509 | 与 OFF-A 是同一组差异 |
| OFF-A vs OFF-B | 0/351 | 0/158 | 0/509 | baseline 完全可复现 |
| ON-A vs ON-B | 0/351 | 0/158 | 0/509 | instrumentation 完全可复现 |
| OFF-A vs ON-A | 0/351 | 0/158 | 0/509 | reductions 未改变输出 |
| OFF-A vs ON-B | 0/351 | 0/158 | 0/509 | 独立 ON 重复仍未改变输出 |

历史 15 个 mismatch 的字段分布也被精确复现：ARC 为 5 个答案变化、11 个完整生成文本变化、7 个生成长度变化；OpenBookQA 为 3 个答案变化、4 个完整生成文本变化、2 个生成长度变化。

NOOP-A/NOOP-B 没有运行。冻结判定树只在 OFF 稳定而 ON 不稳定或不同于 OFF 时触发 NOOP；本次 ON 与 OFF 已是 509/509 exact。因此本报告不提出或启动新的 instrumentation/selector 实验。

## 根因证据

1. 上一阶段 Phase 2A-2a 的 Llama3.2 ON 结果与本次 OFF-A 在 509 条上完全一致。上一阶段 ARC/OpenBookQA 使用 `GPU-1b779...` 与 `GPU-4696...`，本次全部运行使用 `GPU-c1ff...`；跨不同物理 GPU 仍 509/509 exact，排除了特定 GPU 的数值差异。
2. Llama3.2 tokenizer chat template 在未显式传入 `date_string` 时调用 `strftime_now("%d %b %Y")`，把运行日期写入 system prompt。历史 reference 文件生成于 2026-07-17；上一阶段 ON 与本次复跑均在 UTC 2026-07-19，当前 evaluator 日志明确记录 `Today Date: 19 Jul 2026`。
3. 无样本、无标签的 tokenizer probe 显示 `17 Jul 2026` 与 `19 Jul 2026` 的渲染文本 SHA 和 token-id SHA 均不同；日期 token 首项由 `1114` 变为 `777`。历史代码与当前代码都没有显式冻结 `date_string`。
4. TinyLlama/Qwen2.5 在上一阶段没有 Gate-1 mismatch，而失败集中于使用该动态日期模板的 Llama3.2 pair，这与日期输入漂移机制一致。

因此，最具体且被直接验证的输入漂移来源是 **Llama3.2 动态 Today Date**。由于历史运行没有保存完整 tokenized prompt，本报告将“动态日期导致全部 15 个输出变化”表述为强证据推断，而不是用一次未授权的 date-17 GPU replay 冒充直接因果证明。

## 执行与 provenance

- Base commit：`00db4c7eeffc57a852c67fd1aedad9fd823ca528`
- GPU execution commit：`7e6c62c4ba46f5825dda59667b449838eafa251b`
- Manifest SHA256：`e02aaf6bfb10708e39ba07b44ceff7aef556d20ae7c4101886d8c6ac305546a4`
- Submitted Job YAML SHA256：`009fbba850c9e4b00b697b96a870d069938a8f05954d9bf2bb0e2e1816390422`
- B6 checkpoint directory SHA256：`ca789cc72884de477c5f02349a156c25774f0095f1d3a0f544bfba9929547cc5`
- Runtime fingerprint：`e727d731b896b8e6e0ee93320fcb2eb127f0ead4ba23be2af04a025d8d95233c`
- Image：PyTorch 2.6.0 CUDA 12.4 runtime；Python 3.11.11；Torch 2.6.0+cu124；cuDNN 9.1；driver 595.71.05。
- Node/physical GPU：`4090-24gx4` / `GPU-c1ff6c70-6280-a8be-78a6-5575bb320859`；8 个 evaluator run 的 GPU UUID 在每次运行前后均一致。
- Kubernetes Job：`p2a2-eq-debug-7e6c62c4-e02aaf6b`，完成后已删除；无残留 Job/Pod。
- 正式运行 8 个 cell：OFF-A/OFF-B/ON-A/ON-B × ARC/OpenBookQA。MMLU、NOOP、selector、geometry predictability 与 sealed test 均未运行/读取。

## 交付物

- 小型正式 aggregate：`recipe/eval_recipe/phase2a_2a_equivalence_debug/aggregate.json`，源 SHA256 `359497eeb0e12e4ad7f47244f6f860212da02d59e23659ce160017b2ca2fa9b4`。
- 确定性复跑检查：`recipe/eval_recipe/phase2a_2a_equivalence_debug/determinism_rerun_checks.json`，源 SHA256 `ae4da01e2d00fd620958b4bec07cd85cf6bed282ad1224c62c5183727dd896a5`。
- 环境/GPU provenance：`recipe/eval_recipe/phase2a_2a_equivalence_debug/environment_gpu_provenance.json`。
- 动态日期 probe：`recipe/eval_recipe/phase2a_2a_equivalence_debug/date_template_probe.json`，源 SHA256 `3df33e3e7e91c2c6d57579ccacf5436de74a54ae8a9a90c2c5fe65aa1831d44b`。
- 上一阶段 ON vs 当前 OFF secondary comparison：`recipe/eval_recipe/phase2a_2a_equivalence_debug/secondary_phase2a2_on_vs_current_off.json`，源 SHA256 `ba06ddebc034cce547d64729d7b418597d86db5e02aef5a909ac43d62e6c7625`。
- 冻结 manifest、Job YAML 与 12 个 config 均位于同一 recipe 目录。
- 含完整生成文本的 509 行逐例文件只保留在 `/netdisk/lijunsi/c2c-phase2a2-equivalence-debug/results/diagnostic/per_example_outputs.jsonl`，SHA256 `119f3e158da304b5a030bf6493de4bdbde5a604c3bd704ce13eeba4f418462e6`，不提交 Git。

## 审查建议

上一阶段 Llama3.2 Gate-1 NO_GO 不能再解释为 instrumentation 改变生成；它是把不同日期 prompt 的历史 reference 与当前输出做 exact 比较造成的无效等价性检查。后续若获授权重做 Gate 1，应先冻结 `date_string` 并记录最终 prompt/token-id hash；本任务在此停止，不自动恢复 geometry pilot 或 selector。
