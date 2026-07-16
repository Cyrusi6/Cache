# AGENTS.md

## 1. 目标与范围

- 本文件适用于整个仓库。
- 默认使用中文沟通和记录；代码、命令、配置键保留英文。
- 目标是在 C2C/Rosetta 基础上完成 ICLR 2027 论文研究。
- 工作包括调研、设计、实现、实验、分析和论文材料整理。
- 用户当前指令优先；文档与代码冲突时以当前代码、配置和测试为准。
- 开始任务前运行 `git status --short`，保留用户已有改动。

## 2. 当前研究基线

- 研究对象是跨模型 KV Cache 投影、对齐与融合。
- Receiver 负责生成，Sharer 提供语义，Projector 转换 KV Cache。
- 当前主线为 Route-1 v2.2，tag 为 `v2.2_token_mlp_entropy050-baseline`。
- 基线配置：`soft_span_overlap_v2`、`uniform`、entropy alpha `0.5`、`token_mlp`。
- v2.2 小闭环均值 `50.82`，v2.1 对照 `49.77`。
- validation loss 不能替代下游 benchmark。
- 不继续无目的叠加 gate、scale、calibration 或正则项。
- 下一主线是候选约束下的 learned flexible alignment。
- 新方法必须与 v2.2 做同配置公平对照，并保留负结果。

## 3. 必读资料

- 总览：`README.md`；运行：`RUNBOOK.md`；架构：`CORE_ARCHITECTURE.md`。
- 工程背景：`CLAUDE.md`，其中“没有 test 目录”的描述已过时。
- 迭代总结：`C2C_Route1_v2.x迭代总结与后续安排备忘.md`。
- 论文对照：`C2C_论文技术部分代码对照解读.md`。
- 对齐方向：`C2C_跨Tokenizer柔性对齐改进方向研究备忘.md`。
- 只读取当前任务需要的内容。

## 4. 项目结构

- `rosetta/model/wrapper.py`：`RosettaModel` 与 KV 融合执行。
- `rosetta/model/projector.py`：Projector；`aligner.py`：跨 tokenizer 对齐。
- `rosetta/train/dataset_adapters.py`：数据、对齐元数据、collator。
- `rosetta/train/model_utils.py`：层映射；`answer_*.py`：辅助损失。
- `rosetta/utils/evaluate.py`：评测加载；`registry.py`：注册与序列化。
- `script/train/SFT_train.py`：训练入口。
- `script/evaluation/unified_evaluator.py`：评测入口。
- `script/analysis/`：诊断；`recipe/`：配置；`test/`：回归测试。
- `local/`：临时配置、checkpoint 和结果，不提交 Git。

## 5. 核心工程约定

- Recipe 是训练和评测的主要接口；一次性实验先改 recipe，不先改入口。
- `model.mapping` 必填，使用 `last_aligned` 或 `k_nearest`。
- 多 Sharer 通过 `kv_cache_index[:, 0]` bitmask 选择。
- `-1` 表示不投影，`1` 表示 Sharer 1，其他正数表示多来源组合。
- 保持 chat template 的 `enable_thinking=False`，除非实验专门研究 prompt。
- 不改 checkpoint 布局，除非同时补兼容加载与测试。
- 活跃入口优先于 `deprecated`、oracle 或旧实验脚本。

## 6. Conda 环境

- 唯一默认环境：`c2c-py310-cu124`；禁止用 Python 3.7 的 `base` 跑项目。
- 默认命令变量：

```bash
export C2C_ENV=/home/lijunsi/miniconda3/envs/c2c-py310-cu124
export PYTHON="$C2C_ENV/bin/python"
export TORCHRUN="$C2C_ENV/bin/torchrun"
```

- 需要激活时：

```bash
source /home/lijunsi/miniconda3/etc/profile.d/conda.sh
conda activate c2c-py310-cu124
```

- 基线：Python 3.10.20、PyTorch 2.6.0+cu124、Transformers 4.52.4。
- 验证命令：

```bash
"$PYTHON" -c "import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
```

- `environment.yml` 含外部用户 prefix，不直接照搬。
- 缺依赖时使用 `pip install -e ".[dev,training,evaluation]"`。
- 未经要求不升级 torch、transformers 或 CUDA 依赖。

## 7. 本地 GPU

- 使用前运行 `nvidia-smi`；本机当前有 4 张 RTX 4090。
- 小测试优先单卡，用 `CUDA_VISIBLE_DEVICES` 明确选卡。
- 不抢占高负载 GPU；多卡进程数必须等于可见 GPU 数。
- 并行实验使用不同 `master_port`。

## 8. 测试与静态检查

- 修改前跑相关测试；修改后先定向测试，再跑全量测试。
- Pytest 临时目录放在 `local/tmp/`，避免 `/tmp/pytest-of-lijunsi` 所有权错误。
- 全量测试：

```bash
"$PYTHON" -m pytest -q --no-cov --basetemp=local/tmp/pytest-all
```

- 当前基线：`65 passed`；已知 Pydantic warning 不算失败。
- 对齐测试：

```bash
"$PYTHON" -m pytest test/test_aligner_span_overlap.py -q --no-cov --basetemp=local/tmp/pytest-aligner
```

- 答案路由测试：

```bash
"$PYTHON" -m pytest test/test_answer_margin_routing.py test/test_answer_prior_regularization.py -q --no-cov --basetemp=local/tmp/pytest-answer
```

- 静态检查按需运行：

```bash
"$C2C_ENV/bin/black" --check rosetta script test
"$C2C_ENV/bin/isort" --check-only rosetta script test
"$C2C_ENV/bin/flake8" rosetta script test
"$C2C_ENV/bin/mypy" rosetta
```

- 只格式化本次涉及文件，不制造历史清理 diff。

## 9. 实验闭环

1. 写清研究假设、机制和预期现象。
2. 指定 v2.2 或其他明确 baseline。
3. 一次只改变一个关键变量。
4. 补单元测试、断言或诊断指标。
5. 运行 `eval_only` 或最小 smoke test。
6. 运行小样本闭环并检查失败样例。
7. 确认有效后扩大训练规模和 seed。
8. 运行统一 benchmark、效率和稳定性分析。
9. 更新 `FRAMEWORK_UPDATE.md` 后提交。
- 不以单次最好结果、多次挑选或单一 loss 宣称提升。

## 10. 配置与产物

- 临时训练配置：`local/tmp/train_recipes/<study>/`。
- 临时评测配置：`local/tmp/eval_configs/<study>/`。
- Checkpoint：`local/checkpoints/<study>/`。
- 结果：`local/final_results/<study>/`。
- 不覆盖标准 recipe；配置名包含方法、模型、数据量和 seed。
- 公平对照保持模型、数据、步数、生成参数和评测集一致。
- Smoke test 使用约 128 条或更少；小闭环可从 2048 条开始。
- 无 W&B 登录时使用 offline mode；不在代码中硬编码输出路径。

## 11. 训练

- 单卡：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" script/train/SFT_train.py --config <train-config>
```

- 链路检查：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" script/train/SFT_train.py --config <train-config> --eval_only
```

- 多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN" --nproc_per_node=2 --master_port=29501 script/train/SFT_train.py --config <train-config>
```

- 参考配置：`recipe/train_recipe/C2C_0.6+0.5.json`。
- 仅训练 Projector 时确认 base 和 teacher 已冻结。
- 训练后检查 `config.json`、`final/projector_*.pt/json` 和 `projector_config.json`。

## 12. 评测与分析

- 统一评测：

```bash
"$PYTHON" script/evaluation/unified_evaluator.py --config <eval-config>
```

- 参考配置：`recipe/eval_recipe/unified_eval.yaml`。
- 首次评测使用单数据集、单 subject、小 `limit`。
- YAML 中的 `eval.gpu_ids` 为准；`checkpoints_dir` 指向 `final/`。
- 主结果至少覆盖 MMLU-Redux、ARC-Challenge、OpenBookQA。
- 有条件时补 GSM8K、GPQA、延迟、吞吐和显存。
- Gated dataset 无权限时记录原因，不伪造结果。
- 对齐、置信度、flip 和 candidate 分析使用 `script/analysis/route1_*`、`route3_*`。
- 图表必须能从保存的 CSV/JSON 重建，不手工改原始数值。

## 13. 文献研究

- 优先查论文原文、OpenReview、arXiv 和官方仓库。
- 记录检索日期、链接、方法差异和可验证结论。
- 区分论文主张、代码事实、项目实验和推断。
- 未复现结果不得写成项目已验证结论。
- 新方法必须说明与 C2C、token alignment、latent communication 的区别。

## 14. Kubernetes 边界

- 每次先确认 `kubectl config current-context`；当前 context 为 `default`。
- 仓库没有 C2C 专属 namespace 或现成 manifest。
- `agentic-grpo`、`matrix-demo` 等属于其他工作负载，禁止误用。
- 未获用户明确 namespace、镜像、挂载和资源要求时，只做只读检查。
- 所有写命令显式提供 `--context` 和 `-n`，禁止依赖默认 namespace。
- 禁止操作 `kube-system`、`headlamp`，禁止调度到 `NotReady` 节点。

## 15. Kubernetes 命令

- 只读检查：

```bash
kubectl config current-context
kubectl config get-contexts
kubectl get namespaces
kubectl get nodes -o wide
kubectl top nodes
kubectl get pods -A -o wide
```

- 查看 GPU 与权限：

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl auth can-i create jobs.batch -n "$NS"
```

- 提交前必须 dry-run：

```bash
kubectl apply --dry-run=server --context "$KUBE_CONTEXT" -n "$NS" -f <manifest.yaml>
kubectl apply --context "$KUBE_CONTEXT" -n "$NS" -f <manifest.yaml>
```

- 监控：

```bash
kubectl get jobs,pods --context "$KUBE_CONTEXT" -n "$NS" -o wide
kubectl describe job/<job-name> --context "$KUBE_CONTEXT" -n "$NS"
kubectl logs -f job/<job-name> --context "$KUBE_CONTEXT" -n "$NS"
kubectl get events --context "$KUBE_CONTEXT" -n "$NS" --sort-by=.lastTimestamp
```

- GPU Job 声明 `nvidia.com/gpu` request/limit。
- 只删除本任务创建且用户允许删除的 Job。
- 禁止删除 namespace、PVC、系统资源、他人 Pod；禁止 force 删除。

## 16. 结果记录

- 记录日期、实验 ID、研究假设、baseline 和唯一变量。
- 记录 git commit、环境版本、GPU、卡数和运行时长。
- 记录模型、数据集、样本量、seed、配置和完整命令。
- 记录 checkpoint、结果目录、逐任务指标、均值和方差。
- 记录显存、延迟、吞吐、异常、失败样例和负结果。
- 明确区分事实、解释、局限和下一步假设。

## 17. FRAMEWORK_UPDATE.md

- 每次方法、训练、评测或基础设施改进后更新。
- 固定结构：日期与主题、研究目标、核心改动、实验配置、验证结果、结论与下一步。
- 只写实际完成内容；未完成实验标记“待验证”。
- 不删除历史和负结果；更新与代码放在同一 commit。

## 18. Git 管理

- 不使用 Pull Request，不创建 PR。
- 通过 Git commit 管理变更，默认直接提交并推送 `main`。
- 提交前确认分支为 `main`，只暂存本任务文件。
- 不提交 `local/`、checkpoint、权重、缓存或密钥。
- 禁止 `git reset --hard`、修改已有 commit 和 `git push --force`。
- 提交与推送：

```bash
git commit -m "提交信息"
git push origin main
```

- 推送被拒绝时停止并报告，不改写远端历史。

## 19. 完成标准

- 用户要求已实现；相关测试和全量测试无新增失败。
- 配置、命令、产物和结论可复现。
- `FRAMEWORK_UPDATE.md` 已更新，`git diff --check` 通过。
- 无无关文件被提交；已按要求 commit 并推送 `origin/main`。
