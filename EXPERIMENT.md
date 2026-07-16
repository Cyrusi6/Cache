# EXPERIMENT.md

## 2026-07-16：Kubernetes C2C Route-1 v2.2 Smoke

### 研究目标

验证 Route-1 v2.2 `token_mlp + entropy050` checkpoint 能否通过 Kubernetes 在本机 RTX 4090 上完成真实 C2C 加载、跨 tokenizer 对齐和生成式评测。

### 实验配置

- Receiver：`Qwen/Qwen3-0.6B`。
- Sharer：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`。
- Alignment：`soft_span_overlap_v2`，top-k 4，uniform weighting。
- Confidence：entropy alpha 0.5，floor 0.5，fallback 0.25。
- Gate：`token_mlp`，max delta 2.0。
- Checkpoint：2048 个 MMLU auxiliary_train 样本、seed 42、28 个 projector。
- Benchmark：AI2-ARC Challenge，generate，greedy，`limit=4`。
- Kubernetes：`c2c-research`，节点 `4090-24gx4`，1 GPU。
- 配置：`local/tmp/eval_configs/k8s_v22_smoke/route1_v22_token_mlp_ai2_arc_limit4.yaml`。
- 结果：`local/final_results/k8s_v22_smoke/ai2_arc_limit4/`。

### 运行命令

```bash
bash bash/k8s/gpu_job.sh submit \
  --name c2c-v22-ai2arc-mirror-limit4 \
  --gpus 1 \
  --follow \
  -- python script/evaluation/unified_evaluator.py \
     --config local/tmp/eval_configs/k8s_v22_smoke/route1_v22_token_mlp_ai2_arc_limit4.yaml
```

### 验证结果

- 最终 Job：`c2c-v22-ai2arc-mirror-limit4-20260716-170321-690328`。
- 评测完成：4/4，accuracy 100%，skipped 0。
- 平均输入长度：153.75 tokens。
- 平均生成长度：7.0 tokens。
- 示例输出：`The correct answer is C.`。
- Summary：`Rosetta_ai2-arc_generate_20260716_090408_summary.json`。
- 初次运行访问 `huggingface.co` 失败；确认 `hf-mirror.com` 可访问后，将调度器默认 `HF_ENDPOINT` 设为镜像站，普通命令复跑成功。
- 完成后测试 Job 已删除，namespace 和运行环境缓存保留。

### 结论

当前 Kubernetes 调度器可运行真实 C2C v2.2 checkpoint，模型、projector、对齐、数据集和生成链路均已打通。4 样本 100% 仅证明 smoke 链路成功，不用于论文性能比较。

## 2026-07-16：Kubernetes 统一数据目录 Smoke

### 研究目标

验证宿主机数据软链接、Kubernetes 只读挂载、`C2C_DATA_ROOT` 和本地优先加载器在真实 Pod 中能够协同工作。

### 实验配置

- Namespace：`c2c-research`。
- 节点：`4090-24gx4`。
- GPU：1 × RTX 4090。
- 数据根：`/datasets/c2c`。
- 加载数据：OpenBookQA `main/test`、LongBench-E `qasper_e/test`。
- Job：`dataset-mount-smoke-20260716-184328-905304`。

### 验证结果

- Pod 内 `C2C_DATA_ROOT=/datasets/c2c` 存在。
- OpenHermes、MMLU、MMLU-Redux、LongBench、OpenBookQA、AI2-ARC、GSM8K 七个链接全部可解析。
- OpenBookQA 成功加载 500 条，Qasper-E 成功加载 224 条，日志明确显示使用本地路径。
- Job 状态为 `Complete`，完成后已删除 Job 与 Pod。

### 结论

统一数据目录已在真实 Kubernetes 任务中验证可用；后续训练和评测可直接沿用原命令，由代码自动选择本地数据，C-Eval 缺失时回退 Hugging Face。

## 2026-07-16：Kubernetes 统一模型目录 Smoke

### 研究目标

验证 C2C 底座模型和官方 Fuser 的只读挂载、环境变量及本地优先路径解析在真实 Pod 中可用。

### 实验配置

- Namespace：`c2c-research`。
- 节点：`4090-24gx4`。
- GPU：1 × RTX 4090。
- 宿主机模型根：`/home/lijunsi/projects/KVcache/models/c2c`。
- Pod 模型根：`/models/c2c`。
- Jobs：`model-mount-smoke-20260716-202557-004837`、`model-readonly-smoke-20260716-202659-775799`。

### 验证结果

- Pod 内 `C2C_MODEL_ROOT=/models/c2c`。
- Qwen2.5-0.5B 配置、C2C_Fuser 目录和 Qwen3-0.6B 跨挂载软链接均可读取。
- `Qwen/Qwen3-8B` 自动解析为 `/models/c2c/Qwen3-8B`。
- 写入模型目录被内核以 `EROFS`（errno 30）拒绝，确认只读挂载生效。
- 两个 Job 均为 `Complete`，完成后已删除 Job 与 Pod。

### 结论

Kubernetes 任务现在可以直接复用统一模型库；现有 Hugging Face ID 和旧模型绝对路径可本地优先解析，通过 `/models/c2c` 写入公共权重会被拒绝。
