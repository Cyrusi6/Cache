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

## 2026-07-17：Route-1 v2.2 Identifiability 第一阶段（运行中）

### 研究目标

第一阶段只回答三个问题，不开发任何新方法：

1. v2.2 的提升是否来自保留多个 source candidates？
2. entropy confidence 是否提供与正确/错误迁移相关的有效信息？
3. token/head gate 是否只是增加了自适应容量？

### 固定实验配置

- Receiver：`Qwen/Qwen3-0.6B`。
- 主 Sharer：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`。
- 跨模型 Sharer：`Qwen/Qwen3-1.7B`、`Qwen/Qwen2.5-0.5B-Instruct`、`meta-llama/Llama-3.2-1B-Instruct`。
- 训练数据：MMLU `auxiliary_train` 2,048 条。
- Seeds：42、43、44；相同 seed 的方法共享冻结的 train/eval indices 与顺序。
- 评测：MMLU-Redux、AI2-ARC Challenge、OpenBookQA，只作为开发集。
- 所有方法固定 epoch、per-device batch、gradient accumulation、学习率、fuser 架构、数据顺序与 final-checkpoint 选择规则，禁止单独调参。
- 禁止加入 RoPE correction、OT、byte transport、Route3 或新 loss。

### 实验矩阵

- B0：Receiver-only，无 cache transfer、无训练。
- B1：C2C longest，原始 hard remapping。
- B2：hard offset span，top-k=1，无 token/head gate。
- B2-constant：B2 的常数 confidence 匹配控制，用于与 B5 做纯 gate 容量比较。
- B3：soft uniform span，无 alignment confidence、无 gate。
- B4：soft uniform span + static entropy，无 token/head gate。
- B5：top-k=1 + constant confidence + token/head gate。
- B6：soft span + entropy + token/head gate，完整 v2.2。
- B6-constant：以常数 confidence/零 entropy 替换原 entropy 信号。
- B6-shuffle：在序列内联合打乱 confidence 与 entropy，保留数值分布但破坏位置对应。

### B6 seed 42 复现门控

预注册门槛为：旧 checkpoint 用当前 evaluator 重评后的 macro 与历史 50.8176% 相差不超过 0.10pp；新复训相对旧 checkpoint 的 macro 绝对差不超过 1.0pp，且每个任务绝对差不超过 2.0pp。

历史 B6 checkpoint 在当前 evaluator 上得到：

- MMLU-Redux：47.0347%，5,615 条有效样本。
- ARC：54.7826%，1,150 条。
- OpenBookQA：50.6000%，500 条。
- Macro mean：50.8058%，与原记录 50.8176% 相差 -0.0118pp，历史 checkpoint 重评通过。

第一次复训使用独立 `torch.Generator().manual_seed(42)` 创建 split，结果为：

- MMLU-Redux：44.6126%，相对 reference -2.4221pp。
- ARC：54.9565%，相对 reference +0.1739pp。
- OpenBookQA：49.8000%，相对 reference -0.8000pp。
- Macro mean：49.7897%，相对 reference -1.0161pp。
- 判定：同时违反 macro 1.0pp 和单任务 2.0pp 门槛；按协议立即停止后续方法训练并排查。

根因是 split 实现漂移，而不是模型、数据文件或 v2.2 模块漂移：April v2.2 在模型和 `token_mlp` projector 初始化之后，使用进程全局 Torch RNG 执行 `random_split`；后续独立 seeded generator 改变了 train/eval 成员及样本顺序。为消除该混杂，seed 42 的 April indices 已冻结到 `recipe/train_recipe/identifiability/splits/mmlu_aux2048_seed42_april_v22.json`，seeds 43、44 也各自冻结；所有方法直接加载 manifest，不再现场重抽 split。

使用 `legacy_global_rng` 恢复 April 路径后的复训结果：

- Training Job：`route1-ident-b6-legacy-s42-20260717-163859-544026`。
- 历史与复现均为 64 个 optimizer steps，逐步训练轨迹逐位一致。
- 28 层 projector 共比较 1,148 个 tensors、485,647,428 个参数，全部 `torch.equal`。
- 两个 checkpoint 目录 SHA256 均为 `a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`。
- 最终判定：B6 seed 42 reproduction gate 通过。由于 checkpoint 逐位相同，无需重复一次只验证等价性的 gate 评测；canonical suite 仍会用当前 evaluator 完整重评，以产生统一 prediction 与 diagnostics。

门控记录：`local/final_results/route1_identifiability/reproduction_gate_seed42.json` 与 `local/final_results/route1_identifiability/reproduction_gate_seed42_final.json`。

### 权重复用规则

- 仅复用 TinyLlama→Qwen3 的 B6 seed 42，来源为逐位验证的 checkpoint，运行时通过 `recipe/train_recipe/identifiability/reuse_step1_b6.json` 校验目录 SHA256。
- 复用只跳过训练，不复用旧评测；MMLU-Redux、ARC、OpenBookQA 必须由当前 evaluator 重新运行。
- 历史 B3/B4 虽有权重，但其 split/data order 与冻结协议不一致，不进入主表；B1/B2 的旧权重训练规模也不符合 2,048 条固定条件。
- Suite 共 67 runs、66 个名义训练 runs、67 组三任务评测；66 个名义训练中扣除 checkpoint-only B6 后，需要新训练 65 个方法/seed/pair 组合，B0 本身无训练。

### 三条四卡跨节点流水线

- Phase1 共 37 runs：lane A/B/C 分别为 12/13/12。
- Conditional 跨模型多 seed 阶段共 30 runs：每条 lane 10 个；只有 seed 42 的跨模型方向满足门控后才释放。
- Lane A：`4090-24gx4`，4 GPUs；Llama3.2 pair 固定 affinity 到该 lane。
- Lane B/C：`4090-24gx8` 上两个相互独立的 4-GPU Pods，共享节点但不共享 GPU allocation。
- 每条 lane 内按 run 串行执行 train→ARC/OpenBookQA/MMLU-Redux，并用状态文件、依赖和 gate 防止越级运行；不同 lane 可并行。
- 三条 lane 的 `C2C_MODEL_ROOT` 与 `C2C_DATA_ROOT` 均指向同一份 `/netdisk` 资产，不再按节点使用不同的 local-first 路径。
- 本条记录不列尚未稳定确认的 lane Job IDs，后续在实际运行记录中补充。

### `/netdisk` 共享资产与跨节点审计

共享根为 `/netdisk/lijunsi/c2c-route1-identifiability`，用于服务器间复用模型、数据、checkpoint、workspace、lane state 和结果；当外部下载慢于共享盘约 100MB/s 的传输速度时，优先从共享盘复制。

- 数据：MMLU、MMLU-Redux、AI2-ARC、OpenBookQA 的固定本地副本。
- 模型：Qwen3-0.6B、TinyLlama-1.1B、Qwen2.5-0.5B、Qwen3-1.7B、Llama3.2-1B。
- B6：bitwise-verified seed 42 checkpoint 已复制到共享 checkpoint 目录。
- 五套模型同时冻结关键文件 SHA256 与完整目录树 SHA256，覆盖 `generation_config.json`、special token、vocab/merges 等运行时文件；四套数据也冻结完整目录树 SHA256。Qwen3-1.7B 与 Llama3.2 均直接使用共享副本，避免慢速重复下载。
- 24gx8 已确认可访问共享根；stager 在发布 workspace-ready marker 前会一次性核对固定 HF revision、Python package 版本、五套共享模型及四套共享数据的完整目录树哈希。不一致时三条 lane 都不会启动。
- Llama3.2 保持 lane A affinity 仅用于稳定的任务分配与三 lane 负载均衡，不再依赖 lane A 的节点本地模型来源。
- 基础 PyTorch 镜像未内置 `git` 时，允许由控制节点把已提交的 detached checkout 预置到共享盘；stager 仅在 commit-specific ready marker 存在且 `.git/HEAD` 精确等于目标 40 位 SHA 时跳过容器内 clone，之后仍执行同一套资产、环境和计划审计。
- `/netdisk` 是 autofs 根，Pod 不设置其深层目录为 OCI `workingDir`；所有 Python 命令使用绝对入口，待 volume mount 完成后由 `container_entrypoint.py` 校验并切换到 project root，避免 runtime 在挂载前创建 `/netdisk/lijunsi/...` 导致权限错误。
- 共享 runtime bootstrap 使用 tracked pip constraints 固定 `transformers/datasets/accelerate/wandb/peft` 的审计版本；constraint 文件 SHA256 纳入 runtime fingerprint，防止已安装的漂移环境被错误复用。
- suite revision resolver 优先使用 `git rev-parse HEAD`；无 `git` runtime 中仅接受 40 位 detached `.git/HEAD` fallback，stager 随后仍将 manifest revision 与请求 commit 严格比对。

### 必须产出的指标与统计

- 每任务 accuracy、macro mean、按样本 weighted mean。
- Receiver wrong→Fused correct 与 Receiver correct→Fused wrong 的计数和条件率。
- 1-to-1/one-to-many、candidate count、entropy、boundary mismatch 分桶结果。
- 每例 `candidate_count`、`alignment_entropy`、`boundary_mismatch`、`confidence`、fallback 等 diagnostics 与 prediction CSV。
- K/V gate 按 layer/head/token 的均值、方差、饱和率，以及 early/middle/late layer 的 K/V 分工。
- 完整三任务评测只记录轻量的 per-example K/V gate 汇总；详细 layer/head/relative-token 轴统计在 checkpoint 后以 batch size 1 的固定小样本 post-hoc 诊断在线聚合，不保存 raw gate tensor。
- confidence 与正/负迁移的相关性。
- 每 seed 结果、三 seed mean±std、paired bootstrap 95% CI、McNemar，以及跨 seeds/pairs 的聚合配对比较。
- 组件比较以 B3−B2、B4−B3、B5−B2-constant、B6−B4、B6−B5、B6−B6-constant、B6−B6-shuffle 为主；B5−B2 标记为 static-scale-confounded 次要结果。

### 当前结论

实验运行中，尚未形成任何组件贡献或机制结论。只有 B6 稳定优于 hard-span 与 gate-only control，且预注册的聚合 paired 95% CI 不跨 0，才考虑下一阶段。train/eval loss 只能诊断优化过程，不能判定方法优劣；现有 learned-affine 已显示 eval loss 更低但下游明显更差，因此第一阶段禁止依据 eval loss 得出机制结论。
