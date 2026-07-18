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
- 所有方法固定 epoch、per-device batch、有效全局 batch、学习率、fuser 架构、数据 split/order 与 final-checkpoint 选择规则，禁止单独调参。Canonical 四卡配置为 `1×4×8=32`，两卡基础设施 profile 为 `1×2×16=32`；world size 与 accumulation 的等效转换必须写入 provenance，不宣称不同 world size 的训练轨迹 bitwise 相同。
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

### 三条异构卡数跨节点流水线

- Phase1 共 37 runs：lane A/B/C 分别为 12/13/12。
- Conditional 跨模型多 seed 阶段共 30 runs：每条 lane 10 个；只有 seed 42 的跨模型方向满足门控后才释放。
- Lane A：`4090-24gx4`，4×24GB，4-process DDP、gradient accumulation 8；Llama3.2 pair 固定 affinity 到该 lane。
- Lane B：`4090-24gx8`，2×24GB，2-process DDP、gradient accumulation 16。启动前要求两张可见卡的已用显存均不超过 4,096 MiB，以避开节点上 Kubernetes 未记录的高占用 GPU。
- Lane C：`4090-48gx2`，2×48GB，2-process DDP、gradient accumulation 16。
- 三条 lane 的 per-device batch 均为 1、有效全局 batch 均为 32。两卡评测先并发运行 ARC `[0]` 与 OpenBookQA `[1]`，再运行 MMLU-Redux `[0,1]`。
- 每条 lane 内按 run 串行执行 train→ARC/OpenBookQA/MMLU-Redux，并用状态文件、依赖和 gate 防止越级运行；不同 lane 可并行。
- 三条 lane 的 `C2C_MODEL_ROOT` 与 `C2C_DATA_ROOT` 均指向同一份 `/netdisk` 资产，不再按节点使用不同的 local-first 路径。
- 当前 Job：Lane A `route1-id-v22-9b06d173-lane-a-e1e95b27`；Lane B `r1id-v22-9b06-lane-b-2gpu-24g-1a7dd1d2`；Lane C `r1id-v22-9b06-lane-c-2gpu-48g-1a7dd1d2-cache`。

### 2026-07-17 实际启动记录

- Stager `route1-id-v22-9b06d173-stager-69d7dbd9` 已完成，67-run manifest、共享模型/数据树哈希和固定 Python package 版本审计全部通过。
- Canonical B6 seed 42 checkpoint 目录 SHA256 为 `a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`，Lane A 复用该权重，仅重新评测。
- Lane B 的第一次四卡执行在 B0 三任务评测完成后遭遇共享 NFS `completed/` 目录创建竞态；预创建状态目录后重试可复用 B0 完整产物，不重复评测。
- Lane B 的第二次四卡执行被分配到含隐藏高占用卡的 GPU 集合，rank 2 在模型 `.to(device)` 时 OOM。该节点标准 Pod 中观测到一张卡约占用 21,992 MiB，但 Kubernetes 没有对应 GPU request，因此后续改为两卡准入检查，不对该卡执行未授权 reset。
- 两卡 adapter SHA256 为 `1a7dd1d25dc4ac9cf208676a6403e0c4938a47e2bd2b21cb7de3a7d6b6f9d6bb`。每个适配后的 train config 都重新计算 `train_config_sha256` 并写入 checkpoint provenance。
- Lane B 两张启动卡均为 1 MiB，已跳过完成的 B0 并进入 TinyLlama B2 seed 42 的 64-step 训练。
- Lane C 首次拉取约 3.3GB 固定 PyTorch runtime 镜像后，暴露出该节点与其他节点不同的 hostPath 权限：共享 `/netdisk` 需要 supplemental GID 31000，节点本地 `/cache/huggingface` 又由 root 创建且不可写。最终 Job 显式加入 GID 31000，并把只用于 datasets 临时索引的 cache 改成 Pod `emptyDir`；模型、数据和 checkpoint 仍只读取已审计的 `/netdisk` 资产。
- 修复后的 Lane C 两张启动卡均为 1 MiB，已进入 TinyLlama B1 seed 42 的 64-step 两进程训练；运行时显存约 10.7/11.7 GiB，未观察到 OOM 或重启。
- 两卡输入 Job manifests 已保存到 `/netdisk/lijunsi/c2c-route1-identifiability/status/job-manifests/`；Lane B/C manifest SHA256 分别为 `299868b8aca0ab41986a2262ea67ba6b08a716244ef9bb21a852336493e0e143` 与 `6f168f648c84979aeec527d6f53a7792da601e67f4f277f402960f9b31e02746`。

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

### 2026-07-17 Phase1 二次加速记录

- 旧 Lane A 已完成 TinyLlama B3 seed 44，并在 reproduction gate 临时设为 pending 后于 run 边界退出；恢复 Job `r1id-v22-9b06-lane-a-resume4` 已跳过三个完整 run，进入 B6 seed 44。
- 旧 Lane B/C 分别保留正在执行的 B2 seed 42 与 B1 seed 42；两者完成全部训练、评测和 marker 后退出，不进入后续 run。
- 除上述两个 reserved runs 与已有 B0 外，B/C 剩余 22 runs 被确定性拆成四个互斥 shard，plan SHA256 分别为 `cc45e28522a643e82e5fddad063f364e371897b1ef01d544c7bcd6c7b592b75b`、`bd0e432d29284a775a05a220303b500d24ead4f0c45f5b7325c142b4315b8b33`、`510ff6356038c7c79381aec6106f62f734210b16b428c8cd724de970b715ad68`、`aa7c98505f270d69e7462278dbe6fcf4c9e877d8b5742229cd9c46e80b4de37b`。
- Shards 1–3 使用 `4090-24gx8` 各 2 卡，当前 Jobs 为 `r1id-v22-bc4-s1-x8-r2-933bc186`、`r1id-v22-bc4-s2-x8-r2-933bc186`、`r1id-v22-bc4-s3-x8-r2-933bc186`；三者均已通过启动显存检查并进入训练。
- Shard 4 Job `r1id-v22-bc4-s4-x48-r2-933bc186` 已创建，在旧 C worker 释放 `4090-48gx2` 后自动调度。
- Adapter SHA256 更新为 `933bc1868f319e718ae30bcc22f37211b43d61d42813af7a620914fcd9aed3e9`；实际选择的 GPU UUID 与启动显存写入每个 adapted plan 的 `.allocation.json`。
- 两个短暂的 3-card-reserve shard 仅运行数分钟即被主动删除并重新均衡，没有产生 completion marker；对应 partial checkpoint 不满足 reuse 条件，正式四 shard 计划会从头训练相关 run。
- 长期并行布局为 A 的一条 4 卡 lane、`24gx8` 的三条 2 卡 shard、`48gx2` 的一条 2 卡 shard，共五条。Phase1 新 ETA 为约 7–10 小时；若 seed-42 跨模型方向通过并立即以相同方式释放 conditional 30 runs，全部 67 runs 预计还需约 15–20 小时。

### 2026-07-17 Max7 终态调度

- 五线布局进一步被七 worker 终态替代。当前正在运行的 6 个 runs 被标记为 reserved，完成后退出：TinyLlama B6 seed 44、B2 seed 42、B1 seed 42、B3 seed 42、B2-constant seed 43、B5 seed 42。
- 其余 27 runs 重新分成 7 个互斥 plans，SHA256 为 `3bdd060601f30c502f0dc3f291253c605d572b24be535ab37aa57133a989d1a2`、`4037cdc5908f166d8c6ffc36338e9c231522ec8e7825384b5a9628e167ca17bb`、`c67ab3a17f5fbd60425285fbf3958d3d0aa681acb3d0dcf4ab269b7cd1d8024d`、`38a7082d394788bb070e5f1c2c3154a730b5501fad64bb20cdd09f2902c69b9d`、`a62a8a82165de5a075572b7771b79bb215af03eff5a8af084319f96f21c89512`、`bba0d70fb86531c8cc9a53378bec2f90cd2e50566fd3ed56f182a037df157fd7`、`a029475b19093fbb222e22db3632f94486c061e89bff3a1f587c1bb0484f550c`。
- Max7 Jobs 为 `r1id-v22-max7-s1-x4-r2-933bc186`、`s2-x4`、`s3`–`s6-x8`、`s7-x48`；均已创建并使用独立 pass gate，在旧 worker 释放资源后自动调度。
- 终态资源布局：`4090-24gx4` 两条 2 卡 worker，`4090-24gx8` 四条 2 卡 worker，`4090-48gx2` 一条 2 卡 worker，14 张 NVIDIA GPU 全部进入调度合同。
- Max7 的 shard manifest、gate 与七份 Job 输入清单保存于 `/netdisk/lijunsi/c2c-route1-identifiability/status/job-manifests/max7-phase1/`，目录内文件均记录 SHA256。
- Phase1 ETA 更新为约 6–7 小时，预计 2026-07-18 05:00–06:00 CST 左右完成；若 conditional 阶段立即按相同布局释放，全部 67 runs 预计约 12–16 小时完成。

### 2026-07-18 Phase1 完成与 conditional 放行

- Phase1 的 37 个计划 runs 已全部完成，实验失败为 0；七个 Max7 Jobs 全部 Kubernetes Complete，wall time 为 5 小时至 7 小时 8 分。旧过渡 Jobs 的 Kubernetes Failed 来自 pending gate 在 run 边界主动返回非零，不代表训练或评测失败。
- Phase1 报告保存于 `/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/route1_identifiability/rev_9b06d173eada/phase1_report/`。TinyLlama B6 seed 42 macro mean 为 50.806%，与预期 50.82% 接近。
- seed 42 sample-weighted 结果满足 conditional 筛选：B6−B2 在 TinyLlama、Qwen3-1.7B、Qwen2.5-0.5B、Llama3.2 四个 pair 全部为正；B6−B5 在前三个 pair 为正，Llama3.2 为 −0.509 percentage points。该筛选只决定是否补 seed，不替代最终三 seed paired CI。
- TinyLlama 三 seed 的 provisional component contrasts：B3−B2 `+2.09 pp`、B4−B3 `+1.29 pp`、B5−B2-constant `+0.93 pp`，三者 cluster CI 均跨 0；B6−B4 `+1.34 pp` 与 B6−B5 `+2.50 pp` 的 cluster CI 均高于 0。B6−constant 为 `+0.94 pp`，B6−shuffle 为 `+2.28 pp`，当前反事实支持 entropy 数值和位置含有信息，但仍需结合跨模型多 seed 与 gate 饱和诊断解释。
- 30 个 conditional runs 已拆为七路，plan SHA256 为 `2f9d6cb6c0d7943d6aa873887484a5418fda635977ce74d89e1b1b0cf697ce1a`、`c9396dbe5d7c1f8c3f651b4af5185b64ef171abf8d48b8c8f5828a821bc5b2e7`、`1f5cf0b60fdbca591ae2974399585d61c2eba6cb9ec074d75731c9584bc2bf74`、`ec4d8d2df69a28f5bb8cb2523bc9fe890269084098e2ac5d1c3bc80a6f0c8e7c`、`74d2b9537023888f1180bf7412f366475a3588b7efafab523323ffd430b2ccf4`、`d0203899937fd660b766818690c58c735d25029a4ffc7837134c21b870b828a0`、`91637d1043ffa7ba28189baa383f84609ce6a65c5bfec0883fb5bfa92f4c1a29`。
- 正式 Jobs 为 `r1id-v22-c7-s1-x4-r4-e796f4df`、`s2-x4`、`s3`–`s6-x8`、`s7-x48`；七个 Pods 均已进入训练、restart 0，共占用 14 张 NVIDIA GPU。第一批 r3 Jobs 仅因 adapter 不接受 conditional phase 而在训练前退出，修复后 adapter SHA256 为 `e796f4df99e362cfd83e5510b92955a06fac0b08bcd2aace1921ceb7607b9416`。
- 按 Phase1 实测吞吐与 conditional 分片权重，预计 conditional 阶段约需 7–10 小时；从 2026-07-18 08:10 CST 启动计，预计在 15:00–18:00 CST 左右完成，之后生成完整 67-run 统计报告。

### 2026-07-18 最终结果：identifiability gate 未通过

- Conditional 七路在 14:46 CST 全部结束；67/67 runs 完成、failed marker 0，最终严格 materialized manifest 为 234 rows，`conditional_complete=true`。
- 最终报告目录：`/netdisk/lijunsi/c2c-route1-identifiability/workspace/Cache/local/final_results/route1_identifiability/rev_9b06d173eada/final_report/`；主文件为 `report.md`、`summary.json` 和 `MECHANISM_SUMMARY_ZH.md`。
- 三 seed sample-weighted B6：TinyLlama `47.13±1.24`、Qwen3-1.7B `50.27±0.74`、Qwen2.5-0.5B `43.62±4.19`、Llama3.2 `46.69±0.97`。
- B3−B2 跨 pair 为 `+1.29 pp`，CI `[-0.61,+3.58]`；只有 Qwen3-1.7B pair 为 `+3.12 pp`、CI `[+0.47,+6.59]`。该 pair 的 1-to-1 与 one-to-many gains 分别约 `+3.10/+4.49 pp`，收益并非只存在于 tokenizer ambiguity bucket。
- TinyLlama B4−B3 为 `+1.29 pp`、CI `[-2.78,+5.98]`，不能证明 static entropy 独立贡献；但 B6−constant `+0.94 pp`、CI `[+0.11,+1.75]`，B6−shuffle `+2.28 pp`、CI `[+1.39,+3.18]`，支持该 pair 中 entropy 数值与位置对应关系包含信息。
- Clean gate capacity B5−B2-constant 为 `+0.93 pp`、CI `[-2.66,+3.90]`；B5−B2 跨 pair 为 `+0.35 pp`、CI `[-1.15,+2.13]`，均不支持 gate capacity 的稳定独立贡献。
- B6 post-hoc gate 在三个跨模型 pair 上几乎始终开启：Qwen3/Qwen2.5 high-saturation 约 99.93%，Llama3.2 为 100%；TinyLlama 约 85.86%。B5 的 key gate 比 value gate 更动态，但未带来稳定下游提升。
- Final gate：B6−B2 delta `+1.54 pp`、pair-cluster CI `[-1.14,+4.05]`，3/4 pairs 为正；B6−B5 delta `+1.19 pp`、CI `[-0.92,+3.31]`，2/4 pairs 为正。两个预注册条件均失败，第一阶段在此停止，不进入下一阶段。
- 为便于 GitHub 直接查看，最终完整报告与中文机制总结同时发布到仓库根目录：[`ROUTE1_V22_IDENTIFIABILITY_REPORT.md`](ROUTE1_V22_IDENTIFIABILITY_REPORT.md) 和 [`ROUTE1_V22_IDENTIFIABILITY_SUMMARY_ZH.md`](ROUTE1_V22_IDENTIFIABILITY_SUMMARY_ZH.md)。逐例 prediction、CSV 与约 13MB 的 `summary.json` 继续保留在 `local/`，不提交仓库。

### 2026-07-18 Phase 1.5 因果诊断预注册与启动前审计

Phase 1.5 基于 main `0d308525860d27897bde6d558798e468cf113281` 的 Phase 1 完整产物继续执行，不开发 query-time transport、router、replay、OT、RoPE、新 gate 或新 loss。

产物审计：

- 67 个 completion markers、201 个 prediction CSV、26 组 post-hoc gate diagnostics 均存在；实验失败 marker 为 0。
- 65 个新训练 checkpoint 加 TinyLlama B6 seed 42 的 bitwise-verified 复用 checkpoint，共 66 个必需 checkpoint 均可完整加载 28 层 projector，tensor 全部有限。
- 65 个本地 checkpoint 的 run id、执行 commit、训练配置 SHA、split 与数据 hash 均与 provenance 一致；复用 checkpoint 目录 SHA256 为 `a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`。
- 共享根 `/netdisk/lijunsi/c2c-route1-identifiability` 可被三个 NVIDIA 节点访问，空闲容量约 923GB；当前 14 张 NVIDIA GPU 无活动实验 request。

Qwen2.5-0.5B→Qwen3-0.6B B6 seed 44 异常诊断：

- 不是 checkpoint 损坏、NaN/Inf、梯度爆炸或 optimizer 数值崩塌。seed 44 train/eval loss 反而低于另外两个 seeds，所有 7,265 个预测均合法。
- seed 44 与 receiver 预测一致率为 84.1%，明显高于 seed 42/43 的 63.8%/65.5%；正迁移降到 10.3%，负迁移约 10.8%。异常本质是几乎失去正迁移，而非负迁移爆炸。
- alignment-confidence gate 三个 seeds 均约 99.93% high-saturated，不能解释 seed 特异异常。
- checkpoint 内 legacy scalar K/V gate 的 eval 决策是接近零的单标量 logit 硬阈值。seed 44 前 9 层有 7 层 K/V 同时关闭，前 6 层完全无 transfer；微小训练轨迹差异会翻转整层开关。
- 当前最符合证据的解释是 `seed training trajectory × legacy scalar hard mask × model-pair compatibility`。因此额外预注册 Qwen2.5 B6 seed 44 的 alignment-only 与 legacy-only forced-on 推理干预。

同 checkpoint 主矩阵：

- B2 eval-k4；B3 eval-k1，配合 Phase 1 native B2 eval-k1 与 B3 eval-k4 形成 train-k × eval-k 2×2。
- B6 entropy constant-0.93、entropy shuffled、gate static、gate forced-on；Phase 1 native B6 作为共同 comparator。
- 4 个模型对 × 3 seeds × 6 个新干预 = 72 个三任务 triplets；另有 Qwen2.5 seed 44 的 2 个异常拆分 triplets。
- 72 triplets 保持七个逻辑双卡 shards `[11,11,10,10,10,10,10]`。三个整节点 Jobs 请求全部 14 张 Kubernetes GPU，但会过滤 x4/x8 上各一张约 19/22GiB 的外部 busy 卡；实际最多五个双卡 shards 同时评测，其余 shard 在节点内自动排队。

统计协议：

- 使用与 Phase 1 一致的 pair-balanced hierarchical paired bootstrap，报告 accuracy delta、95% CI、McNemar、正向 pair 数、三 seed 方差。
- ambiguity bucket 固定取 native B3/B6 diagnostics，避免干预后 bucket 漂移；同时报告 absolute high ambiguity 与 pair/seed/task 内 composite score top quartile 的 interaction。
- 使用现有 receiver-only 与 fused 逐例结果计算 oracle abstention accuracy、理想 abstain rate 和相对最佳固定策略的 headroom。
- 只有同 checkpoint top-k4 在至少两个真正异构模型对上为正、跨 pair CI 下界大于 0 且收益集中在高 ambiguity 时，才允许进入小型 query-time prototype。

启动前验证：项目全量测试 `223 passed`，三份节点级 Kubernetes Jobs API server dry-run 全部通过。首次真实创建在评测开始前发现 Job 漏传固定 `PIP_CONSTRAINT`，导致准备建立未锁定的新 venv，并在两个节点上触发未发布目录清理竞态；七个 Job 已立即删除，prediction 产物为 0。补齐 constraints 后，第二次启动直接复用 Phase 1 已审计环境，同时暴露两个既有基础设施事实：共享 NFS 上多个 evaluator 递归创建共同结果父目录会发生 `FileExistsError`，且 x4/x8 各有一张 Kubernetes 不可见的高占用 GPU。GPU 调度改为三个整节点池，按实际空闲显存选择 UUID 卡组，并发或排队覆盖全部七个逻辑 shards；x8 对其他节点新建目录名的持续 negative dentry 通过 shards 2–5 的 node-isolated sibling 共享结果根规避，最终 execution manifest 仍统一记录所有逐例路径。统计命令通过 `--anomaly-manifest` 把 Qwen2.5 seed 44 的两个额外 triplets 与主八项对照统一输出，避免手工合并。

### 2026-07-18 Phase 1.5 运行中吞吐审计与单卡预取

- 19:02 CST 的严格输出审计：主矩阵完成 19/216 个 task-level outputs、3/72 个完整 triplets；所有已完成 ARC 1,150 行、OpenBookQA 500 行、MMLU-Redux 5,615 行，未发现重复 CSV/summary、坏 JSON、错误行数或主/x8 双写。24 个已出现的 provenance 均记录正确 intervention、manifest/config hash 与 `training_state_mutated=false`。
- 三个活跃 Pods 均 restart 0、无 OOM/Traceback：x4 运行 shard 0 并在其后串行 shard 1，x8 以三个双卡组并行 shards 2/3/4 后接 shard 5，x48 运行 shard 6。新的 x48 resume Job 与 Qwen2.5 seed44 anomaly Job 因该节点 2/2 GPU 已占用而 Pending，会在 shard 6 释放后自动接续。
- GPU 实测显示每个 evaluator 子进程都在单卡完整加载模型：TinyLlama pair 约 5.8–6.1GiB，Qwen3-1.7B pair 约 6.6–7.0GiB。x4 的第三张 24GB 卡仅占约 396MiB，具备预取 ARC/OpenBookQA 的显存余量；第四张卡由 Kubernetes 外部进程占约 19.2GiB，不参与实验。
- 新增受控 `stage-small-benchmarks`：不改 YAML，只用 CUDA UUID mask 把原 ARC `[0]` 与 OpenBookQA `[1]` 映射到 spare GPU；新版主 lane 与 stager 通过 per-run lock 互斥，状态 JSON 原子记录，MMLU 已开始或小任务存在 partial artifacts 时不重复启动。
- 预取器只用于尚未开始的 shard 1；MMLU 仍保持原双卡 subject 分片。按已观察首个 triplet 约 31.6 分钟与历史子任务耗时估算，预取可把 x4 关键路径缩短约 28%。x48 完成 shard 6 与 anomaly 后，还会按已完成 summary 对 x4/x8 未启动尾部做无重叠再平衡。
- 调度改动验证为聚焦 `15 passed`、全量 `228 passed`，保留 2 个既有 Pydantic warnings；未改变任何 Phase 1.5 方法、checkpoint 或统计比较。
