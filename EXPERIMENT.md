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
- 为避免修改仍被 x8/x48 Pending Jobs 核验的 canonical checkout，节点级 launcher 支持 `C2C_PHASE15_WORKSPACE_ROOT`；x4 resume 使用独立的精确 commit checkout，manifest、checkpoint 与结果路径仍指向同一份共享 `local/` 资产。
- 预取器只用于尚未开始的 shard 1；MMLU 仍保持原双卡 subject 分片。按已观察首个 triplet 约 31.6 分钟与历史子任务耗时估算，预取可把 x4 关键路径缩短约 28%。x48 完成 shard 6 与 anomaly 后，还会按已完成 summary 对 x4/x8 未启动尾部做无重叠再平衡。
- 调度改动验证为预取器聚焦 `15 passed`、全量 `229 passed`，保留 2 个既有 Pydantic warnings；未改变任何 Phase 1.5 方法、checkpoint 或统计比较。

### 2026-07-19 Phase 1.5 x8 失联恢复与机会式再均衡

- x8 节点失联后，主 manifest 的 shards 2–5 严格审计为 84/120 个 dataset outputs 完整、3 个 MMLU provenance-only partial、33 个完全未启动；即 14 个 incomplete triplets、36 个待恢复 dataset eval。shards 0/1/6 与 Qwen2.5 seed 44 anomaly 已完成。
- 旧 x8 Job 基于未实现 run lock 的 `2b0d6a2`；只有在 Kubernetes 已将该 Job 标为 `FailureTarget`/failed、旧 worker 不再可能写结果后，才允许 x4/x48 接管。恢复继续使用同一 manifest SHA `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1` 和原绝对输出路径。
- 新增 `run-shard-opportunistic`：逐 run 非阻塞获取 NFS lock，锁忙立即后移；取得锁后重新检查 triplet 完整性。shards 2–4 先 resume 三个 MMLU-only partial 和一个 full triplet，随后 x4/x48 可同时指向 shard 5，动态分担其 10 个 full triplets。
- 节点级 `route1_phase15_jobs.py run-shard-opportunistic` 复用 `nvidia-smi` 显存过滤、双 UUID 选择、`CUDA_VISIBLE_DEVICES` 隔离、manifest SHA 校验、NFS 目录预创建和原子状态记录；不同节点必须使用不同 `--state-dir`，实验输出仍由共同 per-run lock 协调。
- evaluator 非零或成功返回但 ARC/OpenBookQA/MMLU-Redux 未全部满足唯一 CSV、summary 与 provenance 契约时立即失败；整轮锁忙时按 `(0,60]` 秒有界轮询。原 `run-shard` 与 `run-node` 默认行为不变。
- Phase 1.5 调度相关测试 `30 passed`；项目全量测试 `236 passed`、保留 2 个既有 Pydantic warnings。该恢复只改变固定矩阵的执行位置与吞吐，不改变方法、checkpoint、逐例预测定义或统计协议。

运行时更正：当前共享 NFS 上的 advisory `flock` 实测不能提供跨 Pod 互斥。两个 shard-5 opportunistic workers 对 5 个 dataset 产生重复 bundle，因此该模式不再用于本轮尾部，也不作为当前集群的安全恢复合同。重复 CSV 除 `answer_latency_ms` 外逐单元格一致，预测、正确性、ambiguity 与 gate 完全一致；gate/length 原始 SHA 一致，summary 仅引用的时间戳 artifact 名不同。20 个 ghost-writer 文件已成组移入 `local/tmp/phase1_5_causal_diagnostics/duplicate_quarantine_20260719/`，保留取证但不进入统计。剩余 run 改为一 run 一 Job 的显式不重叠分配，并用固定 GPU UUID、manifest SHA、run index/id 与空输出 init 校验防止再次竞态。

### 2026-07-19 Phase 1.5 最终因果结果与放行判决

- 最终严格审计通过 74/74 runs、222/222 dataset outputs：主矩阵 96、x8 隔离根 120、Qwen2.5 seed44 anomaly 6；每个 dataset 恰好一份 CSV/summary/provenance/gate/length，ARC/OBQA/MMLU 行数分别为 1,150/500/5,615，sample key、JSON、checkpoint/intervention、内部 provenance SHA 与当前 config SHA 全部一致。主/anomaly manifest SHA 分别为 `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1` 与 `bd305268e9a8527cb75407293b49cae4e577bb10516e9643781573e861cfa5d2`。
- 第一次最终审计发现 TinyLlama B2 eval-k4 seed43 三任务来自较早 manifest，当前 YAML 字节 SHA 无法重现。旧 bundle 完整移入 `local/tmp/phase1_5_causal_diagnostics/provenance_quarantine_20260719/` 后按最终 config 显式重跑；新旧除 latency、时间戳与 evaluator checkout 路径外所有逐例科学字段完全一致，gate/length bitwise identical，因此修复 provenance 未改变结果。
- 统计使用 5,000 次 pair→seed→paired-example hierarchical bootstrap、95% CI、seed `20260718`。Kubernetes 正式运行与本地 Conda 独立复核的 paired/oracle/ambiguity CSV 字节级一致；其余差异仅为 sample std 浮点序列化，最大绝对差 `4.337e-19`。正式输出位于 `local/final_results/phase1_5_causal_diagnostics/rev_0d30852/analysis/`，不提交 Git。
- 同 checkpoint top-k：B2 eval-k4−k1 `−0.01 pp`、CI `[−0.13,+0.11]`，1/3 异构 pair 为正；B3 `+0.03 pp`、CI `[−0.06,+0.14]`，3/3 为正但实际量级不足 0.1 pp。两个对照均无可靠正向 ambiguity concentration，因此未识别到推理期多 candidate 的平均因果收益。
- B3-trained−B2-trained 在 eval-k1/k4 下仍为 `+1.25/+1.30 pp`，但 CI `[−0.64,+3.56]` / `[−0.67,+3.63]` 均跨 0、seed std 约 2.4 pp。差异最大的是同 tokenizer Qwen3（+3.12 pp）与 TinyLlama（约 +2.02 pp），Qwen2.5/Llama3.2 约为 0；只能解释为 checkpoint/training-regime 且 pair-dependent，不能单独归因于训练期 k4、tokenizer 身份或随机轨迹。
- Entropy：native−constant `+0.13 pp`、CI `[−0.13,+0.40]`；native−shuffled `+0.04 pp`、CI `[−0.07,+0.22]`，均仅 1/3 异构 pair 为正且无可靠 ambiguity 集中。预注册的 TinyLlama constant/shuffle seeds43/44 条件不成立，不补重训；删除 entropy-aware 的稳定机制主张。
- Gate：learned−static `−0.01 pp`、CI `[−0.10,+0.07]`，不支持 learned token/head modulation；learned−forced-on `−0.21 pp`、CI `[−0.98,+0.62]` 且方向随 pair 翻转。后者同时改变 legacy scalar K/V masks，不能作为纯 token/head 对照，也不能声称 forced-on 普遍更优。
- Qwen2.5 seed44：alignment-confidence forced-on `+0.00 pp`、0 correctness flips；legacy scalar forced-on `+1.91 pp`、CI `[+1.31,+2.53]`，336 改善/197 回退。legacy hard masks 是该 checkpoint 崩塌的部分因果来源，但只恢复约 26% gap，不是通用解法。
- B6 native oracle abstention headroom 为 `+8.24 pp`、CI `[+6.28,+10.19]`，4/4 pairs 为正。该值是 label-aware 上界，统计文件不含 selector AUROC、校准或 selective-risk 结果，不能推断现有 gate 能实现该收益；只支持后续优先审计 calibrated null/no-transfer。
- Ambiguity interaction 只作探索性敏感性分析：pooled q75 在多个 pair 上混入 MMLU 与 ARC/OBQA task 差异，absolute 定义又存在 TinyLlama 全 high、其他 pair 稀疏的问题。因此只用“没有可靠正向集中”判定放行失败，不把负 interaction 包装成普适反机制。
- Query-time prototype 放行失败：B2 正向异构 pair 数不足；B3 虽 3/3 为正，但跨 pair CI 下界 `−0.06 pp`，且无可靠高 ambiguity concentration。本阶段停止，不进入 query-time transport；根目录报告为 `PHASE1_5_CAUSAL_DIAGNOSTICS_REPORT.md` 与 `PHASE1_5_CAUSAL_DIAGNOSTICS_SUMMARY_ZH.md`。

### 2026-07-19 Phase 2A-0 calibrated null/no-transfer 零 GPU opportunity audit

- 基线：Cache `main` commit `29a96947fc5d5e0a8f457f75edf7f2e745932bce`；只读取 Phase 1 artifact commit `9b06d173eada148343ddfb71a31721c7ae5f7ad5` 的 receiver-only 与 B6-native 逐例 CSV。
- 约束：未启动 GPU/Kubernetes，未训练 adapter/router，未修改 B6 checkpoint，未进入 instrumentation rerun 或 selector 训练。
- 输入合同：receiver-only 3 文件/7,265 行；B6-native 36 文件/87,180 重复观测；39 文件 schema、sample keys、输入内容与标签完全匹配。7,265 rows 对应 7,233 normalized-content groups，MMLU 内 32 个重复内容组后续必须绑定 split。
- 四事件 sample-weighted：both-correct 27.9158%，beneficial 19.0158%，harmful 8.2439%，both-wrong 44.8245%。Receiver 36.1597%，fused 46.9316%，oracle 55.1755%。
- Pair-balanced sample-weighted oracle-over-best-fixed `+8.2439 pp`，10,000-draw best-fixed-aware hierarchical paired bootstrap 95% CI `[+6.2318,+10.2019] pp`；task-macro 为 `+8.5352 pp`、CI `[+6.3438,+10.7561] pp`。
- B6-native 在全部 36/36 pair×seed×task 单元中优于 receiver，故 Phase 1.5 的 `+8.24 pp` 点估计确实也是相对 retrospective best fixed 的真实空间，而非仅相对 fused；旧正式 CI 技术上仍是 fused-only，新统计在每个 draw 内重算 `max(receiver,fused)`。
- 统计脚本自动校验 Phase 1 suite manifest、Phase 1.5 execution manifest 与旧 oracle CSV 的 SHA，并解析旧 `b6_native` across-pair 行；旧点估计与新 best-fixed point 必须一致。
- 字段审计：真正 A 类现有特征为 input length 与 alignment summaries；gate/fused output 为 B 类，receiver full-forward output 为 C 类，label/correctness/event/identity 为 D 类。`entropy == one_to_many_rate`、`confidence == 1-0.5×entropy`，`fallback_rate` 恒为 0，说明可部署特征有效维度有限且强烈 pair-coded。
- 预注册：后续 primary comparator 固定为 calibration-selected global best fixed policy；primary aggregation 为 pair-balanced task-macro；fit/calibration/model-selection/test 采用 30/15/15/40 的 content-group hash split，并冻结 leave-one-seed/task/pair-out 与六项 conjunctive GO 条件。
- 可复现命令：`python script/analysis/phase2a_0_opportunity_audit.py --manifest recipe/eval_recipe/phase2a_0/opportunity_audit_manifest.json --output-csv PHASE2A_0_OPPORTUNITY_AGGREGATES.csv --output-json PHASE2A_0_OPPORTUNITY_AGGREGATES.json`。
- 交付：`PHASE2A_0_OPPORTUNITY_AUDIT.md`、`PHASE2A_PREREGISTRATION.md`、小型 aggregate CSV/JSON、manifest、统计脚本与定向测试。大体积逐例文件继续只保留在 `/netdisk/.../local/`。
- 验证：定向测试 `3 passed`；项目全量 `239 passed, 2 warnings`；正式 10,000-draw CSV 确定性重跑 byte-identical，JSON 除输出路径外 scientific content identical；`py_compile` 与 `git diff --check` 通过。
- 结论：opportunity 存在且跨 pair/seed/task 稳定，但仍是 label-aware oracle 上界，不证明当前已有可实现 selector。Phase 2A-0 到此停止，等待审查。

### 2026-07-20 FPCT-GPU-R2 零输出根因诊断与 CPU/HF recovery

- Prospective protocol commit：`f7a5f3c421a7738c9f69224cff1cebb53205c2e2`，发生在任何新 tokenizer、自然 prompt、pretrained forward、GPU、训练或 accuracy 输出之前。
- 旧 immutable execution：scientific SHA `371e72f14da41f5509eafa21553c7a7418c9a53e`，image digest `sha256:c851056733f3b7affc85ae5dabd870043f3ae7d3010d245705f5b9ded8dc36ab`，run-lock SHA256 `2a4db8f26def997c95b590a34718916b772f686f5c00eabb2f2b69f0dfe5e5ec`；不 patch、不 resume。
- Zero-output probe：projector checkpoint 为空，fresh 28 层 key/value gate logits 全部精确为 0；checkpoint-native `(logit>0)` 选择 native，因此旧 activation=0 分类为 `EXPECTED_NATIVE_NULL`。旧 config 未显式 eager，只允许称“按当前 dispatch 规则可能使用 SDPA”。
- Local-only probe：`local/final_results/fpct_factorized_transport/fpct_gpu_r2/rev_f7a5f3c421a7738c9f69224cff1cebb53205c2e2/zero_output_provenance.json`；firewall 记录 tokenizer/natural prompt/model forward/accuracy 均为 false。
- Recovery config：canonical FP32 A/logA/mask/C_post reduction/softmax；C_post/F shared eager adapter；FP32/BF16 × 8 isolated operator conditions；18-row label-free panel；metric-specific null floors；P0–P6 scoped profiler；四步 seed 104729 matched integrity。
- 当前验证：R2 扩展定向集 `53 passed`；首次 repo full suite 使用 repo 外 `/tmp` basetemp 时为 `396 passed, 2 path-contract failures, 2 warnings`，随后在 `local/tmp` 按仓库约定重跑为 `401 passed, 2 warnings`。
- 新 two-lock：scientific SHA `9f2ffcd9ff21e4575f8fe870167eb04a7c86edb5`；image digest `sha256:d04455bf67177792548c3add74214f23ce097a004131481624886631725817ef`；operative run-lock SHA `c4b0ca20bea54f2dbbb9eaabf5bdbb0dc5b74835a3284e5d846b46f1e6a2a331`；run root `fpct-r2-9f2ffcd9-v1`。
- 第一次 gate Pod 仅在 image resolve 阶段 `ImagePullBackOff`，容器未启动、无 output；pending Job删除后，loader增加 `repository@digest` alias并重新冻结lock。
- Lock时仍没有新 pretrained output、GPU/K8s、训练、checkpoint 或 accuracy；下一步仅为 image import + complete synthetic gate。
- R2 v1 complete numerical gate=`GO`；artifact SHA：sequence `9cf75832...`、numerical `588c5c42...`、floors `f0d01226...`、runtime `69101e4d...`。
- 首个 FP32 C_post condition在第一 model forward前因scalar state hash失败；已加载label-free panel/tokenizer/weights，但0 forward/0 accuracy/0 training/checkpoint。v1 controller terminal，不resume。
- v2 prospective repair仅修正0-D tensor hashing并新增回归；targeted `25 passed`、full `402 passed, 2 warnings`。新 run必须从image/GPU gate重启。
- R2b two-lock：SHA `7ceae185...`；image digest `sha256:d035cb31...`；run UID `fpct-r2b-7ceae185-v1`；run-lock SHA `99dcb811...`。不复用v1 numerical/condition artifacts。
- R2b numerical=`GO`，但首个FP32 C_post trace forward在第一层attention因`packed`未初始化失败；0 complete output/0 condition artifact/0 accuracy。R2b terminal，不resume。
- R2c修复`packed=None`并新增actual Qwen C_post trace regression；targeted `28 passed`、full `403 passed, 2 warnings`。
- R2c two-lock：scientific SHA `e1133549...`；image digest `sha256:94437d56...`；run-lock `3ea3c3ea...`；run UID `fpct-r2c-e1133549-v1`；image tar SHA `aeda9aab...`。新run从complete GPU numerical gate开始，不复用R2/R2b artifacts，lock前无pretrained output或accuracy。
- R2c complete numerical=`GO`，16 conditions+5 profiles完成；forced-on activation与resource gates通过，但precollapse/bypass/replicated/m1/hot-sync硬门失败，terminal `GPU_ENGINEERING_BLOCKED_R2`。未进入matched smoke、training、checkpoint或accuracy；R2c不resume。
- R2d prospective repair：shared C_post/F sidecar adapter、m≤1 exact parent path、replicated expanded-local canary+analytic parent return、non-persistent device scale buffers；targeted `53 passed`、full `406 passed, 2 warnings`。尚无R2d pretrained/GPU/training/accuracy output。
- R2d two-lock：scientific SHA `71ba96d...`；image `sha256:04b7b642...`；run-lock `2e1c998f...`；run UID `fpct-r2d-71ba96d-v1`；全新root/tar/sidecar copy，必须从complete GPU numerical gate开始。
- R2d pretrained terminal：FP32 exact controls恢复；BF16 pre-sidecar adapter分叉、expanded output canary `0.0625`、residual-scale hot sync 280导致BLOCKED。0 matched smoke/training/checkpoint/accuracy；R2d不resume。
- R2e prospective repair：pre-sidecar shared adapter、tensor-only exact-replicated group collapse、FP32 grouped-probability canary、residual-scale device-native constants；targeted `53 passed`、full `406 passed, 2 warnings`。尚无R2e GPU/pretrained/training/accuracy output。
- R2e two-lock：scientific SHA `2653930...`；image `sha256:50b89faa...`；run-lock `e4d4392f...`；run UID `fpct-r2e-2653930-v1`。新run只可从complete GPU numerical gate开始。
- R2e-v1 loader因纯数字`git_sha`未quoted在Job创建前被client拒绝，0 container/output；v2 run UID `fpct-r2e-2653930-v2`、run-lock `05c100a7...`，scientific image与阈值不变。
- R2e-v2 terminal：exact controls与hot-sync恢复，但flat expanded kernel造成native-null FP32 `4.12e-5`/BF16 `0.625`；inactive diagnostic atoms抬高D_K/D_V floors。未训练、未读accuracy。
- R2f prospective hierarchical adapter：β/γ global-equivalent、exact parent branch、inactive diagnostics修正；targeted `55 passed`、full `408 passed, 2 warnings`。尚无R2f GPU/pretrained/training/accuracy output。
