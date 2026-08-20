# EXPERIMENT.md

## 2026-08-21 FPCT-E1-FAST-RCA fixed-checkpoint result

- Execution/root：`07b2a3f5...` /
  `/netdisk/lijunsi/fpct-e1/fpct-e1-fast-rca-07b2a3f5-v1`。
- Job：`fpct-e1-fast-formal-07b2a3f5-20260820-233130-701513`；node=
  `4090-24gx4`；restart=`0`；Succeeded。
- Coverage：18/18 shards；17,604 compact rows；1,956 matched group cells；expanded rows=`0`。
- Verifier：`GO_ALL_SHARDS_COMPLETE`；formal inventory SHA256=
  `f12d8b7da233f437916b98a9e796c68c30ca9531d5698ce3eed281d0ee1eb275`。
- Root-cause result SHA256=
  `4467bf4b8c6966c838f66e2f4548b4e4650717a8f2e41bae9380281a40789c1f`。
- Preregistered decision=`NO_EXPLOITABLE_FIXED_CHECKPOINT_HEADROOM`；all selectable
  interventions ineligible；winner/root-cause=`null`；training authorization=`false`。
- Original F-C_post task-macro `delta logp(y*)=-0.00011850`；ARC/OBQA/MMLU=
  `-0.00010409/-0.00038335/+0.00013196`；positive group-cell=`0.49847`。
- Mechanism descriptive summary SHA256=
  `a7a869ae239dc29c4d9a961428942ba9d95299b951947a0af7ccf4ee340b09c2`；
  source→fused relative-dispersion retention K/V=`0.000302/0.0000170`，parent mass=
  `0.002056`，output delta L2=`0.0006855`。
- No winner implementation；0 training steps；E1-pilot/model-selection/test/confirmatory sealed。

## 2026-08-20 FPCT-E1-FAST-RCA pre-output lock

- Goal：直接回答当前 F 的负增益来自 candidate contraction、Jensen parent evidence、
  partition topology 还是不可识别因素；当前阶段不训练。
- Inputs：六个 E0 step-64/final checkpoints；冻结 E0-design groups
  ARC/OpenBookQA/MMLU-Redux=`128/70/128`。
- Compact input cache：
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-1c64b606-v1/input_lock/e0_design_input_lock.pt`，
  SHA256=`d843512b7e446efd229fe3097030407e59acb20b4de591a84b88176bd8b0eec9`。
- Representation：运行时 full-response teacher forcing；one sample/intervention row + online
  layer/head moments；expanded 30M rows=`forbidden`。
- Frozen interventions：C_post、F、centered λ、K-only/V-collapse、parent-mass preserving、
  partition overlap-length composition。Exact RoPE formula 在当前 frozen projector 上不可分离，
  所以不运行近似替代。
- Output root 将在 clean pushed execution SHA 后生成；本记录时 model/GPU/K8s/training=0，
  E1-pilot/model-selection/test/confirmatory 均 sealed。
- Pre-output verification：六 checkpoint tree hash 全匹配；targeted suite=`99/99`；full
  CPU-safe suite=`1082 passed + 42 expected historical fail-closed guards`。旧 A4/A5/R2
  immutable gates 保持不变，没有为 successor source 回写或放宽。
- Execution `1c74a522...` 的首次 `prepare` 在创建 root/model load 前因 direct-file
  `sys.path` bootstrap 缺失而终止；natural/model output=0，旧目标 root 不复用。
- `86b4b240...` plan 已验证全部输入/checkpoint，但 model load 前新增 1-sample smoke
  namespace；smoke 永不进入分析，完整 shard 仍从 group 1 重跑。因 executor 变化，
  `86b4...` root 只作 pre-model plan record。
- `70f64c78...` smoke 的 path/bootstrap/Git failures 全部发生在 model load 前。后继仅新增
  container execution-SHA injection contract：40-hex、与 plan exact equal；不改变任何
  scientific field。六 checkpoint stage 可按原 tree SHA 复用，新 plan/root 重新冻结。
- `2c1f79a6...` r6 已加载模型/checkpoint，但在任何 forward/logits 前因 collator metadata
  的 nested dict device move 停止，JSONL 为 0 bytes。只修复递归 tensor movement 后以
  new commit/root 重启，旧 temp 保留为工程记录。
- `df6629f7...` r7 的首个 forward 后、logp row 前，capture 发现 global labels 与 final
  wrapper section 坐标不一致（117 vs 17）。修复使用冻结 `kv_cache_index[-1]` 同时切分
  capture mask 和 logits labels；8 个 answer targets 不变，旧 forward 不进入分析。

## 2026-07-24 FPCT-E0 TMPDIR closure recovery

- Seed `2026072201` attempt 3 completed C_post training but failed the final sealed closure check; no F arm or accuracy evaluation ran.
- Provenance comparison identified an orchestration-only mismatch: moving `TMPDIR` under `/fpct-e0` disabled the immutable bootstrap's `/tmp` content-SHA normalization for PyTorch's generated remote module.
- The prospective recovery restores only `TMPDIR=/tmp`, keeps all persistent runtime caches outside `/opt/fpct`, requires an exact-image CPU closure preflight, and then restarts the full matched seed from step 0 before continuing the remaining two seeds.
- The first CPU preflight remained `BLOCKED`: torch-path canonicalization and source-tree invariance passed, while another stable-projection field changed. No GPU job was released; a replacement diagnostic preflight now records the exact recursive projection diff.
- The remaining difference was solely W&B's `ImportHookFinder` added to `sys.meta_path`. A presealed launcher now installs that same finder before fingerprinting, without changing the immutable SFT target or any training/evaluation setting; a third CPU preflight is required before GPU release.
- The presealed run completed both 64-step arms with matched integrity GO. Evaluation then skipped every attempted sample because the FPCT sidecar parent range exceeded the decode cache source length; those empty 0% summaries are invalid. A one-prompt, no-correctness geometry diagnostic is used to capture the exact runtime lengths before any evaluation recovery.
- The diagnostic captured prefill length 129 and first-decode length 1 for the unchanged sidecar range `[3,120)`. The wrapper incorrectly sliced a full cached-decode mask by current-section length only. The repair uses `initial_past_length + end`; actual Qwen3/DynamicCache and focused FPCT tests pass 11/11 and 66/66. A new immutable image is required before clean evaluation recovery.
- Corrected image digest=`sha256:19c7a815...`, source commit=`6a51ad4...`, embedded tree=`1534f7fe...`, tar SHA=`e13dbf2e...`. A no-correctness smoke must recover geometry 129→130 before seed 2201 evaluation attempt 5 and fresh seeds 2202/2203 are released.

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
- R2f two-lock：scientific SHA `d08b22b...`；image `sha256:cb91ec54...`；run-lock `1990589f...`；run UID `fpct-r2f-d08b22b-v1`；独立root/tar/sidecar copy。Lock前无R2f GPU/pretrained/training/checkpoint/accuracy output，首个授权步骤仅为complete synthetic GPU gate。
- R2f terminal：numerical gate GO，16 conditions+5 profiles完成；22/23 checks通过。唯一失败是FP32 checkpoint-native null `4.2915e-5 > 4.0e-5`（BF16=0）；504 panel-layer fused/collapsed KV均exact native，差异只在hierarchical执行顺序后深层累积。R2f不patch/resume，未运行matched smoke、训练、checkpoint或accuracy。
- R2g prospective repair：仅把shared parent eager adapter前置到hierarchical atom/group kernels之前，保持beta/gamma、tensor-only mixed-batch selection与全部冻结threshold不变；新增parent-first call-order regression。Targeted `68 passed`，full `409 passed, 2 warnings`；尚无R2g GPU/pretrained/training/accuracy output。
- R2g two-lock：scientific SHA `509a68a...`；image `sha256:e7061bb8...`；run-lock `4ba3cb77...`；run UID `fpct-r2g-509a68a-v1`。独立root/tar/sidecar完成，lock前无R2g GPU/pretrained/training/checkpoint/accuracy output。
- R2g-v1 infra terminal：target synthetic JSON为GO，但sealed bootstrap因缺少`attestations/`父目录退出1，0 attestation；v1 artifacts不复用。V2保持science/image/threshold不变，run UID `fpct-r2g-509a68a-v2`、run-lock `8f59e0b1...`，新root预建全部父目录后从complete GPU gate重跑。
- R2g-v2 terminal：sealed GPU gate GO、16 conditions+5 profiles完整，22/23 checks通过；唯一失败FP32 native-null仍为`4.2915e-5 > 4.0e-5`，parent-first假设被证伪。504/504 trace cells存在微小fused-vs-native RMS；未训练、未生成checkpoint、未读accuracy。
- R2h prospective repair：C_post/F共享candidate边界以tensor-only where强制hard gate=0 exact native，非零/训练gate不变；trace snapshots改为clone。Targeted `70 passed`，full `411 passed, 2 warnings`；尚无R2h GPU/pretrained/training/accuracy output。
- R2h two-lock：scientific SHA `39af03d...`；image `sha256:a1d9041f...`；run-lock `d34f17eb...`；run UID `fpct-r2h-39af03d-v1`。新root预建attestations/results，lock前无R2h GPU/pretrained/training/checkpoint/accuracy output。
- R2h terminal：sealed GPU gate GO，16 conditions+5 profiles完整，22/23 checks通过；唯一失败是FP32 native-null `4.2915e-5 > 4.0e-5`，BF16为0。504/504 FP32 trace cells的hard gates为0且candidate/collapsed K/V逐元素exact native，因此candidate canonicalization成立，但最终packed branch仍未恢复parent；0 matched smoke/training/checkpoint/accuracy。
- R2i prospective repair：新增parameter-free逐parent `parent_force_native` metadata，从C_post/F共享candidate boundary携带到packed parent-equivalence mask；不改变candidate、prior、mask、parameter、threshold或formal recipe。Targeted CPU/HF/numerical tests `80 passed`，full suite `414 passed, 2 warnings`；尚无R2i GPU/pretrained/training/accuracy output。
- R2i two-lock：scientific SHA `8d21c72...`；image `sha256:9ac006fe...`（复用R2h frozen runtime layers、离线重建源码与provenance）；run-lock `ee377b39...`；run UID `fpct-r2i-8d21c72-v1`。新root预建attestations/results，lock前无R2i GPU/pretrained/training/checkpoint/accuracy output。
- R2i-v1 pretrained gate：complete GPU GO、16 conditions+5 profiles完整、23/23 checks通过；checkpoint-native FP32/BF16 delta均0，forced-on activation、exact controls、no-sync与resource全通过；未读accuracy。
- R2i-v1 matched smoke infra failure：c_pre setup时W&B offline写入`/opt/fpct/wandb`，post-attestation source-tree hash mismatch；无optimizer-step/checkpoint/result artifact，formal training未启动。V1不resume；v2只把W&B输出目录重定向到`/fpct-run`，science/image/threshold/recipe不变，并需从完整GPU gate重跑。
- R2i-v2 infrastructure lock：science/image保持`8d21c72...`/`sha256:9ac006fe...`，operational source `ff473e1...`；run-lock `0eb0133e...`；run UID `fpct-r2i-8d21c72-v2`。V1 GPU/pretrained evidence不复用，新root从complete GPU gate重启。
- R2i-v2 repeated gate：新root complete GPU/pretrained再次23/23 GO；checkpoint-native FP32/BF16 delta均0，未读accuracy。Matched smoke的W&B隔离成功，但step-0在0-D BF16 byte hash与DDP duplicate identical remote-module sealing处失败；0 optimizer-step/checkpoint/result，v2 terminal。
- R2j prospective repair：state hash先flatten再uint8 byte view；sealed sys.path严格验证后只dedup byte-identical Torch remote-module marker；不改operator、recipe、threshold、panel、data或seed。需新scientific SHA/image/run-lock全量重启。
- R2j CPU验证：targeted trainer/bootstrap integrity tests `29 passed, 2 warnings`；full suite `416 passed, 2 warnings`。尚无R2j GPU/pretrained/training/checkpoint/accuracy output。
- R2j two-lock：scientific SHA `efa02fb...`；image `sha256:8eac5693...`；run-lock `51ce0a5e...`；run UID `fpct-r2j-efa02fb-v1`。新root预建attestations/results/W&B目录，lock前无GPU/pretrained/training/checkpoint/accuracy output。
- R2j terminal：complete GPU GO、16 conditions+5 profiles完整、22/23 checks通过。Checkpoint-native FP32/BF16 delta均0，forced-on `0.2450/0.71875`，exact/no-sync/HBM/expansion均通过；唯一失败为median latency ratio `1.6886 > 1.50`（p95 `1.3455`通过）。Controller=`GPU_ENGINEERING_BLOCKED_R2`，未运行matched smoke/formal training/checkpoint/accuracy/model-selection/held-out。

### 2026-07-25 FPCT-E0 三 seed 最终探索性结果

- Kubernetes Job `fpct-e0-decode-recovery` 正常完成，Pod restart=0；三个 seed、两条训练臂、四个 inference cells 和三任务评测全部完成，正式评测 `skipped=0`。
- 三个 seed 的 matched integrity 全部为 `GO`，真实 candidate-factorization activation 全部非零。旧 decode-mask 错误产生的空 0% 结果保持隔离，未进入最终统计。
- Seed-level `T` 为 `+1.2128/-3.3780/+1.3467 pp`，均值 `-0.2728 pp`；`O` 为 `-0.1302/-1.0826/-0.2158 pp`，均值 `-0.4762 pp`，0/3 为正。
- Task-level mean `T`：ARC `-1.5625 pp`、MMLU-Redux `+3.1250 pp`、OpenBookQA `-2.3810 pp`。
- 冻结 GO 规则中“至少 2/3 T>0”通过，但 `mean(T)>=+1.00 pp` 失败，OpenBookQA 低于 `-2.00 pp` task floor，query-time mechanism positive gate 也失败。
- 最终分类=`E0_NO_GO_FOR_FURTHER_SPEND`：当前 TinyLlama→Qwen3、2,048 examples/64 steps recipe 不支持投入 36-run confirmatory campaign；不外推为 FPCT 普遍无效。
- 小型正式结果已版本化到 `recipe/eval_recipe/fpct_e0/versioned_outputs/`；完整 55GB checkpoints、逐样本输出和运行日志保留在 `/netdisk/lijunsi/fpct-e0/fpct-e0-20260722-v1`，不提交 Git。

### 2026-07-26 FPCT-E1 阶段 0 前瞻锁定

- 从冻结 E0 结果 commit `613958af...` 建立独立 `research/fpct-e1-mechanism-audit` worktree；E0 compact results、main 与 Phase2A 均保持只读。
- 原“remaining calibration 再取 128/70/128”因 ARC 仅余 32、OpenBookQA 余 0 而不可行；用户在任何 E1-pilot forward/outcome 前批准 prospective amendment：改用 support-fit-only certified groups。
- E1-pilot 按 domain-separated SHA256 排序锁定 ARC 128、OpenBookQA 70、MMLU-Redux 128；E0-design 同规模，两者交集为 0。652-row manifest SHA256=`030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d`。
- 当前执行顺序冻结为 instrumentation hard gate → E0-design mechanism/topology audit → E0-design centered-λ sweep。Grid=`{0,0.25,0.5,1,2}`，λ0=C_post、λ1=F。
- 本阶段只运行 CPU/hash-only locker 与 synthetic unit test（`2 passed`）；未运行 tokenizer/model forward、GPU、训练或 E1-pilot，confirmatory outcome 未访问。

### 2026-07-26 FPCT-E1 instrumentation hard gate

- 修复旧 probe 的两项根本缺陷：显式 begin/end capture 取代 last-forward overwrite；query variance 改为跨 eligible answer queries 的 Welford，而非单次 decode `q_length=1` 方差。
- Teacher-forced mask 固定为 query `t` 预测 `labels[t+1]`，只纳入 shifted label 非 `-100` 的 response positions；prompt/invisible candidate 不进入统计。
- Random Qwen3 eager + DynamicCache 的 instrumentation OFF/ON logits、loss、cache bitwise equal；三步 greedy decode logits/cache/tokens均相同。
- Synthetic query-changing variance=`0.1973072141`、top1 change=true；identical candidates 的 KL/TV/Jensen 均精确为0；capture不保存raw KV。
- Instrumentation + E1 oracle + Qwen/reference/production targeted suite=`82 passed`。E1-1=`GO`；尚未加载 E0 pretrained checkpoint、运行自然 audit、GPU或训练。

### 2026-07-26 FPCT-E1 pre-data execution lock

- 研究目标：在 E0-design mechanism/topology output 前封存完整 producer-consumer contract，避免 baseline/F 竞态、resume 绕过、host/container path 漂移、整表 OOM 或事后修改 λ/operator。
- 冻结输入：E0-design 326 distinct groups；E1-pilot 326 groups 的 hash-only membership 可验证但不可 render/tokenize/align/forward。Checkpoint runtime 实际加载六个 immutable `final` projector trees，同时验证同 attempt `checkpoint-64` projector set byte-identical。
- 执行 recipe：18 个 C_post baseline shards 完整 marker 后才运行 18 个 F endpoints；36-shard analyzer 与 immutable `FINALIZED_E1_2` receipt deep-verify 后才运行 72 个 F λ additions。`λ=0` 是真实 F runtime control，`λ=1` 复用 endpoint；最终 108 shards 全量报告。
- Artifact recipe：input sidecar、raw topology JSONL/Parquet/aggregates、每 shard capture Parquet、stage merged Parquet、六个 analyzer artifacts、source/runtime/ConfigMap/finalized receipts。大文件只写 `local/` 或 `/netdisk`，tracked manifests 记录 path/bytes/rows/SHA256。
- 可复现 pre-data tests：`CUDA_VISIBLE_DEVICES='' python -m pytest -q --no-cov -p no:cacheprovider` 加 15 个 E1 instrumentation/executor/reference/Qwen test files，结果=`169 passed`。All-FPCT CPU suite=`356 passed`；项目 CPU-safe full suite=`595 passed, 2 warnings`。R2l/R2m 两个历史 immutable-lock failures 不属于 functional regression，未修改其历史 allowlist。
- 预期 Commit A 后命令顺序：`fpct_e1_source_snapshot_lock.py create`；`fpct_e1_prepare_input_lock.py`；`fpct_e1_mechanism_audit.py raw-topology/verify-raw-topology`；immutable-image `fpct_e1_runtime_probe.py`；`fpct_e1_capture_runner.py prepare`；`fpct_e1_k8s_lock_bundle.py build/verify`；随后才允许 render baseline K8s jobs。
- 当前观测：0 个新自然 E1 row、0 pretrained forward、0 GPU/K8s job、0 optimizer step、0 checkpoint modification、0 E1-pilot/confirmatory outcome。当前 GO 只表示 execution protocol/code 完整，不是 mechanism 或 performance GO。

### 2026-07-26 FPCT-E1 pre-data operational closure amendment

- `744a943ea804dfebe4e6d3cba756b6a89763002f` 只产生 git archive、外置 source receipt 与空 input-lock directory；在 0 dataset lookup/tokenization/alignment/model/GPU 时被 `ABANDONED_BEFORE_NATURAL_DATA`，不 resume、不复用、不是科学 NO-GO。
- 后继 execution 必须使用全新 run root，receipt 位于 snapshot root；runtime renderer/template/本地 imports 均来自该 exact snapshot。Host 与 Pod 分别复验 mounted receipt、canonical raw bytes 和 mounted tree，probe 在 torch/CUDA 前完成验证。
- Capture Job 必须携带 expected plan SHA 并重算 claim ID；initial/finalized ConfigMap `verify --output` receipts 是 renderer 的必需输入。Output 与 source/E0/models/input/raw 在 physical host path 和 lexical container path 上 fail-closed 隔离；rootfs read-only，cache 仅写 `/tmp` emptyDir。
- 镜像固定为命名 OCI digest `repository@sha256:<64hex>`；裸 digest/tag 均拒绝。Fresh-subprocess hostile-PYTHONPATH/bytecode test、CLI forwarding、receipt/plan/path tamper tests全部通过。
- 可复现 15-file CPU pre-data suite 重新执行为 `211 passed, 0 failed`；all-FPCT CPU=`400 passed`；项目 CPU-safe full suite=`639 passed`（`CUDA_VISIBLE_DEVICES=''`，`--no-cov -p no:cacheprovider`）。截至锁定仍无 E0-design 自然 row、pretrained output、checkpoint load、GPU/K8s、training、E1-pilot 或 confirmatory outcome。

### 2026-07-26 FPCT-E1 d169 pre-artifact integrity failure

- Execution=`d1698177e60455a42731165b88a66374b8826718`；run root=`/netdisk/lijunsi/fpct-e1/fpct-e1-d1698177-v1`；source receipt file SHA=`84811c64...`、mounted tree SHA=`5d7b4da6...`，均验证通过。
- CPU input lock 已解析本地 Qwen3/TinyLlama tokenizer，并至少在 MMLU-Redux/high_school_geography 运行自然 tokenization/alignment；raw topology 分类随后抛出 `ValueError: certified partition geometry is not a disjoint complete cover`。Exact processed row count/hash=`UNKNOWN_NOT_MATERIALIZED`。
- 输出目录实测为空：0 input sidecar/manifest、0 raw artifact/plan/runtime probe/ConfigMap、0 model/checkpoint load、0 forward、0 GPU/K8s/training、0 scientific result。D169 标为 `INCONCLUSIVE_INTEGRITY_FAILURE`，不 resume/reuse。
- 独立代码审计确认 sanitizer 以 span order 认证，旧 classifier 却以 top-k slot order cursor 覆盖。最小 A3 修复只排序 derived intersections；candidate records、indices、weights、A 与 slot-0 语义不动。Mechanism tests=`22 passed`，15-file=`213 passed`，all-FPCT=`402 passed`，项目 CPU-safe full suite=`641 passed`。
- 下一次只允许从新 clean/pushed A3 建全新 snapshot/run root 后，从 CPU input lock 开始全量重跑；任何 d169 partial state 均不得进入后续分析。

### 2026-07-26 FPCT-E1 A3 pre-artifact resource-ceiling failure

- Execution=`612697dfc44ab46699728b8d2de0a6fce980a889`；run root=`/netdisk/lijunsi/fpct-e1/fpct-e1-612697df-v1`。Source git tree=`163a4eca...`、mounted tree SHA256=`c55a1d96...`、receipt file SHA256=`3e5720a3...`，均验证通过。
- CPU input lock 只处理 E0-design，自然 lookup/tokenization/alignment 日志至少覆盖 ARC、OpenBookQA 与 MMLU-Redux；在 artifact 写出前抛出 `ValueError: input-lock sample exceeds long-form row ceiling: 616448 > 262144`。精确 completed task set、processed sample/row count 与 failing group hash=`UNKNOWN_NOT_MATERIALIZED`。
- `616448=28×16×1376` 是精确 logical row product，不是 duplicate runtime emission。预注册 `262144` 是累计 logical-row memory guard；Parquet physical batch/row-group 是独立的 `4096`。
- 输出目录实测为空：0 input sidecar/manifest、0 raw topology、0 plan/runtime probe/ConfigMap、0 model/checkpoint load、0 forward、0 GPU/K8s、0 training、0 scientific result。E1-pilot 未 render/tokenize/align/run/read。
- 当前状态=`INCONCLUSIVE_RESOURCE_CEILING / REVIEW REQUIRED`；本 run 不 resume/reuse。不得自动提高 ceiling、截断 rows、删除超限样本或把 ceiling 静默改称 chunk size。只有人工前瞻批准的新 protocol/schema 与全新 execution 才可继续；E1-2/E1-3 当前 blocked。

### 2026-07-26 FPCT-E1 A4 streaming pre-natural experiment lock

- 批准：用户在新自然 input-lock 与任何 model output 之前批准 `APPROVED_PROSPECTIVE_AMENDMENT_E1_A4_STREAMING`，用 representation-preserving deterministic bounded streaming 替换 A3 cumulative per-sample row list。
- 科学不变量：保留完整 logical-row universe、原 15-field key、全部 metrics/weights/topology/prior/mask/estimands、operator、checkpoints、centered-λ grid 与 E0-design；physical chunk rows=`4096`。
- 历史边界：`744a943...`、`d1698177...`、`612697df...` 仍为 abandoned/non-reusable，不得 resume、复用 partial artifacts 或原地修补重跑。
- 新产物：`FPCT_E1_STREAMING_AMENDMENT.md`、`e1_streaming_contract.json`、`e1_streaming_schema.json`、streaming verifier/synthetic-gate code 及 input-lock/runtime/analyzer 的 bounded-streaming 消费链。
- Synthetic gate=`GO`：reference row key/权重/topology/aggregate、ordinal identity、chunk partition、semantic replay、atomic no-overwrite、crash/resume 与 bounded RSS checks 全部成立，`whole_table_materialization_detected=false`；测试=`395 passed in 4020.78s`。
- Gate artifact SHA256=`42b6c98fd5f27a49e258bae4b79ff3c4673c9144465a3c43f9a672d228c82df4`，evidence SHA256=`d22bb1997234dd0c895f1b819e9cbf3bb1102633502f482cbb7b2d4d4ecc8e43`；amendment/contract/schema SHA256=`2fd6412a5533ecc4085e52a81e514425ae5fe6ffa64ec47ebbd7bc9669a20c74` / `919d7c9955c749d2d3603a2356701c65e5b552d22b8ab8cc23c91ea7246e6f22` / `5d389e81f87a18e02889204f055e61c9a7fed8957e8d990f9642558700a39618`。
- Full-schema baseline=`4480 rows / 2 chunks / 147111936 B peak RSS`；stress=`1000384 rows / 245 chunks / 192376832 B peak RSS`；阈值=`283295744 B`（canonical row bound=`2078 B`，allocator/buffer allowance=`136183808 B`），故 bounded-RSS hard gate 通过。
- A4 instrumentation re-attestation=`GO`：`110 passed`，gamma query variance=`0.19730721414089203`，parity/synthetic/hard-gate SHA256=`d7e78e5e8b54aa2ae21e105b53ad5cbdc8b52d78b6e422c36bdeed919d40c2f3` / `7524fb2ce873dde3450ad12ec26e39d6497f0f99fd9fe755edcfafda44c2ab40` / `0cd401328c5942f51a21f98a3418eb3a18c08a489c6a5cc8c99ef6775b1d6b3e`；historical v1/A3 evidence 未覆盖。
- Gate 外 project CPU-safe complement=`449 passed, 2 deselected`；两项 deselected 是只接受旧 R2l/R2m frozen tree identity 的历史 guards，在 A4 后继分支上按设计 fail-closed。Gate+complement 共 `844 passed`，没有把这两项历史身份检查伪报为 A4 通过。
- 执行 firewall：synthetic GO 以前禁止新自然数据；clean pushed A4 以前禁止新 snapshot/run；streaming CPU input-lock GO 以前禁止 runtime/checkpoint/plan 与 E1-2；E1-3 只能在 finalized E1-2 后执行。
- 当前计数：A4 授权后新 natural lookup/render/tokenize/alignment=`0`，pretrained model/checkpoint forward=`0`，GPU/Kubernetes/training=`0`；clean pushed A4 commit、successor snapshot/run root 和 CPU input lock 均仍 `PENDING/NOT STARTED`；E1-pilot=`SEALED / NOT RUN / NOT READ`，confirmatory 仍 sealed。

### 2026-07-27 FPCT-E1 A4 CPU input-lock fail-closed record

- Execution：`07755a4039e89700e59af9e141026a57142f9da0`；UID=`fpct-e1-a4-streaming-07755a40-v1`；root=`/netdisk/lijunsi/fpct-e1/fpct-e1-a4-07755a40-v1`。
- Immutable receipts：blocked SHA256=`bc9002daebd1e8921d5fd5ca0705ab59efd115b350a785e0c368d322161cdec0`；identity SHA256=`bdafa8df8d609b9bbf4c3468a7bb1452ec459c6e1adb5201a85ea772aa3f40df`；source receipt file SHA256=`b792803b23e22d082743d7f39aefcebd7178c5cd6e10108756f6693d7a778a8d`；source receipt 内部 `receipt_sha256`=`a5b5e87a816377fcaa4cb2080031da22532239a08c65ef5b07265ca4759b0afa`。
- Producer 原生状态：`A4_INPUT_LOCK_BLOCKED`，failed check=`input_lock_rendered_prompt_sha_mismatch`。下列逐行定位是 read-only post-block diagnostic，不是对原 blocked receipt 的追溯改写。
- Prefix：前 `160/326` groups exact；首个 mismatch ordinal=`161`。ARC group=`2ac15877caa468bf7ee3f2c16bcb9fd7b6e122bc2081c7eaaf2ecd0f64af42e4`；sample=`55c885c900afb3b5f7a4541c68797000852c31971527148f5c29b6ccf783b4d8`；source row=`836`；eval qid=`32`。
- Rendered expected/actual SHA256=`2b933c569545f2e26944f1702c1e878c20cb9554cd7d679cd48395b9da6e2828` / `ad7828e4c67fad1515e4bb768624114be20091fd6bbb86960a3931d439a04a9d`；alignment expected/actual SHA256=`1440a0c16db39ecb915318fd836dbad959c1ca5fb275b145343e8011cd79657f` / `206b4ddc9b24874ea7b2d892e7c770901d18ccb5eb30e7cf3ac5d22fea5f03ed`。
- Root cause：historical anchor 从 first-four label-free projection 渲染 A-D；materialized production row 保留额外 E 选项，真实 E0/A4 formatter 遍历完整 choices。该差异与 A4 streaming、operator 或 mechanism 无关。
- Persistent outputs：仅 execution identity 与 generic blocked receipt；`0` usable input sidecar/manifest/templates/raw/runtime/plan，`0` model/checkpoint load/forward，`0` GPU/K8s/training。E1-pilot 与 confirmatory 继续 sealed。
- Disposition：run root 永久 abandoned，resume/reuse=`false`。下一步=`HUMAN REVIEW REQUIRED`；未批准的候选合同为 strict historical first-four anchor 或 actual E0 production-runtime prompt。不得在同一 execution 自动选择、放宽或重跑。

### 2026-07-28 FPCT-E1 A5 production-prompt prospective experiment lock

- Human decision：
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5_RUNTIME_PROMPT`；operative input=
  `ACTUAL_E0_PRODUCTION_RUNTIME_PROMPT`。First-four 仅保留为 immutable
  selection/content-group/provenance anchor。
- Population：E0-design 326 groups，ARC/OpenBookQA/MMLU-Redux=`128/70/128`；
  membership、answers、checkpoints、operators、lambda grid、estimands 与
  statistical weights 不变。E1-pilot 和 confirmatory 全部 sealed。
- Planned no-model experiment：从 group 1 对完整 population 生成双锚点 census，
  每 row 记录 historical/production choice counts、raw labels/gold、first4/full
  raw/prompt/alignment hashes、production token/certified-parent/logical-row/chunk
  geometry。Classifier 只允许 exact match 或 extra choices only；question、前四
  text/order、gold A-D、template/whitespace 或唯一映射异常均 fail-closed。
- Pre-natural gate：新 gate path=
  `recipe/eval_recipe/fpct_e1/e1_a5_prompt_synthetic_gate.json`，必须重新运行
  A4 streaming oracles、A5 instrumentation、dual-anchor/classifier/partition/
  corruption tests。Historical A4 gate 保持只读，不对 A5 current tree 重验。
- Pre-natural result：A5 gate=`GO`；`248 passed / 0 failed`。Gate artifact /
  evidence SHA256=`464e646c336d33c97a03c603d283a95b1579508d63ae0f19d7c700ccf2dcfc02`
  / `2fc5c70da231cfd0aba54656c5a34ff142594208f4a006348f73607559af4a47`。
  Million-row stress=`1,000,384` logical/emitted rows、245 chunks、semantic replay
  exact，peak RSS=`676,200,448 B` < frozen threshold=`812,384,256 B`。此 gate
  明确记录 natural E0-design access=false、model instantiated/forward=false。
  在该 pre-execution gate 冻结时，326-group affected count、choice/token/
  alignment/topology/row geometry 仍未知；随后 `9b248d20` execution 的 terminal
  结果见下一节，不能把本条历史 `PENDING` 当作当前授权。
- 先前未提交的 gates `6dc44be9...` / `b116670b...` 分别因 source closure
  不完整与 tracked amendment Markdown 空白清理而在自然数据前失效，原字节隔离到
  `local/fpct_e1/a5_precommit_invalidated/`；它不属于正式结果。随后补齐
  source-snapshot producer、真实 A5 gate-shape 与 producer/verifier 回归后，
  从头重跑得到上述 operative gate。
- Execution firewall：`07755a40` 永久 abandoned；A5 要求新 clean pushed
  commit、immutable snapshot、UID/root。在该 gate 冻结时尚未进行 A5 natural
  census/input lock，未加载 model/checkpoint，未运行 forward、GPU、Kubernetes、
  training、E1-2/E1-3，未访问 E1-pilot/confirmatory；后续 BLOCKED 结果见下一节。

### 2026-07-28 FPCT-E1 A5 CPU input-lock pre-group-1 failure

- Execution/source SHA=`9b248d2094b684f5d9e9a218919a354b7d97468e`；UID=
  `fpct-e1-a5-runtime-prompt-9b248d20-v1`；root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5-9b248d20-v1`。Source snapshot 为 641
  entries；Git tree=`ec8254302e831d83a169971964e848550e659f75`；mounted
  tree SHA256=`ee824b0d51d015a615f0db836ab1d0e2b555e36bcc55a5a284245dcc98f1cb65`。
- Sealed bootstrap 与 source-snapshot provenance 通过；本地 tokenizer paths/
  files 被解析和枚举、tokenizer objects 被加载，但 strict tokenizer-bundle/
  template attestation 尚未完成。Producer 在 E0-design group 1 lookup/render/
  tokenize/align 前因 `a5_materialized_e0_development_tree_sha_changed` 终止。
  状态=`A5_INPUT_LOCK_BLOCKED`；natural group count=`0`；无 census/alignment/
  sidecar/geometry/runtime/plan/scientific artifact。
- Blocked receipt/identity SHA256=
  `fe305b2c9d881b202f4a096646e9530e8d630a6fcfa095754d3101450ad7ab79` /
  `2247ccf274ea8c55b25b55fd5ac3fb2a191d6796f5ea06a9b18b4a9d8e6b7a0b`；
  source receipt file/internal SHA256=
  `dc4396d50594f0db120fcba9252e85c92e343a014230379ebe8c51e645082981` /
  `9d051cca1290e4d90560587e5d95ed8f81c8e4a156090baf323614a0a07ec95e`。
- Read-only diagnostic 在相同 51 files / 233,569 bytes 上重现 frozen E0 declared
  SHA=`f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`
  与 A5 generic manifest SHA=
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`。
  两者使用不同算法，故 root cause=
  `HASH_ALGORITHM_DOMAIN_MISMATCH_DATA_BYTES_UNCHANGED`；无数据漂移或损坏证据。
- Run/root 永久 abandoned，resume/reuse=`false`。A5R1 双 hash-domain 修复仅以
  draft addendum 记录，等待
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`；不得提前运行新 gate、
  successor execution 或自然 census。
- 未加载 model/checkpoint、未运行 forward/GPU/Kubernetes/training，未进入
  E1-2/E1-3，未访问 E1-pilot/confirmatory。

### 2026-07-30 FPCT-E1 A5R1 pre-natural dual-hash gate

- Approval=`APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS`；用户原始
  回复=`可以`。授权范围仅为 v8 hash-domain repair、CPU/offline synthetic gate、
  新 commit/snapshot/root 与从 group 1 开始的 CPU input lock。
- Configuration：declared algorithm/value=
  `relative_path_nul_file_sha256_bytes_v1` /
  `f3dcf2c77e6c5f90946994488fcb86f67dcdc590510a9f469f32e86773492c73`；
  generic algorithm/value=`canonical_json_file_manifest_v1` /
  `12f537cade1a30f6fd4e7a146c58311412f8824a6845ea4e6f7a6b5651bcb405`。
  禁止跨域 equality；旧 v7 gate=`464e646c...` 只作 historical predecessor。
- Verification：v8 gate=`GO`；`291 passed / 0 failed`，test-output SHA256=
  `652e62bfb95a6acc1cfe156aa67b7c8010856105f85ef329f5dfa8847fe96250`；
  gate/evidence SHA256=
  `a2e53784fa46d2f63963bdb41e327a4e17936cce2bff765802b34f3d87869a9f` /
  `14225f02046f0fe626b3fab1cd8d07d5e45e873ac54f3d2e0ba78d78a6fdd003`。
  Stress=`1,000,384` logical/emitted rows、245 chunks、replay exact；peak RSS=
  `676,876,288 B`，threshold=`813,060,096 B`。独立 verify 与 v7 SHA closure GO。
- Non-operative attempts：一次 direct-file import failure；一次完整 CPU compute
  在最后 publish 因 sibling worktree read-only 失败。两次均未生成 gate/partial
  artifact。`681532e9...` precommit gate 随后因 schema/terminal-atomicity 缺口
  被隔离为 local-only invalid evidence；修复并重新完整执行后才得到唯一正式 gate。
- 在 gate 冻结时没有自然 census/output，successor commit/snapshot/UID/root
  尚未建立；随后获授权的单次 CPU input-lock 结果见下一节。0 model/checkpoint
  load、0 forward、0 GPU/CUDA/K8s/training；E1-2/E1-3 与 E1-pilot/
  confirmatory 继续 NOT AUTHORIZED/sealed。

### 2026-07-30 FPCT-E1 A5R1 CPU input-lock terminal failure

- Execution/source SHA=`37be816ad611b8b0d916bd98c840c5f31efe2b50`；UID=
  `fpct-e1-a5r1-hash-domains-37be816a-v1`；root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-37be816a-v1`。Snapshot=649 entries；
  Git tree=`7aea57815c6abb0005739f77d9ea753f22e26d85`；mounted tree SHA256=
  `1c2f3a1db2036104d13e1f1f07225d5eeaf8e8c4276c827c1b59340354875508`。
- Canonical CPU/offline bootstrap 从 group 1 启动，source snapshot、tokenizer
  assets 与 declared/generic E0 data hashes 通过。本地 tokenizer 与 ARC dataset
  被加载后，choice-cardinality hard check 抛出
  `A5 materialized row has fewer than four choices`。Native status=
  `A5_INPUT_LOCK_BLOCKED`；resume/reuse/scientific result 均为 false。
- Blocked/identity SHA256=
  `46143877891c15fab1b5ebd3d359b80f3d9aa7464ceb961a0e8f353b0873bee2` /
  `c0a1e1b1b0de7700e4a7d7ce79c3317af30c183b1f321187b2cd505c592b27e5`；
  source receipt file/internal SHA256=
  `9e0ac64a3653b58a7c518650f5795ebd10ecc24ffe702def754e663ba480339b` /
  `1bbb0c4e38b36138b8d7c3592f8bbe10d91c1b174860610f947feec18086b6d4`。
- Failure 前没有持久化 group ordinal；本轮不做 post-failure natural population
  scan。只对 frozen ordinal-1 ARC row 做单行 replay，确认其为 canonical 四选项，
  但不能定位失败。Persisted census rows=`0`，未生成 sidecar、manifest、geometry、
  streaming 或 scientific result。Closure=
  `recipe/eval_recipe/fpct_e1/executions/37be816a/input_lock_failure_receipt.json`，
  SHA256=`ec3ae949b557da957a1a1f295f41b91b8522420445b5479a945b1f59b5de8e8a`。
- 0 model/checkpoint load、0 forward、0 GPU/CUDA/K8s/training；未进入 E1-2/E1-3，
  未读取 E1-pilot/confirmatory。任何修复都需要新 prospective amendment、commit、
  snapshot、UID/root；不得复用 `37be816a`。

### 2026-07-31 FPCT-E1 A5R2 pre-natural choice-cardinality gate

- Approval=
  `APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R2_CHOICE_CARDINALITY_RECOVERY`；批准发生在
  新自然 choice audit、tokenization/alignment、模型输出和 E1-pilot outcome 前。
  A5R2 允许 ARC 的实际 `n>=2` cardinality（2/3-choice 是合法 low-cardinality），
  OpenBookQA/MMLU-Redux 仍要求 exact-four；production choice list 不 padding、不
  truncation，historical count 固定为 `min(4,n)`。
- Configuration：protocol/contract/schema SHA256=
  `89753bcbdec66d07c36bcfc3a5c636e66704cddce5546ac0e321d7a9ea053384` /
  `a4bdf4a229d26b367fb8ea7c90adf39d72e94c56daf4d095346bf673c5c2eb1b` /
  `9cb387628e4b8fcf6c978e082b8c3380dbc62406c6aa460892c679872d656d7f`。
  Audit projection 仅包含 question、choices 和冻结的定位/provenance metadata；
  label/answer/correctness/prediction/accuracy/selector outcome 不进入 row、hash、
  log 或 summary。
- Verification：CPU/offline pre-natural synthetic gate=`GO`；`325 passed / 0
  failed`；tracked tree SHA256=
  `60946d56e4c11ebcff8ac44b94b8bcdedbd6a6c5aea71ed168f854d6dce996ca`；
  evidence SHA256=
  `8788694b7715676a84e453f80b3bb596c5688a7660fa916964ab15ddaf9213c4`；
  gate artifact SHA256=
  `cfc8c7da3bb074b0a4fcb23276af8d8404516b656337abb5c3034724272d2cd9`。
  Gate 验证 variable-cardinality parser、完整 326-row synthetic census、taxonomy、
  ordinal/summary/audit lock、dual hash binding、atomic/no-overwrite 与 tamper
  fail-closed；不包含任何自然 population 统计。
- Execution status：到本记录冻结时，新的 326-row label-free natural audit 未运行、
  未读取；clean successor commit/snapshot/UID/root 尚未物化，CPU input-lock 也未
  运行。A5R1 execution/root `37be816a...` 永久禁止 resume/reuse。未加载
  model/checkpoint，未运行 forward/GPU/CUDA/Kubernetes/training，未进入
  E1-2/E1-3；E1-pilot 和 confirmatory 保持 sealed。

### 2026-07-31 FPCT-E1 A5R2 CPU choice-audit terminal closure

- Execution=`e765d493733d9eec94c152506a1c57781e26fb41`；UID=
  `fpct-e1-a5r2-choice-cardinality-e765d493-v1`；root=
  `/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-e765d493-v1`。Source snapshot 658
  entries；mounted-tree SHA256=
  `ebaa80303447e931dd0695ba49d76d21d1f630f87e94577828b53d15239e2c5f`。
- CPU/offline audit 完成 326 个 frozen E0-design group 的 label-free 内存
  reduction，但在发布 immutable `choice_audit` 目录时，
  `renameat2(RENAME_NOREPLACE)` 返回 `EINVAL`。无 final ledger/summary/lock；
  staging 已清理，不能据此读取或推断 natural audit decision。
- Terminal=`A5R2_INPUT_LOCK_BLOCKED`；blocked/identity SHA256=
  `498a3571a8933d34c3bdf7e4f2641b8da5f863bff7a5f87124488794dcc71fd2` /
  `a2c8786c290be420dad6a89a7615fb1db49c7092c0ce600c5d2add8f0534df8f`。
  Closure=
  `recipe/eval_recipe/fpct_e1/executions/e765d493/input_lock_failure_receipt.json`，
  SHA256=`f6715f9d86b23ccc265f6ead7b24af47acaf1e1ecd9eb14552af7940d68f798c`。
- No post-failure natural scan；未修改 protocol/code/threshold，未运行 tokenizer/
  alignment、model/checkpoint forward、GPU/CUDA/K8s 或 training。`e765d493` root
  永久 no-resume/no-reuse，E1-2/E1-3 未进入；后续修复需新的 prospective approval。

### 2026-07-31 FPCT-E1 A5R3 portable publication pre-natural gate

- Human approval：`可以`，在任何 successor natural access 前冻结；唯一 amendment
  是 choice-audit publication primitive。A5R2 v9 population、parser、taxonomy、
  threshold、prompt/tokenizer/alignment 与 operator/estimand 不变。
- Protocol：natural row 1 前创建并 fsync retained O_EXCL claim；完整 reduction 后
  创建 fixed staging；ordinary same-filesystem rename exactly once；以 O_EXCL+fsync
  completion receipt 作为 durable commit point。只有 exact claim/final/receipt 可
  消费，失败 root 不清理、不 resume、不 reuse。
- CPU/offline synthetic gate：`42 passed`；`/netdisk` scratch rename/fsync/
  same-device probe GO；tracked tree=`659151ff...`，gate evidence=`b4c090fa...`，
  gate artifact SHA256=`92c63e3b922405b8c0dcb42a642eeef7b528f1115ea7fc3ba4ab325c61c87bac`。
  Inherited 353-test closure 的五个初始 fixture/environment failures 均按既定合同
  修正或在规范 temp-domain 复验通过，没有改变 scientific rule/tolerance。
- 本记录不含新的 natural audit result、tokenizer/alignment、model/checkpoint、
  forward、GPU/CUDA/Kubernetes 或 training；E1-2/E1-3/E1-pilot/confirmatory 未授权。

### 2026-07-31 FPCT-E1 A5R3 pre-natural verifier recovery

- Execution `b6109443...` 在 natural row 1 和 publication claim 前以
  `A5R2_TRACKED_SOURCE_TEST_MAP_IS_STALE` 终止；原因是 successor 错用 immutable
  A5R2 gate 的 live-tree verifier。该 root 永久 no-resume/no-reuse。
- Closure SHA256=`efbecc09...`；natural/tokenizer/alignment/model/GPU/training 全为
  0。修复只保留 frozen A5R2 gate SHA/schema/status 验证并移除错误的 successor-vs-
  historical live-tree comparison。
- Replacement synthetic gate `43 passed`；tracked tree=`54f108bb...`，evidence=
  `907f3b6a...`，artifact SHA256=`004a8feb...`。下一次从新 SHA/root 的 group 1 重启。

### 2026-07-31 FPCT-E1 A5R3 post-reduction publication-mode closure

- Execution `3263531e...` 使用 fresh 666-entry immutable snapshot 与 sealed CPU
  bootstrap。Claim 在 natural row 1 前创建；完整 326-row label-free reduction 后，
  fixed staging 因 `/netdisk` setgid inheritance 得到 `02700`，exact-`00700` safety
  predicate fail-closed。Final/receipt/tokenizer/alignment input-lock 均未生成。
- Root 永久 no-cleanup/no-resume/no-reuse；tracked closure SHA256=`8ca35379...`。
  程序已执行 label-free reduction，但 human/agent/reviewer 未打开、汇总或使用
  unpublished natural choice statistics；0 model/checkpoint/forward、0 GPU/CUDA/K8s/
  training。E1-2 未授权，下一步需要人工决定是否批准最小 A5R4 mode-predicate
  amendment；当前未修改代码。

### 2026-07-31 FPCT-E1 A5R4 inherited-setgid mode pre-natural gate

- Human approval=`可以`；只改变 staging/final directory mode predicate，接受 `00700`
  或严格 parent-bound inherited `02700`。v9 scientific choice semantics 与 A5R3 v10
  claim/receipt/publication bytes 不变；`3263531e` root 永久 no-resume/no-reuse。
- CPU/offline gate=`78 passed`，tree=`753f8af5...`，evidence=`7e3ee42b...`，artifact
  SHA256=`0f347757...`。真实 target-FS probe evidence=`ad7bfe38...`，验证 setgid、
  UID/GID/device/inode/mode、ordinary rename、bytes 与 fsync，唯一 scratch 已清理。
- Supplementary inherited suites=`155 passed / 2 failed`、`254 passed / 2 failed`；四项
  failure 在 immutable predecessor snapshot 原样复现，属于历史 fixture/API drift，
  不属于 A5R4 regression，未修改也未纳入 v11 gate GO。
- Sealed-import/prepare/source-snapshot targeted suite=`94 passed`；active loader 与 gate
  verifier 均返回同一 `78/tree/evidence` 闭包。
- 本记录生成前未运行 successor natural audit、tokenizer/alignment、model/checkpoint/
  forward、GPU/CUDA/K8s/training；E1-2 未授权。下一执行必须使用新 clean pushed commit、
  immutable snapshot、UID/root，从 group 1 重启。
### 2026-08-14 FPCT-E1 A5R5 fresh-root interruption recovery

- `a47d52f8...` root 的 choice audit 已发布，但 input lock 仅完成 geometry 与首个 sample
  的 21 chunks；无 sidecar/main manifest/GO/BLOCKED。External termination cause 未验证，
  该 root 作为 hash-only forensic evidence 永久 no-resume/no-reuse。
- 用户批准完整 root-cause→单因素修复→单 seed→条件三 seed计划，并选择全新 A5R5
  root。A5R5 不改 prepare、A5R4、math、population、alignment、operator 或 threshold。
- 新 detached controller 使用外置 sibling state、O_EXCL materialization/launch/worker-start
  claims、Popen PID/starttime/cmdline binding、snapshot-origin deep verifier、CPU/offline env 和
  no-relaunch terminal semantics。新 run 必须从 group 1 重算。
- Pre-natural tests=`147 passed`；test SHA=`a4c9e7fb...`，gate evidence=`f2da57e2...`，
  artifact=`6190db2d...`。本记录生成时 0 successor natural/model/GPU/training/output。

### 2026-08-15 FPCT-E1 A5R6 schema-binding recovery

- A5R5 `ab4052cd...` 完成 326 groups/30,370,816 rows/7,564 chunks 后，在 final
  manifest schema validation 因额外 `choice_audit.publication` fail-closed；BLOCKED=
  `9fc9c3a6...`，不是机制或性能结果。
- A5R6 唯一 producer 修复为 five-field v9 binding；A5R3 publication receipt 保持独立
  strict verification。v9/A5R3 schema SHA 不变。
- Failed-root portable inventory=`9,343 entries / 67c572aa...`，跨 namespace digest 不含
  numeric UID/GID，环境内 owner/group/mode safety GO。
- Pre-natural operative suite=`318 passed, 4 deselected`；gate evidence=`c6402176...`，
  artifact=`9f4787a6...`，test output=`92e4a68c...`。Exact tracked/immutable map、
  required test nodes、四个 historical deselection 和旧 worker result/log live SHA 均已
  冻结；controller 在 root 创建前强制 gate，deep GO 以 strict durable receipt 封口。
  本记录生成时尚未创建 successor root，0 model/GPU/training。

### 2026-08-15 FPCT-E1 A5R7 active-gate recovery

- A5R6 `b2e34999...` terminal=`PRECOMPUTATION /
  A5R4_TRACKED_SOURCE_TEST_MAP_IS_STALE`；natural rows=`0`，input root 只有 identity 与
  BLOCKED。Worker result/log SHA=`99fc0245...` / `44a6052c...`，root tree=
  `6c308644...`；永久 no-resume/no-relaunch/no-reuse。
- A5R7 configuration：historical A5R4 gate SHA=`0f347757...`，strict v11 static
  consumption；current-source v14 gate；A5R3 AST unchanged=`10`，A5R4 mode AST
  unchanged except loader=`13`；新 loader AST 单独冻结。
- CPU/offline targeted production-preflight/tamper/controller suite=`103 passed`；最终
  frozen suite=`326 passed, 4 deselected`。Test output SHA=`cfac8e70...`，final gate
  evidence=`728cfb0f...`，artifact SHA=`15da7e33...`；独立 verifier 与真实 production
  loader 均通过。生成 gate 时未运行 natural tokenizer/alignment、模型、GPU 或训练，
  未进入 E1-2/3。
