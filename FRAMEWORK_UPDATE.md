# FRAMEWORK_UPDATE.md

## 2026-07-16：建立 ICLR 2027 研究协作规范

### 研究目标

建立适用于 C2C/Rosetta 项目的统一研究、实验、验证和版本管理规则，为 ICLR 2027 论文研究提供可复现工作流。

### 核心改动

- 新增根目录 `AGENTS.md`。
- 固化 Route-1 v2.2 为当前对照基线。
- 明确 learned flexible alignment 为下一研究主线。
- 统一 Conda、本地 GPU、测试、训练、评测和分析命令。
- 增加 Kubernetes namespace、资源提交和清理边界。
- 规定不使用 Pull Request，直接通过 commit 管理并推送 `main`。
- 建立本文件的固定更新结构。

### 实验配置

- Conda：`c2c-py310-cu124`。
- Python：3.10.20。
- PyTorch：2.6.0+cu124。
- Transformers：4.52.4。
- 本地硬件：4 × RTX 4090。
- Kubernetes context：`default`，未指定 C2C 专属 namespace。

### 验证结果

- `AGENTS.md` 共 284 行，符合 200–300 行要求。
- 全量测试通过：65 passed，2 个已知 Pydantic warning。
- `git diff --check` 通过。

### 结论与下一步

后续每次研究改进都必须同步记录假设、改动、配置、验证、结论和下一步；新方法首先与 v2.2 做公平小闭环对照，再决定是否扩大实验。

## 2026-07-16：精简研究协作规范

### 研究目标

将 `AGENTS.md` 压缩为约 200–300 个中文字，只保留执行研究所需的核心规则。

### 核心改动

- 固定为“目标、项目结构、Environment、常用命令、Commit”五个章节。
- 保留 v2.2 基线、Conda、测试、训练、评测、Kubernetes 和 Git 规则。
- 删除重复解释和扩展说明。

### 实验配置

- 文档变更，不涉及模型或实验配置。

### 验证结果

- `AGENTS.md` 含 229 个中文字，符合 200–300 字要求。
- 全量测试通过：65 passed，2 个已知 Pydantic warning。
- `git diff --check` 通过。

### 结论与下一步

后续按精简规范执行研究任务，并持续结构化更新本文件。

## 2026-07-16：Kubernetes GPU 任务调度器

### 研究目标

在四卡服务器上通过统一命令提交 1–4 GPU 项目进程，由 Kubernetes 自动分配空闲 GPU，并在资源不足时排队。

### 核心改动

- 新增 GPU Job CLI、容器入口和 Conda 环境 Shell 包装。
- 支持 init、submit、list、logs、describe、wait、delete 和 server-side dry-run。
- 固定任务到 `4090-24gx4`，挂载项目、Hugging Face、pip 和持久 runtime。
- 使用依赖指纹隔离不可变 venv，避免并发任务删除或原地更新运行环境。
- `--no-bootstrap` 直接执行镜像命令，可用于 `nvidia-smi` 和完整自定义镜像。
- 增加中文使用文档和 23 个纯单元测试。

### 实验配置

- Namespace：`c2c-research`。
- 节点：`4090-24gx4`，4 × RTX 4090。
- 默认镜像：PyTorch 2.6.0、CUDA 12.4、Python 3.11 国内代理镜像。
- 默认资源：每卡 8 CPU、32Gi 内存，最长运行 72 小时。
- 本地验证环境：`c2c-py310-cu124`。

### 验证结果

- 新增调度器测试 23 passed；项目全量测试 88 passed，2 个已知 Pydantic warning。
- 1 卡与 4 卡 Job 均通过 Kubernetes server-side dry-run。
- 1-GPU `nvidia-smi` Job 成功调度到本机，容器仅看到 1 张 RTX 4090。
- 初版 `--no-bootstrap` 因基础镜像无 Python 失败，修复为直接执行 argv 后通过。
- 默认环境 Job 成功导入 `torch` 与 `rosetta`：CUDA 可用、可见 GPU 数为 1。
- 第二个默认环境 Job 输出“复用现有运行环境”，验证缓存有效。
- 所有测试 Job 已删除；`c2c-research` namespace 与 runtime 缓存保留。

### 结论

项目现可通过单条命令安全提交单卡、并行单卡或四卡任务；正式 GPU 工作负载应统一由 Kubernetes 管理，避免与宿主机直跑进程混用。

## 2026-07-16：Kubernetes Route-1 v2.2 实测

### 研究目标

使用现有 Route-1 v2.2 `token_mlp + entropy050` checkpoint 验证 Kubernetes 上的真实 C2C 评测闭环。

### 核心改动

- 调度器默认设置 `HF_ENDPOINT=https://hf-mirror.com` 和 600 秒下载超时。
- 将旧工作目录中的 v2.2 checkpoint 复制到当前项目 `local/checkpoints/`。
- 新增 AI2-ARC 4 样本 smoke 配置，实验细节记录于 `EXPERIMENT.md`。

### 实验配置

- Qwen3-0.6B Receiver + TinyLlama-1.1B Sharer。
- `soft_span_overlap_v2`、uniform、entropy050、`token_mlp`。
- AI2-ARC Challenge，limit 4，greedy generation，单张 RTX 4090。

### 验证结果

- 首次访问 Hugging Face 官方站失败；Pod 内 `hf-mirror.com` 返回 HTTP 200。
- 使用默认镜像站后，普通提交命令完成真实 v2.2 评测。
- 结果为 4/4、100%，平均输入 153.75 tokens，平均生成 7 tokens。
- Job 已删除，输出与 checkpoint 保留在 `local/`。

### 结论

集群已具备运行 C2C v2.2 的能力；本次只验证链路，不将 4 样本结果作为论文性能结论。

## 2026-07-16：统一 C2C 数据目录与 Kubernetes 挂载

### 研究目标

将 C2C 论文相关数据统一暴露到固定目录，使本地训练、评测和 Kubernetes Job 使用相同路径，避免重复下载和数据版本漂移。

### 核心改动

- 在 `/home/lijunsi/projects/KVcache/datasets/c2c` 补齐 7 个现有数据集软链接，保留原始物理文件。
- Kubernetes 将数据总目录只读挂载到 `/datasets`，设置 `C2C_DATA_ROOT=/datasets/c2c`。
- 新增集中数据加载器，按 dataset、config 和 split 精确读取本地 Parquet、Arrow、JSON 或 CSV，缺失时回退 Hugging Face。
- 训练适配器、统一评测器及 MMLU、OpenHermes、GSM8K 数据脚本统一接入该加载器。
- `init` 可幂等补齐软链接；C-Eval 未下载时不创建悬空链接。

### 实验配置

- 宿主机：`4090-24Gx4`。
- Namespace：`c2c-research`。
- Pod 数据根：`/datasets/c2c`，数据源只读。
- Smoke：1 GPU，加载 OpenBookQA `main/test` 与 LongBench-E `qasper_e/test`。

### 验证结果

- 7 个现有数据集软链接均可解析，无断链；C-Eval 明确为缺失。
- 本地加载 MMLU `auxiliary_train` 99,842 条、LongBench-E `qasper_e` 224 条。
- Route-1 v2.2 的 `MMLUChatDataset` 成功通过统一目录加载 2 条样本。
- Kubernetes 1 卡与 4 卡 Job 均通过 server-side dry-run。
- 真实 Pod 中 7 个链接全部可见，OpenBookQA 500 条、Qasper-E 224 条加载成功；测试 Job 已删除。
- 聚焦测试 31 passed；项目全量测试 96 passed，保留 2 个已知 Pydantic warning。

### 结论与下一步

本地与 Kubernetes 已共享统一数据入口，现有 Route-1 v2.2 无需修改 recipe 即可优先使用本地数据。后续下载 C-Eval 后重新执行 `gpu_job.sh init`，再补齐论文四项主评测。

## 2026-07-16：统一 C2C 模型目录与 Kubernetes 挂载

### 研究目标

让训练和评测任务直接复用已下载的 C2C 论文模型与官方 Fuser，避免重复下载和模型版本漂移。

### 核心改动

- 将宿主机 `/home/lijunsi/projects/KVcache/models/c2c` 只读挂载到 Pod `/models/c2c`。
- 设置 `C2C_MODEL_ROOT=/models/c2c`，`init` 和 `submit` 在模型根缺失时提前失败。
- 新增模型路径解析器，将 Hugging Face ID 或旧绝对路径映射到统一目录，不存在时回退原路径。
- SFT、Oracle、统一评测、数据 tokenizer 和多阶段加载入口接入本地优先解析。
- 公共底座及官方 Fuser 与项目 `local/checkpoints/` 实验产物继续分层保存。

### 实验配置

- Namespace：`c2c-research`。
- 节点：`4090-24gx4`，1 GPU smoke。
- 模型根：宿主机 `/home/lijunsi/projects/KVcache/models/c2c`，Pod `/models/c2c`。
- 验证对象：Qwen2.5-0.5B、Qwen3-0.6B 软链接、Qwen3-8B 路径解析和 C2C_Fuser。

### 验证结果

- 模型挂载聚焦测试 31 passed；项目全量测试 102 passed，保留 2 个已知 Pydantic warning。
- 1 卡与 4 卡 Job 均通过 Kubernetes server-side dry-run。
- 真实 Pod 中模型配置、Fuser 和跨挂载软链接均可读取，Qwen3-8B 本地路径解析正确。
- 模型目录写入返回 `EROFS`，只读约束生效；两个 smoke Job 均完成并已删除。

### 结论与下一步

统一模型库已同时供本地与 Kubernetes 使用，正式任务可继续在 recipe 中保留标准 Hugging Face ID。后续新增模型只需放入统一目录，并保持目录名与模型 ID basename 一致。

## 2026-07-17：Route-1 v2.2 可识别性实验与跨节点流水线

### 研究目标

在不引入新 transport、Route3、OT、RoPE correction 或新 loss 的前提下，分离验证 v2.2 的提升究竟来自多个 source candidates、entropy confidence、token/head gate，还是模块交互；同时建立可复现、可中断续跑的三条四卡跨节点实验流水线。

### 核心改动

- 为 soft alignment 增加 `native`、`constant`、`shuffle` 三种 confidence control；显式传递 `source_entropy` 与 override mask，保证常数和序列内打乱反事实不会从 source weights 旁路泄漏原 entropy。
- 增加 `B2-constant` 匹配控制，用相同 top-k=1 与常数 confidence 比较 B5，避免把静态缩放差异误判为 gate 容量；原始 B2→B5 仅作为带混杂的次要比较。
- 训练入口支持加载和校验冻结的 train/eval index manifest，并拒绝覆盖内容不一致的 split；seed 42 固定为 April v2.2 的历史 split，seeds 43、44 也分别冻结，保证同一 seed 的所有变体使用完全相同的数据成员与顺序。
- 新增 67-run identifiability suite 生成器与 lane runner：覆盖 B0–B6、B2-constant、B6-constant、B6-shuffle、三 seeds 和四个 Sharer pair；runner 串行执行单次训练及三任务评测，支持 reproduction/conditional gate、依赖检查、完整产物复用和断点续跑。
- 复用策略收紧为 checkpoint-only：只有逐位验证通过的 TinyLlama B6 seed 42 可复用，且必须由当前 evaluator 重新评测并生成新 diagnostics；历史 B3/B4 等因 split 不同，不进入主结果复用。
- 统一 evaluator 增加 per-example alignment diagnostics，并在多进程评测后校验所有 worker 的退出状态和返回结果，避免静默合并不完整样本。
- 新增统计报告器，输出每任务 accuracy、macro/weighted mean、正负迁移、三 seed mean±std、paired bootstrap 95% CI、McNemar、跨 seed/pair 聚合比较、alignment 分桶、confidence 相关性和 gate 统计；机制结论只有在对应置信区间不跨 0 时才标为支持。
- 新增跨节点 Kubernetes 基础设施：lane A 使用 `4090-24gx4` 的 4 卡，lane B/C 是 `4090-24gx8` 上两个独立 4 卡 Pod；三者通过 `/netdisk/lijunsi/c2c-route1-identifiability` 共享 workspace、模型、数据、checkpoint、状态与结果。
- 将五套所需模型全部准备到 `/netdisk`，同时记录关键文件和完整目录树 SHA256，覆盖 generation config、special token、vocab/merges 等运行时文件；四套固定数据也记录完整目录树 SHA256。三条 lane 统一从共享根加载模型与数据，节点本地 cache 只用于来源 provenance，不参与实际实验输入选择。
- stager 在发布 ready marker 前统一审计固定 HF revision、Python package 版本、五套共享模型和四套共享数据的完整目录树哈希；任一不一致都会阻止三条 lane 启动。Llama-3.2-1B 的 lane A affinity 只保留为确定性负载分配。
- 完整评测仅采集轻量 per-example gate 汇总；详细 K/V layer/head/early-middle-late/relative-token 统计由 checkpoint 后单 GPU、batch size 1 的固定样本诊断在线聚合，禁止落盘 raw gate tensor。
- 增加无 `git` 基础镜像的受控 fallback：控制节点可将目标 commit 的 clean detached checkout 预置到 `/netdisk`；checkout init 只有在 commit-specific marker 存在且共享 workspace 的 `.git/HEAD` 精确匹配目标 SHA 时才跳过 clone，其余审计与计划生成保持不变。
- 移除所有容器的深层 `/netdisk/...` OCI `workingDir`；命令保持绝对路径，并由入口脚本在共享 volume 挂载完成后执行 `chdir(project_root)`，兼容 autofs 根挂载。
- 新增 tracked runtime pip constraints，精确固定审计合同中的 `transformers`、`datasets`、`accelerate`、`wandb` 与 `peft`；constraints 内容哈希进入共享 venv fingerprint，避免宽松 optional dependency 将 `wandb` 解到 0.28.1 后仍复用该环境。
- suite revision resolver 在 `git` 可用时保持 `git rev-parse HEAD`；无 `git` 镜像只允许读取预置 checkout 的 40 位 detached `.git/HEAD`，且 stage-plans 继续核对生成 manifest 与请求 commit 完全相同。

### 实验配置

- Receiver：Qwen3-0.6B。
- Sharer：TinyLlama-1.1B、Qwen3-1.7B、Qwen2.5-0.5B、Llama3.2-1B。
- 训练：MMLU auxiliary 2,048 条；seeds 42、43、44；所有变体保持相同 epoch、batch、学习率、fuser 和 checkpoint 选择规则。
- 评测：MMLU-Redux、ARC、OpenBookQA，均由当前 evaluator 重新运行并保存 per-example prediction。
- 方法：B0、B1、B2、B2-constant、B3、B4、B5、B6、B6-constant、B6-shuffle。
- Suite：67 个评测 run、66 个名义训练 run、67 组三任务评测；其中 B0 无训练，TinyLlama B6 seed 42 使用严格校验后的 checkpoint-only reuse。
- Lane 计划：phase1 为 37 runs，A/B/C 分别 12/13/12；方向一致后释放的 conditional 阶段为 30 runs，每条 lane 10 个。
- 冻结 split：`recipe/train_recipe/identifiability/splits/`；B6 复用声明：`recipe/train_recipe/identifiability/reuse_step1_b6.json`。

### 验证结果

- 第一次使用独立 `Generator(seed)` 的 seeded split 复训未通过 reproduction gate：历史 checkpoint 在当前 evaluator 上 macro 为 50.8058%，新 checkpoint 为 49.7897%，macro 下降 1.0161 个百分点；其中 MMLU-Redux 下降 2.4221 个百分点，同时超过预注册的 macro 1.0pp 与单任务 2.0pp 阈值，因此下游流水线被停止。
- 根因定位为数据 split 漂移：April v2.2 在 `token_mlp` projector 初始化之后使用进程全局 Torch RNG 调用 `random_split`，后来改成独立 seeded generator 后同时改变了 train/eval 成员和样本顺序，并非 v2.2 算法本身不可复现。
- 使用 `legacy_global_rng` 捕获并冻结 April split 后重新训练，64 个 optimizer steps 的训练轨迹逐步、逐位一致。
- 历史与复现 checkpoint 的 28 层 projector 共 1,148 个 tensors、485,647,428 个参数全部 `torch.equal`；两个目录 SHA256 均为 `a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404`，最终 reproduction gate 通过。
- 24gx8 已验证可直接读写 `/netdisk`；共享盘中已准备四项固定数据、五个所需模型和 bitwise-verified B6 checkpoint。共享模型与共享数据的完整目录树哈希均已冻结，lane 启动前仍会执行 revision、文件、数据和环境审计。
- 首次 stager 现场验证发现 PyTorch runtime 镜像没有 `git`（checkout init exit 127），未进入资产审计或训练；已改用上述 exact-HEAD 共享预置路径，避免在计算节点临时安装软件。
- exact-HEAD checkout 随后通过，但 audit init 在 OCI 启动阶段因深层 `/netdisk` workingDir 创建顺序触发 permission denied；同样未运行审计或训练，已按上述 post-mount chdir 方式修复。
- post-mount chdir 修复后 runtime bootstrap 开始正常执行；在进入资产审计前观察到宽松 `wandb>=0.13` 会解析为 0.28.1，与固定合同 0.28.0 不符，因此主动停止该 stager 并加入 constraints，未产生训练任务。
- constraints 版本的 runtime 构建及模型/数据/package 审计均已通过；主容器随后在 suite revision 解析处再次因镜像无 `git` 退出，尚未发布 workspace-ready marker 或训练任务，已加入 detached-HEAD fallback。
- 生成清单确认 67 runs、三条四卡 lane 和两阶段 gate/依赖关系一致；本条记录不将调度清单误写为已完成实验。

### 结论与下一步

复现门控已通过，且历史 v2.2 的偏差已被确认来自 split 实现变化并通过冻结 index 消除。整套组件实验与跨模型验证任务状态记为“运行中”，目前不能回答 soft candidates、entropy 或 token/head gate 谁是主要来源，也不能据此进入下一阶段。第一阶段最终判断只使用下游 accuracy、配对迁移和统计检验；train/eval loss 仅作优化诊断，尤其不得因 learned-affine 的 eval loss 更低而宣称机制更优。

## 2026-07-17：异构卡数流水线与共享 NFS 稳定性修复

### 研究目标

在不改变方法、学习率、epoch、数据 split、checkpoint 选择规则及有效全局 batch 的前提下，把空闲的 `2×48GB` 节点和 `24gx8` 的健康 GPU 纳入第一阶段流水线，并消除并发 lane 写共享状态目录时的 NFS 竞态。

### 核心改动

- 新增两卡 lane adapter：把 canonical `1×4 GPUs×8 accumulation=32` 显式转换为 `1×2 GPUs×16 accumulation=32`，训练 recipe、checkpoint provenance 和硬件 profile 均记录实际值，不在运行时隐式伪装原配置。
- 两卡评测 profile 改为 ARC 使用 GPU 0、OpenBookQA 使用 GPU 1 并发；二者结束后 MMLU-Redux 使用 GPU `[0,1]`，避免四卡布局 `[0] / [1] / [2,3]` 在两卡 Pod 中越界。
- 作业启动前读取两张可见 GPU 的已用显存；任一卡超过 4,096 MiB 即拒绝启动，避免 `4090-24gx8` 上未被 Kubernetes 记录的高占用 GPU 混入正式训练。
- suite 的共享目录创建增加窄范围重试：仅吞掉“并发写者已成功创建且路径确为目录”的 `FileExistsError`，真实文件碰撞仍直接失败。
- 所有 lane 的 `lane_state/completed`、`lane_state/failed`、checkpoint、评测和 post-hoc 输出目录在运行前预创建，降低 NFS 元数据竞态。

### 实验配置

- Lane A：`4090-24gx4`，4×24GB，`num_processes=4`、gradient accumulation 8。
- Lane B：`4090-24gx8`，2×24GB，`num_processes=2`、gradient accumulation 16。
- Lane C：`4090-48gx2`，2×48GB，`num_processes=2`、gradient accumulation 16。
- 三条 lane 的 per-device batch 均为 1，有效全局 batch 均为 32；两卡 DDP 与四卡 DDP 不宣称 bitwise 等价，world size 和硬件 profile 进入运行 provenance。

### 验证结果

- 两卡 adapter 聚焦测试与 suite NFS 竞态测试共 25 passed。
- 两份 Kubernetes Job 均通过 server-side dry-run。
- Lane B 实际分配的两张 GPU 启动显存均为 1 MiB，成功复用已完成的 B0 评测，并以两进程进入 TinyLlama B2 seed 42 训练。
- Lane C 首次拉取固定 PyTorch runtime 镜像后，补充了共享盘 supplemental GID 31000，并将不可写的节点本地 HF cache 替换为 Pod `emptyDir`；两张启动卡均为 1 MiB，已进入 TinyLlama B1 seed 42 两进程训练，显存约 10.7/11.7 GiB。

### 结论与下一步

当前三条物理流水线均已进入真实训练且无 Pod 重启。卡数变化只用于基础设施等效批量调度，不构成新方法；最终统计必须保留 world-size 字段，并在解释很小的 seed 差异时把非 bitwise 的 DDP 轨迹作为潜在工程噪声披露。

## 2026-07-17：Phase1 动态重分片到五条长期流水线

### 研究目标

在不终止已接近完成的 B2/B1 run、不重复任何正式输出的前提下，利用 `4090-24gx8` 后续恢复为空闲状态的全部 GPU，把 B/C 剩余串行计划拆成四个互斥 shard，降低 Phase1 的关键路径。

### 核心改动

- 新增 deterministic lane sharder：读取多个 phase1 plan，排除 completed 与显式 reserved runs，清除仅用于旧 lane 串行化的 `depends_on_runs`，按 pair 规模与 post-hoc gate 成本做 LPT 负载均衡。
- 当前 B2 seed 42 与 B1 seed 42 继续由旧 B/C worker 完整执行；旧 worker 使用独立 pending gate，在当前 run 写完 checkpoint、三任务评测和 completion marker 后退出。
- A 使用新的 pass gate 立即恢复 canonical 4 卡计划；B/C 的其余 22 runs 被拆为 4 个互斥 shard，estimated weights 为 6.60、5.65、5.65、6.55。
- `4090-24gx8` 同时运行三个 2 卡 shard；`4090-48gx2` 的第四个 shard 在旧 C worker 完成 B1 后自动获得两张卡。长期并行度由三条提升为五条。
- 两卡 adapter 现在允许 Pod 看到多于两张卡时按 UUID 选择两张低占用卡，并把实际 UUID/启动显存写入 allocation provenance；post-hoc 单卡 diagnostics 命令保持 passthrough，不再被误判为 torchrun 命令。

### 实验配置

- Lane A：`r1id-v22-9b06-lane-a-resume4`，`4090-24gx4`，4-process DDP。
- B/C shards 1–3：`4090-24gx8`，各 2-process DDP、per-device batch 1、accumulation 16。
- B/C shard 4：`4090-48gx2`，2-process DDP；旧 C 完成后由 Kubernetes 自动调度。
- 所有 shard 共享同一 reproduction-pass ConfigMap、冻结 split、输出根与 completion state；run IDs、checkpoint 和评测目录在 shard 之间不重叠。

### 验证结果

- 三个 `24gx8` shard 均完成 GPU 准入并进入不同方法训练；启动检查未发现高占用卡。
- A resume 跳过三个 completed runs 后进入 TinyLlama B6 seed 44 训练。
- Sharder 与 adapter 聚焦测试 5 passed；项目全量测试 197 passed，保留 2 个已知 Pydantic warnings。

### 结论与下一步

Phase1 的长期并行线提升为五条，预计关键路径从约 14–18 小时缩短到约 7–10 小时。该变化只重排互相独立的 run，不改变任何方法或单个 run 的有效全局 batch、数据、超参数与 checkpoint 规则。

## 2026-07-17：吃满 14 张 NVIDIA GPU 的七 worker 终态

### 研究目标

在五线方案基础上继续利用 `4090-24gx4` 可拆成两个 2 卡 worker、`4090-24gx8` 可容纳四个 2 卡 worker 的事实，把 Phase1 剩余关键路径压到当前硬件的物理上限。

### 核心改动

- 保留 6 个已经在运行的 reserved runs：B6 seed 44、B2 seed 42、B1 seed 42、B3 seed 42、B2-constant seed 43、B5 seed 42；这些 worker 在各自 run 完整落盘后由 pending gate 退出。
- 将其余 27 个 runs 从 A/B/C 三份原始计划统一重分为 7 个互斥 worker，estimated weights 为 4.55、3.60、4.50、4.50、4.45、4.45、4.45。
- `4090-24gx4` 部署两个 2 卡 worker，`4090-24gx8` 部署四个 2 卡 worker，`4090-48gx2` 部署一个 2 卡 worker；总计使用全部 14 张 NVIDIA GPU。
- 七个终态 Job 使用独立 reproduction-pass gate，已提前创建为 Pending；旧 worker 释放对应节点资源后由 Kubernetes 自动填充，无人工切换窗口。

### 验证结果

- 七份 Job 均通过 server-side dry-run 并已创建；当前因旧 reserved workers 占满节点而处于 Pending，资源释放后自动调度。
- Max7 shard plan SHA256 和输入 Job manifests 已保存到 `/netdisk/lijunsi/c2c-route1-identifiability/status/job-manifests/max7-phase1/`。
- 全量测试仍为 197 passed、2 个已知 Pydantic warnings；终态只改变计划分配和 Kubernetes placement。

### 结论与下一步

七 worker 是当前 CUDA 代码不迁移 Ascend 的安全并行上限。Phase1 新 ETA 为约 6–7 小时；若 conditional 30 runs 通过方向门控后立即采用同一七 worker 布局，全部 67 runs 预计约 12–16 小时完成。

## 2026-07-18：Phase1 完成并释放七路 conditional 多 seed 验证

### 研究目标

在 Phase1 完整落盘后按预注册规则检查 seed 42 的跨模型方向；若方向满足要求，立即把 30 个跨模型 seeds 43/44 runs 分到全部 14 张 NVIDIA GPU，不改变任何方法、训练超参数或统计规则。

### 核心改动

- deterministic lane sharder 从仅支持 `phase1` 扩展为同时支持既有 `conditional` phase；所有输入计划必须具有同一合法 phase，输出文件名和 manifest 显式记录该 phase。
- 两卡 adapter 同样接受命名的 `phase1` 或 `conditional` lane plan，继续保持 `4 processes × accumulation 8` 到 `2 processes × accumulation 16` 的有效全局 batch 32 等价转换。
- 30 个 conditional runs 被确定性分成 7 个互斥 worker，每路 4–5 个 run；`4090-24gx4` 部署两路、`4090-24gx8` 部署四路、`4090-48gx2` 部署一路。
- conditional gate 在 seed-42 方向筛选通过后从 pending 切换为 pass；七个正式 Jobs 使用新 adapter SHA256 `e796f4df99e362cfd83e5510b92955a06fac0b08bcd2aace1921ceb7607b9416`。

### 实验配置

- Phase1 计划中的 37 个 runs 全部完成，completion state 中实验失败为 0；七个 Max7 Jobs 均为 Kubernetes Complete，最长 wall time 为 7 小时 8 分。
- Phase1 报告使用当前 37-run 产物并允许 30 个条件项缺失，生成 126 个 task-level rows；B6 seed 42 的 TinyLlama macro mean 为 50.806%，复现目标 50.82%。
- seed 42 的 sample-weighted 方向：B6−B2 在 4/4 模型对为正；B6−B5 在 3/4 模型对为正，仅 Llama3.2 pair 为 −0.509 percentage points。因此满足“方向一致后补 seeds 43/44”的筛选条件，但最终进入下一阶段仍取决于完整 paired CI。
- Conditional shard estimated weights 为 5.75、5.75、4.80、4.80、4.90、4.90、4.90；plan 与 Job manifests 保存到 `/netdisk/lijunsi/c2c-route1-identifiability/status/job-manifests/max7-conditional/`。

### 验证结果

- Sharder/adapter 聚焦测试 7 passed；项目全量测试 199 passed，保留 2 个已知 Pydantic warnings。
- 第一批 Job 在训练前暴露 adapter 的 phase1-only 校验并立即退出；没有写 completion marker，未产生可复用 partial checkpoint。扩展校验并执行 prepare-only 后，以新 Job 名重新提交。
- 七个正式 conditional Pods 均完成 GPU 启动显存检查、使用共享模型/数据，并以两进程进入首个 run 的真实训练；Pod restart 均为 0。

### 结论与下一步

Phase1 的筛选门控已通过，conditional 30 runs 正在使用全部 14 张 NVIDIA GPU。当前组件证据仍是 provisional：TinyLlama 三 seed 的 B3−B2 与 B4−B3 cluster CI 跨 0，而 B6 相对 B4/B5 为正；entropy constant/shuffle 反事实也显示 full B6 更好。最终结论必须等待跨模型 seeds 43/44 完成后重新生成完整报告与预注册的聚合 paired 95% CI。

## 2026-07-18：67-run identifiability 最终报告与阶段门控结论

### 研究目标

汇总全部四个模型对、三个 seeds 和三个开发任务的 per-example 结果，严格回答 multiple candidates、entropy confidence 与 token/head gate 的独立贡献，并按预注册规则决定是否进入下一阶段。

### 核心改动

- 修复 analysis materializer 的完成状态推断：当所有标记为 `conditional` 的 run/task 均已严格解析时，`conditional_complete` 自动为 true；Phase1 的 `--allow-missing` 报告仍保持 false。
- 最终报告物化 234 个 pair/method/seed/task rows，输出 task/aggregate accuracy、transfer、paired comparison、bucket、correlation、post-hoc gate、cluster bootstrap 与 final gate 等 CSV/JSON/Markdown 产物。
- 新增中文机制汇总 `local/final_results/route1_identifiability/rev_9b06d173eada/final_report/MECHANISM_SUMMARY_ZH.md`，只总结预注册实验，不引入新方法。

### 验证结果

- 67/67 runs 完成，failed marker 为 0；report contract 的四个 pairs、三个 seeds 和三个固定任务行数全部完整。
- Soft candidates：跨 pair B3−B2 `+1.29 pp`，95% cluster CI `[-0.61,+3.58]`，仅 Qwen3-1.7B pair 显著为正。
- Static entropy：TinyLlama B4−B3 `+1.29 pp`，CI `[-2.78,+5.98]`；但 seed-42 反事实 B6−constant `+0.94 pp`、B6−shuffle `+2.28 pp` 的 CI 均高于 0。
- Gate capacity：干净的 TinyLlama B5−B2-constant `+0.93 pp`，CI `[-2.66,+3.90]`；跨 pair confounded B5−B2 `+0.35 pp`，CI `[-1.15,+2.13]`。
- 完整 B6 gate 在 Qwen3/Qwen2.5 上约 99.93% high-saturated，Llama3.2 为 100%，TinyLlama 约 85.86%；B5 有一定动态变化，但没有稳定 accuracy 贡献。
- Final gate：B6−B2 在 3/4 pairs 为正但 cluster CI `[-1.14,+4.05]` 跨 0；B6−B5 仅 2/4 pairs 为正且 CI `[-0.92,+3.31]` 跨 0。combined gate 为 fail。
- Materializer/report 聚焦测试 33 passed；完整项目测试 199 passed，保留 2 个已知 Pydantic warnings。

### 结论与下一步

第一阶段没有支持一个跨模型统一的 v2.2 机制解释。Multiple candidates 只在特定 pair 显著；entropy 在 TinyLlama 反事实中确有信息，但 static entropy 的普适独立贡献未成立；token/head gate 的额外自适应容量未获得支持，完整 B6 反而接近 always-on 饱和。按预注册标准，B6 未稳定优于 B2 与 B5，不进入下一阶段，也不开发新 transport、Route3、OT、RoPE correction 或新 loss。

## 2026-07-18：最终实验报告发布到仓库根目录

### 研究目标

让 GitHub 页面无需访问集群 `local/` 产物即可直接审阅第一阶段的完整统计报告与中文机制结论。

### 核心改动

- 将最终 992 行完整报告发布为根目录 `ROUTE1_V22_IDENTIFIABILITY_REPORT.md`。
- 将三问题结论、关键三 seed 表格和最终门控发布为根目录 `ROUTE1_V22_IDENTIFIABILITY_SUMMARY_ZH.md`。
- 在根目录 `EXPERIMENT.md` 增加两个报告链接；逐例 CSV、bootstrap 中间表和大体积 JSON 仍遵守 `local/` 不提交规则。

### 验证结果

- 两份根目录 Markdown 与最终 `local/final_results/.../final_report/` 来源逐字一致，合计约 92KB。
- 本次只发布实验文档，不改变训练、评测或统计代码。

### 结论

第一阶段实验记录现可从 GitHub 仓库根目录直接访问，同时避免把大体积逐例产物纳入版本控制。

## 2026-07-18：Phase 1.5 同 checkpoint 因果干预与节点级 GPU 池调度

### 研究目标

在不训练新方法、不修改 checkpoint、也不进入 query-time transport 的前提下，对 Phase 1 的 B2、B3、B6 权重执行推理期交叉干预，分离 train/eval top-k、entropy 数值与位置、learned confidence gate 和 legacy scalar K/V gate 的贡献。

### 核心改动

- 新增统一的 evaluation-only intervention 配置与 provenance：支持 `top_k=1/4`、native/constant/shuffled entropy，以及 learned/static/forced-on gate 视图；重复应用配置保持幂等。
- `forced_on` 同时把 alignment-confidence gate 与 checkpoint 中既有的 scalar K/V gate 置为 1；另提供只用于 Qwen2.5 seed 44 异常拆分的 `alignment_forced_on`、`legacy_forced_on`，不进入主矩阵。
- 新增 Phase 1.5 manifest 生成与断点续跑入口。主矩阵为 4 pairs × 3 seeds × 6 非 native 干预，共 72 个三任务 triplets；36 个 native comparator 直接复用 Phase 1 逐例结果。
- 保留七个逻辑双卡 shards，但改由三个整节点 Kubernetes Jobs 管理：x4 请求 4 卡并串行覆盖 shards 0/1，x8 请求 8 卡、最多三个两卡组并行覆盖 shards 2–5，x48 请求 2 卡覆盖 shard 6。运行时按 `nvidia-smi` 实际显存过滤 Kubernetes 不可见的 busy GPU，再用空闲 UUID 两两分组。
- evaluator 仅在显式 `C2C_PRESERVE_CUDA_VISIBLE_DEVICES=1` 时保留调度器设置的 UUID mask；默认行为不变。这样同一 x8 Pod 内的三个 shard 能在互斥 GPU 对上安全并行。
- Kubernetes init 同时核验精确 Git commit 与 execution manifest SHA；当基础镜像没有 `git` 时，允许读取 detached checkout 的 40 位 `.git/HEAD`，避免 init 阶段误阻塞。
- Job 显式传入 Phase 1 已审计的 `runtime_constraints.txt`，运行时 fingerprint 与既有固定 venv 保持一致，禁止因漏传 constraints 而安装漂移依赖。
- Manifest 生成时把全部结果目录解析为绝对路径并串行预创建，避免 ARC/OBQA 与多 shard 在共享 NFS 上递归创建共同父目录时触发 `FileExistsError`。
- 每个 node-pool launcher 在启动并行 evaluator 前，还会从本节点串行预创建其负责的结果树，并对 autofs/NFS 的短暂 `ENOENT/EEXIST` 不一致做有界重试。
- x8 的 autofs 客户端对由其他节点首次创建的目录名存在持续 negative dentry；因此 shards 2–5 使用由 x8 自己创建的 sibling 共享结果根，execution manifest 继续统一记录绝对路径，统计接口无需变化。
- 新增 Phase 1.5 统计入口，复用 pair→seed→example hierarchical paired bootstrap，输出同 checkpoint accuracy delta、McNemar、seed 方差、ambiguity interaction 与 receiver/fused oracle abstention headroom。

### 实验配置

- 固定四个 sender→Qwen3-0.6B 模型对、seeds 42/43/44 和完整 MMLU-Redux、ARC、OpenBookQA dev 集。
- train-k × eval-k：B2 checkpoint 在 eval-k1/k4、B3 checkpoint 在 eval-k1/k4 形成 2×2 对照。
- B6 checkpoint：native/constant-0.93/shuffled entropy，以及 learned/static/forced-on gate。
- Qwen2.5 B6 seed 44 额外执行两个 gate-component isolation triplets；它们与主 72 矩阵分开记录。
- 大体积配置、逐例 prediction、状态和统计中间文件全部保留在 `local/` 或共享 `/netdisk`，不提交 Git。

### 验证结果

- 项目全量测试 `222 passed`，保留 2 个已知 Pydantic warnings。
- 72-run execution manifest 完整性、输出目录隔离、checkpoint-only 约束和七 shard 数量 `[11,11,10,10,10,10,10]` 均通过校验。
- 三份节点级 Kubernetes Jobs 已通过 API server dry-run；逻辑 shard 数和 run 分布仍为 `[11,11,10,10,10,10,10]`。
- 首次真实创建在进入评测前暴露了 `PIP_CONSTRAINT` 漏传：两个跨节点 Pod 清理同一未发布 venv 时发生竞态，其余 Pod 尚在等待/安装。七个 Job 随即全部删除，没有产生 prediction；补齐 constraints 后将复用已审计环境重新提交。

### 结论与下一步

推理期因果诊断基础设施已经就绪。正式实验必须基于本次实现的最终 commit 重新生成共享 manifest 与三节点 Job YAML，再启动最多五个同时运行的双卡逻辑 shard；是否补 TinyLlama constant/shuffle seeds 43/44 训练，严格取决于同 checkpoint entropy 干预是否跨模型对稳定有效。
