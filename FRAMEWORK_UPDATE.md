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

- 项目全量测试 `223 passed`，保留 2 个已知 Pydantic warnings。
- 统计 CLI 支持可选 `--anomaly-manifest`，严格合并 Qwen2.5 B6 seed 44 的 alignment-only 与 legacy-only forced-on 结果；原八项主对照保持不变，异常两项同时进入 paired、oracle 与层级统计。
- 72-run execution manifest 完整性、输出目录隔离、checkpoint-only 约束和七 shard 数量 `[11,11,10,10,10,10,10]` 均通过校验。
- 三份节点级 Kubernetes Jobs 已通过 API server dry-run；逻辑 shard 数和 run 分布仍为 `[11,11,10,10,10,10,10]`。
- 首次真实创建在进入评测前暴露了 `PIP_CONSTRAINT` 漏传：两个跨节点 Pod 清理同一未发布 venv 时发生竞态，其余 Pod 尚在等待/安装。七个 Job 随即全部删除，没有产生 prediction；补齐 constraints 后将复用已审计环境重新提交。

### 结论与下一步

推理期因果诊断基础设施已经就绪。正式实验必须基于本次实现的最终 commit 重新生成共享 manifest 与三节点 Job YAML，再启动最多五个同时运行的双卡逻辑 shard；是否补 TinyLlama constant/shuffle seeds 43/44 训练，严格取决于同 checkpoint entropy 干预是否跨模型对稳定有效。

## 2026-07-18：Phase 1.5 单卡小任务预取与安全并发

### 研究目标

利用 `4090-24gx4` 上无法再组成双卡组、但显存足够承载完整 sender/receiver/projector 的一张空闲 24GB GPU，提前完成后续 triplet 的 ARC 与 OpenBookQA，缩短双卡主 lane 的关键路径；不改变 checkpoint、干预配置、数据、预测逻辑或 MMLU 两卡分片。

### 核心改动

- `route1_phase15_interventions.py` 新增 `stage-small-benchmarks` CLI，按逻辑 shard 串行预取尚未完成的 ARC/OpenBookQA。
- 预取器直接使用 manifest 中原始 YAML：ARC 保持 `gpu_ids=[0]`，OpenBookQA 保持 `gpu_ids=[1]`；只通过进程级 `CUDA_VISIBLE_DEVICES` 把两者映射到同一张 spare GPU，因此不重写 config，也不改变 intervention provenance。
- 新增 shard 级非阻塞锁和 run 级 NFS advisory lock；新版主 `run-shard` 在同一 run lock 内重新检查完成状态，避免主 lane 与预取器写入相同 triplet。
- MMLU provenance 已出现时默认安全跳过，亦可配置为立即报错；ARC/OpenBookQA 存在 partial/active artifacts 时拒绝重复启动。
- 状态以原子 JSON 记录 manifest SHA、GPU mask、逐 run/dataset 时间、返回码和最终输出契约；每个已完成数据集仍严格要求唯一 prediction CSV、summary JSON 与 provenance。
- 节点级 Job renderer 支持通过 `C2C_PHASE15_WORKSPACE_ROOT` 指向独立、精确 commit 的 resume checkout，并把该路径传入 Pod；因此运行中的 canonical checkout 无需被原地切换或修改。

### 实验配置

- 单卡显存审计：TinyLlama pair 活跃 evaluator 约占 5.8–6.1GiB；最重 Qwen3-1.7B pair 约占 6.6–7.0GiB，均明显低于 24.56GiB。
- 双卡并非模型并行要求：每个 evaluator 进程在单卡完整加载 sender、Qwen3-0.6B receiver 与 28 层 projector；MMLU 使用两卡仅用于 subject 数据并行。
- x4 预取计划只覆盖尚未由主 lane 进入的 shard 1；主 lane 继续用两卡执行 MMLU，外部高占用 GPU 不参与实验。

### 验证结果

- 新增测试覆盖原始配置不变、ARC/OBQA 设备映射、MMLU-started skip/error、run lock 互斥、锁内完成状态重查和输出契约失败。
- 预取器聚焦测试 `15 passed`；加入独立 resume checkout 后项目全量测试更新为 `229 passed`，保留 2 个既有 Pydantic warnings；`git diff --check` 通过。

### 结论与下一步

该改动只优化同一评测矩阵的调度，不引入新方法或统计口径。按历史 ARC、OpenBookQA 与 MMLU 耗时，x4 节点的单 triplet 关键路径预计由约 32 分钟降至约 23 分钟；正式启用时必须在旧 commit 主 lane 完成当前 triplet 后 resume-safe 切换到新版，使主 lane 与预取器共同遵守 run lock。

## 2026-07-19：Phase 1.5 x8 失联恢复与机会式 work stealing

### 研究目标

x8 节点失联后，把 shards 2–5 的未完成工作安全迁移到已释放的 x4/x48 节点，并允许两个恢复 worker 动态分担 shard 5 的 10 个 full triplets；不改变 checkpoint、干预方法、评测配置、逐例输出或统计口径。

### 核心改动

- `route1_phase15_interventions.py` 新增独立 `run-shard-opportunistic` CLI；原 `run-shard` 的阻塞串行语义与默认 Job 行为保持完全不变。
- 机会式 runner 按 manifest 顺序扫描指定逻辑 shard，对每个 incomplete run 使用既有 per-run NFS advisory lock 的非阻塞获取；锁忙时记录并跳到后续 run，使同一 shard 的两个 worker 能安全分担不同 triplet。
- 每次成功取得锁后重新检查完整性，防止 sibling worker 刚完成的 triplet 被重复执行；evaluator 返回 0 后仍要求三个任务全部满足唯一 prediction CSV、summary JSON 和 provenance 的既有输出契约。
- evaluator 非零或成功返回但输出契约不完整时立即停止并传播失败；若整轮没有可取得的 run，按 `--idle-poll-seconds` 休眠后重扫，该间隔被限制在 `(0, 60]` 秒，避免 busy spin 或过长失联。
- 日志逐轮报告 complete/incomplete/locked/remaining 状态，并逐 run 标记 deferred、starting、complete 或 failure，便于 Kubernetes logs 直接审计 work stealing 过程。
- `route1_phase15_jobs.py` 新增同名显式恢复入口：核验 immutable manifest SHA，预创建指定 shard 的 NFS 输出树，从 `nvidia-smi` 过滤物理高占用卡并选择一对空闲 UUID，再以 `C2C_PRESERVE_CUDA_VISIBLE_DEVICES=1` 调用机会式 runner。
- 节点入口继续原子记录所选 GPU UUID、启动显存、run ids、poll interval 和退出码；默认 `run-shard`/`run-node` 命令和三节点 renderer 均未改为机会式模式。

### 实验配置

- 失联时审计结果：shards 2–5 共 120 个 dataset outputs，其中 84 个完整、3 个 MMLU provenance-only partial、33 个完全未启动；剩余等价于 36 个 dataset eval，涉及 14 个 incomplete triplets。shards 0/1/6 和 anomaly 已完成。
- 旧 x8 Job 运行 commit `2b0d6a2`，尚未使用 per-run lock；因此恢复的安全前提是 Kubernetes 已确认旧 Job `FailureTarget`/failed、x8 worker 不再写入。不能在旧 x8 worker 仍活跃时直接 work-steal。
- shards 2–4 先 resume 三个 MMLU-only partial 和一个 full triplet；其后 x4 与 x48 使用同一 manifest SHA `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1` 同时运行 shard 5 的机会式 worker。
- 节点内入口示例：`python script/k8s/route1_phase15_jobs.py run-shard-opportunistic --execution-manifest <phase1.5-manifest.json> --expected-manifest-sha256 424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1 --shard-index 5 --num-shards 7 --state-dir <node-unique-state-dir> --max-startup-used-mib 4096 --idle-poll-seconds 5`。
- x4/x48 必须使用不同 state directories，但共享同一 manifest、结果目录和 run locks；任一 worker 遇到 sibling 正在执行的 run 会跳到后续 incomplete run，全部完成后均可 resume-safe 退出。

### 验证结果

- 新增测试覆盖：run 0 锁忙时先执行 run 1、锁释放后下一轮回补 run 0、整 shard 已完成时零评测/零等待退出、整轮锁忙时按受限间隔轮询，以及 evaluator 非零时不启动后续 run 并原样传播返回码。
- K8s 入口测试额外覆盖 hidden-busy GPU 过滤、空闲双 UUID 选择、poll 参数转发、CUDA mask 保留、NFS 结果树预创建、机会式状态记录、CLI 路由和非法 poll interval 在 GPU 查询前失败。
- Phase 1.5 调度相关测试 `30 passed`；项目全量测试 `236 passed`，保留 2 个既有 Pydantic warnings；`git diff --check` 和两个修改脚本的 `py_compile` 均通过。

### 结论

设计时预期该改动可把 x8 节点故障转换为 resume-safe 的跨节点尾部恢复，不增加实验样本、改变方法或改变统计比较；其中“共同 per-run NFS lock 可安全协调多个 Pod”的预期已被下述真实运行更正所否定。

### 运行时更正

真实跨节点执行否定了上述 NFS 锁安全假设：当前共享路径上的 advisory `flock` 没有在多个 Pod 间形成可靠互斥，两个 shard-5 worker 对 5 个 dataset 产生了重复 bundle。逐例和诊断文件审计证明科学结果一致，仅 latency 与时间戳文件名不同；重复 bundle 已完整隔离到 `local/tmp`，不提交 Git、不进入统计。

因此，`run-shard-opportunistic` 仅保留为已测试的执行原型，不再宣称适用于当前 NFS 的跨 Pod work stealing。Phase 1.5 最终尾部使用显式、互不重叠的一 run 一 Job 分配：init 校验固定 commit、manifest SHA、run index/id、配置存在和输出未启动，容器固定物理 GPU UUID；最终统计前再执行 exactly-one CSV/summary/provenance/gate/length 的严格审计。该更正不改变任何研究方法、checkpoint、逐例预测或统计协议。

## 2026-07-19：Phase 1.5 因果诊断收口与机制主张收缩

### 研究目标

在不新增 transport、router、gate 或 loss 的前提下，用同 checkpoint 推理干预区分 inference-time candidate 数、entropy、learned token/head modulation 与 checkpoint/training-regime、legacy scalar mask、模型对兼容性的作用，并按预注册门槛决定是否进入 query-time prototype。

### 核心改动

- 完成 72 个主 triplets 与 2 个 Qwen2.5 seed44 anomaly triplets 的最终输出合同，统计脚本统一生成 paired accuracy、pair-balanced hierarchical CI、seed 方差、ambiguity interaction 与 oracle abstention。
- 最终严格审计新增 current-config SHA 与产物 provenance 的字节级对应检查，发现并修复 TinyLlama B2 eval-k4 seed43 的旧 manifest provenance 漂移；旧 bundle 保留在 `local/tmp`，最终 config 显式重跑不改变任何科学字段。
- `route1_phase15_interventions.py` 与 `route1_phase15_jobs.py` 的 opportunistic CLI 说明改为明确警告：当前共享 NFS 的 advisory lock 不能作为跨 Pod mutex，生产恢复必须显式分配互不重叠的 run ids。
- 根目录新增英文完整报告与中文机制摘要；大体积逐例和统计文件继续只保留在 `local/` 或 `/netdisk`。

### 实验配置

- 四个 sender→Qwen3-0.6B 模型对、seeds 42/43/44、MMLU-Redux/ARC/OpenBookQA 完整开发集。
- 同 checkpoint 干预：B2 eval-k4、B3 eval-k1、B6 constant/shuffled entropy、static/forced-on gate；Qwen2.5 seed44 额外拆分 alignment-confidence 与 legacy scalar forced-on。
- 主 manifest SHA `424d0468a624fee6cd31932bf3795fa42b98bf20f9563ebf84a0afaca5605dd1`；anomaly manifest SHA `bd305268e9a8527cb75407293b49cae4e577bb10516e9643781573e861cfa5d2`。
- 层级 bootstrap 5,000 次、95% CI、seed `20260718`；Kubernetes 与本地 Conda 独立运行复核。

### 验证结果

- 严格产物审计：74/74 runs、222/222 datasets；x8 120、main 96、anomaly 6，全部 exactly-one CSV/summary/provenance/gate/length，行数、sample key、JSON、checkpoint/intervention 与 SHA 合同通过。
- 新旧 provenance-repair triplet 除 latency、时间戳与执行路径记录外逐单元格相同；gate/length bitwise identical。
- 两次统计的 paired/oracle/ambiguity CSV 字节级一致；其余 sample-std 浮点差异最大 `4.337e-19`，不影响任何表格或放行字段。
- 同 checkpoint B2/B3 eval-k4−k1 为 `−0.01/+0.03 pp`，跨 pair CI 均跨 0；entropy native−constant/shuffled 为 `+0.13/+0.04 pp`，CI 均跨 0；learned−static 为 `−0.01 pp`，CI 跨 0。
- Qwen2.5 seed44 alignment forced-on 为严格 accuracy null；legacy scalar forced-on `+1.91 pp`、CI `[+1.31,+2.53]`，定位为部分 under-transfer 原因。
- B6 native oracle headroom `+8.24 pp`、CI `[+6.28,+10.19]`，4/4 pairs 为正；未评测可实现 selector 的校准或预测能力。
- ambiguity 分桶覆盖存在 task-confounding 与稀疏/退化，故只用于确认没有可靠正向 concentration，不解释负 interaction 为普适机制。
- 定向 Phase 1.5 测试 `35 passed`；项目全量 `236 passed, 2 warnings`；相关脚本 `py_compile` 与 `git diff --check` 通过。

### 结论

Phase 1.5 没有识别到 inference-time 多 candidate、position-matched entropy 或 learned token/head modulation 是 v2.2 开发集收益的稳定平均因果来源。剩余差异更符合 checkpoint/training-regime 与模型对兼容性的 pair-dependent 变化，Qwen2.5 seed44 还受到 legacy scalar K/V hard masks 的部分影响，但本实验不单独识别训练期 k4、tokenizer 身份或随机优化轨迹。预注册 query-time release gate 失败；删除 entropy-aware 与 adaptive-gate 的已验证机制主张，不进入 query-time transport prototype，后续优先考虑 calibrated null/no-transfer 与 sender–receiver compatibility 诊断。

## 2026-07-19：Phase 2A-0 calibrated null/no-transfer opportunity audit

### 研究目标

在不启动 GPU、不训练 selector、不修改 B6 checkpoint 的前提下，严格确认 receiver-only 与 B6-native 之间的逐例互补空间是否相对真实 best fixed policy 仍然充足，并冻结后续 selector 的数据划分、比较对象、统计协议和 GO 条件。

### 核心改动

- 新增 CPU-only `phase2a_0_opportunity_audit.py`，从冻结 Phase 1 analysis manifest 解析唯一 receiver/B6 artifacts，校验 SHA、schema、row count、sample key、输入内容与标签。
- 将逐例结果统一编码为 both-correct、beneficial、harmful、both-wrong 四事件，输出 pair×seed×task、pair、seed、task、all-pair 与 hetero-pair 的 task-macro/sample-weighted 指标。
- bootstrap 改为 canonical sample 跨 pair/seed 同步重采样，并在 aggregate 中按 pair→seed 分层；每个 draw 内重新计算 `max(receiver,fused)`，得到真正 best-fixed-aware headroom CI。
- 正式复现同时校验 Phase 1 suite、Phase 1.5 execution 与旧 oracle CSV 的 SHA/关键行，避免 `+8.24 pp` 解释只依赖手工交叉检查。
- 完成 45 列 selector 字段的 A/B/C/D 审计，记录 entropy/confidence/one-to-many 的确定性冗余、fallback 恒零、pair-specific alignment coverage 与当前缺失的 uncertainty instrumentation。
- 新增 Phase 2A preregistration，冻结 content-group split、calibration-selected global best fixed comparator、primary pair-balanced task-macro、leave-one-out 和六项 conjunctive GO 条件。

### 实验配置

- 数据：MMLU-Redux 5,615、ARC 1,150、OpenBookQA 500；4 pairs、seeds 42/43/44。
- 复用：receiver-only 3 CSV 与 B6-native 36 CSV；artifact commit `9b06d173eada148343ddfb71a31721c7ae5f7ad5`。
- 统计：10,000 draws、95% percentile CI、seed `20260719`；pair/seed cluster resampling 与 task-stratified synchronous paired-sample resampling。
- Primary future estimand：selector vs calibration-selected best fixed，pair-balanced task-macro；sample-weighted mandatory secondary。

### 验证结果

- 定向测试 `3 passed`；39 个正式输入文件、7,265 unique rows、87,180 repeated observations 的合同全部通过。
- Sample-weighted：receiver 36.1597%、fused 46.9316%、oracle 55.1755%、oracle-over-best-fixed `+8.2439 pp`，95% CI `[+6.2318,+10.2019] pp`。
- Task-macro：receiver 38.2684%、fused 50.1531%、oracle 58.6883%、oracle-over-best-fixed `+8.5352 pp`，95% CI `[+6.3438,+10.7561] pp`。
- Fused 在 36/36 pair×seed×task 单元中优于 receiver；因此 Phase 1.5 的 `+8.24 pp` 点估计也是真实 retrospective best-fixed headroom。
- 发现 7,233 content groups 与 32 个 MMLU 重复内容组，后续按 content hash 绑定 split；禁止 row-order 或 question-id-only split。
- 正式统计完成 81 个 aggregate rows；定向测试 `3 passed`，项目全量 `239 passed, 2 warnings`；正式 CSV 确定性重跑 byte-identical，JSON 除输出路径外 scientific content identical；`py_compile` 与 `git diff --check` 通过。

### 结论

Calibrated no-transfer 具有足够大的 oracle opportunity，但当前 A 类特征高度冗余且缺少 receiver/fused uncertainty，不能从 headroom 直接推断 selector 可实现性。Phase 2A-0 只完成机会审计与预注册，明确停止在 selector 训练和 instrumentation rerun 之前，等待审查。

## 2026-07-19：FPCT-0 研究线隔离与预注册

### 研究目标

在 Phase 2A-1 结果揭晓前建立完全独立的 query-time factorization-preserving cache transport 研究线，冻结核心问题、`C_pre`/`C_post`/`F` 三个 operator、阶段门槛和机制主张边界，避免并行研究结果造成后验改写。

### 核心改动

- 从固定 commit `9fa1f0ac3bedefd282961a853278ab88fb376fa2` 创建 sibling worktree `/home/lijunsi/projects/Cache-fpct-factorized-transport` 和 branch `research/fpct-factorized-transport`。
- 新增 `FPCT_PREREGISTRATION.md`，冻结三个 operator 的公式、三个因果对比、第一轮方法限制、endpoints、controls、matched-training 证据规则、六项数值不变量及 FPCT-0 至 FPCT-11 gates。
- 新增 `FPCT_STATUS.md`，记录阶段状态、依赖、资源授权、决策日志、artifact/path contract 和待人工批准事项。
- 明确 `F-C_post` 是 query-time factorization preservation 的 headline 主效应；旧 B6 `+8.24pp` 不作为 FPCT headroom；新 checkpoint 必须重新审计 beneficial/harmful events。

### 实验配置

- 文档与 Git 隔离阶段，仅使用 `R0` 环境/路径检查。
- 未修改模型运行代码，未运行 model forward，未使用 GPU/Kubernetes，未训练或修改 checkpoint。
- 未查看或使用 Phase 2A-1 未公开结果；原 Phase 2A-1 工作树保持在 `main` 且未被修改。

### 验证结果

- 固定 base SHA、branch、worktree 和初始路径隔离已核对。
- 未发现 FPCT worktree 中的 `PHASE2A_*`/`phase2a_*` 文件改动；原 Phase 2A-1 工作树保持 `main`/固定 HEAD，其并行产生的 tracked/untracked 状态变化未被打开、修改或停止。
- 用户确认 FPCT worktree 中的 `math.md` 是其自有公式讨论稿；本任务未修改，并将其作为 non-normative reference 与正式预注册隔离，冲突时以后者为准。
- FPCT-0 文件保持未提交状态，未 push。

### 结论与下一步

FPCT-0 的隔离、预注册内容和一致性检查已完成，当前判定 `GO`；该判定不授权 FPCT-1。primary task suite、统计规则、seed/budget、数值容差和资源上限等缺乏客观依据的项目等待人工批准，不得在看到自然数据结果后补定。

## 2026-07-19：FPCT-1A ambiguity support audit protocol locking

### 研究目标

在任何新的自然 ambiguity 分布或逐样本 alignment 统计被查看之前，冻结 FPCT-1B 的 input/pair/split universe、candidate legality、effective-support threshold、pair eligibility、pilot selection、zero-support contract 和输出 schema。

### 核心改动

- 新增 `FPCT_1A_AMBIGUITY_PROTOCOL.md` 与 `recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest.json`，记录三个 heterogeneous pairs、Qwen3 same-tokenizer control、三任务输入、tokenizer/config/code/input hashes 和 FPCT-1B 的机器可读输出合同。
- 将合法 candidate 固定为 mask/index/padding/finite/positive 的合取，先 mask 后 L1 renormalize；zero-support 独立报告，不填伪 candidate。
- 推荐 gate-primary `S>=2/3`，并冻结 `S>=1/2`、`S>=3/4` sensitivities；明确 formal matched-training ambiguity population 仍为 `m>=2`。
- 将 exact no-factorization control（全体 parent `m<=1`）与含 `m=2` 的 below-gate nominal stratum 分开，避免把操作性 support ceiling 误报为数学等价。
- Pair eligibility 只使用 fit+calibration 的 label-free distinct-content-group support，推荐 3-task task-macro harmonic effective size、9-cell simultaneous Wilson LCB 和结果无关 tie-break。

### 实验配置

- 资源等级 `R0`，只读取静态代码/配置/已有公开 provenance，并编写协议文档与 manifest。
- 固定 `soft_span_overlap_v2`、top-k 4、uniform score、candidate window 0、legacy position、`a=1`、`g=1`；未修改模型运行路径。
- FPCT-1B audit、threshold 和 support-floor 数值仍未授权；`delta_pos`、`delta_direct_all` 等待人工批准。

### 验证结果

- FPCT-0 一致性门通过：`F-C_post` 仍是唯一 headline contrast；selector/native null/de-RoPE/re-RoPE/new gate 仍排除，`math.md` 仍为 non-normative。
- Protocol/manifest 已冻结 split 使用、pilot ranking、GO/LIMITED GO/CROSS-PAIR GO/NO-GO 优先级、CSV/JSON schema、aggregation 与 canonical output root。
- 本阶段未运行自然 ambiguity audit、完整数据 tokenizer/alignment、model forward、GPU/Kubernetes 或训练；未读取 Phase 2A-1 未公开结果。

### 结论与下一步

FPCT-1A 当前为 `REVIEW REQUIRED`：协议结构已完整，但 gate-primary threshold、`task_macro_cluster` support estimand 和 practical/statistical parameters 需要人工在任何自然 audit 前批准。FPCT-1B 及 FPCT-2 以后均保持 `NOT AUTHORIZED`；未 commit、未 push。

## 2026-07-19：FPCT-1A-R human decision lock and prospective amendment

### 研究目标

在任何新的自然 structural-support 数据、逐样本 alignment 统计或 operator result 出现前，完成 FPCT-1A 人工决策锁定，将 v1 中过强的 high-cardinality/power gate 修订为可审计的 structural-opportunity 与工程 readiness 协议。

### 核心改动

- 保留 `FPCT_1A_AMBIGUITY_PROTOCOL.md` 与 v1 manifest 为 never-executed 历史记录，新建 `FPCT_1A_APPROVAL_ADDENDUM.md` 和 schema-v2 `ambiguity_protocol_manifest_v2.json`。
- 将 primary structural support 冻结为 `primary_structural_m2: m>=2`；`m>=3` 与 `m=4` 仅作 sensitivity/enrichment，不能否决 pair、改变 ranking 或替代 headline ceiling。
- 冻结 `D_s=1[样本存在 m>=2 parent]` 与 direct structural-support ceiling；明确 `m=2` 是 mechanism-positive low-cardinality，只有所有 eligible parents `m<=1` 才是 exact `F=C_post` control。
- 冻结 `m=0/1/2/3/4` 五个互斥 strata，并完整保留 zero-support/no-pseudo-candidate contract。
- 将 FPCT-1B 限定为 label-free structural-support audit：普通 95% Wilson interval用于描述，9-cell Bonferroni LCB 只作 sensitivity；formal delta/n_req/power gate deferred。
- 工程 readiness 冻结为每 task 至少 30、三任务 pooled 至少 100 个 primary-positive distinct groups，并使用最弱任务 count、task-macro observed rate、pooled count、pair ID 进行 label-free ranking。

### 实验配置

- Source commit：`7207aafffc7f72976473815bc11102f8b06dddc1`；独立 branch `research/fpct-factorized-transport`。
- v1 的 `commit=false/push=false` 是当时执行边界；用户随后单独授权并已将 source commit `7207aaff...` 推送到同名 research branch，不视为历史错误。
- 资源等级 `R0`；只修改 FPCT protocol 文档与 manifest。
- v1 input/pair/split/alignment/candidate legality/provenance/serialization contracts中未冲突部分继续继承。
- `D_K`、`D_V`、candidate-logit variance、Jensen gap 只登记为后续 operator pilot diagnostics，本阶段不计算。

### 验证结果

- v1 protocol/manifest SHA256 被固定并保持原样；v1 标记为 `superseded_before_natural_data`、`executed=false`。
- Operative protocol 明确由 v1 + approval addendum + v2 manifest 组成。
- `math.md` 保持 non-normative 且 byte-identical；未修改 `main`、其他 worktree、Phase2A 文件或模型代码。
- 本阶段未运行 ambiguity audit、完整 tokenizer/alignment、model forward、GPU/Kubernetes、训练或 checkpoint 操作。

### 结论与下一步

FPCT-1A-R 判定 `GO`，只表示 human decision lock 和 prospective amendment 完成。FPCT-1B、FPCT-1C、FPCT-2 及以后继续 `NOT AUTHORIZED`。验证通过后只允许 commit/push 当前 research branch；不创建 PR、不合并 `main`、不 rebase。

## 2026-07-19：FPCT-1B CPU structural-support audit

### 研究目标

在任何 FPCT operator 实现或 accuracy 评测前，按 v1 + approval addendum + v2 manifest 完整执行 label-free structural-support audit，确定现有三任务/四 pair 数据是否为 `F-C_post` 提供可识别的一对多 candidate support。

### 核心改动

- 新增六模式 CPU audit：`prepare`、`freeze`、`selection`、`lock-selection`、`reporting`、`verify`；只读复用 production prompt/chat-template/offset mapping、`TokenAligner.align_chat_messages_soft` 与 `soft_span_overlap_v2`。
- Commit A `7f8af71968a39bc6cba2e4e34de762b291cda834` 在自然 audit 前推送并作为 execution SHA；随后冻结 pre-audit lock，任何 code/config/test/input/split 改动均会使本轮失效。
- Prepare 独立复现 7,265 canonical rows、7,233 distinct content groups、三个 task content SHA 与 aggregate dataset SHA；生成 hash-only split manifest，不保存 label/answer。
- 自然 audit 生成逐 parent/sample/group local artifacts，selection 仅使用 fit+calibration；pilot lock 后才补 model-selection/test reporting。
- 新增 compact report、aggregate CSV/JSON、result manifest，并保留详细 local artifacts 不提交。

### 实验配置

- CPU-only：所有命令显式 `CUDA_VISIBLE_DEVICES=""`、HF offline、Transformers offline；未实例化或运行 HF/LLM model。
- Alignment：`soft_span_overlap_v2`、top-k 4、uniform、boundary bonus 0.5、tolerance 1、min-weight 0、candidate-window 0、reweight none、confidence control off。
- Primary：`m>=2`；m3/m4 sensitivity-only。Readiness：每任务 positive groups 至少 30 且 pooled 至少 100。
- Selection unit：fit+calibration distinct content group；reporting split 不能改变 selection。

### 验证结果

- 12 个 selection shard 与 12 个 reporting shard 全部完成，无 invalid/negative/nonfinite mass、duplicate legal source index、normalization/uniform failure、`m>4` 或 content-group inconsistency。
- TinyLlama 三任务 fit+cal positive groups 为 511/511、228/228、2495/2495，唯一通过工程门槛并成为 rank-1 selected pilot。
- Qwen2.5 与 Llama3.2 只在 MMLU-Redux 分别有 56/2495 与 50/2495 positive groups，均未通过每任务与 pooled 门槛。
- Same-tokenizer Qwen3 control 永不参与 readiness/ranking；reporting split 未改变 pilot lock。
- 独立 verifier 通过：4,345,744 parent rows、29,060 sample rows、28,932 group rows、60 aggregate rows；schema、Wilson、provenance、deterministic reread/reduction 全部一致。

### 结论

FPCT-1B 判定 `SINGLE_PAIR_PILOT_READY`。这只说明 TinyLlama pair 在现有数据上具有充分 structural opportunity，允许 single-pair operator pilot；不支持 cross-pair confirmatory claim，也不证明 query-time separability、accuracy benefit 或 FPCT 数学有效性。因 audit 完整且无 integrity failure，本次条件授权允许进入 FPCT-1C reference oracle；仍未授权任何 GPU、训练、checkpoint 或正式 accuracy evaluation。

## 2026-07-19：FPCT-1C/2/3 CPU operator gate 与 FPCT-4 non-operative draft

### 研究目标

在 FPCT-1B 唯一 ready 的 TinyLlama structural-support 条件下，先建立独立 pure-tensor reference oracle，再验证现有 `C_pre` 生产路径是否存在无歧义的 candidate-factorization seam；只有数学、mask、gradient、nuisance 与 legacy-default invariants 全部通过后，生成 GPU pilot 的非操作性草案。

### 核心改动

- 新增 `FPCT_1C_OPERATOR_CONTRACT.md`、pure-tensor reference 与 manifest，完整实现 `C_pre`、`C_post`、flat/hierarchical `F` 和 replicated-collapse diagnostic。
- 新增 `FPCT_NUISANCE_CALLGRAPH.md`，将 alignment prior、entropy、confidence、legacy scalar gate、residual、fuser、position 与 attention 的现有所有权映射到候选轴处理边界。
- 新增 `rosetta/model/fpct_attention.py`：保存 `[B,Hkv,N,K,D]` candidate sidecar，只对 `m>=2` parents 做 ambiguous-only packing；children 继承 parent mask/bias，并在同一个 global denominator 中加入一次 `log A`。
- 修改 wrapper/projector，使 `C_post/F` 使用同一个 candidate-specific fuser；parent confidence、entropy 与 legacy Gumbel/hard gate 只计算/采样一次后广播。`m=1` 保持单 slot，`m=0` 保留 native fallback。
- 增加 `fpct_operator=c_pre|c_post|f` 的训练/评测配置 plumbing；flag 未设置时保持 legacy 路径和 state dict，不新增 F-only trainable parameter。
- 新增 CPU production tests，覆盖 reference/production、dense/packed、K=1 eval/training、shared Gumbel、global/hierarchical、replicated collapse、refinement/permutation、causal/padding/zero support、all-invalid、GQA/MQA、gradient、config/state-dict/default regression 与 prior single-use。
- 基于 FPCT-1B 的真实 `m` 分布和静态 receiver config 生成 expansion/cache/FLOP 估算；新增 FPCT-4 non-operative GPU draft，但没有生成 Kubernetes Job。

### 实验配置

- 全部命令显式 `CUDA_VISIBLE_DEVICES=""`；Conda 环境 `c2c-py310-cu124`。
- Reference tolerance：float64 `atol=1e-10, rtol=1e-8`；float32 `atol=rtol=2e-5`；gradcheck `eps=1e-6, atol=1e-5, rtol=1e-3`；invalid probability/gradient exact zero。
- 第一轮仍固定 `a=1`、`g=1`、`position_mode=legacy`；无 native null、selector、Route3 router、de-RoPE/re-RoPE 或新 gate。
- CPU-safe full suite 排除 `test/test_phase2a_0_opportunity_audit.py`，避免越过独立研究线的 Phase2A outcome firewall。
- 资源估算使用本地 Qwen3-0.6B `config.json`：28 layers、16 query heads、8 KV heads、head dim 128、bfloat16；未加载模型权重。

### 验证结果

- FPCT-1C reference tests：`19 passed`；全部冻结 invariants 和预批准 tolerance 通过，未放宽 tolerance。
- FPCT targeted（1B audit + 1C reference + production path）：`52 passed`。
- CPU-safe full suite：`288 passed, 2 warnings`。
- Default、`c_pre` 与 `f` 在 matched seed 下 state-dict keys/tensors 相同；legacy weighted-gather regression 通过。
- Selected TinyLlama pair 的 mean expansion ratio 为 ARC `1.2376`、OpenBookQA `1.2392`、MMLU-Redux `1.2271`；p95 分别为 `1.2975/1.2821/1.3015`。Dense top-k4 mean ratio 为 `3.7387/3.6921/3.7621`。
- TinyLlama ambiguous-only mean K/V sidecar 每层约 `236.42/200.78/265.32 KiB`；28 层增量 cache 约 `3.63/3.07/4.04 MiB`。

### 结论与下一步

FPCT-1C、FPCT-2 和 FPCT-3 判定 `GO`，只建立 mathematical/reference 与 CPU implementation correctness，不代表真实模型中机制已激活或 task accuracy 改善。FPCT-4 draft 标记 `REVIEW REQUIRED / GPU NOT AUTHORIZED`。下一次人工批准必须在任何 GPU 结果前冻结 seed、matched training budget、checkpoint rule、formal effect/power gate、fp16/bfloat16 tolerance、resource ceiling 与 stopping rules；`F-C_post` 仍是唯一 headline。

## 2026-07-20：FPCT-3.5 pre-data alignment-correctness protocol

### 研究目标

在任何新的逐 parent 自然异常、pretrained-model forward、GPU、训练或 accuracy 输出出现前，核查 FPCT-1B raw soft-span one-to-many 是否真正来自 tokenizer partition，而不是 same-tokenizer duplicate offsets、top-k 截断、错误路径或 content-span/offset alias。

### 核心改动

- 新增 `FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md` 与 machine-readable manifest，冻结 exact runtime identity、互斥 offset taxonomy、message-relative certified one-to-many oracle、Qwen3/Qwen2.5 row-level set comparison和 local-only forensic schema。
- 明确现有实现中的 `slm=receiver`、`llm=sender`；exact identity 同时要求 runtime backend、vocab/added/special、chat template、tokenizer-relevant files，以及逐样本 rendered text/IDs/offsets/content spans/message ranges 全部一致。
- Tokenizer behavior fingerprint 不纳入模型 `config.json`、generation config 或 `name_or_path`；这些只保留为独立 provenance，避免把模型架构差异误判为 tokenizer 不同。
- Certified support 增加 top-k exhaustiveness：retained candidates 必须等于全部 positive-overlap source tokens；source truncation 后必须重新认证或降级。
- 预注册 common sanitizer：uncertified raw `m>=2` 在 `C_pre/C_post/F` 中共同变成 raw slot-0 one-hot；slot 0 非法则 hard error；legacy/default path 不变。
- 新增 resumable task-sharded diagnostic skeleton；每个自然 shard 使用 atomic artifact/manifest，完成 shard 可安全 resume-skip，不覆盖历史 FPCT-1B artifacts。

### 实验配置

- 当前只运行 CPU synthetic tests，显式 `CUDA_VISIBLE_DEVICES=""`。
- Starting HEAD：`d296a18be9cc3b0dce3c07f4c2d7244145f2e3ac`。
- 历史 consistency facts：fit+calibration Qwen3 raw 56 positive groups、410 m2 parents；只作固定一致性检查。
- Natural forensic 必须等待本次 protocol/code/tests commit 和 push 后才能运行。

### 验证结果

- Synthetic exact-identity/certification/sanitizer suite：`13 passed`。
- Diagnostic `py_compile` 与 `git diff --check` 通过。
- 未运行自然 tokenizer/alignment forensic、HF/LLM forward、GPU、Kubernetes、训练或 accuracy evaluation。

### 结论与下一步

FPCT-3.5 当前为 `PRE-DATA LOCK PENDING COMMIT`。下一步只允许提交并推送 pre-data protocol commit，再从 clean execution SHA 生成 local lock。若 Qwen3 runtime identity、410-parent taxonomy或路径 provenance 任一失败，整个 overnight run 立即停止且不进入 GPU。

## 2026-07-20：FPCT-3.5 identity forensic 与 FPCT-3.6 conditional correction

### 研究目标

在 pre-data protocol commit 后，对全部 7,265 canonical samples 解释 Qwen3 same-tokenizer raw soft-span anomalies；只有 identity/path/taxonomy 全部通过时，才实现 exact identity 和三臂共用的 conservative sanitizer。

### 核心改动

- 从 pre-data commit `0398d26b63e96263b813730368275ee66e313f66` 生成 local lock 后，完成 ARC、OpenBookQA、MMLU 三个 resumable forensic shards。
- 新增完整 correctness report/result manifest，记录 local ledger、row comparison、runtime fingerprint 和 root-cause hashes。
- Production `TokenAligner` 增加 `exact_identity`：tokenizer behavior fingerprint 与逐样本 rendered text/IDs/offset/content spans/message ranges 全部相等后，每个 eligible parent 强制 identity one-hot。
- 增加 `certified_slot0_v1` sanitizer：只在显式 FPCT recipe 中启用；certified rows 保留，uncertified m>=2 三臂共同退化为 raw slot-0 one-hot，source truncation 后重验并重算 entropy。
- 训练 dataset 与 unified evaluator 接入同一 opt-in sanitizer；非 FPCT legacy/default 路径保持不变。
- 新增 certified-support audit/state machine，冻结 12-cell raw/certified support、resource、Qwen raw/exact control 和 readiness/ranking 输出；shard 采用 staging-directory 原子发布，可清理未完成 staging 后安全恢复，完成 shard 不覆盖。
- Freeze/verify 直接锁定并复核 current HEAD/upstream、FPCT-1B helper、prompt source、7,265-row split manifest、tokenizer/dataset assets、runtime、sanitizer tests 与所有 production entry SHA。
- Independent verify 从 shard CSV 重算 60-row aggregate、ordinary/raw Wilson、Bonferroni-9 sensitivity、readiness/ranking、all-split Qwen identity 和 resource estimates；合法 NO-GO 结果完整保存，不冒充 integrity failure。

### 实验配置

- Forensic 全程 CPU/offline，未运行模型 forward、GPU、Kubernetes、训练或 accuracy。
- Runtime identity fingerprint：Qwen3 receiver/control 均为 `ccf72f82...`；Qwen2.5 为 distinct `8fe3f6d6...`。
- Pre-data consistency：fit+calibration 56 positive groups、410 m2 parents。
- Corrected production max-length certification boundary：1024 tokens；receiver/source retained universe 在 truncation 后重验。

### 验证结果

- 7,265/7,265 Qwen3 samples 的 rendered text、IDs、offsets、content spans、message ranges 和 tokenizer behavior 全部相等。
- All-split raw anomalies 为 802 m2 parents、104 groups；802/802 taxonomy 均为 `duplicate_or_overlap_receiver_offsets`，0 unexplained。
- 802 parents 形成 401 个成对 alias：常见形式是 leading-space-plus-symbol token 与 symbol-only token offsets 嵌套；每行 raw weights 为 `[0.5,0.5]` 且包含 identity candidate。
- Qwen3 与 Qwen2.5 的 positive group、`(sample,parent)`、relative offset、candidate-ID sets 全部逐行相等，Jaccard 1.0；sender path 和 runtime fingerprint 不同，排除 path/object mix-up。
- Exact-identity/sanitizer/certified-audit/legacy aligner targeted suite：`102 passed, 2 warnings`，其中包含 synthetic finalize + independent verify 全链路 round trip。
- Sanitizer 对 nonfinite、negative 和原始序列非法 positive index hard error；合法 source truncation 先 mask，再按 retained overlap universe 重算 exhaustiveness。
- Certified aggregates 同时保留 raw/certified m0–m4、m>=3/m=4 sensitivity、uncertified parent/sample/group、support delta、raw/certified expansion、sidecar cache、packed-extra KV、attention FLOP 与 dense-top-k4 对照。

### 结论与下一步

FPCT-3.5 和 FPCT-3.6 判定 `GO`。Same-tokenizer raw soft-span support 被确认是 offset alias，不能作为 tokenizer factorization support。下一步必须先提交并推送 corrected execution commit，再运行全 pair/task/split certified reaudit；TinyLlama 未通过 certified readiness 前仍禁止 GPU。

## 2026-07-20：FPCT-3.7 execution provenance hard stop

### 研究目标

从 clean/pushed corrected execution SHA 运行 12-cell raw-versus-certified structural-support reaudit，并只在 TinyLlama certified readiness 通过后进入 production hardening。

### 实验配置

- Corrected execution commit：`b11a046597b2466c1c6ba95c4d3693e76523c3b3`，已推送且 local/upstream 相同。
- Pre-audit lock SHA256：`311ddf36bc0ab598ec52eae5236ad14f007a4645373200d58a301c9fcfd9cdb5`。
- 全程显式 `CUDA_VISIBLE_DEVICES=''` 与 HF/Transformers/Datasets offline；未启动模型 forward、GPU、Kubernetes、训练或 accuracy evaluation。

### 执行结果

- TinyLlama/ARC 与 Llama3.2/ARC 都在首个 natural alignment 调用前抛出 `TypeError`：runtime `TokenAligner.align_chat_messages_soft` 不接受冻结调用的 `apply_confidence_control` 参数。
- 冻结 worktree source `rosetta/model/aligner.py` SHA256 为 `fe77d72f...`，方法签名包含该参数；Conda editable mapping 却指向 `/home/lijunsi/projects/KVcache/C2C/rosetta`，其 aligner SHA256 `1d68fe69...` 且签名不包含该参数。
- 因 script-mode `sys.path` 未把 frozen research worktree root 放在 editable installation 之前，实际 runtime source 与 lock 中验证的 production source 不一致。
- Qwen2.5 delegated command 未启动；Qwen3 在 hard stop 后未启动。最终 0/12 shards、0 alignment rows、无 support/readiness/resource 结果。

### 决策与结论

FPCT-3.7 判定 `INCONCLUSIVE`。补 `PYTHONPATH`、改变 invocation mode 或 patch audit 都会改变 freeze 后的 execution contract，因此本 revision 不原地修复或重跑。FPCT-3.8、FPCT-3.9、GPU/K8s、pretrained smoke、12-seed training、model-selection 与 held-out test 全部未进入。

该失败只识别到 execution-provenance sealing 缺口，不构成 tokenizer factorization support、机制或 accuracy 的正负证据。未来若重试，必须由新的 prospective execution revision 在任何自然调用前冻结并测试 imported module path/SHA。

## 2026-07-20：FPCT-OVERNIGHT-R1 sealed-import prospective recovery

### 研究目标

在不改动旧 `b11a046...` execution、failure record、lock 或 local artifacts 的前提下，建立唯一且可机器验证的 Python import contract；在任何新自然 tokenizer/alignment row 前冻结 FPCT-3.5P deterministic replay 与 FPCT-3.7-R1 certified-support revision。

### 核心改动

- 新增最小无副作用 `rosetta/__init__.py`，把可跨 worktree 合并的 namespace package 收窄为唯一 current-worktree regular package。
- 新增 absolute-realpath `python -I` bootstrap：固定 cwd，忽略 `PYTHONPATH/PYTHONHOME` 与 user site；保留但不修改旧 Conda editable metadata。
- Bootstrap 在 target 前后重算 module closure，冻结 interpreter SHA、Python ABI/flags、git identity、`sys.path/meta_path`、distribution/direct-url、module origin/file SHA、完整 aligner API signature 与所有已加载 `rosetta.*` origins。
- 在任何受保护自然路径读取前，用 fake tokenizer 真实调用 production `TokenAligner.align_chat_messages_soft(..., apply_confidence_control=False)` 的 exact-identity probe。
- 新增 FPCT-3.5P/3.7-R1 protocol、manifest 与 machine-verifiable zero-scientific-change diff；exact identity、certifier、`certified_slot0_v1`、top-k、readiness/ranking 与 resource formulas 均不变。
- 预先实现新 revision replay/audit targets：3.5P 比较 ordered/multiset candidate atoms、固定 context window、401 overlap clusters 与 geometry co-occurrence；3.7-R1 增加 raw-pre-truncation→retained→sanitized、exposure/geometry/p99/resource/sanitizer-integrity 描述，均不改变 primary certification 或 readiness。

### 实验配置

- 当前仅 CPU synthetic/subprocess tests；`CUDA_VISIBLE_DEVICES=''`、HF/Transformers/Datasets offline。
- Starting HEAD/upstream：`e3943216ee7324b2a010ee006a30dcfa6145284f`。
- Canonical interpreter：`/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python3.10`。
- 当前未执行新自然 tokenizer/data audit；未运行 HF/LLM forward、GPU、Kubernetes、训练或 accuracy evaluation。

### 验证结果

- Hostile real-subprocess sealed-import suite：初版 `20 passed`；stable-temp amendment 后 `21 passed`。
- FPCT alignment/sanitizer/certified-audit + sealed-import targeted suite：`57 passed`。
- Full suite 首次使用 `/tmp` basetemp 时两项既有 Route1 K8s 测试因 manifest 路径不在 repo/workspace 而拒绝；按仓库约定改用 repo-local basetemp 后正式 gate 为 `348 passed, 2 warnings`。
- `py_compile`、JSON syntax 与 `git diff --check` 通过。

### 结论与下一步

状态为 `PRE-DATA PROTOCOL READY`，不是自然 forensic 或 certified-support GO。下一步必须先提交并推送当前 prospective code；clean containing commit 才能成为新 execution SHA。随后 3.5P exact replay 任一差异均停止；只有 provenance-confirmed 后才允许新的 3.7-R1 12-cell CPU audit。

### Pre-data attestation amendment

首个 pushed prospective SHA `9e501d7...` 的 freeze 成功，但三个自然 shard 都在 target 前被 fingerprint mismatch 拦截，因此仍为 0 natural rows。唯一不稳定字段是 Torch 每进程生成的 `/tmp/tmpXXXX/_remote_module_non_scriptable.py` 路径名。

修订后的 bootstrap 继续记录完整随机路径；stable projection 只在目录精确包含 generated source 与匹配 `__pycache__`、生成源码 SHA 被记录、且不存在 foreign `rosetta` candidate 时，把随机目录名替换为 source-SHA identity。任何其他临时 payload 仍 hard error。双进程 fingerprint 稳定性已加入 hostile suite，结果为 `21 passed`。该修订需要新的 clean/pushed execution SHA；`9e501d7...` lock/artifacts 不复用。

## 2026-07-20：FPCT-3.5P/3.7-R1 completion and CPU/HF hardening

### 研究目标

在 sealed import contract 下精确重放 FPCT-3.5，重新执行 12-cell certified-support audit；仅在 TinyLlama readiness、Qwen identity 与 sanitizer integrity 全通过后，消除 FPCT attention hot-path host sync 并用真实 Qwen3 类完成 CPU integration。

### 核心改动

- `7aecf237...` FPCT-3.5P replay 精确复现历史 ordered/multiset row/candidate/context projection；新增 primary reason × secondary geometry co-occurrence。
- FPCT-3.7-R1 新 root 完成 12 shards、three-stage transition、exposure、geometry、resource p99/correlation 与 sanitizer-input hash 描述。
- 新增 reusable `FPCTPackedLayout`；结构 map 在 attention 前建立并跨层复用，per-layer path 只使用 gather/scatter/indexing。
- `m=1` 保持单 slot，`m=0` native，只有 certified `m>=2` 展开；parent mask/bias 与 log prior 正确广播，generated tail 保持同一 denominator。
- 新增 parameter-free replicated-collapse 和 off-by-default aggregate mechanism instrumentation；无 F-only trainable parameter。
- 训练/evaluation loader 透传 replicated-collapse/instrumentation；默认行为和 state dict 不变。

### 实验配置与结果

- FPCT-3.5P stable fingerprint：`609e3f05...`；FPCT-3.7-R1：`5be64db7...`。
- Replay：7,265 identity、802 parents、104 groups、410/56 fit+cal、401 clusters；全部 equality checks pass。
- Certified readiness：TinyLlama 511/228/2495 positive groups；唯一 ready pair。
- TinyLlama all-split certified expansion mean 为 1.2259–1.2391，p95 为 1.2818–1.3000，低于预注册 1.35/1.50 resource support gate。
- Projector dropout 0.1 下 C_post/F pre-collapse fused candidates 在同 RNG/call order 下 bitwise equal；K=1 training-mode equality pass。
- Actual random-config `Qwen3ForCausalLM` eager/DynamicCache CPU tests 覆盖 prefill/decode、batch/padding、causal、GQA/MQA、m0/m1/m>=2、replicated collapse、forward/backward 和 config roundtrip。
- Targeted reference/production/Qwen/sanitizer：`64 passed`；complete CPU-safe suite：`360 passed, 2 warnings`。

### 结论与下一步

FPCT-3.5P、FPCT-3.7-R1、FPCT-3.8 与 FPCT-3.9 均通过对应硬门。结论仍只到 sealed structural support 和 CPU/HF implementation correctness；尚无 pretrained output、GPU、训练或 accuracy。下一步先 commit/push 当前 hardening，再单独冻结 confirmatory/GPU/K8s code、statistics、image 与 execution manifest。

## 2026-07-20：FPCT confirmatory scientific-code pre-output lock

### 研究目标

在任何 pretrained output、GPU training loss 或 benchmark accuracy 前，冻结 TinyLlama→Qwen3 单-pair 12-seed 三臂设计、GPU 数值门、K8s triplet execution、split firewall 与正式统计实现。

### 核心改动

- 新增 confirmatory、GPU numerical、K8s 三份 normative protocol 与机器可读 manifest。
- 固定 seeds 45–56 的六种平衡三臂顺序、2,048 examples、2 processes、gradient accumulation 16、64 optimizer steps 和唯一 step-64 正式 checkpoint。
- 新增 recoverable controller：状态和 append-only ledger 原子写入；12 个完整 triplet 前禁止 model-selection；futility 通过前 held-out 最多释放一次。
- 新增 seed/group/task hierarchical paired bootstrap、exact 4096 sign flips、paired-t 和 exact-sign sensitivities；明确 sign-flip 依赖 sharp/symmetric sign-exchangeability，不冒充 composite mean-null 的无假设精确检验。
- 正式训练器新增 sealed-bootstrap hard guard、distributed step-0/trainable-key/data-order identity、精确 training-example/optimizer-step gate 和 atomic checkpoint SHA manifest。
- 新增 digest-addressed immutable image recipe与 init/rank attestation K8s templates；不挂载 host Conda、site-packages 或 source worktree。

### 实验配置

- 当前只运行 CPU-only syntax/targeted tests；`CUDA_VISIBLE_DEVICES=''`。
- 未运行 pretrained model forward、GPU/CUDA、Kubernetes、训练、checkpoint accuracy 或 held-out evaluation。
- 后续采用 two-lock flow：先 commit/push scientific code；再从该 clean SHA 构建 image，冻结 model/data/image/runtime hashes 和 run lock 后才能进入 GPU numerical gate。

### 验证结果

- confirmatory statistics/controller/runner、sealed import、dataset split、FPCT production 和真实 random-config Qwen CPU integration targeted tests 通过；首次 lock 为 `372 passed, 2 warnings`。
- 首个镜像 build 后的 contract review 发现三臂会分别重新 tokenization/alignment。该镜像未运行 GPU；新增一次性 2048-example certified alignment cache materializer、只读 dataset cache 与三臂相同 sidecar SHA 绑定，diagnostic 128 只读取固定 prefix。修订后 targeted 为 `72 passed`，完整 CPU-safe suite 为 `373 passed, 2 warnings`。
- 集群只读盘点确认 namespace 无共享 PVC、三台 amd64 GPU 节点共同可见 `/netdisk` 冻结模型资产。K8s 模板因此只挂载 `/netdisk/lijunsi/c2c-route1-identifiability/models`（只读）与 `/netdisk/lijunsi/fpct-confirmatory/<run_uid>`（本次独占），不挂载 host Conda、site-packages、source worktree 或其他实验路径。
- JSON syntax、Python compile 与 `git diff --check` 通过。

### 结论与下一步

状态为 `SCIENTIFIC CODE LOCK IN REVIEW`，不是 GPU GO。下一步运行完整 CPU-safe suite；通过后提交并推送 scientific-code commit，随后构建不可变镜像并提交 pre-output run lock。

### Final pre-output lock

- Operative scientific SHA：`850b9d1a2298f5026abba61d59d0f07cb73d29c0`。
- Image：`docker.io/library/fpct-confirmatory:850b9d1a@sha256:447ab481673803fdaf362956e674daad46c4b500ad0d2f2a950491d6ed91dded`。
- Frozen 2,048-example sidecar：`48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990`。
- Sealed no-model container fingerprint：`a56ee395994013b35809612c2ddddc74b5c47f3b998d0195b22fddab51f31614`。
- Candidate images 对缺 evaluator dependency 与 Python symlink 均在模型加载前硬停并废弃；无 GPU/pretrained output。

下一步仅授权 synthetic GPU numerical gate；其通过前不运行 pretrained smoke 或训练。

Pre-output K8s storage probes进一步确认 `/netdisk` 只在 `4090-48gx2` 可见，`/home/lijunsi` 在三台 worker 上均不提供冻结模型。为避免 lock 后复制资产形成 divergent local copies，operative hardware pool 收窄到该双 GPU 节点，formal seed parallelism 固定为 1。

首次 synthetic GPU gate 的 FP16/BF16 forward/gradient/row-sum/invalid checks 全部通过，但 runner 错把 BF16 row-sum error 乘 10 作为 natural activation floor，生成 `0.0244140625`。该 artifact 在任何 pretrained output 前作废；实现改为独立 FP32 oracle forward/gradient、replicated-collapse 与 m≤1 output/greedy exact null，并仅由两类 output-null delta 前瞻生成 activation floor。按 image invalidation rule 从 GPU gate 重新开始。
