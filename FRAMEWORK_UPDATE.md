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
- 生成清单确认 67 runs、三条四卡 lane 和两阶段 gate/依赖关系一致；本条记录不将调度清单误写为已完成实验。

### 结论与下一步

复现门控已通过，且历史 v2.2 的偏差已被确认来自 split 实现变化并通过冻结 index 消除。整套组件实验与跨模型验证任务状态记为“运行中”，目前不能回答 soft candidates、entropy 或 token/head gate 谁是主要来源，也不能据此进入下一阶段。第一阶段最终判断只使用下游 accuracy、配对迁移和统计检验；train/eval loss 仅作优化诊断，尤其不得因 learned-affine 的 eval loss 更低而宣称机制更优。
