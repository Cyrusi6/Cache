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
