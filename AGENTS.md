# AGENTS.md

## 目标

面向 ICLR 2027 开展 C2C/Rosetta 研究。重点推进Cross-tokenizer KV-cache alignment and transport for heterogeneous LLM communication

## 项目结构

- 训练入口：`script/train/SFT_train.py`
- 评测入口：`script/evaluation/unified_evaluator.py`
- 分析脚本：`script/analysis/`
- 核心代码：`rosetta/`
- 配置文件：`recipe/`
- 测试目录：`test/`
- 实验产物：`local/`，不得提交

## Environment

本地测试使用 Conda 环境 `c2c-py310-cu124`；正式训练和评测通过 Kubernetes 调度 GPU,尽量使用多个卡

## 常用命令

```bash
# 本地测试
python -m pytest -q --no-cov --basetemp=local/tmp/pytest-all
# 提交任务
bash bash/k8s/gpu_job.sh submit \
  --name <name> \
  --gpus <num> \
  --follow \
  -- python <script> <args>
# 管理任务
bash bash/k8s/gpu_job.sh list
bash bash/k8s/gpu_job.sh logs <job-name> --follow
bash bash/k8s/gpu_job.sh delete <job-name>
```

## Commit

每次改进后按“日期与主题、研究目标、核心改动、实验配置、验证结果、结论”更新 `FRAMEWORK_UPDATE.md`。如有实验，配置及结果数据放入`EXPERIMENT.md`，不使用 Pull Request；只提交本次文件，禁止强推，直接提交并推送 `main`。