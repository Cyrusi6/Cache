# AGENTS.md

## 目标

面向 ICLR 2027 开展 C2C/Rosetta 研究。以 Route-1 v2.2 为基线，重点推进候选约束的 learned flexible alignment。结论须来自公平、可复现实验；保留负结果，不以 validation loss 代替下游指标。代码与文档冲突时以当前代码和测试为准。

## 项目结构

`rosetta/` 存放模型、对齐、训练与评测逻辑；`script/` 存放运行入口；`recipe/` 存放配置；`test/` 存放测试；临时配置、checkpoint 和结果统一放入 `local/`，不得提交。

## Environment

固定使用 `c2c-py310-cu124`，禁止使用 Python 3.7 的 `base`。本机有 4 张 RTX 4090，运行前检查 GPU。

```bash
source /home/lijunsi/miniconda3/etc/profile.d/conda.sh
conda activate c2c-py310-cu124
nvidia-smi
```

## 常用命令

```bash
python -m pytest -q --no-cov --basetemp=local/tmp/pytest-all
CUDA_VISIBLE_DEVICES=0 python script/train/SFT_train.py --config <train-config>
python script/evaluation/unified_evaluator.py --config <eval-config>
kubectl config current-context
kubectl get nodes -o wide
kubectl get pods -A -o wide
```

训练先跑小样本闭环，再扩大规模。Kubernetes 写操作必须指定用户授权的 namespace，禁止误用其他项目或系统 namespace。

## Commit

每次改进后按“日期与主题、研究目标、核心改动、实验配置、验证结果、结论与下一步”更新 `FRAMEWORK_UPDATE.md`。不使用 Pull Request；只提交本次文件，禁止强推，直接提交并推送 `main`。

```bash
git commit -m "提交信息"
git push origin main
```
