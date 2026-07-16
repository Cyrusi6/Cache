# AGENTS.md

## 目标

面向 ICLR 2027 开展 C2C/Rosetta 研究。重点推进Cross-tokenizer KV-cache alignment and transport for heterogeneous LLM communication

## 项目结构

`rosetta/` 存放模型、对齐、训练与评测逻辑；
`script/` 存放运行入口；
`recipe/` 存放配置；
`test/` 存放测试；
临时配置、checkpoint 和结果统一放入 `local/`，不得提交。

## Environment

使用kubectl调用合适的集群

## 常用命令

```bash
python -m pytest -q --no-cov --basetemp=local/tmp/pytest-all
CUDA_VISIBLE_DEVICES=0 python script/train/SFT_train.py --config <train-config>
python script/evaluation/unified_evaluator.py --config <eval-config>
kubectl config current-context
kubectl get nodes -o wide
kubectl get pods -A -o wide
```

## Commit

每次改进后按“日期与主题、研究目标、核心改动、实验配置、验证结果、结论”更新 `FRAMEWORK_UPDATE.md`。如有实验，配置及结果数据放入`EXPERIMENT.md`，不使用 Pull Request；只提交本次文件，禁止强推，直接提交并推送 `main`。