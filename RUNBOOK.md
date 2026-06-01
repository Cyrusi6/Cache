# 跑通训练与评测

这份文档给出一条最短闭环：先用小规模配置跑通训练，再用训练产物完成一次评测。

## 1. 准备环境

```bash
conda create -n rosetta python=3.10
conda activate rosetta
pip install -e ".[dev,training,evaluation]"
```

如果你更想复现仓库原始环境，也可以直接使用根目录的 `environment.yml`。

## 2. 先改成最小训练配置

从 `recipe/train_recipe/C2C_0.6+0.5.json` 开始，不建议第一次就直接跑默认大规模配置。先把这些字段改小：

- `data.kwargs.num_samples`: 改成 `128` 或 `1000`
- `training.num_epochs`: 改成 `1`
- `training.per_device_train_batch_size`: 根据显存调小
- `training.gradient_accumulation_steps`: 根据显存调小
- `output.output_dir`: 改成你自己的目录，例如 `local/checkpoints/smoke_test`
- `output.wandb_config.mode`: 如果没有登录 Weights & Biases，改成 `offline`

这个配置默认会从 Hugging Face 拉取：

- base model: `Qwen/Qwen3-0.6B`
- teacher model: `Qwen/Qwen2.5-0.5B-Instruct`
- dataset: `teknium/OpenHermes-2.5`

## 3. 跑训练

单卡直接运行：

```bash
python script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json
```

仓库也提供了一个单卡脚本：

```bash
bash bash/train/sft_train.sh
```

如果只想检查数据和前向是否正常，不真正训练，可以先跑：

```bash
python script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_0.6+0.5.json \
  --eval_only
```

训练成功后，关键产物会出现在：

- `<output_dir>/config.json`
- `<output_dir>/checkpoint-*`
- `<output_dir>/final/projector_*.pt`
- `<output_dir>/final/projector_*.json`
- `<output_dir>/final/projector_config.json`

后续评测要用的就是 `<output_dir>/final/`。

## 4. 改评测配置

编辑 `recipe/eval_recipe/unified_eval.yaml`，至少改这几项：

- `model.rosetta_config.checkpoints_dir`: 指向训练输出的 `final/` 目录
- `output.output_dir`: 评测结果输出目录
- `eval.gpu_ids`: 例如 `[0]`
- `eval.subjects`: 第一次建议只保留一个科目，例如 `["nutrition"]`
- `eval.limit`: 第一次建议加上，例如 `20`

示例：

```yaml
model:
  model_name: Rosetta
  rosetta_config:
    base_model: Qwen/Qwen3-0.6B
    teacher_model: Qwen/Qwen2.5-0.5B-Instruct
    checkpoints_dir: local/checkpoints/smoke_test/final

output:
  output_dir: local/final_results/smoke_test

eval:
  dataset: mmlu-redux
  gpu_ids: [0]
  subjects: ["nutrition"]
  limit: 20
```

## 5. 跑评测

```bash
python script/evaluation/unified_evaluator.py \
  --config recipe/eval_recipe/unified_eval.yaml
```

或者使用仓库脚本：

```bash
bash bash/eval/run_eval.sh
```

## 6. 常见坑

- 训练输出目录变了，评测里的 `checkpoints_dir` 也要一起改。
- 评测脚本以 YAML 中的 `eval.gpu_ids` 为准，不要只改 `CUDA_VISIBLE_DEVICES`。
- 第一次建议先用小样本和单科目确认链路通了，再扩大数据规模和评测范围。
