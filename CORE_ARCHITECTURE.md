# 核心架构与调用链

这份文档说明这个仓库里最关键的类、入口脚本、模型装配方式，以及训练/评测时它们如何串起来。

## 1. 项目核心对象

这个项目实现的是一种 **Cache-to-Cache (C2C)** 通信框架。运行时通常有三类对象：

- `base model`：最终负责生成答案的接收方模型
- `teacher/sharer model`：提供 KV Cache 的共享方模型，可以是一个或多个
- `projector`：把 sharer 的 KV Cache 投影到 base model 的 KV 空间

把三者组合起来的总封装是 `RosettaModel`，定义在 `rosetta/model/wrapper.py`。它不是重新训练一个新大模型，而是把现有 Hugging Face Causal LM 和一组 projector 拼成一个“组合模型”。

## 2. 核心文件对应关系

### 运行入口

- `script/train/SFT_train.py`
  训练主入口。负责读 recipe、加载模型、创建数据集、训练 projector、保存 checkpoint。
- `script/evaluation/unified_evaluator.py`
  评测主入口。负责加载模型、构造 prompt、按 benchmark 执行 generate 或 logits 评测。
- `script/playground/live_chat_example.py`
  最直接的交互式推理示例，适合看单轮推理时模型如何被装起来。

### 核心模型层

- `rosetta/model/wrapper.py`
  定义 `RosettaModel`，是整个项目最核心的执行器。
- `rosetta/model/projector.py`
  定义 `Projector` 基类、`C2CProjector` 主实现，以及 projector 的保存/加载/工厂函数。
- `rosetta/model/aligner.py`
  定义 `TokenAligner`，用于 base/teacher tokenizer 不一致时的双路对齐。
- `rosetta/model/sampling.py`
  定义 `sample_token()`，供 `RosettaModel.generate()` 的自定义解码循环使用。

### 数据与配置层

- `rosetta/train/dataset_adapters.py`
  定义训练数据集、`ChatDataset`、`AlignedChatDataset`、`RosettaDataCollator` 和 `generate_kv_cache_index()`。
- `rosetta/train/model_utils.py`
  定义层映射策略，如 `last_aligned_sources()` 和 `k_nearest_sources()`。
- `rosetta/utils/evaluate.py`
  定义评测时的 `load_rosetta_model()`，负责从 checkpoint 重建 `RosettaModel`。
- `rosetta/utils/registry.py`
  定义 registry 与对象序列化机制，projector 的配置保存依赖它。
- `rosetta/utils/core.py`
  定义多 sharer 的 bitmask 约定，如 `sharers_to_mask()`、`all_sharers_mask()`。

## 3. `RosettaModel` 怎么工作

`RosettaModel` 的关键成员有：

- `model_list`
  结构通常是 `[base_model, teacher_model1, teacher_model2, ...]`
- `projector_list`
  保存所有 projector 模块
- `projector_dict`
  描述“哪个 source layer 用哪个 projector 投影到哪个 target layer”
- `kv_cache_dict`
  运行时缓存 base/sharer 的 KV Cache

### `projector_dict` 的作用

`set_projector_config()` 会把层映射注册到 `projector_dict`。逻辑上它表达的是：

`(source_model_idx, source_layer_idx) -> (target_model_idx, target_layer_idx, projector_idx)`

训练阶段和评测阶段都会依赖这个映射。如果没有 `projector_config.json`，评测时就不知道每个 projector 应该作用到哪一层。

### `kv_cache_index` 的作用

`kv_cache_index` 是 C2C 的控制面，定义在 `rosetta/train/dataset_adapters.py` 和 `unified_evaluator.py` 的输入准备逻辑中。

它按“段”控制当前 token 区间是否启用投影：

- `[1, 0]`：启用 sharer 1 的投影
- `[-1, 0]`：不做投影，只用 base model
- 多 sharer 时，第一列是 bitmask，例如 `3` 表示同时启用 sharer 1 和 2

训练里默认做法是：

- instruction 段用 `[1, 0]`
- answer 段用 `[-1, 0]`

也就是“先让 sharer 参与理解 prompt，再让 base model 自己生成答案”。

## 4. 训练调用链

训练主链路在 `script/train/SFT_train.py`：

1. `main()` 读取 train recipe
2. `detect_training_mode()` 判断是 baseline 还是 rosetta 模式
3. `setup_models()` 加载：
   - `base_model`
   - `teacher_model`
   - projector 列表
   - `RosettaModel`
4. `last_aligned_sources()` 或 `k_nearest_sources()` 生成层映射
5. 循环调用 `set_projector_config()` 注册 source/target layer 对应关系
6. 根据 `training.freeze` 冻结 base/teacher，只训练 projector
7. `create_dataset()` 创建原始数据集，如 `OpenHermesChatDataset`
8. 再包装成：
   - `ChatDataset`：单 tokenizer 路径
   - `AlignedChatDataset`：双 tokenizer 对齐路径
9. `RosettaDataCollator` 把样本切成 section，并拼出 batch 级 `kv_cache_index`
10. `train_step()` 调用 `model.forward(...)`
11. 训练结束后保存：
   - `projector_*.pt`
   - `projector_*.json`
   - `projector_config.json`

### 训练时 `forward()` 内部发生了什么

`RosettaModel.forward()` 会先按照 `kv_cache_index` 把序列切成多个 section。对每个 section：

1. 先让 base model 前向，得到当前 base KV Cache
2. 再让每个 sharer model 在 `torch.no_grad()` 下前向，得到各自 KV Cache
3. 根据 `projector_dict` 找到应该投影的 source layer / target layer
4. 调用 `projector.forward(source_kv, target_kv)` 得到投影后的 key/value
5. 把结果写回 base cache

多 sharer 时有两种融合模式：

- `sequential`
  每个 sharer 依次修改 base cache，后面的 sharer 看到的是前面已修改过的 cache
- `parallel`
  每个 sharer 都基于同一份干净 base cache 计算增量，最后统一叠加

如果 `include_response=True`，最后一段还会通过 hook/monkeypatch 让“修改后的 KV”在当前 forward 中立即生效，而不是只影响下一步。

## 5. `C2CProjector` 怎么工作

主 projector 在 `rosetta/model/projector.py` 的 `C2CProjector`。

它的输入是：

- `source_kv = (source_key, source_value)`
- `target_kv = (target_key, target_value)`

核心流程是：

1. 把 source 和 target 的 key/value 展平到 head 维之后拼接
2. 经过 `key_in/value_in + RegularMLP`
3. 一条分支生成投影后的 key/value
4. 另一条分支生成逐 token、逐 head 的 scalar 权重
5. 再通过 gate 控制是否把投影结果加到 target 上

输出形式是：

- `output_key = target_key + gate * scalar * projected_key`
- `output_value = target_value + gate * scalar * projected_value`

所以 projector 不是简单替换 target KV，而是在 target KV 上叠加一个“来自 sharer 的语义增量”。

### 为什么 projector 可以被保存和重建

`C2CProjector` 用了 `@capture_init_args`，而 `save_projector()` / `load_projector()` 走的是 `rosetta/utils/registry.py` 的序列化逻辑。  
这意味着 checkpoint 里不仅保存权重，还保存构造参数，所以评测时可以先按 JSON 重新实例化 projector，再加载 `.pt` 权重。

## 6. 数据如何进入模型

### `ChatDataset`

`ChatDataset` 会把一条对话拆成：

- instruction：最后一条 assistant 回复之前的内容
- full_text：完整对话

然后构造：

- `input_ids`
- `labels`
- `kv_cache_index`

其中 instruction 部分的 label 被置为 `-100`，只监督回答部分。

### `AlignedChatDataset`

如果 base 和 teacher tokenizer 不一致，就走 `AlignedChatDataset`：

1. `TokenAligner.align_chat_messages()` 同时生成 SLM 和 LLM token 序列
2. 把两路 token pad 到同长度
3. 保留 `message_mask`
4. 基于 message 段生成 `kv_cache_index`

这样 `RosettaModel.forward()` 就能接收一个 `input_ids` 列表：

- `input_ids[0]` 给 base model
- `input_ids[1]` 给 teacher model

## 7. 评测调用链

评测主链路在 `script/evaluation/unified_evaluator.py`：

1. `main()` 读取 eval YAML
2. `UnifiedEvaluator` 根据 `model.model_name` 选择模型类型
3. Rosetta 路径下调用 `rosetta/utils/evaluate.py` 的 `load_rosetta_model()`
4. `load_rosetta_model()` 会：
   - 加载 base model 与 teacher model
   - 从 checkpoint 目录读取 `projector_*.json/.pt`
   - 创建 `RosettaModel`
   - 读取 `projector_config.json` 恢复层映射
5. `prepare_model_inputs()` 构造 prompt、chat template、tokenized inputs 和 `kv_cache_index`
6. 根据 `answer_method` 走：
   - `model.forward()`：logits 评测
   - `model.generate()`：生成式评测

### `generate()` 为什么是自定义的

`RosettaModel.generate()` 没直接复用 HF `generate()`，而是实现了自己的解码循环：

1. 先调用一次 `forward()` 完成 prompt prefill，构建 KV Cache
2. 取最后一个 token 的 logits
3. 调用 `sample_token()` 采样下一个 token
4. 把新 token 作为单步输入继续调用 `forward()`
5. 继续迭代直到到达 `max_new_tokens` 或 EOS

生成阶段默认把新的 `kv_cache_index` 设为 `[-1, 0]`，也就是生成 token 时不再继续做新的 C2C 注入，重点让 prompt 编码阶段受 sharer 帮助。

## 8. Checkpoint 格式

训练输出目录下最重要的是：

- `config.json`
  对应这次训练用的 recipe 副本
- `final/projector_0.pt`
- `final/projector_0.json`
- `final/projector_config.json`

对于多层 projector，会有多个 `projector_i.pt/json`。  
评测加载时依赖的是 `final/` 目录，而不是训练输出根目录。

## 9. 可以把这个项目理解成什么

如果只看工程形态，这个项目可以理解成一个“**多模型 KV Cache 投影与融合运行时**”：

- `wrapper.py` 负责执行
- `projector.py` 负责变换
- `dataset_adapters.py` 负责把训练数据变成可控的 KV 段
- `SFT_train.py` 负责训练 projector
- `unified_evaluator.py` 负责把训练好的 projector 接回 base/sharer 模型做评测

最核心的一句话是：  
**base model 负责最终输出，teacher model 提供缓存语义，projector 决定如何把这些语义写进 base model 的 KV Cache。**
