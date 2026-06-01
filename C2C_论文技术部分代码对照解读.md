# C2C 论文技术部分代码对照解读

这份文档按照论文 `Section 3 Method` 的写作顺序，解释 C2C 的技术设计，并把每一部分映射到仓库中的具体实现。

## 0. 阅读范围与术语对照

这次对照主要覆盖论文的以下部分：

1. `3.1 Preliminaries`
2. `3.2 Oracles for Cache-to-Cache Communication`
3. `3.3 C2C Design`

在仓库实现里，论文术语和代码命名并不完全一致，先做一个对照：

| 论文术语 | 仓库里的常见名字 | 说明 |
| --- | --- | --- |
| Receiver | `base_model` / `slm` / `target_model` | 最终负责生成答案的模型 |
| Sharer | `teacher_model` / `llm` / `source_model` | 提供 KV-Cache 语义的模型 |
| C2C Fuser | `C2CProjector` | 把 Sharer 的 KV 投影并融合到 Receiver 中 |
| Layer mapping `G` | `projector_dict` + `last_aligned_sources()` / `k_nearest_sources()` | 指定哪一层对哪一层 |
| 整体 C2C 系统 | `RosettaModel` | 把 base model、teacher model、projector 组装成统一运行时 |

还要先说明一件事：

- 论文 `3.2` 的两个 oracle 更像“验证性实验”。
- 仓库里真正的主线实现，从工程角度看主要落在 `3.3 C2C Design` 这一部分。
- 因此 `3.2` 在代码里更多体现为实验脚手架和辅助分析代码，而不是最终日常推理入口。

核心代码入口如下：

- `rosetta/model/wrapper.py`：C2C 运行时主执行器 `RosettaModel`
- `rosetta/model/projector.py`：`C2CProjector` 与 projector 的保存/加载
- `rosetta/model/aligner.py`：跨 tokenizer 的 token 对齐
- `rosetta/train/dataset_adapters.py`：训练数据、`kv_cache_index`、batch 切段
- `rosetta/train/model_utils.py`：层对齐策略
- `script/train/SFT_train.py`：训练主入口
- `rosetta/utils/evaluate.py`：评测时重建 C2C 模型

---

## 1. 论文 3.1 Preliminaries

### 1.1 论文在说什么

论文先回到标准自回归 LLM 推理过程。

对输入序列 `X[0:n]`，模型在 prefill 阶段先把整段输入编码成 KV-Cache，记作：

`C(X[0:n]) = [c0, ..., c(n-1)]`

之后在 decode 阶段，模型每次利用：

- 输入上下文的 KV-Cache `C(X)`
- 已生成前缀的 KV-Cache `C(Y[0:i])`

来预测下一个 token。论文用公式表达为：

`y(i+1) = P(yi | C(X) ⊕ C(Y[0:i]))`

这里的关键点是：在 LLM 看来，真正驱动后续生成的不是“原始文本本身”，而是经过 prefill 后形成的内部缓存表示。  
这就给 C2C 一个核心出发点：如果缓存本身就是承载上下文语义的中介，那么模型之间是否可以直接交换缓存，而不必先把语义重新压缩成文本？

论文在这一节还定义了两个角色：

- Sharer：提供上下文理解或知识的模型
- Receiver：使用这些信息并负责最终回答的模型

### 1.2 代码怎么实现这个前提

这部分在仓库里主要对应 `rosetta/model/wrapper.py` 的 `RosettaModel`。

#### `RosettaModel.forward()`：负责 prefill 与缓存更新

代码不是把 C2C 单独做成一个“外部预处理器”，而是直接把它塞进一个自定义的模型包装器里。  
`forward()` 会把输入按 section 切开，然后按段执行：

1. 先让 Receiver/base model 前向，拿到 base KV-Cache。
2. 再让 Sharer/teacher model 前向，拿到 source KV-Cache。
3. 根据 `projector_dict` 找到应当做融合的层。
4. 调用 projector，把 source KV 和 target KV 融合。
5. 把融合后的 KV 写回 Receiver 的 cache。

这正对应了论文里“先得到缓存，再用缓存继续生成”的推理前提。

#### `RosettaModel.generate()`：显式区分 prefill 和 decode

`generate()` 没有直接复用 Hugging Face 默认的 `generate()`，而是自己实现了一个解码循环：

1. 先调用一次 `self.forward(...)` 完成 prompt prefill，并建立 fused cache。
2. 读取最后一个位置的 logits。
3. 用 `rosetta/model/sampling.py` 里的 `sample_token()` 采样下一个 token。
4. 把新 token 再送回 `forward()`，继续单步 decode。

这和论文 3.1 讲的 prefill/decode 二阶段是完全一致的，只不过仓库把“prefill 阶段可能发生 cache fusion”显式工程化了。

#### `kv_cache_dict`：论文里 `C(X)` 的工程落点

论文里的 `C(X)` 在实现里并没有抽象成一个单独数学对象，而是落到了 `RosettaModel.kv_cache_dict` 这个运行时状态上。  
它会缓存：

- Receiver 自己的 KV
- 每个 Sharer 的 KV

后面的 projector 融合就是直接对这些缓存切片做读写。

### 1.3 这一节的核心理解

如果只看 3.1，可以把 C2C 理解成一句话：

**论文是在把“LLM 之间通信的对象”从文本换成 KV-Cache；代码是在把这件事做成一个真的可运行的自定义推理循环。**

---

## 2. 论文 3.2 Oracles for Cache-to-Cache Communication

这一节不是最终系统设计，而是论文在回答两个前置问题：

1. KV-Cache 的“语义质量”值不值得通信？
2. 不同模型的 KV-Cache 是否可以互相转换？

从工程上看，这一节对应的是仓库里的 oracle 分支与分析脚本，而不是主线 `RosettaModel + C2CProjector` 推理流程。

### 2.1 论文 3.2.1 Cache Enrichment Oracle

#### 论文在说什么

论文想区分两件事：

- few-shot 提升效果，到底是因为“上下文更长了”？
- 还是因为“问题 token 在有 exemplar 的前缀下，被编码得更好了”？

于是论文设计了三个设置：

1. `Direct`：只用问题 `X` 做 prefill，再 decode。
2. `Few-shot`：用 `E ⊕ X` 一起 prefill，再 decode。
3. `Oracle`：也先用 `E ⊕ X` 做 prefill，但把 exemplar 对应的 cache 丢掉，只保留与问题 `X` 对齐的那一段缓存：

`C*(X) = C[|E|:|E|+|X|](E ⊕ X)`

这样做的意义是：

- decode 时的 cache 长度和 `Direct` 一样长
- 但问题对应 token 的表示已经被 exemplar“润色/增强”过

如果这时效果依然变好，就说明真正有价值的是 cache 语义 enrichment，而不只是“多看了几个 token”。

论文还发现：不是每一层都适合被 enrichment。  
有些层会提升，有些层反而会拖后腿。  
这直接引出了后面 3.3.2 的 gating 设计。

#### 仓库中的对应实现

这一节在仓库里没有一个名字完全一样的 `cache_enrichment_oracle.py` 主入口，但有一组非常接近它的实验支撑代码：

- `rosetta/model/oracle.py`
- `script/evaluation/unified_evaluator_oracle.py`
- `script/train/oracle_train.py`
- `script/train/oracle_train_kvcache_mse.py`

`rosetta/model/oracle.py` 的思路和主线 `RosettaModel.forward()` 很像，也是：

1. 先拿到 Receiver/base 的 cache。
2. 再拿到 Sharer/source 的 cache。
3. 再根据配置把某些层的 source cache 投影/替换到 target cache 上。

它和主线实现不同的地方在于，oracle 分支更强调“分析”和“验证”：

- 会把 `projected_kv` 和 `target_kv` 保存到 `oracle/projected_kv/`、`oracle/target_kv/`
- 更方便做 T-SNE、MSE、层级效果等离线分析

也就是说，论文 3.2.1 的重点是“证明 cache enrichment 这件事本身有价值”，仓库里相应代码的重点也是“把 KV 修改前后保存下来做实验分析”，而不是直接服务最终产品化推理。

#### 这部分和后面主线实现的关系

这节 oracle 的真正作用，是给后面的主线设计两个结论：

1. Receiver 的缓存确实可以被“增强”。
2. 增强不应该无脑作用到所有层，因此需要 layer-wise 的选择机制。

这正是后面 gate 和 layer mapping 的理论前提。

### 2.2 论文 3.2.2 Cache Transformation Oracle

#### 论文在说什么

第二个 oracle 要回答的是：  
一个模型的 KV-Cache，能不能被另一个模型理解？

论文做法是训练一个 3-layer MLP，把 source LLM 的 KV-Cache 映射到 target LLM 的表示空间。  
T-SNE 图显示：

- 原始 source cache 和 target cache 分布相距很远
- 经过转换后，source cache 会落到 target cache 的表示空间内部

这说明不同模型的 KV-Cache 虽然不在同一个空间，但**可以被学习式地转换**。  
这就是 C2C projector 合法性的直接依据。

论文还观察到，转换后的 source cache 只覆盖 target 空间的一部分。  
这意味着：

- source 和 target 的语义结构并不完全相同
- 两个模型对同一上下文往往有互补理解

因此，最合理的设计不是“完全替换 Receiver 的 cache”，而是“把 Sharer 的信息作为增量融合进去”。

#### 仓库中的对应实现

这部分在仓库里主要对应三类代码：

1. `script/train/oracle_train.py`
2. `script/train/oracle_train_kvcache_mse.py`
3. `script/analysis/tsne/tsne.py`

`oracle_train.py` 和 `oracle_train_kvcache_mse.py` 代表了“先证明可转换，再走主线任务损失”的实验分支。  
它们和主线 `SFT_train.py` 的区别在于：

- 主线训练用的是回答任务的 next-token prediction loss
- oracle 分支更偏表征层面的监督，例如 hidden states / KV 的 MSE 或相似度损失

所以这两份脚本更像论文 3.2.2 的实验实现，而不是最终 C2C 系统的默认训练方式。

`script/analysis/tsne/tsne.py` 则直接对应论文里“转换前后表示空间可视化”的分析思路。  
也就是说：

- 论文先用 oracle 说明“空间可变换”
- 仓库主线再把这个可变换性收敛成真正可部署的 `C2CProjector`

### 2.3 这一节的核心理解

3.2 的两个 oracle 合起来，实际上给 3.3 铺了三层逻辑：

1. KV-Cache 不是纯粹的计算副产物，而是有可传递语义的。
2. 不同模型的 cache 空间可以通过可学习变换进行对接。
3. 更合理的设计是“增量融合”，而不是粗暴替换。

这三点，后面都会在 `C2CProjector` 和 `RosettaModel` 里具体落地。

---

## 3. 论文 3.3 C2C Design

这一节就是仓库主线实现的核心。

### 3.3.1 Overview

#### 论文在说什么

论文把 C2C 形式化为：

- 一组 cache fuser `F`
- 一个层映射策略 `G`

对 Receiver 的第 `n` 层 cache `Cn(X)` 和 Sharer 对应层 `C^S_{G(n)}(X)`，生成 fused cache：

`C^F_n = Cn(X) + Fn(Cn(X), C^S_{G(n)}(X))`

这条公式非常重要，因为它直接说明了三件事：

1. C2C 不是替换 Receiver，而是增强 Receiver。
2. 融合是逐层发生的，不是整个 cache 一次性混合。
3. Sharer 先经过映射 `G(n)` 找到应对应的层，再由 fuser `Fn` 处理。

之后 decode 用的就是 fused cache：

`y(i+1) = P(yi | C^F(X) ⊕ C(Y[0:i]))`

也就是说，Sharer 的作用主要发生在 prefill 阶段；之后 Receiver 在自己的生成轨道上继续往下走。

#### 仓库中的对应实现

这部分最直接对应 `rosetta/model/wrapper.py`。

#### `RosettaModel`：把公式变成可执行对象

`RosettaModel` 本质上就是论文 3.3.1 的整体运行时容器。  
它内部有三个关键成员：

- `model_list`：`[base_model, teacher_model1, teacher_model2, ...]`
- `projector_list`：所有 fuser / projector 实例
- `projector_dict`：层映射关系

#### `projector_dict`：就是论文里的 `G`

`set_projector_config()` 负责把“source 哪一层 -> target 哪一层 -> 用哪个 projector”登记到 `projector_dict` 里。  
这相当于把论文中的层映射函数 `G`，落成了一个工程可查询的数据结构。

#### `forward()`：逐段、逐层地做 cache fusion

`RosettaModel.forward()` 的核心流程就是论文 3.3.1 的直接实现：

1. 先对 Receiver 做 forward，得到 target cache。
2. 对每个 Sharer 做 forward，得到 source cache。
3. 按 `projector_dict` 找到 target layer 和 source layer 的对应关系。
4. 调用 projector，得到融合后的 key/value。
5. 把结果写回 Receiver 的 `curr_base_kv_cache`。

如果只用单 Sharer，这就是最标准的论文版 C2C。  
当前仓库还额外支持多 Sharer，这属于论文之后的工程扩展。

#### 评测时如何重建这套结构

训练好之后，`rosetta/utils/evaluate.py` 里的 `load_rosetta_model()` 会负责：

1. 加载 base model 与 teacher model。
2. 从 checkpoint 目录加载 `projector_*.json` 和 `projector_*.pt`。
3. 重建 `RosettaModel`。
4. 从 `projector_config.json` 恢复层映射。

这一步说明论文里的抽象设计在仓库里不是“写死”的，而是可以训练、保存、重建、评测的一套完整对象系统。

### 3.3.2 Fuser Structure

#### 论文在说什么

论文强调一个原则：  
Sharer 的信息不能破坏性地覆盖 Receiver 原本的语义，所以 fuser 必须遵循 residual integration。

论文把 fuser 分成三个模块：

1. Projection module  
   把 Receiver 的 KV 和 Sharer 的 KV 拼起来，做投影与特征融合。

2. Dynamic weighting module  
   做输入相关的动态重加权，决定当前样本、当前 token、当前 head 到底该吸收多少 Sharer 信息。

3. Learnable gate  
   用可学习 gate 决定某一层是否值得注入 Sharer 语义，并通过 Gumbel-sigmoid + temperature annealing，让训练期可微、推理期更接近离散开关。

这部分其实就是整篇论文最核心的技术点：  
**如何把“Sharer 带来的额外语义”以增量形式写进 Receiver 的 cache，而不把 Receiver 自己原来的上下文理解冲掉。**

#### 仓库中的对应实现

这部分几乎完整落在 `rosetta/model/projector.py` 的 `C2CProjector` 上。

#### `C2CProjector.__init__()`：把论文的三个模块拆成具体层

`C2CProjector` 的内部结构和论文描述是一一对应的：

- `key_in` / `value_in`
  先把 source 与 target 的展平特征拼接起来，再映射到隐藏维度  
  对应论文里的 projection module 起点

- `key_mlp1` / `value_mlp1`
  进一步做公共特征提取  
  对应论文里 feature fusion 的主干

- `key_proj_mlp2` / `value_proj_mlp2` + `key_proj_out` / `value_proj_out`
  产出真正要注入 target 的 projected key / value

- `key_scalar_mlp2` / `value_scalar_mlp2` + `key_scalar_head` / `value_scalar_head`
  产出动态权重  
  这就是 dynamic weighting module

- `key_gate_logit` / `value_gate_logit`
  这是 learnable gate 的参数

#### `forward()`：把论文里的融合公式真正算出来

`C2CProjector.forward()` 的实际计算顺序是：

1. 输入 `source_kv` 和 `target_kv`
2. 将 `(B, H, N, D)` 形式的 key/value 展平为 `(B, N, H*D)`
3. 把 source 与 target 在通道维拼接
4. 经过 MLP 提取中间表示
5. 一路得到 `projected_key` / `projected_value`
6. 一路得到 `key_scalar` / `value_scalar`
7. 对 gate logit 做 Gumbel-sigmoid
8. 最终输出：

`output_key = target_key + key_gate * norm_key_scalar * projected_key`

`output_value = target_value + value_gate * norm_value_scalar * projected_value`

这和论文 3.3.2 的 residual integration 思路完全一致：  
不是替换 target，而是做一个“可控的加法增量”。

#### 论文和代码之间的一个细节差异

论文文字里把 gate 讲成“per-layer gate”。  
代码里更细一点：

- 每个 projector 本身通常对应一个 target layer
- projector 内部又拆成 `key_gate_logit` 和 `value_gate_logit`

所以代码实际上是“每层两个 gate 参数，分别管 K 和 V”。  
训练时它们通过 Gumbel 噪声变成可微门控；推理时则变成更接近 0/1 的开关。

#### `update_temperature()`：对应论文的 temperature annealing

论文里提到 gate 使用 temperature annealing。  
代码里 `C2CProjector.update_temperature()` 会按训练步数把 `gate_temperature` 从初始值逐渐退火到更低值。

在 `script/train/SFT_train.py` 的训练循环中，每次优化器 step 之后都会调用每个 projector 的 `update_temperature(global_step)`。  
这说明“可微 gate -> 接近离散 gate”的训练过程不是论文里的概念描述，而是被真实写进训练循环了。

### 3.3.3 Model Alignment

#### 论文在说什么

论文指出跨模型融合要解决两个对齐问题：

1. token 对齐
2. layer 对齐

##### token 对齐

不同 tokenizer 对同一个字符串可能切成不同 token。  
论文的办法是：

1. 取 Receiver 的 token
2. 解码成字符串
3. 用 Sharer 的 tokenizer 重新编码
4. 如果出现 one-to-many，就选覆盖字符串最多的那个 token

##### layer 对齐

论文采用 terminal alignment：

- 最后一层对最后一层
- 倒数第二层对倒数第二层
- 一直往前配

本质上是在优先对齐两个模型更“深语义”的部分。

#### 仓库中的对应实现

这一节在代码里拆成两部分：

- `rosetta/model/aligner.py`
- `rosetta/train/model_utils.py`

#### token 对齐：`TokenAligner`

`TokenAligner.align_tokens()` 正是按论文思路做的：

1. 先把 base/SLM 的 token 解码成字符串
2. 再用 teacher/LLM 的 tokenizer 重新编码
3. 如果是一对多映射，就按策略选一个

论文默认更接近“最大字符串覆盖”，仓库里对应的是 `AlignmentStrategy.LONGEST`。  
如果配置里写的是 `"alignment_strategy": "longest"`，就和论文描述最一致。

#### chat 场景下的工程化补充：`align_chat_messages()`

论文主文只说了 token 对齐原则，但仓库实现更细，因为真实输入不是裸文本，而是 chat template。

`align_chat_messages()` 会把整个 chat 序列拆成：

- template section
- message section
- template section
- message section

然后分别处理：

- template 部分不做语义映射，直接 padding 对齐
- message 部分才做 token 级语义对齐

这是一个非常工程化、但也非常合理的补充。  
因为模板 token 主要承担格式作用，不应该强行拿来做语义通信。

#### layer 对齐：`last_aligned_sources()`

`rosetta/train/model_utils.py` 里的 `last_aligned_sources()` 就是论文 terminal alignment 的直接实现：

- 先把 target 最后几层和 source 最后几层对齐
- 再向前回溯

训练入口 `script/train/SFT_train.py` 默认会根据配置里的 `"mapping": "last_aligned"` 选这个策略。  
如果选 `"k_nearest"`，那是仓库额外提供的替代策略，不是论文主文默认方案。

#### `AlignedChatDataset`：把对齐结果真正喂给模型

`rosetta/train/dataset_adapters.py` 中的 `AlignedChatDataset` 会：

1. 调用 `align_chat_messages()` 得到 base 和 teacher 的对齐 token 序列
2. 生成两路 `input_ids`
3. 用 `message_mask` 标出真正有语义内容的位置
4. 把非 message 的部分在 `kv_cache_index` 中置为 `[-1, 0]`

这一步很关键，因为它说明：

- 论文里的 token alignment 在代码里不是“离线分析”
- 而是直接决定训练和推理时哪些 token 段允许发生 C2C 融合

### 3.3.4 Training Scheme

#### 论文在说什么

论文的训练方案可以概括成三步：

1. Forward  
   Sharer 和 Receiver 先各自编码输入，得到各自的 KV-Cache。

2. Fusion  
   用 C2C 模块把 Sharer 的缓存融合进 Receiver。

3. Supervision  
   让 Receiver 在 fused cache 的条件下继续做响应预测，并对回答部分计算标准 next-token prediction loss。

这里最重要的两个思想是：

- base model 和 sharer model 都冻结
- 只训练中间的 C2C 模块

也就是说，C2C 不是再微调整个大模型，而是在两个现成模型之间学习一个“缓存翻译与融合器”。

#### 仓库中的对应实现

这部分主要落在 `script/train/SFT_train.py` 和 `rosetta/train/dataset_adapters.py`。

#### 第一步：组装模型

`SFT_train.py` 的 `setup_models()` 会：

1. 加载 `base_model`
2. 加载 `teacher_model`
3. 根据配置创建一组 `C2CProjector`
4. 构造 `RosettaModel`
5. 根据 `mapping` 生成层映射
6. 用 `set_projector_config()` 把映射写入 `projector_dict`

如果看训练 recipe，例如 `recipe/train_recipe/C2C_0.6+0.5.json`，可以看到默认配置就是：

- `projector.type = C2CProjector`
- `mapping = last_aligned`
- `freeze = ["teacher", "base"]`

这和论文的默认训练设定完全一致。

#### 第二步：冻结大模型，只训练 projector

`SFT_train.py` 里会根据 `training.freeze` 控制参数是否训练。  
在默认 C2C recipe 下：

- base model 冻结
- teacher model 冻结
- projector 不冻结

因此真正更新的只有 C2C 模块参数。

#### 第三步：把一条样本切成“通信段”和“监督段”

`rosetta/train/dataset_adapters.py` 的 `generate_kv_cache_index()` 是理解训练流程的关键函数。

它会把一个样本分成两段：

- instruction 段：标成 `[1, 0]`
- response 段：标成 `[-1, 0]`

这表示：

- 在 instruction/context 这段，允许 Sharer 参与 cache 融合
- 到 response 生成时，不再继续引入新的 Sharer cache，而是让 Receiver 基于已经融合好的上下文继续预测

这非常接近论文 3.3.4 的描述：  
**先融合上下文 cache，再监督回答。**

#### `ChatDataset` / `AlignedChatDataset`：监督只落在回答部分

这两个数据集包装器都会把：

- instruction 对应位置的 label 设成 `-100`
- 只保留回答部分参与 loss

因此训练目标确实是论文说的标准 next-token prediction loss，只不过 supervision 聚焦在 response 部分。

#### `RosettaDataCollator`：把 `kv_cache_index` 变成分段 forward

`RosettaDataCollator` 会根据 `kv_cache_index` 的变化点把序列切成多个 section。  
之后 batch 输出里的 `kv_cache_index` 不是一整条长向量，而是一组“按段切开的张量”。

这一步非常关键，因为 `RosettaModel.forward()` 正是按这些 section 依次执行：

1. 前一段做 context/instruction prefill
2. 在段末更新 fused cache
3. 后一段继续 forward，并对 response 计算 loss

因此论文里的“三阶段训练流程”，在代码里并不是三份分离脚本，而是被压进了：

- dataset 标注
- collator 切段
- `RosettaModel.forward()` 的顺序执行

#### 训练循环中的 projector 退火与保存

训练主循环还做了两件和论文细节强相关的事情：

1. 每次优化 step 后更新 projector 的 gate temperature
2. 保存 `projector_i.pt`、`projector_i.json` 和 `projector_config.json`

这意味着训练完成后，整个 C2C 系统可以被完整重建：

- projector 权重在 `.pt`
- projector 结构参数在 `.json`
- 层映射关系在 `projector_config.json`

评测时 `load_rosetta_model()` 会把它们全部接回去。

### 3.3 小结

如果把 3.3 压缩成一句工程语言，可以这样理解：

**C2C 在代码里就是“先让多个模型各自预填充上下文，再用 projector 按层把 Sharer 的缓存增量写入 Receiver，然后只训练这个写入器，让 Receiver 在融合后的缓存上完成回答预测”。**

---

## 4. 按论文顺序看，整套系统是怎样串起来的

如果严格按论文技术部分的顺序，把仓库里的执行链串起来，可以得到下面这条主线：

1. `3.1` 告诉我们：LLM 真正用于后续生成的是 KV-Cache，而不是输入文本本身。
2. `3.2.1` 告诉我们：同样长度的 cache，只要语义更好，答案就可能更好。
3. `3.2.2` 告诉我们：不同模型的 cache 空间可以通过可学习模块转换。
4. `3.3.1` 于是定义整体范式：Sharer 的某层缓存，经映射后写入 Receiver 对应层。
5. `3.3.2` 进一步给出具体融合器：投影、动态权重、门控、残差注入。
6. `3.3.3` 解决跨模型 token/layer 不对齐的问题。
7. `3.3.4` 用标准 SFT 风格的回答损失，只训练 projector，让整个系统真正可学。

而在仓库中，这条链路分别落成了：

1. `RosettaModel.generate()` / `RosettaModel.forward()`
2. `oracle.py` + `oracle_train*.py` + `tsne.py`
3. `C2CProjector`
4. `TokenAligner` + `last_aligned_sources()`
5. `ChatDataset` / `AlignedChatDataset` + `RosettaDataCollator`
6. `SFT_train.py`
7. `load_rosetta_model()` + `unified_evaluator.py`

---

## 5. 论文和当前仓库实现之间，最值得注意的几个差异

### 5.1 论文主文是单 Sharer 抽象，仓库已经扩展到多 Sharer

当前 `RosettaModel` 支持多个 sharer，并提供：

- `sequential` 融合模式
- `parallel` 融合模式

这是仓库对论文主线的工程扩展。

### 5.2 论文把 gate 说成 per-layer，代码里拆成了 K/V 两个 gate

代码中 `C2CProjector` 有：

- `key_gate_logit`
- `value_gate_logit`

所以实现比论文主文更细化。

### 5.3 论文主文讲的是抽象 token 对齐，代码把 chat template 也纳入了对齐逻辑

`align_chat_messages()` 对 template section 和 message section 做了不同处理。  
这是工程实现里非常重要，但论文主文里不会铺开写的细节。

### 5.4 论文讲“融合后再解码”，代码通过 `kv_cache_index` 把这个过程切成可控的 section

也就是说，仓库真正的关键控制信号并不是一条抽象公式，而是 `kv_cache_index`。  
它决定：

- 哪一段启用 C2C
- 哪一段只做普通 Receiver 预测

这是理解训练和推理代码时最重要的工程抓手。

---

## 6. 一句话总结

论文从“KV-Cache 是否值得通信、是否可以转换”出发，最终提出了一个逐层、残差式、门控化的缓存融合范式。  
仓库则把这个范式实现成了一个完整可训练系统：`RosettaModel` 负责运行时缓存读写，`C2CProjector` 负责跨模型缓存投影与融合，`TokenAligner` 和 layer mapping 负责对齐，`SFT_train.py` 则用回答损失把整个中间模块训练出来。

如果你只记一句话，可以记成：

**C2C 不是让一个模型“读另一个模型写出来的话”，而是让一个模型“直接读另一个模型已经算好的上下文语义缓存”。**
