# C2C Route-1 v2.x 迭代总结与后续安排备忘

日期：2026-05-03

## 1. 背景问题

当前研究目标是改进 C2C 中跨 tokenizer 的 cache communication。原始代码中的跨模型对齐方式可以工作，但方法上比较启发式：

- 对 source token 单独 decode 成字符串；
- 再用 target tokenizer 对该字符串重新分词；
- 当一个 source token 对应多个 target token 时，通常只保留第一个或最长的一个；
- 最终仍然是 hard 1:1 token 对齐。

这种方式的问题是：它忽略了真实 tokenizer mismatch 中常见的 many-to-many 分段差异。尤其在 Qwen3/TinyLlama 这种 tokenizer 差异较大的模型对上，单 token decode 和 hard index 选择容易造成边界错配、信息丢失和 KV 注入噪声。

因此，Route-1 的核心目标是：

> 将原始 hard heuristic alignment 升级为基于 offset mapping 的 span-aware soft alignment，让跨 tokenizer 的 KV 通信从单点硬匹配变成多 token 柔性聚合。

这对应原研究备忘录中的方向 A：

> 基于 offset mapping 的字符跨度对齐。

并进一步覆盖路线 1：

> Span-overlap soft alignment。

## 2. v2.x 整体迭代脉络

v2.x 系列主要围绕 Route-1 做迭代。整体逻辑不是重新设计整个 C2C，而是在保留原生 C2C projector 和训练流程的基础上，逐步替换跨 tokenizer 对齐和 KV 注入方式。

| 阶段 | 核心改动 | 目的 | 结论 |
| --- | --- | --- | --- |
| 初始 hard baseline | `longest` / hard `span_overlap` | 验证 hard 对齐是否足够 | 强 mismatch 下 hard span 甚至弱于 longest |
| v1 soft span | `soft_span_overlap` top-k soft alignment | 从 hard 1:1 改成 soft top-k | 在 Qwen3/TinyLlama stress test 上明显优于 hard 方法 |
| v2.0 | `soft_span_overlap_v2`，尝试不同 score mode | 比较 overlap / uniform / boundary 等权重策略 | `uniform` 最稳，过度相信 overlap 大小会伤害下游泛化 |
| v2.1 | entropy-based source confidence | 对高歧义 span 降低 KV 注入强度 | `entropy050` 效果最好 |
| v2.2 | learnable token/head confidence gate | 在 v2.1 静态 confidence 基础上学习局部 KV 注入强度 | 当前最强主线结果 |
| v2.3-v2.4 | confidence delta L2 正则 | 限制 gate 偏移，避免过拟合 | 没超过 v2.2 |
| v2.5-v2.6 | layer-aware / learned layer scale | 利用不同层对 key/value 的不同敏感性 | 没超过 v2.2 |
| v2.7 | adaptive overlap reweighting | 动态调整 top-k 内部权重 | 接近 v2.2，但没有超过 |
| v2.8-v2.8b | span MLP weight calibration | 让 projector 学习校准 span 权重 | eval loss 更低，但 benchmark 不升 |
| v2.9 | learned residual scale | 学习 KV residual 注入强度 | 负结果 |
| v2.10-v2.11b | answer prior / answer margin | 诊断 benchmark answer bias | 有诊断价值，但不适合作为主线 |
| v2.12a | alignment-quality-aware token gate | 给 token gate 加入 alignment quality features | 小闭环 loss 降低，但 benchmark 明显下降 |

核心结论：

> v2.2 之后继续堆小型 gate、scale、calibration 模块，边际收益很低，且容易出现 validation loss 下降但 benchmark 泛化下降的问题。

## 3. v2.2 重点做了什么

v2.2 的完整方法可以概括为：

> offset-mapping based span-overlap candidate construction + top-k soft KV aggregation + entropy confidence seed + learnable token/head confidence gate。

它不是单纯的 hard span overlap，而是方向 A 的增强版。

### 3.1 基于完整上下文的 offset mapping

v2.2 不再对单个 source token 做 isolated decode，然后重新 tokenize。它先对完整 chat template tokenize，并要求 tokenizer 返回 offset mapping：

- source/receiver tokenizer 得到 `slm_ids` 和 `slm_offsets`；
- target/sharer tokenizer 得到 `llm_ids` 和 `llm_offsets`；
- 再根据 message content 在各自 template 中的字符位置，找到 message span。

对应代码：

- `rosetta/model/aligner.py::_apply_chat_template_to_ids`
- `rosetta/model/aligner.py::_compute_content_spans`
- `rosetta/model/aligner.py::_spans_to_token_ranges`

这一步解决的问题是：

- 避免单 token decode 造成的残缺字符串；
- 避免重新分词破坏真实上下文边界；
- 让对齐依据来自真实 prompt 中的 token offset。

### 3.2 基于字符跨度重叠构造候选

对于每个 receiver token，v2.2 会：

1. 取它在 receiver message 中的字符跨度；
2. 转换为 message 内的相对字符位置；
3. 投影到 sharer message 的对应字符跨度；
4. 收集所有与该跨度有重叠的 sharer token；
5. 保留 top-k 候选。

对应代码：

- `rosetta/model/aligner.py::_soft_align_message_by_span_overlap`

这一步解决的问题是：

- 不再强行把跨 tokenizer 对齐压成一个 index；
- 能表达 one-to-many 和 many-to-one 的 tokenizer mismatch；
- 给后续 KV 聚合保留多个候选，而不是过早丢弃信息。

### 3.3 top-k soft KV 聚合

v2.2 在 wrapper 中根据 `source_indices` 和 `source_weights` 聚合多个 sharer token 的 KV：

```text
KV_aligned[i] = sum_j w_ij * KV_sharer[j]
```

对应代码：

- `rosetta/model/wrapper.py::_weighted_source_kv_from_indices`

这一步是从 hard alignment 到 soft alignment 的关键变化。它解决的问题是：

- hard `longest` 或 hard `span_overlap` 只能注入一个 token 的 KV；
- 当 tokenizer 边界不一致时，单个 token 可能只覆盖部分语义；
- soft KV 聚合能把局部 span 内多个 sharer token 的信息合成到 receiver token 位置。

### 3.4 uniform weighting 的发现

v2.2 使用的最佳配置是：

```json
"soft_alignment_score_mode": "uniform"
```

这点很重要。虽然方向 A 直觉上会使用 overlap ratio 作为权重，但实验发现，在强 tokenizer mismatch 下，过度相信 overlap 大小并不稳定。

v2.0 对比中：

| 方法 | MMLU-Redux | AI2-ARC | OpenBookQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| `soft_span_overlap_control` | 42.97 | 49.39 | 41.60 | 44.65 |
| `soft_span_overlap_v2_uniform` | 44.83 | 52.61 | 47.00 | 48.15 |
| `soft_span_overlap_v2_overlap_power2` | 42.99 | 51.91 | 44.60 | 46.50 |
| `soft_span_overlap_v2_boundary_power2` | 43.79 | 51.74 | 43.60 | 46.37 |

结论：

> 对候选 token 做 soft 保留比精细化 overlap 权重更关键。uniform top-k 更像一种保守的 span evidence pooling，泛化更稳。

### 3.5 entropy confidence seed

v2.1 引入 entropy confidence，v2.2 沿用最佳设置：

```json
"soft_alignment_confidence_mode": "entropy",
"soft_alignment_confidence_alpha": 0.5,
"soft_alignment_confidence_floor": 0.5,
"soft_alignment_fallback_confidence": 0.25
```

直觉是：如果一个 receiver token 对应多个 sharer 候选，且权重分布更均匀，说明这个位置的对齐更不确定。此时应该降低 sharer KV residual 的注入强度。

这一步解决的问题是：

- soft alignment 虽然保留了更多信息，但也可能带来不确定性；
- entropy confidence 让高歧义位置少注入，低歧义位置正常注入；
- 它把 alignment uncertainty 显式传给 projector。

v2.1 中最佳结果是 `adaptive_entropy050`：

| 方法 | MMLU-Redux | AI2-ARC | OpenBookQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| `soft_span_overlap_v2_uniform_control` | 44.83 | 52.61 | 47.00 | 48.15 |
| `adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 |

### 3.6 learnable token/head confidence gate

v2.2 最核心的新机制是 `alignment_confidence_gate_mode: token_mlp`。

它不是直接替换 entropy confidence，而是在 entropy confidence 的 logit 上学习一个局部 delta：

```text
base_logit = logit(source_confidence)
key_confidence = sigmoid(base_logit + key_delta)
value_confidence = sigmoid(base_logit + value_delta)
```

其中：

- `key_delta` 和 `value_delta` 由 projector hidden state 预测；
- 预测粒度是 token/head 级别；
- key 和 value 分开学习；
- 初始 head 权重为 0，因此训练开始时等价于 v2.1 entropy confidence。

对应代码：

- `rosetta/model/projector.py::key_alignment_confidence_head`
- `rosetta/model/projector.py::_compute_alignment_confidence`

这一步解决的问题是：

- 静态 entropy confidence 只能根据 alignment 权重判断不确定性；
- 它不知道当前 token 的 KV 内容是否真的有帮助；
- token MLP gate 让 projector 根据当前 hidden 表示，自适应决定 key/value 注入强度。

因此，v2.2 的真正提升来自：

> span-level soft alignment 提供候选 KV，entropy confidence 提供不确定性先验，token MLP gate 学习局部修正。

## 4. v2.2 带来的效果

### 4.1 相比 hard alignment

在 Qwen3/TinyLlama 强 mismatch stress test 中：

| 方法 | MMLU-Redux | AI2-ARC | OpenBookQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| `longest` | 45.53 | 56.09 | 48.80 | 50.14 |
| hard `span_overlap` | 43.48 | 52.61 | 40.80 | 45.63 |
| `soft_span_overlap` | 47.07 | 58.17 | 53.80 | 53.01 |

这说明：

- hard `span_overlap` 不一定比 `longest` 好；
- 单点 span 选择会放大边界噪声；
- soft weighted KV gathering 才是关键。

### 4.2 相比 v2.1

v2.2 在小闭环中是当前最强 Route-1 版本：

| 方法 | MMLU-Redux | AI2-ARC | OpenBookQA | Mean |
| --- | ---: | ---: | ---: | ---: |
| `v2.1_adaptive_entropy050` | 45.83 | 54.09 | 49.40 | 49.77 |
| `v2.2_token_mlp_entropy050` | 47.07 | 54.78 | 50.60 | 50.82 |

提升：

- MMLU-Redux：+1.24
- AI2-ARC：+0.69
- OpenBookQA：+1.20
- Mean：+1.05

这说明 token/head 级别的 learnable confidence gate 确实能在静态 entropy confidence 基础上继续提高下游表现。

### 4.3 一个重要负面发现

v2.2 的另一个重要结论是：

> validation loss 不是 Route-1 alignment 质量的可靠代理。

例如：

| 方法 | Final eval loss | Benchmark Mean |
| --- | ---: | ---: |
| `v2.2_token_mlp_entropy050` | 0.1694 | 50.82 |
| `v2.2_learned_affine_entropy050` | 0.1243 | 46.49 |
| `v2.12a_alignment_quality_token_mlp_entropy050` | 0.1007 | 48.77 |

这说明很多后续模块虽然能让小闭环 loss 下降，但会损害 benchmark 泛化。因此后续研究不能只看训练/验证 loss，必须坚持 MMLU-Redux、AI2-ARC、OpenBookQA 的下游评测。

## 5. v2.2 是否满足方向 A

结论：满足，而且超过了方向 A 的最小实现。

方向 A 的要求是：

- 使用 source tokenizer offset mapping；
- 使用 target tokenizer offset mapping；
- 通过字符跨度重叠建立 source-token 和 target-token 的对齐关系。

v2.2 已经实现：

- 完整 chat template 级别的 offset mapping；
- message content span 定位；
- source span 到 target span 的相对字符投影；
- top-k candidate construction；
- soft KV aggregation；
- alignment confidence gating。

需要注意的是，v2.2 最佳配置没有直接使用 overlap ratio 作为最终权重，而是使用 `uniform` score mode。这个选择不是偏离方向 A，而是实验得到的更稳健设计：

> offset overlap 负责找候选，uniform soft aggregation 负责保留多 token 证据，confidence gate 负责控制注入强度。

因此，在论文表述中，v2.2 不应被描述成普通的 overlap-ratio hard alignment，而应描述为：

> Offset-guided soft span alignment with adaptive confidence-gated KV aggregation.

## 6. 为什么 v2.2 后续的小迭代不应继续作为主线

v2.3 到 v2.12a 的结果显示：继续围绕 v2.2 增加小型 gate、scale、regularization、answer-aware loss，收益不稳定。

主要负结果包括：

- delta L2 正则没有提升；
- static / learned layer scaling 没有提升；
- adaptive overlap reweighting 接近但没有超过；
- span MLP calibration 降低 eval loss 但没有提升 benchmark；
- answer prior / answer margin 对诊断有用，但容易引入答案分布偏移；
- alignment-quality-aware token gate 明显降低小闭环 loss，但 benchmark 全面下降。

这说明当前瓶颈不是“再加一个小 gate”，而是：

> 手工构造的 soft alignment 已经接近这一阶段的上限，下一步需要更有方法性的 learned flexible alignment。

## 7. 后续安排建议

### 7.1 短期：固化 v2.2 作为主线方法

接下来应该先把 v2.2 作为主线结果固定下来，而不是继续随意扩展模块。

建议补齐：

- Qwen3/TinyLlama strong mismatch；
- Qwen3/Llama-3.2 mild mismatch；
- `longest` / hard `span_overlap` / `soft_span_overlap_v2` / `v2.2_token_mlp` 对照；
- MMLU-Redux / AI2-ARC / OpenBookQA；
- invalid output、answer prior、prediction flip；
- alignment diagnostics：one-to-many、many-to-one、entropy、fallback rate、token length ratio。

目标是形成论文中的第一张核心实验表：

> Soft span alignment is most useful when tokenizer mismatch is large.

### 7.2 中期：从方向 A 推进到路线 3

如果继续追求更强创新，不建议继续在 v2.2 上堆小补丁，而建议开一个新分支：

> Candidate-constrained learned flexible alignment。

思路：

- dataset 层仍然用方向 A 的 offset mapping 生成候选 span；
- 不再固定 top-k 权重；
- projector 内部根据 source KV、target KV、span features 学习候选权重；
- alignment weight 与 cache projection 联合训练；
- 用 entropy / KL / sparse prior 控制学习出的 alignment 不要退化。

这对应原备忘录中的路线 3：

> Projector 内部学习 alignment。

它比继续调 confidence gate 更适合形成方法论文，因为它把 alignment 从规则模块推进成可学习模块。

### 7.3 论文定位

当前最稳的论文叙事是：

1. 原始 C2C 跨 tokenizer 通信依赖 hard heuristic alignment；
2. tokenizer mismatch 本质是 many-to-many span segmentation mismatch；
3. offset mapping 可以提供 tokenizer-agnostic 的字符跨度候选；
4. soft KV aggregation 比 hard single-index mapping 更稳；
5. confidence-gated injection 能进一步控制 alignment uncertainty；
6. v2.2 是当前最强实证版本；
7. 下一步 learned flexible alignment 是更强方法版本。

可选题目方向：

> Soft Span Alignment for Cross-Tokenizer Cache Communication

或者更偏方法：

> Candidate-Constrained Learned Alignment for Cross-Tokenizer KV Cache Communication

## 8. 当前建议

不建议现在换到完全无关的新方向。

跨 tokenizer alignment 已经被实验初步验证有效，尤其是在 Qwen3/TinyLlama strong mismatch 上，soft alignment 明显优于 hard alignment。

但也不建议继续沿 v2.3-v2.12a 这种局部小模块路线反复试。更合理的下一步是：

1. 固化 v2.2，补完整实验矩阵和诊断；
2. 把 v2.2 写成当前主线方法；
3. 新开一个 learned flexible alignment 分支，作为更强的后续方法候选；
4. 如果 learned 分支有效，再把论文主方法从 v2.2 升级为 learned alignment；
5. 如果 learned 分支不稳定，则以 v2.2 为主方法，后续分支作为负结果和分析支撑。

一句话总结：

> Route-1 方向是值得继续的；v2.2 已经完成方向 A 的核心落地；后续不要再堆小 gate，而应从规则型 soft span alignment 推进到候选约束下的 learned flexible alignment。
