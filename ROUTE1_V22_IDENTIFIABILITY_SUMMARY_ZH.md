# Route-1 v2.2 identifiability 最终机制结论

## 完成状态

- 67/67 runs 完成，failed marker 为 0。
- 四个模型对、seeds 42/43/44、三个开发任务均纳入最终报告。
- 严格报告合同完整：每个模型对的 B2/B5/B6 均具有 3 seeds；任务行数为 ARC 1150、MMLU-Redux 5615、OpenBookQA 500。
- 统计使用 5,000 次分层 paired bootstrap；跨模型结论以 pair→seed→example 的层级 CI 为准，不能用聚合 McNemar 的小 p 值替代跨 pair 稳定性。

## 三个问题的回答

### 1. v2.2 的提升是否来自保留多个 source candidates？

不能作为全局结论成立。

- 全模型对 B3−B2 为 `+1.29 pp`，95% cluster CI `[-0.61, +3.58] pp`，CI 跨 0。
- 只有 Qwen3-1.7B→Qwen3-0.6B 显著：`+3.12 pp`，CI `[+0.47, +6.59] pp`。
- TinyLlama 为 `+2.09 pp`，但 seed 方差很大，CI `[-4.85, +6.50] pp`。
- Qwen2.5 为 `-0.07 pp`，Llama3.2 约为 `+0.01 pp`。
- Qwen3-1.7B 的收益并不只出现在 one-to-many：1-to-1 bucket 为 `+3.10 pp`，one-to-many 为 `+4.49 pp`。因此即便该模型对支持 soft candidates，也不能把全部收益解释成跨 tokenizer ambiguity coverage。

结论：multiple candidates 在特定模型对有价值，但不是 v2.2 跨模型稳定提升的主要、充分解释。

### 2. entropy confidence 是否真的提供有效信息？

有局部反事实证据，但没有证明 static entropy 单独具有稳定贡献。

- TinyLlama 三 seed 的 B4−B3 为 `+1.29 pp`，CI `[-2.78, +5.98] pp`，static entropy 独立贡献不显著。
- TinyLlama seed 42 的反事实中，B6−constant 为 `+0.94 pp`，CI `[+0.11, +1.75] pp`。
- B6−shuffle 为 `+2.28 pp`，CI `[+1.39, +3.18] pp`。打乱位置对应关系会显著变差，说明 entropy 的数值和位置对应关系在该 pair 上含有信息。
- per-example confidence 相关性总体很弱，绝对 Pearson r 通常小于 0.06。Qwen3/Qwen2.5 上 confidence 与正确率小幅正相关、与负迁移小幅负相关，但 TinyLlama 上方向混合或相反。

结论：不能把 v2.2 描述为已经跨模型验证的 entropy-aware 方法；更准确的说法是 entropy 在 TinyLlama 反事实中提供了可检测信息，但 static entropy 的普适、独立贡献尚未成立。

### 3. token/head gate 是否只是增加了自适应容量？

没有证据证明额外 adaptive capacity 本身带来稳定收益，而且完整 B6 的 gate 大多已经饱和。

- 干净的 TinyLlama B5−B2-constant 为 `+0.93 pp`，CI `[-2.66, +3.90] pp`，不显著。
- 跨模型但 confidence-confounded 的 B5−B2 为 `+0.35 pp`，CI `[-1.15, +2.13] pp`，也不显著。
- B5 post-hoc gate 确实具有一定动态性：global mean 约 0.92，key gate 的变化和饱和率高于 value gate；但这种动态没有转化为稳定 accuracy 增益。
- 完整 B6 在 Qwen3、Qwen2.5 上约 99.93% gate 处于高饱和区，Llama3.2 为 100%，TinyLlama 也约为 85.86%。early/middle/late 与 K/V 均接近始终开启。

结论：当前结果不支持“token/head gate 通过自适应选择产生主要收益”。B6 更像 entropy/static scale 把 gate 推到近乎 always-on 后，与投影器和 soft transport 共同改变训练轨迹。

## 三 seed sample-weighted accuracy

| Pair | B2 hard span | B3 soft span | B5 gate-only | B6 full |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 42.42±2.89 | 44.51±3.38 | 44.63±2.64 | **47.13±1.24** |
| Qwen3-1.7B | 47.52±2.66 | **50.63±1.84** | 46.93±2.18 | 50.27±0.74 |
| Qwen2.5-0.5B | **45.11±0.88** | 45.04±0.80 | 44.73±2.24 | 43.62±4.19 |
| Llama3.2-1B | 46.52±0.74 | 46.54±0.75 | 46.69±0.20 | 46.69±0.97 |

## B6 三任务 accuracy mean±std

| Pair | ARC | MMLU-Redux | OpenBookQA |
| --- | ---: | ---: | ---: |
| TinyLlama | 53.62±1.01 | 45.70±1.22 | 48.33±1.97 |
| Qwen3-1.7B | 60.67±1.16 | 47.79±0.81 | 54.27±1.10 |
| Qwen2.5-0.5B | 51.01±6.15 | 41.67±3.78 | 48.60±4.85 |
| Llama3.2-1B | 54.61±0.35 | 44.70±1.19 | 50.87±1.67 |

## 预注册最终门控

- B6−B2：四个 pair 中 3 个为正，聚合 delta `+1.54 pp`，pair-cluster CI `[-1.14, +4.05] pp`，CI 跨 0。
- B6−B5：四个 pair 中只有 2 个为正，聚合 delta `+1.19 pp`，pair-cluster CI `[-0.92, +3.31] pp`，CI 跨 0。
- combined gate：`fail`。

最终决定：B6 没有稳定优于 hard-span 与 gate-only control，不进入下一阶段；不开发新 transport、Route3、OT、RoPE correction 或新 loss。

## 诊断注意事项

- B0/B1 没有 alignment bucket 属于预期，因为它们不使用对应 soft aligner diagnostics。
- `candidate_count` 当前记录的是每例平均 candidate 数，绝大多数落入同一数值桶；机制判断主要依赖预注册的 1-to-1/one-to-many 与 entropy buckets。
- 聚合 McNemar 对 B6−B2/B5 很小，但它由大量 example 驱动，不能覆盖 Qwen2.5 的反向结果和跨 pair 异质性；最终门控应以层级 paired CI 为准。
