完整公式

下面省略 layer (\ell)，但实际每层独立计算；保留 attention head (h)。

符号：

* (i)：receiver 历史 token 位置；
* (j)：与 (i) 对齐的 source candidate；
* (t)：当前 receiver query 位置；
* (A_{ij})：aligner 给出的 candidate prior；
* (g_i^h)：source branch 的 attention prior；
* (a(x)\in{0,1})：Phase 2A 的整样本 hard selector。

## Step 1：Alignment，只产生候选，不平均

对 receiver token (i)：

[
\mathcal C_i={j:A_{ij}>0},
]

[
A_{ij}\geq0,\qquad
\sum_{j\in\mathcal C_i}A_{ij}=1.
]

你现在的 aligner 已经产生了 `source_indices` 和 `source_weights`，即这里的 (j) 和 (A_{ij})。这一部分基本不用重写。[当前 alignment 实现](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/aligner.py#L1164-L1369)

---

## Step 2：先去掉 source/receiver 各自的 RoPE

Receiver native key：

[
k_i^{r,0}=R_r(i)^{-1}K_i^r.
]

Source candidate key：

[
k_j^{s,0}=R_s(j)^{-1}K_j^s.
]

这里的 (0) 表示“去除了位置旋转的 content-space key”。

注意实际实现中应该使用真实的 `position_id`，不一定等于数组下标 (i,j)。

---

## Step 3：对每个 candidate 独立投影和融合

不要再计算：

[
\sum_jA_{ij}K_j^s.
]

而是对每个 (j) 分别计算：

[
\Delta k_{ij}^{h}
=================

F_{K,\theta}^{h}
\left(
k_i^{r,0},
P_K(k_j^{s,0})
\right),
]

[
\Delta v_{ij}^{h}
=================

F_{V,\theta}^{h}
\left(
V_i^r,
P_V(V_j^s)
\right).
]

然后保持 C2C 的 residual enrichment：

[
K_{ij}^{F,h}
============

R_r(i)
\left(
k_i^{r,0,h}+\Delta k_{ij}^{h}
\right),
]

[
V_{ij}^{F,h}
============

V_i^{r,h}+\Delta v_{ij}^{h}.
]

关键点是：

* 每个 (j) 都有独立的 (K_{ij}^F,V_{ij}^F)；
* 所有 candidate 都使用 receiver parent position (i) 的 RoPE；
* (K_{ij}^F,V_{ij}^F) 必须成对保留，不能 K 选择一个 candidate、V 选择另一个。

---

## Step 4：加入 receiver-native component

定义 (c=0) 为 native：

[
K_{i0}^{h}=K_i^{r,h},
\qquad
V_{i0}^{h}=V_i^{r,h}.
]

Source candidate component 为 (c=j)：

[
K_{ij}^{h}=K_{ij}^{F,h},
\qquad
V_{ij}^{h}=V_{ij}^{F,h}.
]

如果将来组合 Phase 2A，定义：

[
\widetilde g_{xi}^{h}
=====================

a(x),r_i,g_i^h,
]

其中 (r_i=0) 表示该位置没有合法 source candidate，否则 (r_i=1)。

Component prior：

[
\pi_{i0}^{h}=1-\widetilde g_{xi}^{h},
]

[
\pi_{ij}^{h}=\widetilde g_{xi}^{h}A_{ij}.
]

因此：

* (a(x)=0)：整道样本只走 receiver；
* (a(x)=1,g=1)：纯 factorized transfer；
* (a(x)=1,0<g<1)：query-time native/source mixture。

第一阶段应该固定：

[
a(x)=1,\qquad g_i^h=1,
]

先只验证 factorization，不要同时加入 null、entropy 和 selector。

---

## Step 5：计算 native 和 candidate logits

Native：

[
\ell_{ti0}^{h}
==============

\frac{(Q_t^h)^\top K_i^{r,h}}
{\sqrt{d_h}}.
]

Source candidate：

[
\ell_{tij}^{h}
==============

\frac{(Q_t^h)^\top K_{ij}^{F,h}}
{\sqrt{d_h}}.
]

加入 prior 和 causal mask：

[
z_{ti0}^{h}
===========

\ell_{ti0}^{h}
+
\log(1-\widetilde g_{xi}^{h})
+
M_{ti},
]

[
z_{tij}^{h}
===========

\ell_{tij}^{h}
+
\log\widetilde g_{xi}^{h}
+
\log A_{ij}
+
M_{ti}.
]

其中：

[
M_{ti}=
\begin{cases}
0,& i\text{ 对 }t\text{ 可见},\
-\infty,&\text{否则}.
\end{cases}
]

所有 candidate 继承 parent (i) 的 mask，不能把展开后的 candidate 当成新的时间位置。

工程上不能真的计算 `log(0)`，而应该直接 mask 成 (-\infty)。

---

## Step 6：必须在全部 ((i,c)) 上做一次全局 softmax

[
p_{tic}^{h}
===========

\frac{\exp z_{tic}^{h}}
{\sum_{i',c'}\exp z_{ti'c'}^{h}}.
]

最终输出：

[
o_t^h
=====

\sum_i
p_{ti0}^{h}V_i^{r,h}
+
\sum_i\sum_{j\in\mathcal C_i}
p_{tij}^{h}V_{ij}^{F,h}.
]

然后接 receiver 原本的：

[
W_O\operatorname{Concat}_h(o_t^h).
]

这一步是整个方法的核心：不能只在每个 (i) 的 candidates 内做 softmax。

---

# 4. 等价的“组内分解 + 全局 attention”形式

为了便于实现和解释，可以把全局 softmax 分成三层。

## Source candidates 内部选择

[
Z_{ti}^{s,h}
============

\sum_jA_{ij}\exp(\ell_{tij}^{h}),
]

[
\gamma_{tij}^{h}
================

\frac{A_{ij}\exp(\ell_{tij}^{h})}
{Z_{ti}^{s,h}}.
]

(\gamma) 回答：

> 如果决定使用 source branch，当前 query 应选哪个 candidate？

## Native/source 选择

[
\rho_{ti}^{h}
=============

\frac{
\widetilde g_i^h Z_{ti}^{s,h}
}{
(1-\widetilde g_i^h)\exp(\ell_{ti0}^{h})
+
\widetilde g_i^h Z_{ti}^{s,h}
}.
]

也可以写成：

[
\rho_{ti}^{h}
=============

\sigma
\left[
\operatorname{logit}(\widetilde g_i^h)
+
\log Z_{ti}^{s,h}
-----------------

\ell_{ti0}^{h}
\right].
]

(\rho) 才是真正依赖 query 的 native/source posterior；(g) 只是 prior。

## Receiver positions 的全局选择

[
S_{ti}^{h}
==========

\log
\left[
(1-\widetilde g_i^h)e^{\ell_{ti0}^{h}}
+
\widetilde g_i^hZ_{ti}^{s,h}
\right],
]

[
\beta_{ti}^{h}
==============

\operatorname{softmax}*i
\left(S*{ti}^{h}+M_{ti}\right).
]

最终：

[
p_{ti0}^{h}
===========

\beta_{ti}^{h}(1-\rho_{ti}^{h}),
]

[
p_{tij}^{h}
===========

\beta_{ti}^{h}\rho_{ti}^{h}\gamma_{tij}^{h},
]

[
o_t^h
=====

\sum_i\beta_{ti}^{h}
\left[
(1-\rho_{ti}^{h})V_i^r
+
\rho_{ti}^{h}
\sum_j\gamma_{tij}^{h}V_{ij}^F
\right].
]

三种权重的含义是：

| 权重             | 回答的问题                  |
| -------------- | ---------------------- |
| (\gamma_{tij}) | source candidates 中选哪个 |
| (\rho_{ti})    | native 还是 source       |
| (\beta_{ti})   | 所有历史 receiver 位置中关注哪个  |

这与直接在全部 ((i,c)) 上做一次 softmax 严格等价。

---

# 5. 对应你当前代码，逻辑上要改哪里

| 当前逻辑                             | 新逻辑                                             |
| -------------------------------- | ----------------------------------------------- |
| aligner 输出 top-k indices/weights | 保留，作为 (J,A)                                     |
| wrapper 在 projector 前加权求和        | gather 后保留 candidate 轴                          |
| projector 只看到单个 source slot      | 同一个 projector 对每个 candidate 独立运行                |
| fused KV 覆盖 native DynamicCache  | native cache 保留，candidate 放 sidecar             |
| 标准 attention 只看一个 cache          | attention 同时读取 native cache 和 candidate sidecar |
| decode 只持有 fused prompt cache    | sidecar 在整个 decode 生命周期中持续存在                    |

当前真正发生提前坍缩的位置是：

```text
(gathered_key * weights).sum(dim=3)
(gathered_value * weights).sum(dim=3)
```

对应 [`_weighted_source_kv_from_indices`](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/wrapper.py#L464-L505)。

仓库已经有不聚合版本 [`_source_kv_candidates_from_indices`](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/wrapper.py#L508-L533)，能够得到：

[
[B,H_s,N,K,D_s].
]

所以 gather 起点已经存在。

但 Route-3 不能直接使用，因为它的 `align_source_kv()` 最后仍然在 candidate 维度上求和，只是一个 attention 前静态 router。[Route-3 candidate aggregation](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/projector.py#L3101-L3254)

当前 `C2CProjector.forward()` 输出一个 residual fused slot，随后 wrapper 直接覆盖 native cache。[当前 projector](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/projector.py#L4456-L4612)、[cache 覆盖位置](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/wrapper.py#L966-L1015)

新方案应变成：

```text
source candidates [B,Hs,N,K,Ds]
target native     [B,Hr,N,Dt]
        ↓ target 沿 K 广播
target expanded   [B,Hr,N,K,Dt]
        ↓ 对每个 candidate 使用共享 fuser
fused candidates  [B,Hr,N,K,Dt]
        ↓ 不对 K 求和
candidate sidecar
```

---

# 6. 推荐的整体伪代码

```text
# Phase 1：factorization-only，暂时固定 a=1, g=1

receiver_ids, source_ids = tokenize_with_own_tokenizers(x)
J, A, valid = span_align(receiver_ids, source_ids)

native_cache = receiver_prefill(receiver_ids)
source_cache = source_prefill(source_ids)

for each receiver layer ℓ:
    source_candidates = gather(
        source_cache[matched_source_layer],
        indices=J
    )  # [B,Hs,N,K,Ds]，不聚合

    for each valid candidate (i,j):
        source_content = inverse_source_rope(
            source_candidates[i,j],
            source_position=j
        )

        receiver_content = inverse_receiver_rope(
            native_cache[ℓ,i],
            receiver_position=i
        )

        delta_k, delta_v = shared_fuser(
            receiver_content,
            source_content
        )

        candidate_K[i,j] = receiver_rope(
            receiver_content.K + delta_k,
            parent_receiver_position=i
        )
        candidate_V[i,j] = receiver_content.V + delta_v

    sidecar[ℓ] = {
        candidate_K,
        candidate_V,
        A,
        valid,
        parent_cache_index,
        parent_receiver_position
    }

for each receiver decode step t:
    for each receiver layer ℓ:
        q, native_K, native_V = original_attention_frontend(...)
        native_cache.update(native_K, native_V)

        native_logits = q @ native_cache.K
        candidate_logits = q @ sidecar[ℓ].candidate_K

        native_logits += log(1-g)
        candidate_logits += log(g) + log(A)

        candidate_mask = gather_native_mask_by_parent_position()
        logits = concatenate(native_logits, candidate_logits)

        p = global_softmax(logits + masks)
        attention_output = weighted_sum(
            p,
            native_V and candidate_V
        )

    append generated token only to native_cache
```

现有 `_monkeypatch_qwen3_attention_forward` 只支持“相同 shape 的 K/V 替换”，无法表示额外 candidate、(\log A) bias 和 parent mask，因此正式 prototype 必须在 `cache.update()` 后、attention softmax 前插入新逻辑。[现有 attention hook](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/wrapper.py#L224-L322)

第一版建议使用 eager attention，先保证公式严格正确；FlashAttention/SDPA 优化放到后面。

---

# 7. Prefill、decode 和 causal mask 的关键边界

Candidate sidecar 至少要保存：

```text
candidate_key
candidate_value
candidate_log_prior
valid_mask
parent_cache_index
parent_rope_position
```

`parent_cache_index` 和 `parent_rope_position` 要分开保存，因为 padding、packed sequence 时两者未必等同。

Candidate mask 必须是：

[
M_{tij}^{\text{candidate}}
==========================

M_{t,\operatorname{parent}(i,j)}^{\text{native}}.
]

不能把 ((i,j)) flatten 后直接套普通三角 mask，否则：

* 第二个 candidate 会被误认为更晚的 token；
* 可能发生 future leakage；
* RoPE position 也会错误膨胀。

Decode 时，当前代码会把后续 `kv_cache_index` 设为 `-1`，不再重新生成 source cache；新方案可以维持这一点，但 candidate sidecar 必须在整个生成过程中持续保留。[当前 generate 流程](https://github.com/Cyrusi6/Cache/blob/9fa1f0ac3bedefd282961a853278ab88fb376fa2/rosetta/model/wrapper.py#L1022-L1297)

新生成 token 没有 source candidate，因此默认：

[
\pi_{i0}=1,\qquad \pi_{ij}=0.
]

---

# 8. 推荐的实验顺序

## Q0：Candidate 可分离性审计，不训练

先测：

[
D_K
===

\sum_jA_{ij}
\left|
K_{ij}^F-\sum_kA_{ik}K_{ik}^F
\right|^2,
]

[
D_V
===

\sum_jA_{ij}
\left|
V_{ij}^F-\sum_kA_{ik}V_{ik}^F
\right|^2,
]

以及 Jensen gap：

[
G_{ti}
======

## \log\sum_jA_{ij}e^{\ell_{tij}}

\sum_jA_{ij}\ell_{tij}
\geq0.
]

如果异构 pair 上 (D_K,D_V,G) 几乎都是零，说明 query 没有可利用的候选差异，昂贵 prototype 可以停止。

需要注意：除了 TinyLlama，当前其他三个 pair 的 one-to-many 样本都很少。因此正式实验最好增加：

* 预注册的高 ambiguity 子集；
* 至少一个 one-to-many rate 足够高的异构 tokenizer pair。

否则全数据平均会严重稀释这个机制。

## Q1：只验证 factorization

固定：

[
a=1,\qquad g=1.
]

同时：

* 不使用 entropy；
* 不使用 confidence；
* legacy K/V gate 在两组中保持完全相同，最好 forced-on；
* 暂不加入 native null；
* 暂不改训练 loss。

训练/评测做一个关键的 (2\times2)：

| 训练 operator |      评测 barycentric |      评测 factorized |
| ----------- | ------------------: | -----------------: |
| barycentric |            baseline | surgery diagnostic |
| factorized  | collapse diagnostic |           proposed |

这能回答 Phase 1.5 暴露出来的问题：收益究竟来自训练轨迹，还是来自推理时保留 candidate。

同一个 factorized checkpoint 还必须比较：

[
\text{eval-top-k4}-\text{eval-top-k1},
]

以及：

[
\text{factorized attention}
---------------------------

\text{evaluation-time candidate collapse}.
]

如果同 checkpoint 干预仍然接近零，就不能声称 query-time candidate 使用有效。

## Q2：加入 RoPE disentanglement

正式论文最好做 aggregation×position 的匹配实验：

* barycentric + legacy position；
* factorized + legacy position；
* barycentric + de/re-RoPE；
* factorized + de/re-RoPE。

这样 RoPE correction 是正确性组件，factorization 才是 headline novelty。

## Q3：加入 native null

先从简单 prior 开始：

[
g_{\ell h}=\sigma(b_{\ell h}),
]

而不是重新做 token entropy MLP。

推荐顺序：

1. 固定 (g=1)；
2. 固定 (g\in{0.25,0.5,0.75})；
3. model-pair/global scalar；
4. layer/head scalar；
5. 最后才考虑 query-independent token feature。

(g) 应在独立语言模型 calibration 数据上按 next-token NLL 校准，不要用 ARC/OBQA/MMLU 的 answer correctness 调 (g)。

## Q4：最后才组合 Phase 2A outer selector

如果 outer selector 通过：

```text
a = pretransfer_selector(x)

if a == 0:
    skip sender and fuser
    run receiver-only
else:
    run factorized transport with native null
```

但换成 factorized checkpoint 后，必须重新定义：

[
u_{\text{fact}}
===============

y_{\text{factorized}}-y_R,
]

重新做 beneficial、harmful 和 oracle audit。旧 B6 的 `+8.24pp` 不能继续作为新系统的 headroom。

---

# 9. 旧 checkpoint 能复用到什么程度

可以复用：

* sender/receiver 权重；
* tokenizer alignment 的 (J,A)；
* 数据、split 和统计框架；
* B6 projector 权重作为 smoke test 或初始化。

不能直接作为正式结果：

* B2/B3/B6 都是在“candidate 已经平均”的输入分布上训练；
* Route-3 最终也聚合成单 slot；
* candidate-specific 调用旧 projector 是 OOD inference；
* de-RoPE 后 key 输入空间改变；
* B6 的 harmful/beneficial 标签也不再对应新模型。

所以正式 factorized-vs-barycentric 比较必须：

* 相同初始化；
* 相同训练数据和顺序；
* 相同 seed；
* 相同训练预算；
* matched retraining。

---

# 10. 可以正式声称的理论性质

这套公式有几项可以写成 proposition：

1. Exact receiver recovery
   (g=0) 时 source 全部 mask，严格恢复 receiver attention。

2. One-to-one degeneration
   (|\mathcal C_i|=1,A=1,g=1) 时退化为单 slot transfer。

3. Refinement invariance
   一个完全相同的 candidate 拆成多个子 candidate，只要 prior 总和不变，输出不变。

4. Flat–hierarchical equivalence
   (\gamma,\rho,\beta) 分解与全局 ((i,c)) softmax 严格等价。

5. Non-commutativity
   旧方法的 evidence 是：

   [
   e^{\sum_jA_{ij}\ell_{tij}},
   ]

   新方法是：

   [
   \sum_jA_{ij}e^{\ell_{tij}},
   ]

   且由 Jensen 不等式：

   [
   \log\sum_jA_{ij}e^{\ell_{tij}}
   \geq
   \sum_jA_{ij}\ell_{tij}.
   ]

   等号只在 candidate logits 相同等特殊情况下成立。

但这不等于“理论保证 benchmark accuracy 一定提升”。数学上保证的是：

* 不提前破坏 candidate factorization；
* exact fallback；
* 正确的全局归一化；
* 对 candidate 复制不敏感；
* query-dependent K/V coupling。

性能是否提高仍然必须由 matched training 和同-checkpoint intervention 证明。

---

## 最终建议

论文主线保留为：

> Factorization-Preserving Cache Transport：不在 attention 前平均一对多 KV，而是让 receiver query 在 attention 内完成 candidate marginalization，并将 receiver-native memory 作为显式 null atom。

Phase 2A 继续作为独立外层安全支线，不要把它和内部 (g,\rho) 混成一个 gate。

现有 Phase 2A 预注册明确限定为 B6-vs-receiver selector，也明确不在该阶段开发新 transport。因此，如果 factorized transport 是你的独立 ICLR 主线，应该在打开 Phase 2A selector test 之前，单独冻结一份 factorization 预注册；不要事后改写 Phase 2A 的 GO/NO-GO 规则。

最先做的不是 GPU 训练，而是 Q0 candidate 可分离性审计。它是目前成本最低、最能判断这条主线值不值得继续的步骤。
