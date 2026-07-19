# Phase 2A-2A Cache-Geometry Instrumentation Preregistration

> 本文件冻结一次有限的、仅面向既有 B6-native transport 的 cache-geometry instrumentation audit。它不授权训练新 transport、修改 checkpoint、打开 Phase 2A-1 calibration/model-selection/test outcomes，或依据本轮结果追加 feature、candidate、pair、task、seed 与阈值。

## 1. 研究问题与范围

Phase 2A-1 已证明原有五项 A-tier summaries 不能预测何时拒绝 transfer。Phase 2A-2A 只检验一个更窄的问题：在完成 sender KV 生成、projection、alignment 与 fusion，但在 receiver generation outcome 已知之前，compact cache-geometry 是否包含可跨 sender pair 泛化的 harmful/beneficial transfer 信号。

本轮只使用 Phase 2A-1 冻结 content split 中标为 `fit` 的 content groups。`calibration`、`model_selection`、`test` groups 均不得运行 geometry pilot、不得进入 outcome join，也不得用于 feature/candidate/threshold 选择。正式 primary pairs 固定为：

- `tinyllama`；
- `qwen25_0p5b`；
- `llama32_1b`。

tasks 固定为 `ai2-arc`、`openbookqa`、`mmlu-redux`；pilot seed 仅固定为 `42`。不得扩展到 seed 43/44。same-tokenizer control 不进入本轮九项 GO gate。

## 2. 不干预性与 outcome 分离

Instrumentation 必须是 read-only observer：不得原地修改 K/V、gate、weight、attention mask、dtype、device、generation configuration 或随机状态。每个正式 run 必须产生 outcome-free sidecars：

- `<prediction-stem>_cache_geometry_layers.jsonl`：`geometry_on` 的逐 projector/layer compact records；
- `<prediction-stem>_cache_geometry_samples.jsonl`：`geometry_on` 或 `geometry_off` 的逐 sample canonical output fingerprint。

canonical output fingerprint 只包含 `pred,cot_pred,cot_output,cot_gen_length,extraction_method_used,extracted_normalized`，缺失字段写为 JSON `null`；使用 UTF-8、sorted keys、compact separators 后计算 SHA256。不得把 label、correctness、true answer、utility 或 event 写入 geometry sidecar。

九个 `pair × task × seed42` geometry-on runs 必须逐 fit row 与冻结 original B6-native reference prediction CSV 比较，比较字段严格固定为 `pred,is_correct,cot_pred,cot_output,cot_gen_length`，key set 与每个 cell 均须完全相同。只有 `tinyllama × ai2-arc × seed42` 额外运行一次新的 geometry-off matched runtime control；该 control 与对应 geometry-on sidecar 的 identity key set 和 outcome-free fingerprint 也必须完全相同。任何一处不一致都使 Gate 1 失败，不要求为其余八个 cells 重跑 geometry-off。

outcomes 使用独立 CSV，只允许 identity fields 加 `receiver_correct,fused_correct`。统计脚本先读取 geometry 与 outcome identity、冻结各文件 SHA256、key digest、join key 和 design manifests，写入 write-once frozen join manifest；在该 manifest 存在且所有 hash 仍一致前，禁止解析 correctness。

## 3. Identity、fit-only 与 join

冻结 join key 为：

```text
(pair, seed, task, subject, question_id, content_hash)
```

不得使用 row order。`content_hash` 必须出现在 Phase 2A-1 `content_group_split_manifest.json` 且 split 必须严格等于 `fit`。geometry rows、on fingerprints、off fingerprints 与 outcomes 必须具有完全相同的 key set；重复 key、缺 key、额外 key 或 content-hash mismatch 均 fail closed。

## 4. Primary compact geometry

所有 primary fields 在 `recipe/eval_recipe/phase2a_2a/feature_manifest.json` 中逐项显式列出；禁止 wildcard expansion、test-time field discovery、raw text 或 metadata feature。

对 K 与 V 分别冻结以下逐 layer metric families：

- `native_norm`；
- `raw_projected_norm`；
- `residual_ratio`；
- `projected_native_norm_ratio`；
- `projected_native_cosine`；
- `fused_native_cosine`；
- `head_residual_mean/std/max/cv`；
- `residual_head_concentration`。

另冻结逐 layer cross-KV：

```text
residual_imbalance = abs(log((key_residual_ratio + 1e-12)
                             / (value_residual_ratio + 1e-12)))
```

以及逐 layer `learned_weight_mean/std`、`alignment_confidence_mean/std`、`effective_gate_mean/std`。每个适用逐 layer metric 仅产生六项 confirmatory aggregates：`all_mean,all_std,all_max,early_mean,middle_mean,late_mean`。layer 按 `(target_layer,projector_index)` 排序，以 `numpy.array_split(...,3)` 定义 early/middle/late，`all_std` 使用 population standard deviation (`ddof=0`)。

sample-global primary scalars 固定为 `source_receiver_length_ratio,valid_alignment_mass,valid_alignment_coverage`；它们必须在同一 sample 的所有 layer records 中数值一致（冻结 tolerance 内），否则拒绝输入。

逐 layer values、额外 raw diagnostics、原 Phase 2A-1 五项 scalar 与 legacy gate summaries 只作 instrumentation/debug diagnostics，不可加入 selector。

## 5. Geometry reality / nonconstant audit

正式输入必须来自真实 benchmark/model run，不接受 synthetic、mock 或 replayed fabricated geometry 作为 GO evidence。数值相等 tolerance 固定为：

```text
tol(x) = 1e-8 * max(1, max_abs(x))
```

每个 primary pair 必须同时满足：

1. 至少一个 K core metric 在 sample 内跨 layer 有真实 variation，且至少一个 V core metric亦然；
2. 至少一个聚合 K primary feature 在该 pair 的 samples 间 nonconstant，且至少一个聚合 V primary feature亦然。

三 pair 全部通过才满足 Gate 2。常量复制、全零占位或只在 metadata 上变化均不通过。

## 6. Content-hash 五折与 leave-one-pair-out

只在既有 fit groups 内进行 cross-fitting。fold 定义为：

```text
digest = SHA256("phase2a2a-outer-v1|"
                + dataset_content_sha256 + "|" + content_hash)
fold   = first_64_big_endian_bits(digest) mod 5
```

同一 content hash 的所有 pair/member rows 永远同 outer fold。另用 outcome-blind dev hash：

```text
dev_digest   = SHA256("phase2a2a-dev-v1|"
                      + dataset_content_sha256 + "|" + content_hash)
dev_fraction = first_64_big_endian_bits(dev_digest) / 2^64
```

固定 `dev_fraction < 0.60` 为 fit，`[0.60,0.80)` 为 calibration，`[0.80,1.00)` 为 model selection。对每个 held-out pair 和 evaluation fold `f`，held-out pair 完全不得进入 development；其他 pairs 中 outer fold `f` 的 observations 全部排除，其余 observations 才按上述 60/20/20 dev hash 使用：

- evaluation：held-out pair 的 fold `f`；
- fit：其他 pairs、outer fold 不为 `f`、dev fraction `<0.60`；
- calibration：其他 pairs、outer fold 不为 `f`、dev fraction `[0.60,0.80)`；
- model selection：其他 pairs、outer fold 不为 `f`、dev fraction `[0.80,1.00)`。

其他 pairs 的 evaluation fold `f` 全部排除，因此同一 content 在其他 pair 的复制 observation 不会泄漏。五个 evaluation folds 合并后，每个 primary observation 恰好获得一次 out-of-pair、out-of-content-fold prediction。

## 7. Candidate、calibration 与 action

Candidate family 与超参数在 `candidate_manifest.json` 冻结：每个显式 primary feature 一个 depth-1 stump；全 feature L2 multinomial logistic regression (`C=0.01,0.1,1,10,100`)；全 feature depth-2 trees (`min_weight_fraction_leaf=0.05,0.01`)。

fit/calibration/model-selection/evaluation 的 objective weights 均为 pair-task balanced：每个 active `pair × task` cell 总权重相等，cell 内 rows（含 seeds 与重复 content members）等权。seed 不另作第三层重加权。

每个 candidate：

1. 仅在 fit folds 拟合；
2. 仅在 calibration fold 使用 frozen sigmoid multiclass calibration，并以 pair-task balanced selector accuracy 选择 action threshold；
3. score 固定为 `P(beneficial)-P(harmful)`；transfer iff `score > threshold`，相等时 receiver；threshold accuracy tie 取更大 threshold；
4. 仅在 model-selection fold 选择唯一 candidate；accuracy tie 按最低 ordinal；
5. 不 refit，直接应用于 held-out pair/evaluation fold。

Cross-fitted constant-prior Brier baseline 在每个 outer fold 仅用相同 fit observations 的 pair-task balanced三类 prevalence，随后应用于 held-out evaluation observations；不得用 held-out pair 或 calibration/selection outcomes估计 prior。

## 8. Primary metrics

所有 pooled 与 per-pair metrics 都在合并后的 cross-fitted predictions 上计算。Primary pooling 为 pair-task balanced。

- harmful target：`u=fused_correct-receiver_correct=-1`；
- beneficial target：`u=+1`；
- harm score：calibrated `P(u=-1)`；
- selector correctness：transfer 时取 fused，否则取 receiver；
- harmful reduction：`1-P(transfer and harm)/P(harm)`；
- beneficial retention：`P(transfer and benefit)/P(benefit)`；
- multiclass Brier：`mean(sum_c (p_c-1[y=c])^2)`。

另报告 binary harm Brier `mean((P(harm)-1[u=-1])^2)` 及其 fit-only cross-fitted constant harm-prior baseline，作为 calibration secondary diagnostic；Gate 9 仍严格使用预注册的三类 multinomial Brier，不得在看到结果后替换。

任何所需 denominator 为零、AUPRC undefined、fold 缺 class、输入不完整或模型 warning 均 fail closed，不得删除 cell/fold 后继续。

## 9. 九项 conjunctive GO gate

以下九项必须同时成立：

1. 九个 geometry-on cells 与冻结 original B6-native reference 在五项 exact columns 上完全一致，且 TinyLlama/ARC 的 matched on/off outcome-free fingerprints 完全一致；
2. 三 pair 均通过真实 within-sample variation 与 K/V nonconstant audit；
3. pooled held-out harm AUPRC `>= pooled harm prevalence + 0.03`；
4. 至少 `2/3` held-out pairs 的 harm AUPRC 严格大于各自 harm prevalence；
5. cross-fitted selector accuracy minus always-fused accuracy `>= +0.005`；
6. harmful reduction `>= 0.15`；
7. beneficial retention `>= 0.90`；
8. 每个 held-out pair 的 selector-minus-fused delta 均 `>= -0.002`；
9. selector multiclass harm/neutral/benefit Brier 严格小于 cross-fitted constant-prior Brier。

任一失败即 `NO_GO`。本轮没有补 feature、补 pair、补 seed、调 fold salt、改 threshold objective、改 weighting 或训练更复杂/神经 selector 的自由度。

## 10. 停止规则

Phase 2A-2A 只产生 instrumentation sanity audit、cross-fitted statistics 与 GO/NO-GO JSON。它不自动授权 query-time deployment、checkpoint 更新、正式 test 打开或下一阶段实验。结果出来后停止并等待新的明确授权。
