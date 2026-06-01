#!/bin/bash
# Run all Phase 1 baseline evaluations (3 models x 3 benchmarks)
# Each batch runs 3 benchmarks in parallel on 3 GPUs
# GPQA excluded (gated dataset, requires HF auth)

set -e
cd "$(dirname "$0")/../.."

PYTHON="/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python"
EVAL_SCRIPT="script/evaluation/unified_evaluator.py"
BASELINE_DIR="recipe/eval_recipe/baseline"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "============================================"
echo " Phase 1: Baseline Evaluations"
echo " 3 models x 3 benchmarks = 9 runs"
echo " Using HF mirror: $HF_ENDPOINT"
echo "============================================"

# Batch 1: Qwen3-0.6B standalone
echo ""
echo "=== Batch 1/3: Qwen3-0.6B standalone ==="
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen3_0.6b_mmlu.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen3_0.6b_gsm8k.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen3_0.6b_arc.yaml &
wait
echo "=== Batch 1 complete ==="

# Batch 2: Qwen2.5-0.5B-Instruct standalone
echo ""
echo "=== Batch 2/3: Qwen2.5-0.5B-Instruct standalone ==="
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen2.5_0.5b_mmlu.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen2.5_0.5b_gsm8k.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_qwen2.5_0.5b_arc.yaml &
wait
echo "=== Batch 2 complete ==="

# Batch 3: Rosetta Fuser (C2C: Qwen3-0.6B + Qwen2.5-0.5B-Instruct)
echo ""
echo "=== Batch 3/3: Rosetta Fuser (C2C) ==="
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_rosetta_fuser_mmlu.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_rosetta_fuser_gsm8k.yaml &
$PYTHON $EVAL_SCRIPT --config $BASELINE_DIR/baseline_rosetta_fuser_arc.yaml &
wait
echo "=== Batch 3 complete ==="

echo ""
echo "============================================"
echo " All Phase 1 evaluations complete!"
echo " Results in: local/final_results/baseline_*/"
echo "============================================"
