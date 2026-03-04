#!/bin/bash
# Phase 2 training script — coarse task (all viral vs non-viral)
#
# Order of operations:
#   1. 8M holdout + SGTM (pipeline validation, ~10 hrs)
#   2. 8M eval (sanity check)
#   3. 35M holdout + SGTM (primary experiment, ~56 hrs)
#   4. 35M eval (full pass)
#   5. Recovery fine-tuning
#   6. HF conversion for evalz benchmarks
#
# Usage:
#   cd ESM-experiments
#   bash scripts/train_phase2.sh          # runs everything
#   bash scripts/train_phase2.sh 8m       # 8M only
#   bash scripts/train_phase2.sh 35m      # 35M only (skip 8M validation)
#   bash scripts/train_phase2.sh eval-8m  # eval 8M only
#   bash scripts/train_phase2.sh eval-35m # eval 35M only

set -euo pipefail

STAGE="${1:-all}"
DATA_DIR="data/sgtm/coarse"
OUTPUT_DIR="models/sgtm_p2"
RESULTS_DIR="results/sgtm_p2"

# Match the original SGTM paper's training regime:
# - No upsampling (natural proportions, ~4% forget similar to paper's 3.7%)
# - 10% retain mode split (90% of retain/adjacent steps use default mode,
#   so forget params get general training signal from most of the data)
# - Embedding masking in forget mode (prevents forget data from updating
#   shared token embeddings)
UPSAMPLE_FORGET=1
UPSAMPLE_ADJACENT=1
RETAIN_RETAIN_PERC=10
ADJACENT_RETAIN_PERC=10

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# ============================================================
# 8M COARSE — pipeline validation
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "8m" ]]; then
  echo ""
  echo "=========================================="
  echo "  8M COARSE: Holdout"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode holdout \
    --model-size 8M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name holdout-coarse-8m \
    --upsample-forget "$UPSAMPLE_FORGET" \
    --upsample-adjacent "$UPSAMPLE_ADJACENT" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  8M COARSE: SGTM (1 head + MLP)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode sgtm \
    --model-size 8M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name sgtm-coarse-8m \
    --upsample-forget "$UPSAMPLE_FORGET" \
    --upsample-adjacent "$UPSAMPLE_ADJACENT" \
    --retain-retain-perc "$RETAIN_RETAIN_PERC" \
    --adjacent-retain-perc "$ADJACENT_RETAIN_PERC" \
    --mask-embeddings \
    --device cuda
fi

# ============================================================
# 8M EVAL — sanity check
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "eval-8m" ]]; then
  echo ""
  echo "=========================================="
  echo "  8M EVAL: PPL + ablation"
  echo "=========================================="
  python -m sgtm.evaluate_sgtm \
    --model-size 8M \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --runs holdout-coarse-8m,sgtm-coarse-8m \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  8M EVAL: Linear probes"
  echo "=========================================="
  python -m sgtm.linear_probe \
    --model-size 8M \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda
fi

# ============================================================
# 35M COARSE — primary experiment
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "35m" ]]; then
  echo ""
  echo "=========================================="
  echo "  35M COARSE: Holdout"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode holdout \
    --model-size 35M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name holdout-coarse-35m \
    --upsample-forget "$UPSAMPLE_FORGET" \
    --upsample-adjacent "$UPSAMPLE_ADJACENT" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  35M COARSE: SGTM (1 head + MLP)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode sgtm \
    --model-size 35M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name sgtm-coarse-35m \
    --upsample-forget "$UPSAMPLE_FORGET" \
    --upsample-adjacent "$UPSAMPLE_ADJACENT" \
    --retain-retain-perc "$RETAIN_RETAIN_PERC" \
    --adjacent-retain-perc "$ADJACENT_RETAIN_PERC" \
    --mask-embeddings \
    --device cuda
fi

# ============================================================
# 35M EVAL — full pass
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "eval-35m" ]]; then
  echo ""
  echo "=========================================="
  echo "  35M EVAL: PPL + ablation"
  echo "=========================================="
  python -m sgtm.evaluate_sgtm \
    --model-size 35M \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --runs holdout-coarse-35m,sgtm-coarse-35m \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  35M EVAL: Linear probes"
  echo "=========================================="
  python -m sgtm.linear_probe \
    --model-size 35M \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  35M: HF conversion for evalz"
  echo "=========================================="
  python -m sgtm.convert_to_hf \
    --checkpoint "$OUTPUT_DIR/holdout-coarse-35m/final_model.pt" \
    --model-size 35M \
    --output-dir models/hf/holdout-coarse-35m

  python -m sgtm.convert_to_hf \
    --checkpoint "$OUTPUT_DIR/sgtm-coarse-35m/final_model.pt" \
    --model-size 35M \
    --output-dir models/hf/sgtm-coarse-35m
fi

# ============================================================
# RECOVERY FINE-TUNING
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "recovery" ]]; then
  echo ""
  echo "=========================================="
  echo "  35M: Recovery fine-tuning"
  echo "=========================================="
  python -m sgtm.recovery_finetune \
    --model-size 35M \
    --checkpoint "$OUTPUT_DIR/holdout-coarse-35m/final_model.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR/recovery" \
    --device cuda
fi

echo ""
echo "=== Phase 2 training complete ==="
