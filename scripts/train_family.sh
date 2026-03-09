#!/bin/bash
# Weekend run: 8M family-level fine task (Coronaviridae)
#
# Forget = Coronaviridae proteins (~486 sequences)
# Adjacent = all other viral proteins
# Retain = non-viral Swiss-Prot
#
# This is the closest analog to the original SGTM paper's design:
#   Paper: forget "dangerous biographies", adjacent "safe biographies", retain "other text"
#   Ours:  forget one viral family, adjacent other viral families, retain non-viral
#
# Usage:
#   bash scripts/train_family.sh data      # generate data splits only
#   bash scripts/train_family.sh train     # run holdout + SGTM training
#   bash scripts/train_family.sh eval      # evaluate all conditions
#   bash scripts/train_family.sh all       # everything

set -euo pipefail

STAGE="${1:-all}"
FAMILY="Coronaviridae"
DATA_DIR="data/sgtm/family_coronaviridae"
OUTPUT_DIR="models/sgtm_p2"
RESULTS_DIR="results/sgtm_p2"

# Hyperparameters (match coarse experiments)
BATCH_SIZE=4
GRAD_ACCUM=32
RETAIN_RETAIN_PERC=25

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# ============================================================
# DATA PREPARATION
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "data" ]]; then
  echo ""
  echo "=========================================="
  echo "  Download viral data with family annotations"
  echo "=========================================="
  python data/download_virus_data.py --output-dir data/raw/

  echo ""
  echo "=========================================="
  echo "  Generate family-based splits: $FAMILY"
  echo "=========================================="
  python -m sgtm.data_pipeline \
    --forget-task family \
    --forget-family "$FAMILY" \
    --data-dir data/sgtm \
    --raw-dir data/raw
fi

# ============================================================
# 8M TRAINING
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  echo ""
  echo "=========================================="
  echo "  8M FAMILY: Holdout (no $FAMILY data)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode holdout \
    --model-size 8M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "holdout-family-${FAMILY,,}-8m" \
    --upsample-forget 1 \
    --upsample-adjacent 1 \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  8M FAMILY: SGTM ret25 (1 head + MLP)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode sgtm \
    --model-size 8M \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "sgtm-family-${FAMILY,,}-8m" \
    --upsample-forget 1 \
    --upsample-adjacent 1 \
    --retain-retain-perc "$RETAIN_RETAIN_PERC" \
    --adjacent-retain-perc "$RETAIN_RETAIN_PERC" \
    --mask-embeddings \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --device cuda
fi

# ============================================================
# 8M EVAL
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
  echo ""
  echo "=========================================="
  echo "  8M EVAL: PPL + ablation"
  echo "=========================================="
  python -m sgtm.evaluate_sgtm \
    --model-size 8M \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --runs "holdout-family-${FAMILY,,}-8m,sgtm-family-${FAMILY,,}-8m" \
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

echo ""
echo "=== Family task ($FAMILY) complete ==="
