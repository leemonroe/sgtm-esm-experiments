#!/bin/bash
# Weekend run: family-level fine task (Coronaviridae)
#
# Forget = Coronaviridae proteins (~486 sequences)
# Adjacent = all other viral proteins
# Retain = non-viral Swiss-Prot
#
# This is the closest analog to the original SGTM paper's design:
#   Paper: forget "dangerous biographies", adjacent "safe biographies", retain "other text"
#   Ours:  forget one viral family, adjacent other viral families, retain non-viral
#
# Usage — run everything for one model size (set-and-forget):
#   bash scripts/train_family.sh 8m       # data + train + eval for 8M
#   bash scripts/train_family.sh 35m      # data + train + eval for 35M
#
# Or run individual stages:
#   bash scripts/train_family.sh data         # generate data splits only
#   bash scripts/train_family.sh train-8m     # train 8M only
#   bash scripts/train_family.sh train-35m    # train 35M only
#   bash scripts/train_family.sh eval-8m      # eval 8M only
#   bash scripts/train_family.sh eval-35m     # eval 35M only
#   bash scripts/train_family.sh recovery-35m # recovery fine-tuning 35M
#   bash scripts/train_family.sh all          # everything (8M + 35M + recovery)

set -euo pipefail

STAGE="${1:-all}"
FAMILY="Coronaviridae"
DATA_DIR="data/sgtm/family_coronaviridae"
OUTPUT_DIR="models/sgtm_p2/family_coronaviridae"
RESULTS_DIR="results/sgtm_p2/family_coronaviridae"

# RTX 4090 (24GB): batch 4 × accum 32 = effective 128
BATCH_SIZE=4
GRAD_ACCUM=32
RETAIN_RETAIN_PERC=25

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

# ============================================================
# DATA PREPARATION (runs once, shared by 8M and 35M)
# ============================================================
run_data() {
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
}

# ============================================================
# TRAINING — parameterized by model size
# ============================================================
run_train() {
  local SIZE="$1"

  echo ""
  echo "=========================================="
  echo "  ${SIZE} FAMILY: Holdout (no $FAMILY data)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode holdout \
    --model-size "$SIZE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "holdout-${SIZE,,}" \
    --upsample-forget 1 \
    --upsample-adjacent 1 \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  ${SIZE} FAMILY: SGTM ret25 (1 head + MLP)"
  echo "=========================================="
  python -m sgtm.train_sgtm \
    --mode sgtm \
    --model-size "$SIZE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "sgtm-ret25-${SIZE,,}" \
    --upsample-forget 1 \
    --upsample-adjacent 1 \
    --retain-retain-perc "$RETAIN_RETAIN_PERC" \
    --adjacent-retain-perc "$RETAIN_RETAIN_PERC" \
    --mask-embeddings \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --device cuda
}

# ============================================================
# EVAL — parameterized by model size
# ============================================================
run_eval() {
  local SIZE="$1"

  echo ""
  echo "=========================================="
  echo "  ${SIZE} EVAL: PPL + ablation"
  echo "=========================================="
  python -m sgtm.evaluate_sgtm \
    --model-size "$SIZE" \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --runs "holdout-${SIZE,,},sgtm-ret25-${SIZE,,}" \
    --device cuda

  echo ""
  echo "=========================================="
  echo "  ${SIZE} EVAL: Linear probes"
  echo "=========================================="
  python -m sgtm.linear_probe \
    --model-size "$SIZE" \
    --models-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR" \
    --device cuda
}

# ============================================================
# RECOVERY FINE-TUNING
# ============================================================
run_recovery() {
  local SIZE="$1"

  echo ""
  echo "=========================================="
  echo "  ${SIZE}: Recovery fine-tuning (holdout → forget data)"
  echo "=========================================="
  python -m sgtm.recovery_finetune \
    --model-size "$SIZE" \
    --checkpoint "$OUTPUT_DIR/holdout-${SIZE,,}/final_model.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RESULTS_DIR/recovery-${SIZE,,}" \
    --device cuda
}

# ============================================================
# DISPATCH
# ============================================================
case "$STAGE" in
  data)
    run_data
    ;;
  train-8m)
    run_train 8M
    ;;
  train-35m)
    run_train 35M
    ;;
  eval-8m)
    run_eval 8M
    ;;
  eval-35m)
    run_eval 35M
    ;;
  recovery-35m)
    run_recovery 35M
    ;;
  8m)
    run_data
    run_train 8M
    run_eval 8M
    ;;
  35m)
    run_data
    run_train 35M
    run_eval 35M
    run_recovery 35M
    ;;
  all)
    run_data
    run_train 8M
    run_eval 8M
    run_train 35M
    run_eval 35M
    run_recovery 35M
    ;;
  *)
    echo "Usage: bash scripts/train_family.sh {data|8m|35m|train-8m|train-35m|eval-8m|eval-35m|recovery-35m|all}"
    exit 1
    ;;
esac

echo ""
echo "=== Family task ($FAMILY, $STAGE) complete ==="
