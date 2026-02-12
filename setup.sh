#!/bin/bash
set -e

echo "=== SGTM Training Pod ==="
echo "TRAIN_ARGS: ${TRAIN_ARGS}"
nvidia-smi 2>/dev/null || echo "No GPU detected"
echo "========================="

if [ -z "$TRAIN_ARGS" ]; then
    echo "ERROR: TRAIN_ARGS environment variable not set."
    echo "Set this in RunPod pod configuration."
    echo "Example: --mode baseline --batch-size 16 --grad-accum 8"
    exit 1
fi

# Install Python dependencies
echo "[1/4] Installing dependencies..."
pip install --no-cache-dir fair-esm datasets wandb tqdm matplotlib numpy

# Pre-cache ESM-2 8M model (downloads ~30MB, needed for alphabet)
echo "[2/4] Caching ESM-2 model..."
python -c "import esm; esm.pretrained.esm2_t6_8M_UR50D()"

# Prepare datasets (downloads Swiss-Prot ~86MB, processes into train/val/test splits)
echo "[3/4] Preparing datasets..."
python -m sgtm.data_pipeline

# Train
echo "[4/4] Starting training..."
python -m sgtm.train_sgtm $TRAIN_ARGS

# Evaluate (auto-detects trained model)
echo "Running evaluation..."
python -m sgtm.evaluate_sgtm --device cuda

echo "=== Training and evaluation complete ==="
echo "Check wandb for results, or inspect models/sgtm/ and results/sgtm/"
