#!/bin/bash
set -e

MODEL_SIZE="${MODEL_SIZE:-8M}"

echo "=== SGTM Training Pod ==="
echo "MODEL_SIZE: ${MODEL_SIZE}"
echo "TRAIN_ARGS: ${TRAIN_ARGS}"
nvidia-smi 2>/dev/null || echo "No GPU detected"
echo "========================="

if [ -z "$TRAIN_ARGS" ]; then
    echo "ERROR: TRAIN_ARGS environment variable not set."
    echo "Example: TRAIN_ARGS='--mode baseline --batch-size 16 --grad-accum 8'"
    exit 1
fi

# Verify required files exist
for f in sgtm/__init__.py sgtm/train_sgtm.py sgtm/model_config.py data/raw/virus_human.tsv data/raw/virus_nonhuman.tsv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        echo "Make sure you're running from the repo root after git clone."
        exit 1
    fi
done

# Install Python dependencies
echo "[1/5] Installing dependencies..."
pip install --no-cache-dir fair-esm datasets wandb tqdm matplotlib numpy scikit-learn

# Pre-cache ESM-2 model weights (needed for alphabet in tests; actual training uses from-scratch init)
echo "[2/5] Caching ESM-2 ${MODEL_SIZE} model..."
if [ "$MODEL_SIZE" = "35M" ]; then
    python -c "import esm; esm.pretrained.esm2_t12_35M_UR50D()"
else
    python -c "import esm; esm.pretrained.esm2_t6_8M_UR50D()"
fi

# Prepare datasets (downloads Swiss-Prot ~86MB, processes into train/val/test splits)
echo "[3/5] Preparing datasets..."
python -m sgtm.data_pipeline

# Train
echo "[4/5] Starting training..."
python -m sgtm.train_sgtm --model-size $MODEL_SIZE $TRAIN_ARGS

# Evaluate (auto-detects trained model)
echo "[5/5] Running evaluation..."
python -m sgtm.evaluate_sgtm --model-size $MODEL_SIZE --device cuda

# Run ablation experiments
echo "Running ablation experiments..."
python -m sgtm.ablation_experiments --model-size $MODEL_SIZE --device cuda

# Run linear probe
echo "Running linear probe evaluation..."
python -m sgtm.linear_probe --model-size $MODEL_SIZE --device cuda

# Sync to network volume if available
if [ -n "$NETWORK_VOL" ] && [ -d "$NETWORK_VOL" ]; then
    echo "Syncing results to network volume..."
    cp -r models/ results/ "$NETWORK_VOL/" 2>/dev/null || true
fi

echo "=== Training and evaluation complete ==="
echo "Check wandb for results, or inspect models/sgtm/ and results/sgtm/"
