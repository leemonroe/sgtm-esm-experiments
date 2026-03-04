#!/bin/bash
# Phase 2 RunPod setup script
# Run once after cloning the repo on a fresh pod.
#
# Usage:
#   cd ESM-experiments
#   bash scripts/setup_pod.sh

set -euo pipefail

echo "=== Phase 2 Pod Setup ==="

# 1. Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install torch fair-esm transformers datasets scikit-learn matplotlib wandb h5py tqdm

# 2. Download curated virus data from UniProt (virus_hosts-based split)
echo ""
echo "--- Downloading curated virus data ---"
python data/download_virus_data.py --output-dir data/raw

# 3. Generate data splits
echo ""
echo "--- Generating coarse splits (all viral vs non-viral) ---"
python -m sgtm.data_pipeline --forget-task coarse --data-dir data/sgtm

echo ""
echo "--- Generating fine splits (human-infecting vs other viral) ---"
python -m sgtm.data_pipeline --forget-task fine --data-dir data/sgtm

echo ""
echo "=== Setup complete ==="
echo ""
echo "Data ready at:"
echo "  data/sgtm/coarse/  (forget + retain)"
echo "  data/sgtm/fine/    (forget + adjacent + retain)"
echo ""
echo "Next: run training with scripts/train_phase2.sh"
