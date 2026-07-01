#!/usr/bin/env bash
#
# Full pipeline: download data → train → evaluate
#
# Usage:
#   bash run_train.sh            # default config
#   bash run_train.sh --epochs 30 --d_model 512 --n_layers 6
#

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1: Download data (if not present) ──────────────────────────────────
if [ ! -f data/train.en ] || [ ! -f data/train.de ]; then
    echo "==> Data not found. Downloading..."
    bash download_data.sh
else
    echo "==> Data already exists, skipping download."
fi

# ── Step 2: Install dependencies ────────────────────────────────────────────
if ! python -c "import torch" 2>/dev/null; then
    echo "==> PyTorch not found. Installing..."
    pip install torch --quiet
fi

# ── Step 3: Train ───────────────────────────────────────────────────────────
echo ""
echo "==> Starting training..."
python train.py "$@"

# ── Step 4: Show best BLEU & run demo inference ─────────────────────────────
echo ""
echo "==> Training complete!"
if [ -f checkpoints/best.pt ]; then
    echo "==> Running demo inference with best checkpoint..."
    python inference.py --checkpoint checkpoints/best.pt
fi
