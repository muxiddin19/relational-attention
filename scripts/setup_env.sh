#!/usr/bin/env bash
# Setup conda environment on the GPU server.
# Usage: bash scripts/setup_env.sh

set -e

ENV_NAME="relattn"
PYTHON_VERSION="3.10"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "=== Activating environment ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing experiment dependencies ==="
pip install -e "$(dirname "$0")/../[experiments]"

echo "=== Installing Spider eval dependencies ==="
pip install networkx==2.8.8 mo-future

echo "=== Verifying GPU access ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo "=== Done. Activate with: conda activate $ENV_NAME ==="
