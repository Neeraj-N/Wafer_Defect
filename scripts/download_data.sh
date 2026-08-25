#!/usr/bin/env bash
# Download the WM-811K wafer map dataset from Kaggle into data/raw/.
#
# Prerequisites:
#   pip install kaggle          (already in requirements.txt)
#   Kaggle API token at ~/.kaggle/kaggle.json
#     -> kaggle.com -> Account -> "Create New API Token"
#     -> mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

mkdir -p data/raw
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw --unzip

echo "Done. Expect data/raw/LSWMD.pkl (~3.5 GB)."
