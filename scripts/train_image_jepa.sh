#!/bin/bash
# Train Image JEPA on the cluster
# Usage:
#   bash scripts/train_image_jepa.sh           # 3-seed sweep (default)
#   bash scripts/train_image_jepa.sh --single  # single job (debug)

source ~/miniforge/bin/activate
conda activate eb_jepa

export EBJEPA_DSETS=/projects/u6da/eb_jepa/data
export EBJEPA_CKPTS=/projects/u6da/eb_jepa/ckpt

cd "$(dirname "$0")/.."

python -m examples.launch_sbatch \
    --example image_jepa \
    --fname examples/image_jepa/cfgs/default.yaml \
    "$@"
