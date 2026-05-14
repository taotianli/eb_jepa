#!/bin/bash
#SBATCH --job-name=eb_jepa_image
#SBATCH --output=/projects/u6da/eb_jepa/ckpt/logs/image_jepa_%j.out
#SBATCH --error=/projects/u6da/eb_jepa/ckpt/logs/image_jepa_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0

source ~/miniforge/bin/activate
conda activate eb_jepa

export EBJEPA_DSETS=/projects/u6da/eb_jepa/data
export EBJEPA_CKPTS=/projects/u6da/eb_jepa/ckpt
export WANDB_DISABLED=true   # remove this line if you want wandb logging

cd ~/eb_jepa

python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/default.yaml
