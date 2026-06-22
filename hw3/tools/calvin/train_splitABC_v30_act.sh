#!/usr/bin/env bash

set -euo pipefail

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

conda run -n lerobot lerobot-train \
  --dataset.repo_id=xiaoma26/calvin-lerobot \
  --dataset.root=/root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitABC_v30 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.repo_id=local/splitABC_v30_act \
  --policy.push_to_hub=false \
  --output_dir=/root/autodl-tmp/cby/homework/ComputerVision/hw3/part2/outputs/splitABC_v30_act \
  --job_name=splitABC_v30_act \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=500 \
  --num_workers=32 \
  --wandb.enable=true
