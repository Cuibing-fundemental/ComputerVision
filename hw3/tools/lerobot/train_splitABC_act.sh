#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

PYTHON=/root/autodl-tmp/miniconda3/envs/lerobot/bin/python
ROOT=/root/autodl-tmp/cby/homework/ComputerVision/hw3
LR_ROOT="$ROOT/lerobot"
DATA_ROOT="$ROOT/data"
OUT_ROOT="$LR_ROOT/output"

SPLITA="$DATA_ROOT/splitA"
SPLITB="$DATA_ROOT/splitB"
SPLITC="$DATA_ROOT/splitC"

SPLITA_V30="$SPLITA"
SPLITB_V30="$DATA_ROOT/splitB_v30"
SPLITC_V30="$SPLITC"

SPLITA_FMT="$DATA_ROOT/splitA_trainfmt"
SPLITB_FMT="$DATA_ROOT/splitB_trainfmt"
SPLITC_FMT="$DATA_ROOT/splitC_trainfmt"
MERGED="$DATA_ROOT/splitABC_trainfmt"

mkdir -p "$OUT_ROOT"

if [ ! -d "$SPLITB_V30" ]; then
  "$PYTHON" -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id local/splitB_v30 --root "$SPLITB" --push-to-hub=false
fi

if [ ! -d "$MERGED" ]; then
  "$PYTHON" tools/rename_dataset_features.py \
    --src "$SPLITA_V30" \
    --dst "$SPLITA_FMT" \
    --rename-map '{"image":"observation.images.image","wrist_image":"observation.images.wrist_image","state":"observation.state","actions":"action"}'
  "$PYTHON" tools/rename_dataset_features.py \
    --src "$SPLITB_V30" \
    --dst "$SPLITB_FMT" \
    --rename-map '{"image":"observation.images.image","wrist_image":"observation.images.wrist_image","state":"observation.state","actions":"action"}'
  "$PYTHON" tools/rename_dataset_features.py \
    --src "$SPLITC_V30" \
    --dst "$SPLITC_FMT" \
    --rename-map '{"image":"observation.images.image","wrist_image":"observation.images.wrist_image","state":"observation.state","actions":"action"}'

  "$PYTHON" - <<'PY'
from pathlib import Path
from lerobot.datasets import LeRobotDataset, merge_datasets

data_root = Path("/root/autodl-tmp/cby/homework/ComputerVision/hw3/data")
datasets = [
    LeRobotDataset("local/splitA_trainfmt", root=data_root / "splitA_v30"),
    LeRobotDataset("local/splitB_trainfmt", root=data_root / "splitB_v30"),
    LeRobotDataset("local/splitC_trainfmt", root=data_root / "splitC_v30"),
]
merge_datasets(
    datasets,
    output_repo_id="local/splitABC",
    output_dir=data_root / "splitABC_v30",
)
PY
fi

OUTPUT_DIR="$OUT_ROOT/act_splitABC_100k"

"$PYTHON" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/splitABC \
  --dataset.root="$MERGED" \
  --dataset.revision=v3.0 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=500 \
  --num_workers=16 \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_splitABC_100k
