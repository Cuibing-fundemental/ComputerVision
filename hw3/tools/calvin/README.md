# LeRobot dataset tools

## merge_lerobot_datasets.sh

Use LeRobot's official CLI `lerobot-edit-dataset` to merge multiple local datasets into a new dataset directory without modifying the source datasets.
If an input dataset still uses the older local schema naming such as `observation.image`, the script first creates a temporary normalized copy and then runs the official merge CLI on those copies.

Example for this homework:

```bash
bash tools/merge_lerobot_datasets.sh \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitABC_v30 \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitA_v30 \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitB_v30 \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitC_v30
```

The script will:

- run inside the `lerobot` Conda environment via `conda run -n lerobot`
- keep `HF_HOME=/root/autodl-tmp/hf_home`
- keep `HF_ENDPOINT=https://hf-mirror.com`
- normalize incompatible local dataset field names in a temporary copy when needed
- refuse to overwrite an existing output directory

General usage:

```bash
bash tools/merge_lerobot_datasets.sh <output_dataset_dir> <input_dataset_dir_1> [input_dataset_dir_2 ...]
```

## normalize_lerobot_dataset.py

This helper copies one dataset to a new location and rewrites old schema names into the newer LeRobot naming used by `merge`.

Example:

```bash
python tools/normalize_lerobot_dataset.py \
  --src /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitB_v30 \
  --dst /tmp/splitB_v30_normalized
```

## train_splitABC_v30_act.sh

Train an ACT policy on the merged `splitABC_v30` dataset using `lerobot-train`.

```bash
bash tools/train_splitABC_v30_act.sh
```

This script uses:

- dataset root: `/root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitABC_v30`
- output dir: `/root/autodl-tmp/cby/homework/ComputerVision/hw3/part2/outputs/splitABC_v30_act`
- policy type: `act`
- conda env: `lerobot`
