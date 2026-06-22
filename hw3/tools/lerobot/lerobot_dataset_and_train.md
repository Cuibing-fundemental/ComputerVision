# LeRobot 数据转换与训练

## 1. v2.1 -> v3.0

```bash
PYTHONPATH=src HF_HOME=/root/autodl-tmp/hf_home HF_ENDPOINT=https://hf-mirror.com \
/root/autodl-tmp/miniconda3/envs/lerobot/bin/python \
src/lerobot/scripts/convert_dataset_v21_to_v30.py \
  --repo-id local/splitD \
  --root /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD \
  --push-to-hub=false
```

可把 `splitD` 替换成 `splitA` / `splitC`。

## 2. 训练用键名重写

```bash
PYTHONPATH=src /root/autodl-tmp/miniconda3/envs/lerobot/bin/python \
tools/rename_dataset_features.py \
  --src /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD \
  --dst /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_v30 \
  --rename-map '{"image":"observation.images.image","wrist_image":"observation.images.wrist_image","state":"observation.state","actions":"action"}'
```

## 3. 50 步训练

```bash
PYTHONPATH=src HF_HOME=/root/autodl-tmp/hf_home HF_ENDPOINT=https://hf-mirror.com \
/root/autodl-tmp/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/splitD_trainfmt \
  --dataset.root=/root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_trainfmt \
  --dataset.revision=v3.0 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --steps=50 \
  --batch_size=8 \
  --num_workers=4 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --output_dir=/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/act_splitD_50step_retry \
  --job_name=act_splitD_50step_retry
```

