# envB 模型评测命令

目标模型：

- checkpoint: `/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model`
- 数据集根目录: `/root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitB_v30`

说明：

- 这个 checkpoint 的 `train_config.json` 里 `env=null`，所以它是一个离线训练产物。
- LeRobot 自带的 `lerobot-eval` 只能做“环境 rollout 评测”，不直接读取 `--dataset.root` 来算 success rate。
- 如果你要的是“模型在某个仿真/真实环境里的成功率”，需要补上和 `splitB_v30` 相匹配的 `--env.type` / `--env.task`。
- 如果你只是想先确认本地数据集和 checkpoint 都能正常加载，可以先跑下面的本地检查命令。

## 1. 本地检查

先确认数据集元信息和 checkpoint 结构：

```bash
export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

/root/autodl-tmp/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_info

python - <<'PY'
from pathlib import Path
import json

ckpt = Path("/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model")
with open(ckpt / "config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
print("policy type:", cfg["type"])
print("input features:", list(cfg["input_features"].keys()))
print("output features:", list(cfg["output_features"].keys()))
PY
```

再抽样看一个数据集 episode：

```bash
export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

/root/autodl-tmp/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/splitB_v30 \
  --root /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitB_v30 \
  --episode-index 0 \
  --mode local
```

## 2. 标准 `lerobot-eval` 命令模板

如果你已经确认 `splitB_v30` 对应的 benchmark 环境是哪一个，就用下面这条命令跑正式评测：

```bash
export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src

/root/autodl-tmp/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_eval \
  --policy.path=/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model \
  --env.type=<匹配 splitB_v30 的 env_type> \
  --env.task=<匹配 splitB_v30 的 task_1,task_2,...> \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --output_dir=/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB_eval
```

## 3. 这份 checkpoint 的关键信息

从 `output/envB/wandb/checkpoints/100000/pretrained_model/config.json` 可以直接读到：

- policy type: `act`
- 输入：
  - `observation.image`
  - `observation.wrist_image`
  - `observation.state`
- 输出：
  - `action`

如果你把对应环境补齐，我可以再把上面 `<匹配 splitB_v30 的 env_type>` 和 `<匹配 splitB_v30 的 task_1,task_2,...>` 直接替换成最终可执行版本。
