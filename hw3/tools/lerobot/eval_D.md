PYTHONPATH=src /root/autodl-tmp/miniconda3/envs/lerobot/bin/python tools/eval_splitD_offline.py \
    --checkpoints \
      /root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model \
    --dataset-root /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_v30 \
    --dataset-repo-id local/splitD_v30 \
    --device cuda \
    --batch-size 16 \
    --output-dir /root/autodl-tmp/cby/homework/ComputerVision/hw3/output/eval_splitD_zero_shot_compare

stdbuf -oL -eL /root/autodl-tmp/miniconda3/envs/lerobot/bin/python \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/tools/eval_splitD_act_calvin.py \
  --checkpoints /root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model \
  --dataset-root /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_v30 \
  --dataset-repo-id local/splitD_v30 \
  --device cuda \
  --batch-size 16 \
  --output-dir /root/autodl-tmp/cby/homework/ComputerVision/hw3/output/eval_splitD_zero_shot_compare \
  > /root/autodl-tmp/cby/projects/log3.txt 2>&1


stdbuf -oL -eL /root/autodl-tmp/miniconda3/envs/lerobot/bin/python \
  /root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/tools/eval_splitD_act_calvin.py \
  --checkpoints /root/autodl-tmp/cby/homework/ComputerVision/hw3/part2/outputs/splitABC_v30_act/checkpoints/100000/pretrained_model \
  --dataset-root /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_v30 \
  --dataset-repo-id local/splitD_v30 \
  --device cuda \
  --batch-size 32 \
  --num-workers 64\
  --output-dir /root/autodl-tmp/cby/homework/ComputerVision/hw3/output/eval_ABC_splitD \
  > /root/autodl-tmp/cby/projects/logABC.txt 2>&1