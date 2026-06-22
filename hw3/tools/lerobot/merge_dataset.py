from pathlib import Path
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm

data_root = Path("/root/autodl-tmp/cby/homework/ComputerVision/hw3/data")

datasets = [
    LeRobotDataset("local/splitA_trainfmt", root=data_root / "splitA_v30"),
    LeRobotDataset("local/splitB_trainfmt", root=data_root / "splitB_v30"),
    LeRobotDataset("local/splitC_trainfmt", root=data_root / "splitC_v30"),
]

# ✅ 创建 merged dataset（以A为模板）
merged = LeRobotDataset.create_empty_like(
    datasets[0],
    root=data_root / "splitABC_v30",
    repo_id="local/splitABC"
)

# =========================
# 🔥 Step 1: 统计总 episode 数
# =========================
total_episodes = sum(len(ds.episodes) for ds in datasets)

pbar = tqdm(total=total_episodes, desc="Merging CALVIN A/B/C", unit="episode")

episode_counter = 0

# =========================
# 🔥 Step 2: merge + 防冲突
# =========================
for ds_id, ds in enumerate(datasets):
    for ep in ds.episodes:

        # ✔ 防止 episode id / index 冲突（非常关键）
        if isinstance(ep, dict):
            ep["episode_index"] = episode_counter

            # 可选：标记来源（debug / ablation 很有用）
            ep["dataset_source"] = ds_id  # 0=A,1=B,2=C

        merged.add_episode(ep)

        episode_counter += 1
        pbar.update(1)

pbar.close()

# =========================
# 🔥 Step 3: 保存 dataset
# =========================
merged.save()

print("✅ Merge finished!")
print(f"Total episodes: {episode_counter}")
print(f"Saved to: {data_root / 'splitABC_v30'}")