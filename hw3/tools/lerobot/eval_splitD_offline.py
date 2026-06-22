#!/usr/bin/env python

"""Offline zero-shot evaluation on CALVIN splitD LeRobot v3.0 datasets.

This script evaluates one or more policy checkpoints against a held-out dataset
by measuring action prediction error. It also reports chunk-aware metrics for
action-chunking policies such as ACT and Diffusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitD_v30")
DEFAULT_DATASET_REPO_ID = "local/splitD_v30"
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/projects/lerobot/output/eval_splitD_zero_shot")
DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/cby/homework/ComputerVision/hw3/lerobot/output/envB/wandb/checkpoints/100000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[DEFAULT_CHECKPOINT],
        help="One or more local pretrained_model directories.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--task-prefix", type=str, default=None, help="Only evaluate tasks with this prefix.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def checkpoint_label(path: Path) -> str:
    parts = path.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx >= 1 and idx + 1 < len(parts):
            return f"{parts[idx - 1]}:{parts[idx + 1]}"
    return path.name


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _get_task_name(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _make_dataset(
    repo_id: str,
    root: Path,
    episodes: list[int] | None,
    action_horizon: int,
) -> LeRobotDataset:
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {"action": [i / meta.fps for i in range(action_horizon)]}
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )


def _select_episodes(
    meta: LeRobotDatasetMetadata,
    max_episodes: int | None,
    task_prefix: str | None,
) -> list[int] | None:
    selected: list[int] | None = None
    if task_prefix:
        selected = meta.filter_episodes(
            lambda ep: any(str(task).startswith(task_prefix) for task in ep["tasks"])
        )
    if selected is not None and max_episodes is not None:
        selected = selected[:max_episodes]
    elif selected is None and max_episodes is not None:
        selected = list(range(min(max_episodes, meta.total_episodes)))
    return selected


def _build_processors(
    policy_cfg: PreTrainedConfig,
    checkpoint: Path,
    device: str,
) -> tuple[Any, Any]:
    rename_map = {
        "observation.images.image": "observation.image",
        "observation.images.wrist_image": "observation.wrist_image",
    }
    return make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
        postprocessor_overrides={"device_processor": {"device": device}},
    )


def _evaluate_checkpoint(
    checkpoint: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    max_episodes: int | None,
    max_batches: int | None,
    task_prefix: str | None,
) -> dict[str, Any]:
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = device
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(checkpoint, config=policy_cfg)
    policy.eval()

    preprocessor, _ = _build_processors(policy_cfg, checkpoint, device)

    action_horizon = int(
        getattr(policy_cfg, "n_action_steps", None)
        or getattr(policy_cfg, "chunk_size", None)
        or 1
    )
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    episodes = _select_episodes(dataset_meta, max_episodes=max_episodes, task_prefix=task_prefix)
    dataset = _make_dataset(dataset_repo_id, dataset_root, episodes, action_horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_valid_steps = 0
    total_valid_scalars = 0
    mae_sum = 0.0
    mse_sum = 0.0
    max_abs_error = 0.0
    first_step_abs_sum = 0.0
    first_step_valid_scalars = 0
    chunk_last_abs_sum = 0.0
    chunk_last_valid_scalars = 0
    episode_final_l2_sum = 0.0
    episode_count = 0
    task_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"mae_sum": 0.0, "valid_scalars": 0.0, "episodes": 0.0, "final_l2_sum": 0.0}
    )
    seen_episode_ids: set[int] = set()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        raw_batch = batch
        task_names = [_get_task_name(task) for task in raw_batch["task"]]
        batch = preprocessor(batch)

        with torch.no_grad():
            pred = policy.predict_action_chunk(batch)

        target = batch["action"]
        if "action_is_pad" in batch:
            action_mask = ~batch["action_is_pad"].bool()
        else:
            action_mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)

        valid_mask = action_mask.unsqueeze(-1).expand_as(target)
        diff = pred - target
        abs_diff = diff.abs()
        sq_diff = diff.square()

        valid_abs = abs_diff[valid_mask]
        valid_sq = sq_diff[valid_mask]
        valid_steps = int(action_mask.sum().item())
        valid_scalars = int(valid_mask.sum().item())

        total_valid_steps += valid_steps
        total_valid_scalars += valid_scalars
        mae_sum += valid_abs.sum().item()
        mse_sum += valid_sq.sum().item()
        if valid_abs.numel() > 0:
            max_abs_error = max(max_abs_error, float(valid_abs.max().item()))

        first_mask = action_mask[:, :1].unsqueeze(-1).expand(-1, 1, target.shape[-1])
        if first_mask.any():
            first_step_abs_sum += abs_diff[:, :1][first_mask].sum().item()
            first_step_valid_scalars += int(first_mask.sum().item())

        last_valid_idx = action_mask.sum(dim=1) - 1
        for sample_idx in range(target.shape[0]):
            task_name = task_names[sample_idx]
            sample_mask = valid_mask[sample_idx]
            sample_abs = abs_diff[sample_idx][sample_mask]
            sample_valid_scalars = int(sample_mask.sum().item())
            if sample_valid_scalars > 0:
                task_stats[task_name]["mae_sum"] += sample_abs.sum().item()
                task_stats[task_name]["valid_scalars"] += sample_valid_scalars

            last_idx = int(last_valid_idx[sample_idx].item())
            if last_idx >= 0:
                last_abs = abs_diff[sample_idx, last_idx]
                chunk_last_abs_sum += last_abs.sum().item()
                chunk_last_valid_scalars += int(last_abs.numel())

                episode_id = _to_int(raw_batch["episode_index"][sample_idx])
                if episode_id not in seen_episode_ids:
                    seen_episode_ids.add(episode_id)
                    episode_count += 1
                    task_stats[task_name]["episodes"] += 1
                    final_l2 = float(torch.linalg.vector_norm(diff[sample_idx, last_idx], ord=2).item())
                    episode_final_l2_sum += final_l2
                    task_stats[task_name]["final_l2_sum"] += final_l2

    task_metrics = []
    for task_name, stats in sorted(task_stats.items()):
        valid_scalars = int(stats["valid_scalars"])
        episodes_count = int(stats["episodes"])
        task_metrics.append(
            {
                "task": task_name,
                "action_mae": stats["mae_sum"] / valid_scalars if valid_scalars else None,
                "episode_final_l2": stats["final_l2_sum"] / episodes_count if episodes_count else None,
                "episodes": episodes_count,
            }
        )

    overall = {
        "label": checkpoint_label(checkpoint),
        "checkpoint": str(checkpoint),
        "policy_type": policy_cfg.type,
        "device": device,
        "frames": len(dataset),
        "episodes": episode_count,
        "action_horizon": action_horizon,
        "valid_action_steps": total_valid_steps,
        "valid_action_scalars": total_valid_scalars,
        "action_mae": mae_sum / total_valid_scalars if total_valid_scalars else None,
        "action_rmse": (mse_sum / total_valid_scalars) ** 0.5 if total_valid_scalars else None,
        "action_max_abs_error": max_abs_error if total_valid_scalars else None,
        "first_step_mae": first_step_abs_sum / first_step_valid_scalars if first_step_valid_scalars else None,
        "last_step_mae": chunk_last_abs_sum / chunk_last_valid_scalars if chunk_last_valid_scalars else None,
        "chunk_drift_ratio": (
            (chunk_last_abs_sum / chunk_last_valid_scalars) / (first_step_abs_sum / first_step_valid_scalars)
            if first_step_valid_scalars and chunk_last_valid_scalars and first_step_abs_sum > 0
            else None
        ),
        "episode_final_l2": episode_final_l2_sum / episode_count if episode_count else None,
        "task_prefix": task_prefix,
    }

    return {"overall": overall, "per_task": task_metrics}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for checkpoint in args.checkpoints:
        results.append(
            _evaluate_checkpoint(
                checkpoint=checkpoint,
                dataset_repo_id=args.dataset_repo_id,
                dataset_root=args.dataset_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                max_episodes=args.max_episodes,
                max_batches=args.max_batches,
                task_prefix=args.task_prefix,
            )
        )

    summary_rows = [result["overall"] for result in results]
    per_task_rows: list[dict[str, Any]] = []
    for result in results:
        label = result["overall"]["label"]
        for row in result["per_task"]:
            per_task_rows.append({"label": label, **row})

    payload = {
        "dataset_root": str(args.dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "results": results,
    }

    json_path = args.output_dir / "metrics.json"
    summary_csv_path = args.output_dir / "summary.csv"
    per_task_csv_path = args.output_dir / "per_task.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(summary_csv_path, summary_rows)
    _write_csv(per_task_csv_path, per_task_rows)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written: {json_path}")
    print(f"written: {summary_csv_path}")
    print(f"written: {per_task_csv_path}")


if __name__ == "__main__":
    main()
