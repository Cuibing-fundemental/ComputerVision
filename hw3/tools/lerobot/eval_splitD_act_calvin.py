#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import functools
import json
import multiprocessing
import math
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
HW3_ROOT = REPO_ROOT.parent
CALVIN_ROOT = HW3_ROOT / "calvin"
CALVIN_MODELS_ROOT = CALVIN_ROOT / "calvin_models"
CALVIN_ENV_ROOT = CALVIN_ROOT / "calvin_env"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if CALVIN_MODELS_ROOT.exists() and str(CALVIN_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(CALVIN_MODELS_ROOT))
if CALVIN_ENV_ROOT.exists() and str(CALVIN_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(CALVIN_ENV_ROOT))
if CALVIN_ROOT.exists() and str(CALVIN_ROOT) not in sys.path:
    sys.path.insert(0, str(CALVIN_ROOT))

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


DEFAULT_DATASET_ROOT = HW3_ROOT / "data" / "splitD_v30"
DEFAULT_DATASET_REPO_ID = "local/splitD_v30"
DEFAULT_CHECKPOINT = (
    HW3_ROOT / "lerobot" / "output" / "envB" / "wandb" / "checkpoints" / "100000" / "pretrained_model"
)
DEFAULT_OUTPUT_DIR = HW3_ROOT / "output" / "eval_splitD_zero_shot_compare"

EP_LEN = 360
NUM_SEQUENCES = 1000

TASK_CATEGORIES = {
    "rotate_red_block_right": 1,
    "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1,
    "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1,
    "rotate_pink_block_left": 1,
    "push_red_block_right": 1,
    "push_red_block_left": 1,
    "push_blue_block_right": 1,
    "push_blue_block_left": 1,
    "push_pink_block_right": 1,
    "push_pink_block_left": 1,
    "move_slider_left": 2,
    "move_slider_right": 2,
    "open_drawer": 3,
    "close_drawer": 3,
    "lift_red_block_table": 4,
    "lift_red_block_slider": 5,
    "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4,
    "lift_blue_block_slider": 5,
    "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4,
    "lift_pink_block_slider": 5,
    "lift_pink_block_drawer": 6,
    "place_in_slider": 7,
    "place_in_drawer": 7,
    "turn_on_lightbulb": 8,
    "turn_off_lightbulb": 8,
    "turn_on_led": 8,
    "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10,
    "unstack_block": 11,
}

TASKS = {
    "rotate_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "rotate_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "move_slider_left": [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}],
    "open_drawer": [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}],
    "lift_red_block_table": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_red_block_slider": [
        {"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
        {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
    ],
    "lift_red_block_drawer": [{"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_blue_block_table": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_blue_block_slider": [
        {"condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
    ],
    "lift_blue_block_drawer": [{"condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_pink_block_table": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "lift_pink_block_slider": [
        {"condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
        {"condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
    ],
    "lift_pink_block_drawer": [{"condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "place_in_slider": [
        {"condition": {"red_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"red_block": "slider_right", "grasped": 0}},
        {"condition": {"red_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"red_block": "slider_left", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"blue_block": "slider_right", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"blue_block": "slider_left", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"pink_block": "slider_right", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"pink_block": "slider_left", "grasped": 0}},
    ],
    "place_in_drawer": [
        {"condition": {"red_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
    "stack_block": [
        {"condition": {"red_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"red_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
    ],
    "unstack_block": [
        {"condition": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "blue_block": "table"}},
        {"condition": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "pink_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "red_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "pink_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "red_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "blue_block": "table"}},
    ],
    "turn_on_lightbulb": [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
    "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
    "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
    "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
    "push_into_drawer": [
        {"condition": {"red_block": "table", "blue_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "table", "red_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "table", "blue_block": ["slider_right", "slider_left"], "red_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot ACT checkpoint on CALVIN splitD for offline action error and rollout success."
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--task-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calvin-env-dataset",
        type=Path,
        default=None,
        help="Root directory for CALVIN environment data. If omitted, the script tries common local paths.",
    )
    parser.add_argument("--max-rollout-sequences", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def checkpoint_label(path: Path) -> str:
    parts = path.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx >= 1 and idx + 1 < len(parts):
            return f"{parts[idx - 1]}:{parts[idx + 1]}"
    return path.name


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


def _build_processors(policy_cfg: PreTrainedConfig, checkpoint: Path, device: str) -> tuple[Any, Any]:
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


def evaluate_offline_checkpoint(
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
    print(f"[offline] loading checkpoint: {checkpoint}", flush=True)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = device
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(checkpoint, config=policy_cfg)
    policy.eval()

    preprocessor, _ = _build_processors(policy_cfg, checkpoint, device)

    action_horizon = int(getattr(policy_cfg, "n_action_steps", None) or getattr(policy_cfg, "chunk_size", None) or 1)
    dataset_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    episodes = _select_episodes(dataset_meta, max_episodes=max_episodes, task_prefix=task_prefix)
    dataset = _make_dataset(dataset_repo_id, dataset_root, episodes, action_horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_batches = len(loader)
    print(
        f"[offline] dataset={dataset_root} frames={len(dataset)} selected_episodes={len(episodes) if episodes is not None else dataset_meta.total_episodes} "
        f"action_horizon={action_horizon} batches={total_batches}",
        flush=True,
    )
    progress_every = max(1, math.ceil(total_batches / 20)) if total_batches > 0 else 1

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

        current_batch = batch_idx + 1
        if current_batch == 1 or current_batch == total_batches or current_batch % progress_every == 0:
            current_mae = mae_sum / total_valid_scalars if total_valid_scalars else None
            print(
                f"[offline] progress {current_batch}/{total_batches} "
                f"episodes={episode_count} valid_steps={total_valid_steps} "
                f"action_mae={current_mae:.6f}" if current_mae is not None else
                f"[offline] progress {current_batch}/{total_batches} episodes={episode_count} valid_steps={total_valid_steps}",
                flush=True,
            )

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


def temp_seed(seed: int):
    class _TempSeed:
        def __enter__(self_inner):
            self_inner.state = np.random.get_state()
            np.random.seed(seed)

        def __exit__(self_inner, exc_type, exc, tb):
            np.random.set_state(self_inner.state)
            return False

    return _TempSeed()


def check_condition(state: dict[str, Any], condition: dict[str, Any]) -> bool:
    for key, value in condition.items():
        if isinstance(value, (str, int)):
            if state[key] != value:
                return False
        elif isinstance(value, list):
            if state[key] not in value:
                return False
        else:
            raise TypeError
    return True


def update_state(state: dict[str, Any], effect: dict[str, Any]) -> dict[str, Any]:
    next_state = deepcopy(state)
    for key, value in effect.items():
        next_state[key] = value
    return next_state


def valid_task(curr_state: dict[str, Any], task: list[dict[str, Any]]) -> list[dict[str, Any]]:
    next_states = []
    for task_option in task:
        if check_condition(curr_state, task_option["condition"]):
            next_states.append(update_state(curr_state, task_option["effect"]))
    return next_states


def check_sequence(state: dict[str, Any], seq: list[str]) -> bool:
    for task_name in seq:
        states = valid_task(state, TASKS[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    categories = [TASK_CATEGORIES[name] for name in seq]
    return len(categories) == len(set(categories))


def get_sequences_for_state2(args: tuple[dict[str, Any], int, int]) -> list[tuple[str, ...]]:
    state, num_sequences, seed = args
    np.random.seed(seed)
    seq_len = 5
    results: list[tuple[str, ...]] = []
    task_names = list(TASKS.keys())
    while len(results) < num_sequences:
        seq = tuple(np.random.choice(task_names, size=seq_len, replace=False).tolist())
        if check_sequence(deepcopy(state), list(seq)):
            results.append(seq)
    return results


def flatten_sequences(nested: list[list[tuple[str, ...]]]) -> list[tuple[str, ...]]:
    return [item for sublist in nested for item in sublist]


@functools.lru_cache
def get_sequences(num_sequences: int = NUM_SEQUENCES, num_workers: int | None = None) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    valid_layout = lambda vals: vals.count("table") in [1, 2] and vals.count("slider_right") < 2 and vals.count("slider_left") < 2
    value_combinations = filter(valid_layout, product(*possible_conditions.values()))
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]
    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    with temp_seed(0):
        worker_count = multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            nested = list(executor.map(get_sequences_for_state2, zip(initial_states, num_sequences_per_state, range(len(initial_states)))))
        sequences = flatten_sequences(nested)
        results = list(zip(np.repeat(initial_states, num_sequences_per_state), sequences))
        np.random.shuffle(results)
    return results


class LeRobotCalvinWrapper:
    def __init__(self, checkpoint: Path, device: str) -> None:
        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cfg.device = device
        policy_cls = get_policy_class(policy_cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint, config=policy_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = _build_processors(policy_cfg, checkpoint, device)
        self.device = torch.device(device)

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def step(self, obs: dict[str, Any], goal: str) -> Any:
        state_obs = obs["robot_obs_raw"] if "robot_obs_raw" in obs else obs["robot_obs"]
        state_obs = np.asarray(state_obs, dtype=np.float32)
        raw_obs = {
            "observation.image": obs["rgb_obs"]["rgb_static"],
            "observation.wrist_image": obs["rgb_obs"]["rgb_gripper"],
            "observation.state": state_obs,
        }
        model_obs = prepare_observation_for_inference(raw_obs, self.device, goal, "")
        processed = self.preprocessor(model_obs)
        with torch.inference_mode():
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)
        action_np = action.squeeze(0).detach().float().cpu().numpy().copy()
        action_np[-1] = 1.0 if action_np[-1] > 0 else -1.0
        return action_np


def count_success(results: list[int]) -> list[float]:
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_env_state_for_initial_condition(initial_condition: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]

    seed = zlib.crc32(repr(sorted(initial_condition.items())).encode("utf-8"))
    rng = np.random.RandomState(seed)
    rng.shuffle(block_table)

    scene_obs = np.zeros(24)
    if initial_condition["slider"] == "left":
        scene_obs[0] = 0.28
    else:
        scene_obs[1] = 0.22
    if initial_condition["drawer"] == "open":
        scene_obs[3] = 0.22
    else:
        scene_obs[3] = 0.0
    scene_obs[4] = initial_condition["lightbulb"]
    scene_obs[5] = initial_condition["led"]

    if initial_condition["red_block"] == "slider_right":
        scene_obs[6:9] = block_slider_right
    elif initial_condition["red_block"] == "slider_left":
        scene_obs[6:9] = block_slider_left
    else:
        scene_obs[6:9] = block_table[0]
    scene_obs[9:12] = np.array([0.0, 0.0, rng.uniform(*block_rot_z_range)])

    if initial_condition["blue_block"] == "slider_right":
        scene_obs[12:15] = block_slider_right
    elif initial_condition["blue_block"] == "slider_left":
        scene_obs[12:15] = block_slider_left
    else:
        scene_obs[12:15] = block_table[1]
    scene_obs[15:18] = np.array([0.0, 0.0, rng.uniform(*block_rot_z_range)])

    if initial_condition["pink_block"] == "slider_right":
        scene_obs[18:21] = block_slider_right
    elif initial_condition["pink_block"] == "slider_left":
        scene_obs[18:21] = block_slider_left
    else:
        scene_obs[18:21] = block_table[rng.randint(0, 2)]
    scene_obs[21:24] = np.array([0.0, 0.0, rng.uniform(*block_rot_z_range)])
    return robot_obs, scene_obs


def evaluate_success_rate_checkpoint(
    checkpoint: Path,
    device: str,
    calvin_env_dataset: Path | None,
    max_rollout_sequences: int | None,
    debug: bool,
) -> dict[str, Any]:
    try:
        from calvin_env.envs.play_table_env import PlayTableSimEnv
        from calvin_env.envs.tasks import Tasks
    except Exception as exc:
        return {"status": "skipped", "reason": "missing_calvin_dependencies", "detail": repr(exc)}

    from omegaconf import OmegaConf
    import hydra

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize_config_dir(version_base=None, config_dir=str(CALVIN_ENV_ROOT / "conf"))

    base_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "config_data_collection.yaml")
    env_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "env" / "play_table_env.yaml")
    scene_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "scene" / "calvin_scene_D_eval.yaml")
    static_camera_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "cameras" / "cameras" / "static.yaml")
    gripper_camera_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "cameras" / "cameras" / "gripper.yaml")
    camera_cfg = OmegaConf.create({"static": static_camera_cfg, "gripper": gripper_camera_cfg})
    robot_base_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "robot" / "panda.yaml")
    robot_variant_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "robot" / "panda_longer_finger.yaml")
    robot_cfg = OmegaConf.merge(robot_base_cfg, robot_variant_cfg)
    conf_dir = CALVIN_MODELS_ROOT / "conf"
    task_cfg = OmegaConf.load(CALVIN_ENV_ROOT / "conf" / "tasks" / "new_playtable_tasks.yaml")
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    task_oracle = Tasks(task_cfg.tasks)
    cfg = OmegaConf.merge(base_cfg, {"env": env_cfg, "scene": scene_cfg, "cameras": camera_cfg, "robot": robot_cfg})
    cfg.use_vr = False
    cfg.data_path = str(CALVIN_ENV_ROOT / "data")

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
    model = LeRobotCalvinWrapper(checkpoint=checkpoint, device=device)
    eval_sequences = get_sequences(NUM_SEQUENCES)
    if max_rollout_sequences is not None:
        eval_sequences = eval_sequences[:max_rollout_sequences]
    total_sequences = len(eval_sequences)
    print(
        f"[success] checkpoint={checkpoint} rollout_sequences={total_sequences} ep_len={EP_LEN}",
        flush=True,
    )
    progress_every = max(1, math.ceil(total_sequences / 20)) if total_sequences > 0 else 1

    results = []
    task_success = Counter()
    task_total = Counter()

    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences, start=1):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        for subtask in eval_sequence:
            model.reset()
            start_info = env.get_info()
            obs = env.get_obs()
            lang_annotation = val_annotations[subtask][0]
            success = False
            for _ in range(EP_LEN):
                action = model.step(obs, lang_annotation)
                obs, _, _, current_info = env.step(action)
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    break
            task_total[subtask] += 1
            if success:
                success_counter += 1
                task_success[subtask] += 1
            else:
                break
        results.append(success_counter)
        if seq_idx == 1 or seq_idx == total_sequences or seq_idx % progress_every == 0:
            avg_seq_len = sum(results) / len(results) if results else 0.0
            sr1 = count_success(results)[0] if results else 0.0
            print(
                f"[success] progress {seq_idx}/{total_sequences} avg_seq_len={avg_seq_len:.3f} sr_1={sr1:.3f}",
                flush=True,
            )

    chain_sr = {str(i + 1): sr for i, sr in enumerate(count_success(results))}
    per_task = []
    for task_name in sorted(task_total):
        total = task_total[task_name]
        succ = task_success[task_name]
        per_task.append(
            {
                "task": task_name,
                "success": succ,
                "total": total,
                "success_rate": succ / total if total else None,
            }
        )

    return {
        "status": "ok",
        "dataset_path": str(CALVIN_ENV_ROOT / "data"),
        "num_sequences": len(results),
        "avg_seq_len": sum(results) / len(results) if results else None,
        "chain_success_rates": chain_sr,
        "sequence_success_histogram": dict(Counter(results)),
        "per_task": per_task,
        "debug": debug,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []
    per_task_rows = []

    for checkpoint in args.checkpoints:
        offline = evaluate_offline_checkpoint(
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
        success = evaluate_success_rate_checkpoint(
            checkpoint=checkpoint,
            device=args.device,
            calvin_env_dataset=args.calvin_env_dataset,
            max_rollout_sequences=args.max_rollout_sequences,
            debug=args.debug,
        )

        record = {
            "label": checkpoint_label(checkpoint),
            "checkpoint": str(checkpoint),
            "offline": offline,
            "success_rate": success,
        }
        all_results.append(record)

        summary = dict(offline["overall"])
        summary["success_eval_status"] = success.get("status")
        summary["avg_seq_len"] = success.get("avg_seq_len")
        chain_sr = success.get("chain_success_rates", {})
        summary["sr_1"] = chain_sr.get("1")
        summary["sr_2"] = chain_sr.get("2")
        summary["sr_3"] = chain_sr.get("3")
        summary["sr_4"] = chain_sr.get("4")
        summary["sr_5"] = chain_sr.get("5")
        summary["success_eval_detail"] = success.get("detail")
        summary_rows.append(summary)

        for row in offline["per_task"]:
            per_task_rows.append({"label": checkpoint_label(checkpoint), "metric_type": "offline", **row})
        for row in success.get("per_task", []):
            per_task_rows.append({"label": checkpoint_label(checkpoint), "metric_type": "success", **row})

    payload = {
        "dataset_root": str(args.dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "calvin_env_dataset": str(args.calvin_env_dataset) if args.calvin_env_dataset else None,
        "results": all_results,
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
