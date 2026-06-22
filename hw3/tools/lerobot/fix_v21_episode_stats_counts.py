#!/usr/bin/env python
"""Add missing per-feature count fields to a LeRobot v2.1 episodes_stats.jsonl."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_episode_lengths(episodes_path: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lengths[int(item["episode_index"])] = int(item["length"])
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    meta_dir = args.dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    backup_path = meta_dir / "episodes_stats.no_count.jsonl"

    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    lengths = load_episode_lengths(episodes_path)
    updated_lines: list[str] = []
    changed = False

    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            count = [lengths[episode_index]]
            for feature_stats in item["stats"].values():
                if "count" not in feature_stats:
                    feature_stats["count"] = count
                    changed = True
            updated_lines.append(json.dumps(item, separators=(",", ":")) + "\n")

    if not changed:
        print(f"No missing count fields found in {stats_path}")
        return

    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)

    with stats_path.open("w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    print(f"Added missing count fields to {stats_path}")
    print(f"Original stats backup: {backup_path}")


if __name__ == "__main__":
    main()
