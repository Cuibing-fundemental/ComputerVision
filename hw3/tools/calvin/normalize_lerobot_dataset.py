#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


FEATURE_RENAMES = {
    "observation.image": "observation.images.image",
    "observation.wrist_image": "observation.images.wrist_image",
}

FEATURE_NAME_RENAMES = {
    ("action", ("action",)): ["actions"],
}

META_COLUMN_RENAMES = {
    "stats/state/min": "stats/observation.state/min",
    "stats/state/max": "stats/observation.state/max",
    "stats/state/mean": "stats/observation.state/mean",
    "stats/state/std": "stats/observation.state/std",
    "stats/state/count": "stats/observation.state/count",
    "stats/actions/min": "stats/action/min",
    "stats/actions/max": "stats/action/max",
    "stats/actions/mean": "stats/action/mean",
    "stats/actions/std": "stats/action/std",
    "stats/actions/count": "stats/action/count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    return parser.parse_args()


def normalize_info(info_path: Path) -> None:
    with info_path.open() as f:
        info = json.load(f)

    features = info["features"]
    normalized = {}
    for key, value in features.items():
        new_key = FEATURE_RENAMES.get(key, key)
        new_value = dict(value)
        names = tuple(new_value.get("names") or [])
        replacement_names = FEATURE_NAME_RENAMES.get((new_key, names))
        if replacement_names is not None:
            new_value["names"] = replacement_names
        normalized[new_key] = new_value

    preferred_order = [
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "source_frame_index",
        "source_episode_index",
    ]

    ordered = {}
    for key in preferred_order:
        if key in normalized:
            ordered[key] = normalized[key]
    for key, value in normalized.items():
        if key not in ordered:
            ordered[key] = value

    info["features"] = ordered

    with info_path.open("w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def normalize_data_parquet(parquet_path: Path) -> None:
    table = pq.read_table(parquet_path)
    rename_map = {old: new for old, new in FEATURE_RENAMES.items() if old in table.column_names}
    if not rename_map:
        return

    df = table.to_pandas()
    df = df.rename(columns=rename_map)
    df.to_parquet(parquet_path, index=False)


def normalize_meta_parquet(parquet_path: Path) -> None:
    table = pq.read_table(parquet_path)
    rename_map = {old: new for old, new in META_COLUMN_RENAMES.items() if old in table.column_names}
    if not rename_map:
        return

    df = table.to_pandas()
    df = df.rename(columns=rename_map)
    df.to_parquet(parquet_path, index=False)


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination dataset directory already exists: {dst}")

    shutil.copytree(src, dst)

    normalize_info(dst / "meta" / "info.json")

    for parquet_path in sorted((dst / "data").rglob("*.parquet")):
        normalize_data_parquet(parquet_path)

    episodes_dir = dst / "meta" / "episodes"
    if episodes_dir.exists():
        for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
            normalize_meta_parquet(parquet_path)


if __name__ == "__main__":
    main()
