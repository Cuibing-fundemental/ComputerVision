#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def rename_stats_dict(stats: dict, rename_map: dict[str, str]) -> dict:
    return {rename_map.get(key, key): value for key, value in stats.items()}


def rename_episode_columns(columns: list[str], rename_map: dict[str, str]) -> list[str]:
    renamed = []
    for col in columns:
        new_col = col
        for old, new in rename_map.items():
            prefix = f"stats/{old}/"
            if col.startswith(prefix):
                new_col = f"stats/{new}/{col[len(prefix):]}"
                break
        renamed.append(new_col)
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--rename-map", required=True, help="JSON object mapping old feature names to new names")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    rename_map = json.loads(args.rename_map)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    info_path = dst / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"] = {rename_map.get(key, key): value for key, value in info["features"].items()}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    stats_path = dst / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    with open(stats_path, "w") as f:
        json.dump(rename_stats_dict(stats, rename_map), f, indent=4)

    for parquet_path in sorted((dst / "data").glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        df = df.rename(columns=rename_map)
        df.to_parquet(parquet_path, index=False)

    for parquet_path in sorted((dst / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        df.columns = rename_episode_columns(list(df.columns), rename_map)
        df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
