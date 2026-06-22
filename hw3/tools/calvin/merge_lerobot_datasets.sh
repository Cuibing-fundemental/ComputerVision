#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash tools/merge_lerobot_datasets.sh <output_dataset_dir> <input_dataset_dir_1> [input_dataset_dir_2 ...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$1"
shift
INPUT_DIRS=("$@")

HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME HF_ENDPOINT

OUTPUT_REPO_ID="$(basename "$OUTPUT_DIR")"

if [ -e "$OUTPUT_DIR" ]; then
  echo "Output path already exists, refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

TMP_PARENT="${TMPDIR:-/root/autodl-tmp/cby/projects/calvin/output}"
mkdir -p "$TMP_PARENT"
TMP_BASE="$TMP_PARENT/lerobot_merge_${OUTPUT_REPO_ID}_$$"
mkdir -p "$TMP_BASE"
trap 'rm -rf "$TMP_BASE"' EXIT

NORMALIZED_DIRS=()
for input_dir in "${INPUT_DIRS[@]}"; do
  if [ ! -d "$input_dir" ]; then
    echo "Input dataset directory not found: $input_dir" >&2
    exit 1
  fi

  normalized_dir="$TMP_BASE/$(basename "$input_dir")"
  python "$SCRIPT_DIR/normalize_lerobot_dataset.py" \
    --src "$input_dir" \
    --dst "$normalized_dir"
  NORMALIZED_DIRS+=("$normalized_dir")
done

REPO_IDS="["
ROOTS="["
for normalized_dir in "${NORMALIZED_DIRS[@]}"; do
  repo_id="$(basename "$normalized_dir")"
  REPO_IDS="${REPO_IDS}'${repo_id}', "
  ROOTS="${ROOTS}'${normalized_dir}', "
done
REPO_IDS="${REPO_IDS%, }]"
ROOTS="${ROOTS%, }]"

conda run -n lerobot lerobot-edit-dataset \
  --new_repo_id "$OUTPUT_REPO_ID" \
  --new_root "$OUTPUT_DIR" \
  --operation.type merge \
  --operation.repo_ids "$REPO_IDS" \
  --operation.roots "$ROOTS"
