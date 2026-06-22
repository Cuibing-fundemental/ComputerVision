#!/usr/bin/env bash
set -euo pipefail

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_ROOT="${REPO_ROOT}/../data/splitA_old"
FALLBACK_ROOT="${REPO_ROOT}/../data/splitA_new"
DATASET_ROOT="${1:-${DEFAULT_ROOT}}"

if [[ ! -d "${DATASET_ROOT}" && "${DATASET_ROOT}" == "${DEFAULT_ROOT}" && -d "${FALLBACK_ROOT}" ]]; then
    DATASET_ROOT="${FALLBACK_ROOT}"
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset directory not found: ${DATASET_ROOT}" >&2
    echo "Pass the dataset root explicitly, for example:" >&2
    echo "  $0 /root/autodl-tmp/cby/homework/ComputerVision/hw3/data/splitA_old" >&2
    exit 1
fi

cd "${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python tools/fix_v21_episode_stats_counts.py "${DATASET_ROOT}"

python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
    --repo-id local/splitA_old \
    --root "${DATASET_ROOT}" \
    --push-to-hub=false \
    --force-conversion
