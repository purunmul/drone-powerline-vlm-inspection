#!/usr/bin/env bash
set -euo pipefail
dataset_project="$(cd -- "$(dirname -- "$0")/.." && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="$dataset_project/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$dataset_project/.venv-eval/bin/python" -u "$dataset_project/pipeline/run_dataset.py" "$@"
