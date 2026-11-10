#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME=$PWD/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export TORCH_HOME=$PWD/.cache/torch
qwen_model=$(.venv-vlm/bin/python -c 'import sys; sys.path.insert(0,"pipeline"); from inspection_io import default_qwen_model; print(default_qwen_model())')
exec .venv-vlm/bin/python -u archive/experiments_20261108/root/vlm_vllm_example.py \
  --model "$qwen_model" --image data/initial_8/images/BEGO7553_frame_001098.jpg --gpu-mem 0.35 "$@"
