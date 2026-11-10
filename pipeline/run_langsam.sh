#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD/vendor/lang-segment-anything
export HF_HOME=$PWD/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export TORCH_HOME=$PWD/.cache/torch
export TORCH_CUDA_ARCH_LIST="$(.venv-langsam/bin/python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))))')"
export TORCH_EXTENSIONS_DIR=$PWD/.cache/torch_extensions
export MAX_JOBS=2
if [ "$#" -eq 0 ]; then set -- pipeline/check_langsam_model.py; fi
exec .venv-langsam/bin/python "$@"
