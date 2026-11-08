#!/usr/bin/env bash
set -euo pipefail
cd /home/work/123-0125
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR=$PWD/.cache/uv
export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
export SAM2_BUILD_ALLOW_ERRORS=0
trap 'printf "%s\n" "$?" > logs/langsam_extensions.exit' EXIT
uv pip install --python .venv-langsam/bin/python --no-cache --no-build-isolation --no-deps --reinstall-package sam-2 'sam-2 @ git+https://github.com/facebookresearch/segment-anything-2@c2ec8e14a185632b0a5d8b161928ceb50197eddc'
.venv-langsam/bin/python -c "import torch; from sam2 import _C; print('SAM_EXTENSION_OK',_C.__file__)"
.venv-langsam/bin/python -u check_langsam_inference.py
uv pip check --python .venv-langsam/bin/python
uv pip freeze --python .venv-langsam/bin/python > logs/langsam_packages.lock.txt
uv pip freeze --python .venv-vlm/bin/python > logs/vlm_packages_after_langsam.txt
cmp logs/vlm_packages_before_langsam.txt logs/vlm_packages_after_langsam.txt
printf '\nEXTENSIONS_CHECK_COMPLETE; VLM_PACKAGES_UNCHANGED\n'
