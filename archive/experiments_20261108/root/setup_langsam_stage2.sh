#!/usr/bin/env bash
set -euo pipefail
cd /home/work/123-0125
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR=$PWD/.cache/uv
export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
mkdir -p logs vendor "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
trap 'printf "%s\n" "$?" > logs/langsam_setup.exit' EXIT
printf '\nSTAGE 1: preserve VLM package inventory\n'
uv pip freeze --python .venv-vlm/bin/python > logs/vlm_packages_before_langsam.txt
uv venv --python .venv-vlm/bin/python .venv-langsam
printf '\nSTAGE 2: install isolated PyTorch\n'
uv pip install --python .venv-langsam/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
printf '\nSTAGE 3: fetch LangSAM source\n'
git clone --depth 1 https://github.com/luca-medeiros/lang-segment-anything.git vendor/lang-segment-anything
git -C vendor/lang-segment-anything rev-parse HEAD > logs/langsam_source_commit.txt
cat > langsam_constraints.txt <<'CONSTRAINTS'
torch==2.4.1
torchvision==0.19.1
transformers==4.44.2
numpy==1.26.4
opencv-python-headless==4.10.0.84
gradio==5.29.0
litserve==0.2.8
CONSTRAINTS
printf '\nSTAGE 4: build prerequisites\n'
uv pip install --python .venv-langsam/bin/python setuptools wheel ninja packaging
printf '\nSTAGE 5: LangSAM and SAM dependencies\n'
uv pip install --python .venv-langsam/bin/python --no-build-isolation -c langsam_constraints.txt ./vendor/lang-segment-anything
printf '\nSTAGE 6: validate installation\n'
uv pip check --python .venv-langsam/bin/python
.venv-langsam/bin/python -c "import torch,torchvision,transformers; from lang_sam import LangSAM; print('IMPORT_OK',torch.__version__,torchvision.__version__,transformers.__version__); print('CUDA_OK',torch.cuda.is_available()); print('CUDA_SUM',torch.ones(4,device='cuda').sum().item())"
uv pip freeze --python .venv-langsam/bin/python > logs/langsam_packages.lock.txt
uv pip freeze --python .venv-vlm/bin/python > logs/vlm_packages_after_langsam.txt
cmp logs/vlm_packages_before_langsam.txt logs/vlm_packages_after_langsam.txt
printf '\nSETUP_COMPLETE; VLM_PACKAGES_UNCHANGED\n'
