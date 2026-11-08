set -euo pipefail
cd /home/work/123-0125
trap 'printf "%s\n" "$?" > logs/langsam_final_check.exit' EXIT
bash run_langsam.sh -c "import torch; from sam2 import _C; print('SAM_EXTENSION_OK')"
bash run_langsam.sh check_langsam_inference.py
uv pip check --python .venv-langsam/bin/python
uv pip freeze --python .venv-langsam/bin/python > logs/langsam_packages.lock.txt
uv pip freeze --python .venv-vlm/bin/python > logs/vlm_packages_after_langsam.txt
cmp logs/vlm_packages_before_langsam.txt logs/vlm_packages_after_langsam.txt
printf '\nFINAL_CHECK_OK; VLM_PACKAGES_UNCHANGED\n'
