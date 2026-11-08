#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 .venv-vlm/bin/python -u vlm_vllm_example.py "$@" 2>&1 | tee vlm_run.log
