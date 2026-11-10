#!/usr/bin/env bash
set -euo pipefail
inspection_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
exec "$inspection_root/.venv-eval/bin/python" -u "$inspection_root/pipeline/inspect_image.py" "$@"
