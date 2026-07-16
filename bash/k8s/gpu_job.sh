#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${C2C_PYTHON:-/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "错误：找不到 Python：$PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/script/k8s/gpu_job.py" "$@"
