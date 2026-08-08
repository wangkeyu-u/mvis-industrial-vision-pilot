#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${MVIS_PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

cd "$PROJECT_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python runtime is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

export MVIS_MODEL_MODE=real
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.

exec "$PYTHON_BIN" -m src.api real-validate \
  --sample assets/system_architecture.png \
  --host 127.0.0.1 \
  --port "${MVIS_PHASE5_PORT:-18086}" \
  --warmup-runs 5 \
  --measured-runs 30 \
  --json-output docs/deployment/phase5_real_runtime_report.json \
  --markdown-output docs/deployment/phase5_real_runtime_report.md
