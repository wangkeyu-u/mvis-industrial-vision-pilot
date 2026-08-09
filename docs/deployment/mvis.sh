#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

cd "$PROJECT_ROOT"

COMMAND=${1:-help}
if [ "$#" -gt 0 ]; then
  shift
fi

case "$COMMAND" in
  sync)
    exec uv sync --group dev "$@"
    ;;
  test)
    uv run ruff check src/api src/core src/observability tests/api
    exec uv run pytest -q tests/api
    ;;
  test-all)
    uv run ruff check src tests
    exec uv run pytest -q
    ;;
  preflight)
    MODEL_MODE=${1:-auto}
    ANALYSIS_MODE=${2:-vlm_only}
    exec env MVIS_MODEL_MODE="$MODEL_MODE" MVIS_ANALYSIS_MODE="$ANALYSIS_MODE" \
      uv run python -m src.api preflight
    ;;
  validate-mock)
    REQUESTS=${1:-100}
    exec env MVIS_MODEL_MODE=mock MVIS_ANALYSIS_MODE=vlm_only \
      uv run python -m src.api validate --requests "$REQUESTS" \
      --output docs/deployment/stage4_validation_report.json
    ;;
  serve)
    MODEL_MODE=${1:-auto}
    ANALYSIS_MODE=${2:-vlm_only}
    PORT=${MVIS_PORT:-8001}
    exec env MVIS_MODEL_MODE="$MODEL_MODE" MVIS_ANALYSIS_MODE="$ANALYSIS_MODE" \
      uv run python -m src.api serve --host 127.0.0.1 --port "$PORT"
    ;;
  release-validate)
    ANALYSIS_MODE=${1:-fused}
    SAMPLE=${2:-}
    if [ -z "$SAMPLE" ]; then
      echo "usage: $0 release-validate <vlm_only|specialist_only|fused> <sample>" >&2
      exit 2
    fi
    exec env MVIS_MODEL_MODE=real MVIS_ANALYSIS_MODE="$ANALYSIS_MODE" \
      uv run python -m src.api release-validate \
      --analysis-mode "$ANALYSIS_MODE" --sample "$SAMPLE" \
      --warmup-runs 5 --measured-runs 30 \
      --json-output docs/deployment/phase7_release_report.json \
      --markdown-output docs/deployment/phase7_release_report.md
    ;;
  *)
    echo "usage: $0 {sync|test|test-all|preflight|validate-mock|serve|release-validate}" >&2
    exit 2
    ;;
esac
