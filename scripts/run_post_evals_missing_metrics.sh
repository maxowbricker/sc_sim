#!/usr/bin/env bash
# Thin wrapper — implementation is scripts/run_post_evals_missing_metrics.py
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  exec "${CONDA_PREFIX}/bin/python" scripts/run_post_evals_missing_metrics.py "$@"
else
  exec python scripts/run_post_evals_missing_metrics.py "$@"
fi
