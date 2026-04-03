#!/usr/bin/env bash
# Run static composite baseline vs both ppo_sc_final and best_model for one RL run folder.
#
# Usage (from repo root):
#   ./scripts/compare_run_checkpoints_to_baseline.sh
#   ./scripts/compare_run_checkpoints_to_baseline.sh rl_logs_sb3/run_20260402_135637
#   ./scripts/compare_run_checkpoints_to_baseline.sh rl_logs_sb3/run_20260402_135637 496528674@qq.com_20161128
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:-rl_logs_sb3/run_20260402_135637}"
DAY="${2:-496528674@qq.com_20161128}"
DATA_ROOT="${DATA_ROOT:-data/didi/full_didi_gaia}"
EVAL_SEED="${EVAL_SEED:-42}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run folder not found: $RUN_DIR"
  exit 1
fi

FINAL="${RUN_DIR}/ppo_sc_final"
BEST="${RUN_DIR}/best_model/best_model"

echo "=========================================="
echo "Run:     $RUN_DIR"
echo "Day:     $DAY"
echo "Seed:    $EVAL_SEED"
echo "=========================================="

echo ""
echo ">>> [1/2] Final checkpoint: ppo_sc_final"
python3 scripts/compare_model_to_baseline.py \
  --model "$FINAL" \
  --day "$DAY" \
  --data-root "$DATA_ROOT" \
  --eval-seed "$EVAL_SEED" \
  --log-weights "${RUN_DIR}/baseline_final_model_weight_outputs.txt" \
  --metrics-out "${RUN_DIR}/baseline_final_model_metrics.txt"

echo ""
echo ">>> [2/2] Best eval checkpoint: best_model/best_model"
python3 scripts/compare_model_to_baseline.py \
  --model "$BEST" \
  --day "$DAY" \
  --data-root "$DATA_ROOT" \
  --eval-seed "$EVAL_SEED" \
  --log-weights "${RUN_DIR}/baseline_best_model_weight_outputs.txt" \
  --metrics-out "${RUN_DIR}/baseline_best_model_metrics.txt"

echo ""
echo "Done. See baseline_*_model_weight_outputs.txt and baseline_*_model_metrics.txt under: $RUN_DIR"
