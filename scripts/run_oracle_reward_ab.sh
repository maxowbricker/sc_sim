#!/usr/bin/env bash
# Oracle reward A/B — same protocol as run_20260522_135121 (25k steps, best_hyperparameters.json).
#
# Arm A: Δ count-JFI vs greedy twin (baseline oracle, run_20260522)
# Arm B: Δ JFI_rate vs greedy twin (supply-aware fairness signal)
#
# Do NOT resume from run_20260522 when switching reward arms — train fresh for a clean A/B.
#
# Usage (from repo root, on oracle-approach branch):
#   bash scripts/run_oracle_reward_ab.sh           # both arms sequentially
#   bash scripts/run_oracle_reward_ab.sh A         # arm A only
#   bash scripts/run_oracle_reward_ab.sh B         # arm B only

set -euo pipefail
cd "$(dirname "$0")/.."

TIMESTEPS="${TIMESTEPS:-25000}"
HYPER="${HYPER:-best_hyperparameters.json}"
EXTRA_ARGS=(--hyperparams "$HYPER")

run_arm() {
  local mode="$1"
  local label="$2"
  echo ""
  echo "================================================================================"
  echo " Oracle A/B — Arm ${label}: --reward-mode ${mode}"
  echo "================================================================================"
  python rl/train_sb3.py --timesteps "$TIMESTEPS" --reward-mode "$mode" "${EXTRA_ARGS[@]}"
}

ARM="${1:-both}"

case "$ARM" in
  A|a|oracle|oracle_delta_jfi)
    run_arm oracle_delta_jfi "A (count-JFI, run_20260522 baseline)"
    ;;
  B|b|rate|oracle_delta_jfi_rate)
    run_arm oracle_delta_jfi_rate "B (JFI_rate)"
    ;;
  both|*)
    run_arm oracle_delta_jfi "A (count-JFI, run_20260522 baseline)"
    run_arm oracle_delta_jfi_rate "B (JFI_rate)"
    ;;
esac

echo ""
echo "Compare TensorBoard:"
echo "  tensorboard --logdir rl_logs_sb3 --logdir_spec \\"
echo "    A_count_jfi:rl_logs_sb3/run_<A_timestamp>,B_jfi_rate:rl_logs_sb3/run_<B_timestamp>"
