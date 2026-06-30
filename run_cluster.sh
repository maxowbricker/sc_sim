#!/bin/bash
# run_cluster.sh — launch all paper experiments in parallel tmux windows
#
# Prerequisites:
#   1. Activate your Python environment (conda activate sc  OR  source venv/bin/activate)
#   2. Run preflight check:  bash preflight_check.sh
#   3. Run this script from the project root:  bash run_cluster.sh
#
# Monitor:
#   tmux attach -t exp          — reconnect after detaching / SSH reconnect
#   Ctrl+b w                    — see all windows with preview, jump to any
#   Ctrl+b n / p                — next / previous window
#   Ctrl+b d                    — detach (scripts KEEP RUNNING in background)
#
# !! Always use Ctrl+b d to detach — closing the SSH terminal kills all scripts !!

set -euo pipefail

SESSION="exp"

# Kill any stale session from a previous run
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure output directories exist
mkdir -p results/s53_scalability results/s54_ablation results/s1_overall_performance results

echo "Starting experiments in tmux session '$SESSION'..."

# ---------------------------------------------------------------------------
# Window 1 — Fleet scalability (~3–4 h, dominates wall time)
# ---------------------------------------------------------------------------
tmux new-session -d -s $SESSION -n fleet
tmux send-keys -t $SESSION:fleet \
  'python scripts/experiments/s53_scalability/run_scalability_fleet.py 2>&1 | tee results/s53_scalability/log_fleet.log; echo "=== fleet DONE ==="' \
  Enter

# ---------------------------------------------------------------------------
# Window 2 — Task scalability (~1.5–2 h)
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n tasks
tmux send-keys -t $SESSION:tasks \
  'python scripts/experiments/s53_scalability/run_scalability_tasks.py 2>&1 | tee results/s53_scalability/log_tasks.log; echo "=== tasks DONE ==="' \
  Enter

# ---------------------------------------------------------------------------
# Window 3 — Gowalla main results (~70 min)
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n gowalla
tmux send-keys -t $SESSION:gowalla \
  'python scripts/experiments/s52_main_results/run_gowalla_comparison.py --compression compressed --ratio 0.2 --output results/s1_overall_performance/gowalla_austin_compressed.csv 2>&1 | tee results/log_gowalla.log; echo "=== gowalla DONE ==="' \
  Enter

# ---------------------------------------------------------------------------
# Window 4 — k-NLF k sweep (~10 min)
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n knlf
tmux send-keys -t $SESSION:knlf \
  'python scripts/experiments/s54_ablation/run_knlf_k_sweep.py 2>&1 | tee results/s54_ablation/log_knlf_k_sweep.log; echo "=== knlf DONE ==="' \
  Enter

# ---------------------------------------------------------------------------
# Window 5 — Signal comparison (~12 min)
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n signal
tmux send-keys -t $SESSION:signal \
  'python scripts/experiments/s54_ablation/run_signal_comparison.py 2>&1 | tee results/s54_ablation/log_signal_comparison.log; echo "=== signal DONE ==="' \
  Enter

# ---------------------------------------------------------------------------
# Window 6 — Fairness weight sweep (~22 min)
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n fw
tmux send-keys -t $SESSION:fw \
  'python scripts/experiments/s54_ablation/run_fairness_weight_sweep.py 2>&1 | tee results/s54_ablation/log_fairness_weight_sweep.log; echo "=== fw_sweep DONE ==="' \
  Enter

# Land on the fleet window (longest — the one to watch)
tmux select-window -t $SESSION:fleet

echo ""
echo "All 6 windows launched. Attaching now..."
echo ""
echo "  Windows:  fleet | tasks | gowalla | knlf | signal | fw"
echo "  Navigate: Ctrl+b w (list)  Ctrl+b n/p (next/prev)"
echo "  Detach:   Ctrl+b d  (scripts keep running)"
echo "  Return:   tmux attach -t $SESSION"
echo ""

tmux attach -t $SESSION
