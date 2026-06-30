#!/bin/bash
# run_rerun.sh — re-run only the experiments whose results are stale after the
# strategy optimisation pass (2026-07-01).
#
# WHAT CHANGED (and why these experiments need new numbers):
#   - greedy:    NEW_TASK now uses spatial index (k=10); runtime ~19s not ~357s
#   - knlf:      FREE_WORKER now uses deferred_task_index (not full linear scan)
#   - laf:       distance function swapped to fast_manhattan_km (tiny numerical delta)
#   - fatp_ann:  same distance swap + pickup_dist no longer triple-computed
#
# WHAT IS SAFE (pure redundancy removal, numerically identical):
#   - biranking, onrta_rt, discrete_review_lp — already used fast_manhattan_km
#   - composite — untouched
#   - fairness_weight_sweep — Composite-only, unaffected → NOT re-run here
#
# NOT IN THIS SCRIPT (manual action required):
#   §5.3 fleet + task scalability runs are mid-flight on the cluster with the
#   old code. Once these finish, discard those results and restart the cluster
#   scripts from scratch with the updated codebase.
#
# OUTPUT FILES: all write to _v2 paths so existing paper-referenced CSVs are
# preserved until the new numbers are verified and the tex tables updated.
#
# Prerequisites:
#   1. conda activate sc   (or your equivalent)
#   2. bash preflight_check.sh
#   3. Run from project root: bash run_rerun.sh
#
# Monitor:
#   tmux attach -t rerun
#   Ctrl+b w   — list windows / jump
#   Ctrl+b d   — detach (scripts keep running)

set -euo pipefail

SESSION="rerun"
RESULTS_52="results/s52_main_results"
RESULTS_53="results/s53_scalability"
RESULTS_54="results/s54_ablation"

tmux kill-session -t $SESSION 2>/dev/null || true

mkdir -p "$RESULTS_52" "$RESULTS_53" "$RESULTS_54"

echo "Launching stale-result re-runs in tmux session '$SESSION'..."

# ---------------------------------------------------------------------------
# Window 1 — §5.2 DiDi main results
# Estimated: ~1–1.5 h (FATP-ANN and LP are now faster due to optimisations,
# but still the heaviest window).
# Includes Greedy (new k=10 numbers), k-NLF, LAF, FATP-ANN.
# BiRanking / ONRTA-RT / Disc. Review LP included so the full table is
# regenerated in one go — their numbers will be identical to before.
# ---------------------------------------------------------------------------
tmux new-session -d -s $SESSION -n didi
tmux send-keys -t $SESSION:didi \
  "python3 scripts/experiments/s52_main_results/run_strategy_comparison.py \
    --day 20161109 \
    --output ${RESULTS_52}/didi_20161109_v2.csv \
    2>&1 | tee ${RESULTS_52}/log_didi_v2.log; echo '=== didi DONE ==='" \
  Enter

# ---------------------------------------------------------------------------
# Window 2 — §5.2 Gowalla main results
# Estimated: ~25–40 min (smaller dataset; FATP-ANN faster with optimisations).
# Greedy, k-NLF, LAF, FATP-ANN all produce slightly different numbers.
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n gowalla
tmux send-keys -t $SESSION:gowalla \
  "python3 scripts/experiments/s52_main_results/run_gowalla_comparison.py \
    --compression compressed \
    --ratio 0.2 \
    --output ${RESULTS_52}/gowalla_austin_compressed_v2.csv \
    2>&1 | tee ${RESULTS_52}/log_gowalla_v2.log; echo '=== gowalla DONE ==='" \
  Enter

# ---------------------------------------------------------------------------
# Window 3 — §5.3 Fleet scalability (vary |W|, fix |T|)
# Config: TARGET_TASKS=50k (not full 224k), 6 fleet sizes, 900s timeout/run.
# Estimated: ~1–1.5 h. Greedy (now k=10 spatial index) and LAF (now
# fast_manhattan_km) are both faster than the old cluster run. FATP-ANN will
# still timeout at larger fleet sizes — expected and reportable.
# Fast strategies (k-NLF, Composite, Greedy) write first; kill anytime safely.
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n fleet
tmux send-keys -t $SESSION:fleet \
  "python3 scripts/experiments/s53_scalability/run_scalability_fleet.py \
    --output ${RESULTS_53}/scalability_fleet_v2.csv \
    2>&1 | tee ${RESULTS_53}/log_fleet_v2.log; echo '=== fleet DONE ==='" \
  Enter

# ---------------------------------------------------------------------------
# Window 4 — §5.3 Task scalability (vary |T|, fix |W|)
# Config: fixed 10k workers, 5 task volumes (50k–224k), 900s timeout/run.
# Estimated: ~45 min–1 h. FATP-ANN will timeout at ≥100k tasks — expected.
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n tasks
tmux send-keys -t $SESSION:tasks \
  "python3 scripts/experiments/s53_scalability/run_scalability_tasks.py \
    --output ${RESULTS_53}/scalability_tasks_v2.csv \
    2>&1 | tee ${RESULTS_53}/log_tasks_v2.log; echo '=== tasks DONE ==='" \
  Enter

# ---------------------------------------------------------------------------
# Window 5 — §5.4.1 k-NLF k-sweep
# Estimated: ~10–15 min (k-NLF FREE_WORKER changed; Greedy anchor will now
# show ~19s not ~357s).
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n knlf
tmux send-keys -t $SESSION:knlf \
  "python3 scripts/experiments/s54_ablation/run_knlf_k_sweep.py \
    --output ${RESULTS_54}/knlf_k_sweep_20161109_v2.csv \
    2>&1 | tee ${RESULTS_54}/log_knlf_k_sweep_v2.log; echo '=== knlf DONE ==='" \
  Enter

# ---------------------------------------------------------------------------
# Window 6 — §5.4.2 Signal comparison
# Estimated: ~12–15 min (Greedy baseline + k-NLF both changed; Composite
# and k-NTF variants unaffected).
# Existing v2 CSV is superseded by v3 here.
# ---------------------------------------------------------------------------
tmux new-window -t $SESSION -n signal
tmux send-keys -t $SESSION:signal \
  "python3 scripts/experiments/s54_ablation/run_signal_comparison.py \
    --output ${RESULTS_54}/signal_comparison_20161109_v3.csv \
    2>&1 | tee ${RESULTS_54}/log_signal_comparison_v3.log; echo '=== signal DONE ==='" \
  Enter

# Land on the heaviest window
tmux select-window -t $SESSION:didi

echo ""
echo "  6 windows launched:"
echo "    didi   — §5.2 DiDi strategy comparison      (~1–1.5 h)"
echo "    gowalla— §5.2 Gowalla comparison             (~25–40 min)"
echo "    fleet  — §5.3 Fleet scalability (vary |W|)   (~3–4 h)"
echo "    tasks  — §5.3 Task scalability (vary |T|)    (~1.5–2 h)"
echo "    knlf   — §5.4.1 k-NLF k-sweep               (~10–15 min)"
echo "    signal — §5.4.2 Signal comparison            (~12–15 min)"
echo ""
echo "  Output files (v2/v3 — original CSVs preserved):"
echo "    ${RESULTS_52}/didi_20161109_v2.csv"
echo "    ${RESULTS_52}/gowalla_austin_compressed_v2.csv"
echo "    ${RESULTS_53}/scalability_fleet_v2.csv"
echo "    ${RESULTS_53}/scalability_tasks_v2.csv"
echo "    ${RESULTS_54}/knlf_k_sweep_20161109_v2.csv"
echo "    ${RESULTS_54}/signal_comparison_20161109_v3.csv"
echo ""
echo "  NOTE: §5.4.3 fairness weight sweep is Composite-only — unaffected, skip."
echo ""
echo "  Navigate: Ctrl+b w (list)  Ctrl+b n/p (next/prev)  Ctrl+b d (detach)"
echo "  Return:   tmux attach -t $SESSION"
echo ""

tmux attach -t $SESSION
