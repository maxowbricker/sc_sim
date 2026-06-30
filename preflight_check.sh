#!/bin/bash
# preflight_check.sh — run this BEFORE run_cluster.sh to verify the environment
# Usage: bash preflight_check.sh
# All checks must pass (green ✓) before launching experiments.

set -euo pipefail

PASS=0
FAIL=0
WARN=0

green()  { echo -e "\033[32m  ✓ $1\033[0m"; }
red()    { echo -e "\033[31m  ✗ $1\033[0m"; FAIL=$((FAIL+1)); }
yellow() { echo -e "\033[33m  ⚠ $1\033[0m"; WARN=$((WARN+1)); }

echo "=================================================="
echo "  Pre-flight Check — sc_sim cluster experiments"
echo "=================================================="

# ---------------------------------------------------------------------------
# 1. Core tools
# ---------------------------------------------------------------------------
echo ""
echo "── System tools ──"

if command -v tmux &>/dev/null; then
    green "tmux $(tmux -V | awk '{print $2}')"
else
    red "tmux not found — install with: sudo apt-get install tmux  OR  brew install tmux"
fi

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &>/dev/null; then
        VER=$($cmd --version 2>&1)
        PYTHON_CMD=$cmd
        green "$cmd — $VER"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    red "No python3 or python found in PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Python packages
# ---------------------------------------------------------------------------
echo ""
echo "── Python packages ──"

check_pkg() {
    PKG=$1
    IMPORT=${2:-$1}
    if $PYTHON_CMD -c "import $IMPORT" &>/dev/null; then
        VER=$($PYTHON_CMD -c "import $IMPORT; print(getattr($IMPORT, '__version__', 'ok'))" 2>/dev/null)
        green "$PKG ($VER)"
    else
        red "$PKG not found — pip install $PKG"
    fi
}

check_pkg numpy
check_pkg pandas
# scipy only needed for NYC taxi adapter — optional for Didi/Gowalla runs
if $PYTHON_CMD -c "import scipy" &>/dev/null; then
    green "scipy (optional, present)"
else
    yellow "scipy not found — only needed for NYC taxi adapter, not for these experiments"
fi

# ---------------------------------------------------------------------------
# 3. Project structure
# ---------------------------------------------------------------------------
echo ""
echo "── Project scripts ──"

check_script() {
    if [ -f "$1" ]; then
        green "$1"
    else
        red "MISSING: $1"
    fi
}

check_script "scripts/experiments/s53_scalability/run_scalability_fleet.py"
check_script "scripts/experiments/s53_scalability/run_scalability_tasks.py"
check_script "scripts/experiments/s52_main_results/run_gowalla_comparison.py"
check_script "scripts/experiments/s54_ablation/run_knlf_k_sweep.py"
check_script "scripts/experiments/s54_ablation/run_signal_comparison.py"
check_script "scripts/experiments/s54_ablation/run_fairness_weight_sweep.py"

# ---------------------------------------------------------------------------
# 4. Didi data
# ---------------------------------------------------------------------------
echo ""
echo "── Didi dataset ──"

DIDI_ROOT="data/didi/full_didi_gaia"
if [ -d "$DIDI_ROOT" ]; then
    DAY_COUNT=$(ls "$DIDI_ROOT" | wc -l | tr -d ' ')
    green "data/didi/full_didi_gaia/ exists ($DAY_COUNT day folders)"
    if ls "$DIDI_ROOT"/*20161109* &>/dev/null 2>&1; then
        green "Target day 20161109 present"
    else
        red "Target day 496528674@qq.com_20161109 not found — scalability and ablation scripts need this"
    fi
else
    red "data/didi/full_didi_gaia/ not found — Didi data missing"
fi

# ---------------------------------------------------------------------------
# 5. Gowalla data
# ---------------------------------------------------------------------------
echo ""
echo "── Gowalla dataset ──"

GOWALLA_ROOT="data/gowalla"
if [ -d "$GOWALLA_ROOT" ]; then
    green "data/gowalla/ exists"
    if ls "$GOWALLA_ROOT"/*.npz &>/dev/null 2>&1 || ls "$GOWALLA_ROOT"/*.gz &>/dev/null 2>&1; then
        green "Gowalla data files present (.npz / .gz)"
    else
        red "No .npz or .gz files in data/gowalla/ — run_gowalla_comparison.py will fail"
    fi
else
    red "data/gowalla/ not found"
fi

# ---------------------------------------------------------------------------
# 6. Quick smoke test — import the simulator (catches broken installs fast)
# ---------------------------------------------------------------------------
echo ""
echo "── Simulator smoke test ──"

if $PYTHON_CMD -c "
import sys; sys.path.insert(0, '.')
from config import create_composite_config
from simulator.simulation import EventSimulator
cfg = create_composite_config(assignment_strategy='greedy')
print('OK')
" 2>/dev/null | grep -q OK; then
    green "Simulator imports cleanly"
else
    red "Simulator import failed — check Python path and dependencies"
    $PYTHON_CMD -c "
import sys; sys.path.insert(0, '.')
from config import create_composite_config
from simulator.simulation import EventSimulator
" 2>&1 | head -5
fi

# ---------------------------------------------------------------------------
# 7. Results directories writable
# ---------------------------------------------------------------------------
echo ""
echo "── Results directories ──"

for DIR in results/s53_scalability results/s54_ablation results/s1_overall_performance results; do
    mkdir -p "$DIR" && green "$DIR (created/exists, writable)" || red "$DIR not writable"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
if [ $FAIL -eq 0 ]; then
    echo -e "\033[32m  ALL CHECKS PASSED ($WARN warning(s))\033[0m"
    echo "  Ready to run: bash run_cluster.sh"
else
    echo -e "\033[31m  $FAIL CHECK(S) FAILED — fix before running experiments\033[0m"
    [ $WARN -gt 0 ] && echo "  $WARN warning(s) (non-blocking)"
fi
echo "=================================================="
exit $FAIL
