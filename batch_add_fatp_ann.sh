#!/bin/bash
# Batch script to add FATP-ANN baseline to experiments 011-016
# This ensures comprehensive baseline coverage across all major experiments

set -e  # Exit on error

cd "$(dirname "$0")"

echo "================================================================================"
echo "BATCH BASELINE SUPPLEMENTATION: Adding FATP-ANN to Experiments 011-016"
echo "================================================================================"
echo ""

# List of experiments to update
experiments=(
    "exp_011_scalability_analysis"
    "exp_012_worker_ratio_analysis"
    "exp_013_fairness_efficiency_tradeoff"
    "exp_014_ewma_tradeoff_exploration"
    "exp_015_temporal_ewma_validation"
    "exp_016_starvation_weight_interaction"
)

total=${#experiments[@]}
current=0

for exp in "${experiments[@]}"; do
    current=$((current + 1))
    
    echo "--------------------------------------------------------------------------------"
    echo "[$current/$total] Processing: $exp"
    echo "--------------------------------------------------------------------------------"
    
    # Check if experiment directory exists
    if [ ! -d "experiments_analysis/$exp" ]; then
        echo "⚠️  Skipping: Directory not found"
        echo ""
        continue
    fi
    
    # Run baseline supplementation
    ./venv/bin/python add_baselines_to_experiment.py \
        --exp "$exp" \
        --baselines fatp_ann
    
    echo "✅ Completed: $exp"
    echo ""
done

echo "================================================================================"
echo "✅ BATCH PROCESSING COMPLETE"
echo "================================================================================"
echo ""
echo "FATP-ANN baseline has been added to all applicable experiments."
echo ""
echo "Next steps:"
echo "  1. Review the updated CSV files in each experiment's data directory"
echo "  2. Re-run analysis notebooks to include FATP-ANN in comparisons"
echo "  3. Update figures and tables in your thesis"
echo ""

