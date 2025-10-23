# Experiment 014: Ready to Execute

✅ **Status**: All setup complete, ready to run!

## Quick Start

```bash
cd experiments_analysis/exp_014_ewma_tradeoff_exploration
python run_experiment.py
```

Or run in background with logging:

```bash
cd experiments_analysis/exp_014_ewma_tradeoff_exploration
nohup python -u ../../venv/bin/python run_experiment.py > experiment_014_run.log 2>&1 &
```

Monitor progress:
```bash
tail -f experiment_014_run.log
```

## What Will Happen

1. **Data Loading**: Loads 3-hour peak Didi dataset (~2 minutes)
2. **Stratified Sampling**: Samples 4K workers, 20K tasks (~30 seconds)
3. **43 Simulations** (~5 hours total):
   - Exp 001-003: Baselines (Greedy, LAF, EWMA-Only)
   - Exp 004-028: Pareto sweep (λ₁ vs λ₃)
   - Exp 029-043: Gamma sensitivity (5 γ × 3 configs)
4. **Results Saved**:
   - `data/experiment_014_aggregate_results.csv` (master file)
   - `data/exp_014_<timestamp>/<individual_jsons>` (detailed results)

## Expected Output

```
================================================================================
Experiment 014: EWMA & Trade-Off Exploration
================================================================================
Start Time: 2025-10-23 12:30:00

📁 Output directory: exp_014_20251023_123000

================================================================================
DATA LOADING AND SAMPLING
================================================================================

📂 Loading didi dataset...
📊 Loading GPS data from: gps_3hour_peak.txt (695.9 MB)
📊 Loading Orders data from: order_3hour_peak.txt (4.1 MB)
📊 Sampling 4,000 workers and 20,000 tasks...
...
✅ Loaded 4,000 workers and 20,000 tasks

📋 Total experiments: 43
   Group 1 (Baselines): 3 simulations
   Group 2 (Pareto Sweep): 25 simulations
   Group 3 (Gamma Sensitivity): 15 simulations

⏱️  Estimated total runtime: ~301 minutes (~5.0 hours)

================================================================================
RUNNING SIMULATIONS
================================================================================

🔄 Experiment 001/043 - Greedy_Baseline
   Strategy: greedy
   ✅ Completed: 18,675/20,000 tasks (TAR: 93.4%)
   JFI: 0.704 | Gini: 0.370 | Wait: 2.24 min
   Runtime: 412.3s
   💾 Saved: exp_001_Greedy_Baseline_summary.json

   Progress: 1/43 | ✅ 1 | ❌ 0 | Elapsed: 7.0m | Est. Remaining: 294.0m

🔄 Experiment 002/043 - LAF_Baseline
   ...
```

## Expected Results

### Baseline Comparison
- **Greedy**: TAR ~93%, JFI ~0.70, Wait ~2.2 min (efficiency baseline)
- **LAF**: TAR ~93%, JFI ~0.92, Wait ~5.1 min (simple fairness)
- **EWMA-Only**: TAR ~93%, JFI ~0.88, Wait ~9.0 min (advanced fairness)

### Pareto Sweep
- **High λ₁, Low λ₃**: JFI ~0.95, Wait ~6-8 min (fairness-optimized)
- **Low λ₁, High λ₃**: JFI ~0.85, Wait ~3-4 min (efficiency-optimized)
- **Balanced**: JFI ~0.92, Wait ~4-5 min (optimal trade-off)

### Gamma Sensitivity
- All γ values should produce similar final JFI (±0.03)
- Confirms EWMA robustness to parameter choice

## After Completion

1. Check the log for any failed simulations
2. Verify CSV has 43 rows:
   ```bash
   wc -l data/experiment_014_aggregate_results.csv
   # Should show: 44 (43 data + 1 header)
   ```
3. Run analysis:
   ```bash
   jupyter notebook analysis.ipynb
   ```

## Troubleshooting

**If simulation crashes:**
- Check available memory (needs ~1-2GB per simulation)
- Check disk space (results ~500MB)
- Review log file for error messages

**If results look wrong:**
- Verify 15-minute expiry is active (check log for "expire_time")
- Check TAR values (should be >85% for all simulations)
- Verify deep copy is working (no mutation warnings)

## Validation Checklist

After completion, verify:
- [ ] All 43 simulations completed successfully
- [ ] CSV file exists with 43 rows
- [ ] All TAR values > 85%
- [ ] Greedy has lowest JFI (~0.70)
- [ ] LAF has highest JFI (~0.92)
- [ ] No NaN values in key metrics
- [ ] Gamma variations show minimal JFI change (<5%)

## Time Estimates

- **Data loading**: ~2 minutes
- **Per simulation**: ~7 minutes average
- **Total**: ~5 hours (301 minutes)
- **With overhead**: ~5.5 hours realistic

Start time: `<now>`  
Expected completion: `<now + 5.5 hours>`

---

**Ready to execute!** 🚀

Run the command above and monitor progress with `tail -f experiment_014_run.log`


