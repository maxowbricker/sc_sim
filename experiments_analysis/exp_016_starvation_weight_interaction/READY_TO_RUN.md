# Experiment 016: Ready to Run Checklist

## Pre-Run Validation

### Data Availability
- [ ] Didi 3-hour peak dataset exists at `data/didi/gps_3hour_peak.txt`
- [ ] Didi 3-hour peak dataset exists at `data/didi/order_3hour_peak.txt`
- [ ] Dataset contains sufficient workers (>= 4,000) and tasks (>= 20,000)

### Dependencies
- [ ] Python 3.8+ installed
- [ ] Required packages installed:
  - numpy
  - pandas
  - scipy (for statistical functions)
- [ ] Virtual environment activated (if using)

### Code Readiness
- [ ] All strategy files exist:
  - `simulator/strategies/greedy.py`
  - `simulator/strategies/laf.py`
  - `simulator/strategies/ewma_only.py`
  - `simulator/strategies/composite.py`
- [ ] Strategies registered in `simulator/strategies/__init__.py`
- [ ] Core simulation module functional (`simulator/simulation.py`)

### Configuration Validation
- [ ] Experiment script is executable: `chmod +x run_experiment.py`
- [ ] Output directory exists: `data/`
- [ ] Sufficient disk space (~50 MB for output)

### Experiment Parameters
- [ ] Total simulations: 28 (3 baselines + 25 starvation sweep)
- [ ] Dataset: 4K workers / 20K tasks
- [ ] Robust parameters confirmed:
  - θ = 0.0 (soft threshold disabled)
  - γ = 0.5 (EWMA gamma)
  - normalize = True
  - k = 15 (nearest workers)

---

## Running the Experiment

### Start Experiment
```bash
# Navigate to experiment directory
cd experiments_analysis/exp_016_starvation_weight_interaction

# Run with output logging
python run_experiment.py 2>&1 | tee experiment_016_run.log

# Or run in background (for long runs)
nohup python run_experiment.py > experiment_016_run.log 2>&1 &
```

### Monitor Progress
```bash
# Watch log file in real-time
tail -f experiment_016_run.log

# Check how many simulations completed
grep "✅ Completed:" experiment_016_run.log | wc -l

# Check current status
ps aux | grep run_experiment.py
```

---

## During-Run Monitoring

### Expected Baseline Performance (First 3 Simulations)

**Greedy (Exp 001)**:
- TAR: ~92-93%
- JFI: ~0.70-0.76
- Mean Wait: ~2.5-2.8 min
- P95 Wait: ~9-11 min

**LAF (Exp 002)**:
- TAR: ~92-93%
- JFI: ~0.82-0.83
- Mean Wait: ~5.5-6.5 min
- P95 Wait: ~13-15 min

**EWMA-Only (Exp 003)**:
- TAR: ~91-92%
- JFI: ~0.80-0.84
- Mean Wait: ~7-9 min
- P95 Wait: ~14-16 min

### Expected Patterns in Starvation Sweep

**λ₂=0.0 (No Starvation Mitigation)**:
- Higher P95 wait times
- Potential for task starvation
- Performance similar to pure fairness+utility

**λ₂=0.5 (Default)**:
- Balanced performance
- Moderate P95 wait times
- Should match prior experiment results

**λ₂=1.0-2.0 (High Starvation Weight)**:
- Lower P95 wait times
- Possible slight efficiency decrease
- Diminishing returns expected

### Red Flags to Watch For
- ❌ Task assignment ratio < 85% (data quality issue)
- ❌ Simulation crashes or hangs
- ❌ Worker pool mutation warnings
- ❌ Extreme outliers in metrics (suggests bug)

---

## Post-Run Validation

### Completeness Check
- [ ] All 28 simulations completed (no failures)
- [ ] Output directory exists: `data/exp_016_YYYYMMDD_HHMMSS/`
- [ ] 28 individual JSON files created
- [ ] Aggregate CSV created: `data/experiment_016_aggregate_results.csv`

### Data Quality Check
```bash
# Count rows in aggregate CSV (should be 29: header + 28 data rows)
wc -l data/experiment_016_aggregate_results.csv

# Check for any missing values
grep -c "nan\|null\|None" data/experiment_016_aggregate_results.csv

# Verify all strategies present
cut -d',' -f3 data/experiment_016_aggregate_results.csv | sort | uniq
# Should show: greedy, laf, ewma_only, composite
```

### Performance Sanity Checks
- [ ] Greedy has lowest mean wait time among baselines
- [ ] LAF has highest JFI among baselines
- [ ] λ₂=0.0 configs show higher P95 wait than λ₂=0.5
- [ ] No extreme outliers (all metrics within reasonable ranges)

### File Size Check
- [ ] Each JSON file: 50-150 KB
- [ ] Aggregate CSV: 10-30 KB
- [ ] Total output: ~2-5 MB

---

## Analysis Preparation

### Next Steps After Validation
1. **Load and Inspect Data**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/experiment_016_aggregate_results.csv')
   print(df.head())
   print(df.describe())
   ```

2. **Create Analysis Notebook**:
   - Interaction heatmaps (JFI, Wait Time vs Config × λ₂)
   - Optimal λ₂ identification for each config
   - Starvation mitigation quantification
   - Comparison to Experiment 009 findings

3. **Key Questions to Answer**:
   - Is λ₂=0.5 optimal across all 5 configurations?
   - Are there interaction effects between λ₂ and (λ₁, λ₃)?
   - How much does starvation mitigation improve P95 wait times?
   - Is there a "safe range" for λ₂?

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'data.loader'"**
- Solution: Ensure you're running from project root or path is set correctly
- Check: `sys.path.insert(0, str(project_root))` in script

**Issue: "File not found: gps_3hour_peak.txt"**
- Solution: Verify data directory path
- Check: `data/didi/gps_3hour_peak.txt` exists
- Alternative: Use full dataset and adjust NUM_WORKERS/NUM_TASKS

**Issue: Simulation hangs or is very slow**
- Check: Dataset size (should be ~4K workers / ~20K tasks)
- Monitor: CPU and memory usage
- Estimated time: ~7 min per simulation

**Issue: Different results than Exp 014/015**
- Expected: Some variation due to sampling randomness
- Check: Same robust parameters (θ=0.0, γ=0.5, normalize=True)
- Verify: Worker gamma is properly set after deep copy

---

## Success Criteria

✅ **Experiment is successful if**:
1. All 28 simulations complete without errors
2. Baseline performance matches expected ranges
3. λ₂ sweep shows expected starvation mitigation patterns
4. Aggregate CSV contains all metrics for all experiments
5. No data quality issues (missing values, extreme outliers)

---

**After completing this checklist, Experiment 016 is ready for analysis!** 📊





