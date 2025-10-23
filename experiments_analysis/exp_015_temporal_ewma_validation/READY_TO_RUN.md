# Experiment 015: Pre-Execution Checklist

**Experiment**: EWMA Temporal & Baseline Validation  
**Total Simulations**: 33  
**Estimated Runtime**: ~3.85 hours

---

## ✅ Pre-Flight Checklist

### 1. Code Implementation

- [x] **Random strategy implemented**: `simulator/strategies/random_assign.py`
- [x] **Random strategy registered**: Added to `__init__.py` and `config.py`
- [x] **Temporal logging added**: Modified `simulator/simulation.py`
  - [x] Initialization variables added (lines ~70-73)
  - [x] Checkpoint logging in event loop (lines ~178-190)
  - [x] Summary fields added (lines ~366-369)

### 2. Test Random Strategy

Before running full experiment, test the random strategy:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim
venv/bin/python -c "
from simulator.strategies import get_strategy
strategy = get_strategy('random_assign')
print('✅ Random strategy loaded successfully')
print(f'Strategy handlers: {list(strategy().keys())}')
"
```

Expected output:
```
✅ Random strategy loaded successfully
Strategy handlers: ['NEW_TASK', 'FREE_WORKER']
```

### 3. Test Temporal Logging

Run a quick test simulation to verify temporal data is captured:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim
venv/bin/python -c "
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
import pandas as pd

# Load small sample
all_workers, all_tasks = load_workers_tasks(dataset='didi')
workers = all_workers[:100]
tasks = all_tasks[:500]

# Run quick sim
config = {'assignment_strategy': 'greedy', 'strategy_params': {}}
summary = run_simulation(workers, tasks, sim_config=config)

# Check temporal data
if 'ewma_temporal_history' in summary:
    print(f'✅ Temporal logging working!')
    print(f'   Snapshots captured: {len(summary[\"ewma_temporal_history\"])}')
    print(f'   Sample: {summary[\"ewma_temporal_history\"][0]}')
else:
    print('❌ No temporal data found')
"
```

### 4. System Requirements

- [x] **Python environment**: Virtual environment activated
- [ ] **Disk space**: Check ~7 MB available for output
  ```bash
  df -h .
  ```
- [ ] **Memory**: Ensure ~4-6 GB RAM available
  ```bash
  free -h  # Linux
  vm_stat  # macOS
  ```

### 5. Experiment Configuration

- [x] **Output directory exists**: `data/` folder created
- [x] **Run script exists**: `run_experiment.py` in place
- [x] **Setup documented**: `setup.md` complete

### 6. Backup Previous Data

If running multiple times:

```bash
cd experiments_analysis/exp_015_temporal_ewma_validation/data
# Backup previous run if exists
if [ -f experiment_015_aggregate_results.csv ]; then
    mv experiment_015_aggregate_results.csv experiment_015_aggregate_results_backup_$(date +%Y%m%d_%H%M%S).csv
    echo "✅ Previous data backed up"
fi
```

### 7. Runtime Planning

**Start Time Considerations**:
- [ ] Runtime: ~3.85 hours
- [ ] Will laptop be available for this duration?
- [ ] Running in background recommended:
  ```bash
  nohup ../../venv/bin/python -u run_experiment.py > experiment_015_run.log 2>&1 &
  ```

---

## 🚀 Ready to Run

Once all checks pass, execute:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_015_temporal_ewma_validation

# Option 1: Foreground (monitor in terminal)
../../venv/bin/python run_experiment.py

# Option 2: Background (recommended)
nohup ../../venv/bin/python -u run_experiment.py > experiment_015_run.log 2>&1 &

# Get process ID
echo $! > experiment_015.pid
```

---

## 📊 Monitoring Progress

### Check Status

```bash
# View live output (if running in background)
tail -f experiment_015_run.log

# Count completed simulations
ls data/exp_015_*/exp_*.json 2>/dev/null | wc -l

# Check process still running
ps aux | grep run_experiment.py | grep -v grep
```

### Expected Progress

| Time | Completed | Remaining | Group |
|------|-----------|-----------|-------|
| 0:00 | 0/33 | 3.85h | Starting |
| 0:21 | 3/33 | 3.5h | Baselines done |
| 0:28 | 4/33 | 3.4h | EWMA-Only done |
| 3:03 | 29/33 | 0.5h | Pareto done |
| 3:51 | 33/33 | 0:00 | Complete! |

---

## 🔍 Post-Run Validation

After completion, verify:

```bash
# Check all simulations completed
cd data
ls exp_015_*/exp_*.json | wc -l  # Should be 33

# Check aggregate CSV
wc -l experiment_015_aggregate_results.csv  # Should be 34 (33 + header)

# Verify temporal data in random sample
python3 -c "
import json
import glob

files = glob.glob('exp_015_*/exp_*.json')[:5]
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        has_temporal = 'ewma_temporal_history' in data
        num_snapshots = len(data.get('ewma_temporal_history', []))
        print(f'{f}: Temporal={has_temporal}, Snapshots={num_snapshots}')
"
```

Expected:
```
✅ All files have temporal data
✅ 390-400 snapshots per simulation
```

---

## ⚠️ Troubleshooting

### Issue: Random strategy not found
**Fix**:
```bash
# Verify registration
grep "random_assign" simulator/strategies/__init__.py
grep "random_assign" config.py
```

### Issue: No temporal data in output
**Check**: 
1. Lines ~70-73 in `simulator/simulation.py` (initialization)
2. Lines ~178-190 (checkpoint logging)
3. Lines ~366-369 (summary fields)

### Issue: Out of memory
**Fix**: Close other applications, or reduce worker/task count for testing

### Issue: Process killed unexpectedly
**Check**:
```bash
dmesg | tail  # Linux
# Check system logs for OOM killer
```

---

## 📋 Expected Output Structure

```
experiments_analysis/exp_015_temporal_ewma_validation/
├── setup.md
├── README.md
├── run_experiment.py
├── READY_TO_RUN.md (this file)
└── data/
    ├── README.md
    ├── experiment_015_aggregate_results.csv  ← 33 rows
    └── exp_015_20251023_HHMMSS/
        ├── exp_001_Greedy_Baseline_summary.json
        ├── exp_002_LAF_Baseline_summary.json
        ├── exp_003_Random_Baseline_summary.json  ← NEW
        ├── exp_004_EWMA_Only_G_0.5_summary.json
        ├── exp_005_Pareto_L1_2.5_L3_0.5_summary.json
        ...
        └── exp_033_Gamma_Balanced_G_0.9_summary.json
```

---

## 🎯 Success Criteria

- [x] **Pre-flight**: All checks passed
- [ ] **Execution**: 33/33 simulations completed
- [ ] **Data Quality**: All JSONs have temporal data
- [ ] **Performance**: TAR >90% for all strategies
- [ ] **Temporal**: ~400 snapshots per simulation
- [ ] **Random**: Random baseline performs as expected

---

**Ready**: If all pre-flight checks pass ✅  
**Start Command**: See "🚀 Ready to Run" section above

Good luck! 🚀

