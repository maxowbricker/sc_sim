# Spatial Crowdsourcing Simulation Guide

## 🚀 Quick Start

This guide covers how to run different types of simulations in the spatial crowdsourcing research codebase.

## 📋 Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure you're in the project root
cd /path/to/sc_sim
```

## 🎯 Simulation Types

### 1. Local Development Simulation

**Use Case**: Quick testing, development, debugging  
**Dataset**: Small synthetic or manual data  
**Duration**: Seconds to minutes  

```python
import pandas as pd
from datetime import datetime, timedelta
from simulator.simulation import Simulation

# Create small test dataset
base_time = datetime.now()

workers_df = pd.DataFrame({
    'worker_id': [f'worker_{i}' for i in range(10)],
    'start_lat': [37.7749 + i*0.01 for i in range(10)],
    'start_lon': [-122.4194 + i*0.01 for i in range(10)],
    'release_time': [base_time for _ in range(10)],
    'deadline': [base_time + timedelta(hours=8) for _ in range(10)]
})

tasks_df = pd.DataFrame({
    'task_id': [f'task_{i}' for i in range(20)],
    'pickup_lat': [37.7749 + i*0.005 for i in range(20)],
    'pickup_lon': [-122.4194 + i*0.005 for i in range(20)],
    'dropoff_lat': [37.7849 + i*0.005 for i in range(20)],
    'dropoff_lon': [-122.4094 + i*0.005 for i in range(20)],
    'release_time': [base_time + timedelta(minutes=i*5) for i in range(20)],
    'expire_time': [base_time + timedelta(hours=2, minutes=i*5) for i in range(20)]
})

# Basic greedy config
config = {
    'assignment_strategy': 'greedy',
    'strategy_params': {}
}

# Run simulation
sim = Simulation(config, workers_df, tasks_df)
results = sim.run()

print(f"Task Assignment Ratio: {results['task_assignment_ratio']*100:.1f}%")
print(f"Jain's Fairness Index: {results['jfi']:.3f}")
```

### 2. Google Colab Full-Scale Experiments

**Use Case**: Research experiments with large datasets  
**Dataset**: Didi Gaia (38K workers, 220K tasks)  
**Duration**: Minutes to hours  

#### 2.1 Setup Colab Environment

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone repository
!git clone https://github.com/yourusername/sc_sim.git
%cd sc_sim

# Cell 3: Install dependencies (if needed)
!pip install pandas numpy

# Cell 4: Load optimized data loading functions
exec(open('notebook_optimized_loader.py').read())
```

#### 2.2 Load Full Dataset

```python
# Cell 5: Load Didi dataset (cached)
print("📊 Loading full Didi dataset...")
workers_df, tasks_df = load_data('didi')  # Uses cached loading
print(f"✅ Loaded {len(workers_df):,} workers, {len(tasks_df):,} tasks")
```

#### 2.3 Run Single Experiment

```python
# Cell 6: Single simulation test
config = create_composite_config(
    λ1=1.0,  # Fairness weight
    λ2=1.0,  # Starvation weight
    λ3=1.0,  # Utility weight
    soft_threshold=0.5,
    assignment_strategy="composite"
)

sim = Simulation(config, workers_df, tasks_df)
results = sim.run()

print(f"JFI: {results['jfi']:.3f}")
print(f"TAR: {results['task_assignment_ratio']*100:.1f}%")
```

### 3. Parameter Sensitivity Analysis

**Use Case**: Research Question 1 - Testing different λ values  
**Dataset**: Full or subset  
**Duration**: Hours (multiple runs)  

```python
# Test different λ2 (starvation) values
λ2_values = [0.1, 0.5, 1.0, 2.0, 5.0]
results = []

for λ2 in λ2_values:
    print(f"🔧 Testing λ2={λ2}")
    
    config = create_composite_config(
        λ1=1.0,
        λ2=λ2,
        λ3=1.0,
        soft_threshold=0.5,
        assignment_strategy="composite"
    )
    
    sim = Simulation(config, workers_df, tasks_df)
    result = sim.run()
    
    results.append({
        'λ2': λ2,
        'jfi': result['jfi'],
        'tar': result['task_assignment_ratio'],
        'wait_time': result['avg_wait_time_minutes']
    })
    
    print(f"   JFI: {result['jfi']:.3f}, TAR: {result['task_assignment_ratio']*100:.1f}%")

# Analyze results
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)
```

### 4. Strategy Comparison

**Use Case**: Comparing different assignment strategies  
**Dataset**: Medium to full  
**Duration**: Variable  

```python
strategies = ['greedy', 'fatp', 'composite']
results = {}

for strategy in strategies:
    print(f"🚀 Testing {strategy} strategy...")
    
    if strategy == 'composite':
        config = create_composite_config(
            λ1=1.0, λ2=1.0, λ3=1.0,
            soft_threshold=0.5,
            assignment_strategy=strategy
        )
    else:
        config = {
            'assignment_strategy': strategy,
            'strategy_params': {}
        }
    
    sim = Simulation(config, workers_df, tasks_df)
    result = sim.run()
    
    results[strategy] = {
        'jfi': result['jfi'],
        'tar': result['task_assignment_ratio'],
        'wait_time': result['avg_wait_time_minutes']
    }

# Compare results
for strategy, metrics in results.items():
    print(f"{strategy:10} | JFI: {metrics['jfi']:.3f} | TAR: {metrics['tar']*100:.1f}% | Wait: {metrics['wait_time']:.1f}min")
```

## 🛠️ Configuration Options

### Assignment Strategies

1. **`greedy`**: Nearest available worker assignment
2. **`fatp`**: First Available Time Priority
3. **`composite`**: Research contribution with fairness weighting

### Composite Strategy Parameters

```python
config = create_composite_config(
    λ1=1.0,        # Fairness weight (EWMA)
    λ2=1.0,        # Starvation weight (idle time)
    λ3=1.0,        # Utility weight (distance)
    soft_threshold=0.5,  # Fairness threshold
    assignment_strategy="composite"
)
```

### Strategy Parameters

```python
config = {
    'assignment_strategy': 'composite',
    'strategy_params': {
        'gamma': 0.3,           # EWMA decay factor
        'soft_threshold': 0.5,  # Fairness threshold
        'λ1': 1.0,             # Fairness weight
        'λ2': 1.0,             # Starvation weight  
        'λ3': 1.0              # Utility weight
    }
}
```

## 📊 Dataset Options

### 1. Manual Test Data (Development)
- **Size**: 10-100 workers, 20-200 tasks
- **Use**: Quick testing, debugging
- **Load Time**: Instant

### 2. Checkin Data (Medium Scale)
- **Size**: Hundreds of workers/tasks
- **Use**: Algorithm validation
- **Load Time**: Seconds
- **Setup**: Run `scripts/CheckinSynthesiser.py` first

### 3. Didi Gaia (Full Scale)
- **Size**: 38,651 workers, 220,139 tasks
- **Use**: Research experiments
- **Load Time**: 30 seconds (with optimizations)
- **Memory**: ~3GB RAM required

## ⚡ Performance Optimizations

### 1. Use Cached Loading
```python
# First load (slow)
workers_df, tasks_df = load_data('didi')

# Subsequent loads (fast - uses cache)
workers_df, tasks_df = load_data('didi')
```

### 2. Progress Indicators
The fixed `Simulation` class now shows progress during object conversion:
```
🚀 Converting DataFrames to objects...
   📊 Converting workers: 5,000/38,651 (12.9%)
   📊 Converting workers: 10,000/38,651 (25.9%)
   ...
   ✅ Converted 38,651 workers
```

### 3. Subset Testing
For development, use smaller datasets:
```python
# Use subset for testing
test_workers = workers_df.head(1000)
test_tasks = tasks_df.head(2000)
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Use Google Colab High-RAM runtime
   - Process data in smaller chunks
   - Clear variables between experiments: `del workers_df, tasks_df`

2. **Timeout Issues**
   - Progress indicators prevent Colab timeouts
   - Use Colab Pro for longer runtimes
   - Break large experiments into smaller cells

3. **Import Errors**
   - Ensure you're in the project root directory
   - Activate virtual environment: `source venv/bin/activate`
   - Check Python path: `import sys; sys.path.append('.')`

### Performance Issues

1. **Slow Object Conversion**
   - **Fixed**: Removed 4x duplicate EWMA calculations
   - **Fixed**: Added progress indicators
   - Use `UltraFastSimulation` for maximum speed

2. **Silent Hangs**
   - **Fixed**: Progress indicators show exactly where the process is
   - Monitor memory usage in Colab

## 📈 Expected Performance

### Local Development
- **Small dataset (10/20)**: 0.02 seconds
- **Medium dataset (100/200)**: 0.1-0.5 seconds

### Google Colab
- **Data loading**: 30 seconds (first time), instant (cached)
- **Object conversion**: 1-3 minutes (38K workers + 220K tasks)
- **Single simulation**: 5-15 minutes (depending on complexity)
- **Full experiment (5 runs)**: 30-60 minutes

## 🎯 Research Experiment Templates

### RQ1: Parameter Sensitivity
```python
# Test λ2 sensitivity (starvation weight)
λ2_values = [0.1, 0.5, 1.0, 2.0, 5.0]
results = run_parameter_sweep('λ2', λ2_values, workers_df, tasks_df)
```

### RQ2: Strategy Comparison
```python
# Compare all strategies
strategies = ['greedy', 'fatp', 'composite']
results = run_strategy_comparison(strategies, workers_df, tasks_df)
```

### RQ3: Scalability Analysis
```python
# Test different dataset sizes
sizes = [(1000, 2000), (5000, 10000), (10000, 20000)]
results = run_scalability_test(sizes, workers_df, tasks_df)
```

## 💡 Best Practices

1. **Start Small**: Test with small datasets first
2. **Use Progress Indicators**: Monitor long-running experiments
3. **Cache Data**: Load once, use many times
4. **Save Results**: Export results to JSON/CSV for analysis
5. **Monitor Resources**: Check RAM usage in Colab
6. **Version Control**: Commit working configurations

## 🚀 Quick Commands

```bash
# Local quick test
source venv/bin/activate && python -c "exec(open('quick_test.py').read())"

# Generate checkin data
python scripts/CheckinSynthesiser.py

# Run parameter sensitivity
python scripts/parameter_sensitivity.py

# Plot results
python scripts/plot_results.py
```

---

## 📝 Notes

- **Fixed Issues**: Removed duplicate EWMA calculations, added progress indicators
- **Colab Ready**: All code works in Google Colab environment
- **Scalable**: Handles datasets from 10 to 250K+ records
- **Research Ready**: Supports all research questions and experiments

For questions or issues, check the troubleshooting section or create an issue in the repository.

