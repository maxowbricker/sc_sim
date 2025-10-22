#!/usr/bin/env python3
"""
Create a comprehensive analysis notebook for Experiment 008 results.
"""

import nbformat as nbf
from pathlib import Path

# Create notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title cell
cells.append(nbf.v4.new_markdown_cell("""# Experiment 008: Results Analysis
## Score Normalization and Threshold Ablation

### Completed: October 19, 2025
**Duration**: 5.74 hours | **Success**: 12/12 experiments

This notebook analyzes the completed Experiment 008 to determine the cause of the worker idle time paradox.

**Hypotheses Tested:**
1. **H1 (Mis-scaled Components)**: Fairness component dominates due to scale mismatch  
2. **H2 (Soft-Threshold Feedback)**: Threshold deferral mechanism creates artificial task shortages

**Experimental Groups:**
- **Group A**: Greedy baseline (efficiency reference) - 3 experiments
- **Group B**: Composite current (replicate paradox) - 3 experiments, `normalize_scores=False`
- **Group C**: Composite + normalization (test H1) - 3 experiments, `normalize_scores=True`
- **Group D**: Composite + normalization + no threshold (test H1+H2) - 3 experiments, `normalize_scores=True, disable_soft_threshold=True`"""))

# Import libraries
cells.append(nbf.v4.new_code_cell("""# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("✅ Libraries loaded")"""))

# Load data
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Experimental Results"""))

cells.append(nbf.v4.new_code_cell("""# Load the completed experiment data
data_dir = Path("data/exp_008_20251019_112545")

# Load aggregate results
results_df = pd.read_csv(data_dir / "experiment_008_aggregate_results.csv")

# Load metadata
with open(data_dir / "experiment_008_metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"📊 Loaded {len(results_df)} experiments")
print(f"⏱️  Total duration: {metadata['duration_hours']:.2f} hours")
print(f"✅ Success rate: {metadata['successful']}/{metadata['total_experiments']}")
print(f"\\nExperiments per group:")
print(results_df.groupby('group')['name'].count())

results_df.head()"""))

# Summary statistics
cells.append(nbf.v4.new_markdown_cell("""## 2. Summary Statistics"""))

cells.append(nbf.v4.new_code_cell("""# Compute mean and std for each group
summary = results_df.groupby('group').agg({
    'completed_tasks': ['mean', 'std'],
    'task_assignment_ratio': ['mean', 'std'],
    'mean_task_wait_time_min': ['mean', 'std'],
    'mean_pickup_distance_km': ['mean', 'std'],
    'jains_fairness_index': ['mean', 'std'],
}).round(3)

print("=" * 80)
print("SUMMARY STATISTICS BY GROUP")
print("=" * 80)
print(summary)
print("\\n")

# Diagnostic metrics for composite groups
composite_df = results_df[results_df['strategy'] == 'composite'].copy()
diag_summary = composite_df.groupby('group').agg({
    'deferral_rate': ['mean', 'std'],
    'dominant_fairness_pct': ['mean', 'std'],
    'dominant_utility_pct': ['mean', 'std'],
    'avg_dominance_ratio': ['mean', 'std']
}).round(3)

print("=" * 80)
print("DIAGNOSTIC METRICS (Composite Groups B, C, D)")
print("=" * 80)
print(diag_summary)"""))

# Wait time analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Key Finding: Task Wait Times

This is our proxy for the "idle time paradox" - higher task wait times indicate workers are idle longer."""))

cells.append(nbf.v4.new_code_cell("""# Compare task wait times across groups
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
results_df.boxplot(column='mean_task_wait_time_min', by='group', ax=ax1)
ax1.set_title('Task Wait Time Distribution by Group', fontsize=14, fontweight='bold')
ax1.set_xlabel('Group', fontsize=12)
ax1.set_ylabel('Mean Task Wait Time (minutes)', fontsize=12)
ax1.grid(True, alpha=0.3)
plt.sca(ax1)
plt.xticks([1, 2, 3, 4], ['A\\n(Greedy)', 'B\\n(Current)', 'C\\n(Normalized)', 'D\\n(Norm+NoThresh)'])

# Bar plot with error bars
group_means = results_df.groupby('group')['mean_task_wait_time_min'].mean()
group_stds = results_df.groupby('group')['mean_task_wait_time_min'].std()
ax2.bar(group_means.index, group_means.values, yerr=group_stds.values, 
        capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Mean Task Wait Time by Group', fontsize=14, fontweight='bold')
ax2.set_xlabel('Group', fontsize=12)
ax2.set_ylabel('Mean Task Wait Time (minutes)', fontsize=12)
ax2.set_xticklabels(['A\\n(Greedy)', 'B\\n(Current)', 'C\\n(Normalized)', 'D\\n(Norm+NoThresh)'])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (g, v) in enumerate(group_means.items()):
    ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('')
plt.tight_layout()
plt.show()

# Statistical comparison
print("\\n" + "="*80)
print("WAIT TIME COMPARISON")
print("="*80)
for group in ['A', 'B', 'C', 'D']:
    group_data = results_df[results_df['group'] == group]['mean_task_wait_time_min']
    print(f"Group {group}: {group_data.mean():.2f} ± {group_data.std():.2f} min")

# Compare B vs C (test H1: normalization)
group_b = results_df[results_df['group'] == 'B']['mean_task_wait_time_min']
group_c = results_df[results_df['group'] == 'C']['mean_task_wait_time_min']
t_stat, p_value = stats.ttest_ind(group_b, group_c)
print(f"\\nB vs C (H1 Test): t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("✅ Significant difference! Normalization DOES affect wait times.")
else:
    print("❌ No significant difference. Normalization does NOT affect wait times.")"""))

# Component dominance
cells.append(nbf.v4.new_markdown_cell("""## 4. Component Dominance Analysis

Test H1: Does fairness dominate in Group B due to mis-scaled components?"""))

cells.append(nbf.v4.new_code_cell("""# Plot component dominance
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

components = ['dominant_fairness_pct', 'dominant_utility_pct', 'dominant_starvation_pct']
titles = ['Fairness Dominance', 'Utility Dominance', 'Starvation Dominance']
colors = ['#ff7f0e', '#2ca02c', '#9467bd']

for ax, comp, title, color in zip(axes, components, titles, colors):
    group_means = composite_df.groupby('group')[comp].mean()
    group_stds = composite_df.groupby('group')[comp].std()
    
    ax.bar(group_means.index, group_means.values, yerr=group_stds.values,
           capsize=5, alpha=0.7, color=color)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Group', fontsize=11)
    ax.set_ylabel('Dominance (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticklabels(['B\\n(Current)', 'C\\n(Normalized)', 'D\\n(Norm+NoThresh)'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (g, v) in enumerate(group_means.items()):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Component Dominance by Group (Composite Strategies)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Print detailed comparison
print("\\n" + "="*80)
print("COMPONENT DOMINANCE PERCENTAGES")
print("="*80)
for group in ['B', 'C', 'D']:
    group_data = composite_df[composite_df['group'] == group]
    print(f"\\nGroup {group}:")
    print(f"  Fairness:   {group_data['dominant_fairness_pct'].mean():.1f}% ± {group_data['dominant_fairness_pct'].std():.1f}%")
    print(f"  Utility:    {group_data['dominant_utility_pct'].mean():.1f}% ± {group_data['dominant_utility_pct'].std():.1f}%")
    print(f"  Starvation: {group_data['dominant_starvation_pct'].mean():.1f}% ± {group_data['dominant_starvation_pct'].std():.1f}%")"""))

# Deferral rate
cells.append(nbf.v4.new_markdown_cell("""## 5. Deferral Rate Analysis

Test H2: Does the soft threshold cause excessive task deferrals?"""))

cells.append(nbf.v4.new_code_cell("""# Plot deferral rates
fig, ax = plt.subplots(figsize=(10, 6))

group_means = composite_df.groupby('group')['deferral_rate'].mean() * 100
group_stds = composite_df.groupby('group')['deferral_rate'].std() * 100

bars = ax.bar(group_means.index, group_means.values, yerr=group_stds.values,
              capsize=5, alpha=0.7, color=['#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Task Deferral Rate by Group', fontsize=14, fontweight='bold')
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Deferral Rate (%)', fontsize=12)
ax.set_xticklabels(['B\\n(With Threshold)', 'C\\n(With Threshold)', 'D\\n(No Threshold)'])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (g, v) in enumerate(group_means.items()):
    ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\\n" + "="*80)
print("DEFERRAL RATE ANALYSIS")
print("="*80)
for group in ['B', 'C', 'D']:
    group_data = composite_df[composite_df['group'] == group]['deferral_rate'] * 100
    print(f"Group {group}: {group_data.mean():.2f}% ± {group_data.std():.2f}%")

# Compare C vs D (test H2: threshold removal)
group_c_def = composite_df[composite_df['group'] == 'C']['deferral_rate']
group_d_def = composite_df[composite_df['group'] == 'D']['deferral_rate']
t_stat, p_value = stats.ttest_ind(group_c_def, group_d_def)
print(f"\\nC vs D (H2 Test): t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("✅ Significant difference! Threshold DOES affect deferral rates.")
else:
    print("❌ No significant difference. Threshold does NOT significantly affect deferrals.")"""))

# Final summary
cells.append(nbf.v4.new_markdown_cell("""## 6. Conclusions

Run the cells above to see:
1. **Did Group B show the paradox?** (Higher wait times than Group A?)
2. **Does normalization help?** (Group C wait times < Group B?)
3. **What dominates in Group B?** (Which component has highest %)
4. **Does threshold contribute?** (Group D deferrals << Group C deferrals?)"""))

cells.append(nbf.v4.new_code_cell("""# Final summary table
print("\\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

summary_table = pd.DataFrame({
    'Group': ['A', 'B', 'C', 'D'],
    'Description': ['Greedy', 'Current', 'Normalized', 'Norm+NoThresh'],
    'Mean Wait (min)': [results_df[results_df['group'] == g]['mean_task_wait_time_min'].mean() 
                        for g in ['A', 'B', 'C', 'D']],
    'Pickup Dist (km)': [results_df[results_df['group'] == g]['mean_pickup_distance_km'].mean() 
                         for g in ['A', 'B', 'C', 'D']],
    'Completed Tasks': [results_df[results_df['group'] == g]['completed_tasks'].mean() 
                        for g in ['A', 'B', 'C', 'D']],
    'JFI': [results_df[results_df['group'] == g]['jains_fairness_index'].mean() 
            for g in ['A', 'B', 'C', 'D']]
})

# Add diagnostic metrics for composite groups
for g in ['B', 'C', 'D']:
    group_data = composite_df[composite_df['group'] == g]
    idx = summary_table[summary_table['Group'] == g].index[0]
    summary_table.loc[idx, 'Deferral %'] = group_data['deferral_rate'].mean() * 100
    summary_table.loc[idx, 'Fairness Dom %'] = group_data['dominant_fairness_pct'].mean()
    summary_table.loc[idx, 'Utility Dom %'] = group_data['dominant_utility_pct'].mean()

print(summary_table.round(2))
print("\\n✅ Analysis complete!")"""))

# Add cells to notebook
nb['cells'] = cells

# Write notebook
output_path = Path("results_analysis.ipynb")
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"✅ Created analysis notebook: {output_path}")

