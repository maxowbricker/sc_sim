"""
Quick Preview of Combined Experiments 009 + 010
================================================

Generates a visual summary of the combined parameter space and key metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
data_dir = Path(__file__).parent / "data"
df = pd.read_csv(data_dir / "experiment_009+010_combined_results.csv")

print("=" * 70)
print("COMBINED DATASET PREVIEW")
print("=" * 70)
print(f"\nTotal Experiments: {len(df)}")
print(f"  - Greedy: {len(df[df['strategy'] == 'greedy'])}")
print(f"  - Composite: {len(df[df['strategy'] == 'composite'])}")
print(f"\nBy Source:")
print(df['source_experiment'].value_counts().to_string())
print()

# Filter to composite only for analysis
df_comp = df[df['strategy'] == 'composite'].copy()

# Create a 2x2 visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Combined Experiments 009 + 010: Parameter Space & Metrics Overview', 
             fontsize=16, fontweight='bold')

# Plot 1: Parameter Space Coverage (λ₁ vs λ₃)
ax = axes[0, 0]
scatter = ax.scatter(df_comp['fairness_weight'], 
                     df_comp['utility_weight'],
                     c=df_comp['jains_fairness_index'],
                     s=100,
                     cmap='viridis',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=0.5)
ax.set_xlabel('λ₁ (Fairness Weight)', fontsize=11, fontweight='bold')
ax.set_ylabel('λ₃ (Utility Weight)', fontsize=11, fontweight='bold')
ax.set_title('A) Parameter Space Coverage\n(Color = JFI)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Jain\'s Fairness Index')

# Annotate by source
for source in ['exp_009', 'exp_010']:
    df_source = df_comp[df_comp['source_experiment'] == source]
    ax.scatter([], [], c='gray', alpha=0.7, s=100, 
               label=f'{source} (n={len(df_source)})')
ax.legend(loc='upper right', fontsize=9)

# Plot 2: Pareto Frontier (Wait Time vs JFI)
ax = axes[0, 1]
for source, marker, color in [('exp_009', 'o', 'steelblue'), 
                                ('exp_010', 's', 'coral')]:
    df_source = df_comp[df_comp['source_experiment'] == source]
    ax.scatter(df_source['mean_task_wait_time_min'],
               df_source['jains_fairness_index'],
               s=100,
               marker=marker,
               color=color,
               alpha=0.7,
               edgecolors='black',
               linewidth=0.5,
               label=source)
ax.set_xlabel('Mean Task Wait Time (min)', fontsize=11, fontweight='bold')
ax.set_ylabel('Jain\'s Fairness Index', fontsize=11, fontweight='bold')
ax.set_title('B) Pareto Frontier\n(Fairness vs Efficiency)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 3: Parameter Distribution Histograms
ax = axes[1, 0]
params = ['fairness_weight', 'starvation_weight', 'utility_weight']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (param, color) in enumerate(zip(params, colors)):
    values = df_comp[param].dropna()
    ax.hist(values, bins=15, alpha=0.5, color=color, 
            label=f'λ{i+1} ({param.split("_")[0].capitalize()})',
            edgecolor='black', linewidth=0.5)
ax.set_xlabel('Parameter Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('C) Parameter Distribution', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Top 10 Configurations Comparison
ax = axes[1, 1]
top_jfi = df_comp.nlargest(10, 'jains_fairness_index').copy()
top_jfi['config_label'] = top_jfi.apply(
    lambda row: f"λ₁={row['fairness_weight']:.1f}\nλ₃={row['utility_weight']:.1f}", 
    axis=1
)
top_jfi = top_jfi.sort_values('jains_fairness_index', ascending=True)

y_pos = range(len(top_jfi))
bars = ax.barh(y_pos, top_jfi['jains_fairness_index'], 
               color=['coral' if src == 'exp_010' else 'steelblue' 
                      for src in top_jfi['source_experiment']],
               alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_jfi['config_label'], fontsize=8)
ax.set_xlabel('Jain\'s Fairness Index', fontsize=11, fontweight='bold')
ax.set_title('D) Top 10 Configurations by JFI', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

# Add legend for source
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='exp_009'),
                   Patch(facecolor='coral', alpha=0.7, label='exp_010')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()

# Save figure
output_file = data_dir.parent / "combined_experiments_preview.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"💾 Saved visualization: {output_file}")
print()

# Print top configurations
print("🏆 TOP 5 CONFIGURATIONS BY FAIRNESS (JFI):")
print("-" * 70)
top5_jfi = df_comp.nlargest(5, 'jains_fairness_index')
for idx, row in top5_jfi.iterrows():
    print(f"{row['name']}")
    print(f"  λ₁={row['fairness_weight']:.1f}, λ₂={row['starvation_weight']:.1f}, "
          f"λ₃={row['utility_weight']:.1f}, θ={row['soft_threshold']:.1f}")
    print(f"  JFI: {row['jains_fairness_index']:.4f} | "
          f"Wait: {row['mean_task_wait_time_min']:.2f} min | "
          f"Source: {row['source_experiment']}")
    print()

print("⚡ TOP 5 CONFIGURATIONS BY EFFICIENCY (Low Wait Time):")
print("-" * 70)
top5_eff = df_comp.nsmallest(5, 'mean_task_wait_time_min')
for idx, row in top5_eff.iterrows():
    print(f"{row['name']}")
    print(f"  λ₁={row['fairness_weight']:.1f}, λ₂={row['starvation_weight']:.1f}, "
          f"λ₃={row['utility_weight']:.1f}, θ={row['soft_threshold']:.1f}")
    print(f"  Wait: {row['mean_task_wait_time_min']:.2f} min | "
          f"JFI: {row['jains_fairness_index']:.4f} | "
          f"Source: {row['source_experiment']}")
    print()

print("=" * 70)
print("✅ Preview complete! Check the visualization:")
print(f"   {output_file}")
print("=" * 70)




