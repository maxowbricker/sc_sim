"""
EWMA Effectiveness Analysis
============================

Creates comprehensive plots to show if EWMA is working to tighten
worker wait time distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path

# Load data
df = pd.read_csv('data/experiment_009+010_combined_results.csv')

# Filter to experiments with EWMA CV data
df_ewma = df[df['ewma_cv'].notna()].copy()
df_composite = df_ewma[df_ewma['strategy'] == 'composite'].copy()

print(f"📊 Creating EWMA analysis plots...")
print(f"   Data: {len(df_composite)} composite experiments with EWMA CV")

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================================================
# Plot 1: EWMA CV vs Fairness Weight
# ============================================================================
ax = axes[0, 0]

scatter = ax.scatter(df_composite['fairness_weight'],
                     df_composite['ewma_cv'],
                     s=100,
                     c=df_composite['jains_fairness_index'],
                     cmap='RdYlGn',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=0.5)

# Trend line
z = np.polyfit(df_composite['fairness_weight'], df_composite['ewma_cv'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_composite['fairness_weight'].min(), 
                     df_composite['fairness_weight'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

# Correlation
corr, pval = pearsonr(df_composite['fairness_weight'], df_composite['ewma_cv'])

ax.set_xlabel('λ₁ (Fairness Weight)', fontsize=12, fontweight='bold')
ax.set_ylabel('EWMA Coefficient of Variation (CV)\n← Lower = Tighter Distribution', 
              fontsize=11, fontweight='bold')
ax.set_title(f'(A) Does Higher Fairness Weight Reduce EWMA Spread?\nr={corr:.3f}, p={pval:.4f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('JFI', fontsize=10, fontweight='bold')

# Annotate best (lowest CV)
best_cv = df_composite.nsmallest(1, 'ewma_cv').iloc[0]
ax.annotate(f"Lowest CV\nλ₁={best_cv['fairness_weight']:.1f}\nCV={best_cv['ewma_cv']:.3f}",
            xy=(best_cv['fairness_weight'], best_cv['ewma_cv']),
            xytext=(20, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5),
            fontsize=9)

# ============================================================================
# Plot 2: JFI vs EWMA CV
# ============================================================================
ax = axes[0, 1]

scatter2 = ax.scatter(df_composite['jains_fairness_index'],
                      df_composite['ewma_cv'],
                      s=df_composite['fairness_weight']*50 + 50,
                      c=df_composite['utility_weight'],
                      cmap='coolwarm',
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.5)

# Trend line
z2 = np.polyfit(df_composite['jains_fairness_index'], df_composite['ewma_cv'], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df_composite['jains_fairness_index'].min(),
                      df_composite['jains_fairness_index'].max(), 100)
ax.plot(x_line2, p2(x_line2), "r--", linewidth=2, alpha=0.7)

# Correlation
corr2, pval2 = pearsonr(df_composite['jains_fairness_index'], df_composite['ewma_cv'])

ax.set_xlabel("Jain's Fairness Index (JFI)\n→ Higher = More Fair", fontsize=11, fontweight='bold')
ax.set_ylabel('EWMA CV\n← Lower = Tighter', fontsize=11, fontweight='bold')
ax.set_title(f'(B) Does Higher JFI Mean Tighter EWMA Distribution?\nr={corr2:.3f}, p={pval2:.4f}',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Colorbar
cbar2 = plt.colorbar(scatter2, ax=ax)
cbar2.set_label('λ₃ (Utility)', fontsize=10, fontweight='bold')

# Ideal region annotation
ax.annotate('← Ideal: High JFI, Low CV',
            xy=(df_composite['jains_fairness_index'].max()*0.95, 
                df_composite['ewma_cv'].min()*1.2),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

# ============================================================================
# Plot 3: EWMA CV Distribution by Fairness Weight Category
# ============================================================================
ax = axes[1, 0]

# Categorize fairness weights
df_composite['fairness_category'] = pd.cut(df_composite['fairness_weight'], 
                                            bins=[0, 1, 2, 3, 4, 5],
                                            labels=['Very Low\n(≤1)', 'Low\n(1-2)', 
                                                   'Medium\n(2-3)', 'High\n(3-4)', 'Very High\n(4-5)'])

# Prepare data for box plot
box_data = []
box_labels = []
for cat in df_composite['fairness_category'].cat.categories:
    cat_data = df_composite[df_composite['fairness_category'] == cat]['ewma_cv'].dropna()
    if len(cat_data) > 0:
        box_data.append(cat_data)
        box_labels.append(cat)

# Box plot
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)

# Color boxes
colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff']
for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Fairness Weight Category', fontsize=11, fontweight='bold')
ax.set_ylabel('EWMA CV\n← Lower = Tighter', fontsize=11, fontweight='bold')
ax.set_title('(C) EWMA Spread Distribution by Fairness Level\n(Box plot shows median, quartiles, outliers)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Summary Statistics Table
# ============================================================================
ax = axes[1, 1]
ax.axis('off')

# Create summary table
summary_data = []

for cat in df_composite['fairness_category'].cat.categories:
    cat_data = df_composite[df_composite['fairness_category'] == cat]
    if len(cat_data) > 0:
        summary_data.append([
            str(cat).replace('\n', ' '),
            f"{len(cat_data)}",
            f"{cat_data['ewma_cv'].mean():.3f}",
            f"{cat_data['ewma_cv'].std():.3f}",
            f"{cat_data['ewma_cv'].min():.3f}",
            f"{cat_data['ewma_cv'].max():.3f}",
            f"{cat_data['jains_fairness_index'].mean():.4f}"
        ])

table = ax.table(cellText=summary_data,
                colLabels=['Fairness\nCategory', 'N', 'Mean\nEWMA CV', 'Std\nCV', 
                          'Min\nCV', 'Max\nCV', 'Mean\nJFI'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
colors_row = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff']
for i, color in enumerate(colors_row[:len(summary_data)], 1):
    for j in range(7):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.5)

ax.set_title('(D) Summary Statistics: EWMA CV by Fairness Category',
             fontsize=12, fontweight='bold', pad=20)

# Overall title
fig.suptitle('🎯 Is EWMA Working to Tighten Worker Wait Time Distribution?\n' +
             'Lower EWMA CV = More Uniform Worker Experience = Tighter Distribution',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('figures/ewma_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Plot saved: figures/ewma_effectiveness_analysis.png")

# Print key findings
print("\n" + "=" * 80)
print("KEY FINDINGS: Is EWMA Working?")
print("=" * 80)

print(f"\n📊 Correlation Analysis:")
print(f"   Fairness Weight vs EWMA CV: r = {corr:.4f}, p = {pval:.4f}")
if corr < 0 and pval < 0.05:
    print(f"   ✅ Higher fairness weight → Lower CV (tighter distribution)")
elif corr > 0 and pval < 0.05:
    print(f"   ⚠️ Higher fairness weight → Higher CV (more spread) - UNEXPECTED!")
else:
    print(f"   ❌ No significant relationship")

print(f"\n   JFI vs EWMA CV: r = {corr2:.4f}, p = {pval2:.4f}")
if corr2 < 0 and pval2 < 0.05:
    print(f"   ✅ Higher JFI → Lower CV (fairness compresses distribution)")
elif corr2 > 0 and pval2 < 0.05:
    print(f"   ⚠️ Higher JFI → Higher CV - UNEXPECTED!")
else:
    print(f"   ❌ No significant relationship")

print(f"\n📈 Range of EWMA CV:")
print(f"   Minimum: {df_composite['ewma_cv'].min():.4f} (tightest)")
print(f"   Maximum: {df_composite['ewma_cv'].max():.4f} (most spread)")
print(f"   Mean: {df_composite['ewma_cv'].mean():.4f} ± {df_composite['ewma_cv'].std():.4f}")
print(f"   Range: {(df_composite['ewma_cv'].max() - df_composite['ewma_cv'].min()):.4f}")

reduction = (df_composite['ewma_cv'].max() - df_composite['ewma_cv'].min()) / df_composite['ewma_cv'].max() * 100
print(f"   Potential reduction: {reduction:.1f}% from worst to best config")

print(f"\n🏆 Best Configuration (Lowest CV):")
best = df_composite.nsmallest(1, 'ewma_cv').iloc[0]
print(f"   {best['name']}")
print(f"   λ₁={best['fairness_weight']:.1f}, λ₂={best['starvation_weight']:.1f}, λ₃={best['utility_weight']:.1f}")
print(f"   EWMA CV: {best['ewma_cv']:.4f} (tightest distribution)")
print(f"   JFI: {best['jains_fairness_index']:.4f}")
print(f"   Wait Time: {best['mean_task_wait_time_min']:.2f} min")

print(f"\n⚠️ Worst Configuration (Highest CV):")
worst = df_composite.nlargest(1, 'ewma_cv').iloc[0]
print(f"   {worst['name']}")
print(f"   λ₁={worst['fairness_weight']:.1f}, λ₂={worst['starvation_weight']:.1f}, λ₃={worst['utility_weight']:.1f}")
print(f"   EWMA CV: {worst['ewma_cv']:.4f} (most spread distribution)")
print(f"   JFI: {worst['jains_fairness_index']:.4f}")
print(f"   Wait Time: {worst['mean_task_wait_time_min']:.2f} min")

print("\n" + "=" * 80)




