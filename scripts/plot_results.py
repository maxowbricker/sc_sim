#!/usr/bin/env python3
"""
Generate publication-quality plots from systematic experiment results.

Creates figures similar to those in spatial crowdsourcing research papers.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class ResultsPlotter:
    """Generate publication-quality plots from experimental results."""
    
    def __init__(self, results_file):
        """Load experimental results from JSON file."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.df = pd.DataFrame(self.results)
        print(f"Loaded {len(self.results)} experimental results")
        
        # Set publication-quality plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_fairness_efficiency_tradeoff(self):
        """Plot fairness vs efficiency trade-off (like your paper examples)."""
        
        fairness_data = self.df[self.df['experiment'] == 'fairness_weight_sweep'].copy()
        
        if fairness_data.empty:
            print("No fairness sweep data found")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: TAR vs λ1
        ax1.plot(fairness_data['λ1'], fairness_data['task_assignment_ratio'], 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Fairness Weight (λ₁)')
        ax1.set_ylabel('Task Assignment Ratio')
        ax1.set_title('(a) Task Assignment vs Fairness Weight')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: JFI vs λ1
        ax2.plot(fairness_data['λ1'], fairness_data['jains_fairness_index'], 'o-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Fairness Weight (λ₁)')
        ax2.set_ylabel("Jain's Fairness Index")
        ax2.set_title('(b) Fairness vs Fairness Weight')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Wait time vs λ1
        ax3.plot(fairness_data['λ1'], fairness_data['avg_wait_time'], 'o-', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Fairness Weight (λ₁)')
        ax3.set_ylabel('Average Wait Time (min)')
        ax3.set_title('(c) Wait Time vs Fairness Weight')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Empty travel vs λ1
        ax4.plot(fairness_data['λ1'], fairness_data['empty_km_share'] * 100, 'o-', linewidth=2, markersize=6, color='red')
        ax4.set_xlabel('Fairness Weight (λ₁)')
        ax4.set_ylabel('Empty Travel (%)')
        ax4.set_title('(d) Empty Travel vs Fairness Weight')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fairness_efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_baseline_comparison(self):
        """Plot comparison against baseline methods."""
        
        baseline_data = self.df[self.df['experiment'] == 'baseline_comparison'].copy()
        
        if baseline_data.empty:
            print("No baseline comparison data found")
            return
        
        metrics = ['task_assignment_ratio', 'jains_fairness_index', 'avg_wait_time', 'empty_km_share']
        metric_labels = ['Task Assignment Ratio', "Jain's Fairness Index", 'Avg Wait Time (min)', 'Empty Travel Share']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            strategies = baseline_data['strategy'].values
            values = baseline_data[metric].values
            
            if metric == 'empty_km_share':
                values = values * 100  # Convert to percentage
            
            bars = ax.bar(strategies, values, alpha=0.8)
            ax.set_ylabel(label)
            ax.set_title(f'({chr(97+i)}) {label} by Strategy')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}' if metric != 'avg_wait_time' else f'{value:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_parameter_sensitivity(self):
        """Plot parameter sensitivity analysis."""
        
        # Get threshold sensitivity data
        threshold_data = self.df[self.df['experiment'] == 'threshold_sensitivity'].copy()
        
        if threshold_data.empty:
            print("No threshold sensitivity data found")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot threshold effects
        ax1.plot(threshold_data['soft_threshold'], threshold_data['task_assignment_ratio'], 'o-', linewidth=2)
        ax1.set_xlabel('Soft Threshold')
        ax1.set_ylabel('Task Assignment Ratio') 
        ax1.set_title('(a) TAR vs Soft Threshold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(threshold_data['soft_threshold'], threshold_data['jains_fairness_index'], 'o-', linewidth=2, color='orange')
        ax2.set_xlabel('Soft Threshold')
        ax2.set_ylabel("Jain's Fairness Index")
        ax2.set_title('(b) Fairness vs Soft Threshold')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(threshold_data['soft_threshold'], threshold_data['avg_wait_time'], 'o-', linewidth=2, color='green')
        ax3.set_xlabel('Soft Threshold')
        ax3.set_ylabel('Average Wait Time (min)')
        ax3.set_title('(c) Wait Time vs Soft Threshold')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(threshold_data['soft_threshold'], threshold_data['peak_backlog'], 'o-', linewidth=2, color='red')
        ax4.set_xlabel('Soft Threshold')
        ax4.set_ylabel('Peak Backlog')
        ax4.set_title('(d) Peak Backlog vs Soft Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_weight_combinations_radar(self):
        """Create radar chart comparing different weight combinations."""
        
        combinations_data = self.df[self.df['experiment'] == 'weight_combinations'].copy()
        
        if combinations_data.empty:
            print("No weight combinations data found")
            return
        
        # Select key metrics for radar chart
        metrics = ['task_assignment_ratio', 'jains_fairness_index', 'avg_wait_time', 'empty_km_share']
        metric_labels = ['TAR', 'JFI', 'Wait Time', 'Empty Travel']
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_data = combinations_data.copy()
        for metric in metrics:
            if metric == 'avg_wait_time' or metric == 'empty_km_share':
                # For these metrics, lower is better - invert them
                normalized_data[metric] = 1 - (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min())
            else:
                # Higher is better
                normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min())
        
        # Create figure for comparison table instead of radar (simpler)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        strategies = combinations_data['label'].values
        data_matrix = []
        
        for _, row in combinations_data.iterrows():
            data_matrix.append([
                row['task_assignment_ratio'],
                row['jains_fairness_index'], 
                row['avg_wait_time'],
                row['empty_km_share'] * 100
            ])
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticklabels(strategies)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Metric Value', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(metric_labels)):
                value = data_matrix[i][j]
                if metric_labels[j] == 'Wait Time':
                    text = f'{value:.1f}'
                elif metric_labels[j] == 'Empty Travel':
                    text = f'{value:.1f}%'
                else:
                    text = f'{value:.3f}'
                ax.text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title("Weight Combinations Performance Comparison")
        plt.tight_layout()
        plt.savefig('weight_combinations_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_all_plots(self):
        """Generate all publication-quality plots."""
        
        print("Generating publication-quality plots...")
        
        plots = []
        
        try:
            plots.append(self.plot_fairness_efficiency_tradeoff())
            print("✓ Fairness-efficiency trade-off plot generated")
        except Exception as e:
            print(f"✗ Error generating fairness-efficiency plot: {e}")
        
        try:
            plots.append(self.plot_baseline_comparison())
            print("✓ Baseline comparison plot generated")
        except Exception as e:
            print(f"✗ Error generating baseline comparison: {e}")
        
        try:
            plots.append(self.plot_parameter_sensitivity())
            print("✓ Parameter sensitivity plot generated")
        except Exception as e:
            print(f"✗ Error generating parameter sensitivity: {e}")
        
        try:
            plots.append(self.plot_weight_combinations_radar())
            print("✓ Weight combinations comparison generated")
        except Exception as e:
            print(f"✗ Error generating weight combinations: {e}")
        
        print(f"\nGenerated {len([p for p in plots if p is not None])} plots")
        print("Plots saved as PNG files in current directory")
        
        return plots


def main():
    """Generate plots from most recent experimental results."""
    
    # Find most recent results file
    results_files = list(Path('.').glob('systematic_experiments_*.json'))
    
    if not results_files:
        print("No experimental results found. Run systematic_experiments.py first.")
        return
    
    # Use most recent file
    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Using results from: {latest_file}")
    
    # Generate plots
    plotter = ResultsPlotter(latest_file)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()