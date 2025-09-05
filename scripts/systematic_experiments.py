#!/usr/bin/env python3
"""
Systematic experimental framework for spatial crowdsourcing research.

This script generates the type of systematic analysis shown in research papers,
with proper parameter sweeps, baseline comparisons, and publication-ready results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation


class SystematicExperiments:
    """Framework for systematic spatial crowdsourcing experiments."""
    
    def __init__(self, dataset="didi"):
        self.dataset = dataset
        self.workers, self.tasks = load_workers_tasks(dataset)
        self.results = []
        print(f"Loaded {len(self.workers)} workers, {len(self.tasks)} tasks from {dataset}")
    
    def run_experiment_set(self, experiment_name, param_grid, description=""):
        """Run a systematic set of experiments with parameter grid."""
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SET: {experiment_name}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        results = []
        total_configs = len(param_grid)
        
        for i, params in enumerate(param_grid):
            print(f"Progress: {i+1}/{total_configs} - {params}")
            
            # Base configuration
            config = {
                'dataset': self.dataset,
                'assignment_strategy': 'composite',
                'strategy_params': {
                    'gamma': 0.3,
                    'k': 15,
                    'soft_threshold': 0.5,
                    **params  # Override with experiment parameters
                }
            }
            
            try:
                summary = run_simulation(self.workers, self.tasks, sim_config=config)
                
                # Extract comprehensive metrics
                result = {
                    'experiment': experiment_name,
                    'config_id': i,
                    **params,
                    
                    # Performance metrics
                    'completed_tasks': summary.get('completed_tasks', 0),
                    'task_assignment_ratio': summary.get('completed_tasks', 0) / len(self.tasks),
                    'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
                    'p90_wait_time': np.percentile(summary.get('wait_times', [0]), 90) if summary.get('wait_times') else 0,
                    'max_wait_time': max(summary.get('wait_times', [0])) if summary.get('wait_times') else 0,
                    
                    # Travel efficiency
                    'avg_travel_distance': summary.get('total_travel_km', 0) / max(1, summary.get('completed_tasks', 1)),
                    'empty_km_share': summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)),
                    'total_travel_km': summary.get('total_travel_km', 0),
                    
                    # Fairness metrics
                    'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                    'utility_difference': summary.get('final_utility_difference_tasks', 0),
                    'fairness_loss': summary.get('final_fairness_loss', 0),
                    'ewma_cv': summary.get('final_ewma_cv', 0),
                    'mean_jfi_over_time': summary.get('mean_jfi_over_time', 0),
                    
                    # System metrics
                    'peak_backlog': summary.get('backlog_peak', 0),
                    'expired_tasks': len(self.tasks) - summary.get('completed_tasks', 0),
                    
                    # Metadata
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        self.results.extend(results)
        print(f"Completed {experiment_name}: {len(results)} successful runs")
        return results
    
    def fairness_weight_sweep(self):
        """Experiment 1: Effect of fairness weight (λ₁) on fairness-efficiency trade-off."""
        
        param_grid = [
            {'λ1': lambda1, 'λ2': 1.0, 'λ3': 0.5}
            for lambda1 in np.linspace(0.2, 3.0, 15)  # 15 points from 0.2 to 3.0
        ]
        
        return self.run_experiment_set(
            "fairness_weight_sweep",
            param_grid,
            "Impact of fairness weight (λ₁) on fairness vs efficiency trade-offs"
        )
    
    def efficiency_weight_sweep(self):
        """Experiment 2: Effect of utility weight (λ₃) on efficiency."""
        
        param_grid = [
            {'λ1': 1.0, 'λ2': 1.0, 'λ3': lambda3}
            for lambda3 in np.linspace(0.1, 2.5, 15)
        ]
        
        return self.run_experiment_set(
            "efficiency_weight_sweep", 
            param_grid,
            "Impact of utility weight (λ₃) on system efficiency"
        )
    
    def starvation_weight_sweep(self):
        """Experiment 3: Effect of starvation weight (λ₂) on task completion."""
        
        param_grid = [
            {'λ1': 1.0, 'λ2': lambda2, 'λ3': 0.5}
            for lambda2 in np.linspace(0.2, 4.0, 15)
        ]
        
        return self.run_experiment_set(
            "starvation_weight_sweep",
            param_grid, 
            "Impact of starvation weight (λ₂) on task completion and wait times"
        )
    
    def threshold_sensitivity(self):
        """Experiment 4: Soft threshold sensitivity analysis."""
        
        param_grid = [
            {'λ1': 1.0, 'λ2': 1.0, 'λ3': 0.5, 'soft_threshold': threshold}
            for threshold in np.linspace(0.1, 1.5, 15)
        ]
        
        return self.run_experiment_set(
            "threshold_sensitivity",
            param_grid,
            "Effect of soft threshold on assignment selectivity"
        )
    
    def weight_ratio_combinations(self):
        """Experiment 5: Key weight combination comparisons."""
        
        param_grid = [
            # Fairness-focused
            {'λ1': 3.0, 'λ2': 1.0, 'λ3': 0.3, 'label': 'High Fairness'},
            {'λ1': 2.0, 'λ2': 1.0, 'λ3': 0.5, 'label': 'Medium Fairness'},
            
            # Efficiency-focused  
            {'λ1': 0.3, 'λ2': 1.0, 'λ3': 2.5, 'label': 'High Efficiency'},
            {'λ1': 0.5, 'λ2': 1.0, 'λ3': 1.5, 'label': 'Medium Efficiency'},
            
            # Starvation-focused
            {'λ1': 1.0, 'λ2': 3.0, 'λ3': 0.5, 'label': 'High Starvation Prevention'},
            
            # Balanced
            {'λ1': 1.0, 'λ2': 1.0, 'λ3': 1.0, 'label': 'Equal Weights'},
            {'λ1': 1.0, 'λ2': 1.0, 'λ3': 0.5, 'label': 'Baseline'},
        ]
        
        return self.run_experiment_set(
            "weight_combinations",
            param_grid,
            "Comparison of key weight ratio combinations"
        )
    
    def baseline_comparison(self):
        """Experiment 6: Compare against baseline strategies."""
        
        baseline_configs = [
            # Your composite approach variants
            {'assignment_strategy': 'composite', 'λ1': 1.0, 'λ2': 1.0, 'λ3': 0.5, 'label': 'Composite-Baseline'},
            {'assignment_strategy': 'composite', 'λ1': 2.0, 'λ2': 1.0, 'λ3': 0.3, 'label': 'Composite-Fairness'},
            {'assignment_strategy': 'composite', 'λ1': 0.5, 'λ2': 1.0, 'λ3': 1.5, 'label': 'Composite-Efficiency'},
            
            # Baselines
            {'assignment_strategy': 'greedy', 'label': 'Greedy-Nearest'},
        ]
        
        results = []
        for i, config_params in enumerate(baseline_configs):
            print(f"Testing: {config_params['label']}")
            
            config = {
                'dataset': self.dataset,
                'assignment_strategy': config_params.pop('assignment_strategy'),
                'strategy_params': {
                    'gamma': 0.3, 'k': 15, 'soft_threshold': 0.5,
                    **{k: v for k, v in config_params.items() if k != 'label'}
                }
            }
            
            summary = run_simulation(self.workers, self.tasks, sim_config=config)
            
            result = {
                'experiment': 'baseline_comparison',
                'strategy': config_params.get('label', 'Unknown'),
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / len(self.tasks),
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
                'empty_km_share': summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)),
            }
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def save_results(self, filename=None):
        """Save all experimental results."""
        if filename is None:
            filename = f"systematic_experiments_{self.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
        return filename
    
    def generate_summary_report(self):
        """Generate a summary report of all experiments."""
        
        print(f"\n{'='*80}")
        print("SYSTEMATIC EXPERIMENTS SUMMARY REPORT")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results to summarize.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Group by experiment type
        for experiment in df['experiment'].unique():
            exp_data = df[df['experiment'] == experiment]
            
            print(f"\n{experiment.upper().replace('_', ' ')}")
            print("-" * 50)
            
            if experiment in ['fairness_weight_sweep', 'efficiency_weight_sweep', 'starvation_weight_sweep']:
                # Show parameter range and key metrics
                param_col = [col for col in exp_data.columns if col.startswith('λ')][0]
                print(f"Parameter range: {param_col} = {exp_data[param_col].min():.2f} → {exp_data[param_col].max():.2f}")
                print(f"JFI range: {exp_data['jains_fairness_index'].min():.3f} → {exp_data['jains_fairness_index'].max():.3f}")
                print(f"Wait time range: {exp_data['avg_wait_time'].min():.1f} → {exp_data['avg_wait_time'].max():.1f} min")
                print(f"Empty travel range: {exp_data['empty_km_share'].min():.1%} → {exp_data['empty_km_share'].max():.1%}")
            
            elif experiment == 'baseline_comparison':
                print("Strategy comparison:")
                for _, row in exp_data.iterrows():
                    print(f"  {row['strategy']:<20}: TAR={row['task_assignment_ratio']:.1%}, JFI={row['jains_fairness_index']:.3f}")


def main():
    """Run systematic experiments for research analysis."""
    
    print("SYSTEMATIC SPATIAL CROWDSOURCING EXPERIMENTS")
    print("=" * 50)
    print("This will generate data for research paper figures and analysis.")
    
    # Initialize experimental framework
    experiments = SystematicExperiments("didi")
    
    # Run core experiment sets
    print("\n1. Running fairness weight sweep...")
    experiments.fairness_weight_sweep()
    
    print("\n2. Running efficiency weight sweep...")  
    experiments.efficiency_weight_sweep()
    
    print("\n3. Running threshold sensitivity...")
    experiments.threshold_sensitivity()
    
    print("\n4. Running weight combinations...")
    experiments.weight_ratio_combinations()
    
    print("\n5. Running baseline comparison...")
    experiments.baseline_comparison()
    
    # Save results and generate report
    filename = experiments.save_results()
    experiments.generate_summary_report()
    
    print(f"\n{'='*80}")
    print("NEXT STEPS FOR YOUR RESEARCH:")
    print(f"{'='*80}")
    print(f"1. Results saved to: {filename}")
    print("2. Use this data to create publication-quality plots")
    print("3. Compare your composite approach against baselines")
    print("4. Analyze fairness-efficiency trade-off curves")
    print("5. Use insights to guide PPO implementation")


if __name__ == "__main__":
    main()