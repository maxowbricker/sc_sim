"""
Lightweight Parameter Sweep Script - No DataFrames, Memory Efficient

Tests key bottleneck parameters that could improve the 86.6% TAR:
1. soft_threshold: Controls assignment strictness 
2. k: Number of nearest workers considered  
3. Weight balance: fairness vs starvation vs utility

Uses object-based loading (not DataFrames) for memory efficiency.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import traceback
from itertools import product
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import load_workers_tasks  # Object-based, memory efficient
from simulator.simulation import run_simulation
from config import get_simulation_config

class BottleneckParameterSweep:
    def __init__(self, output_dir="results", log_level=logging.INFO):
        """Initialize parameter sweep with robust logging and error handling."""
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup timestamped logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"bottleneck_sweep_{self.timestamp}.json"
        self.log_file = self.output_dir / f"bottleneck_sweep_{self.timestamp}.log"
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = []
        self.failed_runs = []
        
        # Performance tracking
        self.start_time = None
        self.total_experiments = 0
        self.completed_experiments = 0

    def define_parameter_space(self, scale="focused"):
        """
        Define parameter combinations to test based on identified bottlenecks.
        
        Args:
            scale: "test" for small validation, "focused" for targeted sweep, "full" for comprehensive
        """
        if scale == "test":
            # TRULY minimal set for validation (9 combinations)
            return {
                'soft_threshold': [0.3, 0.5, 0.7],
                'k': [10, 15, 25], 
                'fairness_weight': [1.0],
                'starvation_weight': [1.0], 
                'utility_weight': [1.0],
                'fairness_metric': ['ewma']
            }
        elif scale == "focused":
            # Focus on most promising parameters - FIXED RANGES (~126 combinations)
            return {
                'soft_threshold': [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],  # More reasonable range around baseline
                'k': [10, 15, 25, 50],  # Test spatial search efficiency
                'fairness_weight': [0.5, 1.0, 2.0],
                'starvation_weight': [0.5, 1.0, 2.0],  
                'utility_weight': [0.5, 1.0, 2.0],
                'fairness_metric': ['ewma']  # Keep metric constant for focused test
            }
        else:  # "full" 
            # Comprehensive parameter space (~800+ combinations)
            return {
                'soft_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'k': [5, 10, 15, 20, 25, 30, 40, 50], 
                'fairness_weight': [0.2, 0.5, 1.0, 1.5, 2.0],
                'starvation_weight': [0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
                'utility_weight': [0.5, 1.0, 1.5],
                'fairness_metric': ['ewma', 'idle_time']  
            }

    def generate_parameter_combinations(self, param_space):
        """Generate all parameter combinations from parameter space."""
        
        # Get all parameter names and values
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Generate cartesian product
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations

    def run_single_experiment(self, workers, tasks, params, experiment_id):
        """
        Run a single simulation experiment with given parameters.
        
        Args:
            workers: List of Worker objects (memory efficient)
            tasks: List of Task objects (memory efficient)
            params: Parameter dictionary
            experiment_id: Unique experiment identifier
            
        Returns:
            dict: Experiment results or None if failed
        """
        try:
            # Build simulation config
            sim_config = {
                "assignment_strategy": "composite",
                "strategy_params": params
            }
            
            experiment_start = time.time()
            
            # Run simulation - using object-based approach
            results = run_simulation(workers, tasks, sim_config=sim_config)
            
            experiment_time = time.time() - experiment_start
            
            # Calculate key metrics
            total_tasks = len(tasks)
            completed_tasks = results.get('completed_tasks', 0)
            tar_percent = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
            
            # Build result record
            result = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'metrics': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'tar_percent': tar_percent,
                    'total_travel_km': results.get('total_travel_km', 0),
                    'empty_km': results.get('empty_km', 0),
                    'passenger_km': results.get('passenger_km', 0),
                    'average_wait_time_min': results.get('total_wait_min', 0) / max(completed_tasks, 1),
                    'experiment_duration_sec': experiment_time
                },
                'raw_results': results
            }
            
            self.logger.info(f"✅ Exp {experiment_id}: TAR={tar_percent:.2f}% "
                           f"(threshold={params['soft_threshold']}, k={params['k']}, "
                           f"weights={params['fairness_weight']}/{params['starvation_weight']}/{params['utility_weight']})")
            
            return result
            
        except Exception as e:
            error_info = {
                'experiment_id': experiment_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.logger.error(f"❌ Exp {experiment_id} failed: {str(e)}")
            self.failed_runs.append(error_info)
            
            return None

    def save_progress(self):
        """Save current progress to file."""
        try:
            progress_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_experiments': self.total_experiments,
                    'completed_experiments': self.completed_experiments,
                    'failed_experiments': len(self.failed_runs),
                    'success_rate': self.completed_experiments / max(self.total_experiments, 1) * 100,
                    'elapsed_time_sec': time.time() - self.start_time if self.start_time else 0
                },
                'results': self.results,
                'failed_runs': self.failed_runs
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            self.logger.info(f"💾 Progress saved: {len(self.results)} successful, {len(self.failed_runs)} failed")
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def run_parameter_sweep(self, dataset="didi", max_workers=None, max_tasks=None, scale="focused"):
        """
        Run the complete parameter sweep experiment.
        
        Args:
            dataset: Dataset to use ("didi", etc.)
            max_workers: Limit number of workers (None for all)
            max_tasks: Limit number of tasks (None for all) 
            scale: Parameter space scale ("test", "focused", "full")
        """
        
        self.start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BOTTLENECK PARAMETER SWEEP")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Dataset: {dataset}")
        self.logger.info(f"👥 Workers limit: {max_workers or 'ALL'}")
        self.logger.info(f"📋 Tasks limit: {max_tasks or 'ALL'}")
        self.logger.info(f"🎯 Scale: {scale}")
        self.logger.info(f"📁 Results file: {self.results_file}")
        self.logger.info(f"📝 Log file: {self.log_file}")
        
        try:
            # Load data - using object-based approach for memory efficiency
            self.logger.info("📊 Loading data...")
            workers, tasks = load_workers_tasks(dataset, limit_workers=None, limit_tasks=None)
            
            # MANUAL LIMIT ENFORCEMENT (data loaders don't respect limits!)
            if max_workers and len(workers) > max_workers:
                workers = workers[:max_workers]
            if max_tasks and len(tasks) > max_tasks:
                tasks = tasks[:max_tasks]
                
            self.logger.info(f"✅ Loaded {len(workers)} workers, {len(tasks)} tasks")
            
            # Generate parameter combinations
            self.logger.info("🎛️ Generating parameter combinations...")
            param_space = self.define_parameter_space(scale)
            param_combinations = self.generate_parameter_combinations(param_space)
            
            self.total_experiments = len(param_combinations)
            self.logger.info(f"🧪 Total experiments: {self.total_experiments}")
            
            # Estimate duration  
            if scale == "test":
                est_time_per_exp = 5  # seconds
            elif scale == "focused":
                est_time_per_exp = 15  # seconds  
            else:
                est_time_per_exp = 30  # seconds
                
            estimated_duration = self.total_experiments * est_time_per_exp / 3600  # hours
            self.logger.info(f"⏱️ Estimated duration: {estimated_duration:.1f} hours")
            
            # Run experiments
            self.logger.info("🧪 Starting experiments...")
            
            for i, params in enumerate(param_combinations):
                experiment_id = f"{self.timestamp}_{i+1:04d}"
                
                # Progress logging
                if i % 10 == 0 or i == 0:
                    progress_pct = (i / self.total_experiments) * 100
                    elapsed_hours = (time.time() - self.start_time) / 3600
                    remaining_exp = self.total_experiments - i
                    if i > 0:
                        avg_time_per_exp = (time.time() - self.start_time) / i
                        eta_hours = (remaining_exp * avg_time_per_exp) / 3600
                        self.logger.info(f"📈 Progress: {i}/{self.total_experiments} ({progress_pct:.1f}%) | "
                                       f"Elapsed: {elapsed_hours:.1f}h | ETA: {eta_hours:.1f}h")
                
                # Run experiment
                result = self.run_single_experiment(workers, tasks, params, experiment_id)
                
                if result:
                    self.results.append(result)
                    self.completed_experiments += 1
                
                # Save progress every 25 experiments
                if (i + 1) % 25 == 0:
                    self.save_progress()
                    
            # Final save
            self.save_progress()
            
            # Summary statistics
            self.logger.info("=" * 80)
            self.logger.info("🎉 PARAMETER SWEEP COMPLETE!")
            self.logger.info("=" * 80)
            
            total_time_hours = (time.time() - self.start_time) / 3600
            self.logger.info(f"⏱️ Total time: {total_time_hours:.2f} hours")
            self.logger.info(f"✅ Successful experiments: {len(self.results)}")
            self.logger.info(f"❌ Failed experiments: {len(self.failed_runs)}")
            self.logger.info(f"📈 Success rate: {(len(self.results) / self.total_experiments) * 100:.1f}%")
            
            if self.results:
                # Find best TAR
                best_tar = max(self.results, key=lambda x: x['metrics']['tar_percent'])
                self.logger.info(f"🏆 Best TAR: {best_tar['metrics']['tar_percent']:.2f}%")
                self.logger.info(f"🏆 Best parameters: {best_tar['parameters']}")
                
                # TAR distribution
                tar_values = [r['metrics']['tar_percent'] for r in self.results]
                self.logger.info(f"📊 TAR range: {min(tar_values):.2f}% - {max(tar_values):.2f}%")
                self.logger.info(f"📊 TAR mean ± std: {np.mean(tar_values):.2f}% ± {np.std(tar_values):.2f}%")
            
            self.logger.info(f"💾 Results saved to: {self.results_file}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"💥 Parameter sweep failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

def main():
    """Main entry point with different run modes."""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "test"
    
    sweep = BottleneckParameterSweep()
    
    if mode == "test":
        print("🧪 Running TEST mode (TINY dataset, minimal parameters)")
        sweep.run_parameter_sweep(
            dataset="checkins", 
            max_workers=10, 
            max_tasks=20, 
            scale="test"
        )
    elif mode == "focused":
        print("🎯 Running FOCUSED mode (medium dataset, key parameters)")
        sweep.run_parameter_sweep(
            dataset="didi", 
            max_workers=1000, 
            max_tasks=2000, 
            scale="focused"
        )
    elif mode == "full":
        print("🚀 Running FULL mode (large dataset, comprehensive parameters)")
        sweep.run_parameter_sweep(
            dataset="didi", 
            max_workers=5000, 
            max_tasks=10000, 
            scale="full"
        )
    elif mode == "background":
        print("⏰ Running BACKGROUND mode (6-7 hour experiment)")
        sweep.run_parameter_sweep(
            dataset="didi", 
            max_workers=2500, 
            max_tasks=5000, 
            scale="focused"
        )
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: test, focused, full, background")
        sys.exit(1)

if __name__ == "__main__":
    main()
