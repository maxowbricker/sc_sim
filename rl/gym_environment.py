"""
Gymnasium Environment for Adaptive Spatial Crowdsourcing.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator
from config import get_simulation_config, create_composite_config

class AdaptiveSpatialCrowdsourcingEnv(gym.Env):
    """
    Gymnasium environment for RL-based control of spatial crowdsourcing strategy weights.
    
    The agent observes the system state (backlog, fairness, etc.) and outputs
    continuous weights (λ1, λ3) for the composite scoring function.
    λ2 is fixed at 0.5 based on empirical findings, reducing action space to 2D.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, dataset="didi", step_duration_minutes=15, reward_weights=None):
        """
        Initialize the environment.
        
        Args:
            dataset: Name of the dataset to load (e.g., 'didi').
            step_duration_minutes: Duration of each simulation step in minutes.
            reward_weights: Weights for reward components [fairness, starvation, throughput].
        """
        super().__init__()
        
        self.dataset = dataset
        self.step_duration = step_duration_minutes * 60  # seconds
        self.reward_weights = reward_weights or [1.0, 1.0, 1.0]
        
        # Load data once
        self.workers, self.tasks = load_workers_tasks(dataset)
        print(f"Loaded {len(self.workers)} workers, {len(self.tasks)} tasks")
        
        # Define Action Space: Continuous [λ1, λ3]
        # λ2 is fixed at 0.5 based on empirical findings
        # We limit the range to reasonable values, e.g., [0, 5.0]
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(2,), dtype=np.float32)
        self.lambda2_fixed = 0.5  # Fixed value for λ2
        
        # Define Observation Space
        # Features:
        # 0. Deferred Tasks Ratio (deferred / total_tasks_released)
        # 1. Worker Availability Ratio
        # 2. Current JFI (Fairness)
        # 3. Step Average Wait Time (Normalized)
        # 4. Peak Backlog (Normalized)
        # 5. Task Release Rate per Worker (tasks/min/worker)
        # 6. Mean Worker Idle Time (normalized)
        # 7. Worker Idle Time Inequality (CV)
        # 8. % Deferrals due to Low Score (below_threshold)
        # 9. % Deferrals due to No Candidates
        # 10. Time of Day (Sine)
        # 11. Time of Day (Cosine)
        # 12. Previous λ1
        # 13. Previous λ3
        # (λ2 is fixed at 0.5, so not included in observation)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        self.simulator = None
        self.current_step_idx = 0
        self.last_action = np.array([1.0, 1.0], dtype=np.float32)  # [λ1, λ3] only
        
        # Baseline for normalization (from baseline_metrics_summary_20251211_165426.json)
        # Values averaged from two configs: λ1=4.0,λ3=4.0 and λ1=5.0,λ3=3.0
        # Both used same sampling: 4000 workers, 20000 tasks, stratified temporal (12 bins, seed=42)
        self.baseline_wait_time = 3.82  # minutes (avg from baseline simulations)
        self.baseline_backlog = 1285    # tasks (avg peak backlog from baseline simulations)
        self.baseline_worker_idle = 146.07  # minutes (avg worker idle time from baseline simulations)
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        """
        super().reset(seed=seed)
        
        # Initialize simulator
        # We use 'composite' strategy by default for RL control
        config = get_simulation_config()
        config['assignment_strategy'] = 'composite'
        config['strategy_params'] = {
            'λ1': 1.0, 'λ2': 1.0, 'λ3': 1.0,
            'enable_deferral_tracking': True # Needed for state
        }
        
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=config)
        self.simulator.reset()
        
        self.current_step_idx = 0
        self.last_action = np.array([1.0, 1.0], dtype=np.float32)  # [λ1, λ3] only
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: Array of shape (2,) containing [λ1, λ3]. λ2 is fixed at 0.5.
        """
        # 1. Apply action (update weights)
        # Action is [λ1, λ3], we fix λ2 at 0.5
        lambda1, lambda3 = action
        lambda2 = self.lambda2_fixed
        self.simulator.update_weights(lambda1, lambda2, lambda3)
        self.last_action = action  # Store [λ1, λ3] for observation
        
        # 2. Run simulation for fixed duration
        done = self.simulator.step(duration_seconds=self.step_duration)
        self.current_step_idx += 1
        
        # 3. Get observation
        obs = self._get_observation()
        
        # 4. Calculate reward
        reward = self._calculate_reward()
        
        # 5. Check termination
        terminated = done
        truncated = False # We rely on simulator completion
        
        info = {
            'step': self.current_step_idx,
            'lambdas': [lambda1, lambda2, lambda3],  # Full [λ1, λ2, λ3] for logging
            'backlog': self.simulator.summary.get('backlog_peak', 0),
            'completed': self.simulator.summary.get('completed_tasks', 0)
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """
        Extract features from simulator state.
        
        Now uses MetricsManager for unified metric calculation.
        """
        sim_state = self.simulator.get_state()
        
        # Get pre-calculated metrics from MetricsManager
        deferred_ratio = sim_state.get('deferred_ratio', 0.0)
        worker_availability_ratio = sim_state.get('worker_availability_ratio', 0.0)
        jfi = sim_state.get('jfi', 1.0)
        step_avg_wait = sim_state.get('step_avg_wait', 0.0)
        backlog_peak = sim_state.get('backlog_peak', 0)
        task_worker_ratio = sim_state.get('task_worker_ratio', 0.0)
        mean_worker_idle = sim_state.get('mean_worker_idle_min', 0.0)
        cv_worker_idle = sim_state.get('cv_worker_idle', 0.0)
        pct_deferrals_below_threshold = sim_state.get('pct_deferrals_below_threshold', 0.0)
        pct_deferrals_no_candidates = sim_state.get('pct_deferrals_no_candidates', 0.0)
        
        # Time encoding (from sim_state)
        time_sin = sim_state.get('time_sin', 0.0)
        time_cos = sim_state.get('time_cos', 0.0)
        
        # Normalize worker idle time
        normalized_idle = mean_worker_idle / self.baseline_worker_idle
            
        obs = np.array([
            deferred_ratio,  # 0. Deferred Tasks Ratio
            worker_availability_ratio,  # 1. Worker Availability Ratio
            jfi,  # 2. Current JFI (Fairness)
            step_avg_wait / self.baseline_wait_time,  # 3. Step Average Wait Time (Normalized)
            backlog_peak / self.baseline_backlog,  # 4. Peak Backlog (Normalized)
            task_worker_ratio,  # 5. Task Release Rate per Worker (tasks/min/worker)
            normalized_idle,  # 6. Mean Worker Idle Time (normalized)
            cv_worker_idle,  # 7. Worker Idle Time Inequality (CV)
            pct_deferrals_below_threshold,  # 8. % Deferrals due to Low Score
            pct_deferrals_no_candidates,  # 9. % Deferrals due to No Candidates
            time_sin,  # 10. Time of Day (Sine)
            time_cos,  # 11. Time of Day (Cosine)
            self.last_action[0],  # 12. Previous λ1
            self.last_action[1]  # 13. Previous λ3
            # (λ2 is fixed at 0.5, so not included in observation)
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self):
        """
        Calculate reward based on recent performance.
        
        Now uses MetricsManager for unified metric calculation - no more duplicate JFI calculation!
        """
        # Get reward stats from MetricsManager (single source of truth)
        reward_stats = self.simulator.metrics.get_reward_stats()
        
        fairness = reward_stats['fairness']  # JFI
        throughput = reward_stats['throughput']  # Negative backlog
        latency = reward_stats['latency']  # Negative wait time
        
        # Normalize components to be roughly in same magnitude
        # JFI is [0, 1]. Target is to boost it.
        # Backlog is [0, 500+].
        # Wait time is [0, 30+].
        
        r_fairness = (fairness - 0.5) * 10.0  # Range [-5, 5]
        r_throughput = throughput / 100.0  # Range [-5, 0] (throughput is already negative)
        r_latency = latency / 5.0  # Range [-6, 0] (latency is already negative)
        
        reward = (self.reward_weights[0] * r_fairness + 
                  self.reward_weights[1] * r_throughput + 
                  self.reward_weights[2] * r_latency)
                  
        return reward
