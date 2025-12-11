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
    continuous weights (λ1, λ2, λ3) for the composite scoring function.
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
        
        # Define Action Space: Continuous [λ1, λ2, λ3]
        # We limit the range to reasonable values, e.g., [0, 5.0]
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(3,), dtype=np.float32)
        
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
        # 13. Previous λ2
        # 14. Previous λ3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.simulator = None
        self.current_step_idx = 0
        self.last_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        # Baseline for normalization (approximate)
        self.baseline_wait_time = 10.0 # minutes
        self.baseline_backlog = 100
        
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
        self.last_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        """
        # 1. Apply action (update weights)
        lambda1, lambda2, lambda3 = action
        self.simulator.update_weights(lambda1, lambda2, lambda3)
        self.last_action = action
        
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
            'lambdas': action,
            'backlog': self.simulator.summary.get('backlog_peak', 0),
            'completed': self.simulator.summary.get('completed_tasks', 0)
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """
        Extract features from simulator state.
        """
        sim_state = self.simulator.get_state()
        
        deferred_tasks = sim_state['deferred_tasks']
        total_tasks_released = sim_state['total_tasks_released']
        available_workers = sim_state['available_workers']
        total_workers = max(1, sim_state['total_workers'])
        
        # Time encoding
        current_time = sim_state['current_time']
        hour = current_time.hour + current_time.minute / 60.0
        time_sin = np.sin(2 * np.pi * hour / 24.0)
        time_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # Fairness (JFI) - calculating from current workers
        workers = sim_state['workers']
        if workers:
            incomes = [w.revenue for w in workers]
            nom = np.sum(incomes)**2
            denom = len(workers) * np.sum(np.array(incomes)**2)
            jfi = nom / denom if denom > 0 else 0
        else:
            jfi = 0
            
        # Windowed Wait time
        step_avg_wait = sim_state.get('step_avg_wait', 0.0)
        
        # Deferred tasks ratio: deferred / total_tasks_released
        deferred_ratio = deferred_tasks / max(1, total_tasks_released)
        
        # NEW: Enhanced metrics
        task_worker_ratio = sim_state.get('task_worker_ratio', 0.0)
        mean_worker_idle = sim_state.get('mean_worker_idle_min', 0.0)
        cv_worker_idle = sim_state.get('cv_worker_idle', 0.0)
        pct_deferrals_below_threshold = sim_state.get('pct_deferrals_below_threshold', 0.0)
        pct_deferrals_no_candidates = sim_state.get('pct_deferrals_no_candidates', 0.0)
        
        # Normalize worker idle time
        normalized_idle = mean_worker_idle / self.baseline_worker_idle
            
        obs = np.array([
            deferred_ratio,  # 0. Deferred Tasks Ratio
            available_workers / total_workers,  # 1. Worker Availability Ratio
            jfi,  # 2. Current JFI (Fairness)
            step_avg_wait / self.baseline_wait_time,  # 3. Step Average Wait Time (Normalized)
            sim_state['backlog_peak'] / self.baseline_backlog,  # 4. Peak Backlog (Normalized)
            task_worker_ratio,  # 5. Task Release Rate per Worker (tasks/min/worker)
            normalized_idle,  # 6. Mean Worker Idle Time (normalized)
            cv_worker_idle,  # 7. Worker Idle Time Inequality (CV)
            pct_deferrals_below_threshold,  # 8. % Deferrals due to Low Score
            pct_deferrals_no_candidates,  # 9. % Deferrals due to No Candidates
            time_sin,  # 10. Time of Day (Sine)
            time_cos,  # 11. Time of Day (Cosine)
            self.last_action[0],  # 12. Previous λ1
            self.last_action[1],  # 13. Previous λ2
            self.last_action[2]  # 14. Previous λ3
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self):
        """
        Calculate reward based on recent performance.
        """
        sim_state = self.simulator.get_state()
        
        # Throughput proxy: Negative backlog (lower backlog is better)
        backlog = sim_state['active_tasks'] + sim_state['deferred_tasks']
        
        # Inequality proxy: JFI (higher is better)
        workers = sim_state['workers']
        if workers:
            incomes = [w.revenue for w in workers]
            nom = np.sum(incomes)**2
            denom = len(workers) * np.sum(np.array(incomes)**2)
            jfi = nom / denom if denom > 0 else 0
        else:
            jfi = 0
            
        # Latency proxy: Windowed Average wait time (lower is better)
        step_avg_wait = sim_state.get('step_avg_wait', 0.0)
            
        # Normalize components to be roughly in same magnitude
        # JFI is [0, 1]. Target is to boost it.
        # Backlog is [0, 500+].
        # Wait time is [0, 30+].
        
        r_fairness = (jfi - 0.5) * 10.0  # Range [-5, 5]
        r_throughput = -(backlog / 100.0) # Range [-5, 0]
        r_latency = -(step_avg_wait / 5.0) # Range [-6, 0]
        
        reward = (self.reward_weights[0] * r_fairness + 
                  self.reward_weights[1] * r_throughput + 
                  self.reward_weights[2] * r_latency)
                  
        return reward
