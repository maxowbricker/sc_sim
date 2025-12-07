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
        # 0. Active Task Ratio (active / total_possible)
        # 1. Deferred Task Ratio
        # 2. Worker Availability Ratio
        # 3. Current JFI (Fairness)
        # 4. Step Average Wait Time (Normalized)
        # 5. Peak Backlog (Normalized)
        # 6. Time of Day (Sine)
        # 7. Time of Day (Cosine)
        # 8. Previous λ1
        # 9. Previous λ2
        # 10. Previous λ3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.simulator = None
        self.last_state = None
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
        
        active_tasks = sim_state['active_tasks']
        deferred_tasks = sim_state['deferred_tasks']
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
            
        obs = np.array([
            active_tasks / 100.0, # Approximate normalization
            deferred_tasks / 100.0,
            available_workers / total_workers,
            jfi,
            step_avg_wait / self.baseline_wait_time,
            sim_state['backlog_peak'] / self.baseline_backlog,
            time_sin,
            time_cos,
            self.last_action[0],
            self.last_action[1],
            self.last_action[2]
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
