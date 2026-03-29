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
import random

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params

class AdaptiveSpatialCrowdsourcingEnv(gym.Env):
    """
    Gymnasium environment for RL-based control of spatial crowdsourcing strategy weights.
    
    The agent observes the system state (backlog, fairness, etc.) and outputs
    continuous weights (λ1, λ2) for the composite scoring function.
    λ3 (Utility/Distance) is fixed at 1.0 as the "Unit Anchor", reducing action space to 2D.
    
    Uses normalized weight space: all weights are scaled by 4.0 from original physics tuning.
    This makes training more stable for neural networks and improves interpretability.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, dataset="didi", step_duration_minutes=5, reward_weights=None, 
                 data_root=None, day_folders=None, warmup_duration_minutes=30, 
                 episode_duration_hours=8, **kwargs):
        """
        Initialize the environment.
        """
        super().__init__()
        
        self.dataset = dataset
        self.step_duration = step_duration_minutes * 60  # seconds
        self.reward_weights = reward_weights or [1.0, 1.0, 1.0]
        self.data_root = data_root
        self.day_folders = day_folders
        
        # Warmup and episode configuration
        self.warmup_duration_seconds = warmup_duration_minutes * 60  # 30 minutes warmup
        self.episode_duration_seconds = episode_duration_hours * 60 * 60  # RL phase length
        self.episode_end_time = None  # Will be set in reset()
        
        # For dynamic loading: Load ONE day initially just to define observation space shape
        if self.day_folders:
            dummy_day = self.day_folders[0]
            workers, tasks = self._load_day_data(dummy_day)
            print(f"Initialized with {len(workers)} workers, {len(tasks)} tasks (from {dummy_day})")
        else:
            workers, tasks = load_workers_tasks(dataset)
            print(f"Loaded {len(workers)} workers, {len(tasks)} tasks")
        
        self.workers = workers
        self.tasks = tasks
        
        # Define Action Space: Continuous [λ1, λ2]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), 
                                        high=np.array([2.0, 0.5], dtype=np.float32), 
                                        dtype=np.float32)
        
        # Fetch defaults from config.py to ensure a single source of truth
        composite_defaults = get_strategy_params('composite')
        
        # Fixed Anchors pulled directly from config
        self.lambda3_fixed = composite_defaults['utility_weight']
        self.gamma_fixed = composite_defaults['gamma']
        self.k_fixed = composite_defaults['k']
        self.threshold_fixed = composite_defaults['soft_threshold']
        
        # Define Observation Space (19 Scalars)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        
        self.simulator = None
        self.current_step_idx = 0
        self.episode_count = 0
        _cd = get_strategy_params("composite")
        self.last_action = np.array(
            [float(_cd["fairness_weight"]), float(_cd["starvation_weight"])], dtype=np.float32
        )
        
        # Delta Tracking Memory
        self.prev_jfi = 1.0
        self.prev_wait = 0.0
        self.prev_delay = 0.0
        self.prev_backlog = 0.0
        self.prev_arrival_rate = 0.0
        
        # Initialize a temporary sim to get spaces
        config = get_simulation_config()
        config['assignment_strategy'] = 'composite'
        _sp = get_strategy_params("composite")
        _sp["normalize_scores"] = True
        _sp["enable_deferral_tracking"] = False
        config["strategy_params"] = _sp
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=config)
        self.simulator.reset()
        
        # Baseline for normalization
        self.baseline_wait_time = 3.82  
        self.baseline_backlog = 1285    
        self.greedy_baseline_jfi = 0.5  # dynamically set in reset() according to the greedy baseline
        
    def _load_day_data(self, day_folder):
        if self.data_root:
            if not os.path.isabs(self.data_root):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_root_clean = self.data_root.lstrip('./')
                data_root_abs = os.path.join(project_root, data_root_clean)
            else:
                data_root_abs = self.data_root
            
            full_path = os.path.join(data_root_abs, day_folder)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Day folder not found: {full_path}")
        else:
            full_path = None
        
        return load_workers_tasks(self.dataset, root_path=full_path)
        
    def reset(self, seed=None, options=None):
        options = options or {}
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1. Load Data (Random Day)
        if self.day_folders:
            selected_day = random.choice(self.day_folders)
            self.workers, self.tasks = self._load_day_data(selected_day)
        else:
            self.workers, self.tasks = load_workers_tasks(self.dataset)
        
        # 2. Determine Time Window (Random Drop-In)
        if not self.tasks:
            raise ValueError("No tasks loaded - cannot determine time window")
        
        earliest = min(t.release_time for t in self.tasks)
        latest = max(t.release_time for t in self.tasks)
        
        total_duration_needed = self.warmup_duration_seconds + self.episode_duration_seconds
        max_start = latest - total_duration_needed
        
        if options.get("start_time") is not None:
            start_time = float(options["start_time"])
        elif max_start < earliest:
            start_time = earliest
        else:
            start_time = random.uniform(earliest, max_start)
        self._eval_drop_in_start_time = start_time
        
        # 3. Initialize Simulator in 'GREEDY' mode for Warmup
        warmup_config = get_simulation_config()
        warmup_config['assignment_strategy'] = 'greedy'
        warmup_config['strategy_params'] = {
            'enable_deferral_tracking': False
        }
        
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=warmup_config)
        self.simulator.reset(start_time=start_time)
        
        # 4. Run Warmup (Pure Greedy)
        self.simulator.step(duration_seconds=self.warmup_duration_seconds)
        
        # --- NEW: Capture the Dynamic Greedy Baseline ---
        warmup_stats = self.simulator.metrics.current_step_stats
        # Set a minimum floor of 0.40 just in case the warmup had a total anomaly
        self.greedy_baseline_jfi = max(0.40, warmup_stats.get('jfi', 0.5))
        
        # 5. Handover to composite (same params as config.py baseline for fair eval vs static)
        rl_params = get_strategy_params('composite')
        rl_params['normalize_scores'] = True
        rl_params['enable_deferral_tracking'] = True
        self.simulator.switch_strategy('composite', rl_params)
        
        self.current_step_idx = 0
        self.last_action = np.array(
            [float(rl_params['fairness_weight']), float(rl_params['starvation_weight'])],
            dtype=np.float32,
        )
        
        # Reset Delta Tracking Memory for new episode
        self.prev_jfi = 1.0
        self.prev_wait = 0.0
        self.prev_delay = 0.0
        self.prev_backlog = 0.0
        self.prev_arrival_rate = 0.0
        
        # 6. Set Hard End Time
        self.episode_end_time = self.simulator.current_time + self.episode_duration_seconds
        
        self.episode_count += 1
        print(f"   ▶️ [Episode {self.episode_count}] Started RL Phase | Tasks: {len(self.tasks):,} | Workers: {len(self.workers):,}")
        
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action):
        # Flatten action for DummyVecEnv compatibility
        action = np.ravel(action)
        lambda1, lambda2 = action
        lambda3 = self.lambda3_fixed
        
        # 1. Apply action
        self.simulator.strategy_params['fairness_weight'] = float(lambda1)
        self.simulator.strategy_params['starvation_weight'] = float(lambda2)
        self.simulator.strategy_params['utility_weight'] = float(lambda3)
        
        self.last_action = action
        
        # 2. Run simulation
        done = self.simulator.step(duration_seconds=self.step_duration)
        self.current_step_idx += 1
        
        # 3. Check termination
        if self.episode_end_time and self.simulator.current_time >= self.episode_end_time:
            done = True
            
        terminated = done
        truncated = False
        
        # 4. Get observation & calculate reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        info = {
            'step': self.current_step_idx,
            'lambdas': [lambda1, lambda2, lambda3],
            'backlog': self.simulator.metrics.summary.get('backlog_peak', 0),
            'completed': self.simulator.metrics.summary.get('completed_tasks', 0),
            'current_time': self.simulator.current_time,
            'episode_end_time': self.episode_end_time
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self):
        """
        Extract 19 features from simulator state, including critical trend deltas.
        """
        sim_obs = self.simulator.metrics.get_observation_data(self.simulator.state, self.simulator.current_time)
        
        # Current Values
        curr_jfi = sim_obs['jfi']
        curr_wait = sim_obs['step_avg_wait']
        curr_delay = sim_obs['step_avg_assignment_delay']
        curr_backlog = sim_obs['backlog_peak']
        curr_arrival = sim_obs['task_arrival_rate']
        
        # Calculate Deltas
        delta_jfi = curr_jfi - self.prev_jfi
        delta_wait = curr_wait - self.prev_wait
        delta_delay = curr_delay - self.prev_delay
        delta_backlog = curr_backlog - self.prev_backlog
        delta_arrival = curr_arrival - self.prev_arrival_rate
        
        # Update Memory for next step
        self.prev_jfi = curr_jfi
        self.prev_wait = curr_wait
        self.prev_delay = curr_delay
        self.prev_backlog = curr_backlog
        self.prev_arrival_rate = curr_arrival
        
        # curr_delay and delta_delay are in seconds; baseline_wait_time is in minutes - convert
        delay_min = curr_delay / 60.0
        delta_delay_min = delta_delay / 60.0
            
        obs = np.array([
            sim_obs['deferred_ratio'],                          # 0
            sim_obs['worker_availability_ratio'],               # 1
            sim_obs['total_workers'] / 10000.0,                 # 2. Perfect 10k normalization
            curr_backlog / self.baseline_backlog,               # 3
            curr_jfi,                                           # 4
            delta_jfi,                                          # 5. Momentum
            curr_wait / self.baseline_wait_time,                # 6
            delta_wait / self.baseline_wait_time,               # 7. Momentum
            delay_min / self.baseline_wait_time,                # 8. Time to Match (minutes)
            delta_delay_min / self.baseline_wait_time,          # 9. Momentum
            delta_backlog / self.baseline_backlog,              # 10. Momentum
            delta_arrival,                                      # 11. Spike Indicator
            sim_obs['is_midweek'],                              # 12
            sim_obs['is_mon_fri'],                              # 13
            sim_obs['is_weekend'],                              # 14
            sim_obs['time_sin'],                                # 15
            sim_obs['time_cos'],                                # 16
            self.last_action[0],                                # 17
            self.last_action[1]                                 # 18
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self):
        stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)

        # 1. The Dynamic Fairness Anchor (Exponential Curve)
        current_jfi = stats['fairness']
        jfi_improvement = current_jfi - self.greedy_baseline_jfi

        if jfi_improvement >= 0:
            r_fairness = jfi_improvement * 20.0
        else:
            r_fairness = -10.0 * (np.exp(abs(jfi_improvement) * 5.0) - 1.0)

        # 2. The Efficiency Anchor
        r_latency = -stats['latency'] / 5.0

        # 3. The Starvation/Expiration Anchor (Relaxed & Capped)
        # Denominator increased to 20.0 to relax sensitivity.
        # Penalty is capped at -5.0 to ensure JFI/Latency dominance.
        r_starvation = -min(3.0, stats['recent_expirations'] / 20)

        reward = (self.reward_weights[0] * r_fairness) + \
                 (self.reward_weights[1] * r_starvation) + \
                 (self.reward_weights[2] * r_latency)

        return float(reward)
