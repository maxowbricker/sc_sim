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
from config import (
    get_simulation_config,
    get_strategy_params,
    get_observation_static_scaling,
)

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
        
        # Define Action Space: Symmetric [-1, 1] × [-1, 1]
        # Mapped in step() to physical ranges:
        #   λ1: [-1, 1] → [0.0, 2.0]  (network output 0 → λ1 = 1.0, the Optuna optimum)
        #   λ2: [-1, 1] → [0.0, 0.5]  (network output 0 → λ2 = 0.25)
        # Symmetric space prevents the policy from sticking to the lower boundary at init.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Fetch defaults from config.py to ensure a single source of truth
        composite_defaults = get_strategy_params('composite')
        
        # Fixed Anchors pulled directly from config
        self.lambda3_fixed = composite_defaults['utility_weight']
        self.gamma_fixed = composite_defaults['gamma']
        self.k_fixed = composite_defaults['k']
        self.threshold_fixed = composite_defaults['soft_threshold']
        
        # 15 scalars: last_action removed from obs to break self-fulfilling-prophecy collapse
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
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
        self.prev_backlog = 0.0
        self.prev_arrival_rate = 0.0

        # Delta JFI tracking for reward (prev_jfi tracks RL; oracle uses same start point)
        # Stored each step before _get_observation() updates self.prev_jfi
        self._delta_jfi_rl = 0.0
        self._delta_jfi_oracle = 0.0

        # Oracle Reward Stats (will be set in step())
        self.oracle_reward_stats = None
        
        # Initialize a temporary sim to get spaces
        config = get_simulation_config()
        config['assignment_strategy'] = 'composite'
        _sp = get_strategy_params("composite")
        _sp["normalize_scores"] = True
        _sp["enable_deferral_tracking"] = False
        config["strategy_params"] = _sp
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=config)
        self.simulator.reset()
        
        self.greedy_baseline_jfi = 0.5  # dynamically set in reset() according to the greedy baseline
        self.obs_scaling = get_observation_static_scaling()

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
        
        # Capture the JFI of the greedy baseline
        warmup_stats = self.simulator.metrics.current_step_stats
        # Compare RL to the exact warmup physics (no artificial floor)
        self.greedy_baseline_jfi = warmup_stats.get('jfi', 0.5)

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
        self.prev_backlog = 0.0
        self.prev_arrival_rate = 0.0
        self._delta_jfi_rl = 0.0
        self._delta_jfi_oracle = 0.0
        
        # 6. Set Hard End Time
        self.episode_end_time = self.simulator.current_time + self.episode_duration_seconds
        
        self.episode_count += 1
        print(f"   ▶️ [Episode {self.episode_count}] Started RL Phase | Tasks: {len(self.tasks):,} | Workers: {len(self.workers):,}")
        
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action):
        # Flatten action for DummyVecEnv compatibility
        action = np.ravel(action)
        # Map symmetric [-1, 1] action to physical weight ranges:
        #   λ1: [-1, 1] → [0.0, 2.0]   (init output 0 → λ1 = 1.0, the Optuna optimum)
        #   λ2: [-1, 1] → [0.0, 0.5]   (init output 0 → λ2 = 0.25)
        lambda1 = float(np.clip(action[0], -1.0, 1.0)) + 1.0
        lambda2 = (float(np.clip(action[1], -1.0, 1.0)) + 1.0) * 0.25
        lambda3 = self.lambda3_fixed

        self.last_action = np.array([lambda1, lambda2], dtype=np.float32)

        # 1. Apply action
        self.simulator.strategy_params['fairness_weight'] = lambda1
        self.simulator.strategy_params['starvation_weight'] = lambda2
        self.simulator.strategy_params['utility_weight'] = lambda3
        
        # 2. Run Oracle (Greedy Baseline) to get "reality check" stats
        oracle_snap = self.simulator.snapshot_state()
        self.simulator.switch_strategy('greedy')
        self.simulator.step(duration_seconds=self.step_duration)
        oracle_stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)
        self.oracle_reward_stats = oracle_stats  # Store for reward calculation
        
        # 3. Restore to pre-oracle state
        self.simulator.restore_state(oracle_snap)
        
        # 4. Run simulation with composite strategy
        self.simulator.switch_strategy('composite', self.simulator.strategy_params)
        done = self.simulator.step(duration_seconds=self.step_duration)
        self.current_step_idx += 1

        # 5. Compute delta JFI BEFORE _get_observation() updates self.prev_jfi
        # Both RL and oracle start from the same snapshot, so they share the same prev_jfi baseline.
        _rl_stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)
        self._delta_jfi_rl     = _rl_stats['fairness']     - self.prev_jfi
        self._delta_jfi_oracle = oracle_stats['fairness']   - self.prev_jfi
        
        # 6. Check termination
        if self.episode_end_time and self.simulator.current_time >= self.episode_end_time:
            done = True
            
        terminated = done
        truncated = False
        
        # 7. Get observation & calculate reward
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
        Extract 15 features from simulator state.
        last_action removed: including it creates a self-fulfilling prophecy where
        the agent observes λ1=0 every step and never explores non-zero values.
        Scaling: config.get_observation_static_scaling() / OBSERVATION_STATIC_SCALING.
        """
        sim_obs = self.simulator.metrics.get_observation_data(self.simulator.state, self.simulator.current_time)
        
        curr_jfi = sim_obs['jfi']
        curr_wait = sim_obs['step_avg_wait']
        curr_backlog = sim_obs['backlog_peak']
        curr_arrival = sim_obs['task_arrival_rate']
        
        delta_jfi = curr_jfi - self.prev_jfi
        delta_wait = curr_wait - self.prev_wait
        delta_backlog = curr_backlog - self.prev_backlog
        delta_arrival = curr_arrival - self.prev_arrival_rate
        
        self.prev_jfi = curr_jfi
        self.prev_wait = curr_wait
        self.prev_backlog = curr_backlog
        self.prev_arrival_rate = curr_arrival

        _o = self.obs_scaling
        eps = 1e-8

        obs = np.array([
            sim_obs['deferred_ratio'],                              # 0
            sim_obs['worker_availability_ratio'],                   # 1
            sim_obs['total_workers'] / max(_o["worker_count_divisor"], eps),  # 2
            curr_backlog / max(_o["ref_backlog"], eps),             # 3
            curr_jfi,                                               # 4
            delta_jfi / max(_o["max_abs_jfi_delta"], eps),          # 5
            curr_wait / max(_o["ref_wait_minutes"], eps),           # 6
            delta_wait / max(_o["max_abs_wait_delta"], eps),        # 7
            delta_backlog / max(_o["max_abs_backlog_delta"], eps),  # 8
            delta_arrival / max(_o["max_abs_arrival_delta"], eps),  # 9
            sim_obs['is_midweek'],                                  # 10
            sim_obs['is_mon_fri'],                                  # 11
            sim_obs['is_weekend'],                                  # 12
            sim_obs['time_sin'],                                    # 13
            sim_obs['time_cos'],                                    # 14
        ], dtype=np.float32)
        
        return obs

    def _calculate_reward(self):
        """
        Delta-JFI Advantage vs Greedy Oracle.

        Uses STEP-OVER-STEP JFI change rather than absolute JFI level.
        This fixes the credit-assignment problem: tasks assigned under a high-fairness λ1
        take several steps to complete and show up in JFI. Absolute JFI advantage is dominated
        by historical assignments and gives no gradient to the current action. Delta JFI
        captures the momentum signal attributable to *this* step's action.

        Both RL and oracle start from the same snapshot, so their delta-JFI baselines are
        identical — the advantage is purely about which strategy moved JFI more this step.

        - Fairness (delta): Heavily scaled (200x). Positive = RL improved JFI more than oracle.
        - Latency: Asymmetric cap — no bonus for beating oracle, only penalty when slower.
        - Starvation: Asymmetric cap — no bonus for fewer expirations, only penalty if more.
        """
        composite_stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)
        oracle_stats = self.oracle_reward_stats or composite_stats

        # 1. DELTA JFI ADVANTAGE (The Carrot — immediate, step-attributable signal)
        # self._delta_jfi_rl and self._delta_jfi_oracle computed in step() before obs update.
        fairness_delta_adv = self._delta_jfi_rl - self._delta_jfi_oracle
        r_fairness = fairness_delta_adv * 200.0

        # 2. ASYMMETRIC LATENCY (The Stick)
        latency_adv = oracle_stats['latency'] - composite_stats['latency']
        r_latency = min(0.0, latency_adv) * 5.0

        # 3. ASYMMETRIC STARVATION
        starvation_adv = oracle_stats['recent_expirations'] - composite_stats['recent_expirations']
        r_starvation = min(0.0, starvation_adv) * 1.0

        reward = r_fairness + r_latency + r_starvation
        return float(reward / 5.0)
