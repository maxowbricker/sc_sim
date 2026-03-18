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
from config import get_simulation_config, create_composite_config

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
                 episode_duration_hours=4, **kwargs):
        """
        Initialize the environment.
        
        Args:
            dataset: Name of the dataset to load (e.g., 'didi').
            step_duration_minutes: Duration of each simulation step in minutes (default: 5).
            reward_weights: Weights for reward components [fairness, starvation, throughput].
            data_root: Base path to dataset folders (e.g., './data/didi/full_didi_gaia').
                      If None, uses default dataset path.
            day_folders: List of folder names to randomly select from on each reset.
                        If None, uses single default dataset path.
            warmup_duration_minutes: Duration of warmup phase in minutes (default: 30).
            episode_duration_hours: Duration of RL episode after warmup in hours (default: 4).
        """
        super().__init__()
        
        self.dataset = dataset
        self.step_duration = step_duration_minutes * 60  # seconds
        self.reward_weights = reward_weights or [1.0, 1.0, 1.0]
        self.data_root = data_root
        self.day_folders = day_folders
        
        # Warmup and episode configuration
        self.warmup_duration_seconds = warmup_duration_minutes * 60  # 30 minutes warmup
        self.episode_duration_seconds = episode_duration_hours * 60 * 60  # 4 hour episodes
        self.episode_end_time = None  # Will be set in reset()
        
        # For dynamic loading: Load ONE day initially just to define observation space shape
        # (We don't keep this data, it's just for setup)
        if self.day_folders:
            # Use first day folder for initialization
            dummy_day = self.day_folders[0]
            workers, tasks = self._load_day_data(dummy_day)
            print(f"Initialized with {len(workers)} workers, {len(tasks)} tasks (from {dummy_day})")
        else:
            # Legacy mode: load data once
            workers, tasks = load_workers_tasks(dataset)
            print(f"Loaded {len(workers)} workers, {len(tasks)} tasks")
        
        # Store for initial space setup (will be replaced in reset())
        self.workers = workers
        self.tasks = tasks
        
        # Define Action Space: Continuous [λ1, λ2]
        # Agent controls [Fairness (λ1), Starvation (λ2)]
        # λ1 (Fairness): Range [0, 2] - can go from "0x" to "2x" importance relative to Distance
        # λ2 (Starvation): Range [0, 0.5] - constrained to lower values (tuned optimal is 0.225)
        # INITIALIZATION BONUS: PPO starts near the center for λ1 (1.0), which is your Thesis Optimal.
        # λ1: 0.0 = "I don't care about fairness", 1.0 = "Equal to Distance", 2.0 = "2x more important"
        # λ2: 0.0 = "I don't care about starvation", 0.225 = "Tuned optimal", 0.5 = "Max allowed"
        self.action_space = spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32), 
                                        high=np.array([2.0, 0.5], dtype=np.float32), 
                                        dtype=np.float32)
        
        # Fixed Anchors (Scaled down by 4.0 from original physics tuning)
        # λ3 (Utility/Distance) is the "Unit Anchor" - always 1.0
        self.lambda3_fixed = 1.0      # The Unit Anchor (normalized from 4.0)
        self.gamma_fixed = 0.1        # EWMA smoothing factor (from best_physics_params.json)
        self.k_fixed = 50             # Number of nearest workers (from best_physics_params.json)
        self.threshold_fixed = 0.3    # Soft threshold (scaled: 1.2 / 4.0 = 0.3)
        
        # --- NEW: SPATIAL GRID CONFIGURATION (STEP 1.1) ---
        # Chengdu 2nd Ring Road Bounding Box (Didi GAIA Standard)
        self.grid_size = 10
        self.min_lat = 30.65
        self.max_lat = 30.73
        self.min_lon = 104.04
        self.max_lon = 104.13
        
        # Pre-calculate cell sizes for O(1) snapping later
        self.lat_step = (self.max_lat - self.min_lat) / self.grid_size
        self.lon_step = (self.max_lon - self.min_lon) / self.grid_size
        # --------------------------------------------------
        
        # --- NEW: MULTI-INPUT OBSERVATION SPACE (STEP 1.2) ---
        # The agent now receives a Dictionary containing both Spatial (Grid) and Global (Scalar) data.
        self.observation_space = spaces.Dict({
            # The Spatial "Image": [Channels, Width, Height] -> [4, 10, 10]
            # Channel 0: Available Worker Density
            # Channel 1: Active Task Density
            # Channel 2: Predicted Future Supply (Drop-offs)
            # Channel 3: Starvation Heat Map (Sum of wait times)
            "spatial_grid": spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(4, self.grid_size, self.grid_size),
                dtype=np.float32
            ),
            
            # The Global "Context": The original 14 metrics (Time of day, JFI, etc.)
            "global_scalars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(14,),
                dtype=np.float32
            )
        })
        # -----------------------------------------------------
        
        self.simulator = None
        self.current_step_idx = 0
        self.last_action = np.array([1.0, 0.225], dtype=np.float32)  # [λ1, λ2] only (initialized: λ1=1.0 baseline, λ2=0.225 from tuned 0.9/4.0)
        
        # Initialize a temporary sim to get spaces (needed for observation_space definition)
        config = get_simulation_config()
        config['assignment_strategy'] = 'composite'
        config['strategy_params'] = {
            'fairness_weight': 1.0,   # Starts at Center of Action Space (Thesis Optimal)
            'starvation_weight': 0.225, # Start at the tuned ratio (0.9/4.0 from best_physics_params.json)
            'utility_weight': self.lambda3_fixed,
            'gamma': self.gamma_fixed,
            'k': self.k_fixed,
            'soft_threshold': self.threshold_fixed,
            'normalize_scores': True,
            'enable_deferral_tracking': False
        }
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=config)
        self.simulator.reset()  # Important to set up the state for dimensions
        
        # Baseline for normalization (from baseline_metrics_summary_20251211_165426.json)
        # Values averaged from two configs: λ1=4.0,λ3=4.0 and λ1=5.0,λ3=3.0
        # Both used same sampling: 4000 workers, 20000 tasks, stratified temporal (12 bins, seed=42)
        self.baseline_wait_time = 3.82  # minutes (avg from baseline simulations)
        self.baseline_backlog = 1285    # tasks (avg peak backlog from baseline simulations)
        self.baseline_worker_idle = 146.07  # minutes (avg worker idle time from baseline simulations)
        
    def _load_day_data(self, day_folder):
        """
        Helper to load a specific day's data on demand.
        
        Args:
            day_folder: Name of the folder (e.g., '496528674@qq.com_20161101')
            
        Returns:
            Tuple of (workers, tasks) lists
        """
        if self.data_root:
            # Construct full path to the day folder
            # Convert to absolute path to ensure correct resolution
            if not os.path.isabs(self.data_root):
                # If relative, make it relative to project root (where this file is in rl/, so go up 2 levels)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                # Remove leading ./ if present
                data_root_clean = self.data_root.lstrip('./')
                data_root_abs = os.path.join(project_root, data_root_clean)
            else:
                data_root_abs = self.data_root
            
            full_path = os.path.join(data_root_abs, day_folder)
            # Ensure path exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Day folder not found: {full_path}")
        else:
            # Fallback to default dataset path
            full_path = None
        
        # Use existing loader with the specific path
        return load_workers_tasks(self.dataset, root_path=full_path)
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state with Random Drop-In + Warmup strategy.
        
        Process:
        1. Load full day data (random day selection)
        2. Pick random start time T
        3. Run warmup phase (T -> T+30min) with Greedy strategy
        4. Switch to Composite strategy for RL training (T+30min -> T+4.5h)
        """
        super().reset(seed=seed)
        
        # 1. Load Data (Random Day)
        if self.day_folders:
            selected_day = random.choice(self.day_folders)
            # Optional: Uncomment for debugging
            # print(f"DEBUG: Resetting Env with Day {selected_day}")
            
            # Load data for the selected day (takes ~0.5s - 1s with optimizations)
            self.workers, self.tasks = self._load_day_data(selected_day)
        else:
            # Legacy mode: load data once
            self.workers, self.tasks = load_workers_tasks(self.dataset)
        
        # 2. Determine Time Window (Random Drop-In)
        # Find valid bounds from loaded data
        if not self.tasks:
            raise ValueError("No tasks loaded - cannot determine time window")
        
        earliest = min(t.release_time for t in self.tasks)
        latest = max(t.release_time for t in self.tasks)
        
        # Ensure we have enough runway for warmup + episode
        # Start time must be at least (Warmup + Episode) before the data ends
        total_duration_needed = self.warmup_duration_seconds + self.episode_duration_seconds
        max_start = latest - total_duration_needed
        
        if max_start < earliest:
            # Fallback for short datasets: just start at beginning
            start_time = earliest
            # print(f"⚠️  Dataset too short, starting at beginning")
        else:
            # Random drop-in: pick a random start time
            start_time = random.uniform(earliest, max_start)
        
        # Convert to datetime for display (if needed for debugging)
        # start_dt = pd.Timestamp.fromtimestamp(start_time)
        # print(f"🔥 Starting at: {start_dt} (Random Drop-In)")
        
        # 3. Initialize Simulator in 'GREEDY' mode for Warmup
        # Greedy is fast and establishes realistic worker positions
        warmup_config = get_simulation_config()
        warmup_config['assignment_strategy'] = 'greedy'
        warmup_config['strategy_params'] = {
            'enable_deferral_tracking': False  # Not needed for warmup
        }
        
        self.simulator = EventSimulator(self.workers, self.tasks, sim_config=warmup_config)
        
        # Reset to specific start time (Random Drop-In)
        self.simulator.reset(start_time=start_time)
        
        # 4. Run Warmup (The "Heuristic Warmup")
        # Greedy strategy moves workers around and assigns tasks
        # This eliminates the "Artificial Idle" spike from cold start
        # print(f"🔥 Warmup: {start_time} -> +{self.warmup_duration_seconds/60:.0f}m (Greedy)")
        self.simulator.step(duration_seconds=self.warmup_duration_seconds)
        
        # 5. Handover to RL (Hot-Swap Strategy)
        rl_params = {
            'fairness_weight': 1.0,   
            'starvation_weight': 0.225, 
            'utility_weight': self.lambda3_fixed,
            'gamma': self.gamma_fixed,
            'k': self.k_fixed,
            'soft_threshold': self.threshold_fixed,
            'normalize_scores': True,
            'enable_deferral_tracking': True  
        }
        self.simulator.switch_strategy('composite', rl_params)
        
        # Reset counters for the actual episode
        self.current_step_idx = 0
        self.last_action = np.array([1.0, 0.225], dtype=np.float32)  # [λ1, λ2] only
        
        # 6. Set Hard End Time for this episode
        # This prevents the episode from running beyond the intended duration
        self.episode_end_time = self.simulator.current_time + self.episode_duration_seconds
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
        
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: Array of shape (2,) containing [λ1, λ2]. λ3 is fixed at 1.0 (Unit Anchor).
        """
        # 1. Apply action (update weights)
        lambda1, lambda2 = action
        lambda3 = self.lambda3_fixed
        
        # Update the simulation physics engine directly using standard python floats
        self.simulator.strategy_params['fairness_weight'] = float(lambda1)
        self.simulator.strategy_params['starvation_weight'] = float(lambda2)
        self.simulator.strategy_params['utility_weight'] = float(lambda3)
        
        self.last_action = action  # Store [λ1, λ2] for observation
        
        # 2. Run simulation for fixed duration
        done = self.simulator.step(duration_seconds=self.step_duration)
        self.current_step_idx += 1
        
        # 3. Check explicit time termination (episode end time)
        if self.episode_end_time and self.simulator.current_time >= self.episode_end_time:
            done = True
        
        # 4. Get observation
        obs = self._get_observation()
        
        # 5. Calculate reward
        reward = self._calculate_reward()
        
        # 6. Check termination
        terminated = done
        truncated = False  # We rely on simulator completion or episode end time
        
        info = {
            'step': self.current_step_idx,
            'lambdas': [lambda1, lambda2, lambda3],  # Full [λ1, λ2, λ3] for logging
            'backlog': self.simulator.metrics.summary.get('backlog_peak', 0),
            'completed': self.simulator.metrics.summary.get('completed_tasks', 0),
            'current_time': self.simulator.current_time,
            'episode_end_time': self.episode_end_time
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

        global_scalars = np.array([
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
            self.last_action[1]  # 13. Previous λ2
        ], dtype=np.float32)

        # --- NEW: CALL THE SPATIAL ENGINE (STEP 2.2) ---
        spatial_grid = self._generate_spatial_grid()

        return {
            "spatial_grid": spatial_grid,
            "global_scalars": global_scalars
        }

    def _generate_spatial_grid(self) -> np.ndarray:
        """
        Generates a 4-channel spatial feature map (10x10) of the city state.
        Channel 0: Available Worker Density
        Channel 1: Active Task Density
        Channel 2: Future Supply (Busy Worker Drop-offs)
        Channel 3: Starvation Heat Map (Sum of wait times)
        """
        # Initialize an empty 4-channel grid (Shape: 4 x 10 x 10)
        grid = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Helper function for O(1) coordinate snapping
        def get_cell(lat, lon):
            x = int((lat - self.min_lat) / self.lat_step)
            y = int((lon - self.min_lon) / self.lon_step)
            # Bound check to ensure we don't crash if a coordinate is slightly outside the Ring Road
            x = max(0, min(x, self.grid_size - 1))
            y = max(0, min(y, self.grid_size - 1))
            return x, y

        # Channel 0: Available Supply
        for worker in self.simulator.state.available_workers:
            x, y = get_cell(worker.start_lat, worker.start_lon)
            grid[0, x, y] += 1.0

        # Channel 1 (Demand) & Channel 3 (Starvation Heat Map)
        current_time = self.simulator.current_time
        for task in self.simulator.state.active_tasks:
            x, y = get_cell(task.pickup_lat, task.pickup_lon)
            grid[1, x, y] += 1.0
            wait_time = current_time - task.release_time  # seconds since release
            grid[3, x, y] += wait_time  # Adds the actual wait time to create the "Pain Map"

        # Channel 2: Future Supply (Proactive Anticipation)
        for worker in self.simulator.state.assigned_workers:
            if worker.assigned_task:
                x, y = get_cell(worker.assigned_task.dropoff_lat, worker.assigned_task.dropoff_lon)
                grid[2, x, y] += 1.0

        # Optional: Normalize the channels so massive numbers don't blow out the neural network
        # (Using a simple max division. Add 1e-8 to prevent divide by zero errors)
        for c in range(4):
            max_val = np.max(grid[c])
            if max_val > 0:
                grid[c] = grid[c] / max_val

        return grid

    def _calculate_reward(self):
        """
        Calculate reward using the standardized MetricsManager.
        
        The manager already calculated JFI, Backlog, and AvgWait at the end of the step.
        We just normalize and combine them with reward weights.
        """
        # Get pre-calculated stats from MetricsManager (single source of truth)
        stats = self.simulator.metrics.get_reward_stats()
        # stats = {'fairness': JFI, 'throughput': -Backlog, 'latency': -AvgWait}
            
        # Normalize components to be roughly in same magnitude
        # JFI is [0, 1]. Target is to boost it.
        # Backlog is [0, 500+]. Throughput is already negative.
        # Wait time is [0, 30+]. Latency is already negative.
        
        r_fairness = (stats['fairness'] - 0.5) * 10.0  # Range [-5, 5]
        r_throughput = stats['throughput'] / 100.0  # Range [-5, 0] (throughput is already negative)
        r_latency = stats['latency'] / 5.0  # Range [-6, 0] (latency is already negative)
        
        reward = (self.reward_weights[0] * r_fairness + 
                  self.reward_weights[1] * r_throughput + 
                  self.reward_weights[2] * r_latency)
                  
        return reward
