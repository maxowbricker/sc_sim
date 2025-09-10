"""
Spatial Crowdsourcing RL Environment for PPO training.

Wraps the spatial crowdsourcing simulator as an RL environment
for training PPO agent to learn optimal weight parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import get_simulation_config, create_composite_config
from rl.ppo_agent import extract_state_features, transform_action_to_weights


class SpatialCrowdsourcingEnv:
    """RL Environment for spatial crowdsourcing task assignment."""
    
    def __init__(self, dataset="didi", episode_length=50, reward_weights=None):
        """
        Initialize SC environment.
        
        Args:
            dataset: Dataset to use for simulation
            episode_length: Number of time steps per episode
            reward_weights: Weights for reward function components [fairness, starvation, utility]
        """
        self.dataset = dataset
        self.episode_length = episode_length
        self.reward_weights = reward_weights or [1.0, 1.0, 1.0]  # β1, β2, β3
        
        # Load data
        self.workers, self.tasks = load_workers_tasks(dataset)
        print(f"Loaded {len(self.workers)} workers, {len(self.tasks)} tasks")
        
        # Environment state
        self.current_step = 0
        self.episode_history = []
        self.baseline_performance = None
        
        # State and action space dimensions
        self.state_dim = 11  # Based on extract_state_features
        self.action_dim = 3   # λ1, λ2, λ3
        
        # Initialize baseline (greedy performance)
        self._calculate_baseline()
    
    def _calculate_baseline(self):
        """Calculate baseline performance using greedy strategy."""
        print("Calculating baseline performance...")
        
        config = get_simulation_config()
        config['assignment_strategy'] = 'greedy'
        
        summary = run_simulation(self.workers, self.tasks, sim_config=config)
        
        self.baseline_performance = {
            'task_assignment_ratio': summary.get('completed_tasks', 0) / len(self.tasks),
            'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
            'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
            'empty_km_share': summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)),
        }
        
        print(f"Baseline performance: TAR={self.baseline_performance['task_assignment_ratio']:.1%}, "
              f"JFI={self.baseline_performance['jains_fairness_index']:.3f}")
    
    def reset(self):
        """Reset environment for new episode."""
        self.current_step = 0
        self.episode_history = []
        
        # Return initial state (dummy state for first step)
        initial_state = np.zeros(self.state_dim, dtype=np.float32)
        return initial_state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: PPO action (λ weights in [-1, 1]^3)
            
        Returns:
            next_state, reward, done, info
        """
        self.current_step += 1
        
        # Transform action to lambda weights
        lambda1, lambda2, lambda3 = transform_action_to_weights(action)
        
        # Run simulation with these weights
        config = create_composite_config(
            λ1=lambda1,
            λ2=lambda2, 
            λ3=lambda3
        )
        
        try:
            summary = run_simulation(self.workers, self.tasks, sim_config=config)
            
            # Extract performance metrics
            performance = {
                'lambda1': lambda1,
                'lambda2': lambda2,
                'lambda3': lambda3,
                'completed_tasks': summary.get('completed_tasks', 0),
                'task_assignment_ratio': summary.get('completed_tasks', 0) / len(self.tasks),
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
                'empty_km_share': summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)),
                'peak_backlog': summary.get('backlog_peak', 0),
                'fairness_loss': summary.get('final_fairness_loss', 0),
                'ewma_cv': summary.get('final_ewma_cv', 0),
            }
            
            # Calculate reward
            reward = self._calculate_reward(performance)
            
            # Store in episode history
            self.episode_history.append(performance)
            
            # Create next state (use recent performance for state features)
            next_state = self._create_state_from_performance(performance)
            
        except Exception as e:
            print(f"Simulation failed with λ=({lambda1:.2f}, {lambda2:.2f}, {lambda3:.2f}): {e}")
            # Penalty for invalid configurations
            reward = -10.0
            performance = {'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3}
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Info dictionary
        info = {
            'performance': performance,
            'step': self.current_step,
            'lambda_weights': (lambda1, lambda2, lambda3)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, performance):
        """
        Calculate reward based on performance metrics.
        
        Reward function: R = β1 * F_t + β2 * S_t + β3 * U_t
        where F_t, S_t, U_t are normalized fairness, starvation, utility improvements
        """
        # Get baseline for normalization
        baseline = self.baseline_performance
        
        # Fairness improvement (higher JFI is better)
        fairness_improvement = performance['jains_fairness_index'] - baseline['jains_fairness_index']
        
        # Starvation mitigation (lower wait time is better) 
        starvation_improvement = -(performance['avg_wait_time'] - baseline['avg_wait_time']) / baseline['avg_wait_time']
        
        # Utility gain (higher task assignment ratio, lower empty travel)
        assignment_improvement = performance['task_assignment_ratio'] - baseline['task_assignment_ratio']
        efficiency_improvement = -(performance['empty_km_share'] - baseline['empty_km_share']) / max(0.01, baseline['empty_km_share'])
        utility_improvement = assignment_improvement + 0.5 * efficiency_improvement
        
        # Combined reward
        reward = (self.reward_weights[0] * fairness_improvement + 
                 self.reward_weights[1] * starvation_improvement + 
                 self.reward_weights[2] * utility_improvement)
        
        # Add penalty for very poor performance
        if performance['task_assignment_ratio'] < 0.5:  # Less than 50% assignment rate
            reward -= 5.0
        
        # Bonus for balanced improvements
        if (fairness_improvement > 0 and starvation_improvement > 0 and utility_improvement > 0):
            reward += 1.0
        
        return reward
    
    def _create_state_from_performance(self, performance):
        """Create state representation from performance metrics."""
        # Normalize metrics relative to baseline
        baseline = self.baseline_performance
        
        features = [
            performance['task_assignment_ratio'],
            performance['jains_fairness_index'], 
            performance['avg_wait_time'] / max(1, baseline['avg_wait_time']),
            performance['empty_km_share'] / max(0.01, baseline['empty_km_share']),
            performance.get('peak_backlog', 0) / len(self.tasks),
            performance.get('fairness_loss', 0),
            performance.get('ewma_cv', 0),
            performance['lambda1'] / 3.0,  # Normalize lambda values
            performance['lambda2'] / 3.0,
            performance['lambda3'] / 2.0,
            len(self.episode_history) / self.episode_length,  # Episode progress
        ]
        
        return np.array(features, dtype=np.float32)
    
    def get_episode_summary(self):
        """Get summary of current episode."""
        if not self.episode_history:
            return {}
        
        # Calculate episode statistics
        performances = self.episode_history
        
        best_performance = max(performances, key=lambda x: x.get('jains_fairness_index', 0))
        worst_performance = min(performances, key=lambda x: x.get('jains_fairness_index', 0))
        
        avg_performance = {
            'avg_jfi': np.mean([p.get('jains_fairness_index', 0) for p in performances]),
            'avg_tar': np.mean([p.get('task_assignment_ratio', 0) for p in performances]),
            'avg_wait': np.mean([p.get('avg_wait_time', 0) for p in performances]),
            'best_jfi': best_performance.get('jains_fairness_index', 0),
            'best_lambda': (best_performance.get('lambda1', 0), 
                           best_performance.get('lambda2', 0), 
                           best_performance.get('lambda3', 0)),
        }
        
        return avg_performance


class RewardShaping:
    """Advanced reward shaping for better learning."""
    
    def __init__(self, target_fairness=0.9, target_efficiency=0.95):
        self.target_fairness = target_fairness
        self.target_efficiency = target_efficiency
        self.episode_rewards = []
    
    def shaped_reward(self, performance, baseline):
        """Calculate shaped reward with multiple objectives."""
        
        # Multi-objective reward components
        fairness_reward = self._fairness_reward(performance, baseline)
        efficiency_reward = self._efficiency_reward(performance, baseline)
        stability_reward = self._stability_reward(performance)
        
        # Adaptive weights based on current performance
        if performance['jains_fairness_index'] < 0.7:
            # Focus on fairness if currently low
            weights = [0.6, 0.3, 0.1]
        elif performance['task_assignment_ratio'] < 0.8:
            # Focus on efficiency if assignment rate low
            weights = [0.2, 0.6, 0.2]
        else:
            # Balanced focus if both are reasonable
            weights = [0.4, 0.4, 0.2]
        
        total_reward = (weights[0] * fairness_reward + 
                       weights[1] * efficiency_reward + 
                       weights[2] * stability_reward)
        
        return total_reward
    
    def _fairness_reward(self, performance, baseline):
        """Reward for fairness improvements."""
        jfi_improvement = performance['jains_fairness_index'] - baseline['jains_fairness_index']
        
        # Progressive reward: more reward for reaching higher fairness levels
        if performance['jains_fairness_index'] >= self.target_fairness:
            return 2.0 + jfi_improvement  # Bonus for reaching target
        else:
            return jfi_improvement
    
    def _efficiency_reward(self, performance, baseline):
        """Reward for efficiency improvements."""
        tar_improvement = performance['task_assignment_ratio'] - baseline['task_assignment_ratio']
        wait_improvement = -(performance['avg_wait_time'] - baseline['avg_wait_time']) / baseline['avg_wait_time']
        
        efficiency_score = tar_improvement + 0.5 * wait_improvement
        
        if performance['task_assignment_ratio'] >= self.target_efficiency:
            return 1.5 + efficiency_score
        else:
            return efficiency_score
    
    def _stability_reward(self, performance):
        """Reward for stable, reasonable parameter choices."""
        lambda1, lambda2, lambda3 = performance['lambda1'], performance['lambda2'], performance['lambda3']
        
        # Penalty for extreme parameter values
        extremity_penalty = 0
        if lambda1 > 2.5 or lambda1 < 0.2:
            extremity_penalty += 0.5
        if lambda2 > 2.5 or lambda2 < 0.2:
            extremity_penalty += 0.5
        if lambda3 > 1.8 or lambda3 < 0.2:
            extremity_penalty += 0.5
        
        return -extremity_penalty