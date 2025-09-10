#!/usr/bin/env python3
"""
Train PPO agent for dynamic weight adaptation in spatial crowdsourcing.

This script trains a PPO agent to learn optimal λ1, λ2, λ3 weights
for the composite scoring function based on real-time system feedback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime
from collections import deque

from rl.ppo_agent import PPOAgent
from rl.sc_environment import SpatialCrowdsourcingEnv


class PPOTrainer:
    """Trainer for PPO agent on spatial crowdsourcing environment."""
    
    def __init__(self, env, agent, log_dir="rl_logs"):
        self.env = env
        self.agent = agent
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_fairness = []
        self.episode_efficiency = []
        self.best_weights = None
        self.best_performance = 0
        
    def train(self, num_episodes=100, update_freq=10, save_freq=25):
        """
        Train PPO agent.
        
        Args:
            num_episodes: Number of training episodes
            update_freq: Update policy every N steps
            save_freq: Save model every N episodes
        """
        print(f"Starting PPO training for {num_episodes} episodes...")
        print(f"Environment: {self.env.dataset}, Episode length: {self.env.episode_length}")
        print(f"Baseline performance: JFI={self.env.baseline_performance['jains_fairness_index']:.3f}")
        
        episode_rewards = deque(maxlen=100)
        recent_performance = deque(maxlen=10)
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while True:
                # Get action from agent
                action, log_prob, value = self.agent.get_action(state)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(
                    state, action, reward, next_state, done, log_prob, value
                )
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Update agent periodically
                if episode_steps % update_freq == 0:
                    loss_info = self.agent.update()
                    if loss_info:
                        print(f"  Step {episode_steps}: Policy loss={loss_info['policy_loss']:.3f}, "
                              f"Value loss={loss_info['value_loss']:.3f}")
                
                if done:
                    break
            
            # Episode summary
            episode_summary = self.env.get_episode_summary()
            episode_rewards.append(episode_reward)
            recent_performance.append(episode_summary)
            
            # Track best performance
            if episode_summary and episode_summary.get('best_jfi', 0) > self.best_performance:
                self.best_performance = episode_summary['best_jfi']
                self.best_weights = episode_summary['best_lambda']
            
            # Logging
            avg_reward = np.mean(episode_rewards)
            print(f"  Episode reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
            if episode_summary:
                print(f"  Best config: λ=({episode_summary['best_lambda'][0]:.2f}, "
                      f"{episode_summary['best_lambda'][1]:.2f}, {episode_summary['best_lambda'][2]:.2f}) "
                      f"→ JFI={episode_summary['best_jfi']:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                self.save_checkpoint(episode + 1)
                self.plot_training_progress(episode + 1)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            if episode_summary:
                self.episode_fairness.append(episode_summary.get('best_jfi', 0))
                self.episode_efficiency.append(episode_summary.get('avg_tar', 0))
        
        print(f"\nTraining completed!")
        print(f"Best performance: JFI={self.best_performance:.3f} with λ={self.best_weights}")
        
        # Final save and plots
        self.save_checkpoint("final")
        self.plot_training_progress("final")
        self.save_training_log()
        
        return self.best_weights, self.best_performance
    
    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, f"ppo_model_episode_{episode}.pt")
        self.agent.save(checkpoint_path)
        print(f"  Model saved: {checkpoint_path}")
    
    def plot_training_progress(self, episode):
        """Plot training progress."""
        if len(self.episode_rewards) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Moving average rewards
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            ax1.plot(range(9, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
            ax1.legend()
        
        # Fairness progress
        if self.episode_fairness:
            ax2.plot(self.episode_fairness, 'g-')
            ax2.axhline(y=self.env.baseline_performance['jains_fairness_index'], color='r', linestyle='--', label='Baseline')
            ax2.set_title('Best Fairness Index per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('JFI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Efficiency progress
        if self.episode_efficiency:
            ax3.plot(self.episode_efficiency, 'b-')
            ax3.axhline(y=self.env.baseline_performance['task_assignment_ratio'], color='r', linestyle='--', label='Baseline')
            ax3.set_title('Task Assignment Ratio per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('TAR')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Reward distribution
        if len(self.episode_rewards) >= 10:
            ax4.hist(self.episode_rewards[-50:], bins=20, alpha=0.7)
            ax4.set_title('Recent Reward Distribution')
            ax4.set_xlabel('Reward')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f"training_progress_episode_{episode}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Progress plot saved: {plot_path}")
    
    def save_training_log(self):
        """Save training log as JSON."""
        log_data = {
            'training_config': {
                'num_episodes': len(self.episode_rewards),
                'dataset': self.env.dataset,
                'episode_length': self.env.episode_length,
                'reward_weights': self.env.reward_weights,
            },
            'baseline_performance': self.env.baseline_performance,
            'best_performance': {
                'jfi': self.best_performance,
                'weights': self.best_weights,
            },
            'training_metrics': {
                'episode_rewards': self.episode_rewards,
                'episode_fairness': self.episode_fairness,
                'episode_efficiency': self.episode_efficiency,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved: {log_path}")


def main():
    """Main training function."""
    print("PPO TRAINING FOR SPATIAL CROWDSOURCING WEIGHT ADAPTATION")
    print("=" * 60)
    
    # Environment setup
    env = SpatialCrowdsourcingEnv(
        dataset="didi",
        episode_length=20,  # Start with shorter episodes for faster training
        reward_weights=[1.0, 1.0, 0.8]  # Slightly less weight on utility
    )
    
    # Agent setup
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.95,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.02,  # Encourage exploration
        value_coef=0.5
    )
    
    # Trainer setup
    trainer = PPOTrainer(env, agent)
    
    # Train
    best_weights, best_performance = trainer.train(
        num_episodes=50,  # Start with fewer episodes for testing
        update_freq=5,    # Update more frequently
        save_freq=10
    )
    
    print(f"\nFINAL RESULTS:")
    print(f"Best λ weights: ({best_weights[0]:.3f}, {best_weights[1]:.3f}, {best_weights[2]:.3f})")
    print(f"Best JFI achieved: {best_performance:.3f}")
    print(f"Baseline JFI: {env.baseline_performance['jains_fairness_index']:.3f}")
    improvement = (best_performance - env.baseline_performance['jains_fairness_index']) / env.baseline_performance['jains_fairness_index'] * 100
    print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)
    
    main()