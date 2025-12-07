"""
Train PPO agent for Adaptive Spatial Crowdsourcing using Stable Baselines 3.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

def main():
    # Create log dir
    log_dir = "rl_logs_sb3/"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize environment
    print("Initializing environment...")
    env = AdaptiveSpatialCrowdsourcingEnv(
        dataset="didi", 
        step_duration_minutes=15,
        reward_weights=[1.0, 1.0, 1.0]
    )
    
    # Check environment compatibility
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")

    # Initialize PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # Train agent
    print("Starting training...")
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_dir,
        name_prefix="ppo_sc_model"
    )
    
    total_timesteps = 10000 # Adjust as needed
    model.learn(
        total_timesteps=total_timesteps, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("Training complete!")
    
    # Save final model
    model.save(os.path.join(log_dir, "ppo_sc_final"))
    print(f"Model saved to {log_dir}")

if __name__ == "__main__":
    main()
