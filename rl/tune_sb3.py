"""
AutoML Hyperparameter Tuning for PPO Agent using Optuna.

This script runs multiple trials to find optimal hyperparameters for the PPO agent.
Each trial trains a model for a short period and evaluates its performance.

Usage:
    python rl/tune_sb3.py [--trials N] [--timesteps N] [--data-root PATH]
    
Examples:
    # Quick test run (10 trials, 10K timesteps each)
    python rl/tune_sb3.py --trials 10 --timesteps 10000
    
    # Full tuning run (30 trials, 50K timesteps each)
    python rl/tune_sb3.py --trials 30 --timesteps 50000
"""

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

# --- DEFAULT CONFIGURATION ---
DEFAULT_N_TRIALS = 30          # How many different combos to try
DEFAULT_N_TIMESTEPS = 50000    # How long to train each combo (short run)
DEFAULT_N_EVAL_EPISODES = 3    # How many test episodes to verify performance
DEFAULT_DATA_ROOT = "./data/didi/full_didi_gaia"  # Ensure this points to your data
# -----------------------------

def sample_ppo_params(trial):
    """
    Define the search space for PPO hyperparameters.
    These are the variables Optuna will optimize.
    """
    return {
        # LEARNING RATE: The most important parameter.
        # Too high = diverge, Too low = slow convergence.
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        
        # N_STEPS: Steps per update.
        # Larger batch size usually better for smoother gradients.
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        
        # BATCH SIZE: Minibatch size for gradient update.
        # Must be a factor of n_steps.
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        
        # GAMMA: Discount factor.
        # 0.99 is standard, 0.95 focuses more on immediate rewards.
        "gamma": trial.suggest_categorical("gamma", [0.95, 0.98, 0.99]),
        
        # GAE LAMBDA: Bias vs Variance trade-off for advantage estimation.
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98]),
        
        # CLIP RANGE: How aggressively to update the policy.
        # Smaller = more conservative.
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        
        # ENTROPY COEFFICIENT: Controls exploration!
        # Very important. If 0, agent exploits too early.
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
        
        # VF COEF: Weight of the Value Function loss.
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        
        # MAX GRAD NORM: Clipping gradients to prevent explosions.
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1.0]),
        
        # NETWORK ARCHITECTURE: Size of the brain.
        # 'net_arch_type' maps to actual array structures below.
        "net_arch_type": trial.suggest_categorical("net_arch_type", ["small", "medium", "large"])
    }

def get_net_arch(net_arch_type):
    """Helper to decode network architecture string."""
    if net_arch_type == "small":
        return [dict(pi=[64, 64], vf=[64, 64])]
    elif net_arch_type == "medium":
        return [dict(pi=[128, 128], vf=[128, 128])]
    elif net_arch_type == "large":
        return [dict(pi=[256, 256], vf=[256, 256])]
    return [dict(pi=[64, 64], vf=[64, 64])]

def objective(trial, data_root, n_timesteps, n_eval_episodes):
    """
    The function Optuna will try to maximize.
    1. Samples params
    2. Builds Agent
    3. Trains for a bit
    4. Returns Evaluation Score
    """
    
    # 1. Sample Hyperparameters
    hyperparams = sample_ppo_params(trial)
    
    # Extract special args that don't go directly into PPO constructor
    net_arch_type = hyperparams.pop("net_arch_type")
    policy_kwargs = dict(
        net_arch=get_net_arch(net_arch_type),
        activation_fn=nn.Tanh
    )
    
    # 2. Create Environment
    # Get available day folders for training
    try:
        if data_root and os.path.exists(data_root):
            day_folders = sorted([d for d in os.listdir(data_root) 
                                if os.path.isdir(os.path.join(data_root, d))])
            # Use first 3 days for tuning (faster)
            train_days = day_folders[:3] if len(day_folders) >= 3 else day_folders
        else:
            train_days = None
        
        env = AdaptiveSpatialCrowdsourcingEnv(
            dataset="didi",
            data_root=data_root,
            day_folders=train_days,
            step_duration_minutes=5,
            warmup_duration_minutes=30,
            episode_duration_hours=4
        )
        env = Monitor(env)  # Wrap for statistics
    except Exception as e:
        print(f"❌ Trial {trial.number}: Skipping due to env error: {e}")
        raise optuna.exceptions.TrialPruned()

    # 3. Initialize Agent
    try:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=None,  # Disable tensorboard for tuning runs
            **hyperparams
        )
    except Exception as e:
        print(f"❌ Trial {trial.number}: Failed to create model: {e}")
        env.close()
        raise optuna.exceptions.TrialPruned()

    # 4. Train Agent
    # We use a try/except block to handle simulation crashes gracefully
    try:
        model.learn(total_timesteps=n_timesteps, progress_bar=False)
    except Exception as e:
        print(f"❌ Trial {trial.number}: Failed during training: {e}")
        env.close()
        raise optuna.exceptions.TrialPruned()

    # 5. Evaluate Performance
    # We evaluate on the SAME env for speed (in rigorous setup, use separate validation env)
    try:
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
    except Exception as e:
        print(f"❌ Trial {trial.number}: Failed during evaluation: {e}")
        env.close()
        raise optuna.exceptions.TrialPruned()
    
    # Clean up
    env.close()
    
    # Report intermediate value for pruning
    trial.report(mean_reward, step=n_timesteps)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return mean_reward

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PPO agent using Optuna")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS,
                       help=f"Number of trials to run (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_N_TIMESTEPS,
                       help=f"Timesteps per trial (default: {DEFAULT_N_TIMESTEPS})")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES,
                       help=f"Evaluation episodes per trial (default: {DEFAULT_N_EVAL_EPISODES})")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                       help=f"Data root directory (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--study-name", type=str, default="ppo_sc_tuning",
                       help="Name for the Optuna study (default: ppo_sc_tuning)")
    parser.add_argument("--output", type=str, default="best_hyperparameters.json",
                       help="Output file for best hyperparameters (default: best_hyperparameters.json)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 OPTUNA HYPERPARAMETER TUNING FOR PPO AGENT")
    print("=" * 80)
    print(f"Trials: {args.trials}")
    print(f"Timesteps per trial: {args.timesteps:,}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Data root: {args.data_root}")
    print(f"Study name: {args.study_name}")
    print("=" * 80)
    print()
    
    # Verify data root exists
    if not os.path.exists(args.data_root):
        print(f"❌ Error: Data root not found: {args.data_root}")
        print("   Please ensure the full_didi_gaia folder exists.")
        return 1
    
    # Create the study with pruning
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
    )
    
    # Create objective function with fixed arguments
    def objective_with_args(trial):
        return objective(trial, args.data_root, args.timesteps, args.eval_episodes)
    
    try:
        print("🔬 Starting optimization...")
        study.optimize(objective_with_args, n_trials=args.trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted by user.")
        print("   Partial results will be saved.")
    except Exception as e:
        print(f"\n❌ Tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show results
    print("\n" + "=" * 80)
    print("🏆 STUDY FINISHED")
    print("=" * 80)
    
    if len(study.trials) == 0:
        print("❌ No trials completed successfully!")
        return 1
    
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print()
    
    if study.best_trial:
        print(f"Best Reward: {study.best_value:.3f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        
        # Save best params (including net_arch_type for reconstruction)
        best_params = study.best_params.copy()
        
        # Save to JSON
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"\n✅ Saved best hyperparameters to: {output_path}")
        
        # Also print summary statistics
        if len(study.trials) > 1:
            rewards = [t.value for t in study.trials if t.value is not None]
            if rewards:
                print(f"\n📊 Reward Statistics:")
                print(f"   Mean: {np.mean(rewards):.3f}")
                print(f"   Std:  {np.std(rewards):.3f}")
                print(f"   Min:  {np.min(rewards):.3f}")
                print(f"   Max:  {np.max(rewards):.3f}")
    else:
        print("❌ No successful trials to report!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


