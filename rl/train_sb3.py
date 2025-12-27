"""
Train PPO agent for Adaptive Spatial Crowdsourcing using Stable Baselines 3.

Usage:
    python rl/train_sb3.py [--timesteps N] [--resume PATH] [--test]
    
Examples:
    # Quick test run (1000 timesteps)
    python rl/train_sb3.py --timesteps 1000
    
    # Full training run
    python rl/train_sb3.py --timesteps 50000
    
    # Resume from checkpoint
    python rl/train_sb3.py --resume rl_logs_sb3/ppo_sc_model_1000_steps.zip --timesteps 50000
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sklearn.model_selection import train_test_split
import argparse
import os
import sys
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

def make_env(data_root, day_folders, rank=0, step_duration_minutes=15, reward_weights=None):
    """
    Utility function for multiprocessed env.
    
    Args:
        data_root: Base path to dataset folders
        day_folders: List of folder names to randomly select from
        rank: Index of the subprocess (useful for seeding)
        step_duration_minutes: Duration of each simulation step
        reward_weights: Weights for reward components
    """
    def _init():
        env = AdaptiveSpatialCrowdsourcingEnv(
            dataset="didi",
            step_duration_minutes=step_duration_minutes,
            reward_weights=reward_weights or [1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=day_folders
        )
        return env
    return _init

def create_env(dataset="didi", step_duration_minutes=15, reward_weights=None, 
               data_root=None, day_folders=None):
    """Create and wrap environment for training (legacy single-env mode)."""
    env = AdaptiveSpatialCrowdsourcingEnv(
        dataset=dataset,
        step_duration_minutes=step_duration_minutes,
        reward_weights=reward_weights or [1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=day_folders
    )
    return env

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Spatial Crowdsourcing")
    parser.add_argument("--timesteps", type=int, default=10000, 
                       help="Total number of training timesteps (default: 10000)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--test", action="store_true",
                       help="Quick test run with minimal timesteps (1000)")
    parser.add_argument("--log-dir", type=str, default="rl_logs_sb3",
                       help="Directory for logs and checkpoints (default: rl_logs_sb3)")
    parser.add_argument("--skip-env-check", action="store_true",
                       help="Skip environment compatibility check (faster startup)")
    parser.add_argument("--data-root", type=str, default="./data/didi/full_didi_gaia",
                       help="Base path to dataset folders (default: ./data/didi/full_didi_gaia)")
    parser.add_argument("--num-cpu", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--train-days", type=int, default=24,
                       help="Number of days to use for training (default: 24)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel environments (use single env)")
    
    args = parser.parse_args()
    
    # Override timesteps for test mode
    if args.test:
        args.timesteps = 1000
        print("🧪 Running in TEST mode (1000 timesteps)")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print("PPO TRAINING FOR ADAPTIVE SPATIAL CROWDSOURCING")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Resume from: {args.resume if args.resume else 'None (new training)'}")
    print("=" * 80)
    
    # Setup Data Paths
    print("\n[1/5] Setting up data paths...")
    data_root = args.data_root
    
    # Get all day folders from the data root
    if os.path.exists(data_root):
        all_days = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))]
        all_days = sorted(all_days)  # Ensure consistent order before splitting
        
        print(f"   Found {len(all_days)} day folders in {data_root}")
        
        if len(all_days) < args.train_days:
            print(f"   ⚠️  Warning: Only {len(all_days)} folders available, but {args.train_days} requested for training")
            print(f"   Using all {len(all_days)} folders for training")
            train_days = all_days
            test_days = []
        else:
            # Split: Training Days, Testing Days
            # shuffle=True ensures we get a random mix of weekdays/weekends in both sets
            train_days, test_days = train_test_split(
                all_days, 
                train_size=args.train_days, 
                random_state=42, 
                shuffle=True
            )
        
        print(f"   🏋️  Training on {len(train_days)} days: {train_days[:3]}...")
        if test_days:
            print(f"   🧪 Testing on {len(test_days)} days: {test_days[:3]}...")
    else:
        print(f"   ⚠️  Data root not found: {data_root}")
        print("   Falling back to legacy single-dataset mode")
        train_days = None
        test_days = None
        data_root = None
    
    # Initialize environment(s)
    print("\n[2/5] Initializing environment(s)...")
    
    if args.no_parallel or train_days is None:
        # Single environment mode (legacy or disabled parallel)
        print("   Using single environment (no parallelization)")
        env = create_env(
            dataset="didi", 
            step_duration_minutes=15, 
            reward_weights=[1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=train_days
        )
    env = Monitor(env, log_dir)
    else:
        # Parallel environments mode
        num_cpu = args.num_cpu
        print(f"   Creating {num_cpu} parallel environments...")
        env = SubprocVecEnv([
            make_env(data_root, train_days, i, step_duration_minutes=15, 
                    reward_weights=[1.0, 1.0, 1.0]) 
            for i in range(num_cpu)
        ])
        print(f"   ✅ Parallel environment created with {num_cpu} workers")
    
    # Check environment compatibility (optional)
    if not args.skip_env_check:
        print("\n[3/5] Checking environment compatibility...")
        try:
            # For parallel envs, check the first one
            if isinstance(env, SubprocVecEnv):
                # Can't easily check SubprocVecEnv, skip for now
                print("   Skipping check for parallel environments (SubprocVecEnv)")
            else:
            check_env(env, warn=True)
            print("✅ Environment check passed!")
        except Exception as e:
            print(f"⚠️  Environment check warning: {e}")
            print("   Continuing anyway...")
    else:
        print("\n[3/5] Skipping environment check (--skip-env-check)")
    
    # Create evaluation environment
    print("\n[4/5] Creating evaluation environment...")
    if not test_days or len(test_days) == 0:
        raise ValueError(
            "No test days available for evaluation. "
            "Cannot evaluate on training data (data leakage). "
            "Ensure train_days < total_days to create a test split."
        )
    
    eval_env = create_env(
        dataset="didi", 
        step_duration_minutes=15, 
        reward_weights=[1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=test_days
    )
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Initialize or load PPO agent
    print("\n[5/5] Initializing PPO agent...")
    if args.resume:
        print(f"   Loading model from: {args.resume}")
        try:
            model = PPO.load(args.resume, env=env, verbose=1)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("   Starting fresh training instead...")
            model = create_model(env, log_dir)
    else:
        model = create_model(env, log_dir)
    
    # Setup callbacks
    print("\n[4/4] Setting up training callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, args.timesteps // 10),  # Save at least 10 times during training
        save_path=log_dir,
        name_prefix="ppo_sc_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(1000, args.timesteps // 20),  # Evaluate at least 20 times
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train agent
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print(f"Monitor progress with: tensorboard --logdir {log_dir}")
    print("=" * 80 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.resume is None  # Reset counter if not resuming
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        
        # Save final model
        final_model_path = os.path.join(log_dir, "ppo_sc_final")
        model.save(final_model_path)
        print(f"📦 Final model saved to: {final_model_path}")
        print(f"📊 TensorBoard logs: {log_dir}")
        print(f"🏆 Best model: {os.path.join(log_dir, 'best_model', 'best_model.zip')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving current model...")
        interrupted_path = os.path.join(log_dir, "ppo_sc_interrupted")
        model.save(interrupted_path)
        print(f"📦 Model saved to: {interrupted_path}")
        print("   Resume with: --resume", interrupted_path)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Attempting to save model...")
        try:
            error_path = os.path.join(log_dir, "ppo_sc_error")
            model.save(error_path)
            print(f"📦 Model saved to: {error_path}")
        except:
            print("   Failed to save model")

def create_model(env, log_dir):
    """Create a new PPO model with default hyperparameters."""
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per update
        batch_size=64,
        n_epochs=10,   # Optimization epochs per update
        gamma=0.99,    # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,   # PPO clip range
        ent_coef=0.01,    # Entropy coefficient
        vf_coef=0.5,      # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Network architecture
        )
    )
    return model

if __name__ == "__main__":
    main()
