"""
Train PPO agent for Adaptive Spatial Crowdsourcing using Stable Baselines 3.

Usage:
    python rl/train_sb3.py [--timesteps N] [--resume PATH] [--test] [--hyperparams PATH]
    
Examples:
    # Quick test run (1000 timesteps)
    python rl/train_sb3.py --timesteps 1000
    
    # Full training run
    python rl/train_sb3.py --timesteps 50000
    
    # Resume from checkpoint
    python rl/train_sb3.py --resume rl_logs_sb3/ppo_sc_model_1000_steps.zip --timesteps 50000
    
    # Use Optuna-tuned hyperparameters (default: rl/best_hyperparameters.json)
    python rl/train_sb3.py --hyperparams best_hyperparameters.json --timesteps 50000
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sklearn.model_selection import train_test_split
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

def make_env(data_root, day_folders, rank=0, step_duration_minutes=5, reward_weights=None,
             warmup_duration_minutes=30, episode_duration_hours=4):
    """
    Utility function for multiprocessed env.
    
    Args:
        data_root: Base path to dataset folders
        day_folders: List of folder names to randomly select from
        rank: Index of the subprocess (useful for seeding)
        step_duration_minutes: Duration of each simulation step (default: 5 minutes)
        reward_weights: Weights for reward components
        warmup_duration_minutes: Duration of warmup phase (default: 30 minutes)
        episode_duration_hours: Duration of RL episode after warmup (default: 4 hours)
    """
    def _init():
        env = AdaptiveSpatialCrowdsourcingEnv(
            dataset="didi",
            step_duration_minutes=step_duration_minutes,
            reward_weights=reward_weights or [1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=day_folders,
            warmup_duration_minutes=warmup_duration_minutes,
            episode_duration_hours=episode_duration_hours
        )
        return env
    return _init

def create_env(dataset="didi", step_duration_minutes=5, reward_weights=None, 
               data_root=None, day_folders=None, warmup_duration_minutes=30,
               episode_duration_hours=4):
    """Create and wrap environment for training (legacy single-env mode)."""
    env = AdaptiveSpatialCrowdsourcingEnv(
        dataset=dataset,
        step_duration_minutes=step_duration_minutes,
        reward_weights=reward_weights or [1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=day_folders,
        warmup_duration_minutes=warmup_duration_minutes,
        episode_duration_hours=episode_duration_hours
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
    parser.add_argument("--hyperparams", type=str, default=None,
                       help="Path to best_hyperparameters.json from Optuna tuning (default: rl/best_hyperparameters.json)")
    
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
            step_duration_minutes=5,  # 5-minute steps for high-frequency decisions
            reward_weights=[1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=train_days,
            warmup_duration_minutes=30,
            episode_duration_hours=4
        )
        env = Monitor(env, log_dir)
    else:
        # Parallel environments mode
        num_cpu = args.num_cpu
        print(f"   Creating {num_cpu} parallel environments...")
        env = SubprocVecEnv([
            make_env(data_root, train_days, i, step_duration_minutes=5, 
                    reward_weights=[1.0, 1.0, 1.0],
                    warmup_duration_minutes=30,
                    episode_duration_hours=4) 
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
        step_duration_minutes=5,  # 5-minute steps for consistency
        reward_weights=[1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=test_days,
        warmup_duration_minutes=30,
        episode_duration_hours=4
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
            model = create_model(env, log_dir, hyperparams_path=args.hyperparams)
    else:
        model = create_model(env, log_dir, hyperparams_path=args.hyperparams)
    
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

def _get_net_arch(net_arch_type):
    """Map Optuna net_arch_type to policy_kwargs. Default to large [256, 256]."""
    if net_arch_type == "small":
        return dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        return dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_type == "large":
        return dict(pi=[256, 256], vf=[256, 256])
    return dict(pi=[256, 256], vf=[256, 256])  # Default: Optuna-recommended large


class SpatialCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN architecture designed specifically for the 10x10 Spatial Crowdsourcing Grid.
    It processes the 4-channel spatial map and concatenates it with the global scalars.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # We start by calling the Base class
        super().__init__(observation_space, features_dim)

        # 1. The Visual Cortex (CNN)
        # Input: (4 channels, 10x10) -> Output: (32 channels, 5x5) -> Flattened: 800
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shrinks 10x10 to 5x5
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the combined flattened arrays
        cnn_output_size = 32 * 5 * 5  # 800
        scalar_size = observation_space.spaces["global_scalars"].shape[0]  # 14
        total_concat_size = cnn_output_size + scalar_size  # 814

        # 2. The Synthesizer (Linear Layer)
        # Compresses the combined visual and scalar data into the final feature dimension
        self.linear = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        # Extract the dual inputs
        grid = observations["spatial_grid"]
        scalars = observations["global_scalars"]

        # Process grid through CNN
        cnn_features = self.cnn(grid)

        # Concatenate CNN output with scalars side-by-side
        combined_features = th.cat([cnn_features, scalars], dim=1)

        # Pass through final linear compression
        return self.linear(combined_features)


def create_model(env, log_dir, hyperparams_path=None):
    """
    Create a new PPO model. Loads hyperparameters from best_hyperparameters.json if available.
    Uses MultiInputPolicy with SpatialCNNExtractor for the dual-modal (spatial + scalar) observation.
    """
    project_root = Path(__file__).resolve().parent.parent
    default_hyperparams_path = project_root / "rl" / "best_hyperparameters.json"
    path = Path(hyperparams_path) if hyperparams_path else default_hyperparams_path

    kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # Default network architecture (using our Custom Extractor)
    policy_kwargs = dict(
        features_extractor_class=SpatialCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    # Load custom hyperparameters if provided
    if path.exists():
        try:
            with open(path) as f:
                hp = json.load(f)
            net_arch_type = hp.pop("net_arch_type", "large")
            if net_arch_type == "large":
                # We update the internal MLPs, but keep our custom extractor!
                policy_kwargs["net_arch"] = dict(pi=[256, 256], vf=[256, 256])
            elif net_arch_type == "medium":
                policy_kwargs["net_arch"] = dict(pi=[128, 128], vf=[128, 128])

            for k, v in hp.items():
                if k in kwargs:
                    kwargs[k] = v
            print(f"   🧠 Loaded hyperparameters from {path} (net_arch: {net_arch_type})")
        except Exception as e:
            print(f"   ⚠️  Could not load {path}: {e}. Using defaults.")
    else:
        print(f"   ⚠️  No hyperparams found at {path}. Using defaults.")

    model = PPO(
        "MultiInputPolicy",  # <--- CRITICAL CHANGE: Changed from MlpPolicy
        env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        **kwargs
    )
    return model

if __name__ == "__main__":
    main()
