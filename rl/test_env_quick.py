#!/usr/bin/env python3
"""
Quick test to verify the RL environment works before training.
"""

import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

def test_environment():
    """Test the environment with a single day folder."""
    print("=" * 60)
    print("🧪 QUICK ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test with multiple day folders to verify dynamic loading
    data_root = './data/didi/full_didi_gaia'
    # Use first 3 available days to test dynamic loading
    import os
    if os.path.exists(data_root):
        available_days = sorted([d for d in os.listdir(data_root) 
                                if os.path.isdir(os.path.join(data_root, d))])
        day_folders = available_days[:3] if len(available_days) >= 3 else available_days[:1]
        print(f"   Using {len(day_folders)} day(s) for testing: {day_folders}")
    else:
        day_folders = ['496528674@qq.com_20161101']  # Fallback
    
    print(f"\n[1/4] Initializing environment...")
    print(f"   Data root: {data_root}")
    print(f"   Day folders: {day_folders}")
    
    try:
        env = AdaptiveSpatialCrowdsourcingEnv(
            dataset='didi',
            data_root=data_root,
            day_folders=day_folders,
            step_duration_minutes=15
        )
        print("   ✅ Environment created successfully!")
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[2/4] Testing reset()...")
    try:
        obs, info = env.reset()
        print(f"   ✅ Reset successful!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[3/4] Testing step() with progress output...")
    try:
        import time
        import sys
        
        # Sample a random action
        action = env.action_space.sample()
        print(f"   Action (λ1, λ3): [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"   Running simulation step (15 minutes of sim time)...")
        print(f"   " + "-" * 50)
        print(f"   Processing events", end="", flush=True)
        
        # Add a simple progress indicator
        start_time = time.time()
        
        # Run step and show progress (with dots)
        def progress_callback():
            print(".", end="", flush=True)
        
        # Run the step
        obs, reward, terminated, truncated, info = env.step(action)
        
        elapsed = time.time() - start_time
        print(f" done! ({elapsed:.1f}s)")
        print(f"   " + "-" * 50)
        print(f"   ✅ Step completed!")
        print(f"   📊 Results:")
        print(f"      Reward: {reward:.3f}")
        print(f"      Backlog: {info.get('backlog', 'N/A')}")
        print(f"      Completed tasks: {info.get('completed', 'N/A')}")
        print(f"      Step index: {info.get('step', 'N/A')}")
        print(f"      Lambdas used: λ1={info.get('lambdas', [0,0,0])[0]:.2f}, "
              f"λ2={info.get('lambdas', [0,0,0])[1]:.2f}, "
              f"λ3={info.get('lambdas', [0,0,0])[2]:.2f}")
        print(f"   📈 Observation stats:")
        print(f"      Shape: {obs.shape}")
        print(f"      Min: {obs.min():.3f}, Max: {obs.max():.3f}, Mean: {obs.mean():.3f}")
        print(f"      Terminated: {terminated}, Truncated: {truncated}")
        
        # Show a few more steps to see progression
        print(f"\n   Running 3 more steps to show progression...")
        for i in range(3):
            action = env.action_space.sample()
            print(f"   Step {i+2}/3: λ=[{action[0]:.2f}, {action[1]:.2f}] ... ", end="", flush=True)
            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            print(f"done ({step_time:.1f}s)")
            print(f"      → Reward={reward:6.2f}, Backlog={info.get('backlog', 0):5d}, "
                  f"Completed={info.get('completed', 0):5d}")
            if terminated:
                print(f"      ⚠️  Episode terminated!")
                break
        
    except Exception as e:
        print(f"   ❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[4/4] Testing multiple resets (dynamic day loading)...")
    try:
        # Test that reset loads different days (if multiple available)
        print(f"   Testing reset with different days...")
        loaded_days = []
        for i in range(5):  # Test 5 resets to see day variation
            print(f"   Resetting environment {i+1}/5...", end="", flush=True)
            obs, info = env.reset()
            
            # Try to detect which day was loaded by checking worker/task counts
            # (Different days will have different counts)
            worker_count = len(env.workers)
            task_count = len(env.tasks)
            
            # Store the signature to detect if different days are loaded
            day_signature = f"{worker_count}w/{task_count}t"
            loaded_days.append(day_signature)
            
            print(f" → Loaded: {worker_count:,} workers, {task_count:,} tasks")
            print(f"      Observation shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Check if we got different days
        unique_days = len(set(loaded_days))
        if unique_days > 1:
            print(f"   ✅ Verified: Loaded {unique_days} different day(s) across resets!")
        else:
            print(f"   ⚠️  Note: All resets loaded same day (expected if only 1 day available)")
        print("   ✅ Multiple resets successful!")
    except Exception as e:
        print(f"   ❌ Multiple resets failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Environment is ready for training!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)



