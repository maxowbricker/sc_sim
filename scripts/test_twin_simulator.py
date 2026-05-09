#!/usr/bin/env python3
"""
Twin Simulator Test: Verify the shadow (greedy) simulator runs correctly
in parallel with the main (composite) simulator.

Tests:
  1. Both simulators initialize at the same time
  2. Shadow stays greedy, main switches to composite after warmup
  3. Both can step independently without crashing
  4. Their metrics diverge over time (not identical)
  5. Reward calculation produces sensible advantage values

Usage:
    cd /path/to/sc_sim
    conda activate sc
    python scripts/test_twin_simulator.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

def run_test():
    print("=" * 70)
    print("Twin Simulator Correctness Test")
    print("=" * 70)

    # 1. Create environment with data from full_didi_gaia (like training would)
    print("\n📂 Initializing environment...")
    data_root = "./data/didi/full_didi_gaia"
    day_folders = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    env = AdaptiveSpatialCrowdsourcingEnv(
        dataset="didi",
        data_root=data_root,
        day_folders=day_folders,
        step_duration_minutes=5,
        warmup_duration_minutes=30,
        episode_duration_hours=8,
    )

    print("✅ Environment created\n")

    # 2. Reset to start an episode
    print("🚀 Resetting environment...")
    obs, _ = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Initial reward_weights: {env.reward_weights}\n")

    # 3. Verify both simulators exist and are at the same time
    assert hasattr(env, 'simulator'), "Main simulator not found"
    assert hasattr(env, 'shadow_simulator'), "Shadow simulator not found"
    
    init_time = env.simulator.current_time
    shadow_init_time = env.shadow_simulator.current_time
    assert init_time == shadow_init_time, \
        f"Initial times don't match: main={init_time}, shadow={shadow_init_time}"
    print(f"✅ Both simulators initialized at time {init_time}\n")

    # 4. Verify strategies are correct
    assert env.simulator.strategy_name == 'composite', \
        f"Main simulator should be composite, got {env.simulator.strategy_name}"
    assert env.shadow_simulator.strategy_name == 'greedy', \
        f"Shadow simulator should be greedy, got {env.shadow_simulator.strategy_name}"
    print(f"✅ Strategies correct: main=composite, shadow=greedy\n")

    # 5. Step through episode with fixed action
    print(f"{'Step':>5}  {'Time':>10}  {'Main JFI':>8}  {'Shadow JFI':>10}  {'Diverged?':>10}  {'Reward':>8}")
    print("-" * 70)

    max_steps = 96  # Full 8-hour episode
    metrics = {
        'steps': 0,
        'time_matches': 0,
        'time_mismatches': 0,
        'divergences': 0,
        'reward_out_of_range': 0,
        'rewards': []
    }

    prev_main_jfi = None
    prev_shadow_jfi = None

    for step in range(1, max_steps + 1):
        # Use a fixed action to isolate simulator behavior
        action = np.array([1.0, 0.3], dtype=np.float32)

        obs, reward, done, _, info = env.step(action)

        # Check 1: Both simulators should have advanced to the same time
        main_time = env.simulator.current_time
        shadow_time = env.shadow_simulator.current_time
        if main_time == shadow_time:
            metrics['time_matches'] += 1
        else:
            metrics['time_mismatches'] += 1
            print(f"  ⚠️  Time mismatch at step {step}: main={main_time}, shadow={shadow_time}")

        # Check 2: Get metrics to verify divergence
        main_jfi = env.simulator.metrics.current_step_stats['jfi']
        shadow_jfi = env.shadow_simulator.metrics.current_step_stats['jfi']

        # After warmup, metrics should eventually diverge (they make different assignments)
        diverged = abs(main_jfi - shadow_jfi) > 0.001
        if diverged:
            metrics['divergences'] += 1

        # Check 3: Reward should be in reasonable range (advantage bounded)
        if not (-2.0 < reward < 2.0):
            metrics['reward_out_of_range'] += 1
            print(f"  ⚠️  Reward out of range at step {step}: {reward}")
        metrics['rewards'].append(reward)

        # Print progress
        diverged_str = "yes" if diverged else "no"
        print(f"{step:>5}  {main_time:>10.0f}  {main_jfi:>8.3f}  {shadow_jfi:>10.3f}  {diverged_str:>10}  {reward:>8.3f}")

        metrics['steps'] += 1

        if done:
            print(f"\n  ℹ️  Episode ended at step {step}")
            break

    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Steps executed            : {metrics['steps']}")
    print(f"  Time matches              : {metrics['time_matches']}")
    if metrics['time_mismatches'] > 0:
        print(f"  ❌ Time mismatches         : {metrics['time_mismatches']}")
    print(f"  Divergences detected      : {metrics['divergences']} (expected > 0 after warmup)")
    if metrics['reward_out_of_range'] > 0:
        print(f"  ❌ Rewards out of range    : {metrics['reward_out_of_range']}")

    mean_reward = np.mean(metrics['rewards']) if metrics['rewards'] else 0.0
    std_reward = np.std(metrics['rewards']) if metrics['rewards'] else 0.0
    print(f"  Reward mean               : {mean_reward:.4f}")
    print(f"  Reward std                : {std_reward:.4f}")
    print(f"  Reward range              : [{min(metrics['rewards']):.4f}, {max(metrics['rewards']):.4f}]")

    print("\n" + "=" * 70)

    # Final verdict
    all_pass = (
        metrics['time_mismatches'] == 0 and
        metrics['divergences'] > 0 and
        metrics['reward_out_of_range'] == 0
    )

    if all_pass:
        print("✅ Twin simulator test PASSED")
        print("   - Both simulators advance in sync")
        print("   - Metrics diverge over time (expected behavior)")
        print("   - Reward advantage signal is reasonable")
    else:
        print("❌ Twin simulator test FAILED")
        if metrics['time_mismatches'] > 0:
            print(f"   - Time sync issue: {metrics['time_mismatches']} mismatches")
        if metrics['divergences'] == 0:
            print(f"   - No metric divergence: simulators not acting independently")
        if metrics['reward_out_of_range'] > 0:
            print(f"   - Reward signal issue: {metrics['reward_out_of_range']} out of range")

    print("=" * 70 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_test())
