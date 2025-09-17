#!/usr/bin/env python3
"""
Quick operational test to verify the system is working correctly.
Tests core functionality before heavy parameter sweeping.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def operational_test():
    """Quick smoke test of core system functionality."""
    
    print("🧪 OPERATIONAL SMOKE TEST")
    print("=" * 50)
    print("Testing core functionality before parameter sweeping...")
    
    # Load small test dataset for speed
    print("\n1️⃣ Loading test data...")
    workers_df, tasks_df = load_data('didi', max_workers=500, max_tasks=200)
    print(f"   ✅ Loaded {len(workers_df)} workers, {len(tasks_df)} tasks")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Greedy Strategy',
            'config': {
                'assignment_strategy': 'greedy'
            }
        },
        {
            'name': 'Composite (Default)',
            'config': create_composite_config(
                fairness_weight=1.0,
                starvation_weight=1.0,
                utility_weight=1.0,
                soft_threshold=0.2,
                assignment_strategy="composite"
            )
        },
        {
            'name': 'Composite (Fairness Focus)',
            'config': create_composite_config(
                fairness_weight=2.0,
                starvation_weight=1.0,
                utility_weight=0.5,
                soft_threshold=0.2,
                assignment_strategy="composite"
            )
        }
    ]
    
    results = []
    
    print(f"\n2️⃣ Testing {len(test_configs)} configurations...")
    
    for i, test in enumerate(test_configs, 1):
        print(f"\n   🧪 Test {i}: {test['name']}")
        
        start_time = time.time()
        
        try:
            # Run simulation
            sim = Simulation(test['config'], workers_df, tasks_df)
            sim_results = sim.run()
            
            duration = time.time() - start_time
            
            # Extract key metrics
            tar = sim_results.get('task_assignment_ratio', 0.0) * 100
            jfi = sim_results.get('jfi', 0.0)
            wait_time = sim_results.get('avg_wait_time_minutes', 0.0)
            
            result = {
                'name': test['name'],
                'tar_percent': tar,
                'jfi': jfi,
                'wait_time_min': wait_time,
                'duration_sec': duration,
                'status': 'SUCCESS'
            }
            
            print(f"      ✅ TAR: {tar:.1f}%")
            print(f"      ✅ JFI: {jfi:.3f}") 
            print(f"      ✅ Wait: {wait_time:.1f} min")
            print(f"      ✅ Time: {duration:.1f}s")
            
        except Exception as e:
            result = {
                'name': test['name'],
                'status': 'FAILED',
                'error': str(e),
                'duration_sec': time.time() - start_time
            }
            print(f"      ❌ FAILED: {str(e)}")
        
        results.append(result)
    
    # Summary analysis
    print(f"\n3️⃣ OPERATIONAL TEST SUMMARY")
    print("=" * 50)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✅ Successful tests: {len(successful)}/{len(results)}")
    if failed:
        print(f"❌ Failed tests: {len(failed)}")
        for fail in failed:
            print(f"   • {fail['name']}: {fail['error']}")
    
    if successful:
        print(f"\n📊 Performance Summary:")
        for result in successful:
            print(f"   {result['name']:<20}: TAR={result['tar_percent']:>5.1f}%, JFI={result['jfi']:>5.3f}, Wait={result['wait_time_min']:>4.1f}min")
    
    # System health checks
    print(f"\n🏥 System Health Checks:")
    
    health_status = "HEALTHY"
    issues = []
    
    if successful:
        # Check for realistic metrics
        avg_tar = sum(r['tar_percent'] for r in successful) / len(successful)
        avg_wait = sum(r['wait_time_min'] for r in successful) / len(successful)
        
        if avg_tar < 50:
            issues.append(f"Low TAR: {avg_tar:.1f}% (expected >70%)")
            health_status = "CONCERN"
            
        if avg_wait == 0:
            issues.append("Zero wait time (unrealistic)")
            health_status = "ISSUE"
            
        if avg_wait > 30:
            issues.append(f"High wait time: {avg_wait:.1f}min (expected <15min)")
            health_status = "CONCERN"
        
        print(f"   📈 Average TAR: {avg_tar:.1f}% {'✅' if avg_tar >= 70 else '⚠️'}")
        print(f"   ⏱️  Average Wait: {avg_wait:.1f} min {'✅' if 1 <= avg_wait <= 15 else '⚠️'}")
        print(f"   🎯 All strategies working: {'✅' if len(successful) == len(results) else '❌'}")
        
    else:
        health_status = "CRITICAL"
        issues.append("No successful test runs")
    
    print(f"\n🎯 FINAL STATUS: {health_status}")
    if issues:
        print("⚠️  Issues detected:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ All systems operational!")
        print("🚀 Ready for parameter sweeping experiments!")
    
    return health_status == "HEALTHY"

if __name__ == "__main__":
    success = operational_test()
    sys.exit(0 if success else 1)
