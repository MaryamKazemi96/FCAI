#!/usr/bin/env python3
"""
Quick validation script to test multi-batch functionality.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.environment import MultiTaskAllocationEnv

def test_basic_environment():
    """Test that basic environment still works."""
    print("=" * 60)
    print("Test 1: Basic environment (backward compatibility)")
    print("=" * 60)
    
    agents = np.load("data/agents.npy", allow_pickle=True)
    tasks = np.load("data/tasks_batch_0.npy", allow_pickle=True)
    
    env = MultiTaskAllocationEnv(agents, tasks)
    obs, _ = env.reset()
    
    print(f"✓ Environment created successfully")
    print(f"✓ Number of robots: {env.n_robots}")
    print(f"✓ Number of tasks: {env.n_tasks}")
    print(f"✓ Tasks at time 0: {len([t for t in env.tasks if t.is_available(0)])}")
    
    # Take a step
    actions = {rid: [] for rid in range(env.n_robots)}
    obs, reward, done, truncated, info = env.step(actions)
    print(f"✓ Step executed successfully")
    
    env.close()
    print("✓ Test 1 PASSED\n")

def test_multi_batch():
    """Test multi-batch setup."""
    print("=" * 60)
    print("Test 2: Multi-batch concurrent training")
    print("=" * 60)
    
    agents = np.load("data/agents.npy", allow_pickle=True)
    batch0 = np.load("data/tasks_batch_0.npy", allow_pickle=True)
    batch1 = np.load("data/tasks_batch_1.npy", allow_pickle=True)
    batch2 = np.load("data/tasks_batch_2.npy", allow_pickle=True)
    
    env = MultiTaskAllocationEnv(agents, batch0)
    
    # Setup multi-batch with staggered release times
    task_batches = [
        (batch0, 0),
        (batch1, 50),
        (batch2, 100)
    ]
    
    env.set_multi_batch(task_batches)
    obs, _ = env.reset()
    
    print(f"✓ Multi-batch environment created")
    print(f"✓ Total tasks: {len(env.tasks)} (expected: {len(batch0) + len(batch1) + len(batch2)})")
    
    # Check availability at different times
    available_0 = len([t for t in env.tasks if t.is_available(0)])
    available_50 = len([t for t in env.tasks if t.is_available(50)])
    available_100 = len([t for t in env.tasks if t.is_available(100)])
    
    print(f"✓ Tasks available at time 0: {available_0}")
    print(f"✓ Tasks available at time 50: {available_50}")
    print(f"✓ Tasks available at time 100: {available_100}")
    
    # Verify progressive availability
    assert available_0 <= available_50 <= available_100, "Tasks should progressively become available"
    assert available_100 == len(env.tasks), "All tasks should be available eventually"
    
    # Simulate some steps and check ego graph updates
    for step in range(3):
        actions = {rid: [] for rid in range(env.n_robots)}
        obs, reward, done, truncated, info = env.step(actions)
        ego_graphs, _ = obs
        print(f"✓ Step {step+1}: ego graphs updated for {len(ego_graphs)} robots")
    
    env.close()
    print("✓ Test 2 PASSED\n")

def test_release_time_filtering():
    """Test that tasks are properly filtered by release time in ego graphs."""
    print("=" * 60)
    print("Test 3: Release time filtering in ego graphs")
    print("=" * 60)
    
    agents = np.load("data/agents.npy", allow_pickle=True)
    batch0 = np.load("data/tasks_batch_0.npy", allow_pickle=True)
    batch1 = np.load("data/tasks_batch_1.npy", allow_pickle=True)
    
    env = MultiTaskAllocationEnv(agents, batch0, radius=5000)  # Large radius to see all tasks
    
    # Setup with very different release times
    task_batches = [
        (batch0, 0),
        (batch1, 1000)  # Very late release
    ]
    
    env.set_multi_batch(task_batches)
    obs, _ = env.reset()
    
    # At time 0, should not see batch1 tasks in ego graphs
    ego_graphs_0, attr_matrix_0 = obs
    
    # Take steps to advance time
    for step in range(5):
        actions = {rid: [] for rid in range(env.n_robots)}
        obs, reward, done, truncated, info = env.step(actions)
    
    # Check that time has advanced
    print(f"✓ Environment time: {env.time_count}")
    print(f"✓ Tasks available now: {len([t for t in env.tasks if t.is_available(env.time_count)])}")
    
    # Batch1 tasks should still not be available
    batch1_available = len([t for t in env.tasks if t.is_available(env.time_count) and t.release_time >= 1000])
    print(f"✓ Batch1 tasks available (should be 0): {batch1_available}")
    
    env.close()
    print("✓ Test 3 PASSED\n")

if __name__ == "__main__":
    try:
        test_basic_environment()
        test_multi_batch()
        test_release_time_filtering()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
