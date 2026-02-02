"""Test multi-batch concurrent training functionality."""
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environment.environment import MultiTaskAllocationEnv

def test_multi_batch_setup():
    """Test that set_multi_batch correctly combines multiple batches with release times."""
    # Paths to the agents and tasks files
    agents_file = Path(__file__).resolve().parent.parent / "data" / "agents.npy"
    batch0_file = Path(__file__).resolve().parent.parent / "data" / "tasks_batch_0.npy"
    batch1_file = Path(__file__).resolve().parent.parent / "data" / "tasks_batch_1.npy"
    
    # Load agents and tasks
    agents = np.load(agents_file, allow_pickle=True)
    batch0 = np.load(batch0_file, allow_pickle=True)
    batch1 = np.load(batch1_file, allow_pickle=True)
    
    # Initialize environment with first batch
    env = MultiTaskAllocationEnv(agents, batch0)
    
    # Set up multi-batch with different release times
    # Batch 0 releases at time 0, Batch 1 releases at time 50
    task_batches_with_release_times = [
        (batch0, 0),
        (batch1, 50)
    ]
    
    env.set_multi_batch(task_batches_with_release_times)
    
    # Reset and check tasks
    obs, _ = env.reset()
    
    print(f"Total tasks loaded: {len(env.tasks)}")
    print(f"Expected tasks: {len(batch0) + len(batch1)}")
    
    # Verify all tasks are loaded
    assert len(env.tasks) == len(batch0) + len(batch1), "All tasks should be loaded"
    
    # Check release times at time 0
    available_at_0 = [t for t in env.tasks if t.is_available(0)]
    print(f"Tasks available at time 0: {len(available_at_0)}")
    assert len(available_at_0) == len(batch0), f"Only batch 0 tasks should be available at time 0"
    
    # Check release times at time 50
    available_at_50 = [t for t in env.tasks if t.is_available(50)]
    print(f"Tasks available at time 50: {len(available_at_50)}")
    assert len(available_at_50) == len(batch0) + len(batch1), "All tasks should be available at time 50"
    
    print("\nMulti-batch setup test passed!")
    
    # Close the environment
    env.close()

def test_backward_compatibility():
    """Test that the environment still works without multi-batch setup."""
    agents_file = Path(__file__).resolve().parent.parent / "data" / "agents.npy"
    batch0_file = Path(__file__).resolve().parent.parent / "data" / "tasks_batch_0.npy"
    
    agents = np.load(agents_file, allow_pickle=True)
    tasks = np.load(batch0_file, allow_pickle=True)
    
    # Initialize environment normally (no multi-batch)
    env = MultiTaskAllocationEnv(agents, tasks)
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Verify all tasks are loaded
    assert len(env.tasks) == len(tasks), "All tasks should be loaded"
    
    # In backward compatible mode, tasks use their original release times from the data file
    # Check that tasks have their original release times (not modified)
    original_release_times = [int(task[7]) for task in tasks]  # t_release is at index 7
    env_release_times = [t.release_time for t in env.tasks]
    
    # Release times should match the original data
    for orig, env_time in zip(original_release_times, env_release_times):
        assert env_time == orig, f"Release times should match original: {env_time} vs {orig}"
    
    # All tasks with release_time <= 0 should be available at time 0
    available_at_0 = [t for t in env.tasks if t.is_available(0)]
    expected_at_0 = [t for t in env.tasks if t.release_time <= 0]
    assert len(available_at_0) == len(expected_at_0), \
        f"Expected {len(expected_at_0)} tasks at time 0, got {len(available_at_0)}"
    
    print(f"Tasks available at time 0 (backward compat): {len(available_at_0)}/{len(env.tasks)}")
    print(f"Release times preserved from original data: {original_release_times[:3]}...")
    print("\nBackward compatibility test passed!")
    
    env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing multi-batch functionality")
    print("=" * 60)
    test_multi_batch_setup()
    print()
    print("=" * 60)
    print("Testing backward compatibility")
    print("=" * 60)
    test_backward_compatibility()
