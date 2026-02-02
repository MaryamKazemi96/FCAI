"""
Test for batch release times in data generation.
Verifies that generated batches have correct release times.
"""
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_generation.generate_data import DataGenerator


def test_batch_release_times():
    """Test that batches have correct release times based on batch index and interval."""
    # Create a simple mock planner for testing
    class MockPlanner:
        def is_point_valid(self, point):
            # Always return True for simplicity
            return True
        
        def get_plan(self, start, goal):
            # Return a simple path for testing
            return True, [(0, 0), (1, 1), (2, 2)]  # path with 3 steps
    
    # Initialize the data generator with mock planner
    generator = DataGenerator(
        x_min=-60, x_max=80, y_min=-40, y_max=20,
        max_waiting_time=10, max_travel_delay_percentage=0.1,
        planning_resolution=1, planner=MockPlanner(),
        origin_x=-60, origin_y=20
    )
    
    # Generate tasks with custom parameters
    n_batches = 5
    n_tasks = 3
    release_time_interval = 50
    
    batches = generator.generate_tasks(n_batches, n_tasks, release_time_interval)
    
    # Verify we got the right number of batches
    assert len(batches) == n_batches, f"Expected {n_batches} batches, got {len(batches)}"
    
    # Verify each batch has correct release times
    for batch_idx, batch in enumerate(batches):
        expected_release_time = batch_idx * release_time_interval
        
        # Each batch should have n_tasks tasks
        assert len(batch) == n_tasks, f"Batch {batch_idx} should have {n_tasks} tasks"
        
        # Check release time for each task in the batch (column 7)
        for task in batch:
            actual_release_time = task[7]
            assert actual_release_time == expected_release_time, \
                f"Batch {batch_idx}: Expected release_time {expected_release_time}, got {actual_release_time}"
    
    print("✓ All batch release times are correct!")


def test_existing_batch_files():
    """Test that existing batch files in data directory have correct release times."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Check if batch files exist
    batch_files = list(data_dir.glob("tasks_batch_*.npy"))
    if not batch_files:
        print("No batch files found, skipping this test")
        return
    
    # Expected interval is 30 (default)
    expected_interval = 30
    
    for batch_file in sorted(batch_files):
        # Extract batch number from filename
        batch_num = int(batch_file.stem.split("_")[-1])
        
        # Load the batch
        tasks = np.load(batch_file)
        
        # Check release times (column 7)
        expected_release_time = batch_num * expected_interval
        actual_release_times = tasks[:, 7]
        
        # All tasks in a batch should have the same release time
        assert np.all(actual_release_times == actual_release_times[0]), \
            f"Tasks in {batch_file.name} have different release times"
        
        # Verify the release time matches expected
        assert actual_release_times[0] == expected_release_time, \
            f"{batch_file.name}: Expected release_time {expected_release_time}, got {actual_release_times[0]}"
    
    print(f"✓ All {len(batch_files)} batch files have correct release times!")


if __name__ == "__main__":
    test_batch_release_times()
    test_existing_batch_files()
    print("\n✓✓ All tests passed!")
