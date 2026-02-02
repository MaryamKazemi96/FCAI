"""
Test script to verify robot capacity constraints and task release time enforcement.
"""
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environment.environment import MultiTaskAllocationEnv


def test_robot_capacity_constraint():
    """Test that robots cannot exceed their capacity of 3 tasks."""
    print("\n" + "="*80)
    print("TEST 1: Robot Capacity Constraint (max 3 tasks)")
    print("="*80)
    
    # Create simple test data
    # 2 robots at different positions
    agents = np.array([
        [0, 100, 100, 0],  # robot 0
        [1, 200, 200, 0],  # robot 1
    ])
    
    # Create 10 tasks all released at time 0 near robot 0
    tasks = []
    for i in range(10):
        # [task_id, w_origin, h_origin, yaw_origin, w_destination, h_destination, yaw_destination, 
        #  t_release, pickupddl, estimatedTravelTime, dropoff_deadline]
        task = [
            i,      # task_id
            105 + i, 105 + i, 0,  # pickup location (near robot 0)
            150 + i, 150 + i, 0,  # dropoff location
            0,      # t_release (all released at time 0)
            60,     # pickupddl
            10,     # estimatedTravelTime
            120     # dropoff_deadline
        ]
        tasks.append(task)
    tasks = np.array(tasks)
    
    # Initialize environment
    env = MultiTaskAllocationEnv(agents, tasks)
    
    # Verify capacity is set to 3
    assert env.robot_capacity == 3, f"Expected robot_capacity=3, got {env.robot_capacity}"
    print(f"✓ Environment robot capacity correctly set to: {env.robot_capacity}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Verify robots have correct capacity
    for robot in env.robots:
        assert robot.maxCapacity == 3, f"Robot {robot.robot_id} has maxCapacity={robot.maxCapacity}, expected 3"
        assert robot.capacity == 0, f"Robot {robot.robot_id} has capacity={robot.capacity}, expected 0 initially"
    print(f"✓ All {len(env.robots)} robots initialized with maxCapacity=3 and capacity=0")
    
    # Try to assign multiple tasks to robot 0
    robot_0 = env.robots[0]
    print(f"\nAttempting to assign 5 tasks to robot 0 (should only accept 3)...")
    
    successful_assignments = 0
    for i in range(5):
        task = env.tasks[i]
        success = robot_0.add_task(task.id, task.pick_up_coord, task.drop_off_coord)
        if success:
            successful_assignments += 1
            print(f"  Task {i} assigned successfully. Robot capacity: {robot_0.capacity}/{robot_0.maxCapacity}")
        else:
            print(f"  Task {i} assignment REJECTED (robot at capacity)")
    
    # Verify capacity constraint was enforced
    assert successful_assignments == 3, f"Expected 3 successful assignments, got {successful_assignments}"
    assert robot_0.capacity == 3, f"Expected final capacity=3, got {robot_0.capacity}"
    print(f"✓ Capacity constraint enforced: Only {successful_assignments} tasks assigned (max capacity reached)")
    print("✓ TEST 1 PASSED: Robot capacity constraint is properly enforced\n")


def test_task_release_time():
    """Test that tasks are only available after their release time."""
    print("\n" + "="*80)
    print("TEST 2: Task Release Time Enforcement")
    print("="*80)
    
    # Create simple test data
    agents = np.array([
        [0, 100, 100, 0],
    ])
    
    # Create tasks with different release times
    tasks = []
    release_times = [0, 10, 20, 30, 40]  # Different release times
    for i, release_time in enumerate(release_times):
        task = [
            i,      # task_id
            105 + i, 105 + i, 0,  # pickup location
            150 + i, 150 + i, 0,  # dropoff location
            release_time,  # t_release (varied)
            release_time + 60,     # pickupddl
            10,     # estimatedTravelTime
            release_time + 120     # dropoff_deadline
        ]
        tasks.append(task)
    tasks = np.array(tasks)
    
    # Initialize environment
    env = MultiTaskAllocationEnv(agents, tasks)
    obs, _ = env.reset()
    
    print(f"Created {len(tasks)} tasks with release times: {release_times}")
    print(f"Task release times in environment:")
    for task in env.tasks:
        print(f"  Task {task.id}: release_time={task.release_time}")
    
    # Test at time 0
    print(f"\nAt time_count=0:")
    available = env.get_available_task_ids()
    print(f"  Available tasks: {available}")
    expected_at_0 = [0]  # Only task 0 should be available
    assert available == expected_at_0, f"Expected {expected_at_0}, got {available}"
    print(f"  ✓ Correct: Only task(s) with release_time <= 0 are available")
    
    # Advance time to 15
    env.time_count = 15
    print(f"\nAt time_count=15:")
    available = env.get_available_task_ids()
    print(f"  Available tasks: {available}")
    expected_at_15 = [0, 1]  # Tasks 0 and 1 should be available (release times 0 and 10)
    assert sorted(available) == sorted(expected_at_15), f"Expected {expected_at_15}, got {available}"
    print(f"  ✓ Correct: Only task(s) with release_time <= 15 are available")
    
    # Advance time to 25
    env.time_count = 25
    print(f"\nAt time_count=25:")
    available = env.get_available_task_ids()
    print(f"  Available tasks: {available}")
    expected_at_25 = [0, 1, 2]  # Tasks 0, 1, 2 should be available
    assert sorted(available) == sorted(expected_at_25), f"Expected {expected_at_25}, got {available}"
    print(f"  ✓ Correct: Only task(s) with release_time <= 25 are available")
    
    # Advance time to 50 (all should be available)
    env.time_count = 50
    print(f"\nAt time_count=50:")
    available = env.get_available_task_ids()
    print(f"  Available tasks: {available}")
    expected_at_50 = [0, 1, 2, 3, 4]  # All tasks should be available
    assert sorted(available) == sorted(expected_at_50), f"Expected {expected_at_50}, got {available}"
    print(f"  ✓ Correct: All tasks with release_time <= 50 are available")
    
    print("✓ TEST 2 PASSED: Task release time is properly enforced\n")


def test_integrated_capacity_and_release():
    """Test that both capacity and release time constraints work together during assignment."""
    print("\n" + "="*80)
    print("TEST 3: Integrated Test - Capacity + Release Time")
    print("="*80)
    
    # Create test data
    agents = np.array([
        [0, 100, 100, 0],
        [1, 200, 200, 0],
    ])
    
    # Create tasks with staggered release times
    tasks = []
    for i in range(8):
        release_time = (i // 2) * 10  # Tasks 0-1 at time 0, 2-3 at time 10, etc.
        task = [
            i, 105 + i, 105 + i, 0, 150 + i, 150 + i, 0,
            release_time, release_time + 60, 10, release_time + 120
        ]
        tasks.append(task)
    tasks = np.array(tasks)
    
    env = MultiTaskAllocationEnv(agents, tasks)
    obs, _ = env.reset()
    
    print(f"Created 2 robots with capacity 3 each")
    print(f"Created 8 tasks with release times: {[t.release_time for t in env.tasks]}")
    
    # At time 0, only tasks 0-1 available
    print(f"\n--- Time 0 ---")
    available = env.get_available_task_ids()
    print(f"Available tasks: {available} (expected: [0, 1])")
    assert len(available) == 2, f"Expected 2 tasks available at time 0, got {len(available)}"
    
    # Assign available tasks to robot 0
    assignments = {0: available[0], 0: available[1]} if len(available) >= 2 else {}
    for tid in available[:2]:
        task = env.taskid_to_task[tid]
        success = env.robots[0].add_task(task.id, task.pick_up_coord, task.drop_off_coord)
        if success:
            task.is_assigned = True
            print(f"  Assigned task {tid} to robot 0, capacity now: {env.robots[0].capacity}/3")
    
    # Advance time to 10
    env.time_count = 10
    print(f"\n--- Time 10 ---")
    available = env.get_available_task_ids()
    print(f"Available tasks: {available}")
    print(f"Robot 0 capacity: {env.robots[0].capacity}/3")
    print(f"Robot 1 capacity: {env.robots[1].capacity}/3")
    
    # Try to assign more tasks
    unassigned = [tid for tid in available if not env.taskid_to_task[tid].is_assigned]
    print(f"Unassigned available tasks: {unassigned}")
    
    # Robot 0 should be able to take 1 more task (at capacity 2, max 3)
    if env.robots[0].capacity < 3 and len(unassigned) > 0:
        task = env.taskid_to_task[unassigned[0]]
        success = env.robots[0].add_task(task.id, task.pick_up_coord, task.drop_off_coord)
        if success:
            task.is_assigned = True
            print(f"  Assigned task {unassigned[0]} to robot 0, capacity now: {env.robots[0].capacity}/3")
    
    # Try to assign one more - should fail if at capacity
    if len(unassigned) > 1:
        task = env.taskid_to_task[unassigned[1]]
        success = env.robots[0].add_task(task.id, task.pick_up_coord, task.drop_off_coord)
        if success:
            print(f"  Assigned task {unassigned[1]} to robot 0, capacity now: {env.robots[0].capacity}/3")
        else:
            print(f"  Task {unassigned[1]} assignment to robot 0 REJECTED (at capacity)")
            # Should succeed with robot 1
            success = env.robots[1].add_task(task.id, task.pick_up_coord, task.drop_off_coord)
            if success:
                task.is_assigned = True
                print(f"  Assigned task {unassigned[1]} to robot 1 instead, capacity now: {env.robots[1].capacity}/3")
    
    print(f"\nFinal state:")
    print(f"  Robot 0: {env.robots[0].capacity}/3 tasks")
    print(f"  Robot 1: {env.robots[1].capacity}/3 tasks")
    
    # Verify no robot exceeded capacity
    for robot in env.robots:
        assert robot.capacity <= 3, f"Robot {robot.robot_id} exceeded capacity: {robot.capacity}/3"
    print("✓ TEST 3 PASSED: Both constraints work correctly together\n")


if __name__ == "__main__":
    test_robot_capacity_constraint()
    test_task_release_time()
    test_integrated_capacity_and_release()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print("✓ Robot capacity constraint is enforced (max 3 tasks per robot)")
    print("✓ Task release times are respected (tasks only available after release)")
    print("✓ Both constraints work together correctly in integrated scenarios")
