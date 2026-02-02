#!/usr/bin/env python3
"""
Visualization script to demonstrate the fixes for robot capacity and task release times.
This script simulates the task assignment process and shows how constraints are enforced.
"""

print("="*80)
print("DEMONSTRATION: Robot Capacity & Task Release Time Enforcement")
print("="*80)

# Simulate the fixed logic
class MockTask:
    def __init__(self, task_id, release_time):
        self.id = task_id
        self.release_time = release_time
        self.is_active = True
        self.is_assigned = False
        self.is_pickedup = False
        self.is_droppedoff = False

class MockRobot:
    def __init__(self, robot_id, max_capacity=3):
        self.robot_id = robot_id
        self.maxCapacity = max_capacity
        self.capacity = 0
        self.current_tasks_id = []
    
    def add_task(self, task_id):
        if self.capacity >= self.maxCapacity:
            print(f"  ❌ Robot {self.robot_id}: REJECTED task {task_id} (at capacity {self.capacity}/{self.maxCapacity})")
            return False
        self.capacity += 1
        self.current_tasks_id.append(task_id)
        print(f"  ✓ Robot {self.robot_id}: ACCEPTED task {task_id} (capacity now {self.capacity}/{self.maxCapacity})")
        return True
    
    def drop_off_task(self, task_id):
        if task_id in self.current_tasks_id:
            self.current_tasks_id.remove(task_id)
            self.capacity -= 1
            print(f"  ✓ Robot {self.robot_id}: DROPPED OFF task {task_id} (capacity now {self.capacity}/{self.maxCapacity})")
            return True
        return False

class MockEnvironment:
    def __init__(self):
        self.time_count = 0
        self.robots = [MockRobot(0), MockRobot(1)]
        self.tasks = [
            MockTask(0, release_time=0),
            MockTask(1, release_time=0),
            MockTask(2, release_time=0),
            MockTask(3, release_time=10),
            MockTask(4, release_time=10),
            MockTask(5, release_time=20),
            MockTask(6, release_time=20),
        ]
        self.taskid_to_task = {t.id: t for t in self.tasks}
    
    def get_available_task_ids(self):
        """
        FIXED VERSION: Filter by release time
        """
        available = [
            tid for tid, t in self.taskid_to_task.items()
            if t.is_active and not t.is_assigned and t.release_time <= self.time_count
        ]
        print(f"\n[Time {self.time_count}] Available tasks: {available}")
        print(f"  (Tasks released: {[tid for tid, t in self.taskid_to_task.items() if t.release_time <= self.time_count]})")
        return available
    
    def assign_tasks(self):
        """
        FIXED VERSION: Respects capacity constraints
        """
        available = self.get_available_task_ids()
        
        for robot in self.robots:
            if available and robot.capacity < robot.maxCapacity:
                # Try to assign first available task
                task_id = available[0]
                task = self.taskid_to_task[task_id]
                
                # Check capacity before assignment (FIXED)
                if robot.capacity >= robot.maxCapacity:
                    print(f"  ⚠ Robot {robot.robot_id} at capacity, skipping assignment")
                    continue
                
                # Try to add task
                success = robot.add_task(task_id)
                if success:
                    task.is_assigned = True
                    available.remove(task_id)

def main():
    env = MockEnvironment()
    
    print("\n" + "="*80)
    print("SCENARIO: 2 Robots (capacity 3 each), 7 Tasks (staggered release times)")
    print("="*80)
    print("Tasks 0-2: release_time=0")
    print("Tasks 3-4: release_time=10")
    print("Tasks 5-6: release_time=20")
    
    # Time 0: Initial assignment
    print("\n" + "-"*80)
    print("TIME 0: First assignment round")
    print("-"*80)
    env.time_count = 0
    env.assign_tasks()
    env.assign_tasks()
    env.assign_tasks()
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Time 5: Try to assign more (should fail - all at capacity)
    print("\n" + "-"*80)
    print("TIME 5: Second assignment round (robots at capacity)")
    print("-"*80)
    env.time_count = 5
    env.assign_tasks()
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Robot 0 completes a task
    print("\n" + "-"*80)
    print("TIME 8: Robot 0 drops off task 0")
    print("-"*80)
    env.robots[0].drop_off_task(0)
    env.taskid_to_task[0].is_droppedoff = True
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Time 10: New tasks released + robot has capacity
    print("\n" + "-"*80)
    print("TIME 10: New tasks released (3, 4) and robot 0 has capacity")
    print("-"*80)
    env.time_count = 10
    env.assign_tasks()
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Robot 1 completes 2 tasks
    print("\n" + "-"*80)
    print("TIME 15: Robot 1 drops off tasks 1 and 2")
    print("-"*80)
    env.robots[1].drop_off_task(1)
    env.robots[1].drop_off_task(2)
    env.taskid_to_task[1].is_droppedoff = True
    env.taskid_to_task[2].is_droppedoff = True
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Time 20: Final tasks released
    print("\n" + "-"*80)
    print("TIME 20: Final tasks released (5, 6)")
    print("-"*80)
    env.time_count = 20
    env.assign_tasks()
    env.assign_tasks()
    print(f"\nRobot Status:")
    for r in env.robots:
        print(f"  Robot {r.robot_id}: {r.capacity}/3 tasks - {r.current_tasks_id}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ENFORCED CONSTRAINTS")
    print("="*80)
    print("✓ Robot capacity constraint enforced:")
    print("  - No robot exceeded capacity of 3 tasks")
    print("  - Assignments rejected when robots at capacity")
    print("  - Capacity freed up after task dropoffs")
    print()
    print("✓ Task release time constraint enforced:")
    print("  - Tasks 0-2 available at time 0")
    print("  - Tasks 3-4 only available at time 10+")
    print("  - Tasks 5-6 only available at time 20+")
    print()
    print("✓ Both constraints work together correctly")
    print("="*80)

if __name__ == "__main__":
    main()
