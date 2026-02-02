# Multi-Batch Concurrent Training

This document explains the multi-batch concurrent training feature added to the FCAI repository.

## Overview

The multi-batch concurrent training feature allows you to train on multiple task batches within a single episode, where each batch becomes available at a specified release time. This is useful for training on complex scenarios where tasks arrive in waves over time.

## Key Changes

### 1. Environment (`src/environment/environment.py`)

#### New Method: `is_available(current_time)`
Added to `Tasks_variable` class to check if a task is available based on its release time:
```python
def is_available(self, current_time=0):
    """Check if task is available based on release_time and current state."""
    if self.is_pickedup:
        return False
    # Task is available if current time has reached its release time
    return current_time >= self.release_time
```

#### New Method: `set_multi_batch(task_batches_with_release_times)`
Added to `MultiTaskAllocationEnv` to configure multiple batches with release times:
```python
task_batches_with_release_times = [
    (batch0_tasks, 0),      # Batch 0 releases at time 0
    (batch1_tasks, 200),    # Batch 1 releases at time 200
    (batch2_tasks, 400),    # Batch 2 releases at time 400
    # ... more batches
]
env.set_multi_batch(task_batches_with_release_times)
```

#### Updated Graph Filtering
The `update_graph()` method now filters out tasks that are:
- Already assigned, OR
- Not yet available (current time < release_time)

This ensures robots only see tasks that are both available and unassigned in their ego graphs.

### 2. Training (`src/training/train_actor_critic.py`)

#### Updated Function Signature
The `train()` function now accepts an optional `task_batches_with_release_times` parameter:
```python
def train(env, num_episodes, actors, critic,
          optimizers_actors, optimizer_critic, gamma=0.99,
          max_steps_per_episode=2000,  # Increased from 200
          device=None, verbose=True,
          task_batches_with_release_times=None):  # New parameter
```

#### Increased Default Episode Length
- `max_steps_per_episode` increased from 200 to 2000
- This accommodates 10 batches × 200 steps per batch

#### Backward Compatibility
If `task_batches_with_release_times` is `None`, the training function uses the environment's current task configuration (original behavior).

## Usage

### Example 1: Multi-Batch Training

```python
import numpy as np
from pathlib import Path
from src.environment.environment import MultiTaskAllocationEnv
from src.training.train_actor_critic import train

# Load agents and task batches
agents = np.load("data/agents.npy", allow_pickle=True)
batch0 = np.load("data/tasks_batch_0.npy", allow_pickle=True)
batch1 = np.load("data/tasks_batch_1.npy", allow_pickle=True)
batch2 = np.load("data/tasks_batch_2.npy", allow_pickle=True)

# Create environment
env = MultiTaskAllocationEnv(agents, batch0)

# Setup multi-batch with release times
task_batches_with_release_times = [
    (batch0, 0),      # Available from start
    (batch1, 200),    # Available at time 200
    (batch2, 400),    # Available at time 400
]

# Train with concurrent batches
episode_rewards = train(
    env,
    num_episodes=100,
    actors=actors,
    critic=critic,
    optimizers_actors=optimizers_actors,
    optimizer_critic=optimizer_critic,
    max_steps_per_episode=1000,
    task_batches_with_release_times=task_batches_with_release_times
)
```

### Example 2: Backward Compatible (Single Batch)

```python
# Original usage still works - no changes needed
agents = np.load("data/agents.npy", allow_pickle=True)
tasks = np.load("data/tasks_batch_0.npy", allow_pickle=True)

env = MultiTaskAllocationEnv(agents, tasks)

# Train without multi-batch parameter (original behavior)
episode_rewards = train(
    env,
    num_episodes=100,
    actors=actors,
    critic=critic,
    optimizers_actors=optimizers_actors,
    optimizer_critic=optimizer_critic,
    max_steps_per_episode=200  # Can use original default
)
```

## Running the Example Script

A complete example script is provided in `example_multi_batch_training.py`:

```bash
# Train on 10 batches with 200 time-step intervals between releases
python example_multi_batch_training.py --num-batches 10 --batch-release-interval 200 --episodes 10

# Train on 5 batches with custom settings
python example_multi_batch_training.py --num-batches 5 --batch-release-interval 300 --episodes 20 --max-steps 1500
```

## Testing

Run the multi-batch test suite:

```bash
# Test multi-batch functionality
python tests/test_multi_batch.py

# Run all tests
python -m pytest tests/
```

## Implementation Details

### Task Release Time Flow

1. **Batch Setup**: When `set_multi_batch()` is called, all tasks from all batches are combined into a single task array, with each task's `t_release` field set to its batch's release time.

2. **Episode Start**: At the beginning of each episode, if `task_batches_with_release_times` is provided, the environment is configured with all batches.

3. **During Episode**: 
   - At each step, `update_graph()` filters tasks based on `is_available(time_count)`
   - Only tasks with `release_time <= time_count` appear in robot ego graphs
   - Robots can only select from available tasks

4. **Episode End**: Episode terminates when either:
   - All tasks across all batches are completed/obsolete, OR
   - `max_steps_per_episode` is reached

### Episode Length Considerations

For N batches with average M tasks each and T timesteps per task:
- Recommended `max_steps_per_episode` = N × M × T
- Example: 10 batches × 10 tasks × 20 steps = 2000 steps

## Benefits

1. **Realistic Training**: Simulates scenarios where tasks arrive over time
2. **Efficient Training**: Trains on multiple batches in a single episode
3. **Flexible Scheduling**: Control when each batch becomes available
4. **Backward Compatible**: Existing code continues to work without changes

## Notes

- Tasks in a batch become available simultaneously when their batch release time is reached
- Individual task deadlines are still respected (from task's original configuration)
- The environment's `time_count` is used to check task availability at each step
- Ego graphs dynamically update to show only available tasks within robot radius
