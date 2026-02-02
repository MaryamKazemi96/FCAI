# Multi-Robot Task Allocation Project

This project implements a multi-robot task allocation system using a centralized training and decentralized execution framework. The system is designed to efficiently manage tasks and agents in a shared environment, utilizing A* planning for pathfinding and reinforcement learning for decision-making.

## Features

### Batch Release Times
The system supports dynamic batch release times, allowing tasks to become available at different points during training:
- Each batch of tasks has a release time (e.g., batch 0 at time 0, batch 1 at time 30, batch 2 at time 60, etc.)
- Tasks become available only when the current time step reaches or exceeds their release time
- The environment persists state across batch releases (robots maintain their capacity, position, etc.)
- Episodes end when all tasks across all batches are completed or the maximum step count is reached

## Usage

### Training with All Batches (Dynamic Release Times)

To train with all batches loaded at once with dynamic release times:

```bash
python3 main_all_batches.py \
  --n-batches 10 \
  --episodes 200 \
  --data-dir data \
  --save-dir checkpoints_all_batches
```

### Training with Individual Batches (Original Method)

To train on individual batches separately:

```bash
python3 main_episodic.py \
  --tasks data \
  --episodes-per-batch 1000 \
  --save-dir checkpoints
```

### Generating Data with Release Times

Generate task batches with release times:

```bash
python3 -m src.data_generation.generate_data \
  --n-batches 10 \
  --n-tasks 10 \
  --n-robots 5 \
  --release-interval 50 \
  --output-dir data
```

This will create batch files with release times: batch 0 at time 0, batch 1 at time 50, batch 2 at time 100, etc.
