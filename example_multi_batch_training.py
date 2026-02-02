#!/usr/bin/env python3
"""
Example: Multi-batch concurrent training with release times.

This demonstrates how to train on multiple task batches concurrently,
where each batch becomes available at a specified time during the episode.
"""
import torch
import numpy as np
from pathlib import Path
import argparse

from src.environment.environment import MultiTaskAllocationEnv
from src.models.actor_critic import ActorGNN, CriticGNN
from src.training.train_actor_critic import train


def main(args):
    # Load agents
    agents_file = Path(args.agents)
    agents = np.load(agents_file, allow_pickle=True)
    
    # Load multiple task batches
    batch_files = [Path(f"data/tasks_batch_{i}.npy") for i in range(args.num_batches)]
    batches = [np.load(f, allow_pickle=True) for f in batch_files if f.exists()]
    
    print(f"Loaded {len(batches)} task batches")
    
    # Create environment (will be configured with multi-batch in train function)
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches[0],  # Initial batch (will be replaced)
        radius=args.radius,
        feature_size=args.feature_size,
        use_true_id=False,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    input_dim = args.feature_size
    hidden_dim = args.hidden_dim
    num_robots = env.n_robots
    
    # Create actors and critic
    actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
    optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
                        for rid, actor in actors.items()}
    
    critic = CriticGNN(input_dim, hidden_dim, args.critic_agg).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    
    # Setup multi-batch with release times
    # Each batch releases at intervals (e.g., batch 0 at time 0, batch 1 at time 200, etc.)
    task_batches_with_release_times = [
        (batch, i * args.batch_release_interval) 
        for i, batch in enumerate(batches)
    ]
    
    print(f"\nBatch release schedule:")
    for i, (batch, release_time) in enumerate(task_batches_with_release_times):
        print(f"  Batch {i}: {len(batch)} tasks, releases at time {release_time}")
    
    # Train with concurrent multi-batch
    print(f"\nTraining for {args.episodes} episodes with concurrent multi-batch...")
    episode_rewards = train(
        env,
        num_episodes=args.episodes,
        actors=actors,
        critic=critic,
        optimizers_actors=optimizers_actors,
        optimizer_critic=optimizer_critic,
        gamma=args.gamma,
        max_steps_per_episode=args.max_steps,
        device=device,
        verbose=True,
        task_batches_with_release_times=task_batches_with_release_times
    )
    
    print(f"\nTraining complete!")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Final episode reward: {episode_rewards[-1]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-batch concurrent training example")
    parser.add_argument("--agents", type=str, default="data/agents.npy")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to load")
    parser.add_argument("--batch-release-interval", type=int, default=200, 
                       help="Time steps between batch releases")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000, 
                       help="Max steps per episode (default 2000 for 10 batches)")
    parser.add_argument("--feature-size", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument("--critic-agg", type=str, default="per_robot", 
                       choices=["per_robot", "joint_mean", "joint_attn"])
    
    args = parser.parse_args()
    main(args)
