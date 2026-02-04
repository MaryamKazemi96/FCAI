#!/usr/bin/env python3
"""
main_all_batches.py - Train with all batches loaded at once, released dynamically

This script loads all task batches at initialization and trains them together,
with tasks becoming available based on their release times during training.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
import random
import time

from src.environment.environment import MultiTaskAllocationEnv, TASK_RELEASE_TIME_INDEX
from src.models.actor_critic import ActorGNN, CriticGNN
from src.training.train_actor_critic import train

# --- Utilities ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_models(save_dir: Path, actors: dict, critic: torch.nn.Module):
    save_dir.mkdir(parents=True, exist_ok=True)
    for rid, actor in actors.items():
        torch.save(actor.state_dict(), save_dir / f"actor_{rid}.pt")
    torch.save(critic.state_dict(), save_dir / "critic.pt")

def plot_rewards(save_dir: Path, episode_rewards):
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (sum over robots)")
        plt.title("Training Rewards - All Batches")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "rewards.png")
        plt.close()
    except Exception as e:
        print("Warning: failed to plot rewards:", e)

def load_all_batches(data_dir: Path, n_batches: int):
    """Load all batch files and return them as a list."""
    batches = []
    for i in range(n_batches):
        batch_file = data_dir / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batch = np.load(batch_file, allow_pickle=True)
            batches.append(batch)
            print(f"Loaded batch {i}: {len(batch)} tasks, release_time={batch[0][TASK_RELEASE_TIME_INDEX] if len(batch) > 0 else 'N/A'}")
        else:
            print(f"Warning: {batch_file} not found")
    return batches

# --- Main ---
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    # Load agents
    agents_file = Path(args.agents)
    agents = np.load(agents_file, allow_pickle=True)
    print(f"Loaded {len(agents)} agents")

    # Load all task batches
    data_dir = Path(args.data_dir)
    all_batches = load_all_batches(data_dir, args.n_batches)
    
    if len(all_batches) == 0:
        raise ValueError("No batches loaded! Check data directory and n_batches parameter.")
    
    print(f"\nTotal batches loaded: {len(all_batches)}")
    total_tasks = sum(len(batch) for batch in all_batches)
    print(f"Total tasks across all batches: {total_tasks}")

    # Create environment with all batches
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=all_batches,  # Pass list of batches
        radius=args.radius,
        feature_size=args.feature_size,
        use_true_id=args.use_true_id,
        all_batches=True  # Enable all_batches mode
    )
    
    print(f"Environment created with {env.n_tasks} tasks total")
    print(f"Max steps per episode (batch_time): {env.batch_time}")
    # Save results and models
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Model hyperparams
    input_dim = args.feature_size
    hidden_dim = args.hidden_dim
    critic_aggregation = args.critic_agg

    # Create actors (one per robot) and optimizers
    num_robots = env.n_robots
    actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
    optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
                         for rid, actor in actors.items()}

    # Shared critic
    critic = CriticGNN(input_dim, hidden_dim, critic_aggregation).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    t0 = time.time()
    episode_rewards = train(
    env,
    num_episodes=args.episodes,
    actors=actors,
    critic=critic,
    optimizers_actors=optimizers_actors,
    optimizer_critic=optimizer_critic,
    gamma=args.gamma,
    max_steps_per_episode=args.max_steps if args.max_steps > 0 else env.batch_time,
    device=device,
    verbose=args.verbose,
    save_dir=save_dir,
    save_every=10,  # Save every 10 episodes
    plot_rewards_fn=plot_rewards,
    save_models_fn=save_models
)
    t1 = time.time()
    print(f"\nTraining finished in {t1 - t0:.1f}s")

    
    
    # Save models
    save_models(save_dir, actors, critic)
    print(f"Saved models to {save_dir}")
    
    # Save rewards
    with open(save_dir / "episode_rewards.json", "w") as f:
        json.dump([float(x) for x in episode_rewards], f)
    
    # Plot rewards
    plot_rewards(save_dir, episode_rewards)
    print(f"Saved rewards and plots to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-robot task allocation with all batches loaded at once"
    )
    parser.add_argument("--agents", type=str, default="data/agents.npy",
                        help="Path to agents.npy file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing task batch files")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of batches to load")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=700,
                        help="Max steps per episode (0 = use env.batch_time)")
    parser.add_argument("--feature-size", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument("--critic-agg", type=str, default="per_robot",
                        choices=["per_robot", "joint_mean", "joint_attn"])
    parser.add_argument("--use-true-id", action="store_true", dest="use_true_id")
    parser.add_argument("--save-dir", type=str, default="checkpoints_all_batches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    main(args)
