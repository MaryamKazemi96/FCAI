# #!/usr/bin/env python3
# """
# main_all_batches.py - Train with all batches loaded at once, released dynamically

# This script loads all task batches at initialization and trains them together,
# with tasks becoming available based on their release times during training.
# """
# import torch
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# import argparse
# import json
# import random
# import time

# from src.environment.environment import MultiTaskAllocationEnv, TASK_RELEASE_TIME_INDEX
# from src.models.actor_critic import ActorGNN, CriticGNN
# from src.training.train_actor_critic import train

# # --- Utilities ---
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def save_models(save_dir: Path, actors: dict, critic: torch.nn.Module):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for rid, actor in actors.items():
#         torch.save(actor.state_dict(), save_dir / f"actor_{rid}.pt")
#     torch.save(critic.state_dict(), save_dir / "critic.pt")

# def plot_rewards(save_dir: Path, episode_rewards):
#     try:
#         plt.figure(figsize=(8, 4))
#         plt.plot(episode_rewards)
#         plt.xlabel("Episode")
#         plt.ylabel("Episode Reward (sum over robots)")
#         plt.title("Training Rewards - All Batches")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "rewards.png")
#         plt.close()
#     except Exception as e:
#         print("Warning: failed to plot rewards:", e)
# # def plot_task_stats(save_dir: Path, episode_task_stats):
# #     try:
# #         if len(episode_task_stats) == 0:
# #             return

# #         episodes = [x["episode"] for x in episode_task_stats]
# #         obsolete = [x["obsolete"] for x in episode_task_stats]
# #         never_picked = [x["never_picked"] for x in episode_task_stats]
# #         completed = [x["completed"] for x in episode_task_stats]

# #         plt.figure(figsize=(10, 5))
# #         plt.plot(episodes, obsolete, label="Obsolete Tasks")
# #         plt.plot(episodes, never_picked, label="Never Picked Tasks")
# #         plt.plot(episodes, completed, label="Completed Tasks")

# #         plt.xlabel("Episode")
# #         plt.ylabel("Number of Tasks")
# #         plt.title("Task Outcomes Per Episode")
# #         plt.legend()
# #         plt.grid(True)
# #         plt.tight_layout()
# #         plt.savefig(save_dir / "task_stats.png")
# #         plt.close()

# #     except Exception as e:
# #         print("Warning: failed to plot task stats:", e)

# def plot_task_stats(save_dir: Path, episode_task_stats):
#     try:
#         if len(episode_task_stats) == 0:
#             return

#         episodes = [x["episode"] for x in episode_task_stats]
#         obsolete = [x["obsolete"] for x in episode_task_stats]
#         never_picked = [x["never_picked"] for x in episode_task_stats]
#         completed = [x["completed"] for x in episode_task_stats]

#         # --- Obsolete ---
#         plt.figure(figsize=(8, 4))
#         plt.plot(episodes, obsolete)
#         plt.xlabel("Episode")
#         plt.ylabel("Obsolete Tasks")
#         plt.title("Obsolete Tasks Per Episode")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "obsolete_tasks.png")
#         plt.close()

#         # --- Never Picked ---
#         plt.figure(figsize=(8, 4))
#         plt.plot(episodes, never_picked)
#         plt.xlabel("Episode")
#         plt.ylabel("Never Picked Tasks")
#         plt.title("Never Picked Tasks Per Episode")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "never_picked_tasks.png")
#         plt.close()

#         # --- Completed ---
#         plt.figure(figsize=(8, 4))
#         plt.plot(episodes, completed)
#         plt.xlabel("Episode")
#         plt.ylabel("Completed Tasks")
#         plt.title("Completed Tasks Per Episode")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "completed_tasks.png")
#         plt.close()

#     except Exception as e:
#         print("Warning: failed to plot task stats:", e)


# def load_all_batches(data_dir: Path, n_batches: int):
#     """Load all batch files and return them as a list."""
#     batches = []
#     for i in range(n_batches):
#         batch_file = data_dir / f"tasks_batch_{i}.npy"
#         if batch_file.exists():
#             batch = np.load(batch_file, allow_pickle=True)
#             batches.append(batch)
#             print(f"Loaded batch {i}: {len(batch)} tasks, release_time={batch[0][TASK_RELEASE_TIME_INDEX] if len(batch) > 0 else 'N/A'}")
#         else:
#             print(f"Warning: {batch_file} not found")
#     return batches
# def plot_values(save_dir: Path, episode_values):
#     try:
#         plt.figure()
#         plt.plot(episode_values)
#         plt.xlabel("Episode")
#         plt.ylabel("Mean Value Prediction")
#         plt.title("Value Function")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "values.png")
#         plt.close()
#     except Exception as e:
#         print("Warning: failed to plot values:", e)

# # --- Main ---
# def main(args):
#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     print("Device:", device)

#     # Load agents
#     agents_file = Path(args.agents)
#     agents = np.load(agents_file, allow_pickle=True)
#     print(f"Loaded {len(agents)} agents")

#     # Load all task batches
#     data_dir = Path(args.data_dir)
#     all_batches = load_all_batches(data_dir, args.n_batches)
    
#     if len(all_batches) == 0:
#         raise ValueError("No batches loaded! Check data directory and n_batches parameter.")
    
#     print(f"\nTotal batches loaded: {len(all_batches)}")
#     total_tasks = sum(len(batch) for batch in all_batches)
#     print(f"Total tasks across all batches: {total_tasks}")

#     # Create environment with all batches
#     env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=all_batches,  # Pass list of batches
#         radius=args.radius,
#         feature_size=args.feature_size,
#         use_true_id=args.use_true_id,
#         all_batches=True  # Enable all_batches mode
#     )
    
#     print(f"Environment created with {env.n_tasks} tasks total")
#     print(f"Max steps per episode (batch_time): {env.batch_time}")
#     # Save results and models
#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     # Model hyperparams
#     input_dim = args.feature_size
#     hidden_dim = args.hidden_dim
#     critic_aggregation = args.critic_agg

#     # Create actors (one per robot) and optimizers
#     num_robots = env.n_robots
#     actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
#     optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
#                          for rid, actor in actors.items()}

#     # Shared critic
#     critic = CriticGNN(input_dim, hidden_dim, critic_aggregation).to(device)
#     optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

#     # Train
#     print(f"\nStarting training for {args.episodes} episodes...")
#     t0 = time.time()
#     episode_rewards, episode_task_stats, episode_value_means = train(
#     env,
#     num_episodes=args.episodes,
#     actors=actors,
#     critic=critic,
#     optimizers_actors=optimizers_actors,
#     optimizer_critic=optimizer_critic,
#     gamma=args.gamma,
#     max_steps_per_episode=args.max_steps if args.max_steps > 0 else env.batch_time,
#     device=device,
#     verbose=args.verbose,
#     save_dir=save_dir,
#     save_every=10,  # Save every 10 episodes
#     plot_rewards_fn=plot_rewards,
#     plot_task_stats=plot_task_stats,
#     plot_values_fn=plot_values,
#     save_models_fn=save_models
# )
#     t1 = time.time()
#     print(f"\nTraining finished in {t1 - t0:.1f}s")

    
    
#     # Save models
#     save_models(save_dir, actors, critic)
#     print(f"Saved models to {save_dir}")
    
#     # Save rewards
#     with open(save_dir / "episode_rewards.json", "w") as f:
#         json.dump([float(x) for x in episode_rewards], f)
#     with open(save_dir / "episode_task_stats.json", "w") as f:
#         json.dump(episode_task_stats, f, indent=2)

#     with open(save_dir / "episode_values.json", "w") as f:
#         json.dump(episode_value_means, f)

#     plot_values(save_dir, episode_value_means)
#     # Plot task stats
#     plot_task_stats(save_dir, episode_task_stats)
    
#     # Plot rewards
#     plot_rewards(save_dir, episode_rewards)
#     print(f"Saved rewards and plots to {save_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train multi-robot task allocation with all batches loaded at once"
#     )
#     parser.add_argument("--agents", type=str, default="data/agents.npy",
#                         help="Path to agents.npy file")
#     parser.add_argument("--data-dir", type=str, default="data",
#                         help="Directory containing task batch files")
#     parser.add_argument("--n-batches", type=int, default=10,
#                         help="Number of batches to load")
#     parser.add_argument("--episodes", type=int, default=1000,
#                         help="Number of training episodes")
#     parser.add_argument("--max-steps", type=int, default=700,
#                         help="Max steps per episode (0 = use env.batch_time)")
#     parser.add_argument("--feature-size", type=int, default=9)
#     parser.add_argument("--hidden-dim", type=int, default=64)
#     parser.add_argument("--lr-actor", type=float, default=1e-3)
#     parser.add_argument("--lr-critic", type=float, default=1e-3)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--radius", type=int, default=30)
#     parser.add_argument("--critic-agg", type=str, default="per_robot",
#                         choices=["per_robot", "joint_mean", "joint_attn"])
#     parser.add_argument("--use-true-id", action="store_true", dest="use_true_id")
#     parser.add_argument("--save-dir", type=str, default="checkpoints_all_batches")
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
#     parser.add_argument("--verbose", action="store_true", help="Print verbose output")
#     args = parser.parse_args()

#     main(args)

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
import copy
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
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, alpha=0.6, label='Raw')
        
        # Add moving average
        if len(episode_rewards) > 10:
            window = min(50, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                    'r-', linewidth=2, label=f'MA({window})')
        
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (sum over robots)")
        plt.title("Training Rewards - All Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "rewards.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Warning: failed to plot rewards:", e)

def plot_task_stats(save_dir: Path, episode_task_stats):
    try:
        if len(episode_task_stats) == 0:
            return

        episodes = [x["episode"] for x in episode_task_stats]
        obsolete = [x["obsolete"] for x in episode_task_stats]
        never_picked = [x["never_picked"] for x in episode_task_stats]
        completed = [x["completed"] for x in episode_task_stats]

        # --- Combined Plot ---
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, completed, label="Completed", color='green', linewidth=2)
        plt.plot(episodes, obsolete, label="Obsolete", color='red', alpha=0.7)
        plt.plot(episodes, never_picked, label="Never Picked", color='orange', alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Number of Tasks")
        plt.title("Task Outcomes Per Episode")
        plt.legend()
        plt.grid(True)
        
        # --- Completion Rate ---
        plt.subplot(1, 2, 2)
        total_tasks = [c + o for c, o in zip(completed, obsolete)]
        completion_rate = [100 * c / t if t > 0 else 0 for c, t in zip(completed, total_tasks)]
        plt.plot(episodes, completion_rate, color='blue', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Completion Rate (%)")
        plt.title("Task Completion Rate")
        plt.grid(True)
        plt.ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(save_dir / "task_stats.png", dpi=150)
        plt.close()

        # --- Individual plots ---
        # Obsolete
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, obsolete, color='red')
        plt.xlabel("Episode")
        plt.ylabel("Obsolete Tasks")
        plt.title("Obsolete Tasks Per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "obsolete_tasks.png", dpi=150)
        plt.close()

        # Never Picked
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, never_picked, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Never Picked Tasks")
        plt.title("Never Picked Tasks Per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "never_picked_tasks.png", dpi=150)
        plt.close()

        # Completed
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, completed, color='green')
        plt.xlabel("Episode")
        plt.ylabel("Completed Tasks")
        plt.title("Completed Tasks Per Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "completed_tasks.png", dpi=150)
        plt.close()

    except Exception as e:
        print("Warning: failed to plot task stats:", e)


def plot_losses(save_dir: Path, critic_losses, actor_losses):
    """Plot critic loss and actor loss over episodes with moving averages."""
    try:
        # Critic Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(critic_losses, alpha=0.6, label='Raw Loss', color='blue')
        
        # Add moving average
        if len(critic_losses) > 10:
            window = min(50, len(critic_losses) // 10)
            moving_avg = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(critic_losses)), moving_avg, 
                     'r-', linewidth=2, label=f'MA({window})')
        
        plt.xlabel("Episode")
        plt.ylabel("Critic Loss")
        plt.title("Critic Loss Over Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "critic_loss.png", dpi=150)
        plt.close()
        print(f"Saved critic loss plot to {save_dir / 'critic_loss.png'}")
        
        # Actor Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, alpha=0.6, label='Raw Loss', color='red')
        
        # Add moving average
        if len(actor_losses) > 10:
            window = min(50, len(actor_losses) // 10)
            moving_avg = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(actor_losses)), moving_avg, 
                     'orange', linewidth=2, label=f'MA({window})')
        
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss (Total)")
        plt.title("Actor Loss Over Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "actor_loss.png", dpi=150)
        plt.close()
        print(f"Saved actor loss plot to {save_dir / 'actor_loss.png'}")
        
    except Exception as e:
        print(f"Warning: failed to plot losses: {e}")
def plot_values(save_dir: Path, episode_values):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_values, alpha=0.6, label='Raw Value')
        
        # Add moving average
        if len(episode_values) > 10:
            window = min(50, len(episode_values) // 10)
            moving_avg = np.convolve(episode_values, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_values)), moving_avg, 
                    'r-', linewidth=2, label=f'MA({window})')
        
        plt.xlabel("Episode")
        plt.ylabel("Mean Value Prediction")
        plt.title("Value Function Over Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "values.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Warning: failed to plot values:", e)

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
    
    optimizers_actors = {
        rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
        for rid, actor in actors.items()
    }

    # Shared critic
    critic = CriticGNN(input_dim, hidden_dim, critic_aggregation).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    #Create learning rate schedulers
    print(f"\nInitial learning rates:")
    print(f"  Actor: {args.lr_actor:.2e}")
    print(f"  Critic: {args.lr_critic:.2e}")
    print(f"  LR decay: {args.lr_gamma} every {args.lr_step_size} episodes")
    
    scheduler_critic = torch.optim.lr_scheduler.StepLR(
        optimizer_critic,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    schedulers_actors = {
        rid: torch.optim.lr_scheduler.StepLR(
            optimizers_actors[rid],
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        for rid in range(num_robots)
    }

    # Train
    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Assignment interval: {args.assignment_interval}")
    print(f"Gamma (discount): {args.gamma}")
    print(f"Max steps per episode: {args.max_steps if args.max_steps > 0 else env.batch_time}")
    
    t0 = time.time()
    episode_rewards, episode_task_stats, episode_value_means, critic_losses, actor_losses = train(
        env,
        num_episodes=args.episodes,
        actors=actors,
        critic=critic,
        optimizers_actors=optimizers_actors,
        optimizer_critic=optimizer_critic,
        schedulers_actors=schedulers_actors,  
        scheduler_critic=scheduler_critic,     
        gamma=args.gamma,
        max_steps_per_episode=args.max_steps if args.max_steps > 0 else env.batch_time,
        device=device,
        verbose=args.verbose,
        save_dir=save_dir,
        save_every=args.save_every,
        plot_rewards_fn=plot_rewards,
        plot_task_stats=plot_task_stats,
        plot_values_fn=plot_values,
        save_models_fn=save_models,
        assignment_interval=args.assignment_interval,
        n_step= 250
    )
    t1 = time.time()
    print(f"\nTraining finished in {t1 - t0:.1f}s ({(t1-t0)/args.episodes:.2f}s per episode)")

    # Save models
    save_models(save_dir, actors, critic)
    print(f"Saved models to {save_dir}")
    
    # Save rewards and stats
    with open(save_dir / "episode_rewards.json", "w") as f:
        json.dump([float(x) for x in episode_rewards], f)
    
    with open(save_dir / "episode_task_stats.json", "w") as f:
        json.dump(episode_task_stats, f, indent=2)

    with open(save_dir / "episode_values.json", "w") as f:
        json.dump(episode_value_means, f)

    with open(save_dir / "critic_losses.json", "w") as f:
        json.dump(critic_losses, f)
    
    with open(save_dir / "actor_losses.json", "w") as f:
        json.dump(actor_losses, f)

    # Generate plots
    plot_values(save_dir, episode_value_means)
    plot_task_stats(save_dir, episode_task_stats)
    plot_rewards(save_dir, episode_rewards)
    plot_losses(save_dir, critic_losses, actor_losses)
    
    # Print final statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total episodes: {args.episodes}")
    print(f"Final 100 episodes:")
    print(f"  Avg reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Avg completed tasks: {np.mean([s['completed'] for s in episode_task_stats[-100:]]):.1f}")
    print(f"  Avg obsolete tasks: {np.mean([s['obsolete'] for s in episode_task_stats[-100:]]):.1f}")
    print(f"  Avg value prediction: {np.mean(episode_value_means[-100:]):.2f}")
    
    if len(episode_task_stats) > 100:
        completion_rate_final = np.mean([
            100 * s['completed'] / (s['completed'] + s['obsolete']) 
            if (s['completed'] + s['obsolete']) > 0 else 0
            for s in episode_task_stats[-100:]
        ])
        print(f"  Completion rate: {completion_rate_final:.1f}%")
    
    print(f"\nResults saved to: {save_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-robot task allocation with all batches loaded at once"
    )
    parser.add_argument("--agents", type=str, default="data/agents.npy",
                        help="Path to agents.npy file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing task batch files")
    parser.add_argument("--n-batches", type=int, default=3,
                        help="Number of batches to load")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=250,
                        help="Max steps per episode (0 = use env.batch_time)")
    parser.add_argument("--feature-size", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    
    # MODIFIED: Lower default learning rates
    parser.add_argument("--lr-actor", type=float, default=1e-4,
                        help="Actor learning rate (default: 1e-4)")
    parser.add_argument("--lr-critic", type=float, default=5e-4,
                        help="Critic learning rate (default: 5e-4)")
    
    # NEW: Learning rate scheduler parameters
    parser.add_argument("--lr-step-size", type=int, default=200,
                        help="LR scheduler step size (episodes)")
    parser.add_argument("--lr-gamma", type=float, default=0.5,
                        help="LR scheduler decay factor")
    
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--radius", type=int, default=30,
                        help="Communication radius for graph edges")
    parser.add_argument("--critic-agg", type=str, default="per_robot",
                        choices=["per_robot", "joint_mean", "joint_attn"])
    parser.add_argument("--use-true-id", action="store_true", dest="use_true_id",
                        help="Use true task IDs in graph")
    
    # NEW: Assignment interval parameter
    parser.add_argument("--assignment-interval", type=int, default=5,
                        help="Steps between assignment decisions")
    
    parser.add_argument("--save-dir", type=str, default="checkpoints_all_batches",
                        help="Directory to save models and results")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save models every N episodes")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", 
                        help="Disable CUDA")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose output")
    
    args = parser.parse_args()

    main(args)