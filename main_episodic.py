# #!/usr/bin/env python3
# """
# main.py - per-episode batch training driver (round-robin or random batch selection)

# Usage examples:
#   python3 main.py --agents data/agents.npy --tasks-pattern "data/task_batch_*.npy" --total-episodes 2000
# """
# import argparse
# from pathlib import Path
# import numpy as np
# import torch
# import random
# import json
# import time
# import re
# from natsort import natsorted  # optional but helpful; fallback to sorted if not installed

# from src.environment.environment import MultiTaskAllocationEnv
# from src.models.actor_critic import ActorGNN, CriticGNN
# from src.training.train_actor_critic import train

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def load_task_batches_from_pattern(pattern):
#     """Load all .npy files matching the glob pattern into a list of batches.
#        Files are naturally sorted by name (natsorted if available).
#     """
#     files = list(Path(".").glob(pattern))
#     if not files:
#         raise FileNotFoundError(f"No files found for pattern: {pattern}")
#     try:
#         # try natural sort if available
#         files = natsorted(files)
#     except Exception:
#         files = sorted(files)
#     batches = [np.load(str(p), allow_pickle=True) for p in files]
#     print(f"Loaded {len(batches)} batches from {pattern}")
#     print("First 3 batch files:", [p.name for p in files[:3]])
#     return batches, [str(p) for p in files]

# def main(args):
#     set_seed(args.seed)

#     # Load agents and all task-batch files
#     agents = np.load(args.agents, allow_pickle=True)
#     tasks_batches, batch_files = load_task_batches_from_pattern(args.tasks_pattern)
#     tasks = np.load('data/tasks_batch_1.npy', allow_pickle=True)

#     # Create environment with empty initial batch; set_batch will populate per episode
#     env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=tasks,   # will be replaced each episode via set_batch
#         radius=args.radius,
#         feature_size=args.feature_size,
#         use_true_id=args.use_true_id,
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     print("Device:", device)

#     # Model setup (create once)
#     input_dim = args.feature_size
#     hidden_dim = args.hidden_dim
#     critic_agg = args.critic_agg

#     num_robots = env.n_robots
#     actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
#     optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor) for rid, actor in actors.items()}

#     critic = CriticGNN(input_dim, hidden_dim, critic_agg).to(device)
#     optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

#     num_batches = len(tasks_batches)
#     episode_rewards_all = []
#     t0 = time.time()

#     for ep in range(args.total_episodes):
#         # choose batch index
#         if args.batch_selection == "round_robin":
#             bidx = ep % num_batches
#         else:  # random
#             bidx = random.randrange(num_batches)

#         batch = tasks_batches[bidx]
#         env.set_batch(batch)

#         # Run exactly one episode (train() will reset observation at start of episode inside)
#         rewards = train(
#             env,
#             num_episodes=10,
#             actors=actors,
#             critic=critic,
#             optimizers_actors=optimizers_actors,
#             optimizer_critic=optimizer_critic,
#             gamma=args.gamma,
#             max_steps_per_episode=args.max_steps,
#             device=device
#         )

#         ep_reward = float(rewards[0]) if rewards else 0.0
#         episode_rewards_all.append(ep_reward)

#         print(f"Episode {ep+1}/{args.total_episodes} batch_file={Path(batch_files[bidx]).name} reward={ep_reward:.2f}")

#         # periodic checkpoint
#         if (ep + 1) % args.checkpoint_every == 0:
#             sd = Path(args.save_dir)
#             sd.mkdir(parents=True, exist_ok=True)
#             for rid, actor in actors.items():
#                 torch.save(actor.state_dict(), sd / f"actor_r{rid}_ep{ep+1}.pt")
#             torch.save(critic.state_dict(), sd / f"critic_ep{ep+1}.pt")
#             with open(sd / "episode_rewards.json", "w") as f:
#                 json.dump([float(x) for x in episode_rewards_all], f)

#     print("Training finished. Time elapsed: {:.1f}s".format(time.time() - t0))
#     # final save
#     sd = Path(args.save_dir)
#     sd.mkdir(parents=True, exist_ok=True)
#     for rid, actor in actors.items():
#         torch.save(actor.state_dict(), sd / f"actor_r{rid}_final.pt")
#     torch.save(critic.state_dict(), sd / "critic_final.pt")
#     with open(sd / "episode_rewards.json", "w") as f:
#         json.dump([float(x) for x in episode_rewards_all], f)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--agents", type=str, default="data/agents.npy")
#     parser.add_argument("--tasks-pattern", type=str, default="data/tasks_batch_*.npy",
#                         help="glob pattern for task batch files (e.g. data/tasks_batch_*.npy)")
#     parser.add_argument("--total-episodes", type=int, default=2)
#     parser.add_argument("--max-steps", type=int, default=45)
#     parser.add_argument("--feature-size", type=int, default=9)
#     parser.add_argument("--hidden-dim", type=int, default=64)
#     parser.add_argument("--lr-actor", type=float, default=1e-3)
#     parser.add_argument("--lr-critic", type=float, default=1e-3)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--radius", type=int, default=3000)
#     parser.add_argument("--critic-agg", type=str, default="per_robot", choices=["per_robot","joint_mean","joint_attn"])
#     parser.add_argument("--use-true-id", action="store_true")
#     parser.add_argument("--save-dir", type=str, default="checkpoints")
#     parser.add_argument("--checkpoint-every", type=int, default=200)
#     parser.add_argument("--batch-selection", type=str, default="round_robin", choices=["round_robin","random"])
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--no-cuda", action="store_true")
#     args = parser.parse_args()
#     main(args)
#!/usr/bin/env python3


#!/usr/bin/env python3
#!/usr/bin/env python3



# import torch
# import numpy as np
# import gym
# from pathlib import Path
# import matplotlib.pyplot as plt
# import time
# import argparse
# import json
# import random

# from src.environment.environment import MultiTaskAllocationEnv
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
#         plt.title("Training Rewards")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(save_dir / "rewards.png")
#         plt.close()
#     except Exception as e:
#         print("Warning: failed to plot rewards:", e)

# def find_task_batches(path: Path):
#     if path.is_dir():
#         files = sorted(path.glob("tasks_batch_*.npy"))
#         return files
#     else:
#         return [path]

# def load_task_batch(file: Path):
#     arr = np.load(file, allow_pickle=True)
#     return arr

# def main(args):
#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     print("Device:", device)

#     # Load agents
#     agents_file = Path(args.agents)
#     agents = np.load(agents_file, allow_pickle=True)

#     # Prepare environment with an initial empty batch (we will replace batches later)
#     # If tasks arg is a file, we still initialize env with that batch then later iterate.
#     tasks_path = Path(args.tasks)
#     task_files = find_task_batches(tasks_path)

#     # Initialize environment using the first batch (or empty if none)
#     initial_tasks = load_task_batch(task_files[0]) if task_files else np.zeros((0,))
#     env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=initial_tasks,
#         radius=args.radius,
#         feature_size=args.feature_size,
#         use_true_id=args.use_true_id,
#     )

#     # Model hyperparams
#     input_dim = args.feature_size
#     hidden_dim = args.hidden_dim

#     # Create actors (one per robot) and optimizers
#     num_robots = env.n_robots
#     actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
#     optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
#                          for rid, actor in actors.items()}

#     # Shared critic
#     critic = CriticGNN(input_dim, hidden_dim, args.critic_agg).to(device)
#     optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

#     all_rewards = []
#     batch_rewards = {}

#     # Iterate over each task batch file and train for episodes_per_batch
#     for b_idx, batch_file in enumerate(task_files):
#         print(f"\n=== Starting training on batch {b_idx}: {batch_file} ===")
#         batch = load_task_batch(batch_file)

#         # Replace env tasks cleanly
#         env.set_batch(batch)

#         # Train for episodes_per_batch (actors/critic are preserved between batches)
#         rewards = train(
#             env,
#             num_episodes=args.episodes_per_batch,
#             actors=actors,
#             critic=critic,
#             optimizers_actors=optimizers_actors,
#             optimizer_critic=optimizer_critic,
#             gamma=args.gamma,
#             max_steps_per_episode=args.max_steps,
#             device=device,
#             verbose=True
#         )

#         # record
#         batch_rewards[str(batch_file.name)] = rewards
#         all_rewards.extend(rewards)

#         # Optionally save intermediate models per batch
#         if args.save_after_each_batch:
#             save_dir = Path(args.save_dir) / f"after_batch_{b_idx}"
#             save_models(save_dir, actors, critic)
#             with open(save_dir / "episode_rewards.json", "w") as f:
#                 json.dump([float(x) for x in rewards], f)
#             print(f"Saved models and rewards after batch {b_idx} to {save_dir}")

#     # Final save
#     save_dir = Path(args.save_dir)
#     save_models(save_dir, actors, critic)
#     with open(save_dir / "episode_rewards_all.json", "w") as f:
#         json.dump(all_rewards, f)
#     with open(save_dir / "episode_rewards_per_batch.json", "w") as f:
#         json.dump(batch_rewards, f)

#     plot_rewards(save_dir, all_rewards)
#     print(f"\nTraining complete. Saved models and rewards to {save_dir.resolve()}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--agents", type=str, default="data/agents.npy")
#     parser.add_argument("--tasks", type=str, default="data")  # can be a single file or directory
#     parser.add_argument("--episodes-per-batch", type=int, default=1000, dest="episodes_per_batch")
#     parser.add_argument("--max-steps", type=int, default=15)
#     parser.add_argument("--feature-size", type=int, default=9)
#     parser.add_argument("--hidden-dim", type=int, default=64)
#     parser.add_argument("--lr-actor", type=float, default=1e-3)
#     parser.add_argument("--lr-critic", type=float, default=1e-3)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--radius", type=int, default=30)
#     parser.add_argument("--critic-agg", type=str, default="per_robot", choices=["per_robot", "joint_mean", "joint_attn"])
#     parser.add_argument("--use-true-id", action="store_true", dest="use_true_id")
#     parser.add_argument("--save-dir", type=str, default="checkpoints")
#     parser.add_argument("--save-after-each-batch", action="store_true", dest="save_after_each_batch")
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
#     args = parser.parse_args()

#     main(args)




import torch
import numpy as np
import gym
from pathlib import Path
import matplotlib.pyplot as plt
import time
import argparse
import json
import random
import os

from src.environment.environment import MultiTaskAllocationEnv
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

def moving_average(x, w=20):
    x = np.array(x, dtype=float)
    if x.size == 0 or w <= 1:
        return x
    if x.size < w:
        # return simple smoothing with available window
        return np.convolve(x, np.ones(x.size)/x.size, mode='valid')
    return np.convolve(x, np.ones(w)/w, mode='valid')

def plot_overall_rewards(all_rewards, save_dir: Path, ma_window=50):
    save_dir.mkdir(parents=True, exist_ok=True)
    all_rewards = np.array(all_rewards, dtype=float)
    plt.figure(figsize=(10,4))
    plt.plot(all_rewards, color='tab:blue', alpha=0.4, label='Episode reward')
    ma = moving_average(all_rewards, w=ma_window)
    if ma.size > 0:
        # align MA to the right side
        x = np.arange(len(all_rewards) - len(ma), len(all_rewards))
        plt.plot(x, ma, color='tab:red', label=f'MA({ma_window})')
    plt.xlabel("Episode (global)")
    plt.ylabel("Episode Reward (sum over robots)")
    plt.title("Learning Curve - All Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curve_all.png", dpi=200)
    plt.close()

def plot_per_batch_curves(batch_rewards, save_dir: Path, ma_window=10):
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,6))
    for name, rewards in batch_rewards.items():
        arr = np.array(rewards, dtype=float)
        if arr.size == 0:
            continue
        ma = moving_average(arr, w=ma_window)
        # plot raw (transparent) and MA
        plt.plot(arr, alpha=0.15)
        if ma.size > 0:
            x = np.arange(len(arr) - len(ma), len(arr))
            plt.plot(x, ma, label=name)
        else:
            plt.plot(arr, label=name)
    plt.xlabel("Episode (within batch)")
    plt.ylabel("Episode Reward")
    plt.title("Per-batch Reward Curves (MA)")
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "per_batch_curves.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_boxplot_per_batch(batch_rewards, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    names = []
    arrs = []
    for name, rewards in batch_rewards.items():
        names.append(name)
        arrs.append(np.array(rewards, dtype=float))
    plt.figure(figsize=(10,6))
    # handle empty lists safely
    nonempty = [a for a in arrs if a.size > 0]
    if len(nonempty) == 0:
        plt.text(0.5, 0.5, "No data", ha='center')
    else:
        plt.boxplot(arrs, labels=names, showmeans=True)
        plt.xticks(rotation=45, ha='right')
    plt.ylabel('Episode reward')
    plt.title('Episode Reward Distribution per Batch')
    plt.tight_layout()
    plt.savefig(save_dir / 'boxplot_per_batch.png', dpi=200, bbox_inches='tight')
    plt.close()

def find_task_batches(path: Path):
    if path.is_dir():
        files = sorted(path.glob("tasks_batch_*.npy"))
        return files
    else:
        return [path]

def load_task_batch(file: Path):
    arr = np.load(file, allow_pickle=True)
    return arr

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    # Load agents
    agents_file = Path(args.agents)
    agents = np.load(agents_file, allow_pickle=True)

    # Prepare environment with an initial empty batch (we will replace batches later)
    tasks_path = Path(args.tasks)
    task_files = find_task_batches(tasks_path)

    # Initialize environment using the first batch (or empty if none)
    initial_tasks = load_task_batch(task_files[0]) if task_files else np.zeros((0,))
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=initial_tasks,
        radius=args.radius,
        feature_size=args.feature_size,
        use_true_id=args.use_true_id,
    )

    # Model hyperparams
    input_dim = args.feature_size
    hidden_dim = args.hidden_dim

    # Create actors (one per robot) and optimizers
    num_robots = env.n_robots
    actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
    optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
                         for rid, actor in actors.items()}

    # Shared critic
    critic = CriticGNN(input_dim, hidden_dim, args.critic_agg).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    all_rewards = []
    batch_rewards = {}

    # Iterate over each task batch file and train for episodes_per_batch
    for b_idx, batch_file in enumerate(task_files):
        print(f"\n=== Starting training on batch {b_idx}: {batch_file} ===")
        batch = load_task_batch(batch_file)

        # Replace env tasks cleanly
        env.set_batch(batch)

        # Train for episodes_per_batch (actors/critic are preserved between batches)
        rewards = train(
            env,
            num_episodes=args.episodes_per_batch,
            actors=actors,
            critic=critic,
            optimizers_actors=optimizers_actors,
            optimizer_critic=optimizer_critic,
            gamma=args.gamma,
            max_steps_per_episode=args.max_steps,
            device=device,
            verbose=True
        )

        # If train returns a tuple (episode_rewards, episode_metrics) handle both cases
        if isinstance(rewards, tuple) or isinstance(rewards, list) and len(rewards) == 2 and not isinstance(rewards[0], (float, int)):
            # previous versions returned only rewards; some modified train return (rewards, metrics)
            episode_rewards = rewards[0]
        else:
            episode_rewards = rewards

        # record
        batch_rewards[str(batch_file.name)] = episode_rewards
        all_rewards.extend(episode_rewards)

        # Optionally save intermediate models per batch
        if args.save_after_each_batch:
            save_dir = Path(args.save_dir) / f"after_batch_{b_idx}"
            save_models(save_dir, actors, critic)
            with open(save_dir / "episode_rewards.json", "w") as f:
                json.dump([float(x) for x in episode_rewards], f)
            print(f"Saved models and rewards after batch {b_idx} to {save_dir}")

    # Final save
    save_dir = Path(args.save_dir)
    save_models(save_dir, actors, critic)
    with open(save_dir / "episode_rewards_all.json", "w") as f:
        json.dump(all_rewards, f)
    with open(save_dir / "episode_rewards_per_batch.json", "w") as f:
        json.dump(batch_rewards, f)

    # Create plots
    plots_dir = save_dir / "plots"
    plot_overall_rewards(all_rewards, plots_dir, ma_window=100)
    plot_per_batch_curves(batch_rewards, plots_dir, ma_window=10)
    plot_boxplot_per_batch(batch_rewards, plots_dir)

    print(f"\nTraining complete. Saved models, rewards and plots to {save_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=str, default="data/agents.npy")
    parser.add_argument("--tasks", type=str, default="data")  # can be a single file or directory
    parser.add_argument("--episodes-per-batch", type=int, default=1000, dest="episodes_per_batch")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--feature-size", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument("--critic-agg", type=str, default="per_robot", choices=["per_robot", "joint_mean", "joint_attn"])
    parser.add_argument("--use-true-id", action="store_true", dest="use_true_id")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-after-each-batch", action="store_true", dest="save_after_each_batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    main(args)