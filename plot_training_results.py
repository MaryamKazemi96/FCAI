# #!/usr/bin/env python3
# """
# Plot PPO training results from tensorboard logs.

# Usage:
#     python plot_training_results.py --log-dir tensorboard_logs
# """

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from pathlib import Path
# from tensorboard.backend.event_processing import event_accumulator
# import glob

# import json
# from pathlib import Path


# def load_baseline_stats(checkpoint_dir: Path):
#     """
#     Load baseline mean/std from eval_baseline.py outputs.
#     Expects: baseline_results_all.json in the same checkpoint dir.
#     Returns: dict like { 'random': {'mean':..., 'std':...}, ... } or None
#     """
#     p = checkpoint_dir / "baseline_results_all.json"
#     if not p.exists():
#         return None

#     with p.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     out = {}
#     for pol in ["random", "greedy", "unique"]:
#         if pol in data and "stats" in data[pol]:
#             out[pol] = {
#                 "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
#                 "std": float(data[pol]["stats"].get("reward_std", 0.0)),
#             }
#     return out


# def add_baseline_lines(ax, baselines, show_std: bool = True):
#     """
#     Draw baseline horizontal lines on an existing matplotlib axis.
#     """
#     if not baselines:
#         return

#     colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
#     labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

#     for pol, stats in baselines.items():
#         mean = stats["mean"]
#         std = stats.get("std", 0.0)
#         c = colors.get(pol, "#999999")
#         label = labels.get(pol, pol)

#         ax.axhline(mean, color=c, linestyle="--", linewidth=2.2, alpha=0.9, label=f"{label} baseline")
#         if show_std and std and std > 0:
#             ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
#             ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
# def moving_average(data, window=50):
#     """Compute moving average."""
#     if len(data) < window:
#         return data
    
#     cumsum = np.cumsum(np.insert(data, 0, 0))
#     ma = (cumsum[window:] - cumsum[:-window]) / window
    
#     # Pad to original length
#     pad_len = len(data) - len(ma)
#     return np.concatenate([data[:pad_len], ma])


# def load_tensorboard_data(log_dir):
#     """Load data from tensorboard logs."""
#     # Find the latest run directory
#     run_dirs = list(Path(log_dir).glob("PPO_*"))
    
#     if not run_dirs:
#         print(f"No tensorboard logs found in {log_dir}")
#         return {}
    
#     latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
#     print(f"Loading data from: {latest_run}")
    
#     ea = event_accumulator.EventAccumulator(str(latest_run))
#     ea.Reload()
    
#     # Get available tags
#     tags = ea.Tags()
    
#     data = {}
#     for tag in tags['scalars']:
#         events = ea.Scalars(tag)
#         data[tag] = {
#             'steps': [e.step for e in events],
#             'values': [e.value for e in events]
#         }
    
#     return data


# def plot_training_results(log_dir, checkpoint_dir):
#     """Generate comprehensive training plots."""
    
#     data = load_tensorboard_data(log_dir)
    
#     if not data:
#         print("No data to plot!")
#         return
    
#     print(f"Available metrics: {list(data.keys())}")
    
#     # Create figure
#     fig = plt.figure(figsize=(18, 12), facecolor='white')
#     gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
#     # ========== 1. Episode Rewards ==========
#     if 'rollout/ep_rew_mean' in data:
#         ax = fig.add_subplot(gs[0, :])
#         ax.set_facecolor('#fafafa')
        
#         steps = data['rollout/ep_rew_mean']['steps']
#         rewards = data['rollout/ep_rew_mean']['values']
        
#         ax.plot(steps, rewards, 'o', markersize=4, alpha=0.3, color='#3498db', label='Raw')
        
#         if len(rewards) > 50:
#             ax.plot(steps, moving_average(rewards, 50), lw=2.5, color='#2980b9', label='MA(50)')
        
#         ax.axhline(np.mean(rewards), color='#e74c3c', lw=2, ls='--', alpha=0.6, 
#                    label=f'Mean: {np.mean(rewards):.1f}')
        
#         # Trend line
#         z = np.polyfit(steps, rewards, 1)
#         ax.plot(steps, np.polyval(z, steps), lw=1.8, color='#27ae60', alpha=0.5,
#                 ls=':', label=f'Trend: {z[0]:+.4f}/step')
        
#         ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
#         ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
#         ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
#         ax.legend(loc='upper left', fontsize=9)
#         ax.grid(alpha=0.25)
        
#         # Stats box
#         if len(rewards) >= 10:
#             last_10 = rewards[-10:]
#             txt = (f'Mean:     {np.mean(rewards):.2f}\n'
#                    f'Last 10:  {np.mean(last_10):.2f}\n'
#                    f'Best:     {np.max(rewards):.2f}')
#             ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=9,
#                     va='bottom', ha='right', family='monospace',
#                     bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))
    
#     # ========== 2. Task Completion Rate ==========
#     if 'task/completion_rate' in data:
#         ax = fig.add_subplot(gs[1, 0])
#         ax.set_facecolor('#fafafa')
        
#         steps = data['task/completion_rate']['steps']
#         comp_rate = data['task/completion_rate']['values']
        
#         ax.plot(steps, comp_rate, 'o', markersize=4, alpha=0.3, color='#27ae60')
        
#         if len(comp_rate) > 20:
#             ax.plot(steps, moving_average(comp_rate, 20), lw=2.5, color='#27ae60')
        
#         ax.axhline(np.mean(comp_rate), color='#16a085', lw=2, ls='--', alpha=0.6)
        
#         ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Completion Rate (%)', fontsize=10, fontweight='bold')
#         ax.set_title(f'Task Completion (mean: {np.mean(comp_rate):.1f}%)', 
#                      fontsize=11, fontweight='bold')
#         ax.grid(alpha=0.25)
#         ax.set_ylim(0, 105)
    
#     # ========== 3. Completed vs Obsolete Tasks ==========
#     if 'task/completed' in data and 'task/obsolete' in data:
#         ax = fig.add_subplot(gs[1, 1])
#         ax.set_facecolor('#fafafa')
        
#         steps_comp = data['task/completed']['steps']
#         completed = data['task/completed']['values']
#         obsolete = data['task/obsolete']['values']
        
#         if len(completed) > 20:
#             ax.plot(steps_comp, moving_average(completed, 20), lw=2.5, 
#                     color='#27ae60', label='Completed')
#             ax.plot(steps_comp, moving_average(obsolete, 20), lw=2.5, 
#                     color='#e74c3c', label='Obsolete')
#         else:
#             ax.plot(steps_comp, completed, lw=2.5, color='#27ae60', label='Completed')
#             ax.plot(steps_comp, obsolete, lw=2.5, color='#e74c3c', label='Obsolete')
        
#         ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Number of Tasks', fontsize=10, fontweight='bold')
#         ax.set_title('Task Outcomes', fontsize=11, fontweight='bold')
#         ax.legend(fontsize=9)
#         ax.grid(alpha=0.25)
    
#     # ========== 4. Value Loss ==========
#     if 'train/value_loss' in data:
#         ax = fig.add_subplot(gs[2, 0])
#         ax.set_facecolor('#fafafa')
        
#         steps = data['train/value_loss']['steps']
#         loss = data['train/value_loss']['values']
        
#         ax.plot(steps, loss, 'o', markersize=3, alpha=0.3, color='#3498db')
        
#         if len(loss) > 20:
#             ax.plot(steps, moving_average(loss, 20), lw=2.5, color='#2980b9')
        
#         ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Value Loss', fontsize=10, fontweight='bold')
#         ax.set_title('Critic Loss', fontsize=11, fontweight='bold')
#         ax.grid(alpha=0.25)
#         ax.set_yscale('log')
    
#     # ========== 5. Policy Entropy ==========
#     if 'train/entropy_loss' in data:
#         ax = fig.add_subplot(gs[2, 1])
#         ax.set_facecolor('#fafafa')
        
#         steps = data['train/entropy_loss']['steps']
#         entropy = data['train/entropy_loss']['values']
        
#         ax.plot(steps, entropy, 'o', markersize=3, alpha=0.3, color='#9b59b6')
        
#         if len(entropy) > 20:
#             ax.plot(steps, moving_average(entropy, 20), lw=2.5, color='#8e44ad')
        
#         ax.axhline(-1.0, color='orange', lw=1.5, ls='--', alpha=0.5, label='Warning')
#         ax.axhline(-0.5, color='red', lw=1.5, ls='--', alpha=0.5, label='Collapse')
        
#         ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
#         ax.set_ylabel('Entropy Loss', fontsize=10, fontweight='bold')
#         ax.set_title('Policy Entropy (Exploration)', fontsize=11, fontweight='bold')
#         ax.legend(fontsize=8)
#         ax.grid(alpha=0.25)
    
#     plt.tight_layout()
#     output_file = checkpoint_dir / "training_results.png"
#     plt.savefig(output_file, dpi=150, bbox_inches='tight')
#     print(f"\n✓ Saved plot to {output_file}")
#     plt.close()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--log-dir', type=str, default='tensorboard_logs',
#                        help='Tensorboard log directory')
#     parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_ppo',
#                        help='Checkpoint directory for saving plots')
#     args = parser.parse_args()
    
#     log_dir = Path(args.log_dir)
#     checkpoint_dir = Path(args.checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
#     if not log_dir.exists():
#         print(f"Error: Log directory {log_dir} does not exist!")
#         return
    
#     print(f"Loading training results from {log_dir}...")
#     plot_training_results(log_dir, checkpoint_dir)
#     print("\n✓ Done!")


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
Plot PPO training results from tensorboard logs and overlay baseline (random/greedy/unique) lines.

Expected baseline file (created by eval_baseline.py):
  <checkpoint-dir>/baseline_results_all.json

Usage:
    python plot_training_results.py --log-dir tensorboard_logs --checkpoint-dir checkpoints_ppo
"""

# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from tensorboard.backend.event_processing import event_accumulator


# # ---------------- Baseline overlay helpers ----------------

# def load_baseline_stats(checkpoint_dir: Path):
#     """
#     Load baseline mean/std from eval_baseline.py outputs.
#     Expects: baseline_results_all.json in checkpoint_dir.

#     Returns:
#         dict like:
#           {
#             "random": {"mean": ..., "std": ...},
#             "greedy": {"mean": ..., "std": ...},
#             "unique": {"mean": ..., "std": ...},
#           }
#         or None if not found.
#     """
#     p = checkpoint_dir / "baseline_results_all.json"
#     if not p.exists():
#         return None

#     with p.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     out = {}
#     for pol in ["random", "greedy", "unique"]:
#         if pol in data and "stats" in data[pol]:
#             out[pol] = {
#                 "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
#                 "std": float(data[pol]["stats"].get("reward_std", 0.0)),
#             }
#     return out


# def add_baseline_lines(ax, baselines, show_std: bool = True):
#     """
#     Draw baseline horizontal lines on an existing matplotlib axis.
#     """
#     if not baselines:
#         return

#     colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
#     labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

#     for pol, stats in baselines.items():
#         mean = float(stats["mean"])
#         std = float(stats.get("std", 0.0))
#         c = colors.get(pol, "#999999")
#         label = labels.get(pol, pol)

#         ax.axhline(
#             mean,
#             color=c,
#             linestyle="--",
#             linewidth=2.2,
#             alpha=0.9,
#             label=f"{label} baseline ({mean:.2f})",
#         )
#         if show_std and std > 0:
#             ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
#             ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)


# # ---------------- Tensorboard utilities ----------------

# def moving_average(data, window=50):
#     """Compute moving average (same length output)."""
#     data = np.asarray(data, dtype=float)
#     if data.size == 0:
#         return data
#     if data.size < window or window <= 1:
#         return data

#     cumsum = np.cumsum(np.insert(data, 0, 0.0))
#     ma = (cumsum[window:] - cumsum[:-window]) / window
#     pad_len = data.size - ma.size
#     return np.concatenate([data[:pad_len], ma])


# def load_tensorboard_data(log_dir: Path):
#     """
#     Load scalar series from the latest PPO_* run directory.
#     Returns dict[tag] = {"steps":[...], "values":[...]}.
#     """
#     run_dirs = list(Path(log_dir).glob("PPO_*"))
#     if not run_dirs:
#         print(f"No tensorboard logs found in {log_dir}")
#         return {}

#     latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
#     print(f"Loading data from: {latest_run}")

#     ea = event_accumulator.EventAccumulator(str(latest_run))
#     ea.Reload()

#     tags = ea.Tags()
#     data = {}
#     for tag in tags.get("scalars", []):
#         events = ea.Scalars(tag)
#         data[tag] = {
#             "steps": [e.step for e in events],
#             "values": [e.value for e in events],
#         }
#     return data


# # ---------------- Plotting ----------------

# def plot_training_results(log_dir: Path, checkpoint_dir: Path, baseline_std: bool = True):
#     data = load_tensorboard_data(log_dir)
#     if not data:
#         print("No data to plot!")
#         return

#     print(f"Available metrics: {list(data.keys())}")

#     baselines = load_baseline_stats(checkpoint_dir)
#     if baselines is None:
#         print(f"[INFO] No baseline_results_all.json found in {checkpoint_dir} (baseline lines will be skipped).")
#         print("[INFO] Run: python eval_baseline.py --output-dir <checkpoint-dir> ...")

#     fig = plt.figure(figsize=(18, 12), facecolor="white")
#     gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

#     # ========== 1. Episode Rewards ==========
#     if "rollout/ep_rew_mean" in data:
#         ax = fig.add_subplot(gs[0, :])
#         ax.set_facecolor("#fafafa")

#         steps = np.asarray(data["rollout/ep_rew_mean"]["steps"], dtype=float)
#         rewards = np.asarray(data["rollout/ep_rew_mean"]["values"], dtype=float)

#         ax.plot(steps, rewards, "o", markersize=4, alpha=0.3, color="#3498db", label="PPO raw")

#         if rewards.size > 50:
#             ax.plot(steps, moving_average(rewards, 50), lw=2.5, color="#2980b9", label="PPO MA(50)")

#         ax.axhline(
#             float(np.mean(rewards)),
#             color="#e74c3c",
#             lw=2,
#             ls="--",
#             alpha=0.6,
#             label=f"PPO mean: {float(np.mean(rewards)):.1f}",
#         )

#         # Trend line
#         if steps.size >= 2:
#             z = np.polyfit(steps, rewards, 1)
#             ax.plot(
#                 steps,
#                 np.polyval(z, steps),
#                 lw=1.8,
#                 color="#27ae60",
#                 alpha=0.5,
#                 ls=":",
#                 label=f"Trend: {z[0]:+.4f}/step",
#             )

#         # ---- Baseline overlay ----
#         add_baseline_lines(ax, baselines, show_std=baseline_std)

#         ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
#         ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
#         ax.set_title("Training Rewards (PPO vs baselines)", fontsize=14, fontweight="bold")
#         ax.legend(loc="upper left", fontsize=9)
#         ax.grid(alpha=0.25)

#         # Stats box
#         if rewards.size >= 10:
#             last_10 = rewards[-10:]
#             txt = (
#                 f"PPO mean:   {float(np.mean(rewards)):.2f}\n"
#                 f"PPO last10: {float(np.mean(last_10)):.2f}\n"
#                 f"PPO best:   {float(np.max(rewards)):.2f}"
#             )
#             ax.text(
#                 0.98,
#                 0.04,
#                 txt,
#                 transform=ax.transAxes,
#                 fontsize=9,
#                 va="bottom",
#                 ha="right",
#                 family="monospace",
#                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9),
#             )
#     else:
#         print("[WARN] Missing 'rollout/ep_rew_mean' in tensorboard logs; reward plot will be skipped.")

#     # ========== 2. Task Completion Rate ==========
#     if "task/completion_rate" in data:
#         ax = fig.add_subplot(gs[1, 0])
#         ax.set_facecolor("#fafafa")

#         steps = np.asarray(data["task/completion_rate"]["steps"], dtype=float)
#         comp_rate = np.asarray(data["task/completion_rate"]["values"], dtype=float)

#         ax.plot(steps, comp_rate, "o", markersize=4, alpha=0.3, color="#27ae60")

#         if comp_rate.size > 20:
#             ax.plot(steps, moving_average(comp_rate, 20), lw=2.5, color="#27ae60")

#         ax.axhline(float(np.mean(comp_rate)), color="#16a085", lw=2, ls="--", alpha=0.6)

#         ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
#         ax.set_ylabel("Completion Rate (%)", fontsize=10, fontweight="bold")
#         ax.set_title(f"Task Completion (mean: {float(np.mean(comp_rate)):.1f}%)", fontsize=11, fontweight="bold")
#         ax.grid(alpha=0.25)
#         ax.set_ylim(0, 105)

#     # ========== 3. Completed vs Obsolete Tasks ==========
#     if "task/completed" in data and "task/obsolete" in data:
#         ax = fig.add_subplot(gs[1, 1])
#         ax.set_facecolor("#fafafa")

#         steps_comp = np.asarray(data["task/completed"]["steps"], dtype=float)
#         completed = np.asarray(data["task/completed"]["values"], dtype=float)
#         obsolete = np.asarray(data["task/obsolete"]["values"], dtype=float)

#         if completed.size > 20:
#             ax.plot(steps_comp, moving_average(completed, 20), lw=2.5, color="#27ae60", label="Completed")
#             ax.plot(steps_comp, moving_average(obsolete, 20), lw=2.5, color="#e74c3c", label="Obsolete")
#         else:
#             ax.plot(steps_comp, completed, lw=2.5, color="#27ae60", label="Completed")
#             ax.plot(steps_comp, obsolete, lw=2.5, color="#e74c3c", label="Obsolete")

#         ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
#         ax.set_ylabel("Number of Tasks", fontsize=10, fontweight="bold")
#         ax.set_title("Task Outcomes", fontsize=11, fontweight="bold")
#         ax.legend(fontsize=9)
#         ax.grid(alpha=0.25)

#     # ========== 4. Value Loss ==========
#     if "train/value_loss" in data:
#         ax = fig.add_subplot(gs[2, 0])
#         ax.set_facecolor("#fafafa")

#         steps = np.asarray(data["train/value_loss"]["steps"], dtype=float)
#         loss = np.asarray(data["train/value_loss"]["values"], dtype=float)

#         ax.plot(steps, loss, "o", markersize=3, alpha=0.3, color="#3498db")

#         if loss.size > 20:
#             ax.plot(steps, moving_average(loss, 20), lw=2.5, color="#2980b9")

#         ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
#         ax.set_ylabel("Value Loss", fontsize=10, fontweight="bold")
#         ax.set_title("Critic Loss", fontsize=11, fontweight="bold")
#         ax.grid(alpha=0.25)
#         ax.set_yscale("log")

#     # ========== 5. Policy Entropy ==========
#     if "train/entropy_loss" in data:
#         ax = fig.add_subplot(gs[2, 1])
#         ax.set_facecolor("#fafafa")

#         steps = np.asarray(data["train/entropy_loss"]["steps"], dtype=float)
#         entropy = np.asarray(data["train/entropy_loss"]["values"], dtype=float)

#         ax.plot(steps, entropy, "o", markersize=3, alpha=0.3, color="#9b59b6")

#         if entropy.size > 20:
#             ax.plot(steps, moving_average(entropy, 20), lw=2.5, color="#8e44ad")

#         ax.axhline(-1.0, color="orange", lw=1.5, ls="--", alpha=0.5, label="Warning")
#         ax.axhline(-0.5, color="red", lw=1.5, ls="--", alpha=0.5, label="Collapse")

#         ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
#         ax.set_ylabel("Entropy Loss", fontsize=10, fontweight="bold")
#         ax.set_title("Policy Entropy (Exploration)", fontsize=11, fontweight="bold")
#         ax.legend(fontsize=8)
#         ax.grid(alpha=0.25)

#     plt.tight_layout()
#     output_file = checkpoint_dir / "training_results.png"
#     plt.savefig(output_file, dpi=150, bbox_inches="tight")
#     print(f"\n✓ Saved plot to {output_file}")
#     plt.close()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--log-dir", type=str, default="tensorboard_logs",
#                         help="Tensorboard log directory containing PPO_* runs")
#     parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
#                         help="Checkpoint directory (also where baseline_results_all.json lives)")
#     parser.add_argument("--no-baseline-std", action="store_true",
#                         help="Do not show ±std dotted lines for baselines")
#     args = parser.parse_args()

#     log_dir = Path(args.log_dir)
#     checkpoint_dir = Path(args.checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)

#     if not log_dir.exists():
#         print(f"Error: Log directory {log_dir} does not exist!")
#         return

#     print(f"Loading training results from {log_dir}...")
#     plot_training_results(log_dir, checkpoint_dir, baseline_std=not args.no_baseline_std)
#     print("\n✓ Done!")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Plot PPO training results from tensorboard logs and overlay baseline (random/greedy/unique) lines.
Also auto-plots ALL reward component scalars logged under "rew/*" as additional subplots.

Usage:
    python plot_training_results.py --log-dir tensorboard_logs --checkpoint-dir checkpoints_ppo

Notes:
- Baselines are loaded from:
    <checkpoint-dir>/baseline_results_all.json
  (created by eval_baseline.py)

- Reward component scalars are expected as TensorBoard tags starting with:
    "rew/"
  Those should be added by:
    1) merging info_reward into info in sb3_env_wrapper.py step()
    2) logging rew/* keys in FinalTaskAllocationCallback
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing import event_accumulator


# ---------------- Baseline overlay helpers ----------------

def load_baseline_stats(checkpoint_dir: Path):
    """
    Load baseline mean/std from eval_baseline.py outputs.
    Expects: baseline_results_all.json in checkpoint_dir.
    """
    p = checkpoint_dir / "baseline_results_all.json"
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for pol in ["random", "greedy", "unique"]:
        if pol in data and "stats" in data[pol]:
            out[pol] = {
                "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
                "std": float(data[pol]["stats"].get("reward_std", 0.0)),
            }
    return out


def add_baseline_lines(ax, baselines, show_std: bool = True):
    """Draw baseline horizontal lines on an existing matplotlib axis."""
    if not baselines:
        return

    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

    for pol, stats in baselines.items():
        mean = float(stats["mean"])
        std = float(stats.get("std", 0.0))
        c = colors.get(pol, "#999999")
        label = labels.get(pol, pol)

        ax.axhline(
            mean,
            color=c,
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            label=f"{label} baseline ({mean:.2f})",
        )
        if show_std and std > 0:
            ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
            ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)


# ---------------- Tensorboard utilities ----------------

def moving_average(data, window=50):
    """Compute moving average (same length output)."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    if window <= 1 or data.size < window:
        return data

    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_len = data.size - ma.size
    return np.concatenate([data[:pad_len], ma])


def load_tensorboard_data(log_dir: Path):
    """
    Load scalar series from the latest PPO_* run directory.
    Returns dict[tag] = {"steps":[...], "values":[...]}.
    """
    run_dirs = list(Path(log_dir).glob("PPO_*"))
    if not run_dirs:
        print(f"No tensorboard logs found in {log_dir}")
        return {}

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_run}")

    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    tags = ea.Tags()
    data = {}
    for tag in tags.get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
        }
    return data


def _plot_scalar(ax, series, title: str, color: str, ma_window: int = 20, yscale: str | None = None):
    ax.set_facecolor("#fafafa")
    steps = np.asarray(series["steps"], dtype=float)
    vals = np.asarray(series["values"], dtype=float)

    ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")
    if vals.size > ma_window:
        ax.plot(steps, moving_average(vals, ma_window), lw=2.2, color=color, label=f"MA({ma_window})")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=7, loc="best")


# ---------------- Plotting ----------------

def plot_training_results(
    log_dir: Path,
    checkpoint_dir: Path,
    baseline_std: bool = True,
    ma_reward_window: int = 50,
    ma_component_window: int = 20,
):
    data = load_tensorboard_data(log_dir)
    if not data:
        print("No data to plot!")
        return

    print(f"Available metrics: {list(data.keys())}")

    baselines = load_baseline_stats(checkpoint_dir)
    if baselines is None:
        print(f"[INFO] No baseline_results_all.json found in {checkpoint_dir} (baseline lines will be skipped).")
        print("[INFO] Run: python eval_baseline.py --output-dir <checkpoint-dir> ...")

    # Collect reward component tags
    rew_tags = sorted([k for k in data.keys() if k.startswith("rew/")])
    if rew_tags:
        print(f"[INFO] Found {len(rew_tags)} reward component tags (rew/*).")
    else:
        print("[INFO] No rew/* tags found. (Did you retrain after adding wrapper+callback logging?)")

    # Layout:
    # Row 0: PPO reward curve (full width)
    # Row 1: Completion + Outcomes (2 plots) if available
    # Row 2+: Auto reward components (N plots, 2 columns)
    #
    # We will keep Value Loss + Entropy at the end if present, after reward components.
    has_value_loss = "train/value_loss" in data
    has_entropy = "train/entropy_loss" in data

    # Compute how many extra rows we need for reward components (2 per row)
    n_comp = len(rew_tags)
    comp_rows = math.ceil(n_comp / 2) if n_comp > 0 else 0

    # Add 3 fixed rows (reward + completion/outcomes) + comp_rows + (maybe 1 row for losses)
    fixed_rows = 2  # row0 reward, row1 completion/outcomes
    loss_rows = 1 if (has_value_loss or has_entropy) else 0
    total_rows = fixed_rows + comp_rows + loss_rows
    total_rows = max(total_rows, 3)  # at least 3 rows for decent sizing

    fig = plt.figure(figsize=(18, 4.2 * total_rows), facecolor="white")
    gs = gridspec.GridSpec(total_rows, 2, figure=fig, hspace=0.45, wspace=0.25)

    # ========== Row 0: Episode Rewards (full width) ==========
    if "rollout/ep_rew_mean" in data:
        ax = fig.add_subplot(gs[0, :])
        ax.set_facecolor("#fafafa")

        steps = np.asarray(data["rollout/ep_rew_mean"]["steps"], dtype=float)
        rewards = np.asarray(data["rollout/ep_rew_mean"]["values"], dtype=float)

        ax.plot(steps, rewards, "o", markersize=4, alpha=0.3, color="#3498db", label="PPO raw")

        if rewards.size > ma_reward_window:
            ax.plot(steps, moving_average(rewards, ma_reward_window), lw=2.5, color="#2980b9",
                    label=f"PPO MA({ma_reward_window})")

        ax.axhline(
            float(np.mean(rewards)),
            color="#e74c3c",
            lw=2,
            ls="--",
            alpha=0.6,
            label=f"PPO mean: {float(np.mean(rewards)):.1f}",
        )

        if steps.size >= 2:
            z = np.polyfit(steps, rewards, 1)
            ax.plot(
                steps,
                np.polyval(z, steps),
                lw=1.8,
                color="#27ae60",
                alpha=0.5,
                ls=":",
                label=f"Trend: {z[0]:+.4f}/step",
            )

        # Baseline overlay
        add_baseline_lines(ax, baselines, show_std=baseline_std)

        ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
        ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
        ax.set_title("Training Rewards (PPO vs baselines)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

        if rewards.size >= 10:
            last_10 = rewards[-10:]
            txt = (
                f"PPO mean:   {float(np.mean(rewards)):.2f}\n"
                f"PPO last10: {float(np.mean(last_10)):.2f}\n"
                f"PPO best:   {float(np.max(rewards)):.2f}"
            )
            ax.text(
                0.98,
                0.04,
                txt,
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="right",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9),
            )
    else:
        print("[WARN] Missing 'rollout/ep_rew_mean' in tensorboard logs; reward plot will be skipped.")

    # ========== Row 1: Task Completion + Task Outcomes ==========
    row_idx = 1

    # Completion rate (left)
    if "task/completion_rate" in data:
        ax = fig.add_subplot(gs[row_idx, 0])
        _plot_scalar(
            ax,
            data["task/completion_rate"],
            title="Task Completion Rate (%)",
            color="#27ae60",
            ma_window=20,
        )
        ax.set_ylim(0, 105)
    else:
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.axis("off")
        ax.set_title("No task/completion_rate logged")

    # Completed vs obsolete (right)
    if "task/completed" in data and "task/obsolete" in data:
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.set_facecolor("#fafafa")

        steps_comp = np.asarray(data["task/completed"]["steps"], dtype=float)
        completed = np.asarray(data["task/completed"]["values"], dtype=float)
        obsolete = np.asarray(data["task/obsolete"]["values"], dtype=float)

        if completed.size > 20:
            ax.plot(steps_comp, moving_average(completed, 20), lw=2.5, color="#27ae60", label="Completed (MA20)")
            ax.plot(steps_comp, moving_average(obsolete, 20), lw=2.5, color="#e74c3c", label="Obsolete (MA20)")
        else:
            ax.plot(steps_comp, completed, lw=2.5, color="#27ae60", label="Completed")
            ax.plot(steps_comp, obsolete, lw=2.5, color="#e74c3c", label="Obsolete")

        ax.set_xlabel("Training Steps", fontsize=9, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9, fontweight="bold")
        ax.set_title("Task Outcomes", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    else:
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.axis("off")
        ax.set_title("No task/completed + task/obsolete logged")

    # ========== Rows 2.. : Reward components (rew/*) ==========
    row_idx = 2
    palette = [
        "#8e44ad", "#16a085", "#c0392b", "#d35400", "#2c3e50",
        "#7f8c8d", "#2980b9", "#27ae60", "#e67e22", "#9b59b6",
    ]

    for i, tag in enumerate(rew_tags):
        r = row_idx + (i // 2)
        c = i % 2
        ax = fig.add_subplot(gs[r, c])
        color = palette[i % len(palette)]
        _plot_scalar(ax, data[tag], title=f"Reward component: {tag}", color=color, ma_window=ma_component_window)

    # ========== Optional last row: value loss / entropy ==========
    if has_value_loss or has_entropy:
        loss_row = row_idx + comp_rows  # the first row after components

        # Value loss
        if has_value_loss:
            ax = fig.add_subplot(gs[loss_row, 0])
            _plot_scalar(ax, data["train/value_loss"], title="Critic Loss (train/value_loss)", color="#2980b9",
                         ma_window=20, yscale="log")
        else:
            ax = fig.add_subplot(gs[loss_row, 0])
            ax.axis("off")
            ax.set_title("No train/value_loss logged")

        # Entropy loss
        if has_entropy:
            ax = fig.add_subplot(gs[loss_row, 1])
            _plot_scalar(ax, data["train/entropy_loss"], title="Policy Entropy Loss (train/entropy_loss)", color="#9b59b6",
                         ma_window=20)
        else:
            ax = fig.add_subplot(gs[loss_row, 1])
            ax.axis("off")
            ax.set_title("No train/entropy_loss logged")

    plt.tight_layout()
    output_file = checkpoint_dir / "training_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs",
                        help="Tensorboard log directory containing PPO_* runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
                        help="Checkpoint directory (also where baseline_results_all.json lives)")
    parser.add_argument("--no-baseline-std", action="store_true",
                        help="Do not show ±std dotted lines for baselines")
    parser.add_argument("--ma-reward-window", type=int, default=50,
                        help="Moving average window for rollout/ep_rew_mean")
    parser.add_argument("--ma-component-window", type=int, default=20,
                        help="Moving average window for rew/* component plots")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist!")
        return

    plot_training_results(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        baseline_std=not args.no_baseline_std,
        ma_reward_window=args.ma_reward_window,
        ma_component_window=args.ma_component_window,
    )
    print("\n✓ Done!")


if __name__ == "__main__":
    main()