# # #!/usr/bin/env python3
# # import argparse
# # import json
# # import math
# # from pathlib import Path

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.gridspec as gridspec
# # from tensorboard.backend.event_processing import event_accumulator


# # # ---------------- Baseline overlay helpers ----------------

# # def load_baseline_stats(checkpoint_dir: Path):
# #     """
# #     Load baseline mean/std from eval_baseline.py outputs.
# #     Expects: baseline_results_all.json in checkpoint_dir.
# #     """
# #     p = checkpoint_dir / "baseline_results_all.json"
# #     if not p.exists():
# #         return None

# #     with p.open("r", encoding="utf-8") as f:
# #         data = json.load(f)

# #     out = {}
# #     for pol in ["random", "greedy", "unique"]:
# #         if pol in data and "stats" in data[pol]:
# #             out[pol] = {
# #                 "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
# #                 "std": float(data[pol]["stats"].get("reward_std", 0.0)),
# #             }
# #     return out


# # def add_baseline_lines(ax, baselines, show_std: bool = True):
# #     """Draw baseline horizontal lines on an existing matplotlib axis."""
# #     if not baselines:
# #         return

# #     colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
# #     labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

# #     for pol, stats in baselines.items():
# #         mean = float(stats["mean"])
# #         std = float(stats.get("std", 0.0))
# #         c = colors.get(pol, "#999999")
# #         label = labels.get(pol, pol)

# #         ax.axhline(mean, color=c, linestyle="--", linewidth=2.2, alpha=0.9,
# #                    label=f"{label} baseline ({mean:.2f})")
# #         if show_std and std > 0:
# #             ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
# #             ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)


# # # ---------------- Tensorboard utilities ----------------

# # def moving_average(data, window=50):
# #     data = np.asarray(data, dtype=float)
# #     if data.size == 0:
# #         return data
# #     if window <= 1 or data.size < window:
# #         return data
# #     cumsum = np.cumsum(np.insert(data, 0, 0.0))
# #     ma = (cumsum[window:] - cumsum[:-window]) / window
# #     pad_len = data.size - ma.size
# #     return np.concatenate([data[:pad_len], ma])


# # def load_tensorboard_data(log_dir: Path):
# #     """
# #     Load scalar series from the latest PPO_* run directory.
# #     Returns dict[tag] = {"steps":[...], "values":[...]}.
# #     """
# #     run_dirs = list(Path(log_dir).glob("PPO_*"))
# #     if not run_dirs:
# #         print(f"No tensorboard logs found in {log_dir}")
# #         return {}

# #     latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
# #     print(f"Loading TensorBoard run: {latest_run}")

# #     ea = event_accumulator.EventAccumulator(str(latest_run))
# #     ea.Reload()

# #     tags = ea.Tags()
# #     data = {}
# #     for tag in tags.get("scalars", []):
# #         events = ea.Scalars(tag)
# #         data[tag] = {
# #             "steps": [e.step for e in events],
# #             "values": [e.value for e in events],
# #         }
# #     return data


# # def _plot_scalar(ax, series, title: str, color: str, ma_window: int = 20, yscale: str | None = None):
# #     ax.set_facecolor("#fafafa")
# #     steps = np.asarray(series["steps"], dtype=float)
# #     vals = np.asarray(series["values"], dtype=float)

# #     ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")
# #     if vals.size > ma_window:
# #         ax.plot(steps, moving_average(vals, ma_window), lw=2.2, color=color, label=f"MA({ma_window})")

# #     ax.set_title(title, fontsize=10, fontweight="bold")
# #     ax.set_xlabel("Training Steps", fontsize=9, fontweight="bold")
# #     ax.grid(alpha=0.25)
# #     if yscale is not None:
# #         ax.set_yscale(yscale)
# #     ax.legend(fontsize=7, loc="best")


# # # ---------------- Plotting ----------------

# # def plot_training_results(
# #     log_dir: Path,
# #     checkpoint_dir: Path,
# #     out_png: Path,
# #     baseline_std: bool = True,
# #     ma_reward_window: int = 50,
# #     ma_component_window: int = 20,
# # ):
# #     data = load_tensorboard_data(log_dir)
# #     if not data:
# #         print("No tensorboard data to plot.")
# #         return

# #     baselines = load_baseline_stats(checkpoint_dir)

# #     rew_tags = sorted([k for k in data.keys() if k.startswith("rew/")])
# #     has_value_loss = "train/value_loss" in data
# #     has_entropy = "train/entropy_loss" in data

# #     n_comp = len(rew_tags)
# #     comp_rows = math.ceil(n_comp / 2) if n_comp > 0 else 0

# #     fixed_rows = 2  # reward + completion/outcomes
# #     loss_rows = 1 if (has_value_loss or has_entropy) else 0
# #     total_rows = max(3, fixed_rows + comp_rows + loss_rows)

# #     fig = plt.figure(figsize=(18, 4.2 * total_rows), facecolor="white")
# #     gs = gridspec.GridSpec(total_rows, 2, figure=fig, hspace=0.45, wspace=0.25)

# #     # Row 0: Episode reward mean
# #     if "rollout/ep_rew_mean" in data:
# #         ax = fig.add_subplot(gs[0, :])
# #         ax.set_facecolor("#fafafa")

# #         steps = np.asarray(data["rollout/ep_rew_mean"]["steps"], dtype=float)
# #         rewards = np.asarray(data["rollout/ep_rew_mean"]["values"], dtype=float)

# #         ax.plot(steps, rewards, "o", markersize=4, alpha=0.3, color="#3498db", label="PPO raw")
# #         if rewards.size > ma_reward_window:
# #             ax.plot(steps, moving_average(rewards, ma_reward_window), lw=2.5, color="#2980b9",
# #                     label=f"PPO MA({ma_reward_window})")

# #         ax.axhline(float(np.mean(rewards)), color="#e74c3c", lw=2, ls="--", alpha=0.6,
# #                    label=f"PPO mean: {float(np.mean(rewards)):.2f}")

# #         if steps.size >= 2:
# #             z = np.polyfit(steps, rewards, 1)
# #             ax.plot(steps, np.polyval(z, steps), lw=1.8, color="#27ae60", alpha=0.5, ls=":",
# #                     label=f"Trend: {z[0]:+.4f}/step")

# #         add_baseline_lines(ax, baselines, show_std=baseline_std)

# #         ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
# #         ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
# #         ax.set_title("Training Rewards (PPO vs baselines)", fontsize=14, fontweight="bold")
# #         ax.legend(loc="upper left", fontsize=9)
# #         ax.grid(alpha=0.25)
# #     else:
# #         ax = fig.add_subplot(gs[0, :])
# #         ax.axis("off")
# #         ax.set_title("Missing rollout/ep_rew_mean in TensorBoard logs")

# #     # Row 1: Completion rate + outcomes
# #     row_idx = 1

# #     if "task/completion_rate" in data:
# #         ax = fig.add_subplot(gs[row_idx, 0])
# #         _plot_scalar(ax, data["task/completion_rate"], "Task Completion Rate (%)", "#27ae60", ma_window=20)
# #         ax.set_ylim(0, 105)
# #     else:
# #         ax = fig.add_subplot(gs[row_idx, 0])
# #         ax.axis("off")
# #         ax.set_title("No task/completion_rate logged")

# #     if "task/completed" in data and "task/obsolete" in data:
# #         ax = fig.add_subplot(gs[row_idx, 1])
# #         ax.set_facecolor("#fafafa")
# #         steps_comp = np.asarray(data["task/completed"]["steps"], dtype=float)
# #         completed = np.asarray(data["task/completed"]["values"], dtype=float)
# #         obsolete = np.asarray(data["task/obsolete"]["values"], dtype=float)
# #         ax.plot(steps_comp, moving_average(completed, 20), lw=2.5, color="#27ae60", label="Completed (MA20)")
# #         ax.plot(steps_comp, moving_average(obsolete, 20), lw=2.5, color="#e74c3c", label="Obsolete (MA20)")
# #         ax.set_xlabel("Training Steps", fontsize=9, fontweight="bold")
# #         ax.set_ylabel("Count", fontsize=9, fontweight="bold")
# #         ax.set_title("Task Outcomes", fontsize=10, fontweight="bold")
# #         ax.legend(fontsize=8)
# #         ax.grid(alpha=0.25)
# #     else:
# #         ax = fig.add_subplot(gs[row_idx, 1])
# #         ax.axis("off")
# #         ax.set_title("No task/completed + task/obsolete logged")

# #     # Reward components
# #     row_idx = 2
# #     palette = [
# #         "#8e44ad", "#16a085", "#c0392b", "#d35400", "#2c3e50",
# #         "#7f8c8d", "#2980b9", "#27ae60", "#e67e22", "#9b59b6",
# #     ]

# #     for i, tag in enumerate(rew_tags):
# #         r = row_idx + (i // 2)
# #         c = i % 2
# #         ax = fig.add_subplot(gs[r, c])
# #         color = palette[i % len(palette)]
# #         _plot_scalar(ax, data[tag], f"Reward component: {tag}", color, ma_window=ma_component_window)

# #     # Loss row
# #     if has_value_loss or has_entropy:
# #         loss_row = row_idx + comp_rows
# #         if has_value_loss:
# #             ax = fig.add_subplot(gs[loss_row, 0])
# #             _plot_scalar(ax, data["train/value_loss"], "Critic Loss (train/value_loss)", "#2980b9", ma_window=20, yscale="log")
# #         else:
# #             ax = fig.add_subplot(gs[loss_row, 0])
# #             ax.axis("off")
# #             ax.set_title("No train/value_loss logged")

# #         if has_entropy:
# #             ax = fig.add_subplot(gs[loss_row, 1])
# #             _plot_scalar(ax, data["train/entropy_loss"], "Policy Entropy Loss (train/entropy_loss)", "#9b59b6", ma_window=20)
# #         else:
# #             ax = fig.add_subplot(gs[loss_row, 1])
# #             ax.axis("off")
# #             ax.set_title("No train/entropy_loss logged")

# #     out_png.parent.mkdir(parents=True, exist_ok=True)
# #     fig.savefig(out_png, dpi=150, bbox_inches="tight")
# #     plt.close(fig)
# #     print(f"✓ Saved: {out_png}")


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--log-dir", type=str, default="tensorboard_logs", help="Tensorboard log directory containing PPO_* runs")
# #     ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo", help="Where baseline_results_all.json lives")
# #     ap.add_argument("--out", type=str, default="checkpoints_ppo/training_results.png")
# #     ap.add_argument("--no-baseline-std", action="store_true", help="Do not show ±std dotted lines for baselines")
# #     ap.add_argument("--ma-reward-window", type=int, default=50)
# #     ap.add_argument("--ma-component-window", type=int, default=20)
# #     args = ap.parse_args()

# #     plot_training_results(
# #         log_dir=Path(args.log_dir),
# #         checkpoint_dir=Path(args.checkpoint_dir),
# #         out_png=Path(args.out),
# #         baseline_std=not args.no_baseline_std,
# #         ma_reward_window=args.ma_reward_window,
# #         ma_component_window=args.ma_component_window,
# #     )


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3#!/usr/bin/env python3
# """
# Simplified training plotter:

# 1) Rewards plot (with baseline overlay)
# 2) Exploration plot (entropy)
# 3) Value plot (approx_kl + clip_fraction + explained_variance + value_loss)
# 4) Value loss plot (standalone, log scale)

# Usage:
#   python3 plot_training.py \
#     --log-dir checkpoints_ppo/seed_456/tensorboard \
#     --checkpoint-dir checkpoints_ppo \
#     --out-dir checkpoints_ppo/seed_456/plots \
#     --ma-window 10
# """
# import argparse
# import json
# from pathlib import Path
# from typing import Optional, Dict, Any, List

# import numpy as np
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator


# # ---------------- Baseline overlay helpers ----------------

# def load_baseline_stats(checkpoint_dir: Path) -> Optional[Dict[str, Dict[str, float]]]:
#     """
#     Load baseline mean/std from eval_baseline.py outputs.
#     Expects: baseline_results_all.json in checkpoint_dir.
#     """
#     p = checkpoint_dir / "baseline_results_all.json"
#     if not p.exists():
#         return None

#     with p.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     out: Dict[str, Dict[str, float]] = {}
#     for pol in ["random", "greedy", "unique"]:
#         if pol in data and "stats" in data[pol]:
#             out[pol] = {
#                 "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
#                 "std": float(data[pol]["stats"].get("reward_std", 0.0)),
#             }
#     return out


# def add_baseline_lines(ax, baselines, show_std: bool = True):
#     """Draw baseline horizontal lines on an existing matplotlib axis."""
#     if not baselines:
#         return

#     colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
#     labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

#     for pol, stats in baselines.items():
#         mean = float(stats["mean"])
#         std = float(stats.get("std", 0.0))
#         c = colors.get(pol, "#999999")
#         label = labels.get(pol, pol)

#         ax.axhline(mean, color=c, linestyle="--", linewidth=2.0, alpha=0.9,
#                    label=f"{label} baseline ({mean:.2f})")
#         if show_std and std > 0:
#             ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.0, alpha=0.75)
#             ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.0, alpha=0.75)


# # ---------------- Tensorboard utilities ----------------

# def moving_average(data, window=50):
#     data = np.asarray(data, dtype=float)
#     if data.size == 0:
#         return data
#     if window <= 1 or data.size < window:
#         return data
#     cumsum = np.cumsum(np.insert(data, 0, 0.0))
#     ma = (cumsum[window:] - cumsum[:-window]) / window
#     pad_len = data.size - ma.size
#     return np.concatenate([data[:pad_len], ma])


# def load_tensorboard_data(log_dir: Path) -> Dict[str, Dict[str, List[float]]]:
#     """
#     Load scalar series from the latest PPO_* run directory.
#     Returns dict[tag] = {"steps":[...], "values":[...]}.
#     """
#     run_dirs = list(Path(log_dir).glob("PPO_*"))
#     if not run_dirs:
#         print(f"No tensorboard logs found in {log_dir}")
#         return {}

#     latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
#     print(f"Loading TensorBoard run: {latest_run}")

#     ea = event_accumulator.EventAccumulator(str(latest_run))
#     ea.Reload()

#     tags = ea.Tags()
#     data: Dict[str, Dict[str, List[float]]] = {}
#     for tag in tags.get("scalars", []):
#         events = ea.Scalars(tag)
#         data[tag] = {
#             "steps": [float(e.step) for e in events],
#             "values": [float(e.value) for e in events],
#         }
#     return data


# def _plot_series(
#     ax,
#     series,
#     title: str,
#     color: str,
#     ma_window: int = 20,
#     yscale: Optional[str] = None,
#     show_raw_points: bool = True,
# ):
#     ax.set_facecolor("#fafafa")
#     steps = np.asarray(series["steps"], dtype=float)
#     vals = np.asarray(series["values"], dtype=float)

#     if show_raw_points:
#         ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")

#     ax.plot(steps, moving_average(vals, ma_window), lw=2.4, color=color, label=f"MA({ma_window})")

#     ax.set_title(title, fontsize=12, fontweight="bold")
#     ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
#     ax.grid(alpha=0.25)
#     if yscale is not None:
#         ax.set_yscale(yscale)
#     ax.legend(fontsize=9, loc="best")


# def _maybe_get(data: Dict[str, Any], tag: str) -> Optional[Dict[str, Any]]:
#     return data.get(tag, None)


# def _save_fig(fig, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Saved: {out_path}")


# # ---------------- Plotting ----------------

# def plot_rewards(data, checkpoint_dir: Path, out_path: Path, ma_window: int, baseline_std: bool):
#     baselines = load_baseline_stats(checkpoint_dir)

#     fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
#     ax.set_facecolor("#fafafa")

#     s = _maybe_get(data, "rollout/ep_rew_mean")
#     if s is None:
#         ax.axis("off")
#         ax.set_title("Missing rollout/ep_rew_mean in TensorBoard logs", fontsize=14, fontweight="bold")
#         _save_fig(fig, out_path)
#         return

#     steps = np.asarray(s["steps"], dtype=float)
#     rewards = np.asarray(s["values"], dtype=float)

#     ax.plot(steps, rewards, "o", markersize=4, alpha=0.25, color="#3498db", label="PPO raw")
#     ax.plot(steps, moving_average(rewards, ma_window), lw=2.8, color="#2980b9", label=f"PPO MA({ma_window})")

#     ax.axhline(float(np.mean(rewards)), color="#2c3e50", lw=2, ls="--", alpha=0.65,
#                label=f"PPO mean: {float(np.mean(rewards)):.2f}")

#     add_baseline_lines(ax, baselines, show_std=baseline_std)

#     ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Episode Reward Mean", fontsize=11, fontweight="bold")
#     ax.set_title("Training Rewards (PPO vs Baselines)", fontsize=15, fontweight="bold")
#     ax.grid(alpha=0.25)
#     ax.legend(loc="best", fontsize=9)

#     _save_fig(fig, out_path)


# def plot_exploration(data, out_path: Path, ma_window: int):
#     # SB3 commonly logs either train/entropy_loss or train/entropy
#     entropy_loss = _maybe_get(data, "train/entropy_loss")
#     entropy = _maybe_get(data, "train/entropy")

#     fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
#     if entropy_loss is not None:
#         _plot_series(ax, entropy_loss, "Exploration: Entropy Loss (train/entropy_loss)", "#9b59b6", ma_window=ma_window)
#     elif entropy is not None:
#         _plot_series(ax, entropy, "Exploration: Entropy (train/entropy)", "#9b59b6", ma_window=ma_window)
#     else:
#         ax.axis("off")
#         ax.set_title("No entropy tag found (train/entropy_loss or train/entropy).", fontsize=14, fontweight="bold")

#     _save_fig(fig, out_path)


# def plot_value_overview(data, out_path: Path, ma_window: int):
#     """
#     One figure with 4 subplots:
#       - train/value_loss
#       - train/explained_variance
#       - train/approx_kl
#       - train/clip_fraction
#     """
#     tags = [
#         ("train/value_loss", "Value: Critic loss (train/value_loss)", "#2980b9", "log"),
#         ("train/explained_variance", "Value: Explained variance (train/explained_variance)", "#16a085", None),
#         ("train/approx_kl", "Value/Policy: Approx KL (train/approx_kl)", "#c0392b", None),
#         ("train/clip_fraction", "Policy: Clip fraction (train/clip_fraction)", "#d35400", None),
#     ]

#     fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
#     axes = axes.reshape(2, 2)

#     any_plotted = False
#     for i, (tag, title, color, yscale) in enumerate(tags):
#         r, c = divmod(i, 2)
#         ax = axes[r, c]
#         s = _maybe_get(data, tag)
#         if s is None:
#             ax.axis("off")
#             ax.set_title(f"Missing {tag}", fontsize=12, fontweight="bold")
#             continue
#         any_plotted = True
#         _plot_series(ax, s, title, color, ma_window=ma_window, yscale=yscale, show_raw_points=True)

#     if not any_plotted:
#         plt.close(fig)
#         fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
#         ax.axis("off")
#         ax.set_title("No value-related tags found (train/value_loss etc.).", fontsize=14, fontweight="bold")

#     _save_fig(fig, out_path)


# def plot_value_loss_only(data, out_path: Path, ma_window: int):
#     s = _maybe_get(data, "train/value_loss")
#     fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
#     if s is None:
#         ax.axis("off")
#         ax.set_title("Missing train/value_loss in TensorBoard logs", fontsize=14, fontweight="bold")
#     else:
#         _plot_series(ax, s, "Critic Loss (train/value_loss) [log scale]", "#2980b9", ma_window=ma_window, yscale="log")
#     _save_fig(fig, out_path)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--log-dir", type=str, required=True,
#                     help="Tensorboard log directory containing PPO_* runs (e.g., checkpoints_ppo/seed_456/tensorboard)")
#     ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
#                     help="Directory containing baseline_results_all.json for overlay.")
#     ap.add_argument("--out-dir", type=str, default="plots",
#                     help="Output directory where PNGs are saved.")
#     ap.add_argument("--no-baseline-std", action="store_true", help="Do not show ±std dotted lines for baselines.")
#     ap.add_argument("--ma-window", type=int, default=20, help="Moving average window.")
#     args = ap.parse_args()

#     data = load_tensorboard_data(Path(args.log_dir))
#     if not data:
#         print("No tensorboard data to plot.")
#         return

#     out_dir = Path(args.out_dir)

#     plot_rewards(
#         data=data,
#         checkpoint_dir=Path(args.checkpoint_dir),
#         out_path=out_dir / "reward.png",
#         ma_window=args.ma_window,
#         baseline_std=not args.no_baseline_std,
#     )
#     plot_exploration(
#         data=data,
#         out_path=out_dir / "exploration_entropy.png",
#         ma_window=args.ma_window,
#     )
#     plot_value_overview(
#         data=data,
#         out_path=out_dir / "value_overview.png",
#         ma_window=args.ma_window,
#     )
#     plot_value_loss_only(
#         data=data,
#         out_path=out_dir / "value_loss.png",
#         ma_window=args.ma_window,
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Training plotter for your checkpoints_ppo/ structure (seed-aligned baselines).

Your directory layout (example):
checkpoints_ppo/
  baseline_random_seed_42.json
  baseline_greedy_seed_42.json
  baseline_unique_seed_42.json
  seed_42/
    tensorboard/PPO_*/
    training_results.png   (output file)
    plots/                 (optional extra output images)

What this script does:
1) Reward plot (PPO reward + baseline lines for SAME seed)
2) Exploration plot (entropy)
3) Value overview (approx_kl, clip_fraction, explained_variance, value_loss)
4) Value loss only (log scale)

It will read baselines from:
  checkpoints_ppo/baseline_{random,greedy,unique}_seed_<seed>.json

Usage (seed 42):
  python3 plot_training.py --seed 42

Usage (seed 456, custom MA):
  python3 plot_training.py --seed 456 --ma-window 10

You can override paths if needed:
  python3 plot_training.py \
    --seed 42 \
    --root checkpoints_ppo \
    --log-dir checkpoints_ppo/seed_42/tensorboard \
    --out checkpoints_ppo/seed_42/training_results.png \
    --out-dir checkpoints_ppo/seed_42/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

# Force a non-Qt backend to avoid QSocketNotifier/Qt backend issues in headless runs
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


# ---------------- Baseline overlay helpers ----------------

def _load_one_baseline_file(p: Path) -> Optional[Dict[str, float]]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read {p}: {e}")
        return None

    st = (data or {}).get("stats", {})
    if not isinstance(st, dict):
        return None

    return {
        "mean": float(st.get("reward_mean", 0.0)),
        "std": float(st.get("reward_std", 0.0)),
    }


def load_baseline_stats(root_dir: Path, seed: int) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load baseline mean/std for a specific seed from root checkpoints_ppo/ directory.
    """
    out: Dict[str, Dict[str, float]] = {}
    for pol in ["random", "greedy", "unique"]:
        p = root_dir / f"baseline_{pol}_seed_{seed}.json"
        if not p.exists():
            print(f"[WARN] Missing baseline file: {p}")
            continue
        stats = _load_one_baseline_file(p)
        if stats is not None:
            out[pol] = stats

    if out:
        loaded = ", ".join([f"{k}(mean={v['mean']:.3f}, std={v['std']:.3f})" for k, v in out.items()])
        print(f"[INFO] Loaded seed={seed} baselines from {root_dir}: {loaded}")
        return out

    print(f"[WARN] No baselines loaded for seed={seed} from {root_dir}")
    return None


def add_baseline_lines(ax, baselines, show_std: bool = True):
    if not baselines:
        return

    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

    for pol in ["random", "greedy", "unique"]:
        if pol not in baselines:
            continue

        stats = baselines[pol]
        mean = float(stats["mean"])
        std = float(stats.get("std", 0.0))
        c = colors.get(pol, "#999999")
        label = labels.get(pol, pol)

        ax.axhline(mean, color=c, linestyle="--", linewidth=2.2, alpha=0.9,
                   label=f"{label} baseline ({mean:.2f})")
        if show_std and std > 0:
            ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)
            ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.2, alpha=0.75)


# ---------------- Tensorboard utilities ----------------

def moving_average(data, window=50):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    if window <= 1 or data.size < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_len = data.size - ma.size
    return np.concatenate([data[:pad_len], ma])


def load_tensorboard_data(tb_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Load scalar series from the latest PPO_* run directory inside tb_dir.
    Returns dict[tag] = {"steps":[...], "values":[...]}.
    """
    run_dirs = list(Path(tb_dir).glob("PPO_*"))
    if not run_dirs:
        print(f"[ERROR] No tensorboard runs found under: {tb_dir}")
        return {}

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Loading TensorBoard run: {latest_run}")

    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    tags = ea.Tags()
    data: Dict[str, Dict[str, List[float]]] = {}
    for tag in tags.get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [float(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }
    return data


def _plot_series(
    ax,
    series,
    title: str,
    color: str,
    ma_window: int,
    yscale: Optional[str] = None,
):
    ax.set_facecolor("#fafafa")
    steps = np.asarray(series["steps"], dtype=float)
    vals = np.asarray(series["values"], dtype=float)

    ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")
    ax.plot(steps, moving_average(vals, ma_window), lw=2.4, color=color, label=f"MA({ma_window})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=9, loc="best")


def _maybe_get(data: Dict[str, Any], tag: str) -> Optional[Dict[str, Any]]:
    return data.get(tag, None)


def _save_fig(fig, out_path: Path):
    # common pitfall: directory named "...png"
    if out_path.exists() and out_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory, not a file: {out_path}\n"
            f"Rename/remove it and re-run."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


# ---------------- Plotting ----------------

def plot_rewards(data: Dict[str, Any], root: Path, seed: int, out_png: Path, ma_window: int, baseline_std: bool):
    baselines = load_baseline_stats(root, seed)

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    ax.set_facecolor("#fafafa")

    s = _maybe_get(data, "rollout/ep_rew_mean")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing rollout/ep_rew_mean in TensorBoard logs", fontsize=14, fontweight="bold")
        _save_fig(fig, out_png)
        return

    steps = np.asarray(s["steps"], dtype=float)
    rewards = np.asarray(s["values"], dtype=float)

    ax.plot(steps, rewards, "o", markersize=4, alpha=0.25, color="#3498db", label="PPO raw")
    ax.plot(steps, moving_average(rewards, ma_window), lw=2.8, color="#2980b9", label=f"PPO MA({ma_window})")

    ax.axhline(float(np.mean(rewards)), color="#2c3e50", lw=2, ls="--", alpha=0.65,
               label=f"PPO mean: {float(np.mean(rewards)):.2f}")

    add_baseline_lines(ax, baselines, show_std=baseline_std)

    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward Mean", fontsize=11, fontweight="bold")
    ax.set_title(f"Training Rewards (Seed {seed}) - PPO vs Baselines", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    _save_fig(fig, out_png)


def plot_exploration(data: Dict[str, Any], out_png: Path, ma_window: int):
    entropy_loss = _maybe_get(data, "train/entropy_loss")
    entropy = _maybe_get(data, "train/entropy")

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    if entropy_loss is not None:
        _plot_series(ax, entropy_loss, "Exploration: Entropy Loss (train/entropy_loss)", "#9b59b6", ma_window)
    elif entropy is not None:
        _plot_series(ax, entropy, "Exploration: Entropy (train/entropy)", "#9b59b6", ma_window)
    else:
        ax.axis("off")
        ax.set_title("No entropy tag found (train/entropy_loss or train/entropy).", fontsize=14, fontweight="bold")

    _save_fig(fig, out_png)


def plot_value_overview(data: Dict[str, Any], out_png: Path, ma_window: int):
    tags = [
        ("train/value_loss", "Value: Critic loss (train/value_loss)", "#2980b9", "log"),
        ("train/explained_variance", "Value: Explained variance (train/explained_variance)", "#16a085", None),
        ("train/approx_kl", "Value/Policy: Approx KL (train/approx_kl)", "#c0392b", None),
        ("train/clip_fraction", "Policy: Clip fraction (train/clip_fraction)", "#d35400", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    axes = axes.reshape(2, 2)

    any_plotted = False
    for i, (tag, title, color, yscale) in enumerate(tags):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        s = _maybe_get(data, tag)
        if s is None:
            ax.axis("off")
            ax.set_title(f"Missing {tag}", fontsize=12, fontweight="bold")
            continue
        any_plotted = True
        _plot_series(ax, s, title, color, ma_window, yscale=yscale)

    if not any_plotted:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
        ax.axis("off")
        ax.set_title("No value-related tags found (train/value_loss etc.).", fontsize=14, fontweight="bold")

    _save_fig(fig, out_png)


def plot_value_loss_only(data: Dict[str, Any], out_png: Path, ma_window: int):
    s = _maybe_get(data, "train/value_loss")

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing train/value_loss in TensorBoard logs", fontsize=14, fontweight="bold")
    else:
        _plot_series(ax, s, "Critic Loss (train/value_loss) [log scale]", "#2980b9", ma_window, yscale="log")
    _save_fig(fig, out_png)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="Seed folder name under checkpoints_ppo/seed_<seed>/")
    ap.add_argument("--root", type=str, default="checkpoints_ppo", help="Root checkpoints directory")
    ap.add_argument("--log-dir", type=str, default=None, help="Override tensorboard directory (otherwise inferred)")
    ap.add_argument("--out", type=str, default=None, help="Reward plot output PNG (otherwise inferred)")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory for extra plots (otherwise inferred)")
    ap.add_argument("--no-baseline-std", action="store_true")
    ap.add_argument("--ma-window", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    seed_dir = root / f"seed_{int(args.seed)}"

    tb_dir = Path(args.log_dir) if args.log_dir else (seed_dir / "tensorboard")
    out_reward = Path(args.out) if args.out else (seed_dir / "training_results.png")
    out_dir = Path(args.out_dir) if args.out_dir else (seed_dir / "plots")

    data = load_tensorboard_data(tb_dir)
    if not data:
        return

    # Reward (with baseline overlay)
    plot_rewards(
        data=data,
        root=root,
        seed=int(args.seed),
        out_png=out_reward,
        ma_window=int(args.ma_window),
        baseline_std=not args.no_baseline_std,
    )

    # Extra plots
    plot_exploration(data=data, out_png=out_dir / "exploration_entropy.png", ma_window=int(args.ma_window))
    plot_value_overview(data=data, out_png=out_dir / "value_overview.png", ma_window=int(args.ma_window))
    plot_value_loss_only(data=data, out_png=out_dir / "value_loss.png", ma_window=int(args.ma_window))


if __name__ == "__main__":
    main()
