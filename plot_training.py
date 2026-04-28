#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


# ---------------- Run selection helpers ----------------

def latest_run_dir(seed_dir: Path) -> Path:
    runs = sorted(seed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {seed_dir}")
    return runs[-1]


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
    return {"mean": float(st.get("reward_mean", 0.0)), "std": float(st.get("reward_std", 0.0))}


def load_baseline_stats(root_dir: Path, seed: int) -> Optional[Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, float]] = {}
    for pol in ["random", "greedy", "unique"]:
        p = root_dir / f"baseline_{pol}_seed_{seed}.json"
        if not p.exists():
            print(f"[WARN] Missing baseline file: {p}")
            continue
        stats = _load_one_baseline_file(p)
        if stats is not None:
            out[pol] = stats
    return out or None


def add_baseline_lines(ax, baselines, show_std: bool = True):
    if not baselines:
        return
    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

    for pol in ["random", "greedy", "unique"]:
        if pol not in baselines:
            continue
        mean = float(baselines[pol]["mean"])
        std = float(baselines[pol].get("std", 0.0))
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
    tb_dir is run_dir/tensorboard. We still need latest PPO_* inside it.
    """
    run_dirs = list(tb_dir.glob("PPO_*"))
    if not run_dirs:
        print(f"[ERROR] No tensorboard runs found under: {tb_dir}")
        return {}

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Loading TensorBoard run: {latest_run}")

    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    data: Dict[str, Dict[str, List[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [float(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }
    return data


def _maybe_get(data: Dict[str, Any], tag: str) -> Optional[Dict[str, Any]]:
    return data.get(tag, None)


def _save_fig(fig, out_path: Path):
    if out_path.exists() and out_path.is_dir():
        raise IsADirectoryError(f"Output path is a directory, not a file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def _plot_series(ax, series, title: str, color: str, ma_window: int, yscale: Optional[str] = None):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", type=str, default="checkpoints_ppo")
    ap.add_argument("--run-id", type=str, default=None, help="run id like 20260421_153012; default: latest run")
    ap.add_argument("--ma-window", type=int, default=20)
    ap.add_argument("--no-baseline-std", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    seed_dir = root / f"seed_{int(args.seed)}"
    run_dir = seed_dir / f"run_{args.run_id}" if args.run_id else latest_run_dir(seed_dir)

    tb_dir = run_dir / "tensorboard"
    out_reward = run_dir / "training_results.png"
    out_dir = run_dir / "plots"

    data = load_tensorboard_data(tb_dir)
    if not data:
        return

    plot_rewards(
        data=data,
        root=root,
        seed=int(args.seed),
        out_png=out_reward,
        ma_window=int(args.ma_window),
        baseline_std=not args.no_baseline_std,
    )
    plot_exploration(data=data, out_png=out_dir / "exploration_entropy.png", ma_window=int(args.ma_window))
    plot_value_overview(data=data, out_png=out_dir / "value_overview.png", ma_window=int(args.ma_window))
    plot_value_loss_only(data=data, out_png=out_dir / "value_loss.png", ma_window=int(args.ma_window))


if __name__ == "__main__":
    main()