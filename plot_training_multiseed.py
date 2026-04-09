#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_baseline_stats(checkpoint_dir: Path):
    p = checkpoint_dir / "baseline_results_all.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())

    out = {}
    for pol in ["random", "greedy", "unique"]:
        if pol in data and "stats" in data[pol]:
            out[pol] = {
                "mean": float(data[pol]["stats"].get("reward_mean", 0.0)),
                "std": float(data[pol]["stats"].get("reward_std", 0.0)),
            }
    return out


def add_baseline_lines(ax, baselines, show_std: bool = True):
    if not baselines:
        return
    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    labels = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}

    for pol, stats in baselines.items():
        mean = float(stats["mean"])
        std = float(stats.get("std", 0.0))
        c = colors.get(pol, "#999999")
        label = labels.get(pol, pol)

        ax.axhline(mean, color=c, linestyle="--", linewidth=2.0, alpha=0.9,
                   label=f"{label} baseline ({mean:.2f})")
        if show_std and std > 0:
            ax.axhline(mean + std, color=c, linestyle=":", linewidth=1.0, alpha=0.75)
            ax.axhline(mean - std, color=c, linestyle=":", linewidth=1.0, alpha=0.75)


def load_scalar_from_run(run_dir: Path, tag: str):
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    steps = np.asarray([e.step for e in events], dtype=np.int64)
    vals = np.asarray([e.value for e in events], dtype=np.float32)
    return steps, vals


def find_latest_run(tb_root: Path):
    runs = list(tb_root.glob("PPO_*"))
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def align_by_index(curves):
    """
    curves: list of (steps, vals) with possibly different lengths.
    Align by truncating to shortest length and using that index grid.
    Returns common_steps, matrix [n_seeds, T]
    """
    lengths = [len(v) for _, v in curves]
    T = min(lengths)
    steps0 = curves[0][0][:T]
    mat = np.stack([v[:T] for _, v in curves], axis=0)
    return steps0, mat


def plot_mean_std(ax, steps, mat, label, color):
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(steps, mean, color=color, lw=2.5, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
                    help="Directory containing seed_*/tensorboard/PPO_*")
    ap.add_argument("--out", type=str, default="checkpoints_ppo/training_results_multiseed.png")
    ap.add_argument("--tag", type=str, default="rollout/ep_rew_mean",
                    help="TensorBoard scalar tag to plot")
    ap.add_argument("--no-baseline-std", action="store_true")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    seed_dirs = sorted(checkpoint_dir.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {checkpoint_dir}")

    curves = []
    used_seeds = []

    for sd in seed_dirs:
        tb_root = sd / "tensorboard"
        run_dir = find_latest_run(tb_root)
        if run_dir is None:
            print(f"[WARN] No PPO_* run found under {tb_root} (skipping)")
            continue

        res = load_scalar_from_run(run_dir, args.tag)
        if res is None:
            print(f"[WARN] Tag '{args.tag}' not found in {run_dir} (skipping)")
            continue

        steps, vals = res
        curves.append((steps, vals))
        used_seeds.append(sd.name)

    if len(curves) == 0:
        raise RuntimeError(f"No curves found for tag '{args.tag}' across seeds.")

    steps, mat = align_by_index(curves)

    baselines = load_baseline_stats(checkpoint_dir)

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafafa")

    plot_mean_std(ax, steps, mat, label=f"{args.tag} (mean±std over {len(curves)} seeds)", color="#2980b9")
    add_baseline_lines(ax, baselines, show_std=(not args.no_baseline_std))

    ax.set_title("Training (multi-seed)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training steps", fontsize=11, fontweight="bold")
    ax.set_ylabel(args.tag, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out}")
    print(f"Used seeds: {used_seeds}")


if __name__ == "__main__":
    main()