#!/usr/bin/env python3
"""
Plot-only script.

Reads:
  - per-seed eval JSONs created by eval_ppo.py
  - TensorBoard logs under <checkpoint-dir>/seed_<seed>/tensorboard/
  - baseline_results_all.json + baseline_{policy}_seed_{seed}.json if present

Writes plots under:
  <checkpoint-dir>/seed_<seed>/eval_plots/
  <checkpoint-dir>/eval_plots/ (aggregate)

Usage:
  python3 plot_evaluation.py --checkpoint-dir checkpoints_ppo
  python3 plot_evaluation.py --checkpoint-dir checkpoints_ppo --seeds 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


# ---------------- helpers ----------------

def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.size == 0 or window <= 1 or data.size < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_len = data.size - ma.size
    return np.concatenate([data[:pad_len], ma])

def _latest_run_dir(seed_dir: Path) -> Path:
    runs = sorted(seed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {seed_dir}")
    return runs[-1]


def _pick_run_dir(seed_dir: Path, run_id: Optional[str]) -> Path:
    if run_id:
        rd = seed_dir / f"run_{run_id}"
        if not rd.exists():
            raise FileNotFoundError(f"Run dir not found: {rd}")
        return rd
    return _latest_run_dir(seed_dir)

# ---------------- baselines ----------------

def _get_baseline_block(data: Dict) -> Dict:
    if "results" in data and isinstance(data["results"], dict):
        return data["results"]
    return data


def load_baselines_all(checkpoint_dir: Path) -> Dict[str, Dict]:
    """Load baseline_results_all.json → {RANDOM: {...}, GREEDY: {...}, UNIQUE: {...}}"""
    p = checkpoint_dir / "baseline_results_all.json"
    if not p.exists():
        return {}
    data = _load_json(p)
    block = _get_baseline_block(data)
    out: Dict[str, Dict] = {}
    for pol in ["random", "greedy", "unique"]:
        b = block.get(pol, None)
        if isinstance(b, dict):
            out[pol.upper()] = {
                "rewards": b.get("rewards", []),
                "completions": b.get("completions", []),
                "obsolete": b.get("obsolete", []),
            }
    return out


def load_baseline_stats_for_seed(root_dir: Path, seed: int) -> Optional[Dict[str, Dict[str, float]]]:
    """Load per-seed baseline JSON files → {random: {mean, std}, ...}"""
    out: Dict[str, Dict[str, float]] = {}
    for pol in ["random", "greedy", "unique"]:
        p = root_dir / f"baseline_{pol}_seed_{seed}.json"
        if not p.exists():
            continue
        try:
            data = _load_json(p)
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}")
            continue
        st = (data or {}).get("stats", {})
        if isinstance(st, dict):
            out[pol] = {"mean": float(st.get("reward_mean", 0.0)), "std": float(st.get("reward_std", 0.0))}
    return out if out else None


# ---------------- tensorboard ----------------

def load_tensorboard_data(tb_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    run_dirs = list(tb_dir.glob("PPO_*"))
    if not run_dirs:
        print(f"[WARN] No TensorBoard runs found under: {tb_dir}")
        return {}
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"  Loading TensorBoard run: {latest_run}")
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    data: Dict[str, Dict[str, List[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {"steps": [float(e.step) for e in events], "values": [float(e.value) for e in events]}
    return data


# ---------------- eval plots ----------------

def plot_eval_rewards_per_episode(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
    ma_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    for label, color, data in [
        ("PPO Deterministic", "#2980b9", det_data),
        ("PPO Stochastic", "#e67e22", stoch_data),
    ]:
        if not data:
            continue
        rewards = np.asarray(data.get("rewards", []), dtype=float)
        if rewards.size == 0:
            continue
        x = np.arange(1, rewards.size + 1)
        ax.plot(x, rewards, alpha=0.18, color=color, linewidth=0.8)
        ax.plot(x, _moving_average(rewards, ma_window), lw=2.6, color=color,
                label=f"{label} MA({ma_window}) – mean {rewards.mean():.2f}")
        ax.axhline(rewards.mean(), color=color, lw=1.4, ls="--", alpha=0.55)

    baseline_order = ["RANDOM", "GREEDY", "UNIQUE"]
    baseline_colors = {"RANDOM": "#e74c3c", "GREEDY": "#f39c12", "UNIQUE": "#8c564b"}
    for name in baseline_order:
        if name not in baselines:
            continue
        rewards = np.asarray(baselines[name].get("rewards", []), dtype=float)
        if rewards.size == 0:
            continue
        x = np.arange(1, rewards.size + 1)
        c = baseline_colors.get(name, "#7f8c8d")
        ax.plot(x, rewards, alpha=0.10, color=c, linewidth=0.8)
        ax.plot(x, _moving_average(rewards, ma_window), lw=2.2, color=c,
                label=f"{name} MA({ma_window}) – mean {rewards.mean():.2f}")
        ax.axhline(rewards.mean(), color=c, lw=1.2, ls="--", alpha=0.45)

    ax.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Per-Episode Rewards (PPO Det/Stoch + Baselines)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_eval_rewards_boxplot(det_data: Optional[Dict], stoch_data: Optional[Dict], baselines: Dict[str, Dict], out_png: Path) -> None:
    ordered: Dict[str, np.ndarray] = {}
    if det_data:
        r = np.asarray(det_data.get("rewards", []), dtype=float)
        if r.size:
            ordered["PPO Det"] = r
    if stoch_data:
        r = np.asarray(stoch_data.get("rewards", []), dtype=float)
        if r.size:
            ordered["PPO Stoch"] = r
    for name, d in baselines.items():
        r = np.asarray(d.get("rewards", []), dtype=float)
        if r.size:
            ordered[name] = r

    if not ordered:
        print("[WARN] No reward data for boxplot.")
        return

    labels = list(ordered.keys())
    series = list(ordered.values())

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    colors_list = ["#2980b9", "#e67e22", "#e74c3c", "#f39c12", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_list[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Episode reward", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Reward Distribution (Det vs Stoch vs Baselines)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_eval_completion_obsolete(det_data: Optional[Dict], stoch_data: Optional[Dict], baselines: Dict[str, Dict], out_png: Path) -> None:
    methods: Dict[str, Dict] = {}
    if det_data:
        methods["PPO Det"] = det_data
    if stoch_data:
        methods["PPO Stoch"] = stoch_data
    methods.update(baselines)

    if not methods:
        return

    labels: List[str] = []
    comp_means: List[float] = []
    obs_means: List[float] = []

    for name, d in methods.items():
        c = np.asarray(d.get("completions", []), dtype=float)
        o = np.asarray(d.get("obsolete", []), dtype=float)
        labels.append(name)
        comp_means.append(float(c.mean()) if c.size else 0.0)
        obs_means.append(float(o.mean()) if o.size else 0.0)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.bar(x - w / 2, comp_means, width=w, label="Completed (mean)", color="#27ae60", alpha=0.8)
    ax.bar(x + w / 2, obs_means, width=w, label="Obsolete (mean)", color="#c0392b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count per episode", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Completions & Obsolete (Det vs Stoch vs Baselines)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_eval_reward_components(det_data: Optional[Dict], stoch_data: Optional[Dict], out_png: Path) -> None:
    keys = ["rew/pickups_this_step", "rew/deliveries_this_step", "rew/obsolete_this_step", "rew/step_penalty"]

    def _extract(d: Optional[Dict]) -> Optional[Dict[str, np.ndarray]]:
        if not d:
            return None
        rc = d.get("reward_components", None)
        if not isinstance(rc, dict):
            return None
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            arr = np.asarray(rc.get(k, []), dtype=float)
            if arr.size:
                out[k] = arr
        return out if out else None

    det_rc = _extract(det_data)
    st_rc = _extract(stoch_data)
    if det_rc is None and st_rc is None:
        print("[INFO] No reward_components in eval JSON; skipping component plot.")
        return

    present = [k for k in keys if (det_rc and k in det_rc) or (st_rc and k in st_rc)]
    if not present:
        return

    x = np.arange(len(present))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(present) * 2.2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")

    for offset, label, color, rc in [
        (-w / 2, "PPO Det", "#2980b9", det_rc),
        (+w / 2, "PPO Stoch", "#e67e22", st_rc),
    ]:
        if rc is None:
            continue
        means = [float(rc[k].mean()) if k in rc else 0.0 for k in present]
        stds = [float(rc[k].std()) if k in rc else 0.0 for k in present]
        ax.bar(x + offset, means, width=w, yerr=stds, capsize=5, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Per-episode sum (mean ± std)", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Component Sums (Det vs Stoch)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)

def plot_training_logits(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    tags = [t for t in tb_data.keys() if t.startswith("logits/")]
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    if not tags:
        ax.axis("off")
        ax.set_title("No logits/* tags found in TensorBoard logs.", fontsize=14, fontweight="bold")
        _save_fig(fig, out_png)
        return

    # If both exist, don't plot NOOP twice:
    # - keep logits/noop_mean
    # - drop logits/action_<noop_idx>_mean where noop_idx inferred as max action index present
    action_tags = []
    for t in tags:
        name = t.split("/", 1)[1]
        if name.startswith("action_") and name.endswith("_mean"):
            action_tags.append(name)

    # infer noop_idx as the largest action index seen (MultiDiscrete head => NOOP is last index)
    action_indices = []
    for name in action_tags:
        try:
            idx = int(name.split("_")[1])
            action_indices.append(idx)
        except Exception:
            pass

    noop_idx = max(action_indices) if action_indices else None
    if noop_idx is not None and "logits/noop_mean" in tags:
        noop_action_tag = f"logits/action_{noop_idx}_mean"
        tags = [t for t in tags if t != noop_action_tag]

    # Plot in stable order: action_0..action_K then noop_mean last
    def _key(t: str):
        name = t.split("/", 1)[1]
        if name.startswith("action_"):
            try:
                idx = int(name.split("_")[1])
                return (0, idx)
            except Exception:
                return (1, name)
        if name == "noop_mean":
            return (2, 999999)
        return (3, name)

    tags = sorted(tags, key=_key)

    for tag in tags:
        s = tb_data[tag]
        steps = np.asarray(s["steps"], dtype=float)
        vals = np.asarray(s["values"], dtype=float)
        ax.plot(steps, _moving_average(vals, ma_window), lw=2.0, label=tag.replace("logits/", ""))

    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean logit (masked, active robots)", fontsize=11, fontweight="bold")
    ax.set_title("Training: Per-Action Mean Logits (meaningful steps)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)

def plot_training_logit_gaps(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    """
    Plot logit-separation diagnostics:
      - gap_best_minus_noop_mean
      - gap_best_minus_second_mean
    (optionally also max_task_logit_mean and noop_logit_mean)
    """
    wanted = [
        "logits/gap_best_minus_noop_mean",
        "logits/gap_best_minus_second_mean",
        "logits/max_task_logit_mean",
        "logits/noop_logit_mean",
        "logits/second_best_task_logit_mean",
    ]
    tags = [t for t in wanted if t in tb_data]

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    if not tags:
        ax.axis("off")
        ax.set_title("No logit-gap tags found (logits/gap_*).", fontsize=14, fontweight="bold")
        _save_fig(fig, out_png)
        return

    # stable order
    order = {t: i for i, t in enumerate(wanted)}
    tags = sorted(tags, key=lambda t: order.get(t, 999))

    for tag in tags:
        s = tb_data[tag]
        steps = np.asarray(s["steps"], dtype=float)
        vals = np.asarray(s["values"], dtype=float)
        ax.plot(
            steps,
            _moving_average(vals, ma_window),
            lw=2.0,
            label=tag.replace("logits/", ""),
        )

    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Logit / gap (moving avg)", fontsize=11, fontweight="bold")
    ax.set_title("Training: Logit Separation Diagnostics (meaningful steps)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)
# ---------------- training plots ----------------

def _plot_series(ax, series: Dict[str, List[float]], title: str, color: str, ma_window: int, yscale: Optional[str] = None) -> None:
    ax.set_facecolor("#fafafa")
    steps = np.asarray(series["steps"], dtype=float)
    vals = np.asarray(series["values"], dtype=float)
    ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")
    ax.plot(steps, _moving_average(vals, ma_window), lw=2.4, color=color, label=f"MA({ma_window})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend(fontsize=9, loc="best")


def _add_baseline_lines(ax, baselines: Optional[Dict[str, Dict[str, float]]], show_std: bool) -> None:
    if not baselines:
        return
    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    display = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}
    for pol in ["random", "greedy", "unique"]:
        if pol not in baselines:
            continue
        mean = float(baselines[pol]["mean"])
        std = float(baselines[pol].get("std", 0.0))
        c = colors[pol]
        ax.axhline(mean, color=c, ls="--", lw=2.2, alpha=0.9, label=f"{display[pol]} baseline ({mean:.2f})")
        if show_std and std > 0:
            ax.axhline(mean + std, color=c, ls=":", lw=1.2, alpha=0.75)
            ax.axhline(mean - std, color=c, ls=":", lw=1.2, alpha=0.75)


def plot_training_rewards(tb_data: Dict, seed: int, baselines: Optional[Dict], out_png: Path, ma_window: int, baseline_std: bool) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    ax.set_facecolor("#fafafa")
    s = tb_data.get("rollout/ep_rew_mean")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing rollout/ep_rew_mean in TensorBoard logs", fontsize=14, fontweight="bold")
        _save_fig(fig, out_png)
        return

    steps = np.asarray(s["steps"], dtype=float)
    rewards = np.asarray(s["values"], dtype=float)
    ax.plot(steps, rewards, "o", markersize=4, alpha=0.25, color="#3498db", label="PPO raw")
    ax.plot(steps, _moving_average(rewards, ma_window), lw=2.8, color="#2980b9", label=f"PPO MA({ma_window})")
    ax.axhline(float(np.mean(rewards)), color="#2c3e50", lw=2, ls="--", alpha=0.65,
               label=f"PPO mean: {float(np.mean(rewards)):.2f}")
    _add_baseline_lines(ax, baselines, baseline_std)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward Mean", fontsize=11, fontweight="bold")
    ax.set_title(f"Training Rewards (Seed {seed}) – PPO vs Baselines", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_training_entropy(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    s = tb_data.get("train/entropy_loss") or tb_data.get("train/entropy")
    if s is None:
        ax.axis("off")
        ax.set_title("No entropy tag found (train/entropy_loss or train/entropy).", fontsize=14, fontweight="bold")
    else:
        tag = "train/entropy_loss" if "train/entropy_loss" in tb_data else "train/entropy"
        _plot_series(ax, s, f"Exploration: Entropy ({tag})", "#9b59b6", ma_window)
    _save_fig(fig, out_png)


def plot_training_policy_behavior(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    tags = [
        ("policy/noop_fraction_meaningful", "NOOP fraction (meaningful)", "#34495e"),
        ("policy/assigned_fraction_meaningful", "Assigned fraction (meaningful)", "#27ae60"),
        ("policy/collision_drop_fraction_meaningful", "Collision-drop fraction (meaningful)", "#c0392b"),
    ]

    present = [(t, name, c) for (t, name, c) in tags if t in tb_data]
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    if not present:
        ax.axis("off")
        ax.set_title(
            "Missing policy behavior tags in TensorBoard.\n"
            "Expected at least one of:\n"
            "  policy/noop_fraction_meaningful\n"
            "  policy/assigned_fraction_meaningful\n"
            "  policy/collision_drop_fraction_meaningful\n\n"
            "Fix: update FinalTaskAllocationCallback to record these tags.",
            fontsize=13,
            fontweight="bold",
        )
        _save_fig(fig, out_png)
        return

    for tag, label, color in present:
        s = tb_data[tag]
        steps = np.asarray(s["steps"], dtype=float)
        vals = np.asarray(s["values"], dtype=float)
        ax.plot(steps, vals, "o", markersize=3, alpha=0.18, color=color)
        ax.plot(steps, _moving_average(vals, ma_window), lw=2.8, color=color,
                label=f"{label} MA({ma_window}) – mean {float(np.mean(vals)):.3f}")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraction", fontsize=11, fontweight="bold")
    ax.set_title("Training: Policy Behavior on Meaningful Decision Steps", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_training_value_overview(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    tags = [
        ("train/value_loss", "Critic loss (train/value_loss)", "#2980b9", "log"),
        ("train/explained_variance", "Explained variance (train/explained_variance)", "#16a085", None),
        ("train/approx_kl", "Approx KL (train/approx_kl)", "#c0392b", None),
        ("train/clip_fraction", "Clip fraction (train/clip_fraction)", "#d35400", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    any_plotted = False
    for i, (tag, title, color, yscale) in enumerate(tags):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        s = tb_data.get(tag)
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
        ax.set_title("No value-related tags found.", fontsize=14, fontweight="bold")
    _save_fig(fig, out_png)


def plot_training_value_loss(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    s = tb_data.get("train/value_loss")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing train/value_loss in TensorBoard logs", fontsize=14, fontweight="bold")
    else:
        _plot_series(ax, s, "Critic Loss (train/value_loss) [log scale]", "#2980b9", ma_window, yscale="log")
    _save_fig(fig, out_png)


# ---------------- aggregate plots ----------------

def plot_agg_rewards_boxplot(all_det: List[Dict], all_stoch: List[Dict], baselines: Dict[str, Dict], out_png: Path) -> None:
    ordered: Dict[str, np.ndarray] = {}
    det_rewards = [r for d in all_det for r in d.get("rewards", [])]
    st_rewards = [r for d in all_stoch for r in d.get("rewards", [])]
    if det_rewards:
        ordered["PPO Det"] = np.asarray(det_rewards, dtype=float)
    if st_rewards:
        ordered["PPO Stoch"] = np.asarray(st_rewards, dtype=float)
    for name, d in baselines.items():
        arr = np.asarray(d.get("rewards", []), dtype=float)
        if arr.size:
            ordered[name] = arr
    if not ordered:
        return

    labels = list(ordered.keys())
    series = list(ordered.values())
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    colors_list = ["#2980b9", "#e67e22", "#e74c3c", "#f39c12", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_list[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Episode reward", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate Evaluation: Reward Distribution (all seeds)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_agg_completion_obsolete(all_det: List[Dict], all_stoch: List[Dict], baselines: Dict[str, Dict], out_png: Path) -> None:
    methods: Dict[str, Dict] = {}
    det_comp = [c for d in all_det for c in d.get("completions", [])]
    det_obs = [o for d in all_det for o in d.get("obsolete", [])]
    st_comp = [c for d in all_stoch for c in d.get("completions", [])]
    st_obs = [o for d in all_stoch for o in d.get("obsolete", [])]
    if det_comp:
        methods["PPO Det"] = {"completions": det_comp, "obsolete": det_obs}
    if st_comp:
        methods["PPO Stoch"] = {"completions": st_comp, "obsolete": st_obs}
    methods.update(baselines)
    if not methods:
        return

    labels = list(methods.keys())
    comp_means = [float(np.mean(d.get("completions", [0]))) for d in methods.values()]
    obs_means = [float(np.mean(d.get("obsolete", [0]))) for d in methods.values()]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.bar(x - w / 2, comp_means, width=w, label="Completed (mean)", color="#27ae60", alpha=0.8)
    ax.bar(x + w / 2, obs_means, width=w, label="Obsolete (mean)", color="#c0392b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count per episode", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate Evaluation: Completions & Obsolete (all seeds)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def process_seed(seed: int, run_dir: Path, root_dir: Path, ma_window: int, baseline_std: bool):
    det_json = run_dir / "eval_results_deterministic.json"
    stoch_json = run_dir / "eval_results_stochastic.json"

    if not det_json.exists() or not stoch_json.exists():
        print(f"[ERROR] Missing eval JSON(s) for seed {seed} in {run_dir}. Run eval_ppo.py first.")
        return None, None

    det_result = _load_json(det_json)
    stoch_result = _load_json(stoch_json)

    plots_dir = run_dir / "eval_plots"

    baselines_for_plot = load_baselines_all(root_dir)
    baselines_stats = load_baseline_stats_for_seed(root_dir, seed)

    print(f"\n  [seed {seed}] run={run_dir.name} Generating evaluation plots → {plots_dir}")
    plot_eval_rewards_per_episode(det_result, stoch_result, baselines_for_plot, plots_dir / "eval_rewards_per_episode.png", ma_window)
    plot_eval_rewards_boxplot(det_result, stoch_result, baselines_for_plot, plots_dir / "eval_rewards_boxplot.png")
    plot_eval_completion_obsolete(det_result, stoch_result, baselines_for_plot, plots_dir / "eval_completion_obsolete.png")
    plot_eval_reward_components(det_result, stoch_result, plots_dir / "eval_reward_components.png")

    # training plots from TensorBoard
    tb_dir = run_dir / "tensorboard"
    if tb_dir.exists():
        tb_data = load_tensorboard_data(tb_dir)
        if tb_data:
            print(f"  [seed {seed}] Generating training plots → {plots_dir}")
            # ---- call your existing training plotters (same as before) ----
            plot_training_rewards(tb_data, seed, baselines_stats, plots_dir / "training_rewards.png", ma_window, baseline_std)
            plot_training_entropy(tb_data, plots_dir / "training_entropy.png", ma_window)
            plot_training_policy_behavior(tb_data, plots_dir / "training_policy_behavior.png", ma_window)
            plot_training_value_overview(tb_data, plots_dir / "training_value_overview.png", ma_window)
            plot_training_value_loss(tb_data, plots_dir / "training_value_loss.png", ma_window)
            plot_training_logits(tb_data, out_png=plots_dir / "training_logits.png", ma_window=ma_window)
            plot_training_logit_gaps(tb_data, out_png=plots_dir / "training_logit_gaps.png", ma_window=ma_window)
    else:
        print(f"  [seed {seed}] No TensorBoard dir at {tb_dir}; skipping training plots.")

    return det_result, stoch_result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot-only: reads eval JSON + TensorBoard and generates plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--run-id", type=str, default=None, help="If set, use run_<id> for all seeds; else latest run per seed.")
    ap.add_argument("--ma-window", type=int, default=5)
    ap.add_argument("--no-baseline-std", action="store_true")
    args = ap.parse_args()

    root_dir = Path(args.checkpoint_dir)

    if args.seeds:
        seed_dirs: List[Tuple[int, Path]] = [(int(s), root_dir / f"seed_{int(s)}") for s in args.seeds]
    else:
        seed_dirs = [
            (int(d.name.replace("seed_", "")), d)
            for d in sorted(root_dir.glob("seed_*"))
            if d.is_dir()
        ]

    if not seed_dirs:
        print(f"[ERROR] No seed directories found in {root_dir}.")
        return

    all_det: List[Dict] = []
    all_stoch: List[Dict] = []

    for seed, seed_dir in seed_dirs:
        if not seed_dir.exists():
            continue
        try:
            run_dir = _pick_run_dir(seed_dir, args.run_id)
        except Exception as e:
            print(f"[WARN] seed {seed}: {e} – skipping")
            continue

        det, st = process_seed(
            seed=seed,
            run_dir=run_dir,
            root_dir=root_dir,
            ma_window=args.ma_window,
            baseline_std=not args.no_baseline_std,
        )
        if det is not None:
            all_det.append(det)
        if st is not None:
            all_stoch.append(st)

    # aggregate plots (all evaluated runs)
    agg_plots_dir = root_dir / "eval_plots"
    baselines = load_baselines_all(root_dir)

    print(f"\nGenerating aggregate plots → {agg_plots_dir}")
    plot_agg_rewards_boxplot(all_det, all_stoch, baselines, agg_plots_dir / "agg_rewards_boxplot.png")
    plot_agg_completion_obsolete(all_det, all_stoch, baselines, agg_plots_dir / "agg_completion_obsolete.png")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()