# #!/usr/bin/env python3
# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt


# def _load_json(p: Path):
#     with p.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def load_baselines(checkpoint_dir: Path):
#     p = checkpoint_dir / "baseline_results_all.json"
#     if not p.exists():
#         return {}
#     data = _load_json(p)

#     out = {}
#     for pol in ["random", "greedy", "unique"]:
#         if pol in data:
#             out[pol] = {
#                 "rewards": data[pol].get("rewards", []),
#                 "completions": data[pol].get("completions", []),
#                 "obsolete": data[pol].get("obsolete", []),
#             }
#     return out


# def load_ppo_eval(checkpoint_dir: Path):
#     """
#     Supports either:
#     - checkpoints_ppo/eval_results.json
#     - checkpoints_ppo/seed_*/eval_results.json (then aggregates across seeds)
#     """
#     direct = checkpoint_dir / "eval_results.json"
#     if direct.exists():
#         d = _load_json(direct)
#         return {"PPO": d}

#     seed_files = sorted(checkpoint_dir.glob("seed_*/eval_results.json"))
#     if not seed_files:
#         return {}

#     # aggregate by concatenation
#     agg = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
#     for sf in seed_files:
#         d = _load_json(sf)
#         for k in agg.keys():
#             if k in d:
#                 agg[k].extend(d[k])
#     return {"PPO": agg}


# def plot_rewards_boxplot(all_methods, out_png: Path):
#     labels = []
#     series = []
#     for name, d in all_methods.items():
#         r = np.asarray(d.get("rewards", []), dtype=float)
#         if r.size == 0:
#             continue
#         labels.append(name)
#         series.append(r)

#     if not series:
#         print("No reward data found to plot.")
#         return

#     fig = plt.figure(figsize=(9, 4.5))
#     plt.boxplot(series, labels=labels, showmeans=True)
#     plt.ylabel("Episode reward")
#     plt.title("Evaluation reward distribution")
#     plt.grid(alpha=0.25, axis="y")
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Saved: {out_png}")


# def plot_completion_obsolete(all_methods, out_png: Path):
#     labels = []
#     comp_means = []
#     obs_means = []

#     for name, d in all_methods.items():
#         c = np.asarray(d.get("completions", []), dtype=float)
#         o = np.asarray(d.get("obsolete", []), dtype=float)
#         if c.size == 0 and o.size == 0:
#             continue
#         labels.append(name)
#         comp_means.append(float(c.mean()) if c.size else 0.0)
#         obs_means.append(float(o.mean()) if o.size else 0.0)

#     if not labels:
#         print("No completion/obsolete data found to plot.")
#         return

#     x = np.arange(len(labels))
#     w = 0.35

#     fig = plt.figure(figsize=(10, 4.5))
#     plt.bar(x - w/2, comp_means, width=w, label="Completed (mean)")
#     plt.bar(x + w/2, obs_means, width=w, label="Obsolete (mean)")
#     plt.xticks(x, labels)
#     plt.ylabel("Count per episode")
#     plt.title("Evaluation outcomes")
#     plt.grid(alpha=0.25, axis="y")
#     plt.legend()
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Saved: {out_png}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
#     ap.add_argument("--out-dir", type=str, default="checkpoints_ppo")
#     args = ap.parse_args()

#     checkpoint_dir = Path(args.checkpoint_dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     baselines = load_baselines(checkpoint_dir)
#     ppo = load_ppo_eval(checkpoint_dir)

#     all_methods = {}
#     all_methods.update(ppo)
#     # add baselines after PPO
#     for k in ["random", "greedy", "unique"]:
#         if k in baselines:
#             all_methods[k.upper()] = baselines[k]

#     plot_rewards_boxplot(all_methods, out_dir / "eval_rewards_boxplot.png")
#     plot_completion_obsolete(all_methods, out_dir / "eval_completion_obsolete.png")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_baseline_block(baseline_json: dict):
    """
    Supports two formats:
      A) {"random": {...}, "greedy": {...}, "unique": {...}}
      B) {"results": {"random": {...}, "greedy": {...}, "unique": {...}}, ...}
    Returns dict with keys random/greedy/unique if present.
    """
    if isinstance(baseline_json, dict) and "results" in baseline_json and isinstance(baseline_json["results"], dict):
        return baseline_json["results"]
    return baseline_json


def load_baselines(checkpoint_dir: Path):
    p = checkpoint_dir / "baseline_results_all.json"
    if not p.exists():
        return {}

    data = _load_json(p)
    block = _get_baseline_block(data)

    out = {}
    for pol in ["random", "greedy", "unique"]:
        if pol in block and isinstance(block[pol], dict):
            out[pol.upper()] = {
                "rewards": block[pol].get("rewards", []),
                "completions": block[pol].get("completions", []),
                "obsolete": block[pol].get("obsolete", []),
            }
    return out


def load_ppo_eval(checkpoint_dir: Path):
    """
    Supports:
      - eval_results_all_seeds.json (preferred for multi-seed)
      - eval_results.json (single run)
      - seed_*/eval_results.json (concatenate)
    Returns {"PPO": dict}
    """
    all_seeds = checkpoint_dir / "eval_results_all_seeds.json"
    if all_seeds.exists():
        d = _load_json(all_seeds)
        # This file has aggregated arrays at top-level: rewards/completions/obsolete
        return {"PPO": d}

    direct = checkpoint_dir / "eval_results.json"
    if direct.exists():
        d = _load_json(direct)
        return {"PPO": d}

    seed_files = sorted(checkpoint_dir.glob("seed_*/eval_results.json"))
    if not seed_files:
        return {}

    agg = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
    comp_keys = ["rew/pickup", "rew/delivery", "rew/obsolete", "rew/step_penalty"]
    agg_components = {k: [] for k in comp_keys}

    for sf in seed_files:
        d = _load_json(sf)
        for k in agg.keys():
            if k in d:
                agg[k].extend(d[k])

        # optional components per seed
        rc = d.get("reward_components", None)
        if isinstance(rc, dict):
            for ck in comp_keys:
                if ck in rc and isinstance(rc[ck], list):
                    agg_components[ck].extend(rc[ck])

    # only attach components if we have at least one entry
    if any(len(v) > 0 for v in agg_components.values()):
        agg["reward_components"] = agg_components

    return {"PPO": agg}


def plot_rewards_boxplot(all_methods, out_png: Path):
    labels = []
    series = []
    for name, d in all_methods.items():
        r = np.asarray(d.get("rewards", []), dtype=float)
        if r.size == 0:
            continue
        labels.append(name)
        series.append(r)

    if not series:
        print("No reward data found to plot.")
        return

    fig = plt.figure(figsize=(10, 4.8))
    plt.boxplot(series, labels=labels, showmeans=True)
    plt.ylabel("Episode reward")
    plt.title("Evaluation reward distribution (boxplot)")
    plt.grid(alpha=0.25, axis="y")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")


def plot_completion_obsolete(all_methods, out_png: Path):
    labels = []
    comp_means = []
    obs_means = []

    for name, d in all_methods.items():
        c = np.asarray(d.get("completions", []), dtype=float)
        o = np.asarray(d.get("obsolete", []), dtype=float)
        if c.size == 0 and o.size == 0:
            continue
        labels.append(name)
        comp_means.append(float(c.mean()) if c.size else 0.0)
        obs_means.append(float(o.mean()) if o.size else 0.0)

    if not labels:
        print("No completion/obsolete data found to plot.")
        return

    x = np.arange(len(labels))
    w = 0.35

    fig = plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - w/2, comp_means, width=w, label="Completed (mean)")
    plt.bar(x + w/2, obs_means, width=w, label="Obsolete (mean)")
    plt.xticks(x, labels)
    plt.ylabel("Count per episode")
    plt.title("Evaluation outcomes (mean completed vs obsolete)")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")


def plot_reward_components(ppo_block: dict, out_png: Path):
    """
    Plot PPO reward component means ± std if eval JSON includes:
      ppo_block["reward_components"] = {"rew/pickup":[...], ...}
    """
    rc = ppo_block.get("reward_components", None)
    if not isinstance(rc, dict):
        print("[INFO] No reward_components in PPO eval json; skipping component plot.")
        return

    keys = ["rew/pickup", "rew/delivery", "rew/obsolete", "rew/step_penalty"]
    means = []
    stds = []
    labels = []
    for k in keys:
        arr = np.asarray(rc.get(k, []), dtype=float)
        if arr.size == 0:
            continue
        labels.append(k)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))

    if not labels:
        print("[INFO] reward_components dict exists but is empty; skipping component plot.")
        return

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(10.5, 4.8))
    plt.bar(x, means, yerr=stds, capsize=6, alpha=0.85)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Reward per episode (mean ± std)")
    plt.title("PPO evaluation reward components")
    plt.grid(alpha=0.25, axis="y")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    ap.add_argument("--out-dir", type=str, default="checkpoints_ppo")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = load_baselines(checkpoint_dir)
    ppo = load_ppo_eval(checkpoint_dir)

    all_methods = {}
    all_methods.update(ppo)
    all_methods.update(baselines)

    plot_rewards_boxplot(all_methods, out_dir / "eval_rewards_boxplot.png")
    plot_completion_obsolete(all_methods, out_dir / "eval_completion_obsolete.png")

    if "PPO" in ppo:
        plot_reward_components(ppo["PPO"], out_dir / "eval_reward_components.png")


if __name__ == "__main__":
    main()