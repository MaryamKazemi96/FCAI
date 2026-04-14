
#!/usr/bin/env python3
"""
Evaluate PPO checkpoints (deterministic + stochastic) and save per-seed JSON results.

This script does NOT generate plots.

Per-seed outputs:
  <checkpoint-dir>/seed_<seed>/eval_results_deterministic.json
  <checkpoint-dir>/seed_<seed>/eval_results_stochastic.json

Aggregate outputs (optional):
  <checkpoint-dir>/eval_results_all_seeds_deterministic.json
  <checkpoint-dir>/eval_results_all_seeds_stochastic.json

Usage:
  python3 eval_ppo.py --checkpoint-dir checkpoints_ppo --episodes 100
  python3 eval_ppo.py --checkpoint-dir checkpoints_ppo --seeds 42 --episodes 200
  python3 eval_ppo.py --checkpoint-dir checkpoints_ppo --skip-existing
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.sb3_gnn_policy import RTGNNPolicy


# ---------------- I/O helpers ----------------

def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✓ Saved JSON: {p}")


# ---------------- env/config helpers ----------------

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: Dict, seed: int):
    env_cfg = config["environment"]
    agents = np.load(env_cfg["agents_file"], allow_pickle=True)

    batches = []
    for i in range(env_cfg["n_batches"]):
        batch_file = Path(env_cfg["data_dir"]) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file, allow_pickle=True))

    base_env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_cfg["radius"],
        feature_size=env_cfg["feature_size"],
        use_true_id=env_cfg["use_true_id"],
        all_batches=True,
    )
    env = WarehouseEnvSB3Final(
        base_env,
        assignment_interval=env_cfg["assignment_interval"],
        k_max=env_cfg.get("k_max", 5),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# These are "info" keys produced by your env debug reward dict merged as rew/<key>.
COMP_KEYS: List[str] = [
    "rew/pickups_this_step",
    "rew/deliveries_this_step",
    "rew/obsolete_this_step",
    "rew/step_penalty",
]


def run_evaluation(
    model_path: Path,
    config: Dict,
    seed: int,
    n_episodes: int,
    deterministic: bool,
) -> Dict[str, Any]:
    set_seed(seed)

    env = make_env(config, seed=seed)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env, custom_objects={"policy_class": RTGNNPolicy})

    rewards: List[float] = []
    completions: List[int] = []
    obsolete: List[int] = []
    lengths: List[int] = []
    comp_sums: Dict[str, List[float]] = {k: [] for k in COMP_KEYS}

    mode_label = "deterministic" if deterministic else "stochastic"

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_comp: Dict[str, float] = {k: 0.0 for k in COMP_KEYS}
        ep_completed = 0
        ep_obsolete = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            ep_reward += float(r[0])
            ep_len += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info, dict):
                for k in COMP_KEYS:
                    v = info.get(k, None)
                    if isinstance(v, (int, float, np.number)):
                        ep_comp[k] += float(v)

                if "episode_completed" in info:
                    ep_completed = int(info["episode_completed"])
                if "episode_obsolete" in info:
                    ep_obsolete = int(info["episode_obsolete"])

        rewards.append(ep_reward)
        lengths.append(ep_len)
        completions.append(ep_completed)
        obsolete.append(ep_obsolete)
        for k in COMP_KEYS:
            comp_sums[k].append(ep_comp[k])

        if (ep + 1) % 10 == 0:
            print(
                f"  [{mode_label}] seed {seed} ep {ep+1}/{n_episodes}: "
                f"reward={ep_reward:.2f} completed={ep_completed} obsolete={ep_obsolete}"
            )

    vec_env.close()

    return {
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "deterministic": bool(deterministic),
        "rewards": rewards,
        "lengths": lengths,
        "completions": completions,
        "obsolete": obsolete,
        "reward_components": comp_sums,
        "stats": {
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "completion_mean": float(np.mean(completions)) if completions else 0.0,
            "obsolete_mean": float(np.mean(obsolete)) if obsolete else 0.0,
        },
    }


def _agg(results: List[Dict], episodes_per_seed: int, deterministic: bool) -> Dict[str, Any]:
    agg_rewards = [r for d in results for r in d.get("rewards", [])]
    agg_comp = [c for d in results for c in d.get("completions", [])]
    agg_obs = [o for d in results for o in d.get("obsolete", [])]
    return {
        "n_seeds": len(results),
        "seeds": [d["seed"] for d in results],
        "episodes_per_seed": int(episodes_per_seed),
        "deterministic": bool(deterministic),
        "rewards": agg_rewards,
        "completions": agg_comp,
        "obsolete": agg_obs,
        "stats": {
            "reward_mean": float(np.mean(agg_rewards)) if agg_rewards else 0.0,
            "reward_std": float(np.std(agg_rewards)) if agg_rewards else 0.0,
            "completion_mean": float(np.mean(agg_comp)) if agg_comp else 0.0,
            "obsolete_mean": float(np.mean(agg_obs)) if agg_obs else 0.0,
        },
        "per_seed": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate PPO (det + stoch) and save JSON results (no plotting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--model-name", type=str, default="ppo_final")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip evaluation if both JSON results already exist for a seed.")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Do not write all-seeds aggregate JSONs.")
    args = ap.parse_args()

    config = load_config(args.config)
    root_dir = Path(args.checkpoint_dir)

    if args.seeds:
        seed_dirs = [(s, root_dir / f"seed_{s}") for s in args.seeds]
    else:
        seed_dirs = [
            (int(d.name.replace("seed_", "")), d)
            for d in sorted(root_dir.glob("seed_*"))
        ]

    if not seed_dirs:
        print(f"[ERROR] No seed directories found in {root_dir}.")
        return

    all_det: List[Dict] = []
    all_stoch: List[Dict] = []

    for seed, seed_dir in seed_dirs:
        model_path = seed_dir / f"{args.model_name}.zip"
        if not model_path.exists():
            print(f"[WARN] Model not found: {model_path} – skipping seed {seed}")
            continue

        det_json = seed_dir / "eval_results_deterministic.json"
        stoch_json = seed_dir / "eval_results_stochastic.json"

        if args.skip_existing and det_json.exists() and stoch_json.exists():
            print(f"  [seed {seed}] JSONs exist, skipping eval.")
            det = _load_json(det_json)
            st = _load_json(stoch_json)
        else:
            print(f"\n{'=' * 70}")
            print(f"  [seed {seed}] Running deterministic eval ({args.episodes} episodes)")
            print(f"{'=' * 70}")
            det = run_evaluation(model_path, config, seed, args.episodes, deterministic=True)
            _save_json(det, det_json)

            print(f"\n{'=' * 70}")
            print(f"  [seed {seed}] Running stochastic eval ({args.episodes} episodes)")
            print(f"{'=' * 70}")
            st = run_evaluation(model_path, config, seed, args.episodes, deterministic=False)
            _save_json(st, stoch_json)

        all_det.append(det)
        all_stoch.append(st)

    if args.no_aggregate:
        return

    if all_det:
        _save_json(_agg(all_det, args.episodes, True), root_dir / "eval_results_all_seeds_deterministic.json")
    if all_stoch:
        _save_json(_agg(all_stoch, args.episodes, False), root_dir / "eval_results_all_seeds_stochastic.json")


if __name__ == "__main__":
    main()