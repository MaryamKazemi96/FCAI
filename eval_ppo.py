
# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import torch
# import yaml

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

# from src.environment.environment import MultiTaskAllocationEnv
# from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
# from src.models.sb3_gnn_policy import RTGNNPolicy


# def load_config(config_path: str):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# def set_seed(seed: int):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def make_env(config, seed: int):
#     env_config = config["environment"]
#     agents = np.load(env_config["agents_file"], allow_pickle=True)

#     batches = []
#     for i in range(env_config["n_batches"]):
#         batch_file = Path(env_config["data_dir"]) / f"tasks_batch_{i}.npy"
#         if batch_file.exists():
#             batches.append(np.load(batch_file, allow_pickle=True))

#     base_env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=batches,
#         radius=env_config["radius"],
#         feature_size=env_config["feature_size"],
#         use_true_id=env_config["use_true_id"],
#         all_batches=True,
#     )

#     env = WarehouseEnvSB3Final(
#         base_env,
#         assignment_interval=env_config["assignment_interval"],
#         k_max=env_config.get("k_max", 5),
#     )

#     env = Monitor(env)
#     env.reset(seed=seed)
#     return env


# def evaluate_one_seed(model_path: Path, config, seed: int, n_episodes: int, deterministic: bool):
#     set_seed(seed)

#     env = make_env(config, seed=seed)
#     vec_env = DummyVecEnv([lambda: env])

#     model = PPO.load(model_path, env=vec_env, custom_objects={"policy_class": RTGNNPolicy})

#     rewards = []
#     completions = []
#     obsolete = []
#     lengths = []

#     # component sums per episode (only if present in info)
#     comp_keys = ["rew/pickup", "rew/delivery", "rew/obsolete", "rew/step_penalty"]
#     comp_sums = {k: [] for k in comp_keys}

#     for ep in range(n_episodes):
#         obs = vec_env.reset()
#         done = False
#         ep_reward = 0.0
#         ep_len = 0

#         ep_comp = {k: 0.0 for k in comp_keys}
#         ep_completed = 0
#         ep_obsolete = 0

#         while not done:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, r, dones, infos = vec_env.step(action)

#             done = bool(dones[0])
#             ep_reward += float(r[0])
#             ep_len += 1

#             info = infos[0] if isinstance(infos, (list, tuple)) else infos

#             # accumulate reward components if they exist
#             if isinstance(info, dict):
#                 for k in comp_keys:
#                     v = info.get(k, None)
#                     if isinstance(v, (int, float, np.number)):
#                         ep_comp[k] += float(v)

#                 # episode stats if wrapper provides them
#                 if "episode_completed" in info:
#                     ep_completed = int(info.get("episode_completed", 0))
#                 if "episode_obsolete" in info:
#                     ep_obsolete = int(info.get("episode_obsolete", 0))

#         rewards.append(ep_reward)
#         lengths.append(ep_len)
#         completions.append(ep_completed)
#         obsolete.append(ep_obsolete)
#         for k in comp_keys:
#             comp_sums[k].append(ep_comp[k])

#         if (ep + 1) % 10 == 0:
#             print(f"[seed {seed}] Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, len={ep_len}, completed={ep_completed}, obsolete={ep_obsolete}")

#     out = {
#         "seed": int(seed),
#         "n_episodes": int(n_episodes),
#         "deterministic": bool(deterministic),
#         "rewards": rewards,
#         "lengths": lengths,
#         "completions": completions,
#         "obsolete": obsolete,
#         "reward_components": comp_sums,  # may be all zeros if not logged
#         "stats": {
#             "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
#             "reward_std": float(np.std(rewards)) if rewards else 0.0,
#             "completion_mean": float(np.mean(completions)) if completions else 0.0,
#             "obsolete_mean": float(np.mean(obsolete)) if obsolete else 0.0,
#         },
#     }
#     vec_env.close()
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str, default="configs/training_config.yaml")
#     ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
#     ap.add_argument("--episodes", type=int, default=100)
#     ap.add_argument("--deterministic", action="store_true")
#     ap.add_argument("--model-name", type=str, default="ppo_final", help="model filename inside seed dir (without .zip)")
#     args = ap.parse_args()

#     config = load_config(args.config)

#     checkpoint_dir = Path(args.checkpoint_dir)
#     seed_dirs = sorted(checkpoint_dir.glob("seed_*"))
#     if not seed_dirs:
#         raise FileNotFoundError(f"No seed_* dirs found in {checkpoint_dir}. Did you train multi-seed?")

#     all_seed_results = []
#     for sd in seed_dirs:
#         seed = int(sd.name.replace("seed_", ""))
#         model_path = sd / f"{args.model_name}.zip"
#         if not model_path.exists():
#             print(f"[WARN] Missing model for {sd.name}: {model_path} (skipping)")
#             continue

#         print("\n" + "=" * 80)
#         print(f"Evaluating seed {seed} model: {model_path}")
#         print("=" * 80)

#         res = evaluate_one_seed(
#             model_path=model_path,
#             config=config,
#             seed=seed,
#             n_episodes=args.episodes,
#             deterministic=args.deterministic,
#         )

#         per_seed_out = sd / "eval_results.json"
#         per_seed_out.write_text(json.dumps(res, indent=2))
#         print(f"✓ Saved per-seed eval to: {per_seed_out}")

#         all_seed_results.append(res)

#     if not all_seed_results:
#         raise RuntimeError("No seed results were evaluated (no models found).")

#     # Aggregate
#     agg_rewards = [r for s in all_seed_results for r in s["rewards"]]
#     agg_completions = [c for s in all_seed_results for c in s["completions"]]
#     agg_obsolete = [o for s in all_seed_results for o in s["obsolete"]]

#     agg = {
#         "n_seeds": len(all_seed_results),
#         "seeds": [s["seed"] for s in all_seed_results],
#         "episodes_per_seed": int(args.episodes),
#         "deterministic": bool(args.deterministic),
#         "rewards": agg_rewards,
#         "completions": agg_completions,
#         "obsolete": agg_obsolete,
#         "stats": {
#             "reward_mean": float(np.mean(agg_rewards)),
#             "reward_std": float(np.std(agg_rewards)),
#             "completion_mean": float(np.mean(agg_completions)),
#             "obsolete_mean": float(np.mean(agg_obsolete)),
#         },
#         "per_seed": all_seed_results,
#     }

#     agg_out = checkpoint_dir / "eval_results_all_seeds.json"
#     agg_out.write_text(json.dumps(agg, indent=2))
#     print(f"\n✓ Saved aggregated eval to: {agg_out}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Evaluate PPO checkpoints (deterministic + stochastic) and save JSON results.

This script does NOT generate plots.

Outputs per-seed:
  checkpoints_ppo/seed_<seed>/eval_results_deterministic.json
  checkpoints_ppo/seed_<seed>/eval_results_stochastic.json

Usage:
  python3 eval_ppo.py --config configs/training_config.yaml --checkpoint-dir checkpoints_ppo --episodes 100
  python3 eval_ppo.py --seeds 42 123 --episodes 200
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


# IMPORTANT: match whatever your wrapper writes into info["rew/<key>"].
# Your env currently returns debug keys like: pickups_this_step, deliveries_this_step, obsolete_this_step, step_penalty, sum_rewards...
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


def evaluate_seed_dir(
    seed: int,
    seed_dir: Path,
    root_dir: Path,
    config: Dict,
    episodes: int,
    model_name: str,
    skip_eval: bool,
) -> None:
    model_path = seed_dir / f"{model_name}.zip"
    if not model_path.exists():
        print(f"[WARN] Model not found: {model_path} – skipping seed {seed}")
        return

    det_json = seed_dir / "eval_results_deterministic.json"
    stoch_json = seed_dir / "eval_results_stochastic.json"

    if skip_eval and det_json.exists():
        print(f"  [seed {seed}] deterministic JSON exists, skipping: {det_json}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  [seed {seed}] Running deterministic evaluation ({episodes} episodes)")
        print(f"{'=' * 70}")
        det = run_evaluation(model_path, config, seed, episodes, deterministic=True)
        _save_json(det, det_json)

    if skip_eval and stoch_json.exists():
        print(f"  [seed {seed}] stochastic JSON exists, skipping: {stoch_json}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  [seed {seed}] Running stochastic evaluation ({episodes} episodes)")
        print(f"{'=' * 70}")
        st = run_evaluation(model_path, config, seed, episodes, deterministic=False)
        _save_json(st, stoch_json)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate PPO (det + stoch) and write JSON results (no plots).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
                    help="Root directory containing seed_*/ sub-directories.")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Seeds to process. If omitted, all seed_*/ dirs are used.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Number of evaluation episodes per seed per mode.")
    ap.add_argument("--model-name", type=str, default="ppo_final",
                    help="Model filename inside seed dir (without .zip).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation if per-seed JSON already exists.")
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

    print(f"Seeds to evaluate: {[s for s, _ in seed_dirs]}")
    for seed, seed_dir in seed_dirs:
        evaluate_seed_dir(
            seed=seed,
            seed_dir=seed_dir,
            root_dir=root_dir,
            config=config,
            episodes=args.episodes,
            model_name=args.model_name,
            skip_eval=args.skip_eval,
        )


if __name__ == "__main__":
    main()