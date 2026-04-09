# """
# Evaluate trained PPO model.

# Usage:
#     python eval_ppo.py --model checkpoints_ppo/ppo_final_masked --episodes 100
# """

# import argparse
# import numpy as np
# from stable_baselines3 import PPO

# # Import the SAME functions as training
# from train_ppo import make_env, load_config


# def evaluate(model, env, n_episodes=100, deterministic=True):
#     episode_rewards = []
#     episode_completions = []
#     episode_obsolete = []
#     episode_lengths = []

#     for ep in range(n_episodes):
#         obs, info = env.reset()
#         done = False
#         episode_reward = 0.0
#         step_count = 0

#         while not done:
#             action, _states = model.predict(obs, deterministic=deterministic)
#             obs, reward, terminated, truncated, info = env.step(action)
#             episode_reward += float(reward)
#             step_count += 1
#             done = terminated or truncated

#         completed = info.get('episode_completed', 0)
#         obsolete = info.get('episode_obsolete', 0)

#         episode_rewards.append(episode_reward)
#         episode_completions.append(completed)
#         episode_obsolete.append(obsolete)
#         episode_lengths.append(step_count)

#         if (ep + 1) % 10 == 0:
#             print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.1f}, "
#                   f"Completed={completed}/15, Steps={step_count}")

#     rewards_array = np.array(episode_rewards)
#     completions_array = np.array(episode_completions)
#     obsolete_array = np.array(episode_obsolete)

#     print("\n" + "="*70)
#     print("EVALUATION SUMMARY")
#     print("="*70)
#     print(f"Episodes:              {n_episodes}")
#     print(f"Mode:                  {'Deterministic' if deterministic else 'Stochastic'}")
#     print(f"\nRewards:")
#     print(f"  Mean:                {rewards_array.mean():.2f} ± {rewards_array.std():.2f}")
#     print(f"  Min/Max:             {rewards_array.min():.2f} / {rewards_array.max():.2f}")
#     print(f"\nTask Completion:")
#     print(f"  Mean:                {completions_array.mean():.2f}/15 "
#           f"({100*completions_array.mean()/15:.1f}%)")
#     print(f"  Best:                {completions_array.max()}/15")
#     print(f"  Worst:               {completions_array.min()}/15")
#     print(f"\nObsolete Tasks:")
#     print(f"  Mean:                {obsolete_array.mean():.2f}")
#     print(f"  Total:               {obsolete_array.sum():.0f}")
#     print(f"\nEpisode Length:")
#     print(f"  Mean:                {np.mean(episode_lengths):.1f} steps")
#     print("="*70)

#     return {
#         'rewards': episode_rewards,
#         'completions': episode_completions,
#         'obsolete': episode_obsolete,
#         'lengths': episode_lengths,
#         'total_tasks': 15
#     }


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, default='checkpoints_ppo/ppo_final_masked',
#                         help='Path to saved model (.zip or directory)')
#     parser.add_argument('--config', type=str, default='configs/training_config.yaml')
#     parser.add_argument('--episodes', type=int, default=100)
#     parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
#     args = parser.parse_args()

#     config = load_config(args.config)
#     env = make_env(config)

#     # ✅ IMPORTANT: load with PPO since you trained with PPO
#     model = PPO.load(args.model)

#     evaluate(model, env, n_episodes=args.episodes, deterministic=not args.stochastic)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.sb3_gnn_policy import RTGNNPolicy


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config, seed: int):
    env_config = config["environment"]
    agents = np.load(env_config["agents_file"], allow_pickle=True)

    batches = []
    for i in range(env_config["n_batches"]):
        batch_file = Path(env_config["data_dir"]) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file, allow_pickle=True))

    base_env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_config["radius"],
        feature_size=env_config["feature_size"],
        use_true_id=env_config["use_true_id"],
        all_batches=True,
    )

    env = WarehouseEnvSB3Final(
        base_env,
        assignment_interval=env_config["assignment_interval"],
        k_max=env_config.get("k_max", 5),
    )

    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_one_seed(model_path: Path, config, seed: int, n_episodes: int, deterministic: bool):
    set_seed(seed)

    env = make_env(config, seed=seed)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=vec_env, custom_objects={"policy_class": RTGNNPolicy})

    rewards = []
    completions = []
    obsolete = []
    lengths = []

    # component sums per episode (only if present in info)
    comp_keys = ["rew/pickup", "rew/delivery", "rew/obsolete", "rew/step_penalty"]
    comp_sums = {k: [] for k in comp_keys}

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        ep_comp = {k: 0.0 for k in comp_keys}
        ep_completed = 0
        ep_obsolete = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = vec_env.step(action)

            done = bool(dones[0])
            ep_reward += float(r[0])
            ep_len += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            # accumulate reward components if they exist
            if isinstance(info, dict):
                for k in comp_keys:
                    v = info.get(k, None)
                    if isinstance(v, (int, float, np.number)):
                        ep_comp[k] += float(v)

                # episode stats if wrapper provides them
                if "episode_completed" in info:
                    ep_completed = int(info.get("episode_completed", 0))
                if "episode_obsolete" in info:
                    ep_obsolete = int(info.get("episode_obsolete", 0))

        rewards.append(ep_reward)
        lengths.append(ep_len)
        completions.append(ep_completed)
        obsolete.append(ep_obsolete)
        for k in comp_keys:
            comp_sums[k].append(ep_comp[k])

        if (ep + 1) % 10 == 0:
            print(f"[seed {seed}] Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, len={ep_len}, completed={ep_completed}, obsolete={ep_obsolete}")

    out = {
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "deterministic": bool(deterministic),
        "rewards": rewards,
        "lengths": lengths,
        "completions": completions,
        "obsolete": obsolete,
        "reward_components": comp_sums,  # may be all zeros if not logged
        "stats": {
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "completion_mean": float(np.mean(completions)) if completions else 0.0,
            "obsolete_mean": float(np.mean(obsolete)) if obsolete else 0.0,
        },
    }
    vec_env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--model-name", type=str, default="ppo_final", help="model filename inside seed dir (without .zip)")
    args = ap.parse_args()

    config = load_config(args.config)

    checkpoint_dir = Path(args.checkpoint_dir)
    seed_dirs = sorted(checkpoint_dir.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs found in {checkpoint_dir}. Did you train multi-seed?")

    all_seed_results = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed_", ""))
        model_path = sd / f"{args.model_name}.zip"
        if not model_path.exists():
            print(f"[WARN] Missing model for {sd.name}: {model_path} (skipping)")
            continue

        print("\n" + "=" * 80)
        print(f"Evaluating seed {seed} model: {model_path}")
        print("=" * 80)

        res = evaluate_one_seed(
            model_path=model_path,
            config=config,
            seed=seed,
            n_episodes=args.episodes,
            deterministic=args.deterministic,
        )

        per_seed_out = sd / "eval_results.json"
        per_seed_out.write_text(json.dumps(res, indent=2))
        print(f"✓ Saved per-seed eval to: {per_seed_out}")

        all_seed_results.append(res)

    if not all_seed_results:
        raise RuntimeError("No seed results were evaluated (no models found).")

    # Aggregate
    agg_rewards = [r for s in all_seed_results for r in s["rewards"]]
    agg_completions = [c for s in all_seed_results for c in s["completions"]]
    agg_obsolete = [o for s in all_seed_results for o in s["obsolete"]]

    agg = {
        "n_seeds": len(all_seed_results),
        "seeds": [s["seed"] for s in all_seed_results],
        "episodes_per_seed": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "rewards": agg_rewards,
        "completions": agg_completions,
        "obsolete": agg_obsolete,
        "stats": {
            "reward_mean": float(np.mean(agg_rewards)),
            "reward_std": float(np.std(agg_rewards)),
            "completion_mean": float(np.mean(agg_completions)),
            "obsolete_mean": float(np.mean(agg_obsolete)),
        },
        "per_seed": all_seed_results,
    }

    agg_out = checkpoint_dir / "eval_results_all_seeds.json"
    agg_out.write_text(json.dumps(agg, indent=2))
    print(f"\n✓ Saved aggregated eval to: {agg_out}")


if __name__ == "__main__":
    main()