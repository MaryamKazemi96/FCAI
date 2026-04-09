# # #!/usr/bin/env python3
# # """
# # Evaluate baseline policies for warehouse task allocation.

# # Usage:
# #     python eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 100
# # """

# # import argparse
# # import yaml
# import numpy as np
# # import json
# # from pathlib import Path
# # from typing import Dict, List

# # # Import environment setup from training script
# from train_ppo import make_env, load_config



# def random_policy(env, obs, info, rng: np.random.Generator):
#     action_space = env.action_space
#     if not hasattr(action_space, "nvec"):
#         return int(action_space.sample())

#     R = len(action_space.nvec)
#     mask, _ = _get_mask_and_cands(obs, info)

#     if mask is None:
#         return action_space.sample()

#     a = np.zeros((R,), dtype=np.int64)
#     for r in range(R):
#         allowed = np.flatnonzero(mask[r] > 0.5)
#         if allowed.size == 0:
#             a[r] = int(action_space.nvec[r] - 1)  # NOOP
#         else:
#             a[r] = int(rng.choice(allowed))
#     return a


# def greedy_nearest_policy(env, obs, info):
#     action_space = env.action_space
#     if not hasattr(action_space, "nvec"):
#         return 0

#     R = len(action_space.nvec)
#     Kp1 = int(action_space.nvec[0])
#     NOOP = Kp1 - 1

#     mask, _ = _get_mask_and_cands(obs, info)
#     if mask is None:
#         return np.full((R,), NOOP, dtype=np.int64)

#     a = np.full((R,), NOOP, dtype=np.int64)
#     for r in range(R):
#         valid_slots = np.flatnonzero(mask[r, :NOOP] > 0.5)
#         a[r] = int(valid_slots[0]) if valid_slots.size > 0 else NOOP
#     return a

# def greedy_unique_policy(env, obs, info, chosen_tasks_this_step=None):
#     """
#     Greedy unique baseline (colleague-style):
#     avoid assigning the SAME TASK ID to two robots in the same step.
#     Requires info['cand_task_ids'] from wrapper: list[R][K] containing task_id or None.
#     """
#     action_space = env.action_space
#     if not hasattr(action_space, "nvec"):
#         return 0  # fallback for old Discrete case

#     R = len(action_space.nvec)
#     Kp1 = int(action_space.nvec[0])
#     NOOP = Kp1 - 1

#     # mask from obs (preferred)
#     mask = None
#     if isinstance(obs, dict) and "action_mask" in obs:
#         mask = np.asarray(obs["action_mask"])
#     elif isinstance(info, dict) and "action_mask" in info:
#         mask = np.asarray(info["action_mask"])

#     cand_ids = info.get("cand_task_ids", None)

#     # fallback to greedy if we can't do true unique
#     if mask is None or cand_ids is None:
#         return greedy_nearest_policy(env, obs, info)

#     chosen = set() if chosen_tasks_this_step is None else chosen_tasks_this_step
#     a = np.full((R,), NOOP, dtype=np.int64)

#     for r in range(R):
#         # scan candidate slots in order (greedy)
#         for k in range(NOOP):
#             if mask[r, k] <= 0.5:
#                 continue
#             task_id = cand_ids[r][k]
#             if task_id is None:
#                 continue
#             if task_id in chosen:
#                 continue
#             chosen.add(task_id)
#             a[r] = int(k)
#             break

#     return a

# #!/usr/bin/env python3
# import argparse
# import json
# from pathlib import Path
# from typing import Dict, Tuple, Any

# import numpy as np

# # assumes you already have these in your file:
# # - load_config
# # - make_env
# # and the policies below

# POLICIES = ["random", "greedy", "unique"]


# def _get_mask_and_cands(obs, info):
#     mask = None
#     if isinstance(obs, dict) and "action_mask" in obs:
#         mask = np.asarray(obs["action_mask"])
#     elif isinstance(info, dict) and "action_mask" in info:
#         mask = np.asarray(info["action_mask"])

#     cand_ids = None
#     if isinstance(info, dict) and "cand_task_ids" in info:
#         cand_ids = info["cand_task_ids"]  # list[R][K] with task_id or None

#     return mask, cand_ids


# def evaluate_policy(env, policy_name: str, n_episodes: int = 100, seed: int = 42) -> Dict[str, Any]:
#     rng = np.random.default_rng(seed)

#     episode_rewards = []
#     episode_completions = []
#     episode_obsolete = []
#     episode_lengths = []

#     # for unique: share set across decision steps inside an episode
#     chosen_tasks_history = set()

#     for ep in range(n_episodes):
#         obs, info = env.reset()
#         done = False
#         ep_rew = 0.0
#         ep_len = 0

#         if policy_name == "unique":
#             chosen_tasks_history = set()

#         while not done:
#             if policy_name == "random":
#                 action = random_policy(env, obs, info, rng)
#             elif policy_name == "greedy":
#                 action = greedy_nearest_policy(env, obs, info)
#             elif policy_name == "unique":
#                 action = greedy_unique_policy(env, obs, info, chosen_tasks_history)
#             else:
#                 raise ValueError(policy_name)

#             # MultiDiscrete action
#             if hasattr(env.action_space, "nvec"):
#                 action = np.asarray(action, dtype=np.int64)
#             else:
#                 action = int(action)

#             obs, reward, terminated, truncated, info = env.step(action)
#             ep_rew += float(reward)
#             ep_len += 1
#             done = bool(terminated or truncated)

#         completed = int(info.get("episode_completed", 0)) if isinstance(info, dict) else 0
#         obsolete = int(info.get("episode_obsolete", 0)) if isinstance(info, dict) else 0

#         episode_rewards.append(ep_rew)
#         episode_completions.append(completed)
#         episode_obsolete.append(obsolete)
#         episode_lengths.append(ep_len)

#     rewards_array = np.asarray(episode_rewards, dtype=float)
#     completions_array = np.asarray(episode_completions, dtype=float)
#     obsolete_array = np.asarray(episode_obsolete, dtype=float)
#     lengths_array = np.asarray(episode_lengths, dtype=float)

#     return {
#         "policy": policy_name,
#         "rewards": [float(r) for r in episode_rewards],
#         "completions": [int(c) for c in episode_completions],
#         "obsolete": [int(o) for o in episode_obsolete],
#         "lengths": [int(l) for l in episode_lengths],
#         "stats": {
#             "reward_mean": float(rewards_array.mean()) if rewards_array.size else 0.0,
#             "reward_std": float(rewards_array.std()) if rewards_array.size else 0.0,
#             "completion_mean": float(completions_array.mean()) if completions_array.size else 0.0,
#             "completion_std": float(completions_array.std()) if completions_array.size else 0.0,
#             "obsolete_mean": float(obsolete_array.mean()) if obsolete_array.size else 0.0,
#             "obsolete_std": float(obsolete_array.std()) if obsolete_array.size else 0.0,
#             "length_mean": float(lengths_array.mean()) if lengths_array.size else 0.0,
#         },
#     }


# def _concat_results(results_list):
#     out = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
#     for r in results_list:
#         for k in out.keys():
#             out[k].extend(r.get(k, []))
#     # stats
#     rr = np.asarray(out["rewards"], dtype=float)
#     cc = np.asarray(out["completions"], dtype=float)
#     oo = np.asarray(out["obsolete"], dtype=float)
#     ll = np.asarray(out["lengths"], dtype=float)
#     out["stats"] = {
#         "reward_mean": float(rr.mean()) if rr.size else 0.0,
#         "reward_std": float(rr.std()) if rr.size else 0.0,
#         "completion_mean": float(cc.mean()) if cc.size else 0.0,
#         "completion_std": float(cc.std()) if cc.size else 0.0,
#         "obsolete_mean": float(oo.mean()) if oo.size else 0.0,
#         "obsolete_std": float(oo.std()) if oo.size else 0.0,
#         "length_mean": float(ll.mean()) if ll.size else 0.0,
#     }
#     return out


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate baseline policies (multi-seed)")
#     parser.add_argument("--config", type=str, default="configs/training_config.yaml")
#     parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed per policy")
#     parser.add_argument("--output-dir", type=str, default="checkpoints_ppo")
#     parser.add_argument("--seed", type=int, default=None, help="(optional) override: run only one seed")
#     args = parser.parse_args()

#     # You said you already have load_config/make_env in this file; keep using them:
#     config = load_config(args.config)

#     # seeds: from config unless overridden
#     seeds = config.get("experiment", {}).get("seeds", None)
#     if args.seed is not None:
#         seeds = [int(args.seed)]
#     elif not seeds:
#         seeds = [int(config["experiment"]["seed"])]
#     else:
#         seeds = [int(s) for s in seeds]

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     per_seed = {}  # per_seed[seed][policy] = result
#     per_policy_allseeds = {p: [] for p in POLICIES}

#     for seed in seeds:
#         print("\n" + "=" * 70)
#         print(f"Seed {seed} | episodes per policy: {args.episodes}")
#         print("=" * 70)

#         per_seed[str(seed)] = {}

#         for policy_name in POLICIES:
#             env = make_env(config, seed=seed)

#             res = evaluate_policy(env, policy_name, n_episodes=args.episodes, seed=seed)
#             per_seed[str(seed)][policy_name] = res
#             per_policy_allseeds[policy_name].append(res)

#             env.close()

#             policy_file = output_dir / f"baseline_{policy_name}_seed_{seed}.json"
#             policy_file.write_text(json.dumps(res, indent=2))
#             print(f"✓ Saved {policy_file}")

#     # Aggregated (this is what plot_training.py expects at top-level)
#     combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
#     combined_results["num_episodes_per_seed"] = int(args.episodes)
#     combined_results["seeds"] = seeds

#     combined_file = output_dir / "baseline_results_all.json"
#     combined_file.write_text(json.dumps(combined_results, indent=2))
#     print(f"\n✓ Saved combined results to {combined_file}")

#     # Extra: keep a per-seed file for debugging
#     per_seed_file = output_dir / "baseline_results_per_seed.json"
#     per_seed_file.write_text(json.dumps(per_seed, indent=2))
#     print(f"✓ Saved per-seed results to {per_seed_file}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Evaluate baseline policies for the warehouse task allocation env.

Fixes vs your current version:
1) Random policy no longer samples NOOP when there are valid task slots.
2) Unique policy reads cand_task_ids from either `info` OR `env.unwrapped._last_cand_task_ids`
   (robust even if wrapper forgets to put cand_task_ids into info).
3) Unique policy resets "chosen" EACH decision step (per-step uniqueness, not whole-episode).
4) Optional debug prints to verify masks/candidates.

Usage:
  python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --output-dir checkpoints_ppo
  python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --seed 456 --output-dir checkpoints_ppo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# Import environment setup from training script
from train_ppo import make_env, load_config


POLICIES = ["random", "greedy", "unique"]


# ---------------- helpers ----------------

def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
    if isinstance(obs, dict) and "action_mask" in obs:
        return np.asarray(obs["action_mask"])
    if isinstance(info, dict) and "action_mask" in info:
        return np.asarray(info["action_mask"])
    return None


def _get_cand_ids(env, info: Any) -> Optional[List[List[Optional[int]]]]:
    """
    Try to get candidate task ids mapping slot->task_id for each robot.
    Preferred: info["cand_task_ids"]
    Fallback: env.unwrapped._last_cand_task_ids (colleague-style)
    """
    if isinstance(info, dict) and "cand_task_ids" in info:
        return info["cand_task_ids"]

    # robust fallback (works if wrapper stores it internally)
    try:
        env0 = env.unwrapped
        cand = getattr(env0, "_last_cand_task_ids", None)
        return cand
    except Exception:
        return None


def _infer_decision_interval(env) -> int:
    # Best effort: your wrapper uses assignment_interval
    for attr in ("assignment_interval", "decision_dt", "decision_interval"):
        if hasattr(env.unwrapped, attr):
            try:
                v = int(getattr(env.unwrapped, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    return 1


# ---------------- policies ----------------

def random_policy(env, obs, info, rng: np.random.Generator) -> np.ndarray:
    """
    Random valid action, but:
    - prefer real task slots (exclude NOOP) when tasks exist
    - only choose NOOP if there are no valid task slots
    """
    action_space = env.action_space
    assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

    R = len(action_space.nvec)
    mask = _get_mask(obs, info)

    NOOP = int(action_space.nvec[0] - 1)
    a = np.full((R,), NOOP, dtype=np.int64)

    if mask is None:
        # fallback: sample raw MultiDiscrete (includes NOOP)
        return np.asarray(action_space.sample(), dtype=np.int64)

    for r in range(R):
        # ✅ exclude NOOP unless no valid task slots
        allowed_tasks = np.flatnonzero(mask[r, :NOOP] > 0.5)
        if allowed_tasks.size > 0:
            a[r] = int(rng.choice(allowed_tasks))
        else:
            a[r] = NOOP
    return a


def greedy_policy(env, obs, info) -> np.ndarray:
    """Pick the first valid task slot for each robot (else NOOP)."""
    action_space = env.action_space
    assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

    R = len(action_space.nvec)
    NOOP = int(action_space.nvec[0] - 1)

    mask = _get_mask(obs, info)
    if mask is None:
        return np.full((R,), NOOP, dtype=np.int64)

    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        valid = np.flatnonzero(mask[r, :NOOP] > 0.5)
        a[r] = int(valid[0]) if valid.size > 0 else NOOP
    return a


def greedy_unique_policy(env, obs, info) -> np.ndarray:
    """
    Greedy unique: avoid assigning the SAME TASK ID to two robots
    in the SAME decision step.

    Requires:
      - action_mask in obs or info
      - candidate id mapping either in info["cand_task_ids"] or env.unwrapped._last_cand_task_ids
    """
    action_space = env.action_space
    assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

    R = len(action_space.nvec)
    NOOP = int(action_space.nvec[0] - 1)

    mask = _get_mask(obs, info)
    cand_ids = _get_cand_ids(env, info)

    # fallback to greedy if missing any needed input
    if mask is None or cand_ids is None:
        return greedy_policy(env, obs, info)

    chosen = set()
    a = np.full((R,), NOOP, dtype=np.int64)

    for r in range(R):
        for k in range(NOOP):
            if mask[r, k] <= 0.5:
                continue
            try:
                task_id = cand_ids[r][k]
            except Exception:
                task_id = None

            if task_id is None:
                continue

            # normalize id (avoid "10001.0" style issues)
            try:
                tid_int = int(task_id)
            except Exception:
                continue

            if tid_int in chosen:
                continue

            chosen.add(tid_int)
            a[r] = int(k)
            break

    return a


# ---------------- evaluation ----------------

def evaluate_policy(
    env,
    policy_name: str,
    n_episodes: int = 20,
    seed: int = 42,
    debug: bool = False,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    ep_rewards: List[float] = []
    ep_completions: List[int] = []
    ep_obsolete: List[int] = []
    ep_lengths: List[int] = []

    decision_interval = _infer_decision_interval(env)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)  # small variation per episode for realism; remove if undesired
        done = False
        ep_rew = 0.0
        ep_len = 0

        while not done:
            if policy_name == "random":
                action = random_policy(env, obs, info, rng)
            elif policy_name == "greedy":
                action = greedy_policy(env, obs, info)
            elif policy_name == "unique":
                action = greedy_unique_policy(env, obs, info)
            else:
                raise ValueError(policy_name)

            action = np.asarray(action, dtype=np.int64)

            if debug and (ep == 0) and (ep_len % max(1, decision_interval) == 0):
                mask = _get_mask(obs, info)
                cand = _get_cand_ids(env, info)
                if mask is not None:
                    print("[DEBUG] valid_slots_per_robot(excl NOOP):", mask[:, :-1].sum(axis=1).astype(int).tolist())
                print("[DEBUG] cand_present:", cand is not None)
                print("[DEBUG] action:", action.tolist())

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

        completed = int(info.get("episode_completed", 0)) if isinstance(info, dict) else 0
        obsolete = int(info.get("episode_obsolete", 0)) if isinstance(info, dict) else 0

        ep_rewards.append(ep_rew)
        ep_completions.append(completed)
        ep_obsolete.append(obsolete)
        ep_lengths.append(ep_len)

    rr = np.asarray(ep_rewards, dtype=float)
    cc = np.asarray(ep_completions, dtype=float)
    oo = np.asarray(ep_obsolete, dtype=float)
    ll = np.asarray(ep_lengths, dtype=float)

    return {
        "policy": policy_name,
        "rewards": [float(x) for x in ep_rewards],
        "completions": [int(x) for x in ep_completions],
        "obsolete": [int(x) for x in ep_obsolete],
        "lengths": [int(x) for x in ep_lengths],
        "stats": {
            "reward_mean": float(rr.mean()) if rr.size else 0.0,
            "reward_std": float(rr.std()) if rr.size else 0.0,
            "completion_mean": float(cc.mean()) if cc.size else 0.0,
            "completion_std": float(cc.std()) if cc.size else 0.0,
            "obsolete_mean": float(oo.mean()) if oo.size else 0.0,
            "obsolete_std": float(oo.std()) if oo.size else 0.0,
            "length_mean": float(ll.mean()) if ll.size else 0.0,
        },
    }


def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
    for r in results_list:
        for k in out.keys():
            out[k].extend(r.get(k, []))

    rr = np.asarray(out["rewards"], dtype=float)
    cc = np.asarray(out["completions"], dtype=float)
    oo = np.asarray(out["obsolete"], dtype=float)
    ll = np.asarray(out["lengths"], dtype=float)

    out["stats"] = {
        "reward_mean": float(rr.mean()) if rr.size else 0.0,
        "reward_std": float(rr.std()) if rr.size else 0.0,
        "completion_mean": float(cc.mean()) if cc.size else 0.0,
        "completion_std": float(cc.std()) if cc.size else 0.0,
        "obsolete_mean": float(oo.mean()) if oo.size else 0.0,
        "obsolete_std": float(oo.std()) if oo.size else 0.0,
        "length_mean": float(ll.mean()) if ll.size else 0.0,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline policies (warehouse)")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed per policy")
    parser.add_argument("--output-dir", type=str, default="checkpoints_ppo")
    parser.add_argument("--seed", type=int, default=None, help="(optional) override: run only one seed")
    parser.add_argument("--debug", action="store_true", help="Print debug info for the first episode")
    args = parser.parse_args()

    config = load_config(args.config)

    seeds = config.get("experiment", {}).get("seeds", None)
    if args.seed is not None:
        seeds = [int(args.seed)]
    elif not seeds:
        seeds = [int(config["experiment"]["seed"])]
    else:
        seeds = [int(s) for s in seeds]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: Dict[str, Dict[str, Any]] = {}
    per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Seed {seed} | episodes per policy: {args.episodes}")
        print("=" * 70)

        per_seed[str(seed)] = {}

        for policy_name in POLICIES:
            env = make_env(config, seed=seed)

            res = evaluate_policy(env, policy_name, n_episodes=args.episodes, seed=seed, debug=args.debug)
            per_seed[str(seed)][policy_name] = res
            per_policy_allseeds[policy_name].append(res)

            try:
                env.close()
            except Exception:
                pass

            policy_file = output_dir / f"baseline_{policy_name}_seed_{seed}.json"
            policy_file.write_text(json.dumps(res, indent=2))
            print(f"✓ Saved {policy_file}")

    combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
    combined_results["num_episodes_per_seed"] = int(args.episodes)
    combined_results["seeds"] = seeds

    combined_file = output_dir / "baseline_results_all.json"
    combined_file.write_text(json.dumps(combined_results, indent=2))
    print(f"\n✓ Saved combined results to {combined_file}")

    per_seed_file = output_dir / "baseline_results_per_seed.json"
    per_seed_file.write_text(json.dumps(per_seed, indent=2))
    print(f"✓ Saved per-seed results to {per_seed_file}")


if __name__ == "__main__":
    main()