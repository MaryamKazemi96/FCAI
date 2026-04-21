# #!/usr/bin/env python3
# """
# Evaluate baseline policies for the warehouse task allocation env.

# Fixes vs your current version:
# 1) Random policy no longer samples NOOP when there are valid task slots.
# 2) Unique policy reads cand_task_ids from either `info` OR `env.unwrapped._last_cand_task_ids`
#    (robust even if wrapper forgets to put cand_task_ids into info).
# 3) Unique policy resets "chosen" EACH decision step (per-step uniqueness, not whole-episode).
# 4) Optional debug prints to verify masks/candidates.

# Usage:
#   python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --output-dir checkpoints_ppo
#   python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --seed 456 --output-dir checkpoints_ppo
# """
# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, Any, Optional, Tuple, List

# import numpy as np

# # Import environment setup from training script
# from train_ppo import make_env, load_config


# POLICIES = ["random", "greedy", "unique"]


# # ---------------- helpers ----------------

# def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
#     if isinstance(obs, dict) and "action_mask" in obs:
#         return np.asarray(obs["action_mask"])
#     if isinstance(info, dict) and "action_mask" in info:
#         return np.asarray(info["action_mask"])
#     return None


# def _get_cand_ids(env, info: Any) -> Optional[List[List[Optional[int]]]]:
#     """
#     Try to get candidate task ids mapping slot->task_id for each robot.
#     Preferred: info["cand_task_ids"]
#     Fallback: env.unwrapped._last_cand_task_ids (colleague-style)
#     """
#     if isinstance(info, dict) and "cand_task_ids" in info:
#         return info["cand_task_ids"]

#     # robust fallback (works if wrapper stores it internally)
#     try:
#         env0 = env.unwrapped
#         cand = getattr(env0, "_last_cand_task_ids", None)
#         return cand
#     except Exception:
#         return None


# def _infer_decision_interval(env) -> int:
#     # Best effort: your wrapper uses assignment_interval
#     for attr in ("assignment_interval", "decision_dt", "decision_interval"):
#         if hasattr(env.unwrapped, attr):
#             try:
#                 v = int(getattr(env.unwrapped, attr))
#                 if v > 0:
#                     return v
#             except Exception:
#                 pass
#     return 1


# # ---------------- policies ----------------

# def random_policy(env, obs, info, rng: np.random.Generator) -> np.ndarray:
#     """
#     Random valid action, but:
#     - prefer real task slots (exclude NOOP) when tasks exist
#     - only choose NOOP if there are no valid task slots
#     """
#     action_space = env.action_space
#     assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

#     R = len(action_space.nvec)
#     mask = _get_mask(obs, info)

#     NOOP = int(action_space.nvec[0] - 1)
#     a = np.full((R,), NOOP, dtype=np.int64)

#     if mask is None:
#         # fallback: sample raw MultiDiscrete (includes NOOP)
#         return np.asarray(action_space.sample(), dtype=np.int64)

#     for r in range(R):
#         # ✅ exclude NOOP unless no valid task slots
#         allowed_tasks = np.flatnonzero(mask[r, :NOOP] > 0.5)
#         if allowed_tasks.size > 0:
#             a[r] = int(rng.choice(allowed_tasks))
#         else:
#             a[r] = NOOP
#     return a


# def greedy_policy(env, obs, info) -> np.ndarray:
#     """Pick the first valid task slot for each robot (else NOOP)."""
#     action_space = env.action_space
#     assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

#     R = len(action_space.nvec)
#     NOOP = int(action_space.nvec[0] - 1)

#     mask = _get_mask(obs, info)
#     if mask is None:
#         return np.full((R,), NOOP, dtype=np.int64)

#     a = np.full((R,), NOOP, dtype=np.int64)
#     for r in range(R):
#         valid = np.flatnonzero(mask[r, :NOOP] > 0.5)
#         a[r] = int(valid[0]) if valid.size > 0 else NOOP
#     return a


# def greedy_unique_policy(env, obs, info) -> np.ndarray:
#     """
#     Greedy unique: avoid assigning the SAME TASK ID to two robots
#     in the SAME decision step.

#     Requires:
#       - action_mask in obs or info
#       - candidate id mapping either in info["cand_task_ids"] or env.unwrapped._last_cand_task_ids
#     """
#     action_space = env.action_space
#     assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"

#     R = len(action_space.nvec)
#     NOOP = int(action_space.nvec[0] - 1)

#     mask = _get_mask(obs, info)
#     cand_ids = _get_cand_ids(env, info)

#     # fallback to greedy if missing any needed input
#     if mask is None or cand_ids is None:
#         return greedy_policy(env, obs, info)

#     chosen = set()
#     a = np.full((R,), NOOP, dtype=np.int64)

#     for r in range(R):
#         for k in range(NOOP):
#             if mask[r, k] <= 0.5:
#                 continue
#             try:
#                 task_id = cand_ids[r][k]
#             except Exception:
#                 task_id = None

#             if task_id is None:
#                 continue

#             # normalize id (avoid "10001.0" style issues)
#             try:
#                 tid_int = int(task_id)
#             except Exception:
#                 continue

#             if tid_int in chosen:
#                 continue

#             chosen.add(tid_int)
#             a[r] = int(k)
#             break

#     return a


# # ---------------- evaluation ----------------
# def evaluate_policy(
#     env,
#     policy_name: str,
#     n_episodes: int = 20,
#     seed: int = 42,
#     debug: bool = False,
# ):

#     rng = np.random.default_rng(seed)

#     ep_rewards = []
#     ep_completions = []
#     ep_obsolete = []
#     ep_lengths = []

#     decision_interval = _infer_decision_interval(env)

#     action_space = env.action_space
#     R = len(action_space.nvec)
#     NOOP = int(action_space.nvec[0] - 1)

#     for ep in range(n_episodes):
#         obs, info = env.reset(seed=seed + ep)

#         done = False
#         ep_rew = 0.0
#         ep_len = 0

#         # 🔑 macro-action memory
#         last_action = np.full((R,), NOOP, dtype=np.int64)

#         while not done:

#             #  DECISION STEP ONLY
#             if ep_len % decision_interval == 0:

#                 # EXACTLY like hers
#                 # if isinstance(info, dict) and "action_mask" in info:
#                 #     mask = info["action_mask"]
#                 # else:
#                 #     mask = env.unwrapped.action_mask()
#                 # mask = info.get("action_mask", env.unwrapped.action_mask())
#                 mask = None

#                 if isinstance(info, dict) and "action_mask" in info:
#                     mask = info["action_mask"]
#                 elif isinstance(obs, dict) and "action_mask" in obs:
#                     mask = obs["action_mask"]

#                 if mask is None:
#                     raise RuntimeError("No action_mask found in info or obs (env does not provide action_mask())")

#                 mask = np.asarray(mask)
#                 mask = (np.asarray(mask) == 1).astype(np.int32)

#                 # ---------- RANDOM ----------
#                 if policy_name == "random":
#                     a = np.full((R,), NOOP, dtype=np.int64)
#                     for r in range(R):
#                         allowed = np.flatnonzero(mask[r] == 1)
#                         if allowed.size > 0:
#                             a[r] = int(rng.choice(allowed))
#                         else:
#                             a[r] = NOOP
#                     last_action = a

#                 # ---------- GREEDY ----------
#                 elif policy_name == "greedy":
#                     a = np.full((R,), NOOP, dtype=np.int64)
#                     for r in range(R):
#                         if mask[r, 0] == 1:
#                             a[r] = 0
#                         else:
#                             a[r] = NOOP
#                     last_action = a

#                 # ---------- GREEDY UNIQUE ----------
#                 elif policy_name == "unique":
#                     env0 = env.unwrapped
#                     cand_ids = getattr(env0, "_last_cand_task_ids", None)

#                     if cand_ids is None:
#                         # fallback to greedy
#                         a = np.full((R,), NOOP, dtype=np.int64)
#                         for r in range(R):
#                             if mask[r, 0] == 1:
#                                 a[r] = 0
#                             else:
#                                 a[r] = NOOP
#                         last_action = a
#                     else:
#                         chosen = set()
#                         a = np.full((R,), NOOP, dtype=np.int64)

#                         for r in range(R):
#                             for k in range(NOOP):
#                                 if mask[r, k] != 1:
#                                     continue

#                                 task_id = int(cand_ids[r][k])

#                                 if task_id < 0:
#                                     continue

#                                 if task_id in chosen:
#                                     continue

#                                 chosen.add(task_id)
#                                 a[r] = k
#                                 break

#                         last_action = a

#                 else:
#                     raise ValueError(policy_name)

#                 if debug and ep == 0:
#                     print(f"[DEBUG] step={ep_len} action={last_action.tolist()}")

#             # 🔁 reuse action
#             action = last_action

#             obs, reward, terminated, truncated, info = env.step(action)

#             ep_rew += float(reward)
#             ep_len += 1
#             done = bool(terminated or truncated)

#         completed = int(info.get("episode_completed", 0)) if isinstance(info, dict) else 0
#         obsolete = int(info.get("episode_obsolete", 0)) if isinstance(info, dict) else 0

#         ep_rewards.append(ep_rew)
#         ep_completions.append(completed)
#         ep_obsolete.append(obsolete)
#         ep_lengths.append(ep_len)

#     rr = np.asarray(ep_rewards, dtype=float)
#     cc = np.asarray(ep_completions, dtype=float)
#     oo = np.asarray(ep_obsolete, dtype=float)
#     ll = np.asarray(ep_lengths, dtype=float)

#     return {
#         "policy": policy_name,
#         "rewards": [float(x) for x in ep_rewards],
#         "completions": [int(x) for x in ep_completions],
#         "obsolete": [int(x) for x in ep_obsolete],
#         "lengths": [int(x) for x in ep_lengths],
#         "stats": {
#             "reward_mean": float(rr.mean()) if rr.size else 0.0,
#             "reward_std": float(rr.std()) if rr.size else 0.0,
#             "completion_mean": float(cc.mean()) if cc.size else 0.0,
#             "completion_std": float(cc.std()) if cc.size else 0.0,
#             "obsolete_mean": float(oo.mean()) if oo.size else 0.0,
#             "obsolete_std": float(oo.std()) if oo.size else 0.0,
#             "length_mean": float(ll.mean()) if ll.size else 0.0,
#         },
#     }

# def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     out = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
#     for r in results_list:
#         for k in out.keys():
#             out[k].extend(r.get(k, []))

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
#     parser = argparse.ArgumentParser(description="Evaluate baseline policies (warehouse)")
#     parser.add_argument("--config", type=str, default="configs/training_config.yaml")
#     parser.add_argument("--episodes", type=int, default=20, help="Episodes per seed per policy")
#     parser.add_argument("--output-dir", type=str, default="checkpoints_ppo")
#     parser.add_argument("--seed", type=int, default=None, help="(optional) override: run only one seed")
#     parser.add_argument("--debug", action="store_true", help="Print debug info for the first episode")
#     args = parser.parse_args()

#     config = load_config(args.config)

#     seeds = config.get("experiment", {}).get("seeds", None)
#     if args.seed is not None:
#         seeds = [int(args.seed)]
#     elif not seeds:
#         seeds = [int(config["experiment"]["seed"])]
#     else:
#         seeds = [int(s) for s in seeds]

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     per_seed: Dict[str, Dict[str, Any]] = {}
#     per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}

#     for seed in seeds:
#         print("\n" + "=" * 70)
#         print(f"Seed {seed} | episodes per policy: {args.episodes}")
#         print("=" * 70)

#         per_seed[str(seed)] = {}

#         for policy_name in POLICIES:
#             env = make_env(config, seed=seed)

#             res = evaluate_policy(env, policy_name, n_episodes=args.episodes, seed=seed, debug=args.debug)
#             per_seed[str(seed)][policy_name] = res
#             per_policy_allseeds[policy_name].append(res)

#             try:
#                 env.close()
#             except Exception:
#                 pass

#             policy_file = output_dir / f"baseline_{policy_name}_seed_{seed}.json"
#             policy_file.write_text(json.dumps(res, indent=2))
#             print(f"✓ Saved {policy_file}")

#     combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
#     combined_results["num_episodes_per_seed"] = int(args.episodes)
#     combined_results["seeds"] = seeds

#     combined_file = output_dir / "baseline_results_all.json"
#     combined_file.write_text(json.dumps(combined_results, indent=2))
#     print(f"\n✓ Saved combined results to {combined_file}")

#     per_seed_file = output_dir / "baseline_results_per_seed.json"
#     per_seed_file.write_text(json.dumps(per_seed, indent=2))
#     print(f"✓ Saved per-seed results to {per_seed_file}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Warehouse baselines, colleague-style.

Policies (matches colleague logic):
- greedy: choose slot 0 if valid else NOOP
- random: choose uniformly among allowed actions in mask (includes NOOP if allowed)
- unique: choose smallest k that is valid and has unseen task_id; else NOOP; fallback to greedy if no cand_ids

Environment:
- MultiDiscrete([K+1]*R), NOOP index is K_max (last action)
- decision steps every assignment_interval (macro-action reused between decision steps)

Usage:
  python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --output-dir checkpoints_ppo
  python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --seed 42 --debug

Optional knobs to make greedy/unique weaker (while keeping same rule shape):
  --shuffle-robots        # randomize robot iteration order each decision step
  --unique-shuffle-k      # randomize the k scan order inside unique policy each decision step
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from train_ppo import make_env, load_config


POLICIES = ["random", "greedy", "unique"]


def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
    if isinstance(info, dict) and "action_mask" in info:
        return np.asarray(info["action_mask"])
    if isinstance(obs, dict) and "action_mask" in obs:
        return np.asarray(obs["action_mask"])
    return None


def _get_last_cand_task_ids(env) -> Optional[List[List[int]]]:
    try:
        env0 = env.unwrapped
        return getattr(env0, "_last_cand_task_ids", None)
    except Exception:
        return None


def _infer_decision_interval(env, config: Dict) -> int:
    # prefer wrapper attr, else config
    for e in [env, getattr(env, "unwrapped", None)]:
        if e is None:
            continue
        if hasattr(e, "assignment_interval"):
            try:
                v = int(getattr(e, "assignment_interval"))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(config.get("environment", {}).get("assignment_interval", 1))


def greedy_nearest_action(mask: np.ndarray, R: int, NOOP: int) -> np.ndarray:
    """Exactly like colleague: slot 0 if valid else NOOP."""
    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        if mask[r, 0] == 1:
            a[r] = 0
        else:
            a[r] = NOOP
    return a


def random_valid_action(mask: np.ndarray, R: int, NOOP: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly like colleague: choose uniformly among allowed actions (including NOOP if mask allows it)."""
    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        allowed = np.flatnonzero(mask[r] == 1)
        if allowed.size > 0:
            a[r] = int(rng.choice(allowed))
        else:
            a[r] = NOOP
    return a


def greedy_unique_action(
    mask: np.ndarray,
    env,
    R: int,
    K_max: int,
    NOOP: int,
    robot_order: np.ndarray,
    shuffle_k: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Exactly like colleague:
    - uses env.unwrapped._last_cand_task_ids
    - assigns first valid k whose task_id not yet chosen
    - fallback to greedy_nearest_action if cand_ids missing
    """
    cand_ids = _get_last_cand_task_ids(env)
    if cand_ids is None:
        return greedy_nearest_action(mask, R, NOOP)

    chosen = set()
    a = np.full((R,), NOOP, dtype=np.int64)

    for r in robot_order:
        if shuffle_k:
            ks = np.arange(K_max, dtype=int)
            rng.shuffle(ks)
        else:
            ks = range(K_max)

        for k in ks:
            if mask[r, k] != 1:
                continue
            try:
                task_id = int(cand_ids[r][k])
            except Exception:
                continue
            if task_id < 0:
                continue
            if task_id in chosen:
                continue
            chosen.add(task_id)
            a[r] = int(k)
            break

    return a


def evaluate_policy(
    env,
    config: Dict,
    policy_name: str,
    n_episodes: int,
    seed: int,
    debug: bool,
    shuffle_robots: bool,
    unique_shuffle_k: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    action_space = env.action_space
    assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"
    R = int(len(action_space.nvec))
    Kp1 = int(action_space.nvec[0])
    K_max = Kp1 - 1
    NOOP = K_max

    decision_interval = _infer_decision_interval(env, config)

    ep_rewards: List[float] = []
    ep_completions: List[int] = []
    ep_obsolete: List[int] = []
    ep_lengths: List[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_rew = 0.0
        ep_len = 0

        last_action = np.full((R,), NOOP, dtype=np.int64)

        while not done:
            if ep_len % max(1, decision_interval) == 0:
                mask = _get_mask(obs, info)
                if mask is None:
                    raise RuntimeError("No action_mask found in info or obs.")
                mask = (np.asarray(mask) == 1).astype(np.int32)

                # robot order (optional weakening knob)
                if shuffle_robots:
                    robot_order = np.arange(R, dtype=int)
                    rng.shuffle(robot_order)
                else:
                    robot_order = np.arange(R, dtype=int)

                if policy_name == "greedy":
                    # colleague logic is always r=0..R-1; robot_order is only for unique.
                    last_action = greedy_nearest_action(mask, R, NOOP)

                elif policy_name == "random":
                    last_action = random_valid_action(mask, R, NOOP, rng)

                elif policy_name == "unique":
                    last_action = greedy_unique_action(
                        mask=mask,
                        env=env,
                        R=R,
                        K_max=K_max,
                        NOOP=NOOP,
                        robot_order=robot_order,
                        shuffle_k=unique_shuffle_k,
                        rng=rng,
                    )
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                if debug and ep == 0:
                    valid_slots = mask[:, :NOOP].sum(axis=1).astype(int).tolist()
                    print("[DEBUG] valid_slots_per_robot(excl NOOP):", valid_slots)
                    print("[DEBUG] cand_present:", _get_last_cand_task_ids(env) is not None)
                    print("[DEBUG] action:", last_action.tolist())

            obs, reward, terminated, truncated, info = env.step(last_action)
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
    return {
        "policy": policy_name,
        "decision_interval": int(decision_interval),
        "rewards": [float(x) for x in ep_rewards],
        "completions": [int(x) for x in ep_completions],
        "obsolete": [int(x) for x in ep_obsolete],
        "lengths": [int(x) for x in ep_lengths],
        "stats": {
            "reward_mean": float(rr.mean()) if rr.size else 0.0,
            "reward_std": float(rr.std()) if rr.size else 0.0,
            "completion_mean": float(np.mean(ep_completions)) if ep_completions else 0.0,
            "obsolete_mean": float(np.mean(ep_obsolete)) if ep_obsolete else 0.0,
            "length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        },
    }


def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
    for r in results_list:
        for k in out.keys():
            out[k].extend(r.get(k, []))

    rr = np.asarray(out["rewards"], dtype=float)
    out["stats"] = {
        "reward_mean": float(rr.mean()) if rr.size else 0.0,
        "reward_std": float(rr.std()) if rr.size else 0.0,
        "completion_mean": float(np.mean(out["completions"])) if out["completions"] else 0.0,
        "obsolete_mean": float(np.mean(out["obsolete"])) if out["obsolete"] else 0.0,
        "length_mean": float(np.mean(out["lengths"])) if out["lengths"] else 0.0,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate baseline policies (warehouse)")
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--output-dir", type=str, default="checkpoints_ppo")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", action="store_true")

    # weakening knobs (optional)
    ap.add_argument("--shuffle-robots", action="store_true",
                    help="Shuffle robot iteration order on each decision step (makes unique less deterministic).")
    ap.add_argument("--unique-shuffle-k", action="store_true",
                    help="Shuffle k scan order inside unique baseline on each decision step (makes unique weaker).")

    args = ap.parse_args()

    config = load_config(args.config)

    seeds = config.get("experiment", {}).get("seeds", None)
    if args.seed is not None:
        seeds = [int(args.seed)]
    elif not seeds:
        seeds = [int(config["experiment"]["seed"])]
    else:
        seeds = [int(s) for s in seeds]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: Dict[str, Dict[str, Any]] = {}
    per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Seed {seed} | episodes per policy: {args.episodes}")
        print("=" * 70)

        per_seed[str(seed)] = {}

        for policy_name in POLICIES:
            env = make_env(config, seed=seed)
            res = evaluate_policy(
                env=env,
                config=config,
                policy_name=policy_name,
                n_episodes=args.episodes,
                seed=seed,
                debug=args.debug,
                shuffle_robots=args.shuffle_robots,
                unique_shuffle_k=args.unique_shuffle_k,
            )
            per_seed[str(seed)][policy_name] = res
            per_policy_allseeds[policy_name].append(res)

            try:
                env.close()
            except Exception:
                pass

            p = out_dir / f"baseline_{policy_name}_seed_{seed}.json"
            p.write_text(json.dumps(res, indent=2))
            print(f"✓ Saved {p}")

    combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
    combined_results["num_episodes_per_seed"] = int(args.episodes)
    combined_results["seeds"] = seeds
    (out_dir / "baseline_results_all.json").write_text(json.dumps(combined_results, indent=2))
    (out_dir / "baseline_results_per_seed.json").write_text(json.dumps(per_seed, indent=2))

    print(f"\n✓ Saved combined results to {out_dir / 'baseline_results_all.json'}")


if __name__ == "__main__":
    main()