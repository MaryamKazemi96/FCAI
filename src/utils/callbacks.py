

"""
Callback that logs episode stats + reward components + policy behavior metrics:
- NOOP fraction on meaningful decision steps
- Assigned fraction on meaningful decision steps
- Collision-drop fraction on meaningful decision steps
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json


class FinalTaskAllocationCallback(BaseCallback):
    def __init__(self, save_freq=1000, save_path="./logs", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.episode_completions = []
        self.episode_obsolete = []
        self.episode_count = 0

        # per-episode reward sums
        self._ep_rew_sums = {}

        # --- policy metrics accumulators (meaningful decision steps only) ---
        self._meaningful_decision_steps = 0
        self._noop_count = 0
        self._action_count = 0
        self._assigned_count = 0
        self._drop_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)  # VecEnv actions

        # ---------- policy behavior metrics (use env0 info + env0 actions) ----------
        if actions is not None and isinstance(infos, (list, tuple)) and len(infos) > 0:
            info0 = infos[0] if isinstance(infos[0], dict) else {}
            is_meaningful = bool(info0.get("meaningful_decision_step", False))

            # Only compute these stats when decisions actually matter
            if is_meaningful:
                mask = info0.get("action_mask", None)
                # mask should be [R, K+1] and NOOP is last index
                if isinstance(mask, np.ndarray) and mask.ndim == 2:
                    R, Kp1 = mask.shape
                    noop_index = int(Kp1 - 1)

                    # actions usually shape (n_envs, R); take env0 and flatten
                    a = np.asarray(actions)
                    a0 = a[0] if a.ndim >= 2 else a
                    a0 = np.asarray(a0).astype(int).reshape(-1)

                    if a0.size == R:
                        resolved = info0.get("resolved_assignments", {}) or {}
                        if not isinstance(resolved, dict):
                            resolved = {}

                        assigned = int(len(resolved))  # after conflict resolution

                        # chosen non-NOOP actions
                        non_noop = int(np.sum(a0 != noop_index))
                        noop = int(np.sum(a0 == noop_index))

                        # robots that tried to assign but got nothing after conflict resolution
                        dropped = max(0, non_noop - assigned)

                        # accumulate
                        self._meaningful_decision_steps += 1
                        self._noop_count += noop
                        self._action_count += int(R)
                        self._assigned_count += assigned
                        self._drop_count += dropped

                        # log running fractions (over meaningful decision steps)
                        denom = float(max(1, self._action_count))
                        self.logger.record("policy/noop_fraction_meaningful", float(self._noop_count) / denom)
                        self.logger.record("policy/assigned_fraction_meaningful", float(self._assigned_count) / denom)
                        self.logger.record("policy/collision_drop_fraction_meaningful", float(self._drop_count) / denom)

        # ---------- reward component logging + episode stats ----------
        for info in infos:
            if not isinstance(info, dict):
                continue

            # log rew/*
            for k, v in info.items():
                if not (isinstance(k, str) and k.startswith("rew/")):
                    continue
                if isinstance(v, (int, float, np.number)):
                    v = float(v)
                    self.logger.record(f"{k}_step", v)
                    self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + v

            # episode end
            if "episode_completed" in info:
                completed = info.get("episode_completed", 0)
                obsolete = info.get("episode_obsolete", 0)

                self.episode_completions.append(completed)
                self.episode_obsolete.append(obsolete)
                self.episode_count += 1

                self.logger.record("task/completed", completed)
                self.logger.record("task/obsolete", obsolete)
                self.logger.record("task/completion_rate", 100 * completed / 24)

                for k, s in self._ep_rew_sums.items():
                    self.logger.record(f"{k}_episode_sum", float(s))
                self._ep_rew_sums = {}

                if self.verbose > 0 and self.episode_count % 10 == 0:
                    recent = min(10, len(self.episode_completions))
                    print(f"\n[Episode {self.episode_count}] Completed: {completed}/24, Obsolete: {obsolete}")
                    print(f"  Last {recent} avg: {np.mean(self.episode_completions[-recent:]):.1f} completed")

        # Save periodically
        if self.n_calls % self.save_freq == 0 and len(self.episode_completions) > 0:
            self._save_metrics()

        return True

    def _save_metrics(self):
        metrics = {
            "episode_completions": self.episode_completions,
            "episode_obsolete": self.episode_obsolete,
            "num_episodes": self.episode_count,
        }
        with open(self.save_path / f"metrics_step_{self.n_calls}.json", "w") as f:
            json.dump(metrics, f, indent=2)