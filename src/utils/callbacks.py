# """
# Final working callback.
# """
# import numpy as np
# from stable_baselines3.common.callbacks import BaseCallback
# from pathlib import Path
# import json


# class FinalTaskAllocationCallback(BaseCallback):
#     """
#     Callback that reads episode stats from info dict and logs additional reward components.
#     """

#     def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = Path(save_path)
#         self.save_path.mkdir(parents=True, exist_ok=True)

#         self.episode_completions = []
#         self.episode_obsolete = []
#         self.episode_count = 0

#         # Track running sums for per-episode reward component aggregation
#         self._ep_rew_sums = {}  # key -> float

#     def _on_step(self) -> bool:
#         infos = self.locals.get('infos', [])

#         for info in infos:
#             # ----------------------------
#             # 1) Log reward-component info (per-step and aggregated per-episode)
#             # ----------------------------
#             for k, v in info.items():
#                 if not (isinstance(k, str) and k.startswith("rew/")):
#                     continue

#                 # Only log scalar-like values
#                 if isinstance(v, (int, float, np.number)):
#                     v = float(v)

#                     # Per-step logging (shows as noisy curve)
#                     self.logger.record(f"{k}_step", v)

#                     # Per-episode aggregation
#                     self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + v

#             # ----------------------------
#             # 2) Episode end stats (your existing logic)
#             # ----------------------------
#             if 'episode_completed' in info:
#                 completed = info.get('episode_completed', 0)
#                 obsolete = info.get('episode_obsolete', 0)

#                 self.episode_completions.append(completed)
#                 self.episode_obsolete.append(obsolete)
#                 self.episode_count += 1

#                 # Existing logs
#                 self.logger.record('task/completed', completed)
#                 self.logger.record('task/obsolete', obsolete)
#                 self.logger.record('task/completion_rate', 100 * completed / 15)  # assuming 15 tasks

#                 # NEW: log per-episode sums of reward components
#                 for k, s in self._ep_rew_sums.items():
#                     self.logger.record(f"{k}_episode_sum", float(s))

#                 # Reset episode accumulators
#                 self._ep_rew_sums = {}

#                 if self.verbose > 0 and self.episode_count % 10 == 0:
#                     recent = min(10, len(self.episode_completions))
#                     print(f"\n[Episode {self.episode_count}] Completed: {completed}/15, Obsolete: {obsolete}")
#                     print(f"  Last {recent} avg: {np.mean(self.episode_completions[-recent:]):.1f} completed")

#         # Save periodically
#         if self.n_calls % self.save_freq == 0 and len(self.episode_completions) > 0:
#             self._save_metrics()

#         return True

#     def _save_metrics(self):
#         metrics = {
#             'episode_completions': self.episode_completions,
#             'episode_obsolete': self.episode_obsolete,
#             'num_episodes': self.episode_count
#         }

#         with open(self.save_path / f'metrics_step_{self.n_calls}.json', 'w') as f:
#             json.dump(metrics, f, indent=2)

"""
Callback that logs episode stats + reward components + NOOP fraction.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json


class FinalTaskAllocationCallback(BaseCallback):
    def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.episode_completions = []
        self.episode_obsolete = []
        self.episode_count = 0

        self._ep_rew_sums = {}
        self._noop_steps = 0
        self._decision_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', None)  # VecEnv actions

        # Track NOOP usage if the env provides noop_index in info (optional)
        # We can infer NOOP fraction from action_mask shape and the fact that NOOP is last index.
        # if actions is not None and len(infos) > 0:
        #     info0 = infos[0]
        #     mask = info0.get("action_mask", None)
        #     # mask is [R, K+1] from wrapper
        #     if isinstance(mask, np.ndarray) and mask.ndim == 2:
        #         R, Kp1 = mask.shape
        #         noop_index = Kp1 - 1
        #         a0 = actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
        #         a0 = np.asarray(a0)
        #         if a0.shape == (R,):
        #             self._decision_steps += 1
        #             self._noop_steps += int(np.sum(a0 == noop_index))
        #             self.logger.record("policy/noop_fraction", float(self._noop_steps) / float(max(1, self._decision_steps * R)))
                # Track NOOP usage ONLY on decision steps (otherwise PPO emits actions that env ignores)
        if actions is not None and len(infos) > 0:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

            # Prefer meaningful_decision_step if available, else decision_step
            is_decision = bool(info0.get("meaningful_decision_step", info0.get("decision_step", False)))

            if is_decision:
                mask = info0.get("action_mask", None)
                if isinstance(mask, np.ndarray) and mask.ndim == 2:
                    R, Kp1 = mask.shape
                    noop_index = Kp1 - 1

                    # VecEnv actions should be shape (n_envs, R); take env0
                    a0 = np.asarray(actions)[0]
                    a0 = np.asarray(a0).reshape(-1)  # ensure shape (R,)

                    if a0.size == R:
                        self._decision_steps += 1
                        self._noop_steps += int(np.sum(a0 == noop_index))

                        noop_frac = float(self._noop_steps) / float(max(1, self._decision_steps * R))
                        self.logger.record("policy/noop_fraction", noop_frac)
        for info in infos:
            # log rew/*
            for k, v in info.items():
                if not (isinstance(k, str) and k.startswith("rew/")):
                    continue
                if isinstance(v, (int, float, np.number)):
                    v = float(v)
                    self.logger.record(f"{k}_step", v)
                    self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + v

            # episode end
            if 'episode_completed' in info:
                completed = info.get('episode_completed', 0)
                obsolete = info.get('episode_obsolete', 0)

                self.episode_completions.append(completed)
                self.episode_obsolete.append(obsolete)
                self.episode_count += 1

                self.logger.record('task/completed', completed)
                self.logger.record('task/obsolete', obsolete)
                self.logger.record('task/completion_rate', 100 * completed / 24)

                for k, s in self._ep_rew_sums.items():
                    self.logger.record(f"{k}_episode_sum", float(s))
                self._ep_rew_sums = {}

                if self.verbose > 0 and self.episode_count % 10 == 0:
                    recent = min(10, len(self.episode_completions))
                    print(f"\n[Episode {self.episode_count}] Completed: {completed}/24, Obsolete: {obsolete}")
                    print(f"  Last {recent} avg: {np.mean(self.episode_completions[-recent:]):.1f} completed")

        if self.n_calls % self.save_freq == 0 and len(self.episode_completions) > 0:
            self._save_metrics()

        return True

    def _save_metrics(self):
        metrics = {
            'episode_completions': self.episode_completions,
            'episode_obsolete': self.episode_obsolete,
            'num_episodes': self.episode_count
        }
        with open(self.save_path / f'metrics_step_{self.n_calls}.json', 'w') as f:
            json.dump(metrics, f, indent=2)