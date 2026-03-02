# """
# Final working callback.
# """
# import numpy as np
# from stable_baselines3.common.callbacks import BaseCallback
# from pathlib import Path
# import json


# class FinalTaskAllocationCallback(BaseCallback):
#     """
#     Callback that reads episode stats from info dict.
#     """
    
#     def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = Path(save_path)
#         self.save_path.mkdir(parents=True, exist_ok=True)
        
#         self.episode_completions = []
#         self.episode_obsolete = []
#         self.episode_count = 0
    
#     def _on_step(self) -> bool:
#         # Check for episode end
#         infos = self.locals.get('infos', [])
        
#         for info in infos:
#             # 🔥 Read from info dict that wrapper provides
#             if 'episode_completed' in info:
#                 completed = info['episode_completed']
#                 obsolete = info['episode_obsolete']
                
#                 self.episode_completions.append(completed)
#                 self.episode_obsolete.append(obsolete)
#                 self.episode_count += 1
                
#                 # Log to tensorboard
#                 self.logger.record('task/completed', completed)
#                 self.logger.record('task/obsolete', obsolete)
#                 self.logger.record('task/completion_rate', 100 * completed / 15)  # Assuming 15 tasks
                
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
Final working callback.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json


class FinalTaskAllocationCallback(BaseCallback):
    """
    Callback that reads episode stats from info dict and logs additional reward components.
    """

    def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.episode_completions = []
        self.episode_obsolete = []
        self.episode_count = 0

        # Track running sums for per-episode reward component aggregation
        self._ep_rew_sums = {}  # key -> float

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])

        for info in infos:
            # ----------------------------
            # 1) Log reward-component info (per-step and aggregated per-episode)
            # ----------------------------
            for k, v in info.items():
                if not (isinstance(k, str) and k.startswith("rew/")):
                    continue

                # Only log scalar-like values
                if isinstance(v, (int, float, np.number)):
                    v = float(v)

                    # Per-step logging (shows as noisy curve)
                    self.logger.record(f"{k}_step", v)

                    # Per-episode aggregation
                    self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + v

            # ----------------------------
            # 2) Episode end stats (your existing logic)
            # ----------------------------
            if 'episode_completed' in info:
                completed = info.get('episode_completed', 0)
                obsolete = info.get('episode_obsolete', 0)

                self.episode_completions.append(completed)
                self.episode_obsolete.append(obsolete)
                self.episode_count += 1

                # Existing logs
                self.logger.record('task/completed', completed)
                self.logger.record('task/obsolete', obsolete)
                self.logger.record('task/completion_rate', 100 * completed / 15)  # assuming 15 tasks

                # NEW: log per-episode sums of reward components
                for k, s in self._ep_rew_sums.items():
                    self.logger.record(f"{k}_episode_sum", float(s))

                # Reset episode accumulators
                self._ep_rew_sums = {}

                if self.verbose > 0 and self.episode_count % 10 == 0:
                    recent = min(10, len(self.episode_completions))
                    print(f"\n[Episode {self.episode_count}] Completed: {completed}/15, Obsolete: {obsolete}")
                    print(f"  Last {recent} avg: {np.mean(self.episode_completions[-recent:]):.1f} completed")

        # Save periodically
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