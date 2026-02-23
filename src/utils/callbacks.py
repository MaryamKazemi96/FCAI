"""
Final working callback.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json


class FinalTaskAllocationCallback(BaseCallback):
    """
    Callback that reads episode stats from info dict.
    """
    
    def __init__(self, save_freq=1000, save_path='./logs', verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.episode_completions = []
        self.episode_obsolete = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check for episode end
        infos = self.locals.get('infos', [])
        
        for info in infos:
            # 🔥 Read from info dict that wrapper provides
            if 'episode_completed' in info:
                completed = info['episode_completed']
                obsolete = info['episode_obsolete']
                
                self.episode_completions.append(completed)
                self.episode_obsolete.append(obsolete)
                self.episode_count += 1
                
                # Log to tensorboard
                self.logger.record('task/completed', completed)
                self.logger.record('task/obsolete', obsolete)
                self.logger.record('task/completion_rate', 100 * completed / 15)  # Assuming 15 tasks
                
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