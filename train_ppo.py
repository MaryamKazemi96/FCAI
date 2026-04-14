import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.models.sb3_gnn_policy import RTGNNPolicy
from stable_baselines3 import PPO

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.gnn_policy_simple import SimpleGNNActorCriticPolicy
from src.utils.callbacks import FinalTaskAllocationCallback


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config, seed: int):
    env_config = config['environment']
    agents = np.load(env_config['agents_file'], allow_pickle=True)

    batches = []
    for i in range(env_config['n_batches']):
        batch_file = Path(env_config['data_dir']) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file, allow_pickle=True))

    base_env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_config['radius'],
        feature_size=env_config['feature_size'],
        use_true_id=env_config['use_true_id'],
        all_batches=True
    )

    env = WarehouseEnvSB3Final(base_env, assignment_interval=env_config['assignment_interval'], k_max=env_config.get("k_max", 5))
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    seeds = config.get("experiment", {}).get("seeds")
    if not seeds:
        seeds = [int(config["experiment"]["seed"])]

    checkpoint_root = Path(config['training']['checkpoint_dir'])
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed = int(seed)
        print("\n" + "="*80)
        print(f"TRAINING SEED {seed}")
        print("="*80)

        set_seed(seed)

        env = make_env(config, seed)
        env = DummyVecEnv([lambda: env])

        seed_dir = checkpoint_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        callback = FinalTaskAllocationCallback(
            save_freq=config['training']['save_freq'],
            save_path=seed_dir / 'metrics'
        )

        model = PPO(
            policy=RTGNNPolicy,
            env=env,
            policy_kwargs=dict(
                in_dim=config['environment']['feature_size'],
                hidden_dim=config['model']['hidden_dim'],
                k_max=config['environment'].get("k_max", 5),
                gnn_type=config['model']['gnn_type'],
                num_gnn_layers=config['model']['num_gnn_layers'],
                activation=config['model']['activation'],
                dropout=config['model']['dropout'],
                noop_init=-1.0,
                noop_frozen=True,
                logit_temperature=5.0,
            ),
             
            learning_rate=config['ppo']['learning_rate'],
            n_steps=config['ppo']['n_steps'],
            batch_size=config['ppo']['batch_size'],
            n_epochs=config['ppo']['n_epochs'],
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_range=config['ppo']['clip_range'],
            ent_coef=config['ppo']['ent_coef'],
            vf_coef=config['ppo']['vf_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            tensorboard_log=str(seed_dir / "tensorboard"),
            verbose=1,
            device=config['experiment']['device'],
            seed=seed,
        )

        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback,
            log_interval=config['training']['log_interval']
        )

        model.save(seed_dir / "ppo_final")
        print(f"✓ Saved: {seed_dir / 'ppo_final'}.zip")

if __name__ == "__main__":
    main()
