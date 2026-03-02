# #!/usr/bin/env python3
# import argparse
# import yaml
# import numpy as np
# import torch
# from pathlib import Path
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

# from src.environment.environment import MultiTaskAllocationEnv
# from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
# from src.models.gnn_policy_simple import SimpleGNNActorCriticPolicy
# from src.utils.callbacks import FinalTaskAllocationCallback


# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def make_env(config):
#     env_config = config['environment']
    
#     agents = np.load(env_config['agents_file'], allow_pickle=True)
    
#     batches = []
#     for i in range(env_config['n_batches']):
#         batch_file = Path(env_config['data_dir']) / f"tasks_batch_{i}.npy"
#         if batch_file.exists():
#             batch = np.load(batch_file, allow_pickle=True)
#             batches.append(batch)
    
#     base_env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=batches,
#         radius=env_config['radius'],
#         feature_size=env_config['feature_size'],
#         use_true_id=env_config['use_true_id'],
#         all_batches=True
#     )
    
#     env = WarehouseEnvSB3Final(base_env, assignment_interval=env_config['assignment_interval'])
#     env = Monitor(env)
    
#     return env


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='configs/training_config.yaml')
#     args = parser.parse_args()
    
#     config = load_config(args.config)
    
#     seed = config['experiment']['seed']
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     env = make_env(config)
#     env = DummyVecEnv([lambda: env])
    
#     print(f"\nEnvironment ready for training!")
    
#     model = PPO(
#         policy=SimpleGNNActorCriticPolicy,
#         env=env,
#         learning_rate=config['ppo']['learning_rate'],
#         n_steps=config['ppo']['n_steps'],
#         batch_size=config['ppo']['batch_size'],
#         n_epochs=config['ppo']['n_epochs'],
#         gamma=config['ppo']['gamma'],
#         gae_lambda=config['ppo']['gae_lambda'],
#         clip_range=config['ppo']['clip_range'],
#         ent_coef=config['ppo']['ent_coef'],
#         vf_coef=config['ppo']['vf_coef'],
#         max_grad_norm=config['ppo']['max_grad_norm'],
#         tensorboard_log=config['training']['tensorboard_log'],
#         policy_kwargs=dict(
#             gnn_type=config['model']['gnn_type'],
#             hidden_dim=config['model']['hidden_dim'],
#             num_gnn_layers=config['model']['num_gnn_layers'],
#             activation=config['model']['activation'],
#             dropout=config['model']['dropout']
#         ),
#         verbose=1,
#         device=config['experiment']['device']
#     )
    
#     checkpoint_dir = Path(config['training']['checkpoint_dir'])
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
#     callback = FinalTaskAllocationCallback(
#         save_freq=config['training']['save_freq'],
#         save_path=checkpoint_dir / 'metrics'
#     )
    
#     print(f"\nStarting training...")
    
#     model.learn(
#         total_timesteps=config['training']['total_timesteps'],
#         callback=callback,
#         log_interval=config['training']['log_interval']
#     )
    
#     model.save(checkpoint_dir / "ppo_final_working")
#     print(f"\n✓ Training complete! Model saved.")
    
#     with open(checkpoint_dir / 'config.yaml', 'w') as f:
#         yaml.dump(config, f)


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.gnn_policy_simple import SimpleGNNActorCriticPolicy
from src.utils.callbacks import FinalTaskAllocationCallback


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config):
    env_config = config['environment']

    agents = np.load(env_config['agents_file'], allow_pickle=True)

    batches = []
    for i in range(env_config['n_batches']):
        batch_file = Path(env_config['data_dir']) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batch = np.load(batch_file, allow_pickle=True)
            batches.append(batch)

    base_env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_config['radius'],
        feature_size=env_config['feature_size'],
        use_true_id=env_config['use_true_id'],
        all_batches=True
    )

    # ✅ you can set k_max in config later; default is 5
    env = WarehouseEnvSB3Final(base_env, assignment_interval=env_config['assignment_interval'], k_max=5)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(config)
    env = DummyVecEnv([lambda: env])

    print("\nEnvironment ready for training!")

    model = PPO(
        policy=SimpleGNNActorCriticPolicy,
        env=env,
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
        tensorboard_log=config['training']['tensorboard_log'],
        policy_kwargs=dict(
            gnn_type=config['model']['gnn_type'],
            hidden_dim=config['model']['hidden_dim'],
            num_gnn_layers=config['model']['num_gnn_layers'],
            activation=config['model']['activation'],
            dropout=config['model']['dropout']
        ),
        verbose=1,
        device=config['experiment']['device']
    )

    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callback = FinalTaskAllocationCallback(
        save_freq=config['training']['save_freq'],
        save_path=checkpoint_dir / 'metrics'
    )

    print("\nStarting training...")

    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callback,
        log_interval=config['training']['log_interval']
    )

    model.save(checkpoint_dir / "ppo_final_maskable")
    print("\n✓ Training complete! Model saved.")


if __name__ == "__main__":
    main()