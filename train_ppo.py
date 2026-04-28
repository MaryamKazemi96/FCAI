# import argparse
# import yaml
# import numpy as np
# import torch
# from pathlib import Path
# import datetime
# import platform
# import subprocess
# import json

# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# from src.models.sb3_gnn_policy import RTGNNPolicy
# from stable_baselines3 import PPO

# from src.environment.environment import MultiTaskAllocationEnv
# from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
# from src.models.gnn_policy_simple import SimpleGNNActorCriticPolicy
# from src.utils.callbacks import FinalTaskAllocationCallback

# def get_git_commit() -> str:
#     try:
#         return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
#     except Exception:
#         return "unknown"
    
# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def set_seed(seed: int):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def make_env(config, seed: int):
#     env_config = config['environment']
#     agents = np.load(env_config['agents_file'], allow_pickle=True)

#     batches = []
#     for i in range(env_config['n_batches']):
#         batch_file = Path(env_config['data_dir']) / f"tasks_batch_{i}.npy"
#         if batch_file.exists():
#             batches.append(np.load(batch_file, allow_pickle=True))

#     base_env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=batches,
#         radius=env_config['radius'],
#         feature_size=env_config['feature_size'],
#         use_true_id=env_config['use_true_id'],
#         all_batches=True
#     )

#     env = WarehouseEnvSB3Final(base_env, assignment_interval=env_config['assignment_interval'], k_max=env_config.get("k_max", 5))
#     env = Monitor(env)
#     env.reset(seed=seed)
#     return env


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='configs/training_config.yaml')
#     args = parser.parse_args()
#     config = load_config(args.config)

#     seeds = config.get("experiment", {}).get("seeds")
#     if not seeds:
#         seeds = [int(config["experiment"]["seed"])]

#     checkpoint_root = Path(config['training']['checkpoint_dir'])
#     checkpoint_root.mkdir(parents=True, exist_ok=True)

#     for seed in seeds:
#         seed = int(seed)
#         print("\n" + "="*80)
#         print(f"TRAINING SEED {seed}")
#         print("="*80)

#         set_seed(seed)

#         env = make_env(config, seed)
#         env = DummyVecEnv([lambda: env])

#         seed_dir = checkpoint_root / f"seed_{seed}"
#         seed_dir.mkdir(parents=True, exist_ok=True)
#                 # ---- save run metadata (hyperparams + config) ----
#         run_meta = {
#             "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
#             "seed": seed,
#             "config_path": args.config,
#             "git_commit": get_git_commit(),
#             "platform": {
#                 "python": platform.python_version(),
#                 "system": platform.platform(),
#                 "torch_cuda_available": bool(torch.cuda.is_available()),
#             },
#             "environment": config.get("environment", {}),
#             "model": config.get("model", {}),
#             "ppo": config.get("ppo", {}),
#             "policy_kwargs": {
#                 "in_dim": config["environment"]["feature_size"],
#                 "hidden_dim": config["model"]["hidden_dim"],
#                 "k_max": config["environment"].get("k_max", 5),
#                 "gnn_type": config["model"]["gnn_type"],
#                 "num_gnn_layers": config["model"]["num_gnn_layers"],
#                 "activation": config["model"]["activation"],
#                 "dropout": config["model"]["dropout"],
#                 "noop_init": -1.0,
#                 "noop_frozen": True,
#                 "logit_temperature": 1.0,
#             },
#         }

#         (seed_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
#         callback = FinalTaskAllocationCallback(
#             save_freq=config['training']['save_freq'],
#             save_path=seed_dir / 'metrics'
#         )

#         model = PPO(
#             policy=RTGNNPolicy,
#             env=env,
#             policy_kwargs=dict(
#                 in_dim=config['environment']['feature_size'],
#                 hidden_dim=config['model']['hidden_dim'],
#                 k_max=config['environment'].get("k_max", 5),
#                 gnn_type=config['model']['gnn_type'],
#                 num_gnn_layers=config['model']['num_gnn_layers'],
#                 activation=config['model']['activation'],
#                 dropout=config['model']['dropout'],
#                 noop_init=-1,
#                 noop_frozen=False,
#                 logit_temperature=1.0,
#             ),
             
#             learning_rate=config['ppo']['learning_rate'],
#             n_steps=config['ppo']['n_steps'],
#             batch_size=config['ppo']['batch_size'],
#             n_epochs=config['ppo']['n_epochs'],
#             gamma=config['ppo']['gamma'],
#             gae_lambda=config['ppo']['gae_lambda'],
#             clip_range=config['ppo']['clip_range'],
#             ent_coef=config['ppo']['ent_coef'],
#             vf_coef=config['ppo']['vf_coef'],
#             max_grad_norm=config['ppo']['max_grad_norm'],
#             tensorboard_log=str(seed_dir / "tensorboard"),
#             verbose=1,
#             device=config['experiment']['device'],
#             seed=seed,
#         )

#         model.learn(
#             total_timesteps=config['training']['total_timesteps'],
#             callback=callback,
#             log_interval=config['training']['log_interval']
#         )

#         model.save(seed_dir / "ppo_final")
#         print(f"✓ Saved: {seed_dir / 'ppo_final'}.zip")

# if __name__ == "__main__":
#     main()
import argparse
import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.sb3_gnn_policy import RTGNNPolicy
from src.utils.callbacks import FinalTaskAllocationCallback
from src.utils.checkpoint_callback import EpisodeTimestepCheckpointCallback, latest_model_path


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
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
    # print("[debug] baches loaded:", len(batches), batches)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier. If not provided, a UTC timestamp is used.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seeds = config.get("experiment", {}).get("seeds")
    if not seeds:
        seeds = [int(config["experiment"]["seed"])]

    checkpoint_root = Path(config["training"]["checkpoint_dir"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # One run_id per script execution (so seed_42/run_<id>, seed_123/run_<id>, etc.)
    run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for seed in seeds:
        seed = int(seed)
        print("\n" + "=" * 80)
        print(f"TRAINING SEED {seed} | run_id={run_id}")
        print("=" * 80)

        set_seed(seed)

        env = make_env(config, seed)
        env = DummyVecEnv([lambda: env])

        seed_dir = checkpoint_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # --- NEW: unique run directory (prevents overwriting old runs) ---
        run_dir = seed_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # dirs like your colleague (inside run_dir)
        init_dir = run_dir / "init_model"
        model_dir = run_dir / "models"
        metrics_dir = run_dir / "metrics"
        tb_dir = run_dir / "tensorboard"

        init_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

        # ---- save run metadata once per run ----
        # Put everything that defines the run here (config + code version + key kwargs)
        policy_kwargs = {
            "in_dim": config["environment"]["feature_size"],
            "hidden_dim": config["model"]["hidden_dim"],
            "k_max": config["environment"].get("k_max", 5),
            "gnn_type": config["model"]["gnn_type"],
            "num_gnn_layers": config["model"]["num_gnn_layers"],
            "activation": config["model"]["activation"],
            "dropout": config["model"]["dropout"],
            "noop_init": -1.0,
            "freeze_noop_logit": True,
            "logit_temperature": 1.0,
        }

        run_meta = {
            "run_id": run_id,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "seed": seed,
            "config_path": args.config,
            "git_commit": get_git_commit(),
            "platform": {
                "python": platform.python_version(),
                "system": platform.platform(),
                "torch_cuda_available": bool(torch.cuda.is_available()),
            },
            "environment": config.get("environment", {}),
            "model": config.get("model", {}),
            "ppo": config.get("ppo", {}),
            "policy_kwargs": policy_kwargs,
            "paths": {
                "run_dir": str(run_dir),
                "init_dir": str(init_dir),
                "model_dir": str(model_dir),
                "metrics_dir": str(metrics_dir),
                "tensorboard_dir": str(tb_dir),
            },
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        metrics_cb = FinalTaskAllocationCallback(
            save_freq=int(config["training"]["save_freq"]),
            save_path=metrics_dir,
        )

        ckpt_cb = EpisodeTimestepCheckpointCallback(
            save_dir=model_dir,
            save_every_steps=int(config["training"]["save_freq"]),
            verbose=1,
        )

        if args.continue_training:
            latest_path, last_ep, last_ts = latest_model_path(str(model_dir))
            print(f"[CONTINUE] Loading model: {latest_path}")
            model = PPO.load(latest_path, env=env, device=config["experiment"]["device"])
            # keep timesteps continuous like colleague
            model.num_timesteps = int(last_ts)
            ckpt_cb.episode_idx = int(last_ep)
        else:
            model = PPO(
                policy=RTGNNPolicy,
                env=env,
                policy_kwargs=dict(policy_kwargs),
                learning_rate=config["ppo"]["learning_rate"],
                n_steps=config["ppo"]["n_steps"],
                batch_size=config["ppo"]["batch_size"],
                n_epochs=config["ppo"]["n_epochs"],
                gamma=config["ppo"]["gamma"],
                gae_lambda=config["ppo"]["gae_lambda"],
                clip_range=config["ppo"]["clip_range"],
                ent_coef=config["ppo"]["ent_coef"],
                vf_coef=config["ppo"]["vf_coef"],
                max_grad_norm=config["ppo"]["max_grad_norm"],
                tensorboard_log=str(tb_dir),
                verbose=1,
                device=config["experiment"]["device"],
                seed=seed,
            )

            # save initial model like colleague
            init_path = init_dir / "model_episode0_ts0.zip"
            model.save(init_path)
            print(f"[INIT] Saved {init_path}")

        model.learn(
            total_timesteps=int(config["training"]["total_timesteps"]),
            callback=[metrics_cb, ckpt_cb],
            log_interval=int(config["training"]["log_interval"]),
            reset_num_timesteps=not args.continue_training,
        )

        # final convenience save (inside run_dir, so never overwrites other runs)
        model.save(run_dir / "ppo_final")
        print(f"✓ Saved: {run_dir / 'ppo_final'}.zip")


if __name__ == "__main__":
    main()