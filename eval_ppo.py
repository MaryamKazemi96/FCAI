# #!/usr/bin/env python3
# """
# Evaluate trained PPO model (fixed).

# Usage:
#     python eval_ppo_fixed.py --model checkpoints_ppo/ppo_final_working --episodes 100
# """

# import argparse
# import yaml
# import numpy as np
# import json
# from pathlib import Path
# from stable_baselines3 import PPO

# # Import the SAME functions as training
from train_ppo import make_env, load_config


# def evaluate(model, env, n_episodes=100, deterministic=True):
#     """Evaluate model."""
#     episode_rewards = []
#     episode_completions = []
#     episode_obsolete = []
#     episode_lengths = []
    
#     for ep in range(n_episodes):
#         obs, info = env.reset()
#         done = False
#         episode_reward = 0
#         step_count = 0
        
#         while not done:
#             action, _states = model.predict(obs, deterministic=deterministic)
#             obs, reward, terminated, truncated, info = env.step(action)
#             episode_reward += reward
#             step_count += 1
#             done = terminated or truncated
        
#         # Get completion from info dict
#         completed = info.get('episode_completed', 0)
#         obsolete = info.get('episode_obsolete', 0)
        
#         episode_rewards.append(episode_reward)
#         episode_completions.append(completed)
#         episode_obsolete.append(obsolete)
#         episode_lengths.append(step_count)
        
#         if (ep + 1) % 10 == 0:
#             print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.1f}, "
#                   f"Completed={completed}/15, Steps={step_count}")
    
#     # Compute statistics
#     rewards_array = np.array(episode_rewards)
#     completions_array = np.array(episode_completions)
#     obsolete_array = np.array(episode_obsolete)
    
#     # Summary
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
#     parser.add_argument('--model', type=str, default='checkpoints_ppo/ppo_final_working',
#                        help='Path to saved model')
#     parser.add_argument('--config', type=str, default='configs/training_config.yaml')
#     parser.add_argument('--episodes', type=int, default=100)
#     parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
#     args = parser.parse_args()
    
#     # Load config and create environment (same as training)
#     config = load_config(args.config)
#     env = make_env(config)
    
#     # Load model
#     print(f"Loading model from {args.model}...")
#     model = PPO.load(args.model, env=env)
#     print("Model loaded successfully!\n")
    
#     # Evaluate
#     results = evaluate(model, env, n_episodes=args.episodes, deterministic=not args.stochastic)
    
#     # Save results
#     model_path = Path(args.model)
#     results_dir = model_path.parent
#     results_file = results_dir / 'eval_results.json'
    
#     with open(results_file, 'w') as f:
#         json_results = {
#             'rewards': [float(r) for r in results['rewards']],
#             'completions': [int(c) for c in results['completions']],
#             'obsolete': [int(o) for o in results['obsolete']],
#             'lengths': [int(l) for l in results['lengths']],
#             'total_tasks': int(results['total_tasks']),
#             'num_episodes': args.episodes
#         }
#         json.dump(json_results, f, indent=2)
    
#     print(f"\n✓ Results saved to {results_file}")


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""
Evaluate trained PPO model.

Usage:
    python eval_ppo.py --model checkpoints_ppo/ppo_final_masked --episodes 100
"""

import argparse
import numpy as np
from stable_baselines3 import PPO

# Import the SAME functions as training
from train_ppo import make_env, load_config


def evaluate(model, env, n_episodes=100, deterministic=True):
    episode_rewards = []
    episode_completions = []
    episode_obsolete = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            step_count += 1
            done = terminated or truncated

        completed = info.get('episode_completed', 0)
        obsolete = info.get('episode_obsolete', 0)

        episode_rewards.append(episode_reward)
        episode_completions.append(completed)
        episode_obsolete.append(obsolete)
        episode_lengths.append(step_count)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.1f}, "
                  f"Completed={completed}/15, Steps={step_count}")

    rewards_array = np.array(episode_rewards)
    completions_array = np.array(episode_completions)
    obsolete_array = np.array(episode_obsolete)

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Episodes:              {n_episodes}")
    print(f"Mode:                  {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"\nRewards:")
    print(f"  Mean:                {rewards_array.mean():.2f} ± {rewards_array.std():.2f}")
    print(f"  Min/Max:             {rewards_array.min():.2f} / {rewards_array.max():.2f}")
    print(f"\nTask Completion:")
    print(f"  Mean:                {completions_array.mean():.2f}/15 "
          f"({100*completions_array.mean()/15:.1f}%)")
    print(f"  Best:                {completions_array.max()}/15")
    print(f"  Worst:               {completions_array.min()}/15")
    print(f"\nObsolete Tasks:")
    print(f"  Mean:                {obsolete_array.mean():.2f}")
    print(f"  Total:               {obsolete_array.sum():.0f}")
    print(f"\nEpisode Length:")
    print(f"  Mean:                {np.mean(episode_lengths):.1f} steps")
    print("="*70)

    return {
        'rewards': episode_rewards,
        'completions': episode_completions,
        'obsolete': episode_obsolete,
        'lengths': episode_lengths,
        'total_tasks': 15
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints_ppo/ppo_final_masked',
                        help='Path to saved model (.zip or directory)')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_env(config)

    # ✅ IMPORTANT: load with PPO since you trained with PPO
    model = PPO.load(args.model)

    evaluate(model, env, n_episodes=args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()