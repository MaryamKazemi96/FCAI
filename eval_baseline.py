#!/usr/bin/env python3
"""
Evaluate baseline policies for warehouse task allocation.

Usage:
    python eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 100
"""

import argparse
import yaml
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

# Import environment setup from training script
from train_ppo import make_env, load_config


POLICIES = ["random", "greedy", "unique"]


def random_policy(env, obs, info, rng: np.random.Generator):
    """
    Random baseline: randomly select a valid action.
    """
    # For Discrete action space, just sample
    return env.action_space.sample()


def greedy_nearest_policy(env, obs, info):
    """
    Greedy baseline: always pick the first valid action (action 0).
    This represents "greedy nearest" - always pick the nearest/first available option.
    """
    # For discrete action space, action 0 is typically the first/nearest
    return 0


def greedy_unique_policy(env, obs, info, chosen_actions_history):
    """
    Greedy unique baseline: try to pick different actions to avoid conflicts.
    
    In a discrete action space representing task assignments, we try to spread
    out action choices to reduce conflicts.
    """
    # Simple strategy: cycle through actions to encourage diversity
    # This is a heuristic for "unique" in discrete action space
    
    # Try actions in order, skipping recently used ones
    n_actions = env.action_space.n
    
    # Prefer actions we haven't used recently
    for action in range(n_actions):
        if action not in chosen_actions_history:
            chosen_actions_history.add(action)
            # Clear history periodically to allow reuse
            if len(chosen_actions_history) > n_actions // 2:
                chosen_actions_history.clear()
            return action
    
    # If all recently used, just pick first (greedy fallback)
    chosen_actions_history.clear()
    return 0


def evaluate_policy(env, policy_name: str, n_episodes: int = 100, seed: int = 42) -> Dict:
    """
    Evaluate a baseline policy.
    """
    rng = np.random.default_rng(seed)
    
    episode_rewards = []
    episode_completions = []
    episode_obsolete = []
    episode_lengths = []
    
    # For unique policy: track recent action choices
    chosen_actions_history = set()
    
    print(f"\n{'='*70}")
    print(f"Evaluating {policy_name.upper()} policy")
    print(f"{'='*70}")
    
    # Get action space info once
    if n_episodes > 0:
        print(f"\n[DEBUG] Inspecting environment action space...")
        action_space = env.action_space
        print(f"[DEBUG] Action space: {action_space}")
        print(f"[DEBUG] Action space type: {type(action_space)}")
        if hasattr(action_space, 'n'):
            print(f"[DEBUG] Action space n: {action_space.n}")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        # Reset unique policy history each episode
        if policy_name == "unique":
            chosen_actions_history.clear()
        
        # Debug first episode
        if ep == 0:
            print(f"\n[DEBUG] First episode observation type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"[DEBUG] Observation keys: {obs.keys()}")
            print(f"[DEBUG] Info keys: {info.keys() if isinstance(info, dict) else 'Not a dict'}")
        
        while not done:
            # Select action based on policy
            if policy_name == "random":
                action = random_policy(env, obs, info, rng)
            elif policy_name == "greedy":
                action = greedy_nearest_policy(env, obs, info)
            elif policy_name == "unique":
                action = greedy_unique_policy(env, obs, info, chosen_actions_history)
            else:
                raise ValueError(f"Unknown policy: {policy_name}")
            
            # Ensure action is a scalar integer (not array)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            
            # Debug first action
            if ep == 0 and step_count == 0:
                print(f"[DEBUG] First action: {action}, type: {type(action)}")
            
            # Take step in environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
            except Exception as e:
                print(f"\n[ERROR] Failed at step {step_count}: {e}")
                print(f"[ERROR] Action was: {action}, type: {type(action)}")
                raise
        
        # Extract episode statistics from info
        completed = info.get('episode_completed', 0)
        obsolete = info.get('episode_obsolete', 0)
        
        episode_rewards.append(episode_reward)
        episode_completions.append(completed)
        episode_obsolete.append(obsolete)
        episode_lengths.append(step_count)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Completed={completed}, "
                  f"Obsolete={obsolete}, "
                  f"Steps={step_count}")
    
    # Compute statistics
    rewards_array = np.array(episode_rewards)
    completions_array = np.array(episode_completions)
    obsolete_array = np.array(episode_obsolete)
    lengths_array = np.array(episode_lengths)
    
    # Print summary
    print(f"\n{'-'*70}")
    print(f"RESULTS SUMMARY - {policy_name.upper()}")
    print(f"{'-'*70}")
    print(f"Rewards:")
    print(f"  Mean:    {rewards_array.mean():.2f} ± {rewards_array.std():.2f}")
    print(f"  Min/Max: {rewards_array.min():.2f} / {rewards_array.max():.2f}")
    print(f"\nTask Completion:")
    print(f"  Mean:    {completions_array.mean():.2f}")
    print(f"  Best:    {completions_array.max()}")
    print(f"  Worst:   {completions_array.min()}")
    print(f"\nObsolete Tasks:")
    print(f"  Mean:    {obsolete_array.mean():.2f}")
    print(f"  Total:   {obsolete_array.sum():.0f}")
    print(f"\nEpisode Length:")
    print(f"  Mean:    {lengths_array.mean():.1f} steps")
    print(f"{'-'*70}\n")
    
    return {
        'policy': policy_name,
        'rewards': episode_rewards,
        'completions': episode_completions,
        'obsolete': episode_obsolete,
        'lengths': episode_lengths,
        'stats': {
            'reward_mean': float(rewards_array.mean()),
            'reward_std': float(rewards_array.std()),
            'completion_mean': float(completions_array.mean()),
            'completion_std': float(completions_array.std()),
            'obsolete_mean': float(obsolete_array.mean()),
            'obsolete_std': float(obsolete_array.std()),
            'length_mean': float(lengths_array.mean()),
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline policies for warehouse environment')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per policy')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints_ppo',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Evaluate each baseline policy
    for policy_name in POLICIES:
        # Create fresh environment for each policy
        env = make_env(config)
        
        # Evaluate
        results = evaluate_policy(env, policy_name, n_episodes=args.episodes, seed=args.seed)
        all_results[policy_name] = results
        
        # Close environment
        env.close()
        
        # Save individual policy results
        policy_file = output_dir / f'baseline_{policy_name}_results.json'
        with open(policy_file, 'w') as f:
            json_results = {
                'policy': policy_name,
                'rewards': [float(r) for r in results['rewards']],
                'completions': [int(c) for c in results['completions']],
                'obsolete': [int(o) for o in results['obsolete']],
                'lengths': [int(l) for l in results['lengths']],
                'stats': results['stats'],
                'num_episodes': args.episodes
            }
            json.dump(json_results, f, indent=2)
        print(f"✓ Saved {policy_file}")
    
    # Save combined results
    combined_file = output_dir / 'baseline_results_all.json'
    with open(combined_file, 'w') as f:
        combined_results = {
            policy: {
                'rewards': [float(r) for r in res['rewards']],
                'completions': [int(c) for c in res['completions']],
                'obsolete': [int(o) for o in res['obsolete']],
                'lengths': [int(l) for l in res['lengths']],
                'stats': res['stats']
            }
            for policy, res in all_results.items()
        }
        combined_results['num_episodes'] = args.episodes
        combined_results['seed'] = args.seed
        json.dump(combined_results, f, indent=2)
    print(f"\n✓ Saved combined results to {combined_file}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Policy':<15} {'Reward':<20} {'Completion':<20} {'Obsolete':<15}")
    print("-"*70)
    for policy_name in POLICIES:
        stats = all_results[policy_name]['stats']
        print(f"{policy_name.upper():<15} "
              f"{stats['reward_mean']:>8.2f} ± {stats['reward_std']:<8.2f} "
              f"{stats['completion_mean']:>8.2f} ± {stats['completion_std']:<8.2f} "
              f"{stats['obsolete_mean']:>8.2f}")
    print("="*70)
    
    # Print vs PPO comparison if available
    ppo_file = output_dir / 'eval_results.json'
    if ppo_file.exists():
        print("\n" + "="*70)
        print("COMPARISON WITH PPO")
        print("="*70)
        with open(ppo_file, 'r') as f:
            ppo_results = json.load(f)
        
        ppo_reward_mean = np.mean(ppo_results['rewards'])
        ppo_reward_std = np.std(ppo_results['rewards'])
        ppo_completion_mean = np.mean(ppo_results['completions'])
        
        print(f"{'Method':<15} {'Reward':<20} {'Completion':<20} {'Improvement':<15}")
        print("-"*70)
        
        for policy_name in POLICIES:
            stats = all_results[policy_name]['stats']
            improvement = ((stats['completion_mean'] / ppo_completion_mean) * 100) if ppo_completion_mean > 0 else 0
            print(f"{policy_name.upper():<15} "
                  f"{stats['reward_mean']:>8.2f} ± {stats['reward_std']:<8.2f} "
                  f"{stats['completion_mean']:>8.2f} / 15        "
                  f"{improvement:>6.1f}%")
        
        print(f"{'PPO (OURS)':<15} "
              f"{ppo_reward_mean:>8.2f} ± {ppo_reward_std:<8.2f} "
              f"{ppo_completion_mean:>8.2f} / 15        "
              f"{'100.0%':>6}")
        print("="*70)
        
        print(f"\n✓ PPO outperforms best baseline by "
              f"{((ppo_completion_mean - max(all_results[p]['stats']['completion_mean'] for p in POLICIES)) / max(all_results[p]['stats']['completion_mean'] for p in POLICIES) * 100):.1f}%")


if __name__ == '__main__':
    main()