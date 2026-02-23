#!/usr/bin/env python3
"""
Plot comparison between PPO model and baseline policies with moving averages.

Usage:
    python plot_comparison.py --checkpoint-dir checkpoints_ppo
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional


def moving_average(data, window=10):
    """Compute moving average with specified window size."""
    if len(data) < window:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad to original length
    pad_len = len(data) - len(ma)
    return np.concatenate([data[:pad_len], ma])


def load_results(checkpoint_dir: Path) -> Dict:
    """Load PPO and baseline results."""
    results = {}
    
    # Load PPO evaluation results
    ppo_file = checkpoint_dir / 'eval_results.json'
    if ppo_file.exists():
        with open(ppo_file, 'r') as f:
            results['ppo'] = json.load(f)
        print(f"✓ Loaded PPO results from {ppo_file}")
    else:
        print(f"⚠ Warning: PPO results not found at {ppo_file}")
    
    # Load baseline results
    baseline_file = checkpoint_dir / 'baseline_results_all.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
            for policy in ['random', 'greedy', 'unique']:
                if policy in baseline_data:
                    results[policy] = baseline_data[policy]
        print(f"✓ Loaded baseline results from {baseline_file}")
    else:
        # Try loading individual baseline files
        for policy in ['random', 'greedy', 'unique']:
            policy_file = checkpoint_dir / f'baseline_{policy}_results.json'
            if policy_file.exists():
                with open(policy_file, 'r') as f:
                    results[policy] = json.load(f)
                print(f"✓ Loaded {policy} baseline from {policy_file}")
    
    return results


def plot_rewards_with_ma(results: Dict, output_file: str, ma_window: int = 10):
    """Plot episode rewards with moving averages for all methods."""
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_facecolor('#fafafa')
    
    policy_order = ['random', 'greedy', 'unique', 'ppo']
    policy_labels = {
        'random': 'Random',
        'greedy': 'Greedy',
        'unique': 'Greedy Unique',
        'ppo': 'PPO (Ours)'
    }
    colors = {
        'random': '#e74c3c', 
        'greedy': '#f39c12', 
        'unique': '#3498db', 
        'ppo': '#27ae60'
    }
    
    max_reward = -np.inf
    min_reward = np.inf
    
    for policy in policy_order:
        if policy not in results:
            continue
        
        rewards = np.array(results[policy]['rewards'])
        episodes = np.arange(1, len(rewards) + 1)
        
        # Raw data (very transparent)
        ax.plot(episodes, rewards, alpha=0.15, color=colors[policy], 
                linewidth=0.8)
        
        # Moving average (prominent)
        ma_rewards = moving_average(rewards, window=ma_window)
        ax.plot(episodes, ma_rewards, alpha=0.9, color=colors[policy], 
                linewidth=2.5, label=f'{policy_labels[policy]} (MA={ma_window})')
        
        # Mean line (dashed)
        mean_reward = np.mean(rewards)
        ax.axhline(mean_reward, color=colors[policy], linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        
        max_reward = max(max_reward, np.max(rewards))
        min_reward = min(min_reward, np.min(rewards))
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=13, fontweight='bold')
    ax.set_title(f'Episode Rewards: Moving Average Comparison (window={ma_window})', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add some padding
    y_range = max_reward - min_reward
    ax.set_ylim(min_reward - 0.05*y_range, max_reward + 0.05*y_range)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved rewards with MA to {output_file}")
    plt.close()


def plot_completion_with_ma(results: Dict, output_file: str, ma_window: int = 10, total_tasks: int = 15):
    """Plot task completion with moving averages."""
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_facecolor('#fafafa')
    
    policy_order = ['random', 'greedy', 'unique', 'ppo']
    policy_labels = {
        'random': 'Random',
        'greedy': 'Greedy',
        'unique': 'Greedy Unique',
        'ppo': 'PPO (Ours)'
    }
    colors = {
        'random': '#e74c3c', 
        'greedy': '#f39c12', 
        'unique': '#3498db', 
        'ppo': '#27ae60'
    }
    
    for policy in policy_order:
        if policy not in results:
            continue
        
        completions = np.array(results[policy]['completions'])
        episodes = np.arange(1, len(completions) + 1)
        
        # Raw data (very transparent)
        ax.plot(episodes, completions, alpha=0.15, color=colors[policy], 
                linewidth=0.8)
        
        # Moving average (prominent)
        ma_completions = moving_average(completions, window=ma_window)
        ax.plot(episodes, ma_completions, alpha=0.9, color=colors[policy], 
                linewidth=2.5, label=f'{policy_labels[policy]} (MA={ma_window})')
        
        # Mean line (dashed)
        mean_completion = np.mean(completions)
        ax.axhline(mean_completion, color=colors[policy], linestyle='--', 
                   linewidth=1.5, alpha=0.5)
    
    # Add reference line for total tasks
    ax.axhline(total_tasks, color='black', linestyle=':', linewidth=2, 
               alpha=0.6, label=f'Max Tasks ({total_tasks})')
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Tasks Completed', fontsize=13, fontweight='bold')
    ax.set_title(f'Task Completion: Moving Average Comparison (window={ma_window})', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, total_tasks + 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved completions with MA to {output_file}")
    plt.close()


def plot_comparison_bars(results: Dict, output_file: str):
    """Create bar chart comparison of all methods."""
    
    methods = []
    rewards_mean = []
    rewards_std = []
    completions_mean = []
    completions_std = []
    obsolete_mean = []
    
    # Define order: baselines first, then PPO
    policy_order = ['random', 'greedy', 'unique', 'ppo']
    policy_labels = {
        'random': 'Random',
        'greedy': 'Greedy',
        'unique': 'Greedy\nUnique',
        'ppo': 'PPO\n(Ours)'
    }
    
    for policy in policy_order:
        if policy not in results:
            continue
        
        data = results[policy]
        methods.append(policy_labels[policy])
        
        # Compute statistics
        if 'stats' in data:
            rewards_mean.append(data['stats']['reward_mean'])
            rewards_std.append(data['stats']['reward_std'])
            completions_mean.append(data['stats']['completion_mean'])
            completions_std.append(data['stats']['completion_std'])
            obsolete_mean.append(data['stats']['obsolete_mean'])
        else:
            rewards_mean.append(np.mean(data['rewards']))
            rewards_std.append(np.std(data['rewards']))
            completions_mean.append(np.mean(data['completions']))
            completions_std.append(np.std(data['completions']))
            obsolete_mean.append(np.mean(data['obsolete']))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']  # red, orange, blue, green
    x_pos = np.arange(len(methods))
    
    # Plot 1: Rewards
    ax = axes[0]
    ax.set_facecolor('#fafafa')
    bars = ax.bar(x_pos, rewards_mean, yerr=rewards_std, capsize=5, 
                   color=colors[:len(methods)], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, rewards_mean, rewards_std)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Task Completion
    ax = axes[1]
    ax.set_facecolor('#fafafa')
    bars = ax.bar(x_pos, completions_mean, yerr=completions_std, capsize=5,
                   color=colors[:len(methods)], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylabel('Tasks Completed', fontsize=12, fontweight='bold')
    ax.set_title('Task Completion Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, completions_mean, completions_std)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Obsolete Tasks
    ax = axes[2]
    ax.set_facecolor('#fafafa')
    bars = ax.bar(x_pos, obsolete_mean, capsize=5,
                   color=colors[:len(methods)], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_ylabel('Obsolete Tasks', fontsize=12, fontweight='bold')
    ax.set_title('Obsolete Task Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, obsolete_mean)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison bars to {output_file}")
    plt.close()


def plot_detailed_comparison_with_ma(results: Dict, output_file: str, ma_window: int = 10):
    """Create comprehensive comparison with moving averages."""
    
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    policy_order = ['random', 'greedy', 'unique', 'ppo']
    policy_labels = {
        'random': 'Random',
        'greedy': 'Greedy',
        'unique': 'Greedy Unique',
        'ppo': 'PPO (Ours)'
    }
    colors = {'random': '#e74c3c', 'greedy': '#f39c12', 'unique': '#3498db', 'ppo': '#27ae60'}
    
    # Panel 1: Reward with MA (takes full width)
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor('#fafafa')
    
    for policy in policy_order:
        if policy not in results:
            continue
        rewards = np.array(results[policy]['rewards'])
        episodes = np.arange(1, len(rewards) + 1)
        
        # Raw (transparent)
        ax.plot(episodes, rewards, alpha=0.1, color=colors[policy], linewidth=0.5)
        
        # MA (prominent)
        ma_rewards = moving_average(rewards, window=ma_window)
        ax.plot(episodes, ma_rewards, alpha=0.9, color=colors[policy], linewidth=2.5,
                label=f'{policy_labels[policy]} ({np.mean(rewards):.1f})')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Episode Rewards with Moving Average (window={ma_window})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', ncol=4)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel 2: Task completion with MA
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor('#fafafa')
    
    for policy in policy_order:
        if policy not in results:
            continue
        completions = np.array(results[policy]['completions'])
        episodes = np.arange(1, len(completions) + 1)
        
        # Raw (transparent)
        ax.plot(episodes, completions, alpha=0.1, color=colors[policy], linewidth=0.5)
        
        # MA (prominent)
        ma_completions = moving_average(completions, window=ma_window)
        ax.plot(episodes, ma_completions, alpha=0.9, color=colors[policy], linewidth=2.5,
                label=f'{policy_labels[policy]} ({np.mean(completions):.2f})')
    
    ax.axhline(15, color='black', linestyle=':', linewidth=2, alpha=0.6, label='Max (15)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasks Completed', fontsize=12, fontweight='bold')
    ax.set_title(f'Task Completion with Moving Average (window={ma_window})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', ncol=5)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 16)
    
    # Panel 3: Reward distribution (box plot)
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor('#fafafa')
    
    data_to_plot = []
    labels = []
    colors_list = []
    
    for policy in policy_order:
        if policy not in results:
            continue
        data_to_plot.append(results[policy]['rewards'])
        labels.append(policy_labels[policy])
        colors_list.append(colors[policy])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel 4: Summary statistics table
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [['Policy', 'Reward', 'Completion', 'Obsolete']]
    for policy in policy_order:
        if policy not in results:
            continue
        data = results[policy]
        if 'stats' in data:
            stats = data['stats']
            row = [
                policy_labels[policy],
                f"{stats['reward_mean']:.1f} ± {stats['reward_std']:.1f}",
                f"{stats['completion_mean']:.2f} ± {stats['completion_std']:.2f}",
                f"{stats['obsolete_mean']:.2f}"
            ]
        else:
            row = [
                policy_labels[policy],
                f"{np.mean(data['rewards']):.1f} ± {np.std(data['rewards']):.1f}",
                f"{np.mean(data['completions']):.2f} ± {np.std(data['completions']):.2f}",
                f"{np.mean(data['obsolete']):.2f}"
            ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows with colors
    for i, policy in enumerate(policy_order, 1):
        if policy not in results:
            continue
        color = colors[policy]
        for j in range(4):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)
    
    ax.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed comparison to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot comparison between PPO and baselines')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_ppo',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as checkpoint-dir)')
    parser.add_argument('--ma-window', type=int, default=10,
                       help='Moving average window size (default: 10)')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {checkpoint_dir}...")
    results = load_results(checkpoint_dir)
    
    if not results:
        print(" No results found! Make sure to run:")
        print("   1. python eval_ppo.py (for PPO results)")
        print("   2. python eval_baselines_warehouse.py (for baseline results)")
        return
    
    print(f"\nFound results for: {', '.join(results.keys())}")
    print(f"Using moving average window: {args.ma_window}")
    
    # Generate plots
    print("\nGenerating comparison plots with moving averages...")
    
    plot_rewards_with_ma(results, output_dir / 'comparison_rewards_ma.png', ma_window=args.ma_window)
    plot_completion_with_ma(results, output_dir / 'comparison_completion_ma.png', ma_window=args.ma_window)
    plot_comparison_bars(results, output_dir / 'comparison_bars.png')
    plot_detailed_comparison_with_ma(results, output_dir / 'comparison_detailed_ma.png', ma_window=args.ma_window)
    
    print(f"\n{'='*70}")
    print("✓ All comparison plots generated successfully!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'comparison_rewards_ma.png'} (Rewards with MA)")
    print(f"  - {output_dir / 'comparison_completion_ma.png'} (Completions with MA)")
    print(f"  - {output_dir / 'comparison_bars.png'} (Bar charts)")
    print(f"  - {output_dir / 'comparison_detailed_ma.png'} (Comprehensive view)")
    print()


if __name__ == '__main__':
    main()