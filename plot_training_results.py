#!/usr/bin/env python3
"""
Plot PPO training results from tensorboard logs.

Usage:
    python plot_training_results.py --log-dir tensorboard_logs
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import glob


def moving_average(data, window=50):
    """Compute moving average."""
    if len(data) < window:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad to original length
    pad_len = len(data) - len(ma)
    return np.concatenate([data[:pad_len], ma])


def load_tensorboard_data(log_dir):
    """Load data from tensorboard logs."""
    # Find the latest run directory
    run_dirs = list(Path(log_dir).glob("PPO_*"))
    
    if not run_dirs:
        print(f"No tensorboard logs found in {log_dir}")
        return {}
    
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_run}")
    
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    
    data = {}
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return data


def plot_training_results(log_dir, checkpoint_dir):
    """Generate comprehensive training plots."""
    
    data = load_tensorboard_data(log_dir)
    
    if not data:
        print("No data to plot!")
        return
    
    print(f"Available metrics: {list(data.keys())}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # ========== 1. Episode Rewards ==========
    if 'rollout/ep_rew_mean' in data:
        ax = fig.add_subplot(gs[0, :])
        ax.set_facecolor('#fafafa')
        
        steps = data['rollout/ep_rew_mean']['steps']
        rewards = data['rollout/ep_rew_mean']['values']
        
        ax.plot(steps, rewards, 'o', markersize=4, alpha=0.3, color='#3498db', label='Raw')
        
        if len(rewards) > 50:
            ax.plot(steps, moving_average(rewards, 50), lw=2.5, color='#2980b9', label='MA(50)')
        
        ax.axhline(np.mean(rewards), color='#e74c3c', lw=2, ls='--', alpha=0.6, 
                   label=f'Mean: {np.mean(rewards):.1f}')
        
        # Trend line
        z = np.polyfit(steps, rewards, 1)
        ax.plot(steps, np.polyval(z, steps), lw=1.8, color='#27ae60', alpha=0.5,
                ls=':', label=f'Trend: {z[0]:+.4f}/step')
        
        ax.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
        ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.25)
        
        # Stats box
        if len(rewards) >= 10:
            last_10 = rewards[-10:]
            txt = (f'Mean:     {np.mean(rewards):.2f}\n'
                   f'Last 10:  {np.mean(last_10):.2f}\n'
                   f'Best:     {np.max(rewards):.2f}')
            ax.text(0.98, 0.04, txt, transform=ax.transAxes, fontsize=9,
                    va='bottom', ha='right', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))
    
    # ========== 2. Task Completion Rate ==========
    if 'task/completion_rate' in data:
        ax = fig.add_subplot(gs[1, 0])
        ax.set_facecolor('#fafafa')
        
        steps = data['task/completion_rate']['steps']
        comp_rate = data['task/completion_rate']['values']
        
        ax.plot(steps, comp_rate, 'o', markersize=4, alpha=0.3, color='#27ae60')
        
        if len(comp_rate) > 20:
            ax.plot(steps, moving_average(comp_rate, 20), lw=2.5, color='#27ae60')
        
        ax.axhline(np.mean(comp_rate), color='#16a085', lw=2, ls='--', alpha=0.6)
        
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Completion Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Task Completion (mean: {np.mean(comp_rate):.1f}%)', 
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 105)
    
    # ========== 3. Completed vs Obsolete Tasks ==========
    if 'task/completed' in data and 'task/obsolete' in data:
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor('#fafafa')
        
        steps_comp = data['task/completed']['steps']
        completed = data['task/completed']['values']
        obsolete = data['task/obsolete']['values']
        
        if len(completed) > 20:
            ax.plot(steps_comp, moving_average(completed, 20), lw=2.5, 
                    color='#27ae60', label='Completed')
            ax.plot(steps_comp, moving_average(obsolete, 20), lw=2.5, 
                    color='#e74c3c', label='Obsolete')
        else:
            ax.plot(steps_comp, completed, lw=2.5, color='#27ae60', label='Completed')
            ax.plot(steps_comp, obsolete, lw=2.5, color='#e74c3c', label='Obsolete')
        
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Tasks', fontsize=10, fontweight='bold')
        ax.set_title('Task Outcomes', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    
    # ========== 4. Value Loss ==========
    if 'train/value_loss' in data:
        ax = fig.add_subplot(gs[2, 0])
        ax.set_facecolor('#fafafa')
        
        steps = data['train/value_loss']['steps']
        loss = data['train/value_loss']['values']
        
        ax.plot(steps, loss, 'o', markersize=3, alpha=0.3, color='#3498db')
        
        if len(loss) > 20:
            ax.plot(steps, moving_average(loss, 20), lw=2.5, color='#2980b9')
        
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value Loss', fontsize=10, fontweight='bold')
        ax.set_title('Critic Loss', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.set_yscale('log')
    
    # ========== 5. Policy Entropy ==========
    if 'train/entropy_loss' in data:
        ax = fig.add_subplot(gs[2, 1])
        ax.set_facecolor('#fafafa')
        
        steps = data['train/entropy_loss']['steps']
        entropy = data['train/entropy_loss']['values']
        
        ax.plot(steps, entropy, 'o', markersize=3, alpha=0.3, color='#9b59b6')
        
        if len(entropy) > 20:
            ax.plot(steps, moving_average(entropy, 20), lw=2.5, color='#8e44ad')
        
        ax.axhline(-1.0, color='orange', lw=1.5, ls='--', alpha=0.5, label='Warning')
        ax.axhline(-0.5, color='red', lw=1.5, ls='--', alpha=0.5, label='Collapse')
        
        ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
        ax.set_ylabel('Entropy Loss', fontsize=10, fontweight='bold')
        ax.set_title('Policy Entropy (Exploration)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    
    plt.tight_layout()
    output_file = checkpoint_dir / "training_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='tensorboard_logs',
                       help='Tensorboard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_ppo',
                       help='Checkpoint directory for saving plots')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist!")
        return
    
    print(f"Loading training results from {log_dir}...")
    plot_training_results(log_dir, checkpoint_dir)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()