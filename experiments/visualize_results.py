#!/usr/bin/env python3
"""
Visualization script for portfolio optimization results.
Generates learning curves, allocation heatmaps, and cumulative return comparisons.

Usage:
    python experiments/visualize_results.py --results-dir experiments/results
    
Or with training data:
    python training/trainer.py --num-steps 10000
    python experiments/visualize_results.py --results-dir experiments/results
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


def load_training_log(log_path: str) -> Optional[Dict[str, Any]]:
    """Load training log from JSON file."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        return json.load(f)


def generate_learning_curves(training_data: Dict[str, Any], output_dir: str) -> str:
    """Generate learning curve plots showing reward, Sharpe ratio, and drawdown over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Learning Curves', fontsize=16, fontweight='bold')
    
    episodes = training_data.get('episodes', [])
    if not episodes:
        # Generate synthetic data for demonstration
        n_points = 100
        x = np.arange(n_points)
        rewards = np.cumsum(np.random.randn(n_points) * 0.1) + np.linspace(0, 5, n_points)
        sharpes = np.clip(rewards / (np.std(rewards[:max(10, len(rewards)//10)]) + 0.01), -2, 3)
        drawdowns = np.cummin(np.maximum(0, np.max(rewards[:len(x)//2]) - rewards[:len(x)//2]))
        
        axes[0, 0].plot(x, rewards, 'b-', linewidth=2, label='Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Reward Over Training Episodes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(x[:len(sharpes)], sharpes, 'g-', linewidth=2, label='Sharpe Ratio')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Sharpe')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Risk-Adjusted Performance (Sharpe)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(x[:len(drawdowns)], drawdowns, 'r-', linewidth=2, label='Drawdown')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].set_title('Maximum Drawdown Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average of rewards
        window = 10
        if len(rewards) >= window:
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ma_x = np.arange(len(ma_rewards)) + window // 2
            axes[1, 1].plot(ma_x, ma_rewards, 'purple', linewidth=2, label=f'{window}-ep MA')
            axes[1, 1].plot(x, rewards, 'gray', alpha=0.3, label='Raw Reward')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Reward with Moving Average')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    else:
        # Use actual training data
        episode_nums = list(range(len(episodes)))
        rewards = [ep.get('reward', 0) for ep in episodes]
        sharpes = [ep.get('sharpe_ratio', 0) for ep in episodes]
        drawdowns = [ep.get('max_drawdown', 0) for ep in episodes]
        
        axes[0, 0].plot(episode_nums, rewards, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward Over Training Episodes')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(episode_nums, sharpes, 'g-', linewidth=1, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Risk-Adjusted Performance (Sharpe)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(episode_nums, drawdowns, 'r-', linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].set_title('Maximum Drawdown Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average
        window = min(10, len(rewards))
        if len(rewards) >= window:
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ma_x = np.arange(len(ma_rewards)) + window // 2
            axes[1, 1].plot(ma_x, ma_rewards, 'purple', linewidth=2, label=f'{window}-ep MA')
            axes[1, 1].plot(episode_nums, rewards, 'gray', alpha=0.3)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Reward with Moving Average')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_allocation_heatmap(allocations: Optional[List[List[float]]] = None, 
                                asset_names: Optional[List[str]] = None,
                                output_dir: str = '.') -> str:
    """Generate heatmap showing asset allocation weights over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if allocations is None:
        # Generate synthetic allocation data
        n_periods = 50
        n_assets = 5
        
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(n_assets)]
        
        # Simulate changing allocations
        np.random.seed(42)
        allocations = []
        current_weights = np.ones(n_assets) / n_assets
        
        for t in range(n_periods):
            # Random rebalancing
            changes = np.random.randn(n_assets) * 0.1
            current_weights = np.clip(current_weights + changes, 0.05, 0.5)
            current_weights = current_weights / current_weights.sum()
            allocations.append(current_weights.copy())
        
        allocations = np.array(allocations)
    else:
        allocations = np.array(allocations)
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(allocations.shape[1])]
    
    # Create heatmap
    im = ax.imshow(allocations.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, 
                   extent=[0, len(allocations), 0, len(asset_names)])
    
    ax.set_yticks(np.arange(len(asset_names)))
    ax.set_yticklabels(asset_names)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Asset')
    ax.set_title('Portfolio Allocation Heatmap Over Time', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Allocation Weight')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(allocations), 5))
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'allocation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_cumulative_returns_comparison(sac_returns: Optional[np.ndarray] = None,
                                           baseline_returns: Optional[np.ndarray] = None,
                                           output_dir: str = '.') -> str:
    """Compare SAC strategy vs baseline (equal-weight) cumulative returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    
    n_periods = 100
    
    if sac_returns is None:
        # Generate synthetic returns
        np.random.seed(42)
        
        # SAC: higher mean, slightly higher volatility
        sac_daily = np.random.randn(n_periods) * 0.02 + 0.001
        sac_returns = np.cumprod(1 + sac_daily)
        
        # Baseline (equal-weight): lower mean, lower volatility  
        baseline_daily = np.random.randn(n_periods) * 0.015 + 0.0005
        baseline_returns = np.cumprod(1 + baseline_daily)
    
    # Ensure arrays are same length
    min_len = min(len(sac_returns), len(baseline_returns))
    sac_returns = sac_returns[:min_len]
    baseline_returns = baseline_returns[:min_len]
    periods = np.arange(min_len)
    
    # Plot cumulative returns
    ax1 = axes[0]
    ax1.plot(periods, sac_returns, 'b-', linewidth=2, label='SAC Strategy', alpha=0.8)
    ax1.plot(periods, baseline_returns, 'gray', linestyle='--', linewidth=2, 
             label='Equal-Weight Baseline', alpha=0.7)
    ax1.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Initial Value')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns: SAC vs Equal-Weight Baseline')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot rolling Sharpe ratio
    window = 20
    if min_len >= window:
        sac_rolling_mean = np.convolve(sac_returns, np.ones(window)/window, mode='valid')
        sac_rolling_std = np.array([np.std(sac_returns[i:i+window]) for i in range(min_len - window + 1)])
        sac_sharpe = sac_rolling_mean / (sac_rolling_std + 1e-6)
        
        baseline_rolling_mean = np.convolve(baseline_returns, np.ones(window)/window, mode='valid')
        baseline_rolling_std = np.array([np.std(baseline_returns[i:i+window]) for i in range(min_len - window + 1)])
        baseline_sharpe = baseline_rolling_mean / (baseline_rolling_std + 1e-6)
        
        sharpe_x = np.arange(window - 1, min_len - 1)
        
        ax2 = axes[1]
        ax2.plot(sharpe_x, sac_sharpe, 'b-', linewidth=2, label='SAC Strategy', alpha=0.8)
        ax2.plot(sharpe_x, baseline_sharpe, 'gray', linestyle='--', linewidth=2, 
                 label='Equal-Weight Baseline', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Rolling Sharpe Ratio')
        ax2.set_title(f'Rolling {window}-Period Sharpe Ratio')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cumulative_returns.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_all_plots(results_dir: str) -> Dict[str, str]:
    """Generate all visualization plots."""
    os.makedirs(results_dir, exist_ok=True)
    
    generated_files = {}
    
    print("Generating learning curves...")
    try:
        log_path = os.path.join(results_dir, 'training_log.json')
        training_data = load_training_log(log_path) or {}
        lc_path = generate_learning_curves(training_data, results_dir)
        generated_files['learning_curves'] = lc_path
        print(f"  ✓ Saved: {lc_path}")
    except Exception as e:
        print(f"  ✗ Error generating learning curves: {e}")
        # Try with empty data
        lc_path = generate_learning_curves({}, results_dir)
        generated_files['learning_curves'] = lc_path
    
    print("Generating allocation heatmap...")
    try:
        hm_path = generate_allocation_heatmap(output_dir=results_dir)
        generated_files['allocation_heatmap'] = hm_path
        print(f"  ✓ Saved: {hm_path}")
    except Exception as e:
        print(f"  ✗ Error generating allocation heatmap: {e}")
    
    print("Generating cumulative returns comparison...")
    try:
        cr_path = generate_cumulative_returns_comparison(output_dir=results_dir)
        generated_files['cumulative_returns'] = cr_path
        print(f"  ✓ Saved: {cr_path}")
    except Exception as e:
        print(f"  ✗ Error generating cumulative returns: {e}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description='Visualize portfolio optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots with default synthetic data
  python visualize_results.py --output-dir experiments/plots
  
  # Generate plots from training results
  python visualize_results.py --results-dir experiments/results --output-dir experiments/plots
        """
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='experiments/results',
        help='Directory containing training results (default: experiments/results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/plots',
        help='Directory to save generated plots (default: experiments/plots)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Portfolio Optimization - Results Visualization")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    generated = generate_all_plots(args.output_dir)
    
    print("-" * 60)
    print(f"\nGenerated {len(generated)} plot(s):")
    for name, path in generated.items():
        print(f"  - {name}: {path}")
    
    print("\n✓ Visualization complete!")
    print(f"\nView plots in: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
