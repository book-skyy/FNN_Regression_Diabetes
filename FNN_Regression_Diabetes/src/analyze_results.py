# analyze_results.py
"""
实验结果分析脚本
用于加载和分析保存到experiments/目录下的所有实验结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd

# 计算相对于脚本文件的路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
DEFAULT_COMPARISON_DIR = DEFAULT_EXPERIMENTS_DIR / 'comparison'


def load_experiment_results(experiments_dir: Path = None) -> Dict:
    """
    Load experiment results
    
    Args:
        experiments_dir: Experiment directory (defaults to project's experiments dir)
        
    Returns:
        Dictionary with experiment name as key and metrics dict as value
    """
    if experiments_dir is None:
        experiments_dir = DEFAULT_EXPERIMENTS_DIR
    results = {}
    
    if not experiments_dir.exists():
        print(f"ERROR: Experiments directory not found: {experiments_dir}")
        return results
    
    for exp_dir in sorted(experiments_dir.glob('*/')):
        metrics_file = exp_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                results[exp_dir.name] = metrics
                print(f"OK: Loaded {exp_dir.name}")
        else:
            print(f"SKIP: {exp_dir.name} (metrics.json not found)")
    
    return results


def print_results_summary(results: Dict):
    """Print results summary"""
    if not results:
        print("ERROR: No experiment results found")
        return
    
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    # 创建数据框用于展示
    data = []
    for exp_name, metrics in results.items():
        data.append({
            'Experiment': exp_name[:40],
            'Epochs': metrics.get('epochs_trained', 'N/A'),
            'Best_Val_Loss': f"{metrics.get('best_val_loss', 0):.6f}",
            'Test_MSE': f"{metrics.get('test_mse', 0):.6f}",
            'Test_RMSE': f"{metrics.get('test_rmse', 0):.6f}",
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("=" * 80)


def plot_comparison(results: Dict, save_path: Path = None):
    """Plot RMSE comparison for different experiments (separate by type)"""
    if save_path is None:
        save_path = DEFAULT_COMPARISON_DIR
    
    if not results:
        return
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Separate experiments by type
    depth_exp = {}
    lr_exp = {}
    activation_exp = {}
    
    for exp_name, metrics in results.items():
        if 'depth' in exp_name:
            depth_exp[exp_name] = metrics
        elif 'lr' in exp_name:
            lr_exp[exp_name] = metrics
        elif 'activation' in exp_name:
            activation_exp[exp_name] = metrics
    
    # Plot depth study
    if depth_exp:
        exp_names = list(depth_exp.keys())
        rmse_values = [depth_exp[exp]['test_rmse'] for exp in exp_names]
        
        short_names = []
        for name in exp_names:
            clean = '_'.join(name.split('_')[:-2])
            parts = clean.split('_')
            short_names.append(f"{parts[2]}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(exp_names)), rmse_values, color='steelblue', alpha=0.8)
        
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
        ax.set_title('Network Depth Study - Test RMSE', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(short_names)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        depth_path = save_path / 'rmse_depth_study.png'
        plt.savefig(depth_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved depth study chart: {depth_path}")
        plt.close()
    
    # Plot learning rate study
    if lr_exp:
        exp_names = list(lr_exp.keys())
        rmse_values = [lr_exp[exp]['test_rmse'] for exp in exp_names]
        
        short_names = []
        for name in exp_names:
            clean = '_'.join(name.split('_')[:-2])
            parts = clean.split('_')
            short_names.append(f"{parts[-1]}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(exp_names)), rmse_values, color='coral', alpha=0.8)
        
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Study - Test RMSE', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        lr_path = save_path / 'rmse_lr_study.png'
        plt.savefig(lr_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved learning rate study chart: {lr_path}")
        plt.close()
    
    # Plot activation function study
    if activation_exp:
        exp_names = list(activation_exp.keys())
        rmse_values = [activation_exp[exp]['test_rmse'] for exp in exp_names]
        
        short_names = []
        for name in exp_names:
            clean = '_'.join(name.split('_')[:-2])
            parts = clean.split('_')
            short_names.append(f"{parts[-1]}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(exp_names)), rmse_values, color='mediumseagreen', alpha=0.8)
        
        for bar, rmse in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Activation Function', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test RMSE', fontsize=12, fontweight='bold')
        ax.set_title('Activation Function Study - Test RMSE', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        activation_path = save_path / 'rmse_activation_study.png'
        plt.savefig(activation_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved activation study chart: {activation_path}")
        plt.close()


def plot_training_curves_overlay(results: Dict, save_path: Path = None):
    """Plot training curves overlay for all experiments"""
    if save_path is None:
        save_path = DEFAULT_COMPARISON_DIR
    
    if not results:
        return
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Separate by experiment type
    depth_exp = {}
    lr_exp = {}
    activation_exp = {}
    
    for exp_name, metrics in results.items():
        if 'depth' in exp_name:
            depth_exp[exp_name] = metrics
        elif 'lr' in exp_name:
            lr_exp[exp_name] = metrics
        elif 'activation' in exp_name:
            activation_exp[exp_name] = metrics
    
    # Plot depth study curves
    if depth_exp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp_name, metrics in depth_exp.items():
            if 'train_losses' in metrics and 'val_losses' in metrics:
                epochs = range(1, len(metrics['train_losses']) + 1)
                clean = '_'.join(exp_name.split('_')[:-2])
                parts = clean.split('_')
                short_name = f"{parts[2]}"  # deep/middle/shallow
                
                ax1.plot(epochs, metrics['train_losses'], alpha=0.7, label=short_name, linewidth=2)
                ax2.plot(epochs, metrics['val_losses'], alpha=0.7, label=short_name, linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Network Depth Study - Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (MSE)', fontsize=11)
        ax2.set_title('Network Depth Study - Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        depth_curves_path = save_path / 'training_curves_depth_study.png'
        plt.savefig(depth_curves_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved depth study curves: {depth_curves_path}")
        plt.close()
    
    # Plot learning rate study curves
    if lr_exp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp_name, metrics in lr_exp.items():
            if 'train_losses' in metrics and 'val_losses' in metrics:
                epochs = range(1, len(metrics['train_losses']) + 1)
                clean = '_'.join(exp_name.split('_')[:-2])
                parts = clean.split('_')
                short_name = f"lr_{parts[-1]}"  # learning rate value
                
                ax1.plot(epochs, metrics['train_losses'], alpha=0.7, label=short_name, linewidth=2)
                ax2.plot(epochs, metrics['val_losses'], alpha=0.7, label=short_name, linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Learning Rate Study - Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (MSE)', fontsize=11)
        ax2.set_title('Learning Rate Study - Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        lr_curves_path = save_path / 'training_curves_lr_study.png'
        plt.savefig(lr_curves_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved learning rate study curves: {lr_curves_path}")
        plt.close()
    
    # Plot activation function study curves
    if activation_exp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp_name, metrics in activation_exp.items():
            if 'train_losses' in metrics and 'val_losses' in metrics:
                epochs = range(1, len(metrics['train_losses']) + 1)
                clean = '_'.join(exp_name.split('_')[:-2])
                parts = clean.split('_')
                short_name = f"{parts[-1]}"  # activation function name
                
                ax1.plot(epochs, metrics['train_losses'], alpha=0.7, label=short_name, linewidth=2)
                ax2.plot(epochs, metrics['val_losses'], alpha=0.7, label=short_name, linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Activation Function Study - Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss (MSE)', fontsize=11)
        ax2.set_title('Activation Function Study - Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        activation_curves_path = save_path / 'training_curves_activation_study.png'
        plt.savefig(activation_curves_path, dpi=300, bbox_inches='tight')
        print(f"OK: Saved activation study curves: {activation_curves_path}")
        plt.close()


def export_csv_summary(results: Dict, save_path: Path = None):
    """Export results as CSV"""
    if save_path is None:
        save_path = DEFAULT_EXPERIMENTS_DIR / 'summary.csv'
    
    if not results:
        return
    
    rows = []
    for exp_name, metrics in results.items():
        rows.append({
            'experiment_name': exp_name,
            'config_file': metrics.get('config_file', 'N/A'),
            'epochs_trained': metrics.get('epochs_trained', 0),
            'best_val_loss': metrics.get('best_val_loss', 0),
            'test_mse': metrics.get('test_mse', 0),
            'test_rmse': metrics.get('test_rmse', 0),
            'test_loss': metrics.get('test_loss', 0),
            'timestamp': metrics.get('timestamp', 'N/A'),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"OK: Exported CSV: {save_path}")


def find_best_model(results: Dict) -> tuple:
    """Find the best performing model"""
    if not results:
        return None, None
    
    best_model = min(results.items(), key=lambda x: x[1].get('test_rmse', float('inf')))
    return best_model[0], best_model[1].get('test_rmse')


def main():
    print("\n" + "=" * 80)
    print("FNN EXPERIMENT RESULT ANALYSIS TOOL")
    print("=" * 80 + "\n")
    
    # Load results
    print("Loading experiment results...\n")
    results = load_experiment_results()
    
    if not results:
        print("\nERROR: No experiment results found")
        print("TIP: Run: python train_multiple_configs.py")
        return
    
    print(f"\nOK: Loaded results from {len(results)} experiments\n")
    
    # Print summary
    print_results_summary(results)
    
    # Find best model
    best_exp, best_rmse = find_best_model(results)
    if best_exp:
        print(f"\nBEST EXPERIMENT: {best_exp}")
        print(f"   Test RMSE: {best_rmse:.6f}")
    
    # Generate visualizations
    print("\nGenerating analysis charts...\n")
    try:
        import pandas as pd
        plot_comparison(results)
        plot_training_curves_overlay(results)
        export_csv_summary(results)
        print("\nOK: All analysis complete!")
    except Exception as e:
        print(f"WARNING: Some visualization generation failed: {e}")
    
    print(f"\nAnalysis results saved to: {DEFAULT_COMPARISON_DIR}/")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
