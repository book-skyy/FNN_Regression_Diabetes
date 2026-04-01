# train_multiple_configs.py
"""
多配置训练脚本
该脚本支持自动加载多个配置文件，对每个配置运行完整的训练流程，
并将所有实验结果（模型权重、训练日志、图表）保存到experiments目录
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

# 添加src目录到路径（当前文件已在src目录中）
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from model import FNN
from data_loader import get_dataloaders
from utils import (
    load_yaml_config,
    merge_configs,
    setup_logger,
    set_seed,
    get_device,
    plot_training_history,
    save_training_metrics,
    save_config,
    create_experiment_directory
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger
) -> dict:
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        dict: 训练指标（损失历史等）
    """
    criterion = nn.MSELoss()
    
    # 根据配置选择优化器
    optimizer_name = config['training'].get('optimizer', 'Adam')
    lr = config['training']['learning_rate']
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = config['training']['epochs']
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20  # 早停耐心值
    patience_counter = 0
    
    logger.info(f"开始训练，共{epochs}个epoch")
    logger.info(f"优化器: {optimizer_name}, 学习率: {lr}")
    
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                num_batches += 1
        
        val_loss /= num_batches
        val_losses.append(val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"触发早停条件 (连续{patience}个epoch无改进)")
                break
    
    logger.info("训练完成！")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'epochs_trained': len(train_losses)
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    logger
) -> dict:
    """
    评估模型在测试集上的性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        dict: 评估指标
    """
    criterion = nn.MSELoss()
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            test_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            num_batches += 1
    
    test_loss /= num_batches
    
    # 计算MSE和RMSE
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
    rmse = np.sqrt(mse)
    
    logger.info(f"测试集评估 - MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return {
        'test_mse': float(mse),
        'test_rmse': float(rmse),
        'test_loss': float(test_loss)
    }


def run_single_experiment(
    config_path: Path,
    base_config: dict,
    output_base_dir: Path,
    device: torch.device,
) -> bool:
    """
    运行单个实验
    
    Args:
        config_path: 实验配置文件路径
        base_config: 基础配置
        output_base_dir: 输出基础目录
        device: 计算设备
        
    Returns:
        bool: 是否成功完成
    """
    try:
        # 加载实验配置
        exp_config = load_yaml_config(str(config_path))
        
        # 合并配置
        config = merge_configs(base_config, exp_config)
        
        # 获取实验名称和创建实验目录
        exp_name = config.get('experiment_name', config_path.stem)
        exp_dir = create_experiment_directory(output_base_dir, exp_name)
        
        # 设置日志
        logger = setup_logger(exp_dir / "logs" / "training.log")
        logger.info("=" * 60)
        logger.info(f"开始实验: {exp_name}")
        logger.info(f"配置文件: {config_path}")
        logger.info(f"实验描述: {config.get('description', 'N/A')}")
        logger.info(f"输出目录: {exp_dir}")
        logger.info("=" * 60)
        
        # 保存配置和对外可见的信息
        save_config(config, exp_dir / "config_used.yaml")
        
        # 设置随机种子
        seed = config['runtime'].get('seed', 42)
        set_seed(seed)
        logger.info(f"随机种子: {seed}")
        
        # 加载数据
        logger.info("加载数据...")
        batch_size = config['data']['batch_size']
        test_size = config['data'].get('test_size', 0.1)
        validation_size = config['data'].get('validation_size', 0.1)
        random_state = config['data'].get('random_state', 42)
        
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
        
        # 构建模型
        logger.info("构建模型...")
        model_config = config['model']
        model = FNN(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            hidden_layers=model_config['hidden_layers'],
            activation=model_config.get('activation', 'relu'),
            dropout_rate=model_config.get('dropout_rate', 0.0)
        )
        
        logger.info(f"模型架构:")
        logger.info(f"  隐藏层: {model_config['hidden_layers']}")
        logger.info(f"  激活函数: {model_config.get('activation', 'relu')}")
        logger.info(f"  Dropout: {model_config.get('dropout_rate', 0.0)}")
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
        
        # 训练模型
        train_metrics = train_model(
            model, train_loader, val_loader, config, device, logger
        )
        
        # 加载最佳模型进行评估
        best_model_path = exp_dir / "models" / "best_model.pt"
        test_metrics = evaluate_model(model, test_loader, device, logger)
        
        # 保存最佳模型
        logger.info(f"保存模型到 {best_model_path}")
        torch.save(model.state_dict(), best_model_path)
        
        # 合并所有指标
        all_metrics = {
            **train_metrics,
            **test_metrics,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': exp_name,
            'config_file': str(config_path)
        }
        
        # 保存指标
        metrics_path = exp_dir / "metrics.json"
        save_training_metrics(all_metrics, metrics_path)
        logger.info(f"保存指标到 {metrics_path}")
        
        # 绘制训练历史
        plot_path = exp_dir / "figures" / "training_history.png"
        plot_training_history(
            train_metrics['train_losses'],
            train_metrics['val_losses'],
            plot_path,
            title=f"{exp_name} - Training History"
        )
        logger.info(f"保存图表到 {plot_path}")
        
        logger.info("=" * 60)
        logger.info(f"实验 '{exp_name}' 完成！")
        logger.info(f"最佳验证损失: {train_metrics['best_val_loss']:.6f}")
        logger.info(f"测试集RMSE: {test_metrics['test_rmse']:.6f}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"FAILED: {config_path}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="多配置训练脚本 - 自动运行多个深度学习实验"
    )
    
    # 计算相对于脚本文件的路径 (使用绝对路径)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    default_configs = str(project_dir / 'configs' / 'exp_configs')
    default_base_config = str(project_dir / 'configs' / 'base_config.yaml')
    default_output = str(project_dir / 'experiments')
    
    parser.add_argument(
        '--configs',
        type=str,
        default=default_configs,
        help='配置文件或目录路径'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default=default_base_config,
        help='基础配置文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=default_output,
        help='实验输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    base_config_path = Path(args.base_config)
    configs_path = Path(args.configs)
    output_dir = Path(args.output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载基础配置
    print(f"Loading base config: {base_config_path}")
    if not base_config_path.exists():
        print(f"ERROR: Base config file not found: {base_config_path}")
        return
    
    base_config = load_yaml_config(str(base_config_path))
    
    # 获取计算设备
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # 收集所有配置文件
    config_files = []
    if configs_path.is_file():
        config_files = [configs_path]
    elif configs_path.is_dir():
        config_files = sorted(configs_path.glob('**/*.yaml'))
    else:
        print(f"ERROR: Config path not found: {configs_path}")
        return
    
    if not config_files:
        print(f"ERROR: No config files found: {configs_path}")
        return
    
    print(f"Found {len(config_files)} experiment configs")
    for cf in config_files:
        try:
            print(f"   - {cf.relative_to(project_dir)}")
        except ValueError:
            print(f"   - {cf}")
    
    # 运行实验
    print("\n" + "=" * 60)
    print("Starting experiments...")
    print("=" * 60 + "\n")
    
    results = []
    for idx, config_file in enumerate(config_files, 1):
        print(f"\n[{idx}/{len(config_files)}] 运行: {config_file.name}")
        success = run_single_experiment(
            config_file,
            base_config,
            output_dir,
            device
        )
        results.append({
            'config': str(config_file),
            'success': success
        })
    
    # Output summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"SUCCESS: {successful}/{len(results)}")
    
    if successful < len(results):
        print("\nFAILED EXPERIMENTS:")
        for r in results:
            if not r['success']:
                print(f"   - {r['config']}")
    
    print(f"\nALL RESULTS SAVED TO: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
