# train_single_config.py
"""
单配置训练脚本
用于运行单个配置的训练，输出结果到experiments目录
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
    """训练模型"""
    criterion = nn.MSELoss()
    
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
    patience = 20
    patience_counter = 0
    
    logger.info(f"开始训练，共{epochs}个epoch")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"触发早停 (连续{patience}个epoch无改进)")
                break
    
    logger.info("训练完成！")
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'epochs_trained': len(train_losses)
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, logger) -> dict:
    """评估模型"""
    criterion = nn.MSELoss()
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
    
    test_loss /= len(test_loader)
    mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
    rmse = np.sqrt(mse)
    
    logger.info(f"测试集 - MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return {'test_mse': float(mse), 'test_rmse': float(rmse), 'test_loss': float(test_loss)}


def main():
    parser = argparse.ArgumentParser(description="Single Config Training Script")
    
    # Calculate paths relative to script location (use absolute paths)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    parser.add_argument('--config', type=str, required=True, help='Experiment config file path')
    parser.add_argument('--base-config', type=str, default=str(project_dir / 'configs' / 'base_config.yaml'), help='Base config')
    parser.add_argument('--output-dir', type=str, default=str(project_dir / 'experiments'), help='Output directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Compute device')
    
    args = parser.parse_args()
    
    # 路径处理
    config_path = Path(args.config)
    base_config_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config: {config_path}")
    base_config = load_yaml_config(str(base_config_path))
    exp_config = load_yaml_config(str(config_path))
    config = merge_configs(base_config, exp_config)
    
    # 设置设备和随机种子
    device = get_device(args.device)
    set_seed(config['runtime'].get('seed', 42))
    
    # 创建实验目录和日志
    exp_name = config.get('experiment_name', config_path.stem)
    exp_dir = create_experiment_directory(output_dir, exp_name)
    logger = setup_logger(exp_dir / "logs" / "training.log")
    
    logger.info(f"实验: {exp_name}")
    logger.info(f"描述: {config.get('description', 'N/A')}")
    
    # 加载数据和构建模型
    logger.info("加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['data']['batch_size'],
        test_size=config['data'].get('test_size', 0.1),
        validation_size=config['data'].get('validation_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
    
    logger.info("构建模型...")
    model = FNN(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        hidden_layers=config['model']['hidden_layers'],
        activation=config['model'].get('activation', 'relu'),
        dropout_rate=config['model'].get('dropout_rate', 0.0)
    )
    
    logger.info(f"隐藏层: {config['model']['hidden_layers']}")
    logger.info(f"参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    train_metrics = train_model(model, train_loader, val_loader, config, device, logger)
    
    # 评估
    test_metrics = evaluate_model(model, test_loader, device, logger)
    
    # 保存结果
    model_path = exp_dir / "models" / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")
    
    save_config(config, exp_dir / "config_used.yaml")
    all_metrics = {**train_metrics, **test_metrics, 'timestamp': datetime.now().isoformat()}
    save_training_metrics(all_metrics, exp_dir / "metrics.json")
    
    # 绘图
    plot_training_history(
        train_metrics['train_losses'],
        train_metrics['val_losses'],
        exp_dir / "figures" / "training_history.png",
        title=f"{exp_name} - Training History"
    )
    logger.info(f"结果已保存到: {exp_dir}")
    
    print(f"\nExperiment completed!")
    print(f"Output directory: {exp_dir}\n")


if __name__ == "__main__":
    main()
