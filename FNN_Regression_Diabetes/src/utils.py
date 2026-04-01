# src/utils.py

import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path (str): YAML配置文件的路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    将override_config深度合并到base_config中
    
    Args:
        base_config (Dict): 基础配置
        override_config (Dict): 覆盖配置
        
    Returns:
        Dict: 合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def setup_logger(log_path: Path, logger_name: str = "FNN_Training") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_path (Path): 日志文件保存路径
        logger_name (str): 日志记录器名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed: int):
    """
    设置全局随机种子以确保可复现性
    
    Args:
        seed (int): 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device_str (str): 设备字符串 ("auto", "cuda", "cpu")
        
    Returns:
        torch.device: PyTorch设备对象
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    return device


def plot_training_history(
    train_losses: list,
    val_losses: list,
    save_path: Path,
    title: str = "Training History"
):
    """
    绘制训练历史图表
    
    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        save_path (Path): 图表保存路径
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_metrics(
    metrics: Dict[str, Any],
    save_path: Path
):
    """
    保存训练指标为JSON文件
    
    Args:
        metrics (Dict[str, Any]): 训练指标字典
        save_path (Path): 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def save_config(
    config: Dict[str, Any],
    save_path: Path
):
    """
    保存配置为YAML文件
    
    Args:
        config (Dict[str, Any]): 配置字典
        save_path (Path): 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def create_experiment_directory(base_dir: Path, experiment_name: str) -> Path:
    """
    创建实验输出目录
    
    Args:
        base_dir (Path): 基础输出目录
        experiment_name (str): 实验名称
        
    Returns:
        Path: 创建的实验目录
    """
    # 使用时间戳避免重名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    
    return exp_dir
