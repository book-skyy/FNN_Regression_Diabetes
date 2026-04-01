# src/data_loader.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def get_dataloaders(
    batch_size: int,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载、预处理Diabetes数据集，并创建PyTorch DataLoaders。

    Args:
        batch_size (int): 训练、验证和测试的批次大小。
        test_size (float): 从总数据中划分出的测试集比例。
        validation_size (float): 从剩余的训练数据中划分出的验证集比例。
        random_state (int): 用于确保数据划分可复现的随机种子。

    Returns:
        A tuple containing:
        - train_loader (DataLoader): 训练集的数据加载器。
        - val_loader (DataLoader): 验证集的数据加载器。
        - test_loader (DataLoader): 测试集的数据加载器。
    """
    # 1. 加载原始数据
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # 2. 划分训练集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. 从剩余数据中划分训练集和验证集
    # 注意：validation_size 是相对于原始训练集大小的比例
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )

    # 4. 数据标准化 (Standardization)
    # 使用训练集计算均值和标准差，并应用到所有数据集上
    # 这是为了防止测试集和验证集的信息泄露到训练过程中
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5. 转换为PyTorch Tensors
    # 需要将y的形状从 (n,) 转换为 (n, 1) 以匹配模型输出
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 6. 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print("数据加载与预处理完成:")
    print(f" - 训练集样本数: {len(train_dataset)}")
    print(f" - 验证集样本数: {len(val_dataset)}")
    print(f" - 测试集样本数: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 这是一个用于测试脚本是否能独立运行的简单示例
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    
    # 检查一个批次的数据形状是否正确
    for X_batch, y_batch in train_loader:
        print("\n一个训练批次的数据形状:")
        print(f" - X_batch shape: {X_batch.shape}") # 应该是 [32, 10]
        print(f" - y_batch shape: {y_batch.shape}") # 应该是 [32, 1]
        break

