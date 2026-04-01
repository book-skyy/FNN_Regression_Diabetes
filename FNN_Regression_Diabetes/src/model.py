# src/model.py

import torch
import torch.nn as nn
from typing import List, Dict

# Custom Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Map activation function names to PyTorch modules
ACTIVATION_MAP: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "swish": Swish
}

class FNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0
    ):
        """
        一个灵活的、可动态配置的前馈神经网络（FNN）模型。

        Args:
            input_dim (int): 输入特征的维度。
            output_dim (int): 输出的维度（对于回归任务通常是1）。
            hidden_layers (List[int]): 一个整数列表，其中每个整数代表一个隐藏层的神经元数量。
                                      例如: [64, 32] 表示一个64神经元的隐藏层后接一个32神经元的隐藏层。
            activation (str): 激活函数的名称 (小写)。必须是 ACTIVATION_MAP 中的键。
            dropout_rate (float): 在每个隐藏层后应用的Dropout比率。0表示不使用。
        """
        super(FNN, self).__init__()

        # 检查激活函数是否受支持
        if activation.lower() not in ACTIVATION_MAP:
            raise ValueError(
                f"不支持的激活函数: '{activation}'. "
                f"可用选项: {list(ACTIVATION_MAP.keys())}"
            )

        layers = []
        current_dim = input_dim

        # 动态构建隐藏层
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(ACTIVATION_MAP[activation.lower()]())
            
            # 如果指定了dropout_rate，则添加Dropout层
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            
            current_dim = h_dim # 更新当前维度，为下一层做准备

        # 添加最终的输出层（通常是线性的，不加激活函数，因为是回归任务）
        layers.append(nn.Linear(current_dim, output_dim))

        # 使用 nn.Sequential 将所有层打包成一个网络
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义模型的前向传播路径。"""
        return self.network(x)

if __name__ == '__main__':
    # 这是一个用于测试模型构建是否正常的简单示例
    print("测试模型动态构建功能...")

    # 示例1：一个浅层网络
    config_shallow = {
        "input_dim": 10,
        "output_dim": 1,
        "hidden_layers": [32],
        "activation": "relu"
    }
    shallow_model = FNN(**config_shallow)
    print("\n浅层模型结构:")
    print(shallow_model)
    # Output:
    # FNN(
    #   (network): Sequential(
    #     (0): Linear(in_features=10, out_features=32, bias=True)
    #     (1): ReLU()
    #     (2): Linear(in_features=32, out_features=1, bias=True)
    #   )
    # )

    # 示例2：一个带有Dropout的深层网络
    config_deep_dropout = {
        "input_dim": 10,
        "output_dim": 1,
        "hidden_layers": [128, 64, 32],
        "activation": "tanh",
        "dropout_rate": 0.5
    }
    deep_model = FNN(**config_deep_dropout)
    print("\n带Dropout的深层模型结构:")
    print(deep_model)
    # Output:
    # FNN(
    #   (network): Sequential(
    #     (0): Linear(in_features=10, out_features=128, bias=True)
    #     (1): Tanh()
    #     (2): Dropout(p=0.5, inplace=False)
    #     (3): Linear(in_features=128, out_features=64, bias=True)
    #     ...
    #   )
    # )

    # 测试前向传播
    dummy_input = torch.randn(5, 10) # 5个样本，10个特征
    output = deep_model(dummy_input)
    print(f"\n模型输入形状: {dummy_input.shape}")
    print(f"模型输出形状: {output.shape}") # 应该是 [5, 1]

