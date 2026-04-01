# FNN回归实验 - 多配置训练系统

本项目为前馈神经网络（FNN）回归任务提供了完整的多配置自动化训练、结果管理与分析工具，支持网络深度、学习率、激活函数等多种实验。

## 核心功能

- 多配置自动化训练：一条命令批量运行多个实验配置
- 自动保存模型权重、训练日志、图表及指标
- 智能可视化：自动生成训练曲线和对比图
- 灵活的YAML配置系统，支持基础配置+差异化实验配置
- 详细日志记录与早停机制
- 支持CPU/GPU自动切换

## 快速开始

1. 安装依赖
   - `pip install -r requirements.txt`
   - 或运行 `check_env.py` 检查环境

2. 运行全部实验
   - `python train_multiple_configs.py`

3. 运行单个实验
   - `python train_single_config.py --config FNN_Regression_Diabetes/configs/exp_configs/depth_study/shallow_network.yaml`

4. 分析实验结果
   - `python analyze_results.py`

## 项目结构

```
FNN_Regression_Diabetes/
├── configs/
│   ├── base_config.yaml
│   └── exp_configs/
│       ├── depth_study/      # 深度研究
│       ├── lr_study/         # 学习率研究
│       └── activation_study/ # 激活函数研究
├── src/
│   ├── model.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── train_multiple_configs.py
│   ├── train_single_config.py
│   └── analyze_results.py
├── experiments/              # 实验输出
├── reports/
│   └── 实验报告_FNN回归.md
└── requirements.txt
```

## 配置说明

### 基础配置（base_config.yaml）
包含所有实验的默认参数：

```yaml
model:
  input_dim: 10
  hidden_layers: [64, 32]
  activation: relu
training:
  learning_rate: 0.001
  epochs: 150
  optimizer: Adam
```

### 实验配置（exp_configs/*）
只需指定与基础配置不同的部分。例如：

```yaml
# depth_study/shallow_network.yaml
model:
  hidden_layers: [64]
```

激活函数实验配置示例：

```yaml
# activation_study/relu_activation.yaml
model:
  activation: relu
# 其他激活函数只需修改 activation 字段
```

## 主要脚本说明

- `train_multiple_configs.py`：自动扫描所有配置并批量训练
- `train_single_config.py`：运行单个配置的训练
- `analyze_results.py`：分析所有实验结果，生成对比图和CSV
- `check_env.py`：环境依赖检查

## 常见问题

- 如何添加新实验？
  - 在 `configs/exp_configs/` 下新建yaml文件即可，脚本会自动发现
- 如何使用GPU？
  - 默认自动检测。强制CPU：`python train_multiple_configs.py --device cpu`
- 实验结果会覆盖吗？
  - 不会，每次实验自动生成唯一目录
- 如何查看详细日志？
  - 查看 `experiments/*/logs/training.log`

## 实验输出说明

每个实验自动生成独立目录，包含：

```
experiments/
└── activation_study_relu_20260331_170457/
    ├── models/best_model.pt
    ├── logs/training.log
    ├── figures/training_history.png
    ├── config_used.yaml
    └── metrics.json
```

## 参考文档

- `深度学习基础实验1.md`：实验任务与分析方向
- `FNN_Regression_Diabetes/README.md`：模型与实现说明
- `reports/实验报告_FNN回归.md`：实验分析报告

## 建议流程

1. 检查环境：`python check_env.py`
2. 配置实验：编辑 `configs/exp_configs/*.yaml`
3. 运行训练：`python train_multiple_configs.py`
4. 分析结果：`python analyze_results.py`
5. 生成报告：查看 `experiments/*/metrics.json` 和图表
