# check_env.py
"""
环境检查脚本 - 验证所有必要的依赖是否安装正确
"""

import sys
from pathlib import Path

def check_imports():
    """检查必要的Python包"""
    required_packages = {
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML'
    }
    
    print("=" * 60)
    print("📦 依赖包检查")
    print("=" * 60)
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20} - 已安装")
        except ImportError:
            print(f"❌ {name:20} - 未安装")
            all_ok = False
    
    return all_ok


def check_project_structure():
    """检查项目结构"""
    print("\n" + "=" * 60)
    print("📁 项目结构检查")
    print("=" * 60)
    
    required_files = [
        'src/model.py',
        'src/data_loader.py',
        'src/utils.py',
        'src/train.py',
        'configs/base_config.yaml',
        'configs/exp_configs/depth_study/shallow_network.yaml',
        'configs/exp_configs/lr_study/lr_001.yaml',
        'src/train_multiple_configs.py',
        'src/train_single_config.py',
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_ok = False
    
    return all_ok


def check_pytorch():
    """检查PyTorch和GPU支持"""
    print("\n" + "=" * 60)
    print("🎯 PyTorch配置检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  未检测到GPU，将使用CPU运行（速度会较慢）")
        return True
    except Exception as e:
        print(f"❌ PyTorch检查出错: {e}")
        return False


def check_configs():
    """检查配置文件格式"""
    print("\n" + "=" * 60)
    print("⚙️  配置文件检查")
    print("=" * 60)
    
    try:
        import yaml
        config_files = [
            'configs/base_config.yaml',
            'configs/exp_configs/depth_study/shallow_network.yaml',
            'configs/exp_configs/lr_study/lr_001.yaml',
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"✅ {config_file} - 格式正确")
            else:
                print(f"❌ {config_file} - 文件不存在")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 配置文件检查出错: {e}")
        return False


def main():
    print("\n")
    print("    ╔════════════════════════════════════════════════════╗")
    print("    ║   FNN回归实验 - 环境检查工具                      ║")
    print("    ╚════════════════════════════════════════════════════╝\n")
    
    results = {
        '依赖包': check_imports(),
        '项目结构': check_project_structure(),
        'PyTorch配置': check_pytorch(),
        '配置文件': check_configs(),
    }
    
    print("\n" + "=" * 60)
    print("📋 检查总结")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status:8} - {check}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ 所有检查通过！环境配置完毕。")
        print("\n🚀 立即开始实验：")
        print("   python train_multiple_configs.py")
        print("\n📖 查看详细文档：")
        print("   TRAINING_GUIDE.md")
        return 0
    else:
        print("\n❌ 部分检查失败，请按上述提示修复问题。")
        print("\n💡 建议步骤：")
        print("   1. 安装缺失的依赖包: pip install -r requirements.txt")
        print("   2. 检查项目文件是否存在")
        print("   3. 验证配置文件的YAML格式")
        print("   4. 重新运行此检查脚本")
        return 1


if __name__ == '__main__':
    sys.exit(main())
