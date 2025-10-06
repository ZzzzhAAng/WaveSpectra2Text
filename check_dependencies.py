#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖检查脚本
验证所需的包是否正确安装
"""

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'), 
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
        ('tqdm', 'TQDM'),
        ('scipy', 'SciPy')
    ]
    
    optional_packages = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('tensorboard', 'TensorBoard'),
        ('matplotlib', 'Matplotlib')
    ]
    
    print("🔍 检查必需依赖...")
    missing_required = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} - 已安装")
        except ImportError:
            print(f"❌ {name} - 未安装")
            missing_required.append(package)
    
    print("\n🔍 检查可选依赖...")
    missing_optional = []
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name} - 已安装")
        except ImportError:
            print(f"⚠️  {name} - 未安装 (可选)")
            missing_optional.append(package)
    
    print("\n" + "="*50)
    
    if not missing_required:
        print("🎉 所有必需依赖都已安装！")
        print("现在可以运行数据预处理脚本了:")
        print("python batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv")
        return True
    else:
        print("❌ 缺少必需依赖:")
        for package in missing_required:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_required)}")
        return False

if __name__ == "__main__":
    check_dependencies()