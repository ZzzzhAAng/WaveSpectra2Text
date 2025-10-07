#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖包检查脚本
检查项目所需的依赖包是否已安装
"""

import sys
from pathlib import Path


def check_dependency(package_name, import_name=None):
    """检查单个依赖包"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "✅"
    except ImportError as e:
        return False, f"❌ {e}"


def check_all_dependencies():
    """检查所有依赖包"""
    print("🔍 检查项目依赖包")
    print("=" * 50)
    
    # 必需依赖
    required_deps = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('librosa', 'librosa'),
        ('tqdm', 'tqdm'),
        ('scipy', 'scipy'),
        ('soundfile', 'soundfile')
    ]
    
    # 可选依赖
    optional_deps = [
        ('matplotlib', 'matplotlib'),
        ('tensorboard', 'tensorboard')
    ]
    
    print("📦 必需依赖:")
    missing_required = []
    for package, import_name in required_deps:
        success, status = check_dependency(package, import_name)
        print(f"  {package}: {status}")
        if not success:
            missing_required.append(package)
    
    print(f"\n📦 可选依赖:")
    missing_optional = []
    for package, import_name in optional_deps:
        success, status = check_dependency(package, import_name)
        print(f"  {package}: {status}")
        if not success:
            missing_optional.append(package)
    
    # 总结
    print(f"\n📊 依赖检查结果:")
    print(f"  必需依赖: {len(required_deps) - len(missing_required)}/{len(required_deps)} 已安装")
    print(f"  可选依赖: {len(optional_deps) - len(missing_optional)}/{len(optional_deps)} 已安装")
    
    if missing_required:
        print(f"\n❌ 缺失必需依赖: {missing_required}")
        print(f"安装命令: pip install {' '.join(missing_required)}")
        return False
    else:
        print(f"\n✅ 所有必需依赖已安装")
        if missing_optional:
            print(f"💡 可选安装: pip install {' '.join(missing_optional)}")
        return True


def test_basic_functionality():
    """测试基础功能（不依赖外部库）"""
    print(f"\n🧪 测试基础功能")
    print("-" * 30)
    
    try:
        # 测试vocab
        from vocab import vocab
        print(f"✅ 词汇表: {vocab.vocab_size}个词汇")
        
        # 测试编码解码
        test_text = "一二三"
        encoded = vocab.encode(test_text)
        decoded = vocab.decode(encoded)
        print(f"✅ 编码解码: {test_text} -> {decoded}")
        
        # 测试文件存在性
        required_files = [
            'data/labels.csv',
            'data/audio',
            'vocab.py',
            'model.py',
            'dual_input_inference.py'
        ]
        
        print(f"✅ 文件检查:")
        for file_path in required_files:
            exists = Path(file_path).exists()
            print(f"  {file_path}: {'✅' if exists else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🤖 WaveSpectra2Text 依赖检查")
    print("=" * 60)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python版本过低，需要3.7+")
        return
    else:
        print("✅ Python版本符合要求")
    
    # 检查依赖包
    deps_ok = check_all_dependencies()
    
    # 测试基础功能
    basic_ok = test_basic_functionality()
    
    print(f"\n" + "=" * 60)
    if deps_ok and basic_ok:
        print("✅ 系统检查通过，可以正常使用")
        print("\n💡 下一步:")
        print("  1. 运行数据设置: python3 setup_data.py")
        print("  2. 开始训练: python3 train_at_different_scales/train_scale_1.py")
        print("  3. 运行推理: python3 dual_input_inference.py --help")
    else:
        print("❌ 系统检查未通过")
        if not deps_ok:
            print("  - 请安装缺失的依赖包")
        if not basic_ok:
            print("  - 请检查基础文件和配置")


if __name__ == "__main__":
    main()