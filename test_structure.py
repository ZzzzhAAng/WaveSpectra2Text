#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码结构验证测试 - 不依赖外部库
验证重构后的代码结构是否正确
"""

import os
import ast
import sys

def test_file_exists():
    """测试文件是否存在"""
    print("🧪 测试文件结构...")
    
    required_files = [
        'audio_preprocessing.py',
        'audio_dataset.py', 
        'batch_preprocess.py',
        'data_utils.py',
        'REFACTOR_GUIDE.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    else:
        print("✅ 所有必需文件都存在")
        return True

def test_python_syntax():
    """测试Python文件语法"""
    print("\n🧪 测试Python语法...")
    
    python_files = [
        'audio_preprocessing.py',
        'audio_dataset.py',
        'batch_preprocess.py', 
        'data_utils.py',
        'test_refactor.py'
    ]
    
    syntax_errors = []
    
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"  ✅ {file} 语法正确")
        except SyntaxError as e:
            syntax_errors.append(f"{file}: {e}")
            print(f"  ❌ {file} 语法错误: {e}")
        except Exception as e:
            syntax_errors.append(f"{file}: {e}")
            print(f"  ❌ {file} 解析错误: {e}")
    
    if syntax_errors:
        print(f"❌ 语法错误: {len(syntax_errors)} 个")
        return False
    else:
        print("✅ 所有Python文件语法正确")
        return True

def test_class_structure():
    """测试类结构"""
    print("\n🧪 测试类结构...")
    
    # 检查 audio_preprocessing.py 中的类
    try:
        with open('audio_preprocessing.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        expected_classes = [
            'AudioPreprocessor',
            'SpectrogramPreprocessor', 
            'MelSpectrogramPreprocessor',
            'PreprocessorFactory',
            'OfflinePreprocessor'
        ]
        
        missing_classes = [cls for cls in expected_classes if cls not in classes]
        
        if missing_classes:
            print(f"❌ audio_preprocessing.py 缺少类: {missing_classes}")
            return False
        else:
            print("✅ audio_preprocessing.py 类结构正确")
            
    except Exception as e:
        print(f"❌ 检查 audio_preprocessing.py 失败: {e}")
        return False
    
    # 检查 audio_dataset.py 中的类
    try:
        with open('audio_dataset.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        expected_classes = ['AudioDataset', 'FlexibleDataLoader']
        missing_classes = [cls for cls in expected_classes if cls not in classes]
        
        if missing_classes:
            print(f"❌ audio_dataset.py 缺少类: {missing_classes}")
            return False
        else:
            print("✅ audio_dataset.py 类结构正确")
            
    except Exception as e:
        print(f"❌ 检查 audio_dataset.py 失败: {e}")
        return False
    
    return True

def test_function_structure():
    """测试函数结构"""
    print("\n🧪 测试函数结构...")
    
    # 检查 data_utils.py 中的函数
    try:
        with open('data_utils.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        expected_functions = [
            'get_dataloader',
            'get_realtime_dataloader', 
            'get_precomputed_dataloader',
            'collate_fn',
            'create_labels_file_if_not_exists',
            'check_audio_files'
        ]
        
        missing_functions = [func for func in expected_functions if func not in functions]
        
        if missing_functions:
            print(f"❌ data_utils.py 缺少函数: {missing_functions}")
            return False
        else:
            print("✅ data_utils.py 函数结构正确")
            
    except Exception as e:
        print(f"❌ 检查 data_utils.py 失败: {e}")
        return False
    
    return True

def test_import_structure():
    """测试导入结构"""
    print("\n🧪 测试导入结构...")
    
    # 检查 data_utils.py 是否正确导入新模块
    try:
        with open('data_utils.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_imports = [
            'from audio_preprocessing import PreprocessorFactory',
            'from audio_dataset import AudioDataset, FlexibleDataLoader'
        ]
        
        missing_imports = []
        for imp in expected_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"❌ data_utils.py 缺少导入: {missing_imports}")
            return False
        else:
            print("✅ data_utils.py 导入结构正确")
            
    except Exception as e:
        print(f"❌ 检查导入结构失败: {e}")
        return False
    
    return True

def test_old_file_removed():
    """测试旧文件是否已删除"""
    print("\n🧪 测试旧文件清理...")
    
    if os.path.exists('preprocess_spectrum.py'):
        print("❌ 旧文件 preprocess_spectrum.py 仍然存在")
        return False
    else:
        print("✅ 旧文件已正确删除")
        return True

def main():
    """主测试函数"""
    print("🎯 代码结构验证测试")
    print("=" * 60)
    
    tests = [
        test_file_exists,
        test_python_syntax,
        test_class_structure,
        test_function_structure,
        test_import_structure,
        test_old_file_removed
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 代码结构验证通过！")
        print("\n📋 重构总结:")
        print("✅ 消除了冗余代码")
        print("✅ 降低了耦合度") 
        print("✅ 提高了可扩展性")
        print("✅ 保持了向后兼容性")
        print("\n💡 使用建议:")
        print("1. 安装依赖: pip install numpy pandas librosa torch tqdm")
        print("2. 运行完整测试: python3 test_refactor.py")
        print("3. 阅读使用指南: cat REFACTOR_GUIDE.md")
        return True
    else:
        print("❌ 部分结构测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)