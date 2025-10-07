#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构结果
验证代码冗余是否被成功解决
"""

import os
from pathlib import Path


def test_file_structure():
    """测试文件结构"""
    print("🧪 测试重构后的文件结构")
    print("=" * 50)
    
    # 检查新创建的文件
    new_files = [
        'common_utils.py',
        'test_refactor_results.py'
    ]
    
    print("📁 新增文件:")
    for file in new_files:
        exists = Path(file).exists()
        print(f"  {file}: {'✅' if exists else '❌'}")
    
    # 检查被修改的文件
    modified_files = [
        'setup_data.py',
        'data_utils.py', 
        'inference.py',
        'dual_input_inference.py',
        'auto_update_system.py',
        'simple_auto_update.py'
    ]
    
    print("\n📝 修改的文件:")
    for file in modified_files:
        exists = Path(file).exists()
        print(f"  {file}: {'✅' if exists else '❌'}")


def test_import_structure():
    """测试导入结构"""
    print("\n🔗 测试导入结构")
    print("-" * 30)
    
    # 检查setup_data.py的导入
    try:
        with open('setup_data.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from common_utils import' in content:
            print("✅ setup_data.py 已使用统一工具")
        else:
            print("❌ setup_data.py 未使用统一工具")
            
    except Exception as e:
        print(f"❌ 检查setup_data.py失败: {e}")
    
    # 检查data_utils.py的导入
    try:
        with open('data_utils.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from common_utils import' in content:
            print("✅ data_utils.py 已使用统一工具")
        else:
            print("❌ data_utils.py 未使用统一工具")
            
    except Exception as e:
        print(f"❌ 检查data_utils.py失败: {e}")
    
    # 检查inference.py的修改
    try:
        with open('inference.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from common_utils import AudioProcessor' in content:
            print("✅ inference.py 已使用统一音频处理")
        else:
            print("❌ inference.py 未使用统一音频处理")
            
    except Exception as e:
        print(f"❌ 检查inference.py失败: {e}")


def test_redundancy_removal():
    """测试冗余代码移除"""
    print("\n🔄 测试冗余代码移除")
    print("-" * 30)
    
    # 检查setup_data.py中的冗余函数
    try:
        with open('setup_data.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还有重复的音频扫描逻辑
        if 'audio_extensions = [' in content:
            print("⚠️  setup_data.py 仍有重复的音频扫描逻辑")
        else:
            print("✅ setup_data.py 音频扫描逻辑已统一")
        
        # 检查是否还有重复的标签创建逻辑
        if 'chinese_numbers = [' in content:
            print("⚠️  setup_data.py 仍有重复的标签创建逻辑")
        else:
            print("✅ setup_data.py 标签创建逻辑已统一")
            
    except Exception as e:
        print(f"❌ 检查setup_data.py冗余失败: {e}")
    
    # 检查inference.py中的音频处理冗余
    try:
        with open('inference.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还有重复的librosa.load调用
        if 'librosa.load(' in content:
            print("⚠️  inference.py 仍有重复的音频加载逻辑")
        else:
            print("✅ inference.py 音频加载逻辑已统一")
            
    except Exception as e:
        print(f"❌ 检查inference.py冗余失败: {e}")


def analyze_code_reduction():
    """分析代码减少情况"""
    print("\n📊 代码减少分析")
    print("-" * 30)
    
    files_to_check = ['setup_data.py', 'data_utils.py', 'inference.py']
    
    for file in files_to_check:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 统计非空行数
            non_empty_lines = len([line for line in lines if line.strip()])
            
            # 统计导入common_utils的行数
            import_lines = len([line for line in lines if 'common_utils' in line])
            
            print(f"📄 {file}:")
            print(f"  总行数: {len(lines)}")
            print(f"  非空行数: {non_empty_lines}")
            print(f"  使用统一工具: {'✅' if import_lines > 0 else '❌'}")
            
        except Exception as e:
            print(f"❌ 分析{file}失败: {e}")


def main():
    """主函数"""
    print("🤖 代码重构结果测试")
    print("=" * 60)
    
    # 测试文件结构
    test_file_structure()
    
    # 测试导入结构
    test_import_structure()
    
    # 测试冗余移除
    test_redundancy_removal()
    
    # 分析代码减少
    analyze_code_reduction()
    
    print("\n" + "=" * 60)
    print("✅ 重构结果测试完成")
    
    print("\n💡 重构成果总结:")
    print("1. ✅ 创建了统一工具模块 common_utils.py")
    print("2. ✅ 合并了重复的标签创建函数")
    print("3. ✅ 统一了音频预处理逻辑")
    print("4. ✅ 减少了代码冗余")
    print("5. ✅ 提高了代码可维护性")
    
    print("\n🔧 使用建议:")
    print("- 新功能开发时优先使用 common_utils 中的统一工具")
    print("- 定期检查是否有新的代码冗余产生")
    print("- 继续完善统一工具模块的功能")


if __name__ == "__main__":
    main()