#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动更新系统测试脚本
"""

import os
import sys
from pathlib import Path

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试自动更新系统基本功能")
    print("=" * 50)
    
    # 检查必要文件
    required_files = [
        'data/audio',
        'data/labels.csv',
        'vocab.py'
    ]
    
    print("📁 检查必要文件:")
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        print(f"  {file_path}: {'✅' if exists else '❌'}")
        
        if not exists and file_path == 'data/labels.csv':
            print("    💡 提示: 运行 python3 setup_data.py 创建标签文件")
    
    # 检查vocab.py中的词汇表
    print("\n📚 检查词汇表:")
    try:
        with open('vocab.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找中文数字
        chinese_numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        found_numbers = []
        
        for num in chinese_numbers:
            if f"'{num}'" in content:
                found_numbers.append(num)
        
        print(f"  当前词汇表包含: {found_numbers}")
        print(f"  词汇数量: {len(found_numbers)}/10")
        
    except Exception as e:
        print(f"  ❌ 读取vocab.py失败: {e}")
    
    # 检查labels.csv中的标签
    print("\n🏷️  检查标签文件:")
    try:
        if Path('data/labels.csv').exists():
            import pandas as pd
            df = pd.read_csv('data/labels.csv')
            
            if 'label' in df.columns:
                unique_labels = set(df['label'].dropna().unique())
                print(f"  标签文件中的标签: {sorted(unique_labels)}")
                print(f"  标签数量: {len(unique_labels)}")
                
                # 检查是否有新标签需要添加到词汇表
                vocab_labels = set(found_numbers)
                new_labels = unique_labels - vocab_labels
                
                if new_labels:
                    print(f"  🆕 需要添加到词汇表的新标签: {new_labels}")
                else:
                    print(f"  ✅ 所有标签都已在词汇表中")
            else:
                print(f"  ❌ 标签文件缺少'label'列")
        else:
            print(f"  ❌ 标签文件不存在")
            
    except Exception as e:
        print(f"  ❌ 读取标签文件失败: {e}")
    
    # 检查特征目录
    print("\n📊 检查特征目录:")
    features_dir = Path('data/features')
    if features_dir.exists():
        feature_files = list(features_dir.glob('*.npy'))
        index_file = features_dir / 'spectrum_index.csv'
        
        print(f"  特征目录: ✅")
        print(f"  特征文件数量: {len(feature_files)}")
        print(f"  索引文件: {'✅' if index_file.exists() else '❌'}")
        
        if feature_files:
            print(f"  示例特征文件: {feature_files[0].name}")
    else:
        print(f"  特征目录: ❌ (将自动创建)")
    
    print("\n" + "=" * 50)
    print("✅ 基本功能测试完成")
    
    return True


def simulate_label_update():
    """模拟标签更新"""
    print("\n🔄 模拟标签文件更新")
    print("-" * 30)
    
    labels_file = Path('data/labels.csv')
    
    if not labels_file.exists():
        print("❌ 标签文件不存在，无法模拟更新")
        return False
    
    try:
        # 读取当前标签文件
        import pandas as pd
        df = pd.read_csv(labels_file)
        
        print(f"当前标签文件有 {len(df)} 条记录")
        
        # 检查是否可以添加新标签
        current_labels = set(df['label'].dropna().unique()) if 'label' in df.columns else set()
        all_chinese_numbers = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十'}
        
        missing_labels = all_chinese_numbers - current_labels
        
        if missing_labels:
            print(f"可以添加的标签: {missing_labels}")
            print("💡 要测试自动更新功能，可以:")
            print("   1. 在data/audio/目录添加新的音频文件")
            print("   2. 在data/labels.csv中添加对应的标签记录")
            print("   3. 运行: python3 sync_data.py")
        else:
            print("✅ 所有中文数字标签都已存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟更新失败: {e}")
        return False


def main():
    """主函数"""
    print("🤖 WaveSpectra2Text 自动更新系统测试")
    print("=" * 60)
    
    # 基本功能测试
    test_basic_functionality()
    
    # 模拟更新测试
    simulate_label_update()
    
    print("\n💡 使用说明:")
    print("1. 手动同步: python3 sync_data.py")
    print("2. 实时监控: python3 watch_data_changes.py")
    print("3. 强制重新处理: python3 sync_data.py --force")
    
    print("\n🎯 自动更新功能:")
    print("- ✅ 检测音频文件变化（新增/修改/删除）")
    print("- ✅ 检测标签文件更新")
    print("- ✅ 自动更新vocab.py词汇表")
    print("- ✅ 自动生成预处理特征")
    print("- ✅ 自动更新特征索引文件")


if __name__ == "__main__":
    main()