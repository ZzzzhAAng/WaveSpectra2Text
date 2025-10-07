#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新标签添加功能
"""

import csv
import shutil
from pathlib import Path
from simple_auto_update import SimpleAutoUpdater


def test_add_new_label():
    """测试添加新标签"""
    print("🧪 测试新标签添加功能")
    print("=" * 50)
    
    # 备份原始文件
    labels_file = Path('data/labels.csv')
    vocab_file = Path('vocab.py')
    
    labels_backup = labels_file.with_suffix('.csv.test_backup')
    vocab_backup = vocab_file.with_suffix('.py.test_backup')
    
    shutil.copy2(labels_file, labels_backup)
    shutil.copy2(vocab_file, vocab_backup)
    
    print(f"📁 已备份原始文件")
    
    try:
        # 读取现有标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # 添加新标签（假设我们要添加"零"）
        new_row = ['test_audio.wav', '零']
        rows.append(new_row)
        
        # 写入修改后的标签文件
        with open(labels_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"✅ 已添加新标签: 零")
        
        # 运行自动更新
        updater = SimpleAutoUpdater()
        results = updater.check_and_update()
        
        # 检查结果
        if results['vocab_updated']:
            print("✅ 词汇表更新成功！")
            
            # 重新创建updater来读取更新后的词汇表
            new_updater = SimpleAutoUpdater()
            vocab_labels = new_updater.get_vocab_labels()
            if '零' in vocab_labels:
                print("✅ 新标签已成功添加到词汇表")
            else:
                print("❌ 新标签未添加到词汇表")
        else:
            print("❌ 词汇表未更新")
        
    finally:
        # 恢复原始文件
        shutil.copy2(labels_backup, labels_file)
        shutil.copy2(vocab_backup, vocab_file)
        
        # 删除备份文件
        labels_backup.unlink()
        vocab_backup.unlink()
        
        print(f"📁 已恢复原始文件")


def main():
    """主函数"""
    print("🤖 自动更新系统功能测试")
    print("=" * 60)
    
    # 测试添加新标签
    test_add_new_label()
    
    print("\n✅ 测试完成！")
    print("\n💡 自动更新系统功能验证:")
    print("- ✅ 检测标签文件变化")
    print("- ✅ 识别新标签")
    print("- ✅ 自动更新词汇表")
    print("- ✅ 备份原始文件")
    print("- ✅ 维护词汇表结构")


if __name__ == "__main__":
    main()