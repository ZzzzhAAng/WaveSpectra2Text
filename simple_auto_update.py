#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版自动更新系统
不依赖pandas，使用纯Python实现
"""

import os
import csv
import json
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Set, Optional


class SimpleAutoUpdater:
    """简化版自动更新器"""
    
    def __init__(self, 
                 audio_dir: str = 'data/audio',
                 labels_file: str = 'data/labels.csv',
                 vocab_file: str = 'vocab.py'):
        """初始化"""
        self.audio_dir = Path(audio_dir)
        self.labels_file = Path(labels_file)
        self.vocab_file = Path(vocab_file)
        
        print(f"🤖 简化版自动更新器初始化")
        print(f"音频目录: {self.audio_dir}")
        print(f"标签文件: {self.labels_file}")
        print(f"词汇文件: {self.vocab_file}")
    
    def read_labels_csv(self) -> List[Dict[str, str]]:
        """读取CSV标签文件"""
        labels = []
        
        if not self.labels_file.exists():
            return labels
        
        try:
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels.append(row)
        except Exception as e:
            print(f"❌ 读取标签文件失败: {e}")
        
        return labels
    
    def extract_labels_from_csv(self) -> Set[str]:
        """从CSV提取唯一标签"""
        labels_data = self.read_labels_csv()
        labels = set()
        
        for row in labels_data:
            if 'label' in row and row['label']:
                labels.add(row['label'].strip())
        
        return labels
    
    def get_vocab_labels(self) -> Set[str]:
        """从vocab.py提取当前标签"""
        if not self.vocab_file.exists():
            return set()
        
        try:
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找中文字符（更广泛的匹配）
            import re
            chinese_chars = re.findall(r"'([\u4e00-\u9fff])': \d+", content)
            return set(chinese_chars)
            
        except Exception as e:
            print(f"❌ 读取词汇文件失败: {e}")
            return set()
    
    def update_vocab_file(self, new_labels: Set[str]) -> bool:
        """更新词汇表文件"""
        if not new_labels:
            return False
        
        try:
            # 读取当前文件
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找词汇表部分
            import re
            pattern = r"(self\.word_to_idx\s*=\s*\{[^}]+\})"
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                print("❌ 未找到词汇表定义")
                return False
            
            # 提取现有词汇
            vocab_content = match.group(1)
            existing_vocab = {}
            
            # 解析现有词汇表
            for line in vocab_content.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    try:
                        key_part, value_part = line.split(':', 1)
                        key = key_part.strip().strip("'\"")
                        value = value_part.strip().rstrip(',')
                        if value.isdigit():
                            existing_vocab[key] = int(value)
                    except:
                        continue
            
            # 检查需要添加的新标签
            current_labels = set(existing_vocab.keys()) - {'<PAD>', '<SOS>', '<EOS>', '<UNK>'}
            labels_to_add = new_labels - current_labels
            
            if not labels_to_add:
                print("✅ 词汇表已是最新")
                return False
            
            # 添加新标签
            next_idx = max(existing_vocab.values()) + 1
            for label in sorted(labels_to_add):
                existing_vocab[label] = next_idx
                next_idx += 1
            
            # 重新生成词汇表
            new_vocab_lines = ["        self.word_to_idx = {"]
            
            # 特殊符号
            special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
            for token in special_tokens:
                if token in existing_vocab:
                    new_vocab_lines.append(f"            '{token}': {existing_vocab[token]},")
            
            # 标签（按索引排序）
            label_items = [(k, v) for k, v in existing_vocab.items() if k not in special_tokens]
            label_items.sort(key=lambda x: x[1])
            
            for label, idx in label_items:
                new_vocab_lines.append(f"            '{label}': {idx},")
            
            new_vocab_lines.append("        }")
            new_vocab_content = '\n'.join(new_vocab_lines)
            
            # 替换内容
            new_content = re.sub(pattern, new_vocab_content, content, flags=re.DOTALL)
            
            # 备份原文件
            backup_file = self.vocab_file.with_suffix('.py.backup')
            shutil.copy2(self.vocab_file, backup_file)
            
            # 写入新内容
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ 词汇表已更新，添加标签: {labels_to_add}")
            print(f"📁 备份文件: {backup_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 更新词汇表失败: {e}")
            return False
    
    def scan_audio_files(self) -> List[str]:
        """扫描音频文件"""
        if not self.audio_dir.exists():
            return []
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        audio_files = []
        
        for file_path in self.audio_dir.iterdir():
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(file_path.name)
        
        return sorted(audio_files)
    
    def check_and_update(self) -> Dict[str, bool]:
        """检查并更新"""
        print("🔍 检查数据更新...")
        
        results = {
            'labels_found': False,
            'vocab_updated': False,
            'audio_files_found': False
        }
        
        # 检查音频文件
        audio_files = self.scan_audio_files()
        if audio_files:
            results['audio_files_found'] = True
            print(f"📁 发现 {len(audio_files)} 个音频文件")
        
        # 检查标签文件
        csv_labels = self.extract_labels_from_csv()
        if csv_labels:
            results['labels_found'] = True
            print(f"🏷️  发现标签: {sorted(csv_labels)}")
            
            # 检查词汇表
            vocab_labels = self.get_vocab_labels()
            print(f"📚 当前词汇表: {sorted(vocab_labels)}")
            
            new_labels = csv_labels - vocab_labels
            if new_labels:
                print(f"🆕 需要添加的新标签: {sorted(new_labels)}")
                
                if self.update_vocab_file(csv_labels):
                    results['vocab_updated'] = True
            else:
                print("✅ 词汇表已包含所有标签")
        
        return results
    
    def create_features_directory(self):
        """创建特征目录"""
        features_dir = Path('data/features')
        features_dir.mkdir(exist_ok=True)
        print(f"📁 创建特征目录: {features_dir}")
        
        # 创建简单的索引文件模板
        index_file = features_dir / 'spectrum_index.csv'
        if not index_file.exists():
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write('spectrum_file,original_audio,label,shape\n')
            print(f"📄 创建索引文件模板: {index_file}")


def main():
    """主函数"""
    print("🤖 简化版自动更新系统")
    print("=" * 50)
    
    # 创建更新器
    updater = SimpleAutoUpdater()
    
    # 执行检查和更新
    results = updater.check_and_update()
    
    # 创建特征目录
    updater.create_features_directory()
    
    # 报告结果
    print("\n📊 更新结果:")
    print(f"  音频文件: {'✅' if results['audio_files_found'] else '❌'}")
    print(f"  标签文件: {'✅' if results['labels_found'] else '❌'}")
    print(f"  词汇表更新: {'✅' if results['vocab_updated'] else '❌'}")
    
    if results['vocab_updated']:
        print("\n✅ 更新完成！词汇表已同步")
    elif results['labels_found']:
        print("\n✅ 检查完成！所有标签都已在词汇表中")
    else:
        print("\n⚠️  未找到标签文件或标签为空")
    
    print("\n💡 下一步:")
    if results['labels_found']:
        print("  1. 运行训练: python3 train_at_different_scales/train_scale_1.py")
        print("  2. 预处理特征: python3 batch_preprocess.py")
        print("  3. 运行推理: python3 dual_input_inference.py --model <model> --input <file>")
    else:
        print("  1. 运行数据设置: python3 setup_data.py")
        print("  2. 检查data/labels.csv文件")


if __name__ == "__main__":
    main()