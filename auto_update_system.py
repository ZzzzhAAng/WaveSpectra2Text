#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动数据更新系统
当data/audio和data/labels.csv更新时，自动同步相关文件
"""

import os
import pandas as pd
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional
import time
from datetime import datetime
import hashlib
import argparse

from vocab import vocab
from audio_preprocess import PreprocessorFactory, OfflinePreprocessor
from common_utils import LabelManager, AudioProcessor, FileUtils


class AutoUpdateSystem:
    """自动数据更新系统"""
    
    def __init__(self, 
                 audio_dir: str = 'data/audio',
                 labels_file: str = 'data/labels.csv',
                 features_dir: str = 'data/features',
                 vocab_file: str = 'vocab.py',
                 config_dir: str = '.',
                 log_file: str = 'auto_update.log'):
        """
        初始化自动更新系统
        
        Args:
            audio_dir: 音频文件目录
            labels_file: 标签文件路径
            features_dir: 特征文件目录
            vocab_file: 词汇表文件路径
            config_dir: 配置文件目录
            log_file: 日志文件路径
        """
        self.audio_dir = Path(audio_dir)
        self.labels_file = Path(labels_file)
        self.features_dir = Path(features_dir)
        self.vocab_file = Path(vocab_file)
        self.config_dir = Path(config_dir)
        self.log_file = Path(log_file)
        
        # 状态跟踪文件
        self.state_file = Path('.auto_update_state.json')
        
        # 创建必要目录
        self.features_dir.mkdir(exist_ok=True)
        
        # 加载或初始化状态
        self.state = self._load_state()
        
        # 初始化预处理器
        self.preprocessor = None
        self._init_preprocessor()
        
        print(f"🤖 自动更新系统初始化完成")
        print(f"监控目录: {self.audio_dir}")
        print(f"标签文件: {self.labels_file}")
        print(f"特征目录: {self.features_dir}")
    
    def _init_preprocessor(self):
        """初始化预处理器"""
        try:
            preprocessor = PreprocessorFactory.create('spectrogram')
            self.offline_processor = OfflinePreprocessor(
                preprocessor, 
                cache_dir=str(self.features_dir)
            )
            self.preprocessor = preprocessor
        except Exception as e:
            print(f"⚠️  预处理器初始化失败: {e}")
    
    def _load_state(self) -> Dict:
        """加载系统状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  状态文件加载失败: {e}")
        
        return {
            'last_update': None,
            'audio_files': {},
            'labels_hash': None,
            'vocab_labels': set(),
            'processed_files': {}
        }
    
    def _save_state(self):
        """保存系统状态"""
        # 转换set为list以便JSON序列化
        state_to_save = self.state.copy()
        state_to_save['vocab_labels'] = list(state_to_save['vocab_labels'])
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  状态保存失败: {e}")
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"⚠️  日志写入失败: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希值"""
        if not file_path.exists():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _scan_audio_files(self) -> Dict[str, Dict]:
        """扫描音频文件"""
        audio_files = {}
        
        if not self.audio_dir.exists():
            return audio_files
        
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        for file_path in self.audio_dir.iterdir():
            if file_path.suffix.lower() in audio_extensions:
                file_info = {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'mtime': file_path.stat().st_mtime,
                    'hash': self._get_file_hash(file_path)
                }
                audio_files[file_path.name] = file_info
        
        return audio_files
    
    def _check_audio_changes(self) -> Dict[str, List[str]]:
        """检查音频文件变化"""
        current_files = self._scan_audio_files()
        previous_files = self.state.get('audio_files', {})
        
        changes = {
            'added': [],
            'modified': [],
            'removed': []
        }
        
        # 检查新增和修改的文件
        for filename, file_info in current_files.items():
            if filename not in previous_files:
                changes['added'].append(filename)
            elif (previous_files[filename]['hash'] != file_info['hash'] or
                  previous_files[filename]['mtime'] != file_info['mtime']):
                changes['modified'].append(filename)
        
        # 检查删除的文件
        for filename in previous_files:
            if filename not in current_files:
                changes['removed'].append(filename)
        
        # 更新状态
        self.state['audio_files'] = current_files
        
        return changes
    
    def _check_labels_changes(self) -> bool:
        """检查标签文件变化"""
        if not self.labels_file.exists():
            return False
        
        current_hash = self._get_file_hash(self.labels_file)
        previous_hash = self.state.get('labels_hash')
        
        if current_hash != previous_hash:
            self.state['labels_hash'] = current_hash
            return True
        
        return False
    
    def _extract_labels_from_csv(self) -> Set[str]:
        """从CSV文件提取标签"""
        labels = set()
        
        if not self.labels_file.exists():
            return labels
        
        try:
            df = pd.read_csv(self.labels_file)
            if 'label' in df.columns:
                labels = set(df['label'].dropna().unique())
        except Exception as e:
            self._log(f"❌ 读取标签文件失败: {e}")
        
        return labels
    
    def _update_vocab_file(self, new_labels: Set[str]) -> bool:
        """更新词汇表文件"""
        if not new_labels:
            return False
        
        try:
            # 读取当前词汇表文件
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找词汇表定义
            import re
            
            # 查找word_to_idx字典
            pattern = r"self\.word_to_idx\s*=\s*\{([^}]+)\}"
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                self._log("❌ 未找到word_to_idx定义")
                return False
            
            # 解析现有词汇表
            existing_vocab = {}
            vocab_content = match.group(1)
            
            # 提取现有词汇
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
                self._log("✅ 词汇表已是最新，无需更新")
                return False
            
            # 添加新标签
            next_idx = max(existing_vocab.values()) + 1
            for label in sorted(labels_to_add):
                existing_vocab[label] = next_idx
                next_idx += 1
            
            # 重新生成词汇表内容
            new_vocab_lines = []
            new_vocab_lines.append("        self.word_to_idx = {")
            
            # 保持特殊符号在前
            special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
            for token in special_tokens:
                if token in existing_vocab:
                    new_vocab_lines.append(f"            '{token}': {existing_vocab[token]},")
            
            # 添加标签（按索引排序）
            label_items = [(k, v) for k, v in existing_vocab.items() if k not in special_tokens]
            label_items.sort(key=lambda x: x[1])
            
            for label, idx in label_items:
                new_vocab_lines.append(f"            '{label}': {idx},")
            
            new_vocab_lines.append("        }")
            
            new_vocab_content = '\n'.join(new_vocab_lines)
            
            # 替换原内容
            new_content = re.sub(pattern, new_vocab_content, content, flags=re.DOTALL)
            
            # 备份原文件
            backup_file = self.vocab_file.with_suffix('.py.backup')
            shutil.copy2(self.vocab_file, backup_file)
            
            # 写入新内容
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self._log(f"✅ 词汇表已更新，添加标签: {labels_to_add}")
            self._log(f"📁 原文件备份至: {backup_file}")
            
            return True
            
        except Exception as e:
            self._log(f"❌ 更新词汇表失败: {e}")
            return False
    
    def _process_new_audio_files(self, filenames: List[str]) -> bool:
        """处理新的音频文件"""
        if not filenames or not self.offline_processor:
            return False
        
        success_count = 0
        
        for filename in filenames:
            audio_path = self.audio_dir / filename
            
            if not audio_path.exists():
                continue
            
            try:
                self._log(f"🎵 处理音频文件: {filename}")
                
                # 提取特征
                features = self.offline_processor.process_file(
                    str(audio_path), 
                    force_recompute=True
                )
                
                # 保存特征文件
                feature_filename = f"{audio_path.stem}.npy"
                feature_path = self.features_dir / feature_filename
                
                import numpy as np
                np.save(feature_path, features)
                
                # 更新处理记录
                self.state['processed_files'][filename] = {
                    'feature_file': feature_filename,
                    'processed_time': time.time(),
                    'feature_shape': features.shape
                }
                
                success_count += 1
                self._log(f"✅ 特征提取完成: {feature_filename}, 形状: {features.shape}")
                
            except Exception as e:
                self._log(f"❌ 处理文件 {filename} 失败: {e}")
        
        if success_count > 0:
            self._update_features_index()
            
        return success_count > 0
    
    def _update_features_index(self):
        """更新特征索引文件"""
        try:
            # 读取标签文件
            if not self.labels_file.exists():
                return
            
            df = pd.read_csv(self.labels_file)
            
            # 创建特征索引
            index_data = []
            
            for _, row in df.iterrows():
                filename = row['filename']
                label = row['label']
                
                # 检查对应的特征文件
                feature_filename = f"{Path(filename).stem}.npy"
                feature_path = self.features_dir / feature_filename
                
                if feature_path.exists():
                    try:
                        import numpy as np
                        features = np.load(feature_path)
                        
                        index_data.append({
                            'spectrum_file': feature_filename,
                            'original_audio': filename,
                            'label': label,
                            'shape': str(features.shape)
                        })
                    except Exception as e:
                        self._log(f"⚠️  读取特征文件 {feature_filename} 失败: {e}")
            
            # 保存索引文件
            if index_data:
                index_df = pd.DataFrame(index_data)
                index_file = self.features_dir / 'spectrum_index.csv'
                index_df.to_csv(index_file, index=False, encoding='utf-8')
                
                self._log(f"✅ 特征索引已更新: {len(index_data)} 个文件")
            
        except Exception as e:
            self._log(f"❌ 更新特征索引失败: {e}")
    
    def _clean_removed_files(self, removed_files: List[str]):
        """清理已删除文件对应的特征"""
        for filename in removed_files:
            # 删除对应的特征文件
            feature_filename = f"{Path(filename).stem}.npy"
            feature_path = self.features_dir / feature_filename
            
            if feature_path.exists():
                try:
                    feature_path.unlink()
                    self._log(f"🗑️  删除特征文件: {feature_filename}")
                except Exception as e:
                    self._log(f"❌ 删除特征文件失败: {e}")
            
            # 从处理记录中移除
            if filename in self.state['processed_files']:
                del self.state['processed_files'][filename]
    
    def check_and_update(self) -> Dict[str, bool]:
        """检查并更新所有相关文件"""
        self._log("🔍 开始检查数据更新...")
        
        results = {
            'audio_changes': False,
            'labels_changes': False,
            'vocab_updated': False,
            'features_updated': False
        }
        
        # 检查音频文件变化
        audio_changes = self._check_audio_changes()
        if any(audio_changes.values()):
            results['audio_changes'] = True
            
            if audio_changes['added']:
                self._log(f"📁 发现新增音频文件: {audio_changes['added']}")
            if audio_changes['modified']:
                self._log(f"📝 发现修改音频文件: {audio_changes['modified']}")
            if audio_changes['removed']:
                self._log(f"🗑️  发现删除音频文件: {audio_changes['removed']}")
            
            # 处理新增和修改的文件
            files_to_process = audio_changes['added'] + audio_changes['modified']
            if files_to_process and self._process_new_audio_files(files_to_process):
                results['features_updated'] = True
            
            # 清理删除的文件
            if audio_changes['removed']:
                self._clean_removed_files(audio_changes['removed'])
                results['features_updated'] = True
        
        # 检查标签文件变化
        if self._check_labels_changes():
            results['labels_changes'] = True
            self._log("📝 检测到标签文件更新")
            
            # 提取新标签
            current_labels = self._extract_labels_from_csv()
            previous_labels = self.state.get('vocab_labels', set())
            
            # 转换为set（如果是list）
            if isinstance(previous_labels, list):
                previous_labels = set(previous_labels)
            
            new_labels = current_labels - previous_labels
            
            if new_labels:
                self._log(f"🆕 发现新标签: {new_labels}")
                
                # 更新词汇表
                if self._update_vocab_file(current_labels):
                    results['vocab_updated'] = True
                
                # 更新状态
                self.state['vocab_labels'] = current_labels
            
            # 更新特征索引
            self._update_features_index()
            results['features_updated'] = True
        
        # 保存状态
        self.state['last_update'] = time.time()
        self._save_state()
        
        return results
    
    def run_continuous_monitoring(self, interval: int = 10):
        """运行持续监控模式"""
        self._log(f"🔄 开始持续监控模式，检查间隔: {interval}秒")
        
        try:
            while True:
                results = self.check_and_update()
                
                if any(results.values()):
                    self._log("✅ 更新完成")
                else:
                    self._log("💤 无变化，继续监控...")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self._log("⏹️  监控已停止")
        except Exception as e:
            self._log(f"❌ 监控过程出错: {e}")
    
    def run_single_check(self) -> bool:
        """运行单次检查"""
        results = self.check_and_update()
        
        if any(results.values()):
            self._log("✅ 检查完成，发现并处理了更新")
            return True
        else:
            self._log("✅ 检查完成，无需更新")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动数据更新系统')
    parser.add_argument('--mode', choices=['check', 'monitor'], default='check',
                        help='运行模式: check(单次检查) 或 monitor(持续监控)')
    parser.add_argument('--interval', type=int, default=10,
                        help='监控间隔（秒）')
    parser.add_argument('--audio_dir', default='data/audio',
                        help='音频文件目录')
    parser.add_argument('--labels_file', default='data/labels.csv',
                        help='标签文件路径')
    parser.add_argument('--features_dir', default='data/features',
                        help='特征文件目录')
    parser.add_argument('--vocab_file', default='vocab.py',
                        help='词汇表文件路径')
    
    args = parser.parse_args()
    
    print("🤖 自动数据更新系统")
    print("=" * 60)
    
    # 创建更新系统
    updater = AutoUpdateSystem(
        audio_dir=args.audio_dir,
        labels_file=args.labels_file,
        features_dir=args.features_dir,
        vocab_file=args.vocab_file
    )
    
    if args.mode == 'check':
        # 单次检查模式
        print("🔍 执行单次检查...")
        updater.run_single_check()
        
    elif args.mode == 'monitor':
        # 持续监控模式
        print(f"👁️  启动持续监控模式（间隔: {args.interval}秒）")
        print("按 Ctrl+C 停止监控")
        updater.run_continuous_monitoring(args.interval)


if __name__ == "__main__":
    main()