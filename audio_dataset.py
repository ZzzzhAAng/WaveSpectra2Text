#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的音频数据集 - 低耦合、高扩展性
支持实时计算和预计算两种模式
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from audio_preprocessing import AudioPreprocessor, PreprocessorFactory, OfflinePreprocessor
from vocab import vocab


class AudioDataset(Dataset):
    """重构后的音频数据集 - 支持多种预处理策略"""
    
    def __init__(self, 
                 labels_file: str,
                 audio_dir: str = None,
                 preprocessor: AudioPreprocessor = None,
                 precomputed_dir: str = None,
                 cache_dir: str = None,
                 mode: str = 'realtime'):
        """
        初始化数据集
        
        Args:
            labels_file: 标签文件路径
            audio_dir: 音频文件目录 (realtime模式需要)
            preprocessor: 预处理器实例 (realtime模式需要)
            precomputed_dir: 预计算特征目录 (precomputed模式需要)
            cache_dir: 缓存目录 (可选)
            mode: 'realtime' 或 'precomputed'
        """
        self.labels_file = labels_file
        self.audio_dir = audio_dir
        self.precomputed_dir = precomputed_dir
        self.cache_dir = cache_dir
        self.mode = mode
        
        # 加载标签
        self.labels_df = pd.read_csv(labels_file)
        
        # 根据模式初始化
        if mode == 'realtime':
            self._init_realtime_mode(preprocessor)
        elif mode == 'precomputed':
            self._init_precomputed_mode()
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        print(f"数据集初始化完成 - 模式: {mode}, 样本数: {len(self.labels_df)}")
    
    def _init_realtime_mode(self, preprocessor: AudioPreprocessor):
        """初始化实时计算模式"""
        if not preprocessor:
            # 使用默认预处理器
            preprocessor = PreprocessorFactory.create('spectrogram')
        
        if self.cache_dir:
            self.offline_processor = OfflinePreprocessor(preprocessor, self.cache_dir)
        else:
            self.preprocessor = preprocessor
            self.offline_processor = None
        
        # 验证音频文件
        if self.audio_dir:
            self._validate_audio_files()
    
    def _init_precomputed_mode(self):
        """初始化预计算模式"""
        if not self.precomputed_dir or not os.path.exists(self.precomputed_dir):
            raise ValueError(f"预计算目录不存在: {self.precomputed_dir}")
        
        # 加载预计算索引
        index_file = os.path.join(self.precomputed_dir, 'spectrum_index.csv')
        if os.path.exists(index_file):
            self.precomputed_index = pd.read_csv(index_file)
            
            # 为了避免列名冲突，先重命名索引文件中的label列
            self.precomputed_index = self.precomputed_index.rename(columns={'label': 'index_label'})
            
            # 合并标签和预计算索引
            self.labels_df = self.labels_df.merge(
                self.precomputed_index, 
                left_on='filename', 
                right_on='original_audio',
                how='inner'
            )
            
            # 验证标签一致性
            if 'index_label' in self.labels_df.columns:
                # 检查标签是否一致
                inconsistent = self.labels_df[self.labels_df['label'] != self.labels_df['index_label']]
                if len(inconsistent) > 0:
                    print(f"警告: {len(inconsistent)} 个文件的标签不一致")
                
                # 删除重复的标签列
                self.labels_df = self.labels_df.drop(columns=['index_label'])
        else:
            raise FileNotFoundError(f"预计算索引文件不存在: {index_file}")
    
    def _validate_audio_files(self):
        """验证音频文件存在性"""
        missing_files = []
        for _, row in self.labels_df.iterrows():
            audio_path = os.path.join(self.audio_dir, row['filename'])
            if not os.path.exists(audio_path):
                missing_files.append(row['filename'])
        
        if missing_files:
            print(f"警告: {len(missing_files)} 个音频文件不存在")
            # 过滤掉不存在的文件
            self.labels_df = self.labels_df[
                ~self.labels_df['filename'].isin(missing_files)
            ].reset_index(drop=True)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        if self.mode == 'realtime':
            return self._get_realtime_item(row)
        else:
            return self._get_precomputed_item(row)
    
    def _get_realtime_item(self, row):
        """获取实时计算的数据项"""
        audio_path = os.path.join(self.audio_dir, row['filename'])
        
        # 提取特征
        if self.offline_processor:
            features = self.offline_processor.process_file(audio_path)
        else:
            features = self.preprocessor.process(audio_path)
        
        # 编码标签
        encoded_label = vocab.encode(row['label'])
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor(encoded_label),
            'text': row['label'],
            'filename': row['filename']
        }
    
    def _get_precomputed_item(self, row):
        """获取预计算的数据项"""
        spectrum_path = os.path.join(self.precomputed_dir, row['spectrum_file'])
        
        # 加载预计算特征
        features = np.load(spectrum_path)
        
        # 编码标签
        encoded_label = vocab.encode(row['label'])
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor(encoded_label),
            'text': row['label'],
            'filename': row['filename']
        }
    
    def get_feature_shape(self):
        """获取特征形状"""
        if self.mode == 'realtime':
            if self.offline_processor:
                return self.offline_processor.preprocessor.get_feature_shape()
            else:
                return self.preprocessor.get_feature_shape()
        else:
            # 从预计算文件获取形状
            sample = self[0]
            return sample['features'].shape


class FlexibleDataLoader:
    """灵活的数据加载器工厂"""
    
    @staticmethod
    def create_dataloader(dataset: AudioDataset, 
                         batch_size: int = 4,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         **kwargs) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=FlexibleDataLoader.collate_fn,
            **kwargs
        )
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        features = []
        labels = []
        texts = []
        filenames = []
        
        for sample in batch:
            features.append(sample['features'])
            labels.append(sample['label'])
            texts.append(sample['text'])
            filenames.append(sample['filename'])
        
        # 堆叠特征
        features = torch.stack(features)
        
        # 填充标签到相同长度
        max_label_len = max(len(label) for label in labels)
        padded_labels = []
        
        for label in labels:
            if len(label) < max_label_len:
                padded = torch.cat([
                    label,
                    torch.full((max_label_len - len(label),), 
                              vocab.get_padding_idx(), dtype=torch.long)
                ])
            else:
                padded = label
            padded_labels.append(padded)
        
        labels = torch.stack(padded_labels)
        
        return {
            'features': features,
            'labels': labels,
            'texts': texts,
            'filenames': filenames
        }


# 便捷函数
def create_realtime_dataset(labels_file: str, 
                           audio_dir: str,
                           preprocessor_type: str = 'spectrogram',
                           cache_dir: str = None,
                           **preprocessor_kwargs) -> AudioDataset:
    """创建实时计算数据集"""
    preprocessor = PreprocessorFactory.create(preprocessor_type, **preprocessor_kwargs)
    
    return AudioDataset(
        labels_file=labels_file,
        audio_dir=audio_dir,
        preprocessor=preprocessor,
        cache_dir=cache_dir,
        mode='realtime'
    )


def create_precomputed_dataset(labels_file: str, 
                              precomputed_dir: str) -> AudioDataset:
    """创建预计算数据集"""
    return AudioDataset(
        labels_file=labels_file,
        precomputed_dir=precomputed_dir,
        mode='precomputed'
    )


def migrate_from_old_dataset(audio_dir: str, 
                           labels_file: str,
                           output_dir: str,
                           preprocessor_type: str = 'spectrogram',
                           **preprocessor_kwargs):
    """从旧数据集迁移到新架构"""
    print("🔄 开始迁移数据集...")
    
    # 创建预处理器
    preprocessor = PreprocessorFactory.create(preprocessor_type, **preprocessor_kwargs)
    offline_processor = OfflinePreprocessor(preprocessor)
    
    # 读取标签文件
    df = pd.read_csv(labels_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    processed_data = []
    success_count = 0
    
    for idx, row in df.iterrows():
        audio_file = row['filename']
        label = row['label']
        audio_path = os.path.join(audio_dir, audio_file)
        
        if os.path.exists(audio_path):
            try:
                # 处理音频
                features = offline_processor.process_file(audio_path)
                
                # 保存特征文件
                feature_filename = f"{Path(audio_file).stem}.npy"
                feature_path = os.path.join(output_dir, feature_filename)
                np.save(feature_path, features)
                
                processed_data.append({
                    'spectrum_file': feature_filename,
                    'original_audio': audio_file,
                    'label': label,
                    'shape': features.shape
                })
                
                success_count += 1
                
            except Exception as e:
                print(f"处理文件 {audio_file} 失败: {e}")
        else:
            print(f"文件不存在: {audio_path}")
    
    # 保存索引文件
    processed_df = pd.DataFrame(processed_data)
    index_file = os.path.join(output_dir, 'spectrum_index.csv')
    processed_df.to_csv(index_file, index=False, encoding='utf-8')
    
    # 保存配置
    offline_processor.save_config(os.path.join(output_dir, 'preprocess_config.json'))
    
    print(f"✅ 迁移完成: {success_count}/{len(df)} 文件成功处理")
    print(f"特征文件保存到: {output_dir}")
    
    return processed_df


if __name__ == "__main__":
    # 使用示例
    print("🎯 重构后的音频数据集测试")
    print("=" * 50)
    
    # 示例1: 实时计算模式
    try:
        realtime_dataset = create_realtime_dataset(
            labels_file='data/labels.csv',
            audio_dir='data/audio',
            preprocessor_type='spectrogram',
            cache_dir='cache/features'
        )
        
        dataloader = FlexibleDataLoader.create_dataloader(realtime_dataset, batch_size=2)
        print(f"实时数据集创建成功，样本数: {len(realtime_dataset)}")
        
    except Exception as e:
        print(f"实时数据集创建失败: {e}")
    
    # 示例2: 预计算模式
    try:
        precomputed_dataset = create_precomputed_dataset(
            labels_file='data/labels.csv',
            precomputed_dir='data/spectrums'
        )
        
        print(f"预计算数据集创建成功，样本数: {len(precomputed_dataset)}")
        
    except Exception as e:
        print(f"预计算数据集创建失败: {e}")