#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具 V2 - 长期重构版本
支持智能模式切换，消除代码冗余
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from vocab import vocab
from spectrum_utils import SpectrumProcessor
import warnings
warnings.filterwarnings('ignore')

class SmartAudioSpectrogramDataset(Dataset):
    """智能音频频谱数据集 - 自动选择最优处理模式"""
    
    def __init__(self, audio_dir, labels_file, spectrum_dir=None, 
                 sample_rate=48000, n_fft=1024, hop_length=512, max_length=200):
        """
        初始化智能数据集
        
        Args:
            audio_dir: 音频文件目录
            labels_file: 标签文件路径
            spectrum_dir: 预处理频谱目录 (可选，如果存在则优先使用)
            其他参数: 频谱处理参数
        """
        self.audio_dir = audio_dir
        self.spectrum_dir = spectrum_dir
        self.labels_file = labels_file
        
        # 创建统一的频谱处理器
        self.spectrum_processor = SpectrumProcessor(sample_rate, n_fft, hop_length, max_length)
        
        # 智能检测处理模式
        self.processing_mode = self._detect_processing_mode()
        
        # 加载标签
        self.labels_df = pd.read_csv(labels_file)
        
        # 准备数据
        self.data = []
        self._prepare_data()
    
    def _detect_processing_mode(self):
        """智能检测最优的处理模式"""
        if self.spectrum_dir and os.path.exists(self.spectrum_dir):
            spectrum_index = os.path.join(self.spectrum_dir, 'spectrum_index.csv')
            if os.path.exists(spectrum_index):
                print("🚀 检测到预处理频谱，使用预处理模式")
                return 'preprocessed'
        
        print("⚡ 未检测到预处理频谱，使用实时处理模式")
        return 'realtime'
    
    def _prepare_data(self):
        """根据处理模式准备数据"""
        if self.processing_mode == 'preprocessed':
            self._prepare_preprocessed_data()
        else:
            self._prepare_realtime_data()
        
        print(f"✅ 数据准备完成: {len(self.data)} 个样本 ({self.processing_mode} 模式)")
    
    def _prepare_preprocessed_data(self):
        """准备预处理数据"""
        spectrum_index_file = os.path.join(self.spectrum_dir, 'spectrum_index.csv')
        
        try:
            spectrum_df = pd.read_csv(spectrum_index_file)
            
            # 合并标签和频谱索引
            merged_df = pd.merge(
                self.labels_df, 
                spectrum_df, 
                left_on='filename', 
                right_on='original_audio',
                how='inner'
            )
            
            for _, row in merged_df.iterrows():
                spectrum_path = os.path.join(self.spectrum_dir, row['spectrum_file'])
                
                if os.path.exists(spectrum_path):
                    encoded_label = vocab.encode(row['label'])
                    
                    self.data.append({
                        'source_path': spectrum_path,
                        'label': encoded_label,
                        'text': row['label'],
                        'filename': row['filename'],
                        'mode': 'preprocessed'
                    })
                else:
                    print(f"⚠️ 频谱文件不存在: {spectrum_path}")
        
        except Exception as e:
            print(f"⚠️ 预处理模式失败，切换到实时模式: {e}")
            self.processing_mode = 'realtime'
            self._prepare_realtime_data()
    
    def _prepare_realtime_data(self):
        """准备实时处理数据"""
        for _, row in self.labels_df.iterrows():
            audio_file = row['filename']
            label = row['label']
            
            audio_path = os.path.join(self.audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                encoded_label = vocab.encode(label)
                
                self.data.append({
                    'source_path': audio_path,
                    'label': encoded_label,
                    'text': label,
                    'filename': audio_file,
                    'mode': 'realtime'
                })
            else:
                print(f"⚠️ 音频文件不存在: {audio_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 根据模式加载频谱 - 使用统一的处理器
        if sample['mode'] == 'preprocessed':
            spectrogram = self.spectrum_processor.load_spectrum_from_file(sample['source_path'])
        else:
            spectrogram = self.spectrum_processor.extract_spectrum_from_audio(sample['source_path'])
        
        # 处理加载失败的情况
        if spectrogram is None:
            print(f"⚠️ 频谱加载失败: {sample['source_path']}")
            # 创建零频谱作为fallback
            spectrogram = self._create_zero_spectrum()
        
        # 转换为tensor
        spectrogram_tensor = torch.FloatTensor(spectrogram)
        label_tensor = torch.LongTensor(sample['label'])
        
        return {
            'spectrogram': spectrogram_tensor,
            'label': label_tensor,
            'text': sample['text'],
            'filename': sample['filename']
        }
    
    def _create_zero_spectrum(self):
        """创建零频谱作为fallback"""
        return np.zeros((self.spectrum_processor.max_length, self.spectrum_processor.freq_bins), dtype=np.float32)
    
    def get_processing_info(self):
        """获取处理信息"""
        return {
            'mode': self.processing_mode,
            'total_samples': len(self.data),
            'spectrum_config': self.spectrum_processor.get_config()
        }

def collate_fn(batch):
    """批处理函数 - 保持与原版兼容"""
    spectrograms = []
    labels = []
    texts = []
    filenames = []
    
    for sample in batch:
        spectrograms.append(sample['spectrogram'])
        labels.append(sample['label'])
        texts.append(sample['text'])
        filenames.append(sample['filename'])
    
    # 堆叠频谱
    spectrograms = torch.stack(spectrograms)
    
    # 填充标签到相同长度
    max_label_len = max(len(label) for label in labels)
    padded_labels = []
    
    for label in labels:
        if len(label) < max_label_len:
            padded = torch.cat([
                label, 
                torch.full((max_label_len - len(label),), vocab.get_padding_idx(), dtype=torch.long)
            ])
        else:
            padded = label
        padded_labels.append(padded)
    
    labels = torch.stack(padded_labels)
    
    return {
        'spectrograms': spectrograms,
        'labels': labels,
        'texts': texts,
        'filenames': filenames
    }

def get_smart_dataloader(audio_dir='data/audio', labels_file='data/labels.csv', 
                        spectrum_dir='data/spectrums', batch_size=4, shuffle=True, num_workers=0):
    """
    获取智能数据加载器 - 自动选择最优模式
    
    Args:
        spectrum_dir: 预处理频谱目录，如果存在且有效则使用预处理模式
    """
    dataset = SmartAudioSpectrogramDataset(
        audio_dir=audio_dir,
        labels_file=labels_file,
        spectrum_dir=spectrum_dir
    )
    
    # 显示处理信息
    info = dataset.get_processing_info()
    print(f"📊 数据加载器信息: {info['mode']} 模式, {info['total_samples']} 样本")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader

# 向后兼容的接口
def get_dataloader(*args, **kwargs):
    """向后兼容的数据加载器接口"""
    return get_smart_dataloader(*args, **kwargs)

# 创建原始数据集的别名 (向后兼容)
AudioSpectrogramDataset = SmartAudioSpectrogramDataset

if __name__ == "__main__":
    print("🧪 测试智能数据加载器")
    
    # 测试智能模式选择
    try:
        dataloader = get_smart_dataloader(batch_size=2)
        
        # 获取一个批次测试
        for batch in dataloader:
            print(f"✅ 批次测试成功:")
            print(f"   频谱形状: {batch['spectrograms'].shape}")
            print(f"   标签形状: {batch['labels'].shape}")
            print(f"   文本: {batch['texts']}")
            break
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请确保安装了必要的依赖包")
    
    print("✅ V2数据处理工具测试完成")