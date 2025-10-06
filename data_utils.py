# -*- coding: utf-8 -*-
"""
数据处理工具
用于处理音频文件，提取频谱特征，加载标签等
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from vocab import vocab
import warnings
warnings.filterwarnings('ignore')

class AudioSpectrogramDataset(Dataset):
    """音频频谱数据集"""
    
    def __init__(self, audio_dir, labels_file, sample_rate=48000, n_fft=1024, 
                 hop_length=512, max_length=200):
        """
        Args:
            audio_dir: 音频文件目录
            labels_file: 标签文件路径
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            max_length: 最大序列长度
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        
        # 加载标签
        self.labels_df = pd.read_csv(labels_file)
        self.data = []
        
        # 预处理数据
        self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理数据，提取频谱特征"""
        print("正在预处理数据...")
        
        for idx, row in self.labels_df.iterrows():
            audio_file = row['filename']
            label = row['label']
            
            audio_path = os.path.join(self.audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                try:
                    # 加载音频
                    audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                    
                    # 提取STFT频谱
                    stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                    magnitude = np.abs(stft)  # 幅度谱
                    
                    # 转换为对数刻度
                    log_magnitude = np.log1p(magnitude)
                    
                    # 转置使时间维度在前
                    spectrogram = log_magnitude.T  # (time_steps, freq_bins)
                    
                    # 编码标签
                    encoded_label = vocab.encode(label)
                    
                    self.data.append({
                        'spectrogram': spectrogram,
                        'label': encoded_label,
                        'text': label,
                        'filename': audio_file
                    })
                    
                except Exception as e:
                    print(f"处理文件 {audio_file} 时出错: {e}")
            else:
                print(f"文件不存在: {audio_path}")
        
        print(f"成功加载 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        spectrogram = sample['spectrogram']
        label = sample['label']
        
        # 填充或截断频谱到固定长度
        if len(spectrogram) > self.max_length:
            spectrogram = spectrogram[:self.max_length]
        else:
            # 零填充
            pad_length = self.max_length - len(spectrogram)
            spectrogram = np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')
        
        # 转换为tensor
        spectrogram = torch.FloatTensor(spectrogram)
        label = torch.LongTensor(label)
        
        return {
            'spectrogram': spectrogram,
            'label': label,
            'text': sample['text'],
            'filename': sample['filename']
        }

def collate_fn(batch):
    """批处理函数"""
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
            # 使用PAD token填充
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

def create_labels_file_if_not_exists(labels_file='data/labels.csv'):
    """如果标签文件不存在，创建示例标签文件"""
    if os.path.exists(labels_file):
        print(f"标签文件已存在: {labels_file}")
        return
    
    # 创建示例标签文件
    labels_data = {
        'filename': [
            '1.wav', '2.wav', '3.wav', '4.wav', '5.wav',
            '6.wav', '7.wav', '8.wav', '9.wav', '10.wav'
        ],
        'label': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    }
    
    df = pd.DataFrame(labels_data)
    df.to_csv(labels_file, index=False, encoding='utf-8')
    print(f"已创建示例标签文件: {labels_file}")
    print("请根据你的实际音频文件修改标签文件中的filename字段")

def check_audio_files(audio_dir, labels_file):
    """检查音频文件是否存在"""
    if not os.path.exists(labels_file):
        print(f"错误: 标签文件不存在 {labels_file}")
        return False
    
    df = pd.read_csv(labels_file)
    missing_files = []
    existing_files = []
    
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(audio_path):
            existing_files.append(row['filename'])
        else:
            missing_files.append(row['filename'])
    
    print(f"音频文件检查结果:")
    print(f"  找到的文件: {len(existing_files)}")
    print(f"  缺失的文件: {len(missing_files)}")
    
    if existing_files:
        print(f"  存在的文件: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}")
    
    if missing_files:
        print(f"  缺失的文件: {missing_files}")
        print("请确保音频文件存在于指定目录中")
    
    return len(missing_files) == 0

def get_dataloader(audio_dir='data/audio', labels_file='data/labels.csv', 
                  batch_size=4, shuffle=True, num_workers=0):
    """获取数据加载器"""
    dataset = AudioSpectrogramDataset(audio_dir, labels_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader

if __name__ == "__main__":
    # 检查并创建标签文件
    create_labels_file_if_not_exists()
    
    # 检查音频文件
    audio_dir = 'data/audio'
    labels_file = 'data/labels.csv'
    
    if check_audio_files(audio_dir, labels_file):
        print("所有音频文件都存在，可以开始训练")
        
        # 测试数据加载
        try:
            dataloader = get_dataloader(batch_size=2)
            
            print(f"数据集大小: {len(dataloader.dataset)}")
            
            # 获取一个批次
            for batch in dataloader:
                print(f"频谱形状: {batch['spectrograms'].shape}")
                print(f"标签形状: {batch['labels'].shape}")
                print(f"文本: {batch['texts']}")
                print(f"文件名: {batch['filenames']}")
                break
                
        except Exception as e:
            print(f"测试数据加载时出错: {e}")
            print("请确保安装了librosa和相关依赖")
    else:
        print("部分音频文件缺失，请检查文件路径")