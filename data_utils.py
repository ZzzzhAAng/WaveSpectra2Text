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

def create_sample_data():
    """创建示例数据文件"""
    # 创建示例标签文件
    labels_data = {
        'filename': [
            '1.wav', '2.wav', '3.wav', '4.wav', '5.wav',
            '6.wav', '7.wav', '8.wav', '9.wav', '10.wav'
        ],
        'label': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    }
    
    df = pd.DataFrame(labels_data)
    df.to_csv('data/labels.csv', index=False, encoding='utf-8')
    print("已创建示例标签文件: data/labels.csv")
    
    # 创建示例音频文件（使用合成音频）
    print("正在创建示例音频文件...")
    
    for i, (filename, label) in enumerate(zip(labels_data['filename'], labels_data['label'])):
        # 生成简单的合成音频（不同频率对应不同数字）
        duration = 1.0  # 1秒
        sample_rate = 48000
        
        # 为每个数字分配不同的基础频率
        base_freq = 200 + i * 50  # 200Hz到 650Hz
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 生成包含多个谐波的音频信号
        audio = np.zeros_like(t)
        for harmonic in range(1, 4):  # 3个谐波
            freq = base_freq * harmonic
            amplitude = 1.0 / harmonic  # 谐波幅度递减
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # 添加一些噪声使其更真实
        noise = np.random.normal(0, 0.05, len(audio))
        audio += noise
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # 保存音频文件
        audio_path = f'data/audio/{filename}'
        
        # 使用librosa保存（需要soundfile库）
        try:
            import soundfile as sf
            sf.write(audio_path, audio, sample_rate)
        except ImportError:
            # 如果没有soundfile，使用scipy
            try:
                from scipy.io import wavfile
                # 转换为16位整数
                audio_int = (audio * 32767).astype(np.int16)
                wavfile.write(audio_path, sample_rate, audio_int)
            except ImportError:
                print(f"无法保存音频文件 {audio_path}，请安装soundfile或scipy")
                continue
        
        print(f"已创建: {audio_path} (标签: {label})")

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
    # 创建示例数据
    create_sample_data()
    
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