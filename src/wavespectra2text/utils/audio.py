#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频工具模块
提供音频处理相关的工具函数
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import csv


class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024,
                 hop_length: int = 512, max_length: int = 200):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"无法加载音频文件 {audio_path}: {e}")
    
    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """提取频谱特征"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 截断或填充到固定长度
        if magnitude.shape[1] > self.max_length:
            magnitude = magnitude[:, :self.max_length]
        else:
            pad_width = self.max_length - magnitude.shape[1]
            magnitude = np.pad(magnitude, ((0, 0), (0, pad_width)), mode='constant')
        
        return magnitude.T  # 转置为 (time, freq)


class LabelManager:
    """标签管理器"""
    
    @staticmethod
    def scan_audio_files(audio_dir: str) -> List[str]:
        """扫描音频目录中的文件"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for file_path in Path(audio_dir).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        return sorted(audio_files)
    
    @staticmethod
    def create_labels_template(audio_files: List[str], output_file: str = 'data/labels.csv'):
        """创建标签模板"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'text'])
            
            for audio_file in audio_files:
                filename = Path(audio_file).name
                writer.writerow([filename, ''])


class FileUtils:
    """文件工具"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]):
        """确保目录存在"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """获取文件大小"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def is_audio_file(file_path: Union[str, Path]) -> bool:
        """检查是否为音频文件"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        return Path(file_path).suffix.lower() in audio_extensions
