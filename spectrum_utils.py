#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一频谱处理工具 - 长期重构方案的核心模块
消除所有频谱处理相关的代码冗余
"""

import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

class SpectrumProcessor:
    """统一的频谱处理器"""
    
    def __init__(self, sample_rate=48000, n_fft=1024, hop_length=512, max_length=200):
        """
        初始化频谱处理器
        
        Args:
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            max_length: 最大序列长度
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.freq_bins = n_fft // 2 + 1  # 513
        
        print(f"频谱处理器初始化: {sample_rate}Hz, {n_fft}FFT, {self.freq_bins}bins")
    
    def extract_spectrum_from_audio(self, audio_path):
        """
        从音频文件提取频谱特征 - 核心统一方法
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            spectrogram: 频谱特征 (max_length, freq_bins) 或 None
        """
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
            
            # 填充或截断到固定长度
            spectrogram = self._normalize_length(spectrogram)
            
            return spectrogram.astype(np.float32)
            
        except Exception as e:
            print(f"提取频谱失败 {audio_path}: {e}")
            return None
    
    def load_spectrum_from_file(self, spectrum_path):
        """
        从预处理的频谱文件加载
        
        Args:
            spectrum_path: 频谱文件路径 (.npy)
            
        Returns:
            spectrogram: 频谱特征 (max_length, freq_bins) 或 None
        """
        try:
            spectrogram = np.load(spectrum_path)
            
            # 确保长度一致
            spectrogram = self._normalize_length(spectrogram)
            
            return spectrogram.astype(np.float32)
            
        except Exception as e:
            print(f"加载频谱失败 {spectrum_path}: {e}")
            return None
    
    def _normalize_length(self, spectrogram):
        """标准化频谱长度"""
        if len(spectrogram) > self.max_length:
            return spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            return np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')
    
    def save_spectrum_to_file(self, spectrogram, output_path):
        """保存频谱到文件"""
        try:
            np.save(output_path, spectrogram)
            return True
        except Exception as e:
            print(f"保存频谱失败 {output_path}: {e}")
            return False
    
    def get_config(self):
        """获取配置参数"""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'max_length': self.max_length,
            'freq_bins': self.freq_bins
        }

# 全局默认处理器实例
_default_processor = None

def get_default_processor():
    """获取默认的频谱处理器"""
    global _default_processor
    if _default_processor is None:
        _default_processor = SpectrumProcessor()
    return _default_processor

def extract_spectrum(audio_path):
    """便捷函数：使用默认处理器提取频谱"""
    return get_default_processor().extract_spectrum_from_audio(audio_path)

def load_spectrum(spectrum_path):
    """便捷函数：使用默认处理器加载频谱"""
    return get_default_processor().load_spectrum_from_file(spectrum_path)

if __name__ == "__main__":
    # 测试频谱处理器
    print("🧪 测试统一频谱处理器")
    
    processor = SpectrumProcessor()
    print(f"配置: {processor.get_config()}")
    
    # 测试便捷函数
    print("✅ 统一频谱处理器创建成功")