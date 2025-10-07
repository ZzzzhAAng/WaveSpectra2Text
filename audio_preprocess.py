#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的音频预处理模块
提供可扩展的预处理策略和低耦合的设计
"""

import os
import numpy as np
import librosa
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path


class AudioPreprocessor(ABC):
    """音频预处理器抽象基类"""

    def __init__(self, sample_rate: int = 48000, **kwargs):
        self.sample_rate = sample_rate
        self.config = kwargs

    @abstractmethod
    def process(self, audio_path: str) -> np.ndarray:
        """处理音频文件，返回特征"""
        pass

    @abstractmethod
    def get_feature_shape(self) -> Tuple[int, ...]:
        """获取特征形状"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """获取配置参数"""
        return {
            'sample_rate': self.sample_rate,
            **self.config
        }


class SpectrogramPreprocessor(AudioPreprocessor):
    """STFT频谱预处理器"""

    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024,
                 hop_length: int = 512, max_length: int = 200, **kwargs):
        super().__init__(sample_rate, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.freq_bins = n_fft // 2 + 1

        # 更新配置
        self.config.update({
            'n_fft': n_fft,
            'hop_length': hop_length,
            'max_length': max_length,
            'freq_bins': self.freq_bins
        })

    def process(self, audio_path: str) -> np.ndarray:
        """提取STFT频谱特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # 提取STFT频谱
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # 转换为对数刻度
            log_magnitude = np.log1p(magnitude)

            # 转置使时间维度在前
            spectrogram = log_magnitude.T  # (time_steps, freq_bins)

            # 填充或截断到固定长度
            spectrogram = self._normalize_length(spectrogram)

            return spectrogram.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"处理音频文件 {audio_path} 时出错: {e}")

    def _normalize_length(self, spectrogram: np.ndarray) -> np.ndarray:
        """标准化序列长度"""
        if len(spectrogram) > self.max_length:
            return spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            return np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')

    def get_feature_shape(self) -> Tuple[int, int]:
        """获取特征形状"""
        return (self.max_length, self.freq_bins)


class MelSpectrogramPreprocessor(AudioPreprocessor):
    """Mel频谱预处理器 - 可扩展示例"""

    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024,
                 hop_length: int = 512, n_mels: int = 128, max_length: int = 200, **kwargs):
        super().__init__(sample_rate, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length = max_length

        self.config.update({
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'max_length': max_length
        })

    def process(self, audio_path: str) -> np.ndarray:
        """提取Mel频谱特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # 提取Mel频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels
            )

            # 转换为对数刻度
            log_mel = np.log1p(mel_spec)

            # 转置使时间维度在前
            mel_spectrogram = log_mel.T  # (time_steps, n_mels)

            # 标准化长度
            mel_spectrogram = self._normalize_length(mel_spectrogram)

            return mel_spectrogram.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"处理音频文件 {audio_path} 时出错: {e}")

    def _normalize_length(self, spectrogram: np.ndarray) -> np.ndarray:
        """标准化序列长度"""
        if len(spectrogram) > self.max_length:
            return spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            return np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')

    def get_feature_shape(self) -> Tuple[int, int]:
        """获取特征形状"""
        return (self.max_length, self.n_mels)


class PreprocessorFactory:
    """预处理器工厂"""

    _preprocessors = {
        'spectrogram': SpectrogramPreprocessor,
        'mel_spectrogram': MelSpectrogramPreprocessor,
    }

    @classmethod
    def create(cls, preprocessor_type: str, **kwargs) -> AudioPreprocessor:
        """创建预处理器"""
        if preprocessor_type not in cls._preprocessors:
            raise ValueError(f"不支持的预处理器类型: {preprocessor_type}")

        return cls._preprocessors[preprocessor_type](**kwargs)

    @classmethod
    def register(cls, name: str, preprocessor_class: type):
        """注册新的预处理器"""
        cls._preprocessors[name] = preprocessor_class

    @classmethod
    def list_available(cls):
        """列出可用的预处理器"""
        return list(cls._preprocessors.keys())


class OfflinePreprocessor:
    """离线预处理器 - 批量处理和缓存"""

    def __init__(self, preprocessor: AudioPreprocessor, cache_dir: str = None):
        self.preprocessor = preprocessor
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def process_file(self, audio_path: str, force_recompute: bool = False) -> np.ndarray:
        """处理单个文件，支持缓存"""
        if self.cache_dir:
            cache_path = self._get_cache_path(audio_path)

            # 检查缓存
            if not force_recompute and os.path.exists(cache_path):
                try:
                    return np.load(cache_path)
                except Exception:
                    pass  # 缓存损坏，重新计算

        # 计算特征
        features = self.preprocessor.process(audio_path)

        # 保存缓存
        if self.cache_dir:
            np.save(cache_path, features)

        return features

    def _get_cache_path(self, audio_path: str) -> str:
        """获取缓存文件路径"""
        audio_name = Path(audio_path).stem
        cache_name = f"{audio_name}_{self._get_config_hash()}.npy"
        return os.path.join(self.cache_dir, cache_name)

    def _get_config_hash(self) -> str:
        """获取配置哈希值"""
        config_str = json.dumps(self.preprocessor.get_config(), sort_keys=True)
        return str(hash(config_str))[:8]

    def save_config(self, config_path: str):
        """保存预处理配置"""
        config = {
            'preprocessor_type': self.preprocessor.__class__.__name__,
            'config': self.preprocessor.get_config(),
            'feature_shape': self.preprocessor.get_feature_shape()
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_config(cls, config_path: str, cache_dir: str = None):
        """从配置文件加载预处理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        preprocessor_type = config['preprocessor_type'].lower().replace('preprocessor', '')
        preprocessor = PreprocessorFactory.create(preprocessor_type, **config['config'])

        return cls(preprocessor, cache_dir)


# 便捷函数
def create_spectrogram_preprocessor(**kwargs) -> SpectrogramPreprocessor:
    """创建STFT频谱预处理器"""
    return SpectrogramPreprocessor(**kwargs)


def create_mel_spectrogram_preprocessor(**kwargs) -> MelSpectrogramPreprocessor:
    """创建Mel频谱预处理器"""
    return MelSpectrogramPreprocessor(**kwargs)


if __name__ == "__main__":
    # 使用示例
    print("🎯 音频预处理模块测试")
    print("=" * 50)

    # 创建预处理器
    preprocessor = PreprocessorFactory.create('spectrogram')
    print(f"预处理器配置: {preprocessor.get_config()}")
    print(f"特征形状: {preprocessor.get_feature_shape()}")

    # 创建离线预处理器
    offline_processor = OfflinePreprocessor(preprocessor, cache_dir='cache/features')

    print(f"可用预处理器: {PreprocessorFactory.list_available()}")