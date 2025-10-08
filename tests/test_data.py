#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块测试
测试音频预处理、数据集和工具功能
"""

import sys
import os
import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from wavespectra2text.data.preprocessing import (
    AudioPreprocessor, SpectrogramPreprocessor, MelSpectrogramPreprocessor,
    PreprocessorFactory, OfflinePreprocessor
)
from wavespectra2text.data.dataset import AudioDataset
from wavespectra2text.data.utils import AudioProcessor, LabelManager, FileUtils
from wavespectra2text.core.vocab import vocab


class TestAudioPreprocessor(unittest.TestCase):
    """测试音频预处理器"""
    
    def setUp(self):
        self.sample_rate = 48000
        self.n_fft = 1024
        self.hop_length = 512
        self.max_length = 200
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.audio_file = os.path.join(self.temp_dir, 'test_audio.wav')
        
        # 创建测试音频文件（使用numpy生成）
        duration = 2.0  # 2秒
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        
        # 保存为WAV文件
        import soundfile as sf
        sf.write(self.audio_file, audio, self.sample_rate)
    
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_spectrogram_preprocessor(self):
        """测试STFT频谱预处理器"""
        preprocessor = SpectrogramPreprocessor(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            max_length=self.max_length
        )
        
        # 处理音频
        features = preprocessor.process(self.audio_file)
        
        # 检查输出
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 2)  # (time, freq)
        self.assertEqual(features.shape[0], self.max_length)
        self.assertEqual(features.shape[1], self.n_fft // 2 + 1)
    
    def test_mel_spectrogram_preprocessor(self):
        """测试Mel频谱预处理器"""
        preprocessor = MelSpectrogramPreprocessor(
            sample_rate=self.sample_rate,
            n_mels=80,
            max_length=self.max_length
        )
        
        # 处理音频
        features = preprocessor.process(self.audio_file)
        
        # 检查输出
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features.shape), 2)  # (time, freq)
        self.assertEqual(features.shape[0], self.max_length)
        self.assertEqual(features.shape[1], 80)
    
    def test_preprocessor_factory(self):
        """测试预处理器工厂"""
        # 测试STFT预处理器创建
        stft_preprocessor = PreprocessorFactory.create_preprocessor('stft')
        self.assertIsInstance(stft_preprocessor, SpectrogramPreprocessor)
        
        # 测试Mel预处理器创建
        mel_preprocessor = PreprocessorFactory.create_preprocessor('mel')
        self.assertIsInstance(mel_preprocessor, MelSpectrogramPreprocessor)
        
        # 测试无效类型
        with self.assertRaises(ValueError):
            PreprocessorFactory.create_preprocessor('invalid')


class TestAudioDataset(unittest.TestCase):
    """测试音频数据集"""
    
    def setUp(self):
        # 创建临时目录和文件
        self.temp_dir = tempfile.mkdtemp()
        self.audio_dir = os.path.join(self.temp_dir, 'audio')
        self.features_dir = os.path.join(self.temp_dir, 'features')
        self.labels_file = os.path.join(self.temp_dir, 'labels.csv')
        
        os.makedirs(self.audio_dir)
        os.makedirs(self.features_dir)
        
        # 创建测试音频文件
        import soundfile as sf
        sample_rate = 48000
        duration = 1.0
        
        for i in range(3):
            audio_file = os.path.join(self.audio_dir, f'test_{i}.wav')
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
            sf.write(audio_file, audio, sample_rate)
        
        # 创建标签文件
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            f.write('filename,text\n')
            f.write('test_0.wav,一\n')
            f.write('test_1.wav,二\n')
            f.write('test_2.wav,三\n')
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_audio_dataset_creation(self):
        """测试音频数据集创建"""
        dataset = AudioDataset(
            audio_dir=self.audio_dir,
            labels_file=self.labels_file,
            vocab=vocab,
            max_length=200
        )
        
        self.assertEqual(len(dataset), 3)
    
    def test_audio_dataset_getitem(self):
        """测试数据集索引访问"""
        dataset = AudioDataset(
            audio_dir=self.audio_dir,
            labels_file=self.labels_file,
            vocab=vocab,
            max_length=200
        )
        
        # 获取第一个样本
        features, labels = dataset[0]
        
        # 检查特征
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(len(features.shape), 2)  # (time, freq)
        
        # 检查标签
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(len(labels.shape), 1)  # (seq_len,)
    
    def test_audio_dataset_with_features(self):
        """测试使用预计算特征的数据集"""
        # 创建预计算特征
        preprocessor = SpectrogramPreprocessor()
        for i in range(3):
            audio_file = os.path.join(self.audio_dir, f'test_{i}.wav')
            features = preprocessor.process(audio_file)
            feature_file = os.path.join(self.features_dir, f'test_{i}.npy')
            np.save(feature_file, features)
        
        # 创建使用预计算特征的数据集
        dataset = AudioDataset(
            audio_dir=self.audio_dir,
            labels_file=self.labels_file,
            vocab=vocab,
            features_dir=self.features_dir,
            max_length=200
        )
        
        self.assertEqual(len(dataset), 3)
        
        # 测试数据访问
        features, labels = dataset[0]
        self.assertIsInstance(features, torch.Tensor)


class TestDataUtils(unittest.TestCase):
    """测试数据工具"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.audio_dir = os.path.join(self.temp_dir, 'audio')
        os.makedirs(self.audio_dir)
        
        # 创建测试音频文件
        import soundfile as sf
        sample_rate = 48000
        duration = 1.0
        
        for i in range(3):
            audio_file = os.path.join(self.audio_dir, f'test_{i}.wav')
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t)
            sf.write(audio_file, audio, sample_rate)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_audio_processor(self):
        """测试音频处理器"""
        processor = AudioProcessor()
        
        # 测试音频加载
        audio_file = os.path.join(self.audio_dir, 'test_0.wav')
        audio, sr = processor.load_audio(audio_file)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 48000)
        
        # 测试频谱提取
        spectrogram = processor.extract_spectrogram(audio)
        self.assertIsInstance(spectrogram, np.ndarray)
        self.assertEqual(len(spectrogram.shape), 2)
    
    def test_label_manager(self):
        """测试标签管理器"""
        # 测试音频文件扫描
        audio_files = LabelManager.scan_audio_files(self.audio_dir)
        self.assertEqual(len(audio_files), 3)
        
        # 测试标签模板创建
        labels_file = os.path.join(self.temp_dir, 'labels.csv')
        LabelManager.create_labels_template(audio_files, labels_file)
        
        self.assertTrue(os.path.exists(labels_file))
        
        # 检查文件内容
        with open(labels_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('filename', content)
            self.assertIn('text', content)
    
    def test_file_utils(self):
        """测试文件工具"""
        # 测试目录创建
        test_dir = os.path.join(self.temp_dir, 'test_dir')
        FileUtils.ensure_dir(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        
        # 测试文件大小
        audio_file = os.path.join(self.audio_dir, 'test_0.wav')
        size = FileUtils.get_file_size(audio_file)
        self.assertGreater(size, 0)
        
        # 测试音频文件检查
        self.assertTrue(FileUtils.is_audio_file(audio_file))
        self.assertFalse(FileUtils.is_audio_file('test.txt'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
