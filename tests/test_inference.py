#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块测试
测试识别器功能
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

from wavespectra2text.inference.recognizer import DualInputSpeechRecognizer
from wavespectra2text.core.model import create_model
from wavespectra2text.core.vocab import vocab


class TestDualInputRecognizer(unittest.TestCase):
    """测试双输入识别器"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时模型文件
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
        model = create_model(device=self.device)
        torch.save(model.state_dict(), self.model_path)
        
        # 创建测试音频文件
        self.audio_file = os.path.join(self.temp_dir, 'test_audio.wav')
        self._create_test_audio()
        
        # 创建测试频谱文件
        self.spectrogram_file = os.path.join(self.temp_dir, 'test_spectrogram.npy')
        self._create_test_spectrogram()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio(self):
        """创建测试音频文件"""
        import soundfile as sf
        sample_rate = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        sf.write(self.audio_file, audio, sample_rate)
    
    def _create_test_spectrogram(self):
        """创建测试频谱文件"""
        # 创建模拟频谱特征
        spectrogram = np.random.randn(200, 513)  # (time, freq)
        np.save(self.spectrogram_file, spectrogram)
    
    def test_recognizer_creation(self):
        """测试识别器创建"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        self.assertIsNotNone(recognizer.model)
        self.assertEqual(recognizer.device, self.device)
        self.assertIsNotNone(recognizer.inference_core)
    
    def test_recognize_from_audio(self):
        """测试从音频文件识别"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        result = recognizer.recognize_from_audio(self.audio_file)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['processing_time'], float)
    
    def test_recognize_from_spectrogram(self):
        """测试从频谱文件识别"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        result = recognizer.recognize_from_spectrogram(self.spectrogram_file)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
    
    def test_auto_mode_detection(self):
        """测试自动模式检测"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        # 测试音频文件自动检测
        result = recognizer.recognize(self.audio_file, mode='auto')
        self.assertIsInstance(result, dict)
        
        # 测试频谱文件自动检测
        result = recognizer.recognize(self.spectrogram_file, mode='auto')
        self.assertIsInstance(result, dict)
    
    def test_batch_recognition(self):
        """测试批量识别"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        # 创建多个测试文件
        audio_files = []
        for i in range(3):
            audio_file = os.path.join(self.temp_dir, f'test_audio_{i}.wav')
            self._create_test_audio_file(audio_file)
            audio_files.append(audio_file)
        
        results = recognizer.batch_recognize(audio_files)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
    
    def _create_test_audio_file(self, file_path):
        """创建测试音频文件"""
        import soundfile as sf
        sample_rate = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(file_path, audio, sample_rate)
    
    def test_error_handling(self):
        """测试错误处理"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        # 测试不存在的文件
        with self.assertRaises(FileNotFoundError):
            recognizer.recognize_from_audio('nonexistent.wav')
        
        # 测试无效的频谱文件
        invalid_spectrogram_file = os.path.join(self.temp_dir, 'invalid.npy')
        np.save(invalid_spectrogram_file, np.array([1, 2, 3]))  # 错误的形状
        
        with self.assertRaises(ValueError):
            recognizer.recognize_from_spectrogram(invalid_spectrogram_file)


class TestInferenceIntegration(unittest.TestCase):
    """推理集成测试"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模型
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
        model = create_model(device=self.device)
        torch.save(model.state_dict(), self.model_path)
        
        # 创建测试数据
        self.audio_file = os.path.join(self.temp_dir, 'test.wav')
        self._create_test_audio()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio(self):
        """创建测试音频"""
        import soundfile as sf
        sample_rate = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(self.audio_file, audio, sample_rate)
    
    def test_end_to_end_recognition(self):
        """端到端识别测试"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        # 测试音频识别
        result = recognizer.recognize_from_audio(self.audio_file)
        
        # 验证结果结构
        expected_keys = ['text', 'confidence', 'processing_time', 'input_type']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # 验证结果类型
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['processing_time'], float)
        self.assertIsInstance(result['input_type'], str)
        
        # 验证置信度范围
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_different_decoding_strategies(self):
        """测试不同解码策略"""
        recognizer = DualInputSpeechRecognizer(self.model_path, self.device)
        
        # 测试不同解码策略
        strategies = ['greedy', 'beam_search', 'sampling']
        
        for strategy in strategies:
            result = recognizer.recognize_from_audio(
                self.audio_file, 
                decoding_strategy=strategy
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('confidence', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
