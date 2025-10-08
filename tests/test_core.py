#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块测试
测试模型、词汇表和推理核心的功能
"""

import sys
import os
import unittest
import torch
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from wavespectra2text.core.model import create_model, Seq2SeqModel
from wavespectra2text.core.vocab import vocab, Vocabulary
from wavespectra2text.core.inference import InferenceCore


class TestVocabulary(unittest.TestCase):
    """测试词汇表功能"""
    
    def setUp(self):
        self.vocab = Vocabulary()
    
    def test_vocab_size(self):
        """测试词汇表大小"""
        self.assertEqual(self.vocab.vocab_size, 14)  # 0-13
    
    def test_encode_decode(self):
        """测试编码和解码"""
        text = "一二三"
        encoded = self.vocab.encode(text)
        decoded = self.vocab.decode(encoded)
        
        # 解码结果应该不包含特殊标记
        self.assertNotIn('<SOS>', decoded)
        self.assertNotIn('<EOS>', decoded)
        
        # 应该包含实际文本
        self.assertEqual(decoded, text)
    
    def test_unknown_token(self):
        """测试未知字符处理"""
        text = "未知字符"
        encoded = self.vocab.encode(text)
        decoded = self.vocab.decode(encoded)
        
        # 应该包含UNK标记
        self.assertIn('<UNK>', decoded)
    
    def test_special_tokens(self):
        """测试特殊标记"""
        self.assertEqual(self.vocab.word_to_idx['<PAD>'], 0)
        self.assertEqual(self.vocab.word_to_idx['<SOS>'], 1)
        self.assertEqual(self.vocab.word_to_idx['<EOS>'], 2)
        self.assertEqual(self.vocab.word_to_idx['<UNK>'], 3)


class TestModel(unittest.TestCase):
    """测试模型功能"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_model(device=self.device)
    
    def test_model_creation(self):
        """测试模型创建"""
        self.assertIsInstance(self.model, Seq2SeqModel)
        self.assertEqual(self.model.device, self.device)
    
    def test_model_forward(self):
        """测试模型前向传播"""
        batch_size = 2
        seq_len = 100
        input_dim = 513
        
        # 创建模拟输入
        input_features = torch.randn(batch_size, seq_len, input_dim)
        
        # 创建目标序列（用于训练）
        tgt_len = 20
        tgt_tokens = torch.randint(0, vocab.vocab_size, (batch_size, tgt_len))
        
        # 前向传播
        with torch.no_grad():
            output = self.model(input_features, tgt_tokens)
        
        # 检查输出形状
        expected_shape = (batch_size, tgt_len, vocab.vocab_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_parameters(self):
        """测试模型参数"""
        # 检查模型是否有参数
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(param_count, 0)
        
        # 检查参数是否在CPU上
        for param in self.model.parameters():
            self.assertEqual(param.device, self.device)


class TestInferenceCore(unittest.TestCase):
    """测试推理核心功能"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        # 创建一个临时模型文件用于测试
        self.temp_model_path = "temp_test_model.pth"
        
        # 创建并保存模型
        model = create_model(device=self.device)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 0,
            'best_val_loss': 0.5,
            'config': {}
        }
        torch.save(checkpoint, self.temp_model_path)
        
        # 创建推理核心
        self.inference_core = InferenceCore(self.temp_model_path, self.device)
    
    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.temp_model_path):
            os.remove(self.temp_model_path)
    
    def test_inference_core_creation(self):
        """测试推理核心创建"""
        self.assertIsInstance(self.inference_core, InferenceCore)
        self.assertEqual(self.inference_core.device, self.device)
    
    def test_inference_core_model_loading(self):
        """测试模型加载"""
        self.assertIsInstance(self.inference_core.model, Seq2SeqModel)
    
    def test_inference_core_predict(self):
        """测试推理预测"""
        # 创建模拟输入 - 应该是2D (time, freq)
        seq_len = 100
        input_dim = 513
        
        input_features = np.random.randn(seq_len, input_dim)
        
        # 进行预测
        result = self.inference_core.infer_from_spectrogram(input_features)
        
        # 检查结果
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('score', result)  # 使用score而不是confidence
        self.assertIsInstance(result['text'], str)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_model(device=self.device)
        self.vocab = vocab
    
    def test_end_to_end_prediction(self):
        """端到端预测测试"""
        # 创建模拟频谱输入
        batch_size = 1
        seq_len = 100
        input_dim = 513
        
        input_features = torch.randn(batch_size, seq_len, input_dim)
        
        # 模型预测
        with torch.no_grad():
            # 创建目标序列（用于训练）
            tgt_len = 20
            tgt_tokens = torch.randint(0, vocab.vocab_size, (batch_size, tgt_len))
            logits = self.model(input_features, tgt_tokens)
        
        # 解码
        predicted_indices = torch.argmax(logits, dim=-1)
        predicted_text = self.vocab.decode(predicted_indices[0].cpu().numpy().tolist())
        
        # 检查结果
        self.assertIsInstance(predicted_text, str)
        self.assertGreater(len(predicted_text), 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
