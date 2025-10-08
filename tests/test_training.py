#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块测试
测试训练器、配置管理和回调功能
"""

import sys
import os
import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from wavespectra2text.training.trainer import BaseTrainer, create_trainer
from wavespectra2text.training.config import ConfigManager, get_default_config
from wavespectra2text.training.callbacks import (
    CallbackList, EarlyStoppingCallback, CheckpointCallback, 
    LoggingCallback, TensorBoardCallback
)
from wavespectra2text.core.model import create_model
from wavespectra2text.core.vocab import vocab
from wavespectra2text.data.dataset import AudioDataset


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # 创建测试配置
        config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'hidden_dim': 256,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'model': {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10
            },
            'data': {
                'max_length': 200,
                'sample_rate': 48000
            }
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """测试配置加载"""
        config_manager = ConfigManager(self.config_file)
        config = config_manager.get_config()
        
        self.assertIn('model', config)
        self.assertIn('training', config)
        self.assertIn('data', config)
        
        self.assertEqual(config['model']['hidden_dim'], 256)
        self.assertEqual(config['training']['batch_size'], 32)
    
    def test_default_config(self):
        """测试默认配置"""
        config = get_default_config()
        
        # 检查必需的配置项
        required_keys = ['batch_size', 'learning_rate', 'num_epochs', 'hidden_dim', 'encoder_layers', 'decoder_layers']
        for key in required_keys:
            self.assertIn(key, config)
        
        # 检查默认值
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
        self.assertIn('num_epochs', config)
    
    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigManager(self.config_file)
        
        # 测试有效配置
        valid_config = config_manager.get_config()
        self.assertTrue(config_manager.validate_config(valid_config))
        
        # 测试无效配置
        invalid_config = {'invalid': 'config'}
        self.assertFalse(config_manager.validate_config(invalid_config))


class TestCallbacks(unittest.TestCase):
    """测试训练回调"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_early_stopping_callback(self):
        """测试早停回调"""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.001)
        
        # 模拟训练过程
        for epoch in range(10):
            val_loss = 1.0 - epoch * 0.1  # 递减的验证损失
            should_stop = callback.on_epoch_end(epoch, {'val_loss': val_loss})
            
            if epoch < 3:
                self.assertFalse(should_stop)
            else:
                # 由于损失持续下降，不应该早停
                self.assertFalse(should_stop)
    
    def test_checkpoint_callback(self):
        """测试检查点回调"""
        callback = CheckpointCallback(
            checkpoint_dir=self.checkpoint_dir,
            save_best=True,
            monitor='val_loss'
        )
        
        # 创建模拟模型
        model = create_model(device=torch.device('cpu'))
        
        # 模拟保存检查点
        callback.on_epoch_end(0, {'val_loss': 0.5}, model=model)
        
        # 检查文件是否创建
        checkpoint_files = os.listdir(self.checkpoint_dir)
        self.assertGreater(len(checkpoint_files), 0)
    
    def test_logging_callback(self):
        """测试日志回调"""
        log_file = os.path.join(self.log_dir, 'training.log')
        callback = LoggingCallback(log_file=log_file)
        
        # 模拟训练日志
        callback.on_epoch_end(0, {'train_loss': 0.5, 'val_loss': 0.6})
        
        # 检查日志文件是否创建
        self.assertTrue(os.path.exists(log_file))
        
        # 检查日志内容
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('train_loss', content)
            self.assertIn('val_loss', content)
    
    def test_callback_list(self):
        """测试回调列表"""
        callback1 = EarlyStoppingCallback(patience=2)
        callback2 = LoggingCallback(log_file=os.path.join(self.log_dir, 'test.log'))
        
        callback_list = CallbackList([callback1, callback2])
        
        # 测试回调执行
        should_stop = callback_list.on_epoch_end(0, {'val_loss': 0.5})
        self.assertFalse(should_stop)


class TestTrainer(unittest.TestCase):
    """测试训练器"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_model(device=self.device)
        
        # 创建模拟数据
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 513
        
        # 创建模拟数据加载器
        self.train_data = []
        self.val_data = []
        
        for _ in range(10):  # 10个训练样本
            features = torch.randn(self.seq_len, self.input_dim)
            labels = torch.randint(0, vocab.vocab_size, (20,))  # 随机标签
            self.train_data.append((features, labels))
        
        for _ in range(5):  # 5个验证样本
            features = torch.randn(self.seq_len, self.input_dim)
            labels = torch.randint(0, vocab.vocab_size, (20,))
            self.val_data.append((features, labels))
        
        # 创建配置
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 2,
            'num_epochs': 2,
            'experiment_name': 'test_experiment',
            'label_smoothing': 0.1
        }
    
    def test_trainer_creation(self):
        """测试训练器创建"""
        # 创建模拟数据加载器
        train_loader = self.train_data
        val_loader = self.val_data
        
        trainer = create_trainer('small', self.model, train_loader, val_loader, self.device, self.config)
        
        self.assertIsInstance(trainer, BaseTrainer)
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.model, self.model)
    
    def test_trainer_optimizer(self):
        """测试优化器创建"""
        train_loader = self.train_data
        val_loader = self.val_data
        
        trainer = create_trainer('small', self.model, train_loader, val_loader, self.device, self.config)
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
    
    def test_trainer_training_step(self):
        """测试训练步骤"""
        train_loader = self.train_data
        val_loader = self.val_data
        
        trainer = create_trainer('small', self.model, train_loader, val_loader, self.device, self.config)
        
        # 测试单个训练步骤
        features, labels = self.train_data[0]
        features = features.unsqueeze(0)  # 添加batch维度
        labels = labels.unsqueeze(0)
        
        loss = trainer._train_step(features, labels)
        
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_trainer_validation_step(self):
        """测试验证步骤"""
        train_loader = self.train_data
        val_loader = self.val_data
        
        trainer = create_trainer('small', self.model, train_loader, val_loader, self.device, self.config)
        
        # 测试验证步骤
        features, labels = self.val_data[0]
        features = features.unsqueeze(0)
        labels = labels.unsqueeze(0)
        
        loss = trainer._validate_step(features, labels)
        
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)


class TestTrainingIntegration(unittest.TestCase):
    """训练集成测试"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建配置
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 2,
            'num_epochs': 1,
            'experiment_name': 'integration_test',
            'label_smoothing': 0.1
        }
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_training_workflow(self):
        """测试完整训练流程"""
        # 创建模型
        model = create_model(device=self.device)
        
        # 创建模拟数据
        train_data = []
        val_data = []
        
        for _ in range(6):  # 6个训练样本
            features = torch.randn(100, 513)
            labels = torch.randint(0, vocab.vocab_size, (20,))
            train_data.append((features, labels))
        
        for _ in range(3):  # 3个验证样本
            features = torch.randn(100, 513)
            labels = torch.randint(0, vocab.vocab_size, (20,))
            val_data.append((features, labels))
        
        # 创建训练器
        trainer = create_trainer('small', model, train_data, val_data, self.device, self.config)
        
        # 运行一个epoch
        trainer.train(1)
        
        # 检查训练状态
        self.assertEqual(trainer.epoch, 1)
        self.assertIsNotNone(trainer.best_val_loss)


if __name__ == '__main__':
    unittest.main(verbosity=2)
