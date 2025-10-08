#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练基类
解决不同规模训练脚本中的代码冗余问题
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from ..core.model import create_model
from ..core.vocab import vocab
from ..data.dataset import AudioDataset


class BaseTrainer(ABC):
    """统一训练基类 - 解决训练脚本代码冗余"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.get_padding_idx(),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # 日志
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.max_patience = config.get('max_patience', 15)
    
    @abstractmethod
    def _create_optimizer(self):
        """创建优化器 - 子类实现"""
        pass
    
    @abstractmethod
    def _create_scheduler(self):
        """创建学习率调度器 - 子类实现"""
        pass
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 兼容新旧接口
            if 'features' in batch:
                spectrograms = batch['features'].to(self.device)
            else:
                spectrograms = batch['spectrograms'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 准备输入和目标
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(spectrograms, tgt_input)
            
            # 计算损失
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            predictions = outputs.argmax(dim=-1)
            mask = (tgt_output != vocab.get_padding_idx())
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # 更新进度条
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{current_acc:.3f}'
            })
            
            # 记录到tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', current_acc, global_step)
        
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 兼容新旧接口
                if 'features' in batch:
                    spectrograms = batch['features'].to(self.device)
                else:
                    spectrograms = batch['spectrograms'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 准备输入和目标
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                
                # 前向传播
                outputs = self.model(spectrograms, tgt_input)
                
                # 计算损失
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                total_loss += loss.item()
                
                # 计算准确率
                predictions = outputs.argmax(dim=-1)
                mask = (tgt_output != vocab.get_padding_idx())
                correct = (predictions == tgt_output) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"检查点已加载: {filepath}")
        print(f"从第 {self.epoch + 1} 轮开始训练")
    
    def train(self, num_epochs: int, resume_path: Optional[str] = None):
        """训练主循环"""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
        
        print(f"开始训练 - 共 {num_epochs} 轮")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 记录到tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.3f}")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.3f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("checkpoints/best_model.pth")
                print(f"  ✅ 新的最佳模型 (验证损失: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  ⏳ 早停计数: {self.patience_counter}/{self.max_patience}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
            
            # 早停
            if self.patience_counter >= self.max_patience:
                print(f"早停触发 - 验证损失 {self.max_patience} 轮未改善")
                break
            
            print("-" * 60)
        
        self.writer.close()
        print("训练完成!")


class SimpleTrainer(BaseTrainer):
    """小数据集训练器"""
    
    def _create_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def _create_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.8
        )


class ImprovedTrainer(BaseTrainer):
    """中等数据集训练器"""
    
    def _create_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def _create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6
        )


class LargeDatasetTrainer(BaseTrainer):
    """大数据集训练器"""
    
    def _create_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.98)
        )
    
    def _create_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )


def create_trainer(trainer_type: str, model, train_loader, val_loader, device, config):
    """创建训练器工厂函数"""
    trainers = {
        'simple': SimpleTrainer,
        'improved': ImprovedTrainer,
        'large': LargeDatasetTrainer
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"不支持的训练器类型: {trainer_type}")
    
    return trainers[trainer_type](model, train_loader, val_loader, device, config)


def split_dataset(audio_dir: str, labels_file: str, test_size: float = 0.2, random_state: int = 42):
    """分割数据集"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    labels_df = pd.read_csv(labels_file)
    
    # 按标签分层分割
    train_df, val_df = train_test_split(
        labels_df,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_df['label']
    )
    
    return train_df, val_df


def create_dataloader_from_df(df, audio_dir: str, batch_size: int, shuffle: bool = True):
    """从DataFrame创建数据加载器"""
    import tempfile
    
    # 创建临时标签文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_labels_file = f.name
    
    try:
        # 创建数据集
        dataset = AudioDataset(
            labels_file=temp_labels_file,
            audio_dir=audio_dir,
            mode='realtime'
        )
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False
        )
        return dataloader
    finally:
        # 清理临时文件
        if os.path.exists(temp_labels_file):
            os.remove(temp_labels_file)


def get_default_config(scale: str) -> Dict[str, Any]:
    """获取不同规模的默认配置"""
    configs = {
        'small': {
            'experiment_name': f'small_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 1,
            'learning_rate': 1e-5,
            'weight_decay': 1e-3,
            'grad_clip': 0.1,
            'num_epochs': 30,
            'save_every': 10,
            'hidden_dim': 64,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'dropout': 0.5,
            'max_patience': 15,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'medium': {
            'experiment_name': f'medium_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'grad_clip': 0.5,
            'num_epochs': 50,
            'save_every': 10,
            'hidden_dim': 128,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'dropout': 0.3,
            'max_patience': 20,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'large': {
            'experiment_name': f'large_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'num_epochs': 100,
            'save_every': 20,
            'hidden_dim': 256,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'dropout': 0.2,
            'max_patience': 25,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'xlarge': {
            'experiment_name': f'xlarge_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 8,
            'learning_rate': 2e-4,
            'weight_decay': 1e-6,
            'grad_clip': 1.0,
            'num_epochs': 200,
            'save_every': 20,
            'hidden_dim': 512,
            'encoder_layers': 6,
            'decoder_layers': 6,
            'dropout': 0.1,
            'max_patience': 30,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        }
    }
    
    return configs.get(scale, configs['small'])


if __name__ == "__main__":
    print("🧪 统一训练基类测试")
    print("=" * 50)
    print("📋 功能列表:")
    print("  ✅ 统一训练基类")
    print("  ✅ 三种训练器类型")
    print("  ✅ 数据集分割功能")
    print("  ✅ 配置管理")
    print("  ✅ 检查点保存/加载")
    print("  ✅ 早停机制")
    print("  ✅ TensorBoard日志")
    
    print("\n💡 使用方式:")
    print("from train_base import create_trainer, get_default_config")
    print("trainer = create_trainer('simple', model, train_loader, val_loader, device, config)")
    print("trainer.train(num_epochs)")
