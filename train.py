# -*- coding: utf-8 -*-
"""
训练脚本
用于训练从频谱到文本的语音识别模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from model import create_model
from data_utils import get_dataloader, create_labels_file_if_not_exists, check_audio_files
from vocab import vocab

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.get_padding_idx())
        
        # 日志
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            spectrograms = batch['spectrograms'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 准备输入和目标
            # 输入：除了最后一个token
            # 目标：除了第一个token
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 创建掩码
            src_mask = None  # 暂时不使用源掩码
            tgt_mask = None  # 模型内部会生成
            
            outputs = self.model(spectrograms, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 记录到tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
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
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        self.val_losses.append(avg_loss)
        
        # 记录到tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, self.epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, self.epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        print(f"检查点已保存: {filename}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            print(f"检查点已加载: {filename}")
            return True
        return False
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"checkpoints/best_model.pth")
                print("  -> 新的最佳模型!")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
            
            print("-" * 50)
        
        print("训练完成!")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='训练语音识别模型')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--create_data', action='store_true', help='创建示例数据')
    
    args = parser.parse_args()
    
    # 默认配置
    default_config = {
        'experiment_name': f'speech_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'num_epochs': 100,
        'save_every': 10,
        'hidden_dim': 256,
        'encoder_layers': 4,
        'decoder_layers': 4,
        'dropout': 0.1,
        'audio_dir': 'data/audio',
        'labels_file': 'data/labels.csv'
    }
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        # 保存默认配置
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"已创建默认配置文件: {args.config}")
    
    # 检查和创建标签文件
    if args.create_data or not os.path.exists(config['labels_file']):
        print("检查标签文件...")
        create_labels_file_if_not_exists(config['labels_file'])
    
    # 检查音频文件
    print("检查音频文件...")
    if not check_audio_files(config['audio_dir'], config['labels_file']):
        print("错误: 部分音频文件缺失，请检查音频文件路径")
        print(f"音频目录: {config['audio_dir']}")
        print(f"标签文件: {config['labels_file']}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # 数据加载器
    try:
        train_loader = get_dataloader(
            audio_dir=config['audio_dir'],
            labels_file=config['labels_file'],
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # 使用相同数据作为验证集（在实际项目中应该分离）
        val_loader = get_dataloader(
            audio_dir=config['audio_dir'],
            labels_file=config['labels_file'],
            batch_size=config['batch_size'],
            shuffle=False
        )
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保安装了所需依赖: pip install librosa pandas tqdm tensorboard")
        return
    
    # 创建模型
    model = create_model(
        vocab_size=vocab.vocab_size,
        hidden_dim=config['hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 恢复训练
    if args.resume and trainer.load_checkpoint(args.resume):
        print("从检查点恢复训练")
    
    # 开始训练
    try:
        trainer.train(config['num_epochs'])
    except KeyboardInterrupt:
        print("训练被中断")
        trainer.save_checkpoint("checkpoints/interrupted.pth")

if __name__ == "__main__":
    main()