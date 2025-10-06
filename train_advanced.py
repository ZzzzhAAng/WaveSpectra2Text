#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级训练脚本
集成学习率调度、验证集分割、更好的监控等改进
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
from sklearn.model_selection import train_test_split
import pandas as pd

from model import create_model
from data_utils import get_dataloader
from vocab import vocab


class AdvancedTrainer:
    """高级训练器 - 集成多种改进"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器
        self.optimizer = optim.AdamW(  # 使用AdamW替代Adam
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器 - 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_T0', 10),  # 重启周期
            T_mult=config.get('scheduler_T_mult', 2),  # 周期倍数
            eta_min=config.get('scheduler_eta_min', 1e-6)  # 最小学习率
        )
        
        # 学习率预热
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.warmup_epochs
        )
        
        # 损失函数 - 标签平滑
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.get_padding_idx(),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # 日志和监控
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # 梯度监控
        self.log_gradients = config.get('log_gradients', False)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')
        
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            mask = (tgt_output != vocab.get_padding_idx())
            correct = (predictions == tgt_output) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # 更新进度条
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'grad_norm': f'{grad_norm:.3f}'
            })
            
            # 记录到tensorboard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
            self.writer.add_scalar('Train/Gradient_Norm', grad_norm, global_step)
            
            # 梯度监控
            if self.log_gradients and batch_idx % 10 == 0:
                self._log_gradients(global_step)
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 兼容新旧接口
                if 'features' in batch:
                    spectrograms = batch['features'].to(self.device)
                else:
                    spectrograms = batch['spectrograms'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                
                outputs = self.model(spectrograms, tgt_input)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                total_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                mask = (tgt_output != vocab.get_padding_idx())
                correct = (predictions == tgt_output) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def _log_gradients(self, step):
        """记录梯度信息"""
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每层的梯度范数
                self.writer.add_scalar(f'Gradients/{name}', param_norm, step)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Gradients/Total_Norm', total_norm, step)
    
    def save_checkpoint(self, epoch, val_loss, val_acc, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        print(f"检查点已保存: {filename}")
    
    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始高级训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # 学习率调度
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 记录到tensorboard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)
            
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.3f}")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.3f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc, "checkpoints/best_model.pth")
                print(f"  🎉 新的最佳验证准确率: {val_acc:.3f}")
            else:
                self.patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")
            
            # 早停检查
            patience = self.config.get('patience', 20)
            if self.patience_counter >= patience:
                print(f"验证准确率连续 {patience} 个epoch未改善，提前停止训练")
                break
        
        print(f"训练完成！最佳验证准确率: {self.best_val_acc:.3f}")
        self.writer.close()


def create_train_val_split(labels_file, val_ratio=0.2, random_state=42):
    """创建训练验证集分割"""
    df = pd.read_csv(labels_file)
    
    # 按类别分层分割
    train_df, val_df = train_test_split(
        df, 
        test_size=val_ratio,
        random_state=random_state,
        stratify=df['label']
    )
    
    # 保存分割后的文件
    train_file = labels_file.replace('.csv', '_train.csv')
    val_file = labels_file.replace('.csv', '_val.csv')
    
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    val_df.to_csv(val_file, index=False, encoding='utf-8')
    
    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_df)} 样本 -> {train_file}")
    print(f"  验证集: {len(val_df)} 样本 -> {val_file}")
    
    return train_file, val_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("🚀 高级训练脚本")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"实验名称: {config['experiment_name']}")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    try:
        # 创建训练验证集分割
        train_labels_file, val_labels_file = create_train_val_split(
            config['labels_file'], 
            val_ratio=args.val_ratio
        )
        
        # 创建数据加载器
        train_loader = get_dataloader(
            audio_dir=config['audio_dir'],
            labels_file=train_labels_file,
            batch_size=config['batch_size'],
            shuffle=True,
            mode='auto'
        )
        
        val_loader = get_dataloader(
            audio_dir=config['audio_dir'],
            labels_file=val_labels_file,
            batch_size=config['batch_size'],
            shuffle=False,
            mode='auto'
        )
        
        # 创建模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=config['hidden_dim'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # 创建训练器
        trainer = AdvancedTrainer(model, train_loader, val_loader, device, config)
        
        # 开始训练
        trainer.train(config['num_epochs'])
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        trainer.save_checkpoint(-1, float('inf'), 0, "checkpoints/interrupted.pth")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()