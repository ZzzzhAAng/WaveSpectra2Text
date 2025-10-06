# -*- coding: utf-8 -*-
"""
简化的训练脚本 - 专门处理小数据集
适用于每个标签只有少量样本的情况
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

from model import create_model
from data_utils import get_dataloader, check_audio_files
from vocab import vocab


class SimpleTrainer:
    def __init__(self, model, dataloader, device, config):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config

        # 优化器 - 针对小数据集的设置
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器 - 更保守的设置
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,  # 每20个epoch降低学习率
            gamma=0.8  # 降低到80%
        )

        # 损失函数 - 添加标签平滑
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.get_padding_idx(),
            label_smoothing=0.1
        )

        # 日志
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")

        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = []
        self.patience_counter = 0
        self.max_patience = 15

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.dataloader)

        progress_bar = tqdm(self.dataloader, desc=f'Epoch {self.epoch + 1}')

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

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

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        self.losses.append(avg_loss)

        return avg_loss, accuracy

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'losses': self.losses,
            'config': self.config
        }

        torch.save(checkpoint, filename)
        print(f"检查点已保存: {filename}")

    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"数据集大小: {len(self.dataloader.dataset)}")
        print(f"批次数量: {len(self.dataloader)}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 学习率调度
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']

            # 记录到tensorboard
            self.writer.add_scalar('Epoch/Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/LR', new_lr, epoch)

            # 打印结果
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Accuracy: {train_acc:.4f}")
            print(f"  LR: {new_lr:.6f}")

            if new_lr != old_lr:
                print(f"  -> 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")

            # 保存最佳模型
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.patience_counter = 0
                self.save_checkpoint(f"checkpoints/best_model.pth")
                print("  -> 新的最佳模型!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    print(f"  -> 早停: 损失连续{self.max_patience}个epoch未改善")
                    break

            # 定期保存检查点
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")

            print("-" * 50)

        print("训练完成!")
        self.writer.close()


def main():
    """主函数"""
    # 针对小数据集优化的配置
    config = {
        'experiment_name': f'simple_speech_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 1,  # 极小批大小
        'learning_rate': 1e-5,  # 很低的学习率
        'weight_decay': 1e-3,  # 强正则化
        'grad_clip': 0.1,  # 强梯度裁剪
        'num_epochs': 30,  # 较少的训练轮数
        'save_every': 10,
        'hidden_dim': 64,  # 很小的模型
        'encoder_layers': 1,  # 最少的层数
        'decoder_layers': 1,
        'dropout': 0.5,  # 很高的dropout
        'audio_dir': 'data/audio',
        'labels_file': 'data/labels.csv'
    }

    print("🎯 简化训练脚本 - 小数据集专用")
    print("配置特点:")
    print("  ✅ 极小模型 (hidden_dim=64, 1层)")
    print("  ✅ 强正则化 (dropout=0.5, weight_decay=1e-3)")
    print("  ✅ 低学习率 (1e-5)")
    print("  ✅ 小批大小 (batch_size=1)")
    print("  ✅ 早停机制")
    print("=" * 60)

    # 检查音频文件
    if not check_audio_files(config['audio_dir'], config['labels_file']):
        print("错误: 音频文件检查失败")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    try:
        # 创建数据加载器 - 使用全部数据
        dataloader = get_dataloader(
            audio_dir=config['audio_dir'],
            labels_file=config['labels_file'],
            batch_size=config['batch_size'],
            shuffle=True
        )

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建小模型
    model = create_model(
        vocab_size=vocab.vocab_size,
        hidden_dim=config['hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        dropout=config['dropout']
    ).to(device)

    print(f"模型大小: {sum(p.numel() for p in model.parameters())} 参数")

    # 创建训练器
    trainer = SimpleTrainer(model, dataloader, device, config)

    # 开始训练
    try:
        trainer.train(config['num_epochs'])
    except KeyboardInterrupt:
        print("训练被中断")
        trainer.save_checkpoint("checkpoints/interrupted.pth")


if __name__ == "__main__":
    main()