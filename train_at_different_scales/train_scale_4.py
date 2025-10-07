# -*- coding: utf-8 -*-
"""
大数据集训练脚本 - 适用于10000+样本
基于原始train.py，针对大数据集优化
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from model import create_model
from data_utils import get_dataloader, check_audio_files
from vocab import vocab


class LargeDatasetTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 优化器 - 大数据集可以用更高学习率
        self.optimizer = optim.AdamW(  # AdamW对大数据集更好
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98)  # 更适合Transformer
        )

        # 学习率调度器 - 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # 20个epoch后重启
            T_mult=2,  # 每次重启周期翻倍
            eta_min=1e-6
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.get_padding_idx(),
            label_smoothing=0.1
        )

        # 日志
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.max_patience = 10

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')

        for batch_idx, batch in enumerate(progress_bar):
            # 兼容新旧接口

            if 'features' in batch:

                spectrograms = batch['features'].to(self.device)

            else:

                spectrograms = (batch['features'] if 'features' in batch else batch['spectrograms']).to(self.device)
            labels = batch['labels'].to(self.device)

            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            self.optimizer.zero_grad()

            outputs = self.model(spectrograms, tgt_input)

            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt_output.reshape(-1)
            )

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
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
                # 兼容新旧接口

                if 'features' in batch:

                    spectrograms = batch['features'].to(self.device)

                else:

                    spectrograms = (batch['features'] if 'features' in batch else batch['spectrograms']).to(self.device)
                labels = batch['labels'].to(self.device)

                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]

                outputs = self.model(spectrograms, tgt_input)

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

    def train(self, num_epochs):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 学习率调度
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']

            # 打印结果
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  LR: {new_lr:.6f}")

            if abs(new_lr - old_lr) > 1e-7:
                print(f"  -> 学习率变化: {old_lr:.6f} -> {new_lr:.6f}")

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f"../checkpoints/best_model.pth")
                print("  -> 新的最佳模型!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    print(f"  -> 早停: 验证损失连续{self.max_patience}个epoch未改善")
                    break

            # 定期保存检查点
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")

            print("-" * 50)

        print("训练完成!")
        self.writer.close()


def split_large_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """分割大数据集"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"数据集分割:")
    print(f"  训练集: {len(train_dataset)} 样本 ({train_ratio * 100:.1f}%)")
    print(f"  验证集: {len(val_dataset)} 样本 ({val_ratio * 100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} 样本 ({(1 - train_ratio - val_ratio) * 100:.1f}%)")

    return train_dataset, val_dataset, test_dataset


def main():
    """主函数 - 大数据集配置"""

    # 大数据集优化配置
    config = {
        'experiment_name': f'large_dataset_speech_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 32,  # 大批大小
        'learning_rate': 3e-4,  # 较高学习率
        'weight_decay': 1e-4,  # 适中正则化
        'grad_clip': 1.0,  # 标准梯度裁剪
        'num_epochs': 200,  # 更多训练轮数
        'save_every': 20,
        'hidden_dim': 512,  # 大模型
        'encoder_layers': 6,  # 更多层
        'decoder_layers': 6,
        'dropout': 0.1,  # 较少dropout
        'audio_dir': 'data/audio',
        'labels_file': 'data/labels.csv'
    }

    print("🚀 大数据集训练脚本 (适用于10000+样本)")
    print("配置特点:")
    print("  ✅ 大模型 (hidden_dim=512, 6层)")
    print("  ✅ 高学习率 (3e-4)")
    print("  ✅ 大批大小 (32)")
    print("  ✅ 余弦退火学习率调度")
    print("  ✅ AdamW优化器")
    print("  ✅ 数据集自动分割 (80%/10%/10%)")
    print("=" * 60)

    # 检查音频文件
    if not check_audio_files(config['audio_dir'], config['labels_file']):
        print("错误: 音频文件检查失败")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../runs', exist_ok=True)

    try:
        # 创建完整数据集
        from data_utils import AudioSpectrogramDataset
        from torch.utils.data import DataLoader

        full_dataset = AudioSpectrogramDataset(
            config['audio_dir'],
            config['labels_file']
        )

        # 分割数据集
        train_dataset, val_dataset, _ = split_large_dataset(full_dataset)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,  # 多进程加载
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建大模型
    model = create_model(
        vocab_size=vocab.vocab_size,
        hidden_dim=config['hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        dropout=config['dropout']
    ).to(device)

    print(f"模型大小: {sum(p.numel() for p in model.parameters())} 参数")

    # 创建训练器
    trainer = LargeDatasetTrainer(model, train_loader, val_loader, device, config)

    # 开始训练
    try:
        trainer.train(config['num_epochs'])
    except KeyboardInterrupt:
        print("训练被中断")
        trainer.save_checkpoint("checkpoints/interrupted.pth")


if __name__ == "__main__":
    main()