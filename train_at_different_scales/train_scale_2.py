# -*- coding: utf-8 -*-
"""
改进的训练脚本
解决过拟合问题，添加数据分割和正则化
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

from model import create_model
from data_utils import AudioSpectrogramDataset, collate_fn
from vocab import vocab


class ImprovedTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 优化器 - 降低学习率
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器 - 更保守的设置
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,  # 更温和的衰减
            patience=10,  # 更大的耐心
            min_lr=1e-6
        )

        # 损失函数 - 添加标签平滑
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.get_padding_idx(),
            label_smoothing=0.1  # 标签平滑防止过拟合
        )

        # 日志
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.max_patience = 20  # 早停耐心

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

            # 更新进度条
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
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # 打印结果
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  LR: {new_lr:.6f}")

            if new_lr != old_lr:
                print(f"  -> 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")

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


def split_dataset(audio_dir, labels_file, test_size=0.2, random_state=42):
    """分割数据集为训练集和验证集"""
    import pandas as pd

    df = pd.read_csv(labels_file)

    # 检查数据集大小
    if len(df) < 5:
        print("⚠️  数据集太小，无法分割，使用全部数据进行训练和验证")
        return df, df

    # 检查每个标签的样本数
    label_counts = df['label'].value_counts()
    min_samples = label_counts.min()

    print(f"标签分布: {dict(label_counts)}")

    if min_samples < 2:
        print("⚠️  部分标签只有1个样本，无法进行分层分割")
        print("使用随机分割代替分层分割")

        # 使用简单的随机分割
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        # 可以进行分层分割
        print("使用分层分割保持标签分布")
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )

    print(f"数据分割结果:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")

    # 显示分割后的标签分布
    print(f"  训练集标签: {dict(train_df['label'].value_counts())}")
    print(f"  验证集标签: {dict(val_df['label'].value_counts())}")

    return train_df, val_df


def create_dataloader_from_df(df, audio_dir, batch_size, shuffle=True):
    """从DataFrame创建数据加载器 - 使用统一工具"""
    from data_utils import get_dataloader
    
    # 创建临时标签文件
    temp_labels_file = f"temp_labels_{hash(str(df.values.tolist()))}.csv"
    df.to_csv(temp_labels_file, index=False, encoding='utf-8')

    try:
        # 使用统一的数据加载器
        dataloader = get_dataloader(
            audio_dir=audio_dir,
            labels_file=temp_labels_file,
            batch_size=batch_size,
            shuffle=shuffle,
            mode='realtime'
        )
        return dataloader
    finally:
        # 清理临时文件
        if os.path.exists(temp_labels_file):
            os.remove(temp_labels_file)


def main():
    """主函数"""
    # 改进的默认配置
    config = {
        'experiment_name': f'improved_speech_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 2,  # 减小批大小
        'learning_rate': 5e-5,  # 降低学习率
        'weight_decay': 1e-4,  # 增加权重衰减
        'grad_clip': 0.5,  # 减小梯度裁剪
        'num_epochs': 50,  # 减少训练轮数
        'save_every': 10,
        'hidden_dim': 128,  # 减小模型大小
        'encoder_layers': 2,  # 减少层数
        'decoder_layers': 2,
        'dropout': 0.3,  # 增加dropout
        'audio_dir': 'data/audio',
        'labels_file': 'data/labels.csv'
    }

    print("🔧 改进的训练脚本")
    print("主要改进:")
    print("  ✅ 数据集分割 (训练/验证)")
    print("  ✅ 降低学习率和模型复杂度")
    print("  ✅ 添加正则化 (dropout, weight_decay, label_smoothing)")
    print("  ✅ 早停机制")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../runs', exist_ok=True)

    try:
        # 分割数据集
        train_df, val_df = split_dataset(
            config['audio_dir'],
            config['labels_file']
        )

        # 创建数据加载器
        train_loader = create_dataloader_from_df(
            train_df,
            config['audio_dir'],
            config['batch_size'],
            shuffle=True
        )

        val_loader = create_dataloader_from_df(
            val_df,
            config['audio_dir'],
            config['batch_size'],
            shuffle=False
        )

    except Exception as e:
        print(f"数据加载失败: {e}")
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
    trainer = ImprovedTrainer(model, train_loader, val_loader, device, config)

    # 开始训练
    try:
        trainer.train(config['num_epochs'])
    except KeyboardInterrupt:
        print("训练被中断")
        trainer.save_checkpoint("checkpoints/interrupted.pth")


if __name__ == "__main__":
    main()