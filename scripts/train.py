#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本
支持不同规模的数据集训练
"""

import sys
import os
import argparse

from wavespectra2text.training.config import ConfigManager, get_default_config
from wavespectra2text.training.trainer import create_trainer
from wavespectra2text.core.model import create_model
from wavespectra2text.core.vocab import vocab
from wavespectra2text.data.dataset import AudioDataset
from wavespectra2text.training.callbacks import CallbackList, EarlyStoppingCallback, CheckpointCallback, LoggingCallback, TensorBoardCallback


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='WaveSpectra2Text 训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large', 'xlarge'], 
                        default='medium', help='数据集规模')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                        default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        config_manager = ConfigManager(args.config)
        config = config_manager.config
    else:
        config = get_default_config(args.scale)
    
    # 设置设备
    if args.device != 'auto':
        config['device'] = args.device
    
    print(f"🎯 WaveSpectra2Text 训练脚本")
    print(f"📊 数据集规模: {args.scale}")
    print(f"🔧 配置文件: {args.config or '默认配置'}")
    print(f"💻 计算设备: {config['device']}")
    print("=" * 60)
    
    # 设置设备
    import torch
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    print(f"使用设备: {device}")
    
    # 创建必要目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['tensorboard_log_dir'], exist_ok=True)
    
    try:
        # 创建数据集
        dataset = AudioDataset(
            labels_file=config['labels_file'],
            audio_dir=config['audio_dir'],
            mode='realtime'
        )
        
        # 分割数据集
        if config['validation_split'] > 0:
            from sklearn.model_selection import train_test_split
            
            # 简化数据集分割逻辑
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=config['validation_split'],
                random_state=config['random_seed']
            )
            
            from torch.utils.data import Subset
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        else:
            train_dataset = dataset
            val_dataset = dataset
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        from wavespectra2text.data.dataset import FlexibleDataLoader
        
        train_loader = FlexibleDataLoader.create_dataloader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        val_loader = FlexibleDataLoader.create_dataloader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        
        # 创建模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=config['hidden_dim'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            dropout=config['dropout']
        ).to(device)
        
        print(f"模型大小: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 创建训练器
        # 将scale参数映射到训练器类型
        scale_to_trainer = {
            'small': 'simple',
            'medium': 'improved', 
            'large': 'large',
            'xlarge': 'large'  # 超大数据集也使用large训练器
        }
        trainer_type = scale_to_trainer.get(args.scale, 'simple')
        print(f"🎯 使用训练器: {trainer_type} (对应数据集规模: {args.scale})")
        trainer = create_trainer(trainer_type, model, train_loader, val_loader, device, config)
        
        # 设置回调
        callbacks = CallbackList([
            EarlyStoppingCallback(patience=config['max_patience']),
            CheckpointCallback(
                filepath=os.path.join(config['checkpoint_dir'], 'checkpoint_epoch_{epoch}.pth'),
                save_every=config['save_every']
            ),
            LoggingCallback(log_dir=config['log_dir']),
            TensorBoardCallback(log_dir=config['tensorboard_log_dir'])
        ])
        
        callbacks.set_trainer(trainer)
        trainer.callbacks = callbacks
        
        # 开始训练
        trainer.train(config['num_epochs'], args.resume)
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
