#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练脚本是否能正常运行
"""

import torch
import json
from data_utils import get_dataloader
from model import create_model
from vocab import vocab

def test_training_loop():
    """测试训练循环"""
    print("🧪 测试训练循环...")
    
    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 创建数据加载器
    dataloader = get_dataloader(
        audio_dir=config['audio_dir'],
        labels_file=config['labels_file'],
        batch_size=1,  # 小批大小用于测试
        shuffle=True
    )
    
    # 创建小模型用于测试
    model = create_model(
        vocab_size=vocab.vocab_size,
        hidden_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.1
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.get_padding_idx())
    
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试几个批次
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # 只测试3个批次
            break
            
        # 兼容新旧接口
        if 'features' in batch:
            spectrograms = batch['features']
        else:
            spectrograms = batch['spectrograms']
        labels = batch['labels']
        
        print(f"批次 {batch_idx + 1}:")
        print(f"  特征形状: {spectrograms.shape}")
        print(f"  标签形状: {labels.shape}")
        
        # 准备输入和目标
        tgt_input = labels[:, :-1]
        tgt_output = labels[:, 1:]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(spectrograms, tgt_input)
        
        # 计算损失
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  损失: {loss.item():.4f}")
    
    avg_loss = total_loss / 3
    print(f"\n平均损失: {avg_loss:.4f}")
    print("✅ 训练循环测试成功!")
    
    return True

if __name__ == "__main__":
    print("🎯 训练功能测试")
    print("=" * 50)
    
    try:
        test_training_loop()
        print("\n🎉 所有测试通过！现在可以开始正式训练了")
        print("\n💡 建议的训练命令:")
        print("python train_small.py --config config.json")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()