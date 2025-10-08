#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本使用示例
展示WaveSpectra2Text的基本功能
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from wavespectra2text import (
    create_model, vocab, DualInputSpeechRecognizer,
    AudioPreprocessor, PreprocessorFactory, AudioDataset
)


def basic_usage_example():
    """基本使用示例"""
    print("🎯 WaveSpectra2Text 基本使用示例")
    print("=" * 50)
    
    # 1. 创建模型
    print("1. 创建模型...")
    model = create_model(
        vocab_size=vocab.vocab_size,
        hidden_dim=128,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1
    )
    print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 2. 创建音频预处理器
    print("\n2. 创建音频预处理器...")
    preprocessor = PreprocessorFactory.create('spectrogram')
    print(f"✅ 预处理器创建成功，特征形状: {preprocessor.get_feature_shape()}")
    
    # 3. 创建数据集
    print("\n3. 创建数据集...")
    try:
        dataset = AudioDataset(
            labels_file='data/labels.csv',
            audio_dir='data/audio',
            mode='realtime'
        )
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
    except Exception as e:
        print(f"⚠️  数据集创建失败: {e}")
        print("   请确保data/labels.csv和data/audio目录存在")
    
    # 4. 创建识别器（需要模型文件）
    print("\n4. 创建识别器...")
    model_path = 'experiments/checkpoints/best_model.pth'
    if os.path.exists(model_path):
        try:
            recognizer = DualInputSpeechRecognizer(model_path)
            print("✅ 识别器创建成功")
            
            # 测试识别
            audio_file = 'data/audio/Chinese_Number_01.wav'
            if os.path.exists(audio_file):
                result = recognizer.recognize_from_audio(audio_file)
                if result['success']:
                    print(f"🎯 识别结果: '{result['text']}'")
                else:
                    print(f"❌ 识别失败: {result['error']}")
            else:
                print("⚠️  音频文件不存在，跳过识别测试")
                
        except Exception as e:
            print(f"❌ 识别器创建失败: {e}")
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   请先训练模型")
    
    print("\n✅ 基本使用示例完成!")


def advanced_usage_example():
    """高级使用示例"""
    print("\n🚀 WaveSpectra2Text 高级使用示例")
    print("=" * 50)
    
    # 1. 自定义预处理器
    print("1. 自定义预处理器...")
    custom_preprocessor = PreprocessorFactory.create(
        'spectrogram',
        sample_rate=48000,
        n_fft=2048,  # 更大的FFT窗口
        hop_length=1024,
        max_length=300  # 更长的序列
    )
    print(f"✅ 自定义预处理器创建成功")
    print(f"   特征形状: {custom_preprocessor.get_feature_shape()}")
    
    # 2. 批量处理示例
    print("\n2. 批量处理示例...")
    audio_files = [
        'data/audio/Chinese_Number_01.wav',
        'data/audio/Chinese_Number_02.wav',
        'data/audio/Chinese_Number_03.wav'
    ]
    
    results = []
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            try:
                # 预处理音频
                features = custom_preprocessor.process(audio_file)
                print(f"✅ 处理完成: {audio_file}, 特征形状: {features.shape}")
                results.append(features)
            except Exception as e:
                print(f"❌ 处理失败: {audio_file}, 错误: {e}")
    
    print(f"📊 批量处理完成，成功处理 {len(results)} 个文件")
    
    # 3. 配置管理示例
    print("\n3. 配置管理示例...")
    from wavespectra2text.training.config import ConfigManager, get_default_config
    
    # 使用默认配置
    config = get_default_config('medium')
    print(f"✅ 默认配置加载成功")
    print(f"   批大小: {config['batch_size']}")
    print(f"   学习率: {config['learning_rate']}")
    print(f"   隐藏层维度: {config['hidden_dim']}")
    
    # 更新配置
    config_manager = ConfigManager()
    config_manager.update_config({'batch_size': 8, 'learning_rate': 0.002})
    print(f"✅ 配置更新成功")
    print(f"   新批大小: {config_manager['batch_size']}")
    print(f"   新学习率: {config_manager['learning_rate']}")
    
    print("\n✅ 高级使用示例完成!")


if __name__ == "__main__":
    basic_usage_example()
    advanced_usage_example()
    
    print("\n" + "=" * 60)
    print("📚 更多示例:")
    print("  - examples/custom_training.py: 自定义训练示例")
    print("  - examples/batch_inference.py: 批量推理示例")
    print("  - scripts/train.py: 训练脚本")
    print("  - scripts/inference.py: 推理脚本")
    print("\n💡 提示:")
    print("  - 确保已安装所有依赖: pip install -r requirements.txt")
    print("  - 准备数据: python setup_data.py")
    print("  - 开始训练: python scripts/train.py --scale small")
