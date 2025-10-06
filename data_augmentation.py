#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的音频数据增强
通过添加噪声、时间拉伸等方式扩充数据集
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import soundfile as sf

def add_noise(audio, noise_factor=0.005):
    """添加高斯噪声"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, rate=1.1):
    """时间拉伸"""
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    """音调变化"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def augment_audio_file(input_path, output_dir, base_name, sr=48000):
    """对单个音频文件进行增强"""
    # 加载音频
    audio, _ = librosa.load(input_path, sr=sr)
    
    augmented_files = []
    
    # 原始文件
    original_path = os.path.join(output_dir, f"{base_name}_original.wav")
    sf.write(original_path, audio, sr)
    augmented_files.append((f"{base_name}_original.wav", "original"))
    
    # 添加噪声 (3个变体)
    for i, noise_level in enumerate([0.003, 0.005, 0.008]):
        noisy_audio = add_noise(audio, noise_level)
        noisy_path = os.path.join(output_dir, f"{base_name}_noise_{i+1}.wav")
        sf.write(noisy_path, noisy_audio, sr)
        augmented_files.append((f"{base_name}_noise_{i+1}.wav", f"noise_{noise_level}"))
    
    # 时间拉伸 (2个变体)
    for i, rate in enumerate([0.9, 1.1]):
        stretched_audio = time_stretch(audio, rate)
        stretched_path = os.path.join(output_dir, f"{base_name}_stretch_{i+1}.wav")
        sf.write(stretched_path, stretched_audio, sr)
        augmented_files.append((f"{base_name}_stretch_{i+1}.wav", f"stretch_{rate}"))
    
    # 音调变化 (2个变体)
    for i, n_steps in enumerate([-1, 1]):
        pitched_audio = pitch_shift(audio, sr, n_steps)
        pitched_path = os.path.join(output_dir, f"{base_name}_pitch_{i+1}.wav")
        sf.write(pitched_path, pitched_audio, sr)
        augmented_files.append((f"{base_name}_pitch_{i+1}.wav", f"pitch_{n_steps}"))
    
    return augmented_files

def create_augmented_dataset(input_dir, labels_file, output_dir):
    """创建增强数据集"""
    print("🎵 开始数据增强...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始标签
    df = pd.read_csv(labels_file)
    
    augmented_data = []
    
    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        
        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print(f"⚠️ 文件不存在: {filename}")
            continue
        
        print(f"🎯 处理: {filename} (标签: {label})")
        
        base_name = Path(filename).stem
        
        try:
            # 生成增强版本
            augmented_files = augment_audio_file(input_path, output_dir, base_name)
            
            # 添加到数据列表
            for aug_filename, aug_type in augmented_files:
                augmented_data.append({
                    'filename': aug_filename,
                    'label': label,
                    'original_file': filename,
                    'augmentation_type': aug_type
                })
            
            print(f"  ✅ 生成 {len(augmented_files)} 个增强版本")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    # 保存增强后的标签文件
    aug_df = pd.DataFrame(augmented_data)
    aug_labels_file = os.path.join(output_dir, 'labels_augmented.csv')
    aug_df.to_csv(aug_labels_file, index=False, encoding='utf-8')
    
    print(f"\n📊 数据增强完成:")
    print(f"  原始样本: {len(df)}")
    print(f"  增强样本: {len(aug_df)}")
    print(f"  增强倍数: {len(aug_df) / len(df):.1f}x")
    print(f"  输出目录: {output_dir}")
    print(f"  标签文件: {aug_labels_file}")
    
    return aug_df

if __name__ == "__main__":
    # 创建增强数据集
    augmented_df = create_augmented_dataset(
        input_dir='data/audio',
        labels_file='data/labels.csv', 
        output_dir='data/audio_augmented'
    )
    
    print("\n💡 使用增强数据集训练:")
    print("python batch_preprocess.py --audio_dir data/audio_augmented --labels_file data/audio_augmented/labels_augmented.csv --output_dir data/features_augmented")
    print("python train_small.py --config config_augmented.json")