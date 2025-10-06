#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练状态和推理问题
"""

import os
import glob
from pathlib import Path

def check_training_status():
    """检查训练状态"""
    print("🔍 检查训练状态")
    print("=" * 40)
    
    # 检查checkpoints目录
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        print(f"✅ checkpoints目录存在")
        
        # 列出所有模型文件
        model_files = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
        if model_files:
            print(f"✅ 找到 {len(model_files)} 个模型文件:")
            for model_file in model_files:
                size = os.path.getsize(model_file) / (1024*1024)  # MB
                print(f"  - {model_file} ({size:.1f} MB)")
            return model_files[0]  # 返回第一个模型文件
        else:
            print("❌ checkpoints目录存在但没有模型文件")
    else:
        print("❌ checkpoints目录不存在")
    
    # 检查其他可能的模型文件位置
    possible_locations = [
        "*.pth",
        "models/*.pth", 
        "saved_models/*.pth",
        "outputs/*.pth"
    ]
    
    print("\n🔍 搜索其他位置的模型文件...")
    for pattern in possible_locations:
        files = glob.glob(pattern)
        if files:
            print(f"✅ 在 {pattern} 找到模型文件:")
            for file in files:
                print(f"  - {file}")
            return files[0]
    
    print("❌ 没有找到任何模型文件")
    return None

def check_training_logs():
    """检查训练日志"""
    print("\n🔍 检查训练日志")
    print("-" * 30)
    
    # 检查TensorBoard日志
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        print(f"✅ TensorBoard日志目录存在: {runs_dir}")
        subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if subdirs:
            print(f"  实验目录: {subdirs}")
            latest_dir = max(subdirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
            print(f"  最新实验: {latest_dir}")
        else:
            print("  没有实验日志")
    else:
        print("❌ 没有TensorBoard日志")

def analyze_inference_issue():
    """分析推理问题"""
    print("\n🔍 分析推理空结果问题")
    print("-" * 30)
    
    print("根据您的输出，推理系统工作正常但结果为空，可能原因:")
    print("1. 🎯 模型训练不充分 (28 epoch可能不够)")
    print("2. 🎯 模型还在学习阶段，没有学会正确的映射")
    print("3. 🎯 解码时立即遇到EOS token")
    print("4. 🎯 模型输出的都是特殊token (PAD, UNK等)")

def provide_solutions():
    """提供解决方案"""
    print("\n💡 解决方案")
    print("=" * 40)
    
    solutions = [
        {
            "title": "1. 检查训练是否还在进行",
            "commands": [
                "ps aux | grep python",  # 检查是否有训练进程
                "top | grep python"      # 检查CPU使用情况
            ]
        },
        {
            "title": "2. 继续训练更多epoch",
            "commands": [
                "python train_small.py --config config_small_data.json"
            ]
        },
        {
            "title": "3. 使用更小的模型和更多epoch",
            "description": "修改config_small_data.json，设置更多epoch"
        },
        {
            "title": "4. 检查训练损失",
            "commands": [
                "tensorboard --logdir=runs",
                "# 然后在浏览器打开 http://localhost:6006"
            ]
        },
        {
            "title": "5. 创建测试模型验证推理逻辑",
            "description": "使用我们之前的虚拟模型测试"
        }
    ]
    
    for i, solution in enumerate(solutions):
        print(f"\n{solution['title']}:")
        if 'commands' in solution:
            for cmd in solution['commands']:
                print(f"  {cmd}")
        if 'description' in solution:
            print(f"  {solution['description']}")

def create_quick_test_model():
    """创建一个快速测试模型"""
    print("\n🚀 创建快速测试模型")
    print("-" * 30)
    
    try:
        import torch
        from model import create_model
        from vocab import vocab
        
        print("创建一个简单的测试模型...")
        
        # 创建超小模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=32,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0
        )
        
        # 手动设置一些权重，让模型输出有意义的结果
        with torch.no_grad():
            # 设置输出层偏置，让某些token更容易被选中
            if hasattr(model.decoder, 'output_projection'):
                # 给中文数字token更高的偏置
                bias = model.decoder.output_projection.bias
                for word, idx in vocab.word_to_idx.items():
                    if word in ['一', '二', '三', '四', '五']:
                        bias[idx] = 2.0  # 增加这些token的概率
        
        # 保存测试模型
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'best_val_loss': 1.0,
            'config': {
                'hidden_dim': 32,
                'encoder_layers': 1,
                'decoder_layers': 1,
                'dropout': 0.0
            }
        }
        
        test_model_path = 'checkpoints/test_model.pth'
        torch.save(checkpoint, test_model_path)
        
        print(f"✅ 测试模型已保存: {test_model_path}")
        print("现在可以测试推理:")
        print(f"  python inference.py --model {test_model_path} --audio data/audio/Chinese_Number_01.wav")
        
        return test_model_path
        
    except Exception as e:
        print(f"❌ 创建测试模型失败: {e}")
        return None

def main():
    """主函数"""
    print("🎯 训练状态和推理问题检查")
    print("=" * 50)
    
    # 检查训练状态
    model_file = check_training_status()
    
    # 检查训练日志
    check_training_logs()
    
    # 分析推理问题
    analyze_inference_issue()
    
    # 提供解决方案
    provide_solutions()
    
    # 如果没有模型文件，创建测试模型
    if not model_file:
        print("\n" + "=" * 50)
        test_model = create_quick_test_model()
        if test_model:
            print(f"\n🎯 立即测试推理:")
            print(f"python inference.py --model {test_model} --audio data/audio/Chinese_Number_01.wav")
    
    print("\n🎉 总结:")
    if model_file:
        print("✅ 找到了训练的模型，推理空结果可能是训练不充分")
        print("💡 建议: 继续训练或检查训练损失")
    else:
        print("❌ 没有找到训练的模型文件")
        print("💡 建议: 检查训练是否在进行，或重新开始训练")

if __name__ == "__main__":
    main()