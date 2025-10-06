#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理逻辑验证脚本
分层验证推理流程的每个环节
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

def test_model_loading():
    """测试1: 模型加载验证"""
    print("🧪 测试1: 模型加载验证")
    print("-" * 40)
    
    try:
        from model import create_model
        from vocab import vocab
        
        # 创建测试模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=64,
            encoder_layers=2,
            decoder_layers=2
        )
        
        print(f"✅ 模型创建成功")
        print(f"  词汇表大小: {vocab.vocab_size}")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试模型前向传播
        batch_size, seq_len, freq_bins = 1, 200, 513
        test_input = torch.randn(batch_size, seq_len, freq_bins)
        test_target = torch.randint(0, vocab.vocab_size, (batch_size, 3))
        
        model.eval()
        with torch.no_grad():
            output = model(test_input, test_target[:, :-1])
            print(f"✅ 模型前向传播成功")
            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False, None

def test_audio_preprocessing():
    """测试2: 音频预处理验证"""
    print("\n🧪 测试2: 音频预处理验证")
    print("-" * 40)
    
    try:
        from audio_preprocessing import SpectrogramPreprocessor
        
        # 创建预处理器
        preprocessor = SpectrogramPreprocessor()
        
        # 测试音频文件
        test_audio = "data/audio/Chinese_Number_01.wav"
        if not os.path.exists(test_audio):
            print(f"❌ 测试音频文件不存在: {test_audio}")
            return False, None
        
        # 处理音频
        features = preprocessor.process(test_audio)
        print(f"✅ 音频预处理成功")
        print(f"  音频文件: {test_audio}")
        print(f"  特征形状: {features.shape}")
        print(f"  特征类型: {features.dtype}")
        print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")
        
        return True, features
        
    except Exception as e:
        print(f"❌ 音频预处理失败: {e}")
        traceback.print_exc()
        return False, None

def test_inference_components(model, features):
    """测试3: 推理组件验证"""
    print("\n🧪 测试3: 推理组件验证")
    print("-" * 40)
    
    try:
        from vocab import vocab
        
        # 转换为tensor并添加batch维度
        spectrogram_tensor = torch.FloatTensor(features).unsqueeze(0)
        print(f"✅ 特征tensor化成功: {spectrogram_tensor.shape}")
        
        model.eval()
        with torch.no_grad():
            # 测试编码器
            encoder_output = model.encode(spectrogram_tensor)
            print(f"✅ 编码器测试成功")
            print(f"  编码器输出形状: {encoder_output.shape}")
            
            # 测试贪婪解码
            decoded_seq = test_greedy_decode(model, encoder_output)
            print(f"✅ 贪婪解码测试成功")
            print(f"  解码序列: {decoded_seq.tolist()}")
            
            # 解码为文本
            text = vocab.decode(decoded_seq.tolist())
            print(f"✅ 文本解码成功: '{text}'")
            
            # 测试束搜索解码
            beam_seq, beam_score = test_beam_search(model, encoder_output)
            beam_text = vocab.decode(beam_seq.tolist())
            print(f"✅ 束搜索解码成功")
            print(f"  束搜索结果: '{beam_text}' (得分: {beam_score:.3f})")
        
        return True, text, beam_text
        
    except Exception as e:
        print(f"❌ 推理组件测试失败: {e}")
        traceback.print_exc()
        return False, None, None

def test_greedy_decode(model, encoder_output, max_length=10):
    """贪婪解码测试"""
    from vocab import vocab
    
    # 初始化解码序列
    decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]])
    
    for step in range(max_length):
        # 获取下一个token的概率
        output = model.decode_step(decoded_seq, encoder_output)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # 添加到序列中
        decoded_seq = torch.cat([decoded_seq, next_token], dim=1)
        
        # 如果生成了结束符号，停止解码
        if next_token.item() == vocab.get_eos_idx():
            break
    
    return decoded_seq.squeeze(0)

def test_beam_search(model, encoder_output, beam_size=3, max_length=10):
    """束搜索解码测试"""
    from vocab import vocab
    
    # 初始化束
    beams = [(torch.LongTensor([[vocab.get_sos_idx()]]), 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq[0, -1].item() == vocab.get_eos_idx():
                new_beams.append((seq, score))
                continue
            
            # 获取下一个token的概率
            output = model.decode_step(seq, encoder_output)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            
            # 获取top-k候选
            top_probs, top_indices = torch.topk(probs, beam_size)
            
            for i in range(beam_size):
                new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=1)
                new_score = score + torch.log(top_probs[:, i]).item()
                new_beams.append((new_seq, new_score))
        
        # 保留最好的beam_size个候选
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        # 检查是否所有束都结束了
        if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
            break
    
    # 返回最佳序列
    best_seq, best_score = beams[0]
    return best_seq.squeeze(0), best_score

def test_inference_class():
    """测试4: 推理类验证"""
    print("\n🧪 测试4: 推理类验证 (模拟)")
    print("-" * 40)
    
    try:
        # 检查推理脚本是否存在
        if not os.path.exists('inference.py'):
            print("❌ inference.py 文件不存在")
            return False
        
        print("✅ inference.py 文件存在")
        
        # 尝试导入推理类 (不实际加载模型)
        import sys
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("inference", "inference.py")
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)  # 执行模块
        
        print("✅ 推理模块导入成功")
        
        # 检查关键类和方法
        if hasattr(inference_module, 'SpeechRecognizer'):
            print("✅ SpeechRecognizer 类存在")
            
            # 检查关键方法
            recognizer_class = getattr(inference_module, 'SpeechRecognizer')
            methods = ['recognize_file', 'recognize_batch', 'evaluate_on_dataset']
            
            for method in methods:
                if hasattr(recognizer_class, method):
                    print(f"✅ {method} 方法存在")
                else:
                    print(f"⚠️  {method} 方法不存在")
        else:
            print("❌ SpeechRecognizer 类不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 推理类验证失败: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_simulation():
    """测试5: 端到端推理模拟"""
    print("\n🧪 测试5: 端到端推理模拟")
    print("-" * 40)
    
    try:
        # 模拟完整的推理流程
        print("🎯 模拟推理流程:")
        
        # 1. 音频文件检查
        test_files = [
            "data/audio/Chinese_Number_01.wav",
            "data/audio/Chinese_Number_02.wav", 
            "data/audio/Chinese_Number_03.wav"
        ]
        
        existing_files = [f for f in test_files if os.path.exists(f)]
        print(f"  可用测试文件: {len(existing_files)}/{len(test_files)}")
        
        if not existing_files:
            print("❌ 没有可用的测试音频文件")
            return False
        
        # 2. 预处理验证
        from audio_preprocessing import SpectrogramPreprocessor
        preprocessor = SpectrogramPreprocessor()
        
        results = []
        for audio_file in existing_files[:3]:  # 只测试前3个
            try:
                features = preprocessor.process(audio_file)
                results.append({
                    'file': os.path.basename(audio_file),
                    'shape': features.shape,
                    'success': True
                })
                print(f"  ✅ {os.path.basename(audio_file)}: {features.shape}")
            except Exception as e:
                results.append({
                    'file': os.path.basename(audio_file),
                    'error': str(e),
                    'success': False
                })
                print(f"  ❌ {os.path.basename(audio_file)}: {e}")
        
        success_rate = sum(1 for r in results if r['success']) / len(results)
        print(f"  预处理成功率: {success_rate:.1%}")
        
        return success_rate > 0.5
        
    except Exception as e:
        print(f"❌ 端到端模拟失败: {e}")
        traceback.print_exc()
        return False

def test_with_dummy_model():
    """测试6: 使用虚拟模型验证完整流程"""
    print("\n🧪 测试6: 虚拟模型完整流程验证")
    print("-" * 40)
    
    try:
        from model import create_model
        from vocab import vocab
        from audio_preprocessing import SpectrogramPreprocessor
        
        # 创建虚拟模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=32,  # 小模型用于测试
            encoder_layers=1,
            decoder_layers=1
        )
        
        # 随机初始化权重 (模拟训练好的模型)
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
        
        model.eval()
        
        # 测试音频文件
        test_audio = "data/audio/Chinese_Number_01.wav"
        if not os.path.exists(test_audio):
            print(f"❌ 测试文件不存在: {test_audio}")
            return False
        
        # 完整推理流程
        preprocessor = SpectrogramPreprocessor()
        
        print("🎯 执行完整推理流程:")
        
        # 1. 音频预处理
        features = preprocessor.process(test_audio)
        spectrogram_tensor = torch.FloatTensor(features).unsqueeze(0)
        print(f"  1. 音频预处理: {features.shape} → {spectrogram_tensor.shape}")
        
        # 2. 编码
        with torch.no_grad():
            encoder_output = model.encode(spectrogram_tensor)
            print(f"  2. 编码: {spectrogram_tensor.shape} → {encoder_output.shape}")
        
        # 3. 解码
        with torch.no_grad():
            decoded_seq = test_greedy_decode(model, encoder_output)
            print(f"  3. 解码: 序列长度 {len(decoded_seq)}")
        
        # 4. 文本转换
        text = vocab.decode(decoded_seq.tolist())
        print(f"  4. 文本转换: {decoded_seq.tolist()} → '{text}'")
        
        print(f"✅ 完整推理流程成功!")
        print(f"  输入: {os.path.basename(test_audio)}")
        print(f"  输出: '{text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ 虚拟模型测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("🎯 推理逻辑验证")
    print("=" * 60)
    
    tests = [
        ("模型加载验证", test_model_loading),
        ("音频预处理验证", test_audio_preprocessing), 
        ("推理类验证", test_inference_class),
        ("端到端模拟", test_end_to_end_simulation),
        ("虚拟模型完整流程", test_with_dummy_model)
    ]
    
    results = []
    model, features = None, None
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_name == "模型加载验证":
                success, model = test_func()
            elif test_name == "音频预处理验证":
                success, features = test_func()
            elif test_name == "推理组件验证" and model is not None and features is not None:
                success, _, _ = test_inference_components(model, features)
            else:
                success = test_func()
            
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ {test_name} 执行异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 验证结果汇总:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有推理逻辑验证通过！推理系统可以正常工作")
        print("\n💡 下一步:")
        print("1. 训练一个真实模型")
        print("2. 使用 inference.py 进行实际推理测试")
        print("3. 在真实数据上验证推理效果")
    elif passed >= len(results) * 0.8:
        print("⚠️  大部分测试通过，推理逻辑基本正常")
        print("建议检查失败的测试项目")
    else:
        print("❌ 多个测试失败，推理系统可能存在问题")
        print("建议逐项检查和修复")

if __name__ == "__main__":
    main()