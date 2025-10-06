#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理问题诊断脚本
分析为什么识别结果为空
"""

import os
import torch
import numpy as np
from inference import SpeechRecognizer
from vocab import vocab

def diagnose_empty_results(model_path, test_audio):
    """诊断空结果问题"""
    print("🔍 诊断推理空结果问题")
    print("=" * 50)
    
    try:
        # 创建识别器
        recognizer = SpeechRecognizer(model_path)
        print(f"✅ 模型加载成功")
        
        # 测试音频预处理
        print(f"\n1️⃣ 测试音频预处理...")
        spectrogram = recognizer._extract_spectrogram(test_audio)
        print(f"  频谱形状: {spectrogram.shape}")
        print(f"  频谱范围: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
        
        # 测试编码器
        print(f"\n2️⃣ 测试编码器...")
        with torch.no_grad():
            encoder_output = recognizer.model.encode(spectrogram)
            print(f"  编码器输出形状: {encoder_output.shape}")
            print(f"  编码器输出范围: [{encoder_output.min():.3f}, {encoder_output.max():.3f}]")
        
        # 详细测试贪婪解码过程
        print(f"\n3️⃣ 详细贪婪解码过程...")
        decoded_seq = diagnose_greedy_decode(recognizer.model, encoder_output)
        print(f"  最终解码序列: {decoded_seq.tolist()}")
        
        # 测试词汇表解码
        print(f"\n4️⃣ 测试词汇表解码...")
        text = vocab.decode(decoded_seq.tolist())
        print(f"  解码文本: '{text}'")
        print(f"  文本长度: {len(text)}")
        
        # 检查词汇表
        print(f"\n5️⃣ 检查词汇表...")
        print(f"  词汇表大小: {vocab.vocab_size}")
        print(f"  特殊token: PAD={vocab.get_padding_idx()}, SOS={vocab.get_sos_idx()}, EOS={vocab.get_eos_idx()}")
        print(f"  词汇映射: {vocab.word_to_idx}")
        
        # 测试束搜索
        print(f"\n6️⃣ 测试束搜索解码...")
        beam_seq, beam_score = diagnose_beam_search(recognizer.model, encoder_output)
        beam_text = vocab.decode(beam_seq.tolist())
        print(f"  束搜索序列: {beam_seq.tolist()}")
        print(f"  束搜索文本: '{beam_text}'")
        print(f"  束搜索得分: {beam_score:.3f}")
        
        return text, beam_text
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def diagnose_greedy_decode(model, encoder_output, max_length=10):
    """诊断贪婪解码过程"""
    device = encoder_output.device
    
    # 初始化解码序列
    decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(device)
    print(f"  初始序列: {decoded_seq.tolist()} (SOS token)")
    
    for step in range(max_length):
        with torch.no_grad():
            # 获取下一个token的概率
            output = model.decode_step(decoded_seq, encoder_output)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            print(f"  步骤 {step+1}:")
            print(f"    输出logits形状: {output.shape}")
            print(f"    下一个token: {next_token.item()} ('{vocab.idx_to_word.get(next_token.item(), 'UNK')}')")
            print(f"    概率: {probs[0, next_token.item()].item():.4f}")
            print(f"    前5个最高概率: {torch.topk(probs, 5)[1].tolist()}")
            
            # 添加到序列中
            decoded_seq = torch.cat([decoded_seq, next_token], dim=1)
            print(f"    当前序列: {decoded_seq.tolist()}")
            
            # 如果生成了结束符号，停止解码
            if next_token.item() == vocab.get_eos_idx():
                print(f"    遇到EOS token，停止解码")
                break
    
    return decoded_seq.squeeze(0)

def diagnose_beam_search(model, encoder_output, beam_size=3, max_length=10):
    """诊断束搜索解码过程"""
    device = encoder_output.device
    
    # 初始化束
    beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]
    print(f"  初始束: {beams[0][0].tolist()}")
    
    for step in range(max_length):
        print(f"  束搜索步骤 {step+1}:")
        new_beams = []
        
        for i, (seq, score) in enumerate(beams):
            if seq[0, -1].item() == vocab.get_eos_idx():
                new_beams.append((seq, score))
                print(f"    束 {i}: 已结束 {seq.tolist()}")
                continue
            
            # 获取下一个token的概率
            with torch.no_grad():
                output = model.decode_step(seq, encoder_output)
                probs = torch.softmax(output[:, -1, :], dim=-1)
            
            # 获取top-k候选
            top_probs, top_indices = torch.topk(probs, beam_size)
            print(f"    束 {i}: 当前序列 {seq.tolist()}")
            print(f"    束 {i}: 候选tokens {top_indices.tolist()[0]} 概率 {top_probs.tolist()[0]}")
            
            for j in range(beam_size):
                new_seq = torch.cat([seq, top_indices[:, j:j + 1]], dim=1)
                new_score = score + torch.log(top_probs[:, j]).item()
                new_beams.append((new_seq, new_score))
        
        # 保留最好的beam_size个候选
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        print(f"    保留的束: {[beam[0].tolist() for beam in beams]}")
        
        # 检查是否所有束都结束了
        if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
            print(f"    所有束都结束")
            break
    
    # 返回最佳序列
    best_seq, best_score = beams[0]
    return best_seq.squeeze(0), best_score

def check_model_training_status(model_path):
    """检查模型训练状态"""
    print("\n🔍 检查模型训练状态")
    print("-" * 30)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"训练epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'Unknown')}")
        print(f"配置: {checkpoint.get('config', 'Unknown')}")
        
        # 检查模型权重
        state_dict = checkpoint['model_state_dict']
        
        # 检查一些关键层的权重
        for name, param in state_dict.items():
            if 'embedding' in name or 'output_projection' in name:
                print(f"  {name}: 形状={param.shape}, 范围=[{param.min():.3f}, {param.max():.3f}]")
                
        return True
        
    except Exception as e:
        print(f"❌ 检查模型失败: {e}")
        return False

def suggest_solutions():
    """提供解决方案建议"""
    print("\n💡 解决方案建议")
    print("=" * 50)
    
    solutions = [
        "1. 继续训练更多epoch (当前28可能不够)",
        "2. 检查训练损失是否还在下降",
        "3. 降低学习率，延长训练时间",
        "4. 检查训练数据是否正确加载",
        "5. 尝试使用更小的模型避免过拟合",
        "6. 增加数据增强提高训练效果"
    ]
    
    for solution in solutions:
        print(f"  {solution}")

def main():
    """主函数"""
    model_path = "checkpoints/test_model.pth"
    test_audio = "data/audio/Chinese_Number_01.wav"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(test_audio):
        print(f"❌ 测试音频不存在: {test_audio}")
        return
    
    # 检查模型状态
    check_model_training_status(model_path)
    
    # 诊断推理问题
    text, beam_text = diagnose_empty_results(model_path, test_audio)
    
    # 提供解决方案
    suggest_solutions()
    
    print(f"\n🎯 诊断总结:")
    print(f"  贪婪解码结果: '{text}'")
    print(f"  束搜索结果: '{beam_text}'")
    
    if not text or text.strip() == '':
        print("❌ 确认存在空结果问题")
        print("💡 建议: 继续训练模型或调整训练参数")
    else:
        print("✅ 解码正常，可能是显示问题")

if __name__ == "__main__":
    main()