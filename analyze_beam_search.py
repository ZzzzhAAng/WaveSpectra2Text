#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
束搜索问题分析脚本
诊断为什么束搜索总是返回空结果
"""

import os
import torch
import numpy as np
from inference_improved import ImprovedSpeechRecognizer
from vocab import vocab

def analyze_model_bias(model_path):
    """分析模型对不同token的偏置"""
    print("🔍 分析模型token偏置")
    print("-" * 40)
    
    try:
        recognizer = ImprovedSpeechRecognizer(model_path)
        
        # 获取一个测试样本
        test_audio = "data/audio/Chinese_Number_01.wav"
        spectrogram = recognizer._extract_spectrogram(test_audio).to(recognizer.device)
        
        with torch.no_grad():
            encoder_output = recognizer.model.encode(spectrogram)
            
            # 分析第一步的输出分布
            initial_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(recognizer.device)
            output = recognizer.model.decode_step(initial_seq, encoder_output)
            logits = output[:, -1, :]  # 第一步的logits
            probs = torch.softmax(logits, dim=-1)
            
            print("第一步解码的token概率分布:")
            for idx, prob in enumerate(probs[0]):
                token_name = vocab.idx_to_word.get(idx, f'IDX_{idx}')
                print(f"  {token_name}: {prob.item():.4f}")
            
            # 检查EOS token的概率
            eos_prob = probs[0, vocab.get_eos_idx()].item()
            print(f"\n⚠️  EOS token概率: {eos_prob:.4f}")
            
            if eos_prob > 0.3:
                print("❌ EOS token概率过高，这解释了为什么束搜索总是选择结束")
            elif eos_prob > 0.1:
                print("⚠️  EOS token概率偏高，可能影响束搜索")
            else:
                print("✅ EOS token概率正常")
            
            return eos_prob
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def test_improved_beam_search(model_path, test_audio):
    """测试改进的束搜索"""
    print("\n🔧 测试改进的束搜索")
    print("-" * 40)
    
    try:
        recognizer = ImprovedSpeechRecognizer(model_path)
        spectrogram = recognizer._extract_spectrogram(test_audio).to(recognizer.device)
        
        with torch.no_grad():
            encoder_output = recognizer.model.encode(spectrogram)
            
            # 测试不同的束搜索配置
            configs = [
                {"beam_size": 1, "max_length": 5},
                {"beam_size": 3, "max_length": 5},
                {"beam_size": 5, "max_length": 5},
                {"beam_size": 3, "max_length": 10},
            ]
            
            for i, config in enumerate(configs):
                print(f"\n配置 {i+1}: 束大小={config['beam_size']}, 最大长度={config['max_length']}")
                
                # 使用改进的束搜索
                beam_seq, beam_score = improved_beam_search(
                    recognizer.model, encoder_output, 
                    beam_size=config['beam_size'], 
                    max_length=config['max_length']
                )
                
                beam_text = vocab.decode(beam_seq.tolist())
                print(f"  结果: '{beam_text}'")
                print(f"  序列: {beam_seq.tolist()}")
                print(f"  得分: {beam_score:.3f}")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def improved_beam_search(model, encoder_output, beam_size=3, max_length=10, eos_penalty=-1.0):
    """改进的束搜索 - 添加EOS惩罚"""
    device = encoder_output.device
    beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]
    
    for step in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq[0, -1].item() == vocab.get_eos_idx():
                new_beams.append((seq, score))
                continue
            
            with torch.no_grad():
                output = model.decode_step(seq, encoder_output)
                logits = output[:, -1, :]
                
                # 对EOS token添加惩罚 (如果序列太短)
                if seq.size(1) < 3:  # 如果序列长度小于3，惩罚EOS
                    logits[0, vocab.get_eos_idx()] += eos_penalty
                
                probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, beam_size)
            
            for i in range(beam_size):
                new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=1)
                new_score = score + torch.log(top_probs[:, i]).item()
                new_beams.append((new_seq, new_score))
        
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
            break
    
    best_seq, best_score = beams[0]
    return best_seq.squeeze(0), best_score

def suggest_beam_search_fixes():
    """建议束搜索修复方案"""
    print("\n💡 束搜索修复建议")
    print("=" * 50)
    
    suggestions = [
        {
            "问题": "EOS token概率过高",
            "解决方案": [
                "1. 在束搜索中对短序列的EOS添加惩罚",
                "2. 调整训练时的标签平滑",
                "3. 增加最小序列长度限制"
            ]
        },
        {
            "问题": "模型训练不充分",
            "解决方案": [
                "1. 继续训练更多epoch",
                "2. 降低学习率精细调优",
                "3. 检查训练损失曲线"
            ]
        },
        {
            "问题": "束搜索参数不当",
            "解决方案": [
                "1. 尝试不同的束大小 (1, 3, 5)",
                "2. 调整最大序列长度",
                "3. 添加长度惩罚机制"
            ]
        }
    ]
    
    for suggestion in suggestions:
        print(f"\n🎯 {suggestion['问题']}:")
        for solution in suggestion['解决方案']:
            print(f"  {solution}")

def create_fixed_beam_search_recognizer():
    """创建修复版束搜索识别器"""
    print("\n🛠️ 创建修复版束搜索识别器")
    print("-" * 40)
    
    code = '''
class FixedBeamSearchRecognizer(ImprovedSpeechRecognizer):
    """修复版束搜索识别器"""
    
    def _beam_search_fixed(self, encoder_output, beam_size=3, max_length=10):
        """修复版束搜索 - 添加EOS惩罚和长度奖励"""
        device = encoder_output.device
        beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == vocab.get_eos_idx():
                    # 对过短的序列添加惩罚
                    if seq.size(1) < 3:
                        score -= 2.0  # 惩罚过短序列
                    new_beams.append((seq, score))
                    continue
                
                with torch.no_grad():
                    output = self.model.decode_step(seq, encoder_output)
                    logits = output[:, -1, :]
                    
                    # 对过早的EOS添加惩罚
                    if seq.size(1) < 3:
                        logits[0, vocab.get_eos_idx()] -= 1.0
                    
                    probs = torch.softmax(logits, dim=-1)
                
                top_probs, top_indices = torch.topk(probs, beam_size)
                
                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=1)
                    new_score = score + torch.log(top_probs[:, i]).item()
                    
                    # 长度奖励
                    if top_indices[:, i].item() != vocab.get_eos_idx():
                        new_score += 0.1  # 鼓励生成更长序列
                    
                    new_beams.append((new_seq, new_score))
            
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            
            if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
                break
        
        best_seq, best_score = beams[0]
        return best_seq.squeeze(0), best_score
    '''
    
    print("修复版束搜索代码已准备好")
    print("主要改进:")
    print("1. ✅ 对过短序列的EOS添加惩罚")
    print("2. ✅ 对非EOS token添加长度奖励")
    print("3. ✅ 动态调整EOS概率")
    
    return code

def main():
    """主函数"""
    model_path = "checkpoints/test_model.pth"
    test_audio = "data/audio/Chinese_Number_01.wav"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print("🎯 束搜索问题分析")
    print("=" * 60)
    
    # 分析模型偏置
    eos_prob = analyze_model_bias(model_path)
    
    # 测试改进的束搜索
    test_improved_beam_search(model_path, test_audio)
    
    # 提供修复建议
    suggest_beam_search_fixes()
    
    # 创建修复版识别器
    create_fixed_beam_search_recognizer()
    
    print(f"\n🎯 总结:")
    if eos_prob and eos_prob > 0.3:
        print("❌ 束搜索问题主要是EOS token概率过高")
        print("💡 建议: 使用贪婪解码或修复束搜索算法")
    elif eos_prob and eos_prob > 0.1:
        print("⚠️  束搜索可能受EOS token影响")
        print("💡 建议: 尝试修复版束搜索或继续训练")
    else:
        print("✅ 模型概率分布相对正常")
        print("💡 建议: 检查束搜索参数或训练更多epoch")

if __name__ == "__main__":
    main()