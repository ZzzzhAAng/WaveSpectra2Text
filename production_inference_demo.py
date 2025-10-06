#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境推理演示
展示成熟模型如何处理任意新音频文件的完整流程
"""

import os
import torch
import numpy as np
import librosa
import argparse
from pathlib import Path
import time

from model import create_model
from vocab import vocab
from audio_preprocessing import SpectrogramPreprocessor
import warnings

warnings.filterwarnings('ignore')


class ProductionSpeechRecognizer:
    """生产环境语音识别器 - 处理任意新音频"""
    
    def __init__(self, model_path, device='cpu'):
        """
        初始化生产环境识别器
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.device = torch.device(device)
        print(f"🚀 初始化生产环境语音识别器")
        print(f"设备: {self.device}")
        
        # 1. 加载训练好的模型
        self.model = self._load_trained_model(model_path)
        
        # 2. 初始化音频预处理器 (与训练时完全一致)
        self.preprocessor = SpectrogramPreprocessor(
            sample_rate=48000,
            n_fft=1024,
            hop_length=512,
            max_length=200
        )
        
        print(f"✅ 初始化完成，准备处理新音频文件")
    
    def _load_trained_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"📂 加载模型: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        print(f"📊 模型信息:")
        print(f"  训练epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  验证准确率: {checkpoint.get('best_val_acc', 'Unknown')}")
        print(f"  模型配置: {config}")
        
        # 创建模型架构
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=config.get('hidden_dim', 256),
            encoder_layers=config.get('encoder_layers', 4),
            decoder_layers=config.get('decoder_layers', 4),
            dropout=config.get('dropout', 0.1)
        )
        
        # 加载训练好的权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()  # 设置为评估模式
        
        print(f"✅ 模型加载成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        return model
    
    def process_new_audio(self, audio_path, show_details=True):
        """
        处理全新的音频文件 - 完整流程演示
        
        Args:
            audio_path: 新音频文件路径
            show_details: 是否显示详细处理过程
        
        Returns:
            识别结果字典
        """
        if show_details:
            print(f"\n🎵 处理新音频文件: {audio_path}")
            print("=" * 60)
        
        try:
            start_time = time.time()
            
            # 步骤1: 验证音频文件
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            file_size = os.path.getsize(audio_path) / 1024  # KB
            if show_details:
                print(f"📁 文件信息:")
                print(f"  路径: {audio_path}")
                print(f"  大小: {file_size:.1f} KB")
            
            # 步骤2: 音频预处理 - 提取频谱特征
            if show_details:
                print(f"\n🔧 步骤1: 音频预处理")
            
            preprocess_start = time.time()
            
            # 使用与训练时完全相同的预处理流程
            spectrogram_features = self.preprocessor.process(audio_path)
            
            preprocess_time = time.time() - preprocess_start
            
            if show_details:
                print(f"  ✅ 频谱提取完成")
                print(f"  📊 频谱形状: {spectrogram_features.shape}")
                print(f"  📊 数据类型: {spectrogram_features.dtype}")
                print(f"  📊 数值范围: [{spectrogram_features.min():.3f}, {spectrogram_features.max():.3f}]")
                print(f"  ⏱️  预处理耗时: {preprocess_time:.3f}秒")
            
            # 步骤3: 模型推理
            if show_details:
                print(f"\n🧠 步骤2: 模型推理")
            
            inference_start = time.time()
            
            # 转换为tensor并添加batch维度
            spectrogram_tensor = torch.FloatTensor(spectrogram_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 编码阶段: 频谱 -> 隐藏表示
                encoder_output = self.model.encode(spectrogram_tensor)
                
                if show_details:
                    print(f"  🔍 编码器输出形状: {encoder_output.shape}")
                
                # 解码阶段: 隐藏表示 -> 文本序列
                # 使用贪婪解码
                decoded_sequence = self._greedy_decode(encoder_output)
                
                # 也可以使用束搜索 (更高质量但更慢)
                beam_sequence, beam_score = self._beam_search_decode(encoder_output)
                
                if show_details:
                    print(f"  🔍 贪婪解码序列: {decoded_sequence.tolist()}")
                    print(f"  🔍 束搜索序列: {beam_sequence.tolist()}")
            
            inference_time = time.time() - inference_start
            
            # 步骤4: 序列解码为文本
            if show_details:
                print(f"\n📝 步骤3: 文本解码")
            
            # 将token序列转换为文本
            greedy_text = vocab.decode(decoded_sequence.tolist())
            beam_text = vocab.decode(beam_sequence.tolist())
            
            if show_details:
                print(f"  🔤 贪婪解码文本: '{greedy_text}'")
                print(f"  🔤 束搜索文本: '{beam_text}' (得分: {beam_score:.3f})")
            
            # 选择最终结果 (可以根据需要选择贪婪或束搜索)
            final_text = beam_text if beam_text and len(beam_text.strip()) > 0 else greedy_text
            
            total_time = time.time() - start_time
            
            if show_details:
                print(f"\n🎯 最终结果:")
                print(f"  识别文本: '{final_text}'")
                print(f"  ⏱️  总耗时: {total_time:.3f}秒")
                print(f"    - 预处理: {preprocess_time:.3f}秒")
                print(f"    - 推理: {inference_time:.3f}秒")
            
            return {
                'text': final_text,
                'greedy_text': greedy_text,
                'beam_text': beam_text,
                'beam_score': beam_score,
                'processing_time': {
                    'total': total_time,
                    'preprocessing': preprocess_time,
                    'inference': inference_time
                },
                'spectrogram_shape': spectrogram_features.shape,
                'success': True
            }
            
        except Exception as e:
            if show_details:
                print(f"❌ 处理失败: {e}")
            
            return {
                'text': '',
                'success': False,
                'error': str(e)
            }
    
    def _greedy_decode(self, encoder_output, max_length=10):
        """贪婪解码"""
        decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(self.device)
        
        for step in range(max_length):
            output = self.model.decode_step(decoded_seq, encoder_output)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_seq = torch.cat([decoded_seq, next_token], dim=1)
            
            if next_token.item() == vocab.get_eos_idx():
                break
        
        return decoded_seq.squeeze(0)
    
    def _beam_search_decode(self, encoder_output, beam_size=3, max_length=10):
        """束搜索解码"""
        device = encoder_output.device
        beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == vocab.get_eos_idx():
                    new_beams.append((seq, score))
                    continue
                
                output = self.model.decode_step(seq, encoder_output)
                logits = output[:, -1, :]
                
                # 对过早EOS添加惩罚
                if seq.size(1) < 3:
                    logits[0, vocab.get_eos_idx()] -= 1.0
                
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
    
    def batch_process_directory(self, audio_dir, output_file=None):
        """批量处理目录中的所有音频文件"""
        print(f"\n📁 批量处理目录: {audio_dir}")
        
        # 支持的音频格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        # 找到所有音频文件
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
            audio_files.extend(Path(audio_dir).glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"❌ 在 {audio_dir} 中没有找到音频文件")
            return []
        
        print(f"📊 找到 {len(audio_files)} 个音频文件")
        
        results = []
        for i, audio_file in enumerate(audio_files):
            print(f"\n处理 {i+1}/{len(audio_files)}: {audio_file.name}")
            
            result = self.process_new_audio(str(audio_file), show_details=False)
            result['filename'] = audio_file.name
            results.append(result)
            
            if result['success']:
                print(f"  ✅ '{result['text']}' ({result['processing_time']['total']:.2f}s)")
            else:
                print(f"  ❌ 失败: {result['error']}")
        
        # 保存结果
        if output_file:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n💾 结果已保存到: {output_file}")
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        avg_time = np.mean([r['processing_time']['total'] for r in results if r['success']])
        
        print(f"\n📊 批量处理统计:")
        print(f"  成功: {success_count}/{len(results)}")
        print(f"  平均耗时: {avg_time:.3f}秒/文件")
        
        return results


def create_test_audio_examples():
    """创建测试音频文件示例"""
    print("🎵 创建测试音频文件示例")
    
    # 这里可以创建一些测试音频文件
    # 实际使用时，用户会提供真实的音频文件
    test_dir = "test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"📁 测试目录: {test_dir}")
    print("💡 请将您的音频文件放入此目录进行测试")
    
    return test_dir


def demo_complete_workflow():
    """演示完整的工作流程"""
    print("🎯 生产环境推理完整流程演示")
    print("=" * 80)
    
    # 模拟场景：用户有一个训练好的模型，想要识别新的音频文件
    
    print("📋 场景描述:")
    print("  - 您已经训练好了一个语音识别模型")
    print("  - 现在有新的音频文件需要识别")
    print("  - 这些音频文件不在原始训练数据中")
    print("  - 需要获得识别结果")
    
    print(f"\n🔧 系统工作流程:")
    print("  1. 加载训练好的模型")
    print("  2. 对新音频进行预处理 (提取频谱特征)")
    print("  3. 使用模型进行推理")
    print("  4. 解码得到文本结果")
    
    # 实际演示代码
    workflow_code = '''
# 完整的生产环境使用流程

# 1. 初始化识别器
recognizer = ProductionSpeechRecognizer(
    model_path="checkpoints/best_model.pth",  # 训练好的模型
    device="cpu"  # 或 "cuda"
)

# 2. 处理单个新音频文件
result = recognizer.process_new_audio("path/to/new_audio.wav")

if result['success']:
    print(f"识别结果: {result['text']}")
    print(f"处理时间: {result['processing_time']['total']:.3f}秒")
else:
    print(f"识别失败: {result['error']}")

# 3. 批量处理多个文件
results = recognizer.batch_process_directory(
    audio_dir="path/to/audio_directory",
    output_file="recognition_results.csv"
)
    '''
    
    print(f"\n💻 使用代码示例:")
    print(workflow_code)


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='生产环境语音识别演示')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--audio', type=str, help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str, help='音频文件目录')
    parser.add_argument('--output', type=str, help='输出结果文件')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--demo', action='store_true', help='显示完整流程演示')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_complete_workflow()
        return
    
    try:
        # 创建识别器
        recognizer = ProductionSpeechRecognizer(args.model, args.device)
        
        if args.audio:
            # 处理单个文件
            result = recognizer.process_new_audio(args.audio, show_details=True)
            
            if result['success']:
                print(f"\n🎉 识别成功!")
                print(f"最终结果: '{result['text']}'")
            else:
                print(f"\n❌ 识别失败: {result['error']}")
        
        elif args.audio_dir:
            # 批量处理
            results = recognizer.batch_process_directory(args.audio_dir, args.output)
            
            print(f"\n🎉 批量处理完成!")
            
        else:
            print("请指定 --audio 或 --audio_dir 参数")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()