#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统推理脚本 - 兼容层
使用统一推理核心，保持向后兼容性
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

from inference_core import InferenceCore, BatchInference
import warnings

warnings.filterwarnings('ignore')


class FinalSpeechRecognizer:
    """传统语音识别器 - 基于统一推理核心的兼容层"""

    def __init__(self, model_path, device='cpu'):
        """
        初始化识别器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        # 使用统一推理核心
        self.inference_core = InferenceCore(model_path, device)
        self.batch_inference = BatchInference(self.inference_core)
        
        # 保持兼容性
        self.device = self.inference_core.device
        self.model = self.inference_core.model
        
        # 音频处理参数 (保持兼容)
        self.sample_rate = 48000
        self.n_fft = 1024
        self.hop_length = 512
        self.max_length = 200

    # 移除重复的方法，使用统一推理核心

    def recognize_file(self, audio_path, method='auto', beam_size=3):
        """智能识别文件 - 使用统一推理核心"""
        result = self.inference_core.infer_from_audio(audio_path, method, beam_size)
        
        # 转换结果格式以保持兼容性
        return {
            'text': result['text'],
            'method': result.get('method', method),
            'success': result['success'],
            'score': result.get('score'),
            'note': result.get('note'),
            'error': result.get('error')
        }

    def recognize_batch(self, audio_paths, method='auto', beam_size=3):
        """批量识别 - 使用统一推理核心"""
        results = self.batch_inference.infer_audio_batch(
            audio_paths, method, beam_size, show_progress=True
        )
        
        # 转换结果格式以保持兼容性
        converted_results = []
        for result in results:
            converted_results.append({
                'text': result['text'],
                'method': result.get('method', method),
                'success': result['success'],
                'score': result.get('score'),
                'file': result.get('file'),
                'error': result.get('error')
            })
        
        return converted_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='最终版语音识别推理')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--audio', type=str, help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str, help='音频文件目录')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'greedy', 'beam'],
                        help='解码方法: auto(智能选择), greedy(贪婪), beam(束搜索)')
    parser.add_argument('--beam_size', type=int, default=3, help='束搜索大小')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')

    args = parser.parse_args()

    # 创建识别器
    try:
        recognizer = FinalSpeechRecognizer(args.model, args.device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    if args.audio:
        # 识别单个文件
        print(f"识别文件: {args.audio}")
        result = recognizer.recognize_file(args.audio, args.method, args.beam_size)

        if result['success']:
            print(f"识别结果: {result['text']}")
            print(f"使用方法: {result['method']}")
            if 'score' in result:
                print(f"得分: {result['score']:.4f}")
            if 'note' in result:
                print(f"注意: {result['note']}")
        else:
            print(f"识别失败: {result['error']}")

    elif args.audio_dir:
        # 批量识别目录中的文件
        print(f"批量识别目录: {args.audio_dir}")
        print(f"解码方法: {args.method}")

        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            audio_files.extend([
                os.path.join(args.audio_dir, f)
                for f in os.listdir(args.audio_dir)
                if f.lower().endswith(ext)
            ])

        if not audio_files:
            print("未找到音频文件")
            return

        results = recognizer.recognize_batch(audio_files, args.method, args.beam_size)

        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        method_counts = {}
        for r in results:
            method = r.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        print(f"\n📊 识别统计:")
        print(f"  总文件数: {len(results)}")
        print(f"  成功识别: {success_count}")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")

        # 打印结果
        print(f"\n📋 识别结果:")
        for result in results:
            filename = os.path.basename(result['file'])
            status = "✅" if result['success'] else "❌"
            method = result.get('method', 'unknown')
            print(f"  {status} {filename}: '{result['text']}' ({method})")

    else:
        print("请指定要识别的音频文件或目录")
        parser.print_help()


if __name__ == "__main__":
    main()