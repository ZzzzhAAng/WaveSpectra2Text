#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理脚本
支持音频和频谱两种输入模式
"""

import sys
import os
import argparse

from wavespectra2text.inference.recognizer import DualInputSpeechRecognizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='WaveSpectra2Text 推理脚本')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--input', type=str, help='输入文件 (音频或频谱)')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'audio', 'spectrogram'],
                        help='输入模式')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--demo', action='store_true', help='显示演示代码')
    parser.add_argument('--compare', action='store_true', help='显示性能对比')

    args = parser.parse_args()

    if args.demo:
        # 显示演示代码
        print("🔧 外部频谱特征处理演示")
        print("=" * 60)
        print("💻 使用示例代码:")
        print()
        print("# 场景: 在其他系统中预处理音频，然后传递频谱特征给识别系统")
        print()
        print("# === 外部系统 (例如: 实时音频处理系统) ===")
        print("import librosa")
        print("import numpy as np")
        print()
        print("def external_audio_preprocessing(audio_path):")
        print("    \"\"\"外部系统的音频预处理 - 使用统一工具\"\"\"")
        print("    from wavespectra2text.data.utils import AudioProcessor")
        print()
        print("    # 使用统一的音频处理器")
        print("    processor = AudioProcessor(sample_rate=48000, n_fft=1024, hop_length=512, max_length=200)")
        print("    return processor.extract_spectrogram(audio_path)")
        print()
        print("# 外部系统处理音频")
        print("audio_file = \"external_audio.wav\"")
        print("spectrogram = external_audio_preprocessing(audio_file)")
        print()
        print("# 保存频谱特征")
        print("np.save(\"external_spectrogram.npy\", spectrogram)")
        print()
        print("# === 语音识别系统 ===")
        print("from wavespectra2text.inference.recognizer import DualInputSpeechRecognizer")
        print()
        print("# 初始化识别器")
        print("recognizer = DualInputSpeechRecognizer(\"experiments/checkpoints/best_model.pth\")")
        print()
        print("# 直接从频谱特征识别 (跳过预处理，速度更快)")
        print("result = recognizer.recognize_from_spectrogram(\"external_spectrogram.npy\")")
        print("print(f\"识别结果: {result['text']}\")")
        return

    if args.compare:
        # 显示性能对比
        print("📊 输入模式性能对比")
        print("=" * 60)
        print(f"{'特征':<15} {'音频输入':<20} {'频谱输入':<20}")
        print("-" * 60)
        
        comparison = {
            "特征": ["预处理时间", "推理时间", "总时间", "内存占用", "适用场景"],
            "音频输入": ["2-3秒", "0.3-0.5秒", "2.5-3.5秒", "中等", "一般使用、开发测试"],
            "频谱输入": ["0秒", "0.3-0.5秒", "0.3-0.5秒", "低", "高性能、批量处理、实时系统"]
        }
        
        for i, feature in enumerate(comparison["特征"]):
            audio_val = comparison["音频输入"][i]
            spec_val = comparison["频谱输入"][i]
            print(f"{feature:<15} {audio_val:<20} {spec_val:<20}")
        
        print(f"\n💡 选择建议:")
        print(f"  🎵 音频输入: 适合一般使用，完整流程")
        print(f"  📊 频谱输入: 适合高性能需求，已有预处理系统")
        print(f"  🤖 自动模式: 根据文件扩展名自动选择")
        return

    # 检查必需参数
    if not args.demo and not args.compare and not args.input:
        parser.error("--input 参数是必需的")

    print("🎯 WaveSpectra2Text 推理系统")
    print("=" * 60)

    try:
        # 创建识别器
        recognizer = DualInputSpeechRecognizer(args.model, args.device)

        # 根据模式处理
        if args.mode == 'auto':
            result = recognizer.auto_recognize(args.input)
        elif args.mode == 'audio':
            result = recognizer.recognize_from_audio(args.input)
        elif args.mode == 'spectrogram':
            result = recognizer.recognize_from_spectrogram(args.input)

        # 显示结果
        if result['success']:
            print(f"\n🎯 识别结果: '{result['text']}'")
            
            # 兼容不同的时间格式
            if 'total_time' in result:
                total_time = result['total_time']
            elif 'processing_time' in result and 'total' in result['processing_time']:
                total_time = result['processing_time']['total']
            else:
                total_time = 0.0
                
            print(f"⏱️  总耗时: {total_time:.3f}秒")
            print(f"📊 使用模式: {result.get('mode', 'unknown')}")
        else:
            print(f"\n❌ 识别失败: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"❌ 系统错误: {e}")
        return


if __name__ == "__main__":
    main()
