"""
双输入模式推理系统
支持两种输入方式：
1. 原始音频文件 (系统内预处理)
2. 对应原始音频的频谱特征 (跳过预处理)
"""

import os
import torch
import numpy as np
import librosa
import argparse
from pathlib import Path
import time
import json

from ..core.model import create_model
from ..core.vocab import vocab
from ..data.preprocessing import SpectrogramPreprocessor
from ..core.inference import InferenceCore, BatchInference
import warnings

warnings.filterwarnings('ignore')


class DualInputSpeechRecognizer:
    """双输入模式语音识别器"""

    def __init__(self, model_path, device='cpu'):
        """
        初始化双输入识别器

        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        print(f"🚀 初始化双输入语音识别器")
        print(f"设备: {device}")

        # 使用统一推理核心
        self.inference_core = InferenceCore(model_path, device)
        self.batch_inference = BatchInference(self.inference_core)

        # 保持兼容性
        self.device = self.inference_core.device
        self.model = self.inference_core.model

        print(f"✅ 支持两种输入模式:")
        print(f"  1. 原始音频文件 (.wav, .mp3, .flac等)")
        print(f"  2. 预处理频谱特征 (.npy文件)")

        # 显示模型信息
        model_info = self.inference_core.get_model_info()
        print(f"📂 模型加载成功: {model_info['path']}")
        print(f"📊 模型参数: {sum(p.numel() for p in self.model.parameters())}")

    # 移除重复的模型加载方法，使用统一推理核心

    def recognize_from_audio(self, audio_path, show_details=True):
        """
        模式1: 从原始音频文件识别
        完整流程: 音频 → 频谱提取 → 模型推理 → 文本
        """
        if show_details:
            print(f"\n🎵 模式1: 原始音频输入")
            print(f"文件: {audio_path}")
            print("-" * 50)

        # 使用统一推理核心
        result = self.inference_core.infer_from_audio(audio_path, method='auto')

        if show_details and result['success']:
            print("🔧 步骤1: 音频预处理")
            print(f"  ✅ 频谱提取: {result['spectrogram_shape']}")
            print(f"  ⏱️  预处理耗时: {result['preprocessing_time']:.3f}秒")
            print("🧠 步骤2: 模型推理")
            print(f"  🎯 最终结果: '{result['text']}'")
            print(f"  ⏱️  推理耗时: {result['inference_time']:.3f}秒")

        # 转换结果格式以保持兼容性
        return {
            'text': result['text'],
            'success': result['success'],
            'processing_time': {
                'preprocessing': result.get('preprocessing_time', 0),
                'inference': result.get('inference_time', 0),
                'total': result.get('total_time', 0)
            },
            'input_type': 'audio_file',
            'spectrogram_shape': result.get('spectrogram_shape'),
            'method': result.get('method', 'auto'),
            'error': result.get('error')
        }

    def recognize_from_spectrogram(self, spectrogram_path, show_details=True):
        """
        模式2: 从预处理频谱特征识别
        快速流程: 频谱特征 → 模型推理 → 文本
        """
        if show_details:
            print(f"\n📊 模式2: 频谱特征输入")
            print(f"文件: {spectrogram_path}")
            print("-" * 50)

        try:
            # 加载频谱特征
            load_start = time.time()
            spectrogram_features = np.load(spectrogram_path)
            load_time = time.time() - load_start

            if show_details:
                print("📂 步骤1: 加载频谱特征")
                print(f"  ✅ 频谱加载: {spectrogram_features.shape}")
                print(f"  📊 数值范围: [{spectrogram_features.min():.3f}, {spectrogram_features.max():.3f}]")
                print(f"  ⏱️  加载耗时: {load_time:.3f}秒")
                print("  🚀 跳过预处理，直接进入推理")

            # 使用统一推理核心
            result = self.inference_core.infer_from_spectrogram(spectrogram_features, method='auto')

            if show_details and result['success']:
                print("🧠 步骤2: 模型推理")
                print(f"  🎯 最终结果: '{result['text']}'")
                print(f"  ⏱️  推理耗时: {result['inference_time']:.3f}秒")

            # 转换结果格式以保持兼容性
            return {
                'text': result['text'],
                'success': result['success'],
                'processing_time': {
                    'preprocessing': 0.0,  # 跳过预处理
                    'loading': load_time,
                    'inference': result.get('inference_time', 0),
                    'total': load_time + result.get('inference_time', 0)
                },
                'input_type': 'spectrogram_file',
                'spectrogram_shape': result.get('spectrogram_shape'),
                'method': result.get('method', 'auto'),
                'error': result.get('error')
            }

        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'input_type': 'spectrogram_file'
            }

    def recognize_from_spectrogram_array(self, spectrogram_array, show_details=True):
        """
        模式3: 从内存中的频谱数组识别
        直接输入: numpy数组 → 模型推理 → 文本
        """
        if show_details:
            print(f"\n🧮 模式3: 内存频谱数组输入")
            print(f"数组形状: {spectrogram_array.shape}")
            print("-" * 50)

        try:
            # 验证输入数组
            if not isinstance(spectrogram_array, np.ndarray):
                raise ValueError("输入必须是numpy数组")

            if show_details:
                print("🧮 步骤1: 验证频谱数组")
                print(f"  ✅ 数组形状: {spectrogram_array.shape}")
                print(f"  📊 数值范围: [{spectrogram_array.min():.3f}, {spectrogram_array.max():.3f}]")
                print("  🚀 直接进入推理 (无需加载)")

            # 使用统一推理核心
            result = self.inference_core.infer_from_spectrogram(spectrogram_array, method='auto')

            if show_details and result['success']:
                print("🧠 步骤2: 模型推理")
                print(f"  🎯 最终结果: '{result['text']}'")
                print(f"  ⏱️  推理耗时: {result['inference_time']:.3f}秒")

            # 转换结果格式以保持兼容性
            return {
                'text': result['text'],
                'success': result['success'],
                'processing_time': {
                    'preprocessing': 0.0,
                    'loading': 0.0,
                    'inference': result.get('inference_time', 0),
                    'total': result.get('inference_time', 0)
                },
                'input_type': 'spectrogram_array',
                'spectrogram_shape': result.get('spectrogram_shape'),
                'method': result.get('method', 'auto'),
                'error': result.get('error')
            }

        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'input_type': 'spectrogram_array'
            }

    # 移除重复的推理方法，已统一到 inference_core 模块

    def auto_recognize(self, input_path, show_details=True):
        """
        自动识别输入类型并处理
        根据文件扩展名自动判断是音频文件还是频谱文件
        """
        if show_details:
            print(f"\n🤖 自动模式: {input_path}")

        file_ext = Path(input_path).suffix.lower()

        # 音频文件扩展名
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        # 频谱文件扩展名
        spectrogram_extensions = ['.npy', '.npz']

        if file_ext in audio_extensions:
            if show_details:
                print("🎵 检测到音频文件，使用音频输入模式")
            return self.recognize_from_audio(input_path, show_details)

        elif file_ext in spectrogram_extensions:
            if show_details:
                print("📊 检测到频谱文件，使用频谱输入模式")
            return self.recognize_from_spectrogram(input_path, show_details)

        else:
            return {
                'text': '',
                'success': False,
                'error': f'不支持的文件格式: {file_ext}',
                'input_type': 'unknown'
            }


def create_external_spectrogram_demo():
    """创建外部频谱特征处理演示"""
    print("🔧 外部频谱特征处理演示")
    print("=" * 60)

    demo_code = '''
# 场景: 在其他系统中预处理音频，然后传递频谱特征给识别系统

# === 外部系统 (例如: 实时音频处理系统) ===
import librosa
import numpy as np

def external_audio_preprocessing(audio_path):
    """外部系统的音频预处理 - 使用统一工具"""
    from common_utils import AudioProcessor
    
    # 使用统一的音频处理器
    processor = AudioProcessor(sample_rate=48000, n_fft=1024, hop_length=512, max_length=200)
    return processor.extract_spectrogram(audio_path)

# 外部系统处理音频
audio_file = "external_audio.wav"
spectrogram = external_audio_preprocessing(audio_file)

# 保存频谱特征
np.save("external_spectrogram.npy", spectrogram)

# === 语音识别系统 ===
from dual_input_inference import DualInputSpeechRecognizer

# 初始化识别器
recognizer = DualInputSpeechRecognizer("checkpoints/best_model.pth")

# 直接从频谱特征识别 (跳过预处理，速度更快)
result = recognizer.recognize_from_spectrogram("external_spectrogram.npy")
print(f"识别结果: {result['text']}")

# 或者使用内存数组
result = recognizer.recognize_from_spectrogram_array(spectrogram)
print(f"识别结果: {result['text']}")
    '''

    print("💻 使用示例代码:")
    print(demo_code)

    return demo_code


def compare_input_modes():
    """对比两种输入模式的性能"""
    print("\n📊 输入模式性能对比")
    print("=" * 60)

    comparison = {
        "特征": ["预处理时间", "推理时间", "总时间", "内存占用", "适用场景"],
        "音频输入": ["2-3秒", "0.3-0.5秒", "2.5-3.5秒", "中等", "一般使用、开发测试"],
        "频谱输入": ["0秒", "0.3-0.5秒", "0.3-0.5秒", "低", "高性能、批量处理、实时系统"]
    }

    print(f"{'特征':<15} {'音频输入':<20} {'频谱输入':<20}")
    print("-" * 55)

    for i, feature in enumerate(comparison["特征"]):
        audio_val = comparison["音频输入"][i]
        spec_val = comparison["频谱输入"][i]
        print(f"{feature:<15} {audio_val:<20} {spec_val:<20}")

    print(f"\n💡 选择建议:")
    print(f"  🎵 音频输入: 适合一般使用，完整流程")
    print(f"  📊 频谱输入: 适合高性能需求，已有预处理系统")
    print(f"  🤖 自动模式: 根据文件扩展名自动选择")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双输入模式语音识别')
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
        create_external_spectrogram_demo()
        return

    if args.compare:
        compare_input_modes()
        return

    # 检查必需参数
    if not args.model or not args.input:
        parser.error("--model 和 --input 参数是必需的")

    print("🎯 双输入模式语音识别系统")
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
            print(f"\n🎉 识别成功!")
            print(f"输入类型: {result['input_type']}")
            print(f"识别结果: '{result['text']}'")
            print(f"总耗时: {result['processing_time']['total']:.3f}秒")

            if 'preprocessing' in result['processing_time']:
                print(f"  预处理: {result['processing_time']['preprocessing']:.3f}秒")
            if 'loading' in result['processing_time']:
                print(f"  加载: {result['processing_time']['loading']:.3f}秒")
            if 'inference' in result['processing_time']:
                print(f"  推理: {result['processing_time']['inference']:.3f}秒")
        else:
            print(f"❌ 识别失败: {result['error']}")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()