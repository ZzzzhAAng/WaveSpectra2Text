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

from model import create_model
from vocab import vocab
from audio_preprocessing import SpectrogramPreprocessor
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
        self.device = torch.device(device)
        print(f"🚀 初始化双输入语音识别器")
        print(f"设备: {self.device}")

        # 加载训练好的模型
        self.model = self._load_trained_model(model_path)

        # 初始化音频预处理器 (仅用于音频输入模式)
        self.preprocessor = SpectrogramPreprocessor(
            sample_rate=48000,
            n_fft=1024,
            hop_length=512,
            max_length=200
        )

        print(f"✅ 支持两种输入模式:")
        print(f"  1. 原始音频文件 (.wav, .mp3, .flac等)")
        print(f"  2. 预处理频谱特征 (.npy文件)")

    def _load_trained_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})

        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=config.get('hidden_dim', 256),
            encoder_layers=config.get('encoder_layers', 4),
            decoder_layers=config.get('decoder_layers', 4),
            dropout=config.get('dropout', 0.1)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"📂 模型加载成功: {model_path}")
        print(f"📊 模型参数: {sum(p.numel() for p in model.parameters())}")

        return model

    def recognize_from_audio(self, audio_path, show_details=True):
        """
        模式1: 从原始音频文件识别
        完整流程: 音频 → 频谱提取 → 模型推理 → 文本
        """
        if show_details:
            print(f"\n🎵 模式1: 原始音频输入")
            print(f"文件: {audio_path}")
            print("-" * 50)

        try:
            start_time = time.time()

            # 验证音频文件
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

            # 步骤1: 音频预处理 (提取频谱特征)
            if show_details:
                print("🔧 步骤1: 音频预处理")

            preprocess_start = time.time()
            spectrogram_features = self.preprocessor.process(audio_path)
            preprocess_time = time.time() - preprocess_start

            if show_details:
                print(f"  ✅ 频谱提取: {spectrogram_features.shape}")
                print(f"  ⏱️  预处理耗时: {preprocess_time:.3f}秒")

            # 步骤2: 模型推理
            result = self._infer_from_spectrogram(spectrogram_features, show_details)

            # 添加预处理时间
            result['processing_time']['preprocessing'] = preprocess_time
            result['processing_time']['total'] = time.time() - start_time
            result['input_type'] = 'audio_file'

            return result

        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'input_type': 'audio_file'
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
            start_time = time.time()

            # 验证频谱文件
            if not os.path.exists(spectrogram_path):
                raise FileNotFoundError(f"频谱文件不存在: {spectrogram_path}")

            # 步骤1: 加载预处理好的频谱特征
            if show_details:
                print("📂 步骤1: 加载频谱特征")

            load_start = time.time()
            spectrogram_features = np.load(spectrogram_path)
            load_time = time.time() - load_start

            # 验证频谱特征格式
            expected_shape = (200, 513)  # 或其他预期形状
            if spectrogram_features.shape != expected_shape:
                print(f"⚠️  频谱形状 {spectrogram_features.shape} 与预期 {expected_shape} 不匹配")
                # 可以尝试调整形状或给出警告

            if show_details:
                print(f"  ✅ 频谱加载: {spectrogram_features.shape}")
                print(f"  📊 数值范围: [{spectrogram_features.min():.3f}, {spectrogram_features.max():.3f}]")
                print(f"  ⏱️  加载耗时: {load_time:.3f}秒")
                print("  🚀 跳过预处理，直接进入推理")

            # 步骤2: 模型推理 (跳过预处理)
            result = self._infer_from_spectrogram(spectrogram_features, show_details)

            # 添加加载时间
            result['processing_time']['preprocessing'] = 0.0  # 跳过预处理
            result['processing_time']['loading'] = load_time
            result['processing_time']['total'] = time.time() - start_time
            result['input_type'] = 'spectrogram_file'

            return result

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
            start_time = time.time()

            # 验证输入数组
            if not isinstance(spectrogram_array, np.ndarray):
                raise ValueError("输入必须是numpy数组")

            expected_shape = (200, 513)
            if spectrogram_array.shape != expected_shape:
                print(f"⚠️  输入形状 {spectrogram_array.shape} 与预期 {expected_shape} 不匹配")

            if show_details:
                print("🧮 步骤1: 验证频谱数组")
                print(f"  ✅ 数组形状: {spectrogram_array.shape}")
                print(f"  📊 数值范围: [{spectrogram_array.min():.3f}, {spectrogram_array.max():.3f}]")
                print("  🚀 直接进入推理 (无需加载)")

            # 直接推理
            result = self._infer_from_spectrogram(spectrogram_array, show_details)

            result['processing_time']['preprocessing'] = 0.0
            result['processing_time']['loading'] = 0.0
            result['processing_time']['total'] = time.time() - start_time
            result['input_type'] = 'spectrogram_array'

            return result

        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'input_type': 'spectrogram_array'
            }

    def _infer_from_spectrogram(self, spectrogram_features, show_details=True):
        """从频谱特征进行推理 (核心推理逻辑)"""
        if show_details:
            print("🧠 步骤2: 模型推理")

        inference_start = time.time()

        # 转换为tensor
        spectrogram_tensor = torch.FloatTensor(spectrogram_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 编码阶段
            encoder_output = self.model.encode(spectrogram_tensor)

            if show_details:
                print(f"  🔍 编码器输出: {encoder_output.shape}")

            # 解码阶段 - 贪婪解码
            greedy_seq = self._greedy_decode(encoder_output)
            greedy_text = vocab.decode(greedy_seq.tolist())

            # 解码阶段 - 束搜索
            beam_seq, beam_score = self._beam_search_decode(encoder_output)
            beam_text = vocab.decode(beam_seq.tolist())

            if show_details:
                print(f"  🔤 贪婪解码: '{greedy_text}'")
                print(f"  🔤 束搜索: '{beam_text}' (得分: {beam_score:.3f})")

        inference_time = time.time() - inference_start

        # 智能选择最终结果
        final_text = beam_text if beam_text and len(beam_text.strip()) > 0 else greedy_text

        if show_details:
            print(f"\n🎯 最终结果: '{final_text}'")
            print(f"⏱️  推理耗时: {inference_time:.3f}秒")

        return {
            'text': final_text,
            'greedy_text': greedy_text,
            'beam_text': beam_text,
            'beam_score': beam_score,
            'processing_time': {
                'inference': inference_time
            },
            'spectrogram_shape': spectrogram_features.shape,
            'success': True
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
    """外部系统的音频预处理"""
    # 使用与训练时相同的参数
    audio, sr = librosa.load(audio_path, sr=48000)
    stft = librosa.stft(audio, n_fft=1024, hop_length=512)
    magnitude = np.abs(stft)
    log_magnitude = np.log1p(magnitude)
    spectrogram = log_magnitude.T

    # 标准化长度
    if len(spectrogram) > 200:
        spectrogram = spectrogram[:200]
    else:
        pad_length = 200 - len(spectrogram)
        spectrogram = np.pad(spectrogram, ((0, pad_length), (0, 0)))

    return spectrogram.astype(np.float32)

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
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--input', type=str, required=True, help='输入文件 (音频或频谱)')
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