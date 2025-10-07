#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版推理脚本
集成修复的束搜索和智能回退机制
"""

import os
import torch
import numpy as np
import librosa
import argparse
from tqdm import tqdm

from model import create_model
from vocab import vocab
import warnings

warnings.filterwarnings('ignore')


class FinalSpeechRecognizer:
    """最终版语音识别器 - 集成所有改进"""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()

        # 音频处理参数
        self.sample_rate = 48000
        self.n_fft = 1024
        self.hop_length = 512
        self.max_length = 200

    def _load_model(self, model_path):
        """加载模型"""
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

        print(f"模型已加载: {model_path}")
        print(f"训练epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'Unknown')}")

        return model

    def _extract_spectrogram(self, audio_path):
        """从音频文件提取频谱特征"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        log_magnitude = np.log1p(magnitude)
        spectrogram = log_magnitude.T

        if len(spectrogram) > self.max_length:
            spectrogram = spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            spectrogram = np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')

        return torch.FloatTensor(spectrogram).unsqueeze(0)

    def _greedy_decode(self, encoder_output, max_length=10):
        """贪婪解码"""
        decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(self.device)

        for step in range(max_length):
            with torch.no_grad():
                output = self.model.decode_step(decoded_seq, encoder_output)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                decoded_seq = torch.cat([decoded_seq, next_token], dim=1)

                if next_token.item() == vocab.get_eos_idx():
                    break

        return decoded_seq.squeeze(0)

    def _fixed_beam_search(self, encoder_output, beam_size=3, max_length=10):
        """修复版束搜索 - 解决EOS过早问题"""
        device = encoder_output.device
        beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]

        for step in range(max_length):
            new_beams = []

            for seq, score in beams:
                if seq[0, -1].item() == vocab.get_eos_idx():
                    # 对过短序列添加惩罚
                    if seq.size(1) < 3:
                        score -= 2.0  # 惩罚过短序列
                    new_beams.append((seq, score))
                    continue

                with torch.no_grad():
                    output = self.model.decode_step(seq, encoder_output)
                    logits = output[:, -1, :]

                    # 对过早的EOS添加惩罚
                    if seq.size(1) < 3:
                        logits[0, vocab.get_eos_idx()] -= 1.5  # 惩罚过早结束

                    probs = torch.softmax(logits, dim=-1)

                top_probs, top_indices = torch.topk(probs, beam_size)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=1)
                    new_score = score + torch.log(top_probs[:, i]).item()

                    # 长度奖励 - 鼓励生成更长序列
                    if top_indices[:, i].item() != vocab.get_eos_idx():
                        new_score += 0.1

                    new_beams.append((new_seq, new_score))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
                break

        best_seq, best_score = beams[0]
        return best_seq.squeeze(0), best_score

    def recognize_file(self, audio_path, method='auto', beam_size=3):
        """智能识别文件 - 支持多种解码方法"""
        try:
            spectrogram = self._extract_spectrogram(audio_path).to(self.device)

            with torch.no_grad():
                encoder_output = self.model.encode(spectrogram)

                if method == 'greedy':
                    # 强制使用贪婪解码
                    seq = self._greedy_decode(encoder_output)
                    text = vocab.decode(seq.tolist())
                    return {
                        'text': text,
                        'method': 'greedy',
                        'success': True
                    }

                elif method == 'beam':
                    # 强制使用束搜索
                    seq, score = self._fixed_beam_search(encoder_output, beam_size)
                    text = vocab.decode(seq.tolist())
                    return {
                        'text': text,
                        'method': 'fixed_beam_search',
                        'score': score,
                        'success': True
                    }

                else:  # method == 'auto'
                    # 智能选择：先尝试修复版束搜索
                    beam_seq, beam_score = self._fixed_beam_search(encoder_output, beam_size)
                    beam_text = vocab.decode(beam_seq.tolist())

                    # 如果束搜索结果合理，使用它
                    if beam_text and len(beam_text.strip()) > 0 and not beam_text.strip() == '':
                        return {
                            'text': beam_text,
                            'method': 'fixed_beam_search',
                            'score': beam_score,
                            'success': True
                        }
                    else:
                        # 否则回退到贪婪解码
                        greedy_seq = self._greedy_decode(encoder_output)
                        greedy_text = vocab.decode(greedy_seq.tolist())
                        return {
                            'text': greedy_text,
                            'method': 'greedy_fallback',
                            'success': True,
                            'note': '束搜索为空，使用贪婪解码'
                        }

        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e)
            }

    def recognize_batch(self, audio_paths, method='auto', beam_size=3):
        """批量识别"""
        results = []

        for audio_path in tqdm(audio_paths, desc="识别中"):
            result = self.recognize_file(audio_path, method, beam_size)
            result['file'] = audio_path
            results.append(result)

        return results


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