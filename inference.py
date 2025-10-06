# -*- coding: utf-8 -*-
"""
推理脚本
用于使用训练好的模型进行语音识别推理
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

class SpeechRecognizer:
    def __init__(self, model_path, device='cpu'):
        """
        初始化语音识别器
        
        Args:
            model_path: 模型检查点路径
            device: 计算设备
        """
        self.device = torch.device(device)
        
        # 加载模型
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
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        config = checkpoint.get('config', {})
        
        # 创建模型
        model = create_model(
            vocab_size=vocab.vocab_size,
            hidden_dim=config.get('hidden_dim', 256),
            encoder_layers=config.get('encoder_layers', 4),
            decoder_layers=config.get('decoder_layers', 4),
            dropout=config.get('dropout', 0.1)
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"模型已加载: {model_path}")
        print(f"训练epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'Unknown')}")
        
        return model
    
    def _extract_spectrogram(self, audio_path):
        """从音频文件提取频谱特征"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 提取STFT频谱
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 转换为对数刻度
        log_magnitude = np.log1p(magnitude)
        
        # 转置使时间维度在前
        spectrogram = log_magnitude.T  # (time_steps, freq_bins)
        
        # 填充或截断到固定长度
        if len(spectrogram) > self.max_length:
            spectrogram = spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            spectrogram = np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')
        
        return torch.FloatTensor(spectrogram).unsqueeze(0)  # (1, seq_len, freq_bins)
    
    def _beam_search(self, encoder_output, beam_size=3, max_length=10):
        """束搜索解码"""
        batch_size = encoder_output.size(0)
        
        # 初始化束
        beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(self.device), 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == vocab.get_eos_idx():
                    # 已经结束的序列
                    new_beams.append((seq, score))
                    continue
                
                # 获取下一个token的概率
                with torch.no_grad():
                    output = self.model.decode_step(seq, encoder_output)
                    probs = torch.softmax(output[:, -1, :], dim=-1)
                
                # 获取top-k候选
                top_probs, top_indices = torch.topk(probs, beam_size)
                
                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_indices[:, i:i+1]], dim=1)
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
    
    def _greedy_decode(self, encoder_output, max_length=10):
        """贪婪解码"""
        batch_size = encoder_output.size(0)
        
        # 初始化解码序列
        decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(self.device)
        
        for step in range(max_length):
            with torch.no_grad():
                # 获取下一个token的概率
                output = self.model.decode_step(decoded_seq, encoder_output)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # 添加到序列中
                decoded_seq = torch.cat([decoded_seq, next_token], dim=1)
                
                # 如果生成了结束符号，停止解码
                if next_token.item() == vocab.get_eos_idx():
                    break
        
        return decoded_seq.squeeze(0)
    
    def recognize_file(self, audio_path, use_beam_search=True, beam_size=3):
        """识别单个音频文件"""
        try:
            # 提取频谱特征
            spectrogram = self._extract_spectrogram(audio_path).to(self.device)
            
            with torch.no_grad():
                # 编码
                encoder_output = self.model.encode(spectrogram)
                
                # 解码
                if use_beam_search:
                    decoded_seq, score = self._beam_search(encoder_output, beam_size)
                    decoded_indices = decoded_seq.tolist()
                else:
                    decoded_seq = self._greedy_decode(encoder_output)
                    decoded_indices = decoded_seq.tolist()
                    score = None
                
                # 解码为文本
                recognized_text = vocab.decode(decoded_indices)
                
                return {
                    'text': recognized_text,
                    'indices': decoded_indices,
                    'score': score,
                    'success': True
                }
                
        except Exception as e:
            return {
                'text': '',
                'indices': [],
                'score': None,
                'success': False,
                'error': str(e)
            }
    
    def recognize_batch(self, audio_paths, use_beam_search=True, beam_size=3):
        """批量识别音频文件"""
        results = []
        
        for audio_path in tqdm(audio_paths, desc="识别中"):
            result = self.recognize_file(audio_path, use_beam_search, beam_size)
            result['file'] = audio_path
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, audio_dir, labels_file):
        """在数据集上评估"""
        import pandas as pd
        
        # 加载标签
        labels_df = pd.read_csv(labels_file)
        
        results = []
        correct = 0
        total = 0
        
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="评估中"):
            audio_file = row['filename']
            true_label = row['label']
            
            audio_path = os.path.join(audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                result = self.recognize_file(audio_path)
                predicted_text = result['text']
                
                is_correct = (predicted_text == true_label)
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'filename': audio_file,
                    'true_label': true_label,
                    'predicted': predicted_text,
                    'correct': is_correct,
                    'success': result['success']
                })
                
                print(f"{audio_file}: 真实={true_label}, 预测={predicted_text}, "
                      f"正确={is_correct}")
            else:
                print(f"文件不存在: {audio_path}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f} ({correct}/{total})")
        
        return results, accuracy

def main():
    parser = argparse.ArgumentParser(description='语音识别推理')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--audio', type=str, help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str, help='音频文件目录')
    parser.add_argument('--labels', type=str, help='标签文件路径（用于评估）')
    parser.add_argument('--output', type=str, help='输出结果文件路径')
    parser.add_argument('--beam_size', type=int, default=3, help='束搜索大小')
    parser.add_argument('--no_beam_search', action='store_true', help='使用贪婪解码')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建识别器
    try:
        recognizer = SpeechRecognizer(args.model, args.device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    use_beam_search = not args.no_beam_search
    
    if args.audio:
        # 识别单个文件
        print(f"识别文件: {args.audio}")
        result = recognizer.recognize_file(args.audio, use_beam_search, args.beam_size)
        
        if result['success']:
            print(f"识别结果: {result['text']}")
            if result['score'] is not None:
                print(f"得分: {result['score']:.4f}")
        else:
            print(f"识别失败: {result['error']}")
    
    elif args.audio_dir and args.labels:
        # 在数据集上评估
        print(f"在数据集上评估: {args.audio_dir}")
        results, accuracy = recognizer.evaluate_on_dataset(args.audio_dir, args.labels)
        
        # 保存结果
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"结果已保存到: {args.output}")
    
    elif args.audio_dir:
        # 批量识别目录中的文件
        print(f"批量识别目录: {args.audio_dir}")
        
        # 获取所有音频文件
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
        
        results = recognizer.recognize_batch(audio_files, use_beam_search, args.beam_size)
        
        # 打印结果
        for result in results:
            status = "成功" if result['success'] else "失败"
            print(f"{result['file']}: {result['text']} ({status})")
        
        # 保存结果
        if args.output:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"结果已保存到: {args.output}")
    
    else:
        print("请指定要识别的音频文件或目录")
        parser.print_help()

if __name__ == "__main__":
    main()