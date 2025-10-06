#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理分离实现 - 离线频谱提取
将音频预处理和模型推理完全分离
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import json
from tqdm import tqdm
import argparse

class SpectrumPreprocessor:
    """频谱预处理器 - 离线提取频谱特征"""
    
    def __init__(self, sample_rate=48000, n_fft=1024, hop_length=512, max_length=200):
        """
        初始化预处理器
        
        Args:
            sample_rate: 目标采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            max_length: 最大序列长度
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        
        # 频谱参数
        self.freq_bins = n_fft // 2 + 1  # 513
        
        print(f"频谱预处理器初始化:")
        print(f"  采样率: {sample_rate}Hz")
        print(f"  FFT窗口: {n_fft}")
        print(f"  跳跃长度: {hop_length}")
        print(f"  频率bins: {self.freq_bins}")
        print(f"  最大长度: {max_length}帧")
    
    def extract_spectrum_from_audio(self, audio_path):
        """从音频文件提取频谱"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 提取STFT频谱
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)  # 幅度谱
            
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
            
            return spectrogram.astype(np.float32)
            
        except Exception as e:
            print(f"处理音频文件 {audio_path} 时出错: {e}")
            return None
    
    def process_audio_directory(self, audio_dir, labels_file, output_dir):
        """批量处理音频目录"""
        print(f"\n开始批量处理音频目录: {audio_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取标签文件
        if not os.path.exists(labels_file):
            print(f"错误: 标签文件不存在 {labels_file}")
            return None
        
        df = pd.read_csv(labels_file)
        
        # 处理结果
        processed_data = []
        success_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
            audio_file = row['filename']
            label = row['label']
            
            audio_path = os.path.join(audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                # 提取频谱
                spectrogram = self.extract_spectrum_from_audio(audio_path)
                
                if spectrogram is not None:
                    # 保存频谱文件
                    spectrum_filename = f"{Path(audio_file).stem}.npy"
                    spectrum_path = os.path.join(output_dir, spectrum_filename)
                    np.save(spectrum_path, spectrogram)
                    
                    processed_data.append({
                        'spectrum_file': spectrum_filename,
                        'original_audio': audio_file,
                        'label': label,
                        'shape': spectrogram.shape
                    })
                    
                    success_count += 1
                else:
                    print(f"跳过文件: {audio_file} (处理失败)")
            else:
                print(f"跳过文件: {audio_file} (文件不存在)")
        
        # 保存处理结果索引
        processed_df = pd.DataFrame(processed_data)
        index_file = os.path.join(output_dir, 'spectrum_index.csv')
        processed_df.to_csv(index_file, index=False, encoding='utf-8')
        
        # 保存预处理参数
        params = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'max_length': self.max_length,
            'freq_bins': self.freq_bins,
            'total_files': len(df),
            'processed_files': success_count
        }
        
        params_file = os.path.join(output_dir, 'preprocess_params.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成:")
        print(f"  总文件数: {len(df)}")
        print(f"  成功处理: {success_count}")
        print(f"  频谱文件保存到: {output_dir}")
        print(f"  索引文件: {index_file}")
        print(f"  参数文件: {params_file}")
        
        return processed_df
    
    def validate_spectrum_files(self, spectrum_dir):
        """验证频谱文件"""
        print(f"\n验证频谱文件: {spectrum_dir}")
        
        index_file = os.path.join(spectrum_dir, 'spectrum_index.csv')
        if not os.path.exists(index_file):
            print("错误: 找不到索引文件 spectrum_index.csv")
            return False
        
        df = pd.read_csv(index_file)
        
        valid_count = 0
        for _, row in df.iterrows():
            spectrum_file = os.path.join(spectrum_dir, row['spectrum_file'])
            
            if os.path.exists(spectrum_file):
                try:
                    spectrum = np.load(spectrum_file)
                    expected_shape = eval(row['shape']) if isinstance(row['shape'], str) else row['shape']
                    
                    if spectrum.shape == expected_shape:
                        valid_count += 1
                    else:
                        print(f"形状不匹配: {row['spectrum_file']} - 期望{expected_shape}, 实际{spectrum.shape}")
                except Exception as e:
                    print(f"加载失败: {row['spectrum_file']} - {e}")
            else:
                print(f"文件缺失: {row['spectrum_file']}")
        
        print(f"验证结果: {valid_count}/{len(df)} 文件有效")
        return valid_count == len(df)

def create_spectrum_only_inference():
    """创建纯频谱推理脚本"""
    
    inference_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯频谱推理脚本 - 完全独立于音频文件
只需要预提取的频谱文件和训练好的模型
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from vocab import vocab

class SpectrumOnlyRecognizer:
    """纯频谱识别器 - 不依赖音频处理"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """加载模型"""
        from model import create_model
        
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
        
        return model
    
    def recognize_from_spectrum(self, spectrum_path, use_beam_search=True, beam_size=3):
        """从频谱文件识别文本"""
        try:
            # 加载频谱
            spectrogram = np.load(spectrum_path)
            spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 编码
                encoder_output = self.model.encode(spectrogram_tensor)
                
                # 解码
                if use_beam_search:
                    decoded_seq, score = self._beam_search(encoder_output, beam_size)
                else:
                    decoded_seq = self._greedy_decode(encoder_output)
                    score = None
                
                # 转换为文本
                text = vocab.decode(decoded_seq.tolist())
                
                return {
                    'text': text,
                    'score': score,
                    'success': True
                }
        
        except Exception as e:
            return {
                'text': '',
                'score': None,
                'success': False,
                'error': str(e)
            }
    
    def batch_recognize(self, spectrum_dir):
        """批量识别频谱目录"""
        index_file = os.path.join(spectrum_dir, 'spectrum_index.csv')
        
        if not os.path.exists(index_file):
            print("错误: 找不到频谱索引文件")
            return []
        
        df = pd.read_csv(index_file)
        results = []
        
        for _, row in df.iterrows():
            spectrum_path = os.path.join(spectrum_dir, row['spectrum_file'])
            result = self.recognize_from_spectrum(spectrum_path)
            
            result.update({
                'spectrum_file': row['spectrum_file'],
                'original_audio': row['original_audio'],
                'expected_label': row['label']
            })
            
            results.append(result)
        
        return results

# 使用示例
if __name__ == "__main__":
    recognizer = SpectrumOnlyRecognizer("checkpoints/best_model.pth")
    
    # 单文件识别
    result = recognizer.recognize_from_spectrum("spectrums/audio_001.npy")
    print(f"识别结果: {result['text']}")
    
    # 批量识别
    results = recognizer.batch_recognize("spectrums/")
    for result in results:
        print(f"{result['original_audio']} -> {result['text']}")
'''
    
    with open('spectrum_only_inference.py', 'w', encoding='utf-8') as f:
        f.write(inference_code)
    
    print("✅ 已创建纯频谱推理脚本: spectrum_only_inference.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='频谱预处理工具')
    parser.add_argument('--audio_dir', default='data/audio', help='音频目录')
    parser.add_argument('--labels_file', default='data/labels.csv', help='标签文件')
    parser.add_argument('--output_dir', default='data/spectrums', help='输出目录')
    parser.add_argument('--validate', action='store_true', help='验证频谱文件')
    parser.add_argument('--create_inference', action='store_true', help='创建纯频谱推理脚本')
    
    args = parser.parse_args()
    
    print("🎯 频谱预处理分离工具")
    print("=" * 60)
    
    if args.create_inference:
        create_spectrum_only_inference()
        return
    
    # 创建预处理器
    preprocessor = SpectrumPreprocessor()
    
    if args.validate:
        # 验证现有频谱文件
        preprocessor.validate_spectrum_files(args.output_dir)
    else:
        # 批量处理音频文件
        result = preprocessor.process_audio_directory(
            args.audio_dir, 
            args.labels_file, 
            args.output_dir
        )
        
        if result is not None:
            print(f"\n✅ 预处理完成!")
            print(f"现在可以使用纯频谱推理，完全独立于音频文件")
            print(f"频谱文件目录: {args.output_dir}")

if __name__ == "__main__":
    main()
'''

# 使用方法说明
使用步骤:
1. 预处理: python preprocess_spectrum.py --audio_dir data/audio --labels_file data/labels.csv --output_dir data/spectrums
2. 创建推理脚本: python preprocess_spectrum.py --create_inference  
3. 纯频谱推理: python spectrum_only_inference.py
4. 验证: python preprocess_spectrum.py --validate --output_dir data/spectrums
'''