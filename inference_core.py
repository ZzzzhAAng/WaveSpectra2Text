#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理核心模块
解决inference.py和dual_input_inference.py之间的代码冗余
提供统一的模型加载、解码算法和推理接口
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

from model import create_model
from vocab import vocab
from common_utils import AudioProcessor


class InferenceCore:
    """统一的推理核心类 - 包含所有共同的推理逻辑"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化推理核心
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 音频处理参数
        self.audio_config = {
            'sample_rate': 48000,
            'n_fft': 1024,
            'hop_length': 512,
            'max_length': 200
        }
        
        # 创建音频处理器
        self.audio_processor = AudioProcessor(**self.audio_config)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        统一的模型加载方法
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            model: 加载的模型
        """
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
        
        # 存储模型信息
        self.model_info = {
            'path': model_path,
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
            'config': config
        }
        
        return model
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return self.model_info.copy()
    
    def extract_spectrogram_from_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        从音频文件提取频谱特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            spectrogram: 频谱特征
        """
        return self.audio_processor.extract_spectrogram(audio_path)
    
    def greedy_decode(self, encoder_output: torch.Tensor, max_length: int = 10) -> torch.Tensor:
        """
        贪婪解码算法
        
        Args:
            encoder_output: 编码器输出
            max_length: 最大解码长度
            
        Returns:
            decoded_seq: 解码序列
        """
        decoded_seq = torch.LongTensor([[vocab.get_sos_idx()]]).to(self.device)
        
        for step in range(max_length):
            with torch.no_grad():
                output = self.model.decode_step(decoded_seq, encoder_output)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                decoded_seq = torch.cat([decoded_seq, next_token], dim=1)
                
                if next_token.item() == vocab.get_eos_idx():
                    break
        
        return decoded_seq.squeeze(0)
    
    def beam_search_decode(self, encoder_output: torch.Tensor, 
                          beam_size: int = 3, max_length: int = 10,
                          use_length_penalty: bool = True) -> Tuple[torch.Tensor, float]:
        """
        束搜索解码算法
        
        Args:
            encoder_output: 编码器输出
            beam_size: 束搜索大小
            max_length: 最大解码长度
            use_length_penalty: 是否使用长度惩罚
            
        Returns:
            best_seq: 最佳序列
            best_score: 最佳得分
        """
        device = encoder_output.device
        beams = [(torch.LongTensor([[vocab.get_sos_idx()]]).to(device), 0.0)]
        
        for step in range(max_length):
            new_beams = []
            
            for seq, score in beams:
                if seq[0, -1].item() == vocab.get_eos_idx():
                    # 对过短序列添加惩罚
                    if use_length_penalty and seq.size(1) < 3:
                        score -= 1.0
                    new_beams.append((seq, score))
                    continue
                
                with torch.no_grad():
                    output = self.model.decode_step(seq, encoder_output)
                    logits = output[:, -1, :]
                    
                    # 对过早的EOS添加惩罚
                    if use_length_penalty and seq.size(1) < 3:
                        logits[0, vocab.get_eos_idx()] -= 1.0
                    
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, beam_size)
                    
                    for i in range(beam_size):
                        new_seq = torch.cat([seq, top_indices[:, i:i + 1]], dim=1)
                        new_score = score + torch.log(top_probs[:, i]).item()
                        
                        # 长度奖励
                        if use_length_penalty and top_indices[:, i].item() != vocab.get_eos_idx():
                            new_score += 0.1
                        
                        new_beams.append((new_seq, new_score))
            
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            
            if all(seq[0, -1].item() == vocab.get_eos_idx() for seq, _ in beams):
                break
        
        best_seq, best_score = beams[0]
        return best_seq.squeeze(0), best_score
    
    def infer_from_spectrogram(self, spectrogram: np.ndarray, 
                              method: str = 'auto', 
                              beam_size: int = 3) -> Dict:
        """
        从频谱特征进行推理
        
        Args:
            spectrogram: 频谱特征
            method: 解码方法 ('greedy', 'beam', 'auto')
            beam_size: 束搜索大小
            
        Returns:
            result: 推理结果字典
        """
        import time
        
        start_time = time.time()
        
        # 转换为tensor
        if isinstance(spectrogram, np.ndarray):
            spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(self.device)
        else:
            spectrogram_tensor = spectrogram.to(self.device)
        
        with torch.no_grad():
            # 编码
            encoder_output = self.model.encode(spectrogram_tensor)
            
            # 解码
            if method == 'greedy':
                seq = self.greedy_decode(encoder_output)
                text = vocab.decode(seq.tolist())
                
                result = {
                    'text': text,
                    'method': 'greedy',
                    'success': True
                }
                
            elif method == 'beam':
                seq, score = self.beam_search_decode(encoder_output, beam_size)
                text = vocab.decode(seq.tolist())
                
                result = {
                    'text': text,
                    'method': 'beam_search',
                    'score': score,
                    'success': True
                }
                
            else:  # method == 'auto'
                # 智能选择：先尝试束搜索
                beam_seq, beam_score = self.beam_search_decode(encoder_output, beam_size)
                beam_text = vocab.decode(beam_seq.tolist())
                
                # 如果束搜索结果合理，使用它
                if beam_text and len(beam_text.strip()) > 0:
                    result = {
                        'text': beam_text,
                        'method': 'beam_search',
                        'score': beam_score,
                        'success': True
                    }
                else:
                    # 否则回退到贪婪解码
                    greedy_seq = self.greedy_decode(encoder_output)
                    greedy_text = vocab.decode(greedy_seq.tolist())
                    
                    result = {
                        'text': greedy_text,
                        'method': 'greedy_fallback',
                        'success': True,
                        'note': '束搜索为空，使用贪婪解码'
                    }
        
        # 添加推理时间
        inference_time = time.time() - start_time
        result.update({
            'inference_time': inference_time,
            'spectrogram_shape': spectrogram.shape if isinstance(spectrogram, np.ndarray) else spectrogram_tensor.shape
        })
        
        return result
    
    def infer_from_audio(self, audio_path: Union[str, Path], 
                        method: str = 'auto', 
                        beam_size: int = 3) -> Dict:
        """
        从音频文件进行推理
        
        Args:
            audio_path: 音频文件路径
            method: 解码方法
            beam_size: 束搜索大小
            
        Returns:
            result: 推理结果字典
        """
        import time
        
        start_time = time.time()
        
        try:
            # 提取频谱特征
            preprocess_start = time.time()
            spectrogram = self.extract_spectrogram_from_audio(audio_path)
            preprocess_time = time.time() - preprocess_start
            
            # 进行推理
            result = self.infer_from_spectrogram(spectrogram, method, beam_size)
            
            # 添加预处理时间
            result.update({
                'preprocessing_time': preprocess_time,
                'total_time': time.time() - start_time,
                'input_type': 'audio_file',
                'audio_path': str(audio_path)
            })
            
            return result
            
        except Exception as e:
            return {
                'text': '',
                'success': False,
                'error': str(e),
                'input_type': 'audio_file',
                'audio_path': str(audio_path)
            }


class BatchInference:
    """批量推理工具类"""
    
    def __init__(self, inference_core: InferenceCore):
        """
        初始化批量推理
        
        Args:
            inference_core: 推理核心实例
        """
        self.core = inference_core
    
    def infer_audio_batch(self, audio_paths: list, 
                         method: str = 'auto', 
                         beam_size: int = 3,
                         show_progress: bool = True) -> list:
        """
        批量音频推理
        
        Args:
            audio_paths: 音频文件路径列表
            method: 解码方法
            beam_size: 束搜索大小
            show_progress: 是否显示进度
            
        Returns:
            results: 推理结果列表
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="批量推理")
            except ImportError:
                iterator = audio_paths
                print(f"开始批量推理 {len(audio_paths)} 个文件...")
        else:
            iterator = audio_paths
        
        for audio_path in iterator:
            result = self.core.infer_from_audio(audio_path, method, beam_size)
            result['file'] = str(audio_path)
            results.append(result)
        
        return results
    
    def infer_spectrogram_batch(self, spectrograms: list, 
                               method: str = 'auto', 
                               beam_size: int = 3,
                               show_progress: bool = True) -> list:
        """
        批量频谱推理
        
        Args:
            spectrograms: 频谱特征列表或文件路径列表
            method: 解码方法
            beam_size: 束搜索大小
            show_progress: 是否显示进度
            
        Returns:
            results: 推理结果列表
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(spectrograms, desc="批量推理")
            except ImportError:
                iterator = spectrograms
                print(f"开始批量推理 {len(spectrograms)} 个频谱...")
        else:
            iterator = spectrograms
        
        for i, spectrogram in enumerate(iterator):
            if isinstance(spectrogram, (str, Path)):
                # 如果是文件路径，加载频谱
                spectrogram_data = np.load(spectrogram)
                result = self.core.infer_from_spectrogram(spectrogram_data, method, beam_size)
                result['spectrogram_file'] = str(spectrogram)
            else:
                # 如果是numpy数组
                result = self.core.infer_from_spectrogram(spectrogram, method, beam_size)
                result['spectrogram_index'] = i
            
            results.append(result)
        
        return results


# 便捷函数
def create_inference_core(model_path: str, device: str = 'cpu') -> InferenceCore:
    """创建推理核心实例"""
    return InferenceCore(model_path, device)


def quick_infer_audio(model_path: str, audio_path: str, 
                     method: str = 'auto', device: str = 'cpu') -> str:
    """快速音频推理 - 返回识别文本"""
    core = InferenceCore(model_path, device)
    result = core.infer_from_audio(audio_path, method)
    return result.get('text', '')


def quick_infer_spectrogram(model_path: str, spectrogram: np.ndarray, 
                           method: str = 'auto', device: str = 'cpu') -> str:
    """快速频谱推理 - 返回识别文本"""
    core = InferenceCore(model_path, device)
    result = core.infer_from_spectrogram(spectrogram, method)
    return result.get('text', '')


if __name__ == "__main__":
    # 测试代码
    print("🧪 统一推理核心模块测试")
    print("=" * 50)
    
    # 模拟测试（需要实际模型文件）
    print("📋 功能列表:")
    print("  ✅ 统一模型加载")
    print("  ✅ 贪婪解码算法")
    print("  ✅ 束搜索解码算法")
    print("  ✅ 音频文件推理")
    print("  ✅ 频谱特征推理")
    print("  ✅ 批量推理支持")
    print("  ✅ 智能解码策略")
    
    print("\n💡 使用方式:")
    print("from inference_core import InferenceCore")
    print("core = InferenceCore('model.pth')")
    print("result = core.infer_from_audio('audio.wav')")
    print("print(result['text'])")