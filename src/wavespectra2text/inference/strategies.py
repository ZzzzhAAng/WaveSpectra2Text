#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解码策略模块
提供不同的解码策略实现
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch.nn.functional as F


class DecoderStrategy(ABC):
    """解码策略抽象基类"""
    
    @abstractmethod
    def decode(self, logits: torch.Tensor, vocab) -> List[str]:
        """
        解码logits为文本
        
        Args:
            logits: 模型输出的logits [batch_size, seq_len, vocab_size]
            vocab: 词汇表对象
        
        Returns:
            解码后的文本列表
        """
        pass


class GreedyDecoder(DecoderStrategy):
    """贪婪解码器"""
    
    def decode(self, logits: torch.Tensor, vocab) -> List[str]:
        """贪婪解码"""
        batch_size = logits.size(0)
        decoded_texts = []
        
        for i in range(batch_size):
            # 获取每个时间步的最大概率索引
            predicted_indices = torch.argmax(logits[i], dim=-1)
            
            # 转换为文本
            text = vocab.decode(predicted_indices.cpu().numpy().tolist())
            decoded_texts.append(text)
        
        return decoded_texts


class BeamSearchDecoder(DecoderStrategy):
    """束搜索解码器"""
    
    def __init__(self, beam_size: int = 5, max_length: int = 50):
        self.beam_size = beam_size
        self.max_length = max_length
    
    def decode(self, logits: torch.Tensor, vocab) -> List[str]:
        """束搜索解码"""
        batch_size = logits.size(0)
        decoded_texts = []
        
        for i in range(batch_size):
            # 对每个样本进行束搜索
            best_sequence = self._beam_search(logits[i], vocab)
            decoded_texts.append(best_sequence)
        
        return decoded_texts
    
    def _beam_search(self, logits: torch.Tensor, vocab) -> str:
        """单个样本的束搜索"""
        seq_len, vocab_size = logits.size()
        
        # 初始化束
        beams = [(torch.tensor([vocab.word_to_idx['<SOS>']]), 0.0)]
        
        for t in range(min(seq_len, self.max_length)):
            new_beams = []
            
            for sequence, score in beams:
                if len(sequence) > 0 and sequence[-1] == vocab.word_to_idx['<EOS>']:
                    # 序列已结束
                    new_beams.append((sequence, score))
                    continue
                
                # 获取当前时间步的logits
                if t < seq_len:
                    current_logits = logits[t]
                    probabilities = F.softmax(current_logits, dim=-1)
                    
                    # 获取top-k个候选
                    top_probs, top_indices = torch.topk(probabilities, self.beam_size)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        new_sequence = torch.cat([sequence, idx.unsqueeze(0)])
                        new_score = score + torch.log(prob)
                        new_beams.append((new_sequence, new_score))
            
            # 选择top beam_size个序列
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_size]
        
        # 选择最佳序列
        best_sequence, _ = max(beams, key=lambda x: x[1])
        
        # 转换为文本
        return vocab.decode(best_sequence.cpu().numpy().tolist())


class SamplingDecoder(DecoderStrategy):
    """采样解码器"""
    
    def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def decode(self, logits: torch.Tensor, vocab) -> List[str]:
        """采样解码"""
        batch_size = logits.size(0)
        decoded_texts = []
        
        for i in range(batch_size):
            sequence = self._sample_sequence(logits[i], vocab)
            decoded_texts.append(sequence)
        
        return decoded_texts
    
    def _sample_sequence(self, logits: torch.Tensor, vocab) -> str:
        """采样生成序列"""
        seq_len = logits.size(0)
        sequence = [vocab.word_to_idx['<SOS>']]
        
        for t in range(min(seq_len, 50)):  # 限制最大长度
            if sequence[-1] == vocab.word_to_idx['<EOS>']:
                break
            
            # 获取当前时间步的logits
            current_logits = logits[t] / self.temperature
            probabilities = F.softmax(current_logits, dim=-1)
            
            # Top-k过滤
            if self.top_k is not None:
                top_probs, top_indices = torch.topk(probabilities, self.top_k)
                probabilities = torch.zeros_like(probabilities)
                probabilities[top_indices] = top_probs
                probabilities = probabilities / probabilities.sum()
            
            # Top-p过滤
            if self.top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 找到累积概率超过top_p的位置
                cutoff = torch.searchsorted(cumulative_probs, self.top_p)
                if cutoff < len(sorted_indices):
                    probabilities[sorted_indices[cutoff:]] = 0
                    probabilities = probabilities / probabilities.sum()
            
            # 采样
            next_token = torch.multinomial(probabilities, 1).item()
            sequence.append(next_token)
        
        return vocab.decode(sequence)
