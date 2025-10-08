#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块
提供语音识别任务的评估指标计算
"""

import numpy as np
from typing import List, Union, Tuple
import re


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
    
    Returns:
        准确率 (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
    
    return correct / len(predictions)


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    计算词错误率 (Word Error Rate)
    
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
    
    Returns:
        词错误率 (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.strip().split()
        target_words = target.strip().split()
        
        total_words += len(target_words)
        
        # 计算编辑距离
        errors = _edit_distance(pred_words, target_words)
        total_errors += errors
    
    return total_errors / total_words if total_words > 0 else 0.0


def calculate_bleu(predictions: List[str], targets: List[str], n_gram: int = 4) -> float:
    """
    计算BLEU分数 (简化版本)
    
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
        n_gram: n-gram的最大长度
    
    Returns:
        BLEU分数 (0-1)
    """
    if len(predictions) != len(targets):
        raise ValueError("预测结果和真实标签数量不匹配")
    
    total_score = 0.0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.strip().split()
        target_words = target.strip().split()
        
        # 计算n-gram精确度
        precisions = []
        for n in range(1, n_gram + 1):
            pred_ngrams = _get_ngrams(pred_words, n)
            target_ngrams = _get_ngrams(target_words, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum(1 for ngram in pred_ngrams if ngram in target_ngrams)
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        # 计算几何平均
        if all(p > 0 for p in precisions):
            score = np.exp(np.mean(np.log(precisions)))
        else:
            score = 0.0
        
        total_score += score
    
    return total_score / len(predictions)


def _edit_distance(words1: List[str], words2: List[str]) -> int:
    """计算两个词序列的编辑距离"""
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充dp表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def _get_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    """获取n-gram"""
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams
