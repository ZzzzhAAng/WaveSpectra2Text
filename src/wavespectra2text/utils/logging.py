#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志管理功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = 'logs',
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件名
        log_dir: 日志目录
        format_string: 日志格式字符串
    
    Returns:
        配置好的logger
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # 设置日志格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 创建logger
    logger = logging.getLogger('wavespectra2text')
    logger.setLevel(numeric_level)
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        if not log_file.endswith('.log'):
            log_file += '.log'
        
        file_path = Path(log_dir) / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'wavespectra2text') -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称
    
    Returns:
        logger实例
    """
    return logging.getLogger(name)
