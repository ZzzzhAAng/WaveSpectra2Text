"""
训练模块 - 包含训练器、配置管理和回调
"""

from .trainer import BaseTrainer, create_trainer, SimpleTrainer, ImprovedTrainer, LargeDatasetTrainer
from .config import ConfigManager, get_default_config
from .callbacks import EarlyStoppingCallback, CheckpointCallback, LoggingCallback

__all__ = [
    "BaseTrainer",
    "create_trainer",
    "SimpleTrainer", 
    "ImprovedTrainer",
    "LargeDatasetTrainer",
    "ConfigManager",
    "get_default_config",
    "EarlyStoppingCallback",
    "CheckpointCallback", 
    "LoggingCallback",
]
