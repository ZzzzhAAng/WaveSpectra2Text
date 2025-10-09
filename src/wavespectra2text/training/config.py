#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理系统
支持YAML配置文件，提供默认配置和配置验证
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self.get_default_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                # 假设是JSON格式
                import json
                self.config = json.load(f)
        
        # 验证配置
        self.validate_config()
        return self.config
    
    def save_config(self, config_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            else:
                import json
                json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def validate_config(self) -> None:
        """验证配置的有效性"""
        required_keys = [
            'batch_size', 'learning_rate', 'num_epochs', 
            'hidden_dim', 'encoder_layers', 'decoder_layers'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"缺少必需的配置项: {key}")
        
        # 验证数值范围
        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size 必须大于0")
        
        if not 0 < self.config['learning_rate'] < 1:
            raise ValueError("learning_rate 必须在0和1之间")
        
        if self.config['num_epochs'] <= 0:
            raise ValueError("num_epochs 必须大于0")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'experiment_name': 'default_experiment',
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'num_epochs': 100,
            'save_every': 10,
            'hidden_dim': 256,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'dropout': 0.1,
            'max_patience': 20,
            'label_smoothing': 0.1,
            'audio_dir': 'data/audio',
            'labels_file': 'data/labels.csv',
            'device': 'auto',  # auto, cpu, cuda
            'num_workers': 0,
            'pin_memory': True,
            'shuffle': True,
            'validation_split': 0.2,
            'random_seed': 42,
            'log_level': 'INFO',
            'tensorboard_log_dir': 'experiments/runs',
            'checkpoint_dir': 'experiments/checkpoints',
            'log_dir': 'experiments/logs'
        }
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
        """
        self.config.update(updates)
        self.validate_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.config[key] = value


def get_default_config(scale: str = 'medium') -> Dict[str, Any]:
    """
    获取不同规模的默认配置
    
    Args:
        scale: 数据集规模 ('small', 'medium', 'large', 'xlarge')
        
    Returns:
        配置字典
    """
    configs = {
        'small': {
            'experiment_name': f'small_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 1,
            'learning_rate': 1e-5,
            'weight_decay': 1e-3,
            'grad_clip': 0.1,
            'num_epochs': 30,
            'save_every': 10,
            'hidden_dim': 64,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'dropout': 0.5,
            'max_patience': 15,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'medium': {
            'experiment_name': f'medium_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'grad_clip': 0.5,
            'num_epochs': 50,
            'save_every': 10,
            'hidden_dim': 128,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'dropout': 0.3,
            'max_patience': 20,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'large': {
            'experiment_name': f'large_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'num_epochs': 100,
            'save_every': 20,
            'hidden_dim': 256,
            'encoder_layers': 4,
            'decoder_layers': 4,
            'dropout': 0.2,
            'max_patience': 25,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'xlarge': {
            'experiment_name': f'xlarge_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'batch_size': 8,
            'learning_rate': 2e-4,
            'weight_decay': 1e-6,
            'grad_clip': 1.0,
            'num_epochs': 200,
            'save_every': 20,
            'hidden_dim': 512,
            'encoder_layers': 6,
            'decoder_layers': 6,
            'dropout': 0.1,
            'max_patience': 30,
            'label_smoothing': 0.1,
            'device': 'auto',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_log_dir': 'runs',
            'labels_file': 'data/labels.csv',
            'audio_dir': 'data/audio',
            'validation_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': False
        }
    }
    
    base_config = ConfigManager().get_default_config()
    scale_config = configs.get(scale, configs['medium'])
    base_config.update(scale_config)
    
    return base_config


def create_config_file(config_path: str, scale: str = 'medium') -> None:
    """
    创建配置文件
    
    Args:
        config_path: 配置文件路径
        scale: 数据集规模
    """
    config = get_default_config(scale)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            import json
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"配置文件已创建: {config_path}")


if __name__ == "__main__":
    # 测试配置管理
    print("🧪 配置管理系统测试")
    print("=" * 50)
    
    # 测试默认配置
    config_manager = ConfigManager()
    print("✅ 默认配置加载成功")
    
    # 测试不同规模配置
    for scale in ['small', 'medium', 'large', 'xlarge']:
        config = get_default_config(scale)
        print(f"✅ {scale} 规模配置: {config['hidden_dim']} 隐藏层, {config['batch_size']} 批大小")
    
    # 测试配置验证
    try:
        config_manager.update_config({'batch_size': -1})
    except ValueError as e:
        print(f"✅ 配置验证正常: {e}")
    
    print("\n💡 使用方式:")
    print("from wavespectra2text.training.config import ConfigManager, get_default_config")
    print("config = ConfigManager('config.yaml')")
    print("config = get_default_config('medium')")
