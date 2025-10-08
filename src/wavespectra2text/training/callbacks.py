#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练回调系统
提供早停、检查点保存、日志记录等回调功能
"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseCallback(ABC):
    """回调基类"""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """设置训练器引用"""
        self.trainer = trainer
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始时的回调"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch结束时的回调"""
        pass
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """训练开始时的回调"""
        pass
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """训练结束时的回调"""
        pass


class EarlyStoppingCallback(BaseCallback):
    """早停回调"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, 
                 min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化早停回调
        
        Args:
            monitor: 监控的指标名称
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善阈值
            mode: 'min' 或 'max'，表示指标越小越好还是越大越好
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """检查是否需要早停"""
        if self.monitor not in logs:
            return
        
        current_value = logs[self.monitor]
        
        if self.best_value is None:
            self.best_value = current_value
            return
        
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            self.trainer.should_stop = True
            print(f"早停触发 - {self.monitor} 连续 {self.patience} 个epoch未改善")
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始时的回调"""
        pass


class CheckpointCallback(BaseCallback):
    """检查点保存回调"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, mode: str = 'min',
                 save_every: int = 10):
        """
        初始化检查点回调
        
        Args:
            filepath: 保存路径模板
            monitor: 监控的指标
            save_best_only: 是否只保存最佳模型
            mode: 'min' 或 'max'
            save_every: 每多少个epoch保存一次
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_every = save_every
        self.best_value = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """保存检查点"""
        should_save = False
        
        if self.save_best_only and self.monitor in logs:
            current_value = logs[self.monitor]
            
            if self.best_value is None:
                self.best_value = current_value
                should_save = True
            elif self.mode == 'min' and current_value < self.best_value:
                self.best_value = current_value
                should_save = True
            elif self.mode == 'max' and current_value > self.best_value:
                self.best_value = current_value
                should_save = True
        elif (epoch + 1) % self.save_every == 0:
            should_save = True
        
        if should_save:
            self._save_checkpoint(epoch, logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始时的回调"""
        pass
    
    def _save_checkpoint(self, epoch: int, logs: Dict[str, Any]) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
            'best_val_loss': self.trainer.best_val_loss,
            'train_losses': self.trainer.train_losses,
            'val_losses': self.trainer.val_losses,
            'config': self.trainer.config,
            'logs': logs
        }
        
        # 生成文件名
        if self.save_best_only:
            filepath = self.filepath.replace('{epoch}', 'best')
        else:
            filepath = self.filepath.format(epoch=epoch + 1)
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")


class LoggingCallback(BaseCallback):
    """日志记录回调"""
    
    def __init__(self, log_dir: str = 'experiments/logs'):
        """
        初始化日志回调
        
        Args:
            log_dir: 日志目录
        """
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """训练开始时记录"""
        print("🚀 训练开始")
        print(f"📊 配置: {self.trainer.config}")
        print(f"📁 日志目录: {self.log_dir}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """记录epoch信息"""
        print(f"Epoch {epoch + 1}/{self.trainer.config['num_epochs']}")
        print(f"  训练损失: {logs.get('train_loss', 0):.4f}")
        print(f"  训练准确率: {logs.get('train_acc', 0):.3f}")
        print(f"  验证损失: {logs.get('val_loss', 0):.4f}")
        print(f"  验证准确率: {logs.get('val_acc', 0):.3f}")
        print(f"  学习率: {logs.get('learning_rate', 0):.6f}")
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始时的回调"""
        pass
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """训练结束时记录"""
        print("✅ 训练完成")
        if hasattr(self.trainer, 'stopped_epoch'):
            print(f"📈 最佳验证损失: {self.trainer.best_val_loss:.4f}")


class TensorBoardCallback(BaseCallback):
    """TensorBoard回调"""
    
    def __init__(self, log_dir: str = 'experiments/runs'):
        """
        初始化TensorBoard回调
        
        Args:
            log_dir: TensorBoard日志目录
        """
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """初始化TensorBoard写入器"""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """记录到TensorBoard"""
        if self.writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Epoch/{key}', value, epoch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """关闭TensorBoard写入器"""
        if self.writer:
            self.writer.close()
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始时的回调"""
        pass


class CallbackList:
    """回调列表管理器"""
    
    def __init__(self, callbacks: list = None):
        """
        初始化回调列表
        
        Args:
            callbacks: 回调列表
        """
        self.callbacks = callbacks or []
        self.trainer = None
    
    def add_callback(self, callback: BaseCallback) -> None:
        """添加回调"""
        self.callbacks.append(callback)
    
    def set_trainer(self, trainer) -> None:
        """设置训练器"""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """训练开始回调"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """训练结束回调"""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch开始回调"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """epoch结束回调"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


if __name__ == "__main__":
    print("🧪 训练回调系统测试")
    print("=" * 50)
    
    print("📋 可用回调:")
    print("  ✅ EarlyStoppingCallback - 早停机制")
    print("  ✅ CheckpointCallback - 检查点保存")
    print("  ✅ LoggingCallback - 日志记录")
    print("  ✅ TensorBoardCallback - TensorBoard记录")
    print("  ✅ CallbackList - 回调列表管理")
    
    print("\n💡 使用方式:")
    print("from wavespectra2text.training.callbacks import CallbackList, EarlyStoppingCallback")
    print("callbacks = CallbackList([EarlyStoppingCallback(patience=10)])")
    print("callbacks.set_trainer(trainer)")
