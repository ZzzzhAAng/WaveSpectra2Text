#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据变化监控脚本 - 简化版
实时监控data/audio和data/labels.csv的变化，自动更新相关文件
"""

import os
import time
import pandas as pd
from pathlib import Path
from auto_update_system import AutoUpdateSystem


def main():
    """主函数 - 简化的监控入口"""
    print("🔄 WaveSpectra2Text 数据监控系统")
    print("=" * 50)
    print("监控以下变化:")
    print("  📁 data/audio/ - 音频文件增删改")
    print("  📄 data/labels.csv - 标签文件更新")
    print("自动更新:")
    print("  📄 vocab.py - 词汇表")
    print("  📁 data/features/ - 预处理特征")
    print("  📄 data/features/spectrum_index.csv - 特征索引")
    print()
    
    # 检查必要文件
    audio_dir = Path('data/audio')
    labels_file = Path('data/labels.csv')
    
    if not audio_dir.exists():
        print(f"❌ 音频目录不存在: {audio_dir}")
        print("请创建目录并放入音频文件")
        return
    
    if not labels_file.exists():
        print(f"❌ 标签文件不存在: {labels_file}")
        print("请运行: python setup_data.py")
        return
    
    # 创建自动更新系统
    try:
        updater = AutoUpdateSystem()
        
        # 首次检查
        print("🔍 执行初始检查...")
        results = updater.check_and_update()
        
        if any(results.values()):
            print("✅ 初始化完成，发现并处理了数据更新")
        else:
            print("✅ 初始化完成，所有文件都是最新的")
        
        print()
        print("🔄 开始实时监控...")
        print("按 Ctrl+C 停止监控")
        print("-" * 50)
        
        # 开始监控
        updater.run_continuous_monitoring(interval=5)
        
    except KeyboardInterrupt:
        print("\n⏹️  监控已停止")
    except Exception as e:
        print(f"\n❌ 监控系统出错: {e}")
        print("请检查文件权限和依赖包")


if __name__ == "__main__":
    main()