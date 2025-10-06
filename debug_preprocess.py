#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试预处理脚本
用于诊断批量预处理失败的原因
"""

import os
import pandas as pd
import traceback

def test_single_file():
    """测试处理单个文件"""
    print("🔍 测试单个文件处理...")
    
    # 测试导入
    try:
        from audio_preprocessing import SpectrogramPreprocessor
        print("✅ 成功导入 SpectrogramPreprocessor")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False
    
    # 创建预处理器
    try:
        preprocessor = SpectrogramPreprocessor()
        print("✅ 成功创建预处理器")
        print(f"配置: {preprocessor.get_config()}")
    except Exception as e:
        print(f"❌ 创建预处理器失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试处理单个文件
    test_file = "data/audio/Chinese_Number_01.wav"
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    try:
        print(f"🎵 处理测试文件: {test_file}")
        features = preprocessor.process(test_file)
        print(f"✅ 成功处理! 特征形状: {features.shape}")
        print(f"特征类型: {type(features)}")
        print(f"特征数据类型: {features.dtype}")
        return True
    except Exception as e:
        print(f"❌ 处理文件失败: {e}")
        traceback.print_exc()
        return False

def test_audio_loading():
    """测试音频加载"""
    print("\n🔍 测试音频加载...")
    
    try:
        import librosa
        import numpy as np
        print("✅ 成功导入 librosa 和 numpy")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    test_file = "data/audio/Chinese_Number_01.wav"
    try:
        print(f"🎵 加载音频文件: {test_file}")
        audio, sr = librosa.load(test_file, sr=48000)
        print(f"✅ 成功加载音频!")
        print(f"音频长度: {len(audio)} 样本")
        print(f"采样率: {sr} Hz")
        print(f"时长: {len(audio)/sr:.2f} 秒")
        
        # 测试 STFT
        stft = librosa.stft(audio, n_fft=1024, hop_length=512)
        print(f"✅ STFT 成功! 形状: {stft.shape}")
        
        magnitude = np.abs(stft)
        print(f"✅ 幅度谱计算成功! 形状: {magnitude.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 音频处理失败: {e}")
        traceback.print_exc()
        return False

def test_file_permissions():
    """测试文件权限"""
    print("\n🔍 测试文件权限...")
    
    # 检查音频文件权限
    audio_dir = "data/audio"
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            if os.path.exists(filepath):
                if os.access(filepath, os.R_OK):
                    print(f"✅ {filename} - 可读")
                else:
                    print(f"❌ {filename} - 不可读")
            else:
                print(f"❌ {filename} - 不存在")
    
    # 检查输出目录权限
    output_dir = "data/features"
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ 输出目录 {output_dir} - 可写")
        return True
    except Exception as e:
        print(f"❌ 输出目录权限问题: {e}")
        return False

def test_batch_processor():
    """测试批量处理器"""
    print("\n🔍 测试批量处理器...")
    
    try:
        from batch_preprocess import BatchPreprocessor
        print("✅ 成功导入 BatchPreprocessor")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False
    
    try:
        processor = BatchPreprocessor(output_dir="data/features_debug")
        print("✅ 成功创建批量处理器")
        
        # 测试处理单个文件
        labels_file = "data/labels.csv"
        df = pd.read_csv(labels_file)
        first_row = df.iloc[0]
        
        audio_file = first_row['filename']
        audio_path = os.path.join("data/audio", audio_file)
        
        print(f"🎵 测试处理: {audio_path}")
        features = processor.offline_processor.process_file(audio_path)
        print(f"✅ 单文件处理成功! 形状: {features.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 批量处理器测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 预处理调试工具")
    print("=" * 60)
    
    tests = [
        ("文件权限检查", test_file_permissions),
        ("音频加载测试", test_audio_loading), 
        ("单文件处理测试", test_single_file),
        ("批量处理器测试", test_batch_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行异常: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    failed_tests = [name for name, result in results if not result]
    if failed_tests:
        print(f"\n❌ 失败的测试: {', '.join(failed_tests)}")
        print("\n💡 建议:")
        if "音频加载测试" in failed_tests:
            print("- 检查 librosa 是否正确安装: pip install librosa soundfile")
        if "文件权限检查" in failed_tests:
            print("- 检查文件权限和路径是否正确")
        if "单文件处理测试" in failed_tests:
            print("- 检查音频文件格式是否支持")
    else:
        print("\n🎉 所有测试通过! 预处理应该可以正常工作")

if __name__ == "__main__":
    main()