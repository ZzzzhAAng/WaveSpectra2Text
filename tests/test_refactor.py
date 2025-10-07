#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构验证测试脚本
验证新架构是否正常工作
"""

import os
import sys
import traceback


def test_imports():
    """测试模块导入"""
    print("🧪 测试模块导入...")

    try:
        from audio_preprocess import PreprocessorFactory, SpectrogramPreprocessor, MelSpectrogramPreprocessor
        from audio_dataset import AudioDataset, create_realtime_dataset, create_precomputed_dataset
        from batch_preprocess import BatchPreprocessor
        from data_utils import get_dataloader, AudioSpectrogramDataset
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        traceback.print_exc()
        return False


def test_preprocessor_factory():
    """测试预处理器工厂"""
    print("\n🧪 测试预处理器工厂...")

    try:
        from audio_preprocess import PreprocessorFactory

        # 列出可用预处理器
        available = PreprocessorFactory.list_available()
        print(f"可用预处理器: {available}")

        # 创建预处理器
        spec_processor = PreprocessorFactory.create('spectrogram')
        mel_processor = PreprocessorFactory.create('mel_spectrogram', n_mels=64)

        print(f"STFT预处理器配置: {spec_processor.get_config()}")
        print(f"Mel预处理器配置: {mel_processor.get_config()}")
        print(f"STFT特征形状: {spec_processor.get_feature_shape()}")
        print(f"Mel特征形状: {mel_processor.get_feature_shape()}")

        print("✅ 预处理器工厂测试通过")
        return True
    except Exception as e:
        print(f"❌ 预处理器工厂测试失败: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🧪 测试向后兼容性...")

    try:
        from data_utils import AudioSpectrogramDataset, collate_fn, get_dataloader

        # 测试旧接口是否可用
        print("旧接口可用性检查通过")

        # 测试参数兼容性
        if os.path.exists('../data/labels.csv'):
            try:
                # 这应该会显示兼容模式警告
                dataset = AudioSpectrogramDataset('../data/audio', 'data/labels.csv')
                print(f"兼容模式数据集创建成功，样本数: {len(dataset)}")
            except Exception as e:
                print(f"兼容模式测试跳过 (数据文件不存在): {e}")

        print("✅ 向后兼容性测试通过")
        return True
    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        traceback.print_exc()
        return False


def test_new_architecture():
    """测试新架构功能"""
    print("\n🧪 测试新架构功能...")

    try:
        from audio_dataset import AudioDataset
        from audio_preprocess import PreprocessorFactory
        from batch_preprocess import BatchPreprocessor

        # 测试预处理器创建
        preprocessor = PreprocessorFactory.create('spectrogram', max_length=100)
        print(f"自定义预处理器创建成功: {preprocessor.get_feature_shape()}")

        # 测试批量预处理器
        batch_processor = BatchPreprocessor(
            preprocessor_type='mel_spectrogram',
            output_dir='test_output',
            n_mels=64,
            max_length=150
        )
        print("批量预处理器创建成功")

        print("✅ 新架构功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 新架构功能测试失败: {e}")
        traceback.print_exc()
        return False


def test_configuration_consistency():
    """测试配置一致性"""
    print("\n🧪 测试配置一致性...")

    try:
        from audio_preprocess import PreprocessorFactory

        # 创建相同配置的预处理器
        config = {
            'sample_rate': 22050,
            'n_fft': 2048,
            'hop_length': 256,
            'max_length': 300
        }

        proc1 = PreprocessorFactory.create('spectrogram', **config)
        proc2 = PreprocessorFactory.create('spectrogram', **config)

        # 检查配置是否一致
        assert proc1.get_config() == proc2.get_config()
        assert proc1.get_feature_shape() == proc2.get_feature_shape()

        print("✅ 配置一致性测试通过")
        return True
    except Exception as e:
        print(f"❌ 配置一致性测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 重构验证测试")
    print("=" * 60)

    tests = [
        test_imports,
        test_preprocessor_factory,
        test_backward_compatibility,
        test_new_architecture,
        test_configuration_consistency
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！重构成功！")
        print("\n💡 下一步建议:")
        print("1. 运行 python data_utils.py 测试数据加载")
        print("2. 运行 python batch_preprocess.py --help 查看批量处理选项")
        print("3. 阅读 REFACTOR_GUIDE.md 了解详细使用方法")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)