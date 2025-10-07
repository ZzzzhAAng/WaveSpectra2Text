# -*- coding: utf-8 -*-
"""
数据处理工具 - 重构版本
使用新的低耦合架构，支持多种预处理策略和数据加载模式
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

# 导入新的模块
from audio_preprocess import PreprocessorFactory
from audio_dataset import AudioDataset, FlexibleDataLoader, create_realtime_dataset, create_precomputed_dataset
from vocab import vocab

warnings.filterwarnings('ignore')


# 为了向后兼容，保留旧的类名和接口
class AudioSpectrogramDataset(AudioDataset):
    """音频频谱数据集 - 兼容旧接口的包装器"""

    def __init__(self, audio_dir, labels_file, sample_rate=48000, n_fft=1024,
                 hop_length=512, max_length=200, use_cache=True):
        """
        Args:
            audio_dir: 音频文件目录
            labels_file: 标签文件路径
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            max_length: 最大序列长度
            use_cache: 是否使用缓存 (新增参数)
        """
        print("⚠️  使用兼容模式 - 建议迁移到新的 AudioDataset 接口")

        # 创建预处理器
        preprocessor = PreprocessorFactory.create(
            'spectrogram',
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            max_length=max_length
        )

        # 使用缓存目录
        cache_dir = 'cache/legacy_features' if use_cache else None

        # 调用父类初始化
        super().__init__(
            labels_file=labels_file,
            audio_dir=audio_dir,
            preprocessor=preprocessor,
            cache_dir=cache_dir,
            mode='realtime'
        )

        # 为了兼容旧接口，设置一些属性
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length

    def __getitem__(self, idx):
        """获取数据项 - 兼容旧接口"""
        sample = super().__getitem__(idx)

        # 重命名键以保持兼容性
        return {
            'spectrogram': sample['features'],  # 重命名 features -> spectrogram
            'label': sample['label'],
            'text': sample['text'],
            'filename': sample['filename']
        }


def collate_fn(batch):
    """批处理函数 - 兼容旧接口"""
    # 转换键名
    converted_batch = []
    for sample in batch:
        converted_batch.append({
            'features': sample['spectrogram'],  # 重命名 spectrogram -> features
            'label': sample['label'],
            'text': sample['text'],
            'filename': sample['filename']
        })

    # 使用新的 collate_fn
    result = FlexibleDataLoader.collate_fn(converted_batch)

    # 重命名输出键以保持兼容性
    return {
        'spectrograms': result['features'],  # 重命名 features -> spectrograms
        'labels': result['labels'],
        'texts': result['texts'],
        'filenames': result['filenames']
    }


def create_labels_file_if_not_exists(labels_file='data/labels.csv'):
    """如果标签文件不存在，创建示例标签文件 - 使用统一工具"""
    from common_utils import LabelManager
    
    if os.path.exists(labels_file):
        print(f"标签文件已存在: {labels_file}")
        return

    # 使用统一工具创建标签模板
    success = LabelManager.create_labels_template('data/audio', labels_file, auto_labels=True)
    
    if not success:
        # 如果没有音频文件，创建示例标签文件
        labels_data = {
            'filename': [
                '1.wav', '2.wav', '3.wav', '4.wav', '5.wav',
                '6.wav', '7.wav', '8.wav', '9.wav', '10.wav'
            ],
            'label': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        }

        df = pd.DataFrame(labels_data)
        df.to_csv(labels_file, index=False, encoding='utf-8')
        print(f"已创建示例标签文件: {labels_file}")
        print("请根据你的实际音频文件修改标签文件中的filename字段")


def check_audio_files(audio_dir, labels_file):
    """检查音频文件是否存在 - 使用统一工具"""
    from common_utils import LabelManager
    
    # 使用统一工具进行验证，但保持原有的输出格式
    if not os.path.exists(labels_file):
        print(f"错误: 标签文件不存在 {labels_file}")
        return False

    df = pd.read_csv(labels_file)
    missing_files = []
    existing_files = []

    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(audio_path):
            existing_files.append(row['filename'])
        else:
            missing_files.append(row['filename'])

    print(f"音频文件检查结果:")
    print(f"  找到的文件: {len(existing_files)}")
    print(f"  缺失的文件: {len(missing_files)}")

    if existing_files:
        print(f"  存在的文件: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}")

    if missing_files:
        print(f"  缺失的文件: {missing_files}")
        print("请确保音频文件存在于指定目录中")

    return len(missing_files) == 0


def get_dataloader(audio_dir='data/audio', labels_file='data/labels.csv',
                   batch_size=4, shuffle=True, num_workers=0, mode='auto', **kwargs):
    """
    获取数据加载器 - 支持多种模式

    Args:
        audio_dir: 音频文件目录
        labels_file: 标签文件路径
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        mode: 数据加载模式 ('auto', 'realtime', 'precomputed', 'legacy')
        **kwargs: 其他参数
    """
    if mode == 'auto':
        # 自动选择模式
        precomputed_dir = kwargs.get('precomputed_dir', 'data/features')
        if os.path.exists(os.path.join(precomputed_dir, 'spectrum_index.csv')):
            print("🚀 检测到预计算特征，使用预计算模式")
            mode = 'precomputed'
        else:
            print("⚡ 使用实时计算模式")
            mode = 'realtime'

    if mode == 'legacy':
        # 兼容旧接口
        dataset = AudioSpectrogramDataset(audio_dir, labels_file, **kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    elif mode == 'realtime':
        # 实时计算模式
        dataset = create_realtime_dataset(
            labels_file=labels_file,
            audio_dir=audio_dir,
            cache_dir=kwargs.get('cache_dir', 'cache/features'),
            **kwargs
        )

    elif mode == 'precomputed':
        # 预计算模式
        precomputed_dir = kwargs.get('precomputed_dir', 'data/features')
        dataset = create_precomputed_dataset(
            labels_file=labels_file,
            precomputed_dir=precomputed_dir
        )

    else:
        raise ValueError(f"不支持的模式: {mode}")

    return FlexibleDataLoader.create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# 新增便捷函数
def get_realtime_dataloader(audio_dir='data/audio', labels_file='data/labels.csv',
                           batch_size=4, shuffle=True, num_workers=0, **kwargs):
    """获取实时计算数据加载器"""
    return get_dataloader(
        audio_dir=audio_dir,
        labels_file=labels_file,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        mode='realtime',
        **kwargs
    )


def get_precomputed_dataloader(labels_file='data/labels.csv', precomputed_dir='data/features',
                              batch_size=4, shuffle=True, num_workers=0):
    """获取预计算数据加载器"""
    return get_dataloader(
        labels_file=labels_file,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        mode='precomputed',
        precomputed_dir=precomputed_dir
    )


if __name__ == "__main__":
    print("🎯 重构后的数据处理工具测试")
    print("=" * 50)

    # 检查并创建标签文件
    create_labels_file_if_not_exists()

    # 检查音频文件
    audio_dir = 'data/audio'
    labels_file = 'data/labels.csv'

    if check_audio_files(audio_dir, labels_file):
        print("✅ 所有音频文件都存在，可以开始训练")

        # 测试不同模式的数据加载
        try:
            print("\n🧪 测试自动模式数据加载...")
            dataloader = get_dataloader(batch_size=2, mode='auto')
            print(f"数据集大小: {len(dataloader.dataset)}")

            # 获取一个批次
            for batch in dataloader:
                print(f"特征形状: {batch['features'].shape}")
                print(f"标签形状: {batch['labels'].shape}")
                print(f"文本: {batch['texts']}")
                print(f"文件名: {batch['filenames']}")
                break

            print("\n🧪 测试兼容模式数据加载...")
            legacy_dataloader = get_dataloader(batch_size=2, mode='legacy')

            for batch in legacy_dataloader:
                print(f"频谱形状 (兼容): {batch['spectrograms'].shape}")
                print(f"标签形状 (兼容): {batch['labels'].shape}")
                break

            print("\n✅ 所有测试通过!")
            print("\n💡 使用建议:")
            print("1. 对于大数据集，建议先运行批量预处理:")
            print("   python batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv")
            print("2. 然后使用预计算模式获得最佳性能:")
            print("   get_dataloader(mode='precomputed')")
            print("3. 对于小数据集或开发阶段，可以使用实时模式:")
            print("   get_dataloader(mode='realtime')")

        except Exception as e:
            print(f"❌ 测试数据加载时出错: {e}")
            print("请确保安装了librosa和相关依赖")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  部分音频文件缺失，请检查文件路径")
        print("💡 提示: 可以运行以下命令创建示例数据:")
        print("   python batch_preprocess.py --migrate")