#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版批量预处理工具
增强错误处理和调试信息
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import traceback

from audio_preprocessing import PreprocessorFactory, OfflinePreprocessor


class BatchPreprocessorFixed:
    """修复版批量预处理器 - 增强错误处理"""

    def __init__(self,
                 preprocessor_type: str = 'spectrogram',
                 output_dir: str = 'data/features',
                 **preprocessor_kwargs):
        """
        初始化批量预处理器
        """
        self.preprocessor_type = preprocessor_type
        self.output_dir = output_dir
        self.preprocessor_kwargs = preprocessor_kwargs

        # 创建预处理器
        self.preprocessor = PreprocessorFactory.create(preprocessor_type, **preprocessor_kwargs)
        self.offline_processor = OfflinePreprocessor(self.preprocessor)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"批量预处理器初始化:")
        print(f"  类型: {preprocessor_type}")
        print(f"  输出目录: {output_dir}")
        print(f"  特征形状: {self.preprocessor.get_feature_shape()}")
        print(f"  配置: {self.preprocessor.get_config()}")

    def process_directory(self, audio_dir: str, labels_file: str,
                          force_recompute: bool = False, verbose: bool = True) -> pd.DataFrame:
        """批量处理音频目录 - 增强版"""
        print(f"\n🎯 开始批量处理: {audio_dir}")

        # 读取标签文件
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")

        df = pd.read_csv(labels_file)
        print(f"读取到 {len(df)} 个标签")

        # 处理结果
        processed_data = []
        success_count = 0
        failed_files = []

        # 进度条
        pbar = tqdm(df.iterrows(), total=len(df), desc="处理音频文件")

        for idx, row in pbar:
            audio_file = row['filename']
            label = row['label']
            audio_path = os.path.join(audio_dir, audio_file)

            # 更新进度条显示
            pbar.set_postfix({
                '当前': audio_file[:15] + '...' if len(audio_file) > 15 else audio_file,
                '成功': success_count,
                '失败': len(failed_files)
            })

            if not os.path.exists(audio_path):
                error_msg = f"文件不存在: {audio_path}"
                failed_files.append({'file': audio_file, 'reason': error_msg})
                if verbose:
                    print(f"\n❌ {error_msg}")
                continue

            try:
                # 处理音频文件
                if verbose:
                    print(f"\n🎵 处理: {audio_file}")

                features = self.offline_processor.process_file(
                    audio_path, force_recompute=force_recompute
                )

                if verbose:
                    print(f"✅ 特征提取成功，形状: {features.shape}")

                # 保存特征文件
                feature_filename = f"{Path(audio_file).stem}.npy"
                feature_path = os.path.join(self.output_dir, feature_filename)

                # 保存特征
                import numpy as np
                np.save(feature_path, features)

                if verbose:
                    print(f"💾 保存到: {feature_path}")

                processed_data.append({
                    'spectrum_file': feature_filename,
                    'original_audio': audio_file,
                    'label': label,
                    'shape': str(features.shape),
                    'feature_path': feature_path
                })

                success_count += 1

            except Exception as e:
                error_msg = f"处理失败: {str(e)}"
                failed_files.append({
                    'file': audio_file,
                    'reason': error_msg,
                    'traceback': traceback.format_exc()
                })

                if verbose:
                    print(f"\n❌ {audio_file}: {error_msg}")
                    print(f"详细错误: {traceback.format_exc()}")

        pbar.close()

        # 保存处理结果
        self._save_results(processed_data, failed_files, len(df))

        return pd.DataFrame(processed_data)

    def _save_results(self, processed_data: list, failed_files: list, total_files: int):
        """保存处理结果"""
        # 保存成功处理的索引
        if processed_data:
            processed_df = pd.DataFrame(processed_data)
            index_file = os.path.join(self.output_dir, 'spectrum_index.csv')
            processed_df.to_csv(index_file, index=False, encoding='utf-8')
            print(f"✅ 索引文件保存: {index_file}")

        # 保存失败文件列表 (包含详细错误信息)
        if failed_files:
            failed_df = pd.DataFrame(failed_files)
            failed_file = os.path.join(self.output_dir, 'failed_files.csv')
            failed_df.to_csv(failed_file, index=False, encoding='utf-8')
            print(f"❌ 失败文件列表: {failed_file}")

            # 保存详细错误日志
            error_log_file = os.path.join(self.output_dir, 'error_log.txt')
            with open(error_log_file, 'w', encoding='utf-8') as f:
                f.write("批量预处理错误日志\n")
                f.write("=" * 50 + "\n\n")
                for error in failed_files:
                    f.write(f"文件: {error['file']}\n")
                    f.write(f"错误: {error['reason']}\n")
                    if 'traceback' in error:
                        f.write(f"详细信息:\n{error['traceback']}\n")
                    f.write("-" * 30 + "\n\n")
            print(f"📝 详细错误日志: {error_log_file}")

        # 保存预处理配置
        config_file = os.path.join(self.output_dir, 'preprocess_config.json')
        self.offline_processor.save_config(config_file)

        # 保存处理统计
        stats = {
            'total_files': total_files,
            'success_count': len(processed_data),
            'failed_count': len(failed_files),
            'success_rate': len(processed_data) / total_files if total_files > 0 else 0,
            'preprocessor_type': self.preprocessor_type,
            'output_directory': self.output_dir,
            'failed_files': [f['file'] for f in failed_files]
        }

        stats_file = os.path.join(self.output_dir, 'process_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print(f"\n📊 处理统计:")
        print(f"  总文件数: {total_files}")
        print(f"  成功处理: {len(processed_data)}")
        print(f"  处理失败: {len(failed_files)}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        print(f"  配置文件: {config_file}")
        print(f"  统计文件: {stats_file}")

        if failed_files:
            print(f"\n❌ 失败文件:")
            for error in failed_files[:5]:  # 只显示前5个
                print(f"  - {error['file']}: {error['reason']}")
            if len(failed_files) > 5:
                print(f"  ... 还有 {len(failed_files) - 5} 个失败文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='修复版批量音频预处理工具')

    # 基本参数
    parser.add_argument('--audio_dir', default='data/audio', help='音频文件目录')
    parser.add_argument('--labels_file', default='data/labels.csv', help='标签文件')
    parser.add_argument('--output_dir', default='data/features', help='输出目录')

    # 预处理器参数
    parser.add_argument('--preprocessor', default='spectrogram',
                        choices=PreprocessorFactory.list_available(),
                        help='预处理器类型')
    parser.add_argument('--sample_rate', type=int, default=48000, help='采样率')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=512, help='跳跃长度')
    parser.add_argument('--max_length', type=int, default=200, help='最大序列长度')
    parser.add_argument('--n_mels', type=int, default=128, help='Mel频谱的mel数量')

    # 操作参数
    parser.add_argument('--force_recompute', action='store_true', help='强制重新计算')
    parser.add_argument('--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    print("🎯 修复版批量音频预处理工具")
    print("=" * 60)

    # 准备预处理器参数
    preprocessor_kwargs = {
        'sample_rate': args.sample_rate,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'max_length': args.max_length
    }

    if args.preprocessor == 'mel_spectrogram':
        preprocessor_kwargs['n_mels'] = args.n_mels

    try:
        # 创建批量预处理器
        batch_processor = BatchPreprocessorFixed(
            preprocessor_type=args.preprocessor,
            output_dir=args.output_dir,
            **preprocessor_kwargs
        )

        # 处理
        result = batch_processor.process_directory(
            args.audio_dir,
            args.labels_file,
            force_recompute=args.force_recompute,
            verbose=not args.quiet
        )

        if len(result) > 0:
            print(f"\n✅ 批量预处理完成!")
            print(f"成功处理 {len(result)} 个文件")
            print(f"现在可以使用预计算模式加载数据集，大幅提升训练速度")
        else:
            print(f"\n❌ 没有文件处理成功，请检查错误日志")

    except Exception as e:
        print(f"\n💥 批量预处理器初始化失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()