#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动数据同步脚本
一键同步所有数据文件，更新词汇表和预处理特征
"""

import argparse
from pathlib import Path
from auto_update_system import AutoUpdateSystem


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手动数据同步工具')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理所有文件')
    parser.add_argument('--audio_dir', default='data/audio',
                        help='音频文件目录')
    parser.add_argument('--labels_file', default='data/labels.csv',
                        help='标签文件路径')
    parser.add_argument('--features_dir', default='data/features',
                        help='特征文件目录')

    args = parser.parse_args()

    print("🔄 WaveSpectra2Text 数据同步工具")
    print("=" * 50)

    # 检查文件存在性
    audio_dir = Path(args.audio_dir)
    labels_file = Path(args.labels_file)

    if not audio_dir.exists():
        print(f"❌ 音频目录不存在: {audio_dir}")
        return

    if not labels_file.exists():
        print(f"❌ 标签文件不存在: {labels_file}")
        print("请先运行: python setup_data.py")
        return

    try:
        # 创建更新系统
        updater = AutoUpdateSystem(
            audio_dir=args.audio_dir,
            labels_file=args.labels_file,
            features_dir=args.features_dir
        )

        if args.force:
            print("🔄 强制模式：重新处理所有文件...")
            # 清空状态以强制重新处理
            updater.state = {
                'last_update': None,
                'audio_files': {},
                'labels_hash': None,
                'vocab_labels': set(),
                'processed_files': {}
            }

        # 执行同步
        print("🔍 检查数据变化...")
        results = updater.check_and_update()

        # 报告结果
        print("\n📊 同步结果:")
        print(f"  音频文件变化: {'✅' if results['audio_changes'] else '❌'}")
        print(f"  标签文件变化: {'✅' if results['labels_changes'] else '❌'}")
        print(f"  词汇表更新: {'✅' if results['vocab_updated'] else '❌'}")
        print(f"  特征文件更新: {'✅' if results['features_updated'] else '❌'}")

        if any(results.values()):
            print("\n✅ 同步完成！所有文件已更新")
        else:
            print("\n✅ 所有文件都是最新的，无需同步")

        # 显示统计信息
        features_dir = Path(args.features_dir)
        if features_dir.exists():
            feature_files = list(features_dir.glob('*.npy'))
            index_file = features_dir / 'spectrum_index.csv'

            print(f"\n📁 特征文件统计:")
            print(f"  特征文件数量: {len(feature_files)}")
            print(f"  索引文件: {'✅' if index_file.exists() else '❌'}")

            if index_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(index_file)
                    print(f"  索引记录数: {len(df)}")
                except:
                    print(f"  索引记录数: 无法读取")

        print(f"\n💡 提示:")
        print(f"  - 现在可以使用预计算模式训练，速度更快")
        print(f"  - 运行训练: python train_at_different_scales/train_scale_1.py")
        print(f"  - 运行推理: python dual_input_inference.py --model <model_path> --input <input_file>")

    except Exception as e:
        print(f"\n❌ 同步过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()