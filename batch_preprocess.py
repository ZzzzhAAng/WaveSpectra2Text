#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量预处理工具 - 统一的离线预处理解决方案
替代原来的 preprocess_spectrum.py，提供更好的扩展性
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

from audio_preprocessing import PreprocessorFactory, OfflinePreprocessor
from audio_dataset import migrate_from_old_dataset


class BatchPreprocessor:
    """批量预处理器"""
    
    def __init__(self, 
                 preprocessor_type: str = 'spectrogram',
                 output_dir: str = 'data/features',
                 **preprocessor_kwargs):
        """
        初始化批量预处理器
        
        Args:
            preprocessor_type: 预处理器类型
            output_dir: 输出目录
            **preprocessor_kwargs: 预处理器参数
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
                         force_recompute: bool = False) -> pd.DataFrame:
        """批量处理音频目录"""
        print(f"\n🎯 开始批量处理: {audio_dir}")
        
        # 读取标签文件
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")
        
        df = pd.read_csv(labels_file)
        
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
            
            if not os.path.exists(audio_path):
                failed_files.append({'file': audio_file, 'reason': '文件不存在'})
                pbar.set_postfix({'失败': len(failed_files)})
                continue
            
            try:
                # 处理音频文件
                features = self.offline_processor.process_file(
                    audio_path, force_recompute=force_recompute
                )
                
                # 保存特征文件
                feature_filename = f"{Path(audio_file).stem}.npy"
                feature_path = os.path.join(self.output_dir, feature_filename)
                
                # 如果使用缓存，复制到输出目录
                if not os.path.samefile(os.path.dirname(feature_path), 
                                       self.offline_processor.cache_dir or ''):
                    import numpy as np
                    np.save(feature_path, features)
                
                processed_data.append({
                    'spectrum_file': feature_filename,
                    'original_audio': audio_file,
                    'label': label,
                    'shape': str(features.shape),
                    'feature_path': feature_path
                })
                
                success_count += 1
                pbar.set_postfix({'成功': success_count, '失败': len(failed_files)})
                
            except Exception as e:
                failed_files.append({'file': audio_file, 'reason': str(e)})
                pbar.set_postfix({'成功': success_count, '失败': len(failed_files)})
        
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
        
        # 保存失败文件列表
        if failed_files:
            failed_df = pd.DataFrame(failed_files)
            failed_file = os.path.join(self.output_dir, 'failed_files.csv')
            failed_df.to_csv(failed_file, index=False, encoding='utf-8')
            print(f"❌ 失败文件列表: {failed_file}")
        
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
            'output_directory': self.output_dir
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
    
    def validate_output(self) -> bool:
        """验证输出文件"""
        print(f"\n🔍 验证输出文件: {self.output_dir}")
        
        index_file = os.path.join(self.output_dir, 'spectrum_index.csv')
        if not os.path.exists(index_file):
            print("❌ 索引文件不存在")
            return False
        
        df = pd.read_csv(index_file)
        valid_count = 0
        invalid_files = []
        
        for _, row in df.iterrows():
            feature_path = os.path.join(self.output_dir, row['spectrum_file'])
            
            if not os.path.exists(feature_path):
                invalid_files.append(f"{row['spectrum_file']} (文件不存在)")
                continue
            
            try:
                import numpy as np
                features = np.load(feature_path)
                expected_shape = eval(row['shape']) if isinstance(row['shape'], str) else row['shape']
                
                if features.shape == expected_shape:
                    valid_count += 1
                else:
                    invalid_files.append(
                        f"{row['spectrum_file']} (形状不匹配: 期望{expected_shape}, 实际{features.shape})"
                    )
            except Exception as e:
                invalid_files.append(f"{row['spectrum_file']} (加载失败: {e})")
        
        # 保存验证结果
        validation_result = {
            'total_files': len(df),
            'valid_files': valid_count,
            'invalid_files': len(invalid_files),
            'validation_passed': len(invalid_files) == 0,
            'invalid_file_details': invalid_files
        }
        
        validation_file = os.path.join(self.output_dir, 'validation_result.json')
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2, ensure_ascii=False)
        
        print(f"📋 验证结果: {valid_count}/{len(df)} 文件有效")
        if invalid_files:
            print(f"❌ 无效文件: {len(invalid_files)} 个")
            for invalid_file in invalid_files[:5]:  # 只显示前5个
                print(f"  - {invalid_file}")
            if len(invalid_files) > 5:
                print(f"  ... 还有 {len(invalid_files) - 5} 个")
        
        print(f"验证报告保存: {validation_file}")
        return validation_result['validation_passed']


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量音频预处理工具')
    
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
    parser.add_argument('--validate', action='store_true', help='验证输出文件')
    parser.add_argument('--force_recompute', action='store_true', help='强制重新计算')
    parser.add_argument('--migrate', action='store_true', help='从旧数据集迁移')
    
    args = parser.parse_args()
    
    print("🎯 批量音频预处理工具")
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
    
    if args.migrate:
        # 迁移模式
        print("🔄 迁移模式")
        migrate_from_old_dataset(
            audio_dir=args.audio_dir,
            labels_file=args.labels_file,
            output_dir=args.output_dir,
            preprocessor_type=args.preprocessor,
            **preprocessor_kwargs
        )
    else:
        # 创建批量预处理器
        batch_processor = BatchPreprocessor(
            preprocessor_type=args.preprocessor,
            output_dir=args.output_dir,
            **preprocessor_kwargs
        )
        
        if args.validate:
            # 验证模式
            batch_processor.validate_output()
        else:
            # 处理模式
            result = batch_processor.process_directory(
                args.audio_dir,
                args.labels_file,
                force_recompute=args.force_recompute
            )
            
            if len(result) > 0:
                print(f"\n✅ 批量预处理完成!")
                print(f"现在可以使用预计算模式加载数据集，大幅提升训练速度")
                
                # 自动验证
                print("\n🔍 自动验证输出...")
                if batch_processor.validate_output():
                    print("✅ 所有文件验证通过")
                else:
                    print("❌ 部分文件验证失败，请检查")


if __name__ == "__main__":
    main()