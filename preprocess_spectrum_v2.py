#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理工具 V2 - 使用统一的频谱处理器
消除与data_utils的代码冗余
"""

import os
import pandas as pd
import json
from tqdm import tqdm
import argparse
from spectrum_utils import SpectrumProcessor

class UnifiedSpectrumPreprocessor:
    """统一的频谱预处理器 - 使用spectrum_utils"""
    
    def __init__(self, sample_rate=48000, n_fft=1024, hop_length=512, max_length=200):
        """初始化预处理器 - 使用统一的频谱处理器"""
        self.spectrum_processor = SpectrumProcessor(sample_rate, n_fft, hop_length, max_length)
        
        print(f"统一预处理器初始化完成")
        print(f"配置: {self.spectrum_processor.get_config()}")
    
    def process_audio_directory(self, audio_dir, labels_file, output_dir):
        """批量处理音频目录 - 核心预处理功能"""
        print(f"\n🚀 开始批量预处理: {audio_dir} -> {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取标签文件
        if not os.path.exists(labels_file):
            print(f"❌ 标签文件不存在: {labels_file}")
            return None
        
        df = pd.read_csv(labels_file)
        
        # 处理结果
        processed_data = []
        success_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
            audio_file = row['filename']
            label = row['label']
            
            audio_path = os.path.join(audio_dir, audio_file)
            
            if os.path.exists(audio_path):
                # 使用统一的频谱处理器提取频谱
                spectrogram = self.spectrum_processor.extract_spectrum_from_audio(audio_path)
                
                if spectrogram is not None:
                    # 保存频谱文件
                    spectrum_filename = f"{os.path.splitext(audio_file)[0]}.npy"
                    spectrum_path = os.path.join(output_dir, spectrum_filename)
                    
                    if self.spectrum_processor.save_spectrum_to_file(spectrogram, spectrum_path):
                        processed_data.append({
                            'spectrum_file': spectrum_filename,
                            'original_audio': audio_file,
                            'label': label,
                            'shape': str(spectrogram.shape)
                        })
                        success_count += 1
                    else:
                        print(f"❌ 保存失败: {spectrum_filename}")
                else:
                    print(f"❌ 处理失败: {audio_file}")
            else:
                print(f"❌ 文件不存在: {audio_file}")
        
        # 保存处理结果索引
        processed_df = pd.DataFrame(processed_data)
        index_file = os.path.join(output_dir, 'spectrum_index.csv')
        processed_df.to_csv(index_file, index=False, encoding='utf-8')
        
        # 保存预处理参数
        params = self.spectrum_processor.get_config()
        params.update({
            'total_files': len(df),
            'processed_files': success_count,
            'success_rate': success_count / len(df) if len(df) > 0 else 0
        })
        
        params_file = os.path.join(output_dir, 'preprocess_params.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 预处理完成:")
        print(f"   总文件数: {len(df)}")
        print(f"   成功处理: {success_count}")
        print(f"   成功率: {params['success_rate']:.1%}")
        print(f"   频谱文件: {output_dir}")
        print(f"   索引文件: {index_file}")
        print(f"   参数文件: {params_file}")
        
        return processed_df
    
    def validate_spectrum_files(self, spectrum_dir):
        """验证频谱文件完整性"""
        print(f"\n🔍 验证频谱文件: {spectrum_dir}")
        
        index_file = os.path.join(spectrum_dir, 'spectrum_index.csv')
        if not os.path.exists(index_file):
            print("❌ 索引文件不存在: spectrum_index.csv")
            return False
        
        df = pd.read_csv(index_file)
        
        valid_count = 0
        for _, row in df.iterrows():
            spectrum_file = os.path.join(spectrum_dir, row['spectrum_file'])
            
            if os.path.exists(spectrum_file):
                # 使用统一处理器验证加载
                spectrogram = self.spectrum_processor.load_spectrum_from_file(spectrum_file)
                
                if spectrogram is not None:
                    valid_count += 1
                else:
                    print(f"❌ 加载失败: {row['spectrum_file']}")
            else:
                print(f"❌ 文件缺失: {row['spectrum_file']}")
        
        success_rate = valid_count / len(df) if len(df) > 0 else 0
        print(f"✅ 验证结果: {valid_count}/{len(df)} 文件有效 ({success_rate:.1%})")
        
        return success_rate == 1.0
    
    def incremental_process(self, audio_dir, labels_file, output_dir):
        """增量处理 - 只处理新文件"""
        print(f"\n🔄 增量预处理模式")
        
        # 检查已处理的文件
        index_file = os.path.join(output_dir, 'spectrum_index.csv')
        processed_files = set()
        
        if os.path.exists(index_file):
            existing_df = pd.read_csv(index_file)
            processed_files = set(existing_df['original_audio'].tolist())
            print(f"   已处理文件: {len(processed_files)} 个")
        
        # 读取当前标签文件
        df = pd.read_csv(labels_file)
        new_files = [row['filename'] for _, row in df.iterrows() 
                    if row['filename'] not in processed_files]
        
        if not new_files:
            print("✅ 没有新文件需要处理")
            return True
        
        print(f"   新文件数量: {len(new_files)}")
        
        # 创建临时标签文件只包含新文件
        temp_df = df[df['filename'].isin(new_files)]
        temp_labels_file = os.path.join(output_dir, 'temp_new_labels.csv')
        temp_df.to_csv(temp_labels_file, index=False, encoding='utf-8')
        
        try:
            # 处理新文件
            result = self.process_audio_directory(audio_dir, temp_labels_file, output_dir)
            
            # 合并索引文件
            if result is not None and os.path.exists(index_file):
                existing_df = pd.read_csv(index_file)
                combined_df = pd.concat([existing_df, result], ignore_index=True)
                combined_df.to_csv(index_file, index=False, encoding='utf-8')
                print("✅ 索引文件已更新")
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_labels_file):
                os.remove(temp_labels_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一频谱预处理工具 V2')
    parser.add_argument('--audio_dir', default='data/audio', help='音频目录')
    parser.add_argument('--labels_file', default='data/labels.csv', help='标签文件')
    parser.add_argument('--output_dir', default='data/spectrums', help='输出目录')
    parser.add_argument('--validate', action='store_true', help='验证频谱文件')
    parser.add_argument('--incremental', action='store_true', help='增量处理模式')
    
    args = parser.parse_args()
    
    print("🎯 统一频谱预处理工具 V2")
    print("基于spectrum_utils，消除代码冗余")
    print("=" * 60)
    
    # 创建预处理器
    preprocessor = UnifiedSpectrumPreprocessor()
    
    if args.validate:
        # 验证现有频谱文件
        preprocessor.validate_spectrum_files(args.output_dir)
        
    elif args.incremental:
        # 增量处理
        preprocessor.incremental_process(args.audio_dir, args.labels_file, args.output_dir)
        
    else:
        # 全量处理
        result = preprocessor.process_audio_directory(
            args.audio_dir, 
            args.labels_file, 
            args.output_dir
        )
        
        if result is not None:
            print(f"\n🎉 预处理完成!")
            print(f"现在训练时会自动检测并使用预处理的频谱文件")

if __name__ == "__main__":
    main()