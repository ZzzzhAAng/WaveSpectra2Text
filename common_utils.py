#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具模块
统一处理项目中的重复功能，减少代码冗余
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import csv

# 可选依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class AudioProcessor:
    """统一的音频处理器 - 解决音频预处理代码冗余"""
    
    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024, 
                 hop_length: int = 512, max_length: int = 200):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            max_length: 最大序列长度
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        统一的音频加载方法
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            audio: 音频数据
            sr: 采样率
        """
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"加载音频文件失败 {audio_path}: {e}")
    
    def extract_spectrogram(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        统一的频谱提取方法 - 替代各个文件中的重复实现
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            spectrogram: 频谱特征 (max_length, freq_bins)
        """
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        
        # 提取STFT频谱
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 转换为对数刻度
        log_magnitude = np.log1p(magnitude)
        
        # 转置使时间维度在前
        spectrogram = log_magnitude.T  # (time_steps, freq_bins)
        
        # 标准化长度
        spectrogram = self._normalize_length(spectrogram)
        
        return spectrogram.astype(np.float32)
    
    def _normalize_length(self, spectrogram: np.ndarray) -> np.ndarray:
        """标准化序列长度"""
        if len(spectrogram) > self.max_length:
            return spectrogram[:self.max_length]
        else:
            pad_length = self.max_length - len(spectrogram)
            return np.pad(spectrogram, ((0, pad_length), (0, 0)), mode='constant')
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> Dict:
        """
        获取音频文件信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            info: 音频信息字典
        """
        try:
            audio, sr = self.load_audio(audio_path)
            duration = len(audio) / sr
            
            return {
                'path': str(audio_path),
                'sample_rate': sr,
                'duration': duration,
                'samples': len(audio),
                'channels': 1 if audio.ndim == 1 else audio.shape[1]
            }
        except Exception as e:
            return {
                'path': str(audio_path),
                'error': str(e)
            }


class LabelManager:
    """统一的标签管理器 - 解决标签文件创建的代码冗余"""
    
    @staticmethod
    def scan_audio_files(audio_dir: Union[str, Path]) -> List[Path]:
        """
        扫描音频文件
        
        Args:
            audio_dir: 音频目录
            
        Returns:
            audio_files: 音频文件路径列表
        """
        audio_dir = Path(audio_dir)
        
        if not audio_dir.exists():
            return []
        
        # 支持的音频格式
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(audio_dir.glob(f'*{ext}')))
            audio_files.extend(list(audio_dir.glob(f'*{ext.upper()}')))
        
        # 按文件名排序
        audio_files.sort(key=lambda x: x.name)
        
        return audio_files
    
    @staticmethod
    def create_labels_template(audio_dir: Union[str, Path], 
                              output_file: Union[str, Path] = 'data/labels.csv',
                              auto_labels: bool = True) -> bool:
        """
        统一的标签模板创建方法 - 合并setup_data.py和data_utils.py中的重复功能
        
        Args:
            audio_dir: 音频文件目录
            output_file: 输出标签文件路径
            auto_labels: 是否自动推断标签
            
        Returns:
            success: 是否创建成功
        """
        # 扫描音频文件
        audio_files = LabelManager.scan_audio_files(audio_dir)
        
        if not audio_files:
            print(f"在 {audio_dir} 目录中没有找到音频文件")
            return False
        
        print(f"找到 {len(audio_files)} 个音频文件:")
        for i, file in enumerate(audio_files[:10]):  # 只显示前10个
            print(f"  {i + 1}. {file.name}")
        if len(audio_files) > 10:
            print(f"  ... 还有 {len(audio_files) - 10} 个文件")
        
        # 创建标签数据
        labels_data = []
        chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        
        for i, audio_file in enumerate(audio_files):
            filename = audio_file.name
            
            if auto_labels:
                # 尝试从文件名推断标签
                label = LabelManager._infer_label_from_filename(filename, chinese_numbers, i)
            else:
                # 使用占位符
                label = "待标注"
            
            labels_data.append({
                'filename': filename,
                'label': label
            })
        
        # 保存到CSV文件
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 优先使用pandas，如果不可用则使用csv模块
            if HAS_PANDAS:
                df = pd.DataFrame(labels_data)
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                # 使用csv模块
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
                    writer.writeheader()
                    writer.writerows(labels_data)
            
            print(f"\n✅ 标签模板已创建: {output_file}")
            print("请检查并修改标签文件中的标签，确保它们与音频内容匹配")
            
            return True
            
        except Exception as e:
            print(f"❌ 创建标签文件失败: {e}")
            return False
    
    @staticmethod
    def _infer_label_from_filename(filename: str, chinese_numbers: List[str], index: int) -> str:
        """从文件名推断标签"""
        # 尝试从文件名中的数字映射到中文
        for j, num in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]):
            if num in filename:
                if j < len(chinese_numbers):
                    return chinese_numbers[j]
                break
        
        # 如果没有找到匹配，使用索引
        if index < len(chinese_numbers):
            return chinese_numbers[index]
        else:
            return "未知"
    
    @staticmethod
    def validate_labels_file(labels_file: Union[str, Path], 
                            audio_dir: Union[str, Path]) -> bool:
        """
        验证标签文件
        
        Args:
            labels_file: 标签文件路径
            audio_dir: 音频目录
            
        Returns:
            valid: 是否验证通过
        """
        labels_file = Path(labels_file)
        audio_dir = Path(audio_dir)
        
        if not labels_file.exists():
            print(f"标签文件不存在: {labels_file}")
            return False
        
        try:
            if HAS_PANDAS:
                df = pd.read_csv(labels_file)
            else:
                # 使用csv模块读取
                labels_data = LabelManager.read_labels_csv(labels_file)
                if not labels_data:
                    print(f"读取标签文件失败: 文件为空或格式错误")
                    return False
                # 模拟pandas DataFrame的接口
                class SimpleDF:
                    def __init__(self, data):
                        self.data = data
                        self.columns = list(data[0].keys()) if data else []
                    
                    def iterrows(self):
                        for i, row in enumerate(self.data):
                            yield i, type('Row', (), row)()
                    
                    def __len__(self):
                        return len(self.data)
                
                df = SimpleDF(labels_data)
        except Exception as e:
            print(f"读取标签文件失败: {e}")
            return False
        
        # 检查必要的列
        required_columns = ['filename', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"标签文件缺少列: {missing_columns}")
            return False
        
        # 检查音频文件是否存在
        missing_files = []
        existing_files = []
        
        for _, row in df.iterrows():
            audio_path = audio_dir / row['filename']
            if audio_path.exists():
                existing_files.append(row['filename'])
            else:
                missing_files.append(row['filename'])
        
        print(f"标签文件验证结果:")
        print(f"  总条目数: {len(df)}")
        print(f"  存在的音频文件: {len(existing_files)}")
        print(f"  缺失的音频文件: {len(missing_files)}")
        
        if missing_files:
            print(f"  缺失文件列表: {missing_files}")
            return False
        
        # 检查标签
        valid_labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        invalid_labels = []
        
        for _, row in df.iterrows():
            if row['label'] not in valid_labels and row['label'] not in ['未知', '待标注']:
                invalid_labels.append((row['filename'], row['label']))
        
        if invalid_labels:
            print(f"  无效标签: {invalid_labels}")
            print(f"  有效标签应为: {valid_labels}")
            return False
        
        print("✅ 标签文件验证通过")
        return True
    
    @staticmethod
    def read_labels_csv(labels_file: Union[str, Path]) -> List[Dict[str, str]]:
        """
        读取标签CSV文件 - 不依赖pandas的版本
        
        Args:
            labels_file: 标签文件路径
            
        Returns:
            labels: 标签数据列表
        """
        labels = []
        labels_file = Path(labels_file)
        
        if not labels_file.exists():
            return labels
        
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels.append(row)
        except Exception as e:
            print(f"❌ 读取标签文件失败: {e}")
        
        return labels


class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def ensure_dir(dir_path: Union[str, Path]) -> Path:
        """确保目录存在"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], suffix: str = '.backup') -> Path:
        """备份文件"""
        file_path = Path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + suffix)
        
        if file_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
            
        return backup_path
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """获取文件哈希值"""
        import hashlib
        
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


# 创建全局实例，方便其他模块使用
default_audio_processor = AudioProcessor()
label_manager = LabelManager()
file_utils = FileUtils()


# 向后兼容的便捷函数
def create_labels_template(audio_files, output_file='data/labels.csv'):
    """向后兼容setup_data.py中的函数"""
    if isinstance(audio_files, (str, Path)):
        # 如果传入的是目录路径
        return LabelManager.create_labels_template(audio_files, output_file)
    else:
        # 如果传入的是文件列表，转换为临时目录处理
        print("⚠️  建议使用 LabelManager.create_labels_template(audio_dir, output_file)")
        return LabelManager.create_labels_template('data/audio', output_file)


def create_labels_file_if_not_exists(labels_file='data/labels.csv'):
    """向后兼容data_utils.py中的函数"""
    if os.path.exists(labels_file):
        print(f"标签文件已存在: {labels_file}")
        return
    
    return LabelManager.create_labels_template('data/audio', labels_file, auto_labels=True)


def extract_spectrogram(audio_path, sample_rate=48000, n_fft=1024, hop_length=512, max_length=200):
    """向后兼容的频谱提取函数"""
    processor = AudioProcessor(sample_rate, n_fft, hop_length, max_length)
    return processor.extract_spectrogram(audio_path)


if __name__ == "__main__":
    # 测试代码
    print("🧪 通用工具模块测试")
    print("=" * 50)
    
    # 测试音频处理器
    processor = AudioProcessor()
    print(f"音频处理器配置: 采样率={processor.sample_rate}, FFT={processor.n_fft}")
    
    # 测试标签管理器
    audio_files = LabelManager.scan_audio_files('data/audio')
    print(f"扫描到 {len(audio_files)} 个音频文件")
    
    # 测试标签文件验证
    if Path('data/labels.csv').exists():
        is_valid = LabelManager.validate_labels_file('data/labels.csv', 'data/audio')
        print(f"标签文件验证: {'✅' if is_valid else '❌'}")
    
    print("✅ 通用工具模块测试完成")