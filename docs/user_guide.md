# 用户指南

## 🎯 快速开始

### 第一次使用

#### 1. 准备数据
```bash
# 将音频文件放入data/audio目录
cp your_audio_files/*.wav data/audio/

# 运行数据设置脚本
python scripts/setup_data.py
```

#### 2. 编辑标签文件
编辑 `data/labels.csv` 文件，确保标签正确：
```csv
filename,label
Chinese_Number_01.wav,一
Chinese_Number_02.wav,二
Chinese_Number_03.wav,三
...
```

#### 3. 开始训练
```bash
# 小数据集训练（推荐新手）
python scripts/train.py --scale small

# 查看训练进度
tensorboard --logdir runs
```

#### 4. 测试识别
```bash
# 测试音频识别
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav

# 测试频谱识别（更快）
python scripts/inference.py --model checkpoints/best_model.pth --input data/features/Chinese_Number_01.npy
```

## 📊 训练指南

### 选择合适的训练规模

| 数据量 | 推荐规模 | 配置文件 | 训练时间 | 内存需求 |
|--------|----------|----------|----------|----------|
| 1-50样本 | `small` | `configs/small_dataset.yaml` | 5-15分钟 | 2GB |
| 50-200样本 | `medium` | `configs/medium_dataset.yaml` | 15-30分钟 | 4GB |
| 200-1000样本 | `large` | `configs/large_dataset.yaml` | 30-60分钟 | 8GB |
| 1000+样本 | `xlarge` | `configs/xlarge_dataset.yaml` | 1-3小时 | 16GB |

### 训练参数调优

#### 小数据集优化
```bash
# 使用小数据集配置
python scripts/train.py --config configs/small_dataset.yaml

# 关键参数：
# - batch_size: 1 (避免过拟合)
# - learning_rate: 1e-5 (小学习率)
# - hidden_dim: 64 (小模型)
# - dropout: 0.5 (高正则化)
```

#### 大数据集优化
```bash
# 使用大数据集配置
python scripts/train.py --config configs/large_dataset.yaml

# 关键参数：
# - batch_size: 4-8 (大批次)
# - learning_rate: 1e-4 (正常学习率)
# - hidden_dim: 256-512 (大模型)
# - dropout: 0.1-0.2 (低正则化)
```

### 训练监控

#### 使用TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir runs

# 在浏览器中打开 http://localhost:6006
```

#### 监控指标
- **训练损失**: 应该逐渐下降
- **验证损失**: 应该与训练损失同步下降
- **学习率**: 根据调度器变化
- **准确率**: 应该逐渐提升

### 训练技巧

#### 1. 数据质量
- 确保音频质量良好
- 标签准确无误
- 数据平衡（各类别样本数量相近）

#### 2. 过拟合处理
```bash
# 增加dropout
# 编辑配置文件
dropout: 0.5  # 从0.1增加到0.5

# 减少模型大小
hidden_dim: 64  # 从256减少到64
encoder_layers: 1  # 从4减少到1
decoder_layers: 1  # 从4减少到1
```

#### 3. 欠拟合处理
```bash
# 减少dropout
dropout: 0.1  # 从0.5减少到0.1

# 增加模型大小
hidden_dim: 256  # 从64增加到256
encoder_layers: 4  # 从1增加到4
decoder_layers: 4  # 从1增加到4

# 增加训练轮数
num_epochs: 200  # 从50增加到200
```

## 🎯 推理指南

### 输入模式选择

#### 音频输入模式
**适用场景**: 一般使用、开发测试
```bash
python scripts/inference.py --model checkpoints/best_model.pth --input audio.wav --mode audio
```

**特点**:
- 完整的音频处理流程
- 预处理时间: 2-3秒
- 推理时间: 0.3-0.5秒
- 总时间: 2.5-3.5秒

#### 频谱输入模式
**适用场景**: 高性能需求、批量处理、实时系统
```bash
python scripts/inference.py --model checkpoints/best_model.pth --input spectrogram.npy --mode spectrogram
```

**特点**:
- 跳过预处理步骤
- 预处理时间: 0秒
- 推理时间: 0.3-0.5秒
- 总时间: 0.3-0.5秒

#### 自动模式
**适用场景**: 不确定输入类型
```bash
python scripts/inference.py --model checkpoints/best_model.pth --input file --mode auto
```

**特点**:
- 根据文件扩展名自动判断
- `.wav`, `.mp3`, `.flac` → 音频模式
- `.npy`, `.npz` → 频谱模式

### 批量推理

#### 预处理音频文件
```bash
# 批量预处理
python scripts/batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv --output_dir data/features
```

#### 批量推理
```python
from wavespectra2text import DualInputSpeechRecognizer

recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')

# 批量音频推理
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = []

for audio_file in audio_files:
    result = recognizer.recognize_from_audio(audio_file)
    results.append(result)
    print(f"{audio_file}: {result['text']}")
```

### 性能优化

#### 1. 使用GPU
```bash
# 训练时使用GPU
python scripts/train.py --scale medium --device cuda

# 推理时使用GPU
python scripts/inference.py --model checkpoints/best_model.pth --input audio.wav --device cuda
```

#### 2. 预计算特征
```bash
# 预处理所有音频文件
python scripts/batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv

# 使用预计算模式训练
python scripts/train.py --scale medium --use_precomputed
```

#### 3. 批量处理
```python
# 使用批量推理
from wavespectra2text.core.inference import BatchInference

batch_inference = BatchInference(recognizer.inference_core)
results = batch_inference.infer_audio_batch(audio_files, show_progress=True)
```

## 🔧 数据管理

### 数据准备

#### 1. 音频文件要求
- **格式**: WAV, MP3, FLAC, M4A, AAC, OGG
- **采样率**: 推荐48kHz（系统会自动重采样）
- **时长**: 1-10秒（推荐2-5秒）
- **质量**: 清晰无噪声

#### 2. 标签文件格式
```csv
filename,label
audio_01.wav,一
audio_02.wav,二
audio_03.wav,三
...
```

#### 3. 数据验证
```bash
# 验证数据完整性
python scripts/setup_data.py

# 检查标签文件
python -c "
import pandas as pd
df = pd.read_csv('data/labels.csv')
print(f'总样本数: {len(df)}')
print(f'标签分布: {df[\"label\"].value_counts()}')
"
```

### 数据增强

#### 1. 音频增强
```python
import librosa
import numpy as np

def add_noise(audio, noise_factor=0.005):
    """添加噪声"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def change_speed(audio, sr, speed_factor=1.2):
    """改变语速"""
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def change_pitch(audio, sr, pitch_factor=2):
    """改变音调"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)
```

#### 2. 数据平衡
```python
import pandas as pd
from collections import Counter

def balance_dataset(df, target_count=100):
    """平衡数据集"""
    label_counts = Counter(df['label'])
    balanced_data = []
    
    for label, count in label_counts.items():
        if count < target_count:
            # 重复采样
            label_data = df[df['label'] == label]
            repeat_times = target_count // count + 1
            balanced_data.append(label_data.sample(n=target_count, replace=True))
        else:
            # 随机采样
            label_data = df[df['label'] == label]
            balanced_data.append(label_data.sample(n=target_count))
    
    return pd.concat(balanced_data, ignore_index=True)
```

### 数据同步

#### 自动更新系统
```bash
# 启动自动更新
python scripts/auto_update_system.py --mode monitor --interval 10

# 单次检查
python scripts/auto_update_system.py --mode check
```

**功能**:
- 监控音频文件变化
- 自动更新词汇表
- 同步预处理特征
- 更新特征索引

## 🛠️ 故障排除

### 常见问题

#### 1. 训练不收敛
**症状**: 损失不下降或准确率不提升

**解决方案**:
```bash
# 检查学习率
python -c "
from wavespectra2text.training.config import get_default_config
config = get_default_config('small')
print(f'学习率: {config[\"learning_rate\"]}')
"

# 调整学习率
# 编辑配置文件，将learning_rate从1e-5改为5e-5
```

#### 2. 内存不足
**症状**: CUDA out of memory 或系统卡死

**解决方案**:
```bash
# 减少批大小
python scripts/train.py --scale small  # 使用小规模配置

# 减少模型大小
# 编辑配置文件
batch_size: 1
hidden_dim: 32
```

#### 3. 推理速度慢
**症状**: 推理时间过长

**解决方案**:
```bash
# 使用频谱输入模式
python scripts/inference.py --model checkpoints/best_model.pth --input spectrogram.npy --mode spectrogram

# 使用GPU
python scripts/inference.py --model checkpoints/best_model.pth --input audio.wav --device cuda
```

#### 4. 识别准确率低
**症状**: 识别结果错误率高

**解决方案**:
```bash
# 检查数据质量
python scripts/setup_data.py

# 增加训练轮数
# 编辑配置文件
num_epochs: 200  # 从50增加到200

# 使用更大的模型
python scripts/train.py --scale large
```

### 调试技巧

#### 1. 启用详细日志
```bash
# 训练时显示详细信息
python scripts/train.py --scale small --verbose

# 推理时显示详细信息
python scripts/inference.py --model checkpoints/best_model.pth --input audio.wav --verbose
```

#### 2. 检查中间结果
```python
# 检查预处理结果
from wavespectra2text import PreprocessorFactory
preprocessor = PreprocessorFactory.create('spectrogram')
features = preprocessor.process('audio.wav')
print(f'特征形状: {features.shape}')
print(f'特征范围: [{features.min():.3f}, {features.max():.3f}]')

# 检查模型输出
from wavespectra2text import create_model, vocab
model = create_model()
# ... 加载模型
output = model(input_tensor)
print(f'输出形状: {output.shape}')
```

#### 3. 使用测试脚本
```bash
# 运行测试套件
python tests/run_tests.py

# 运行特定测试
python tests/test_core.py
python tests/test_data.py
python tests/test_training.py
python tests/test_inference.py
```

## 📈 性能优化

### 训练优化

#### 1. 使用预计算特征
```bash
# 预处理所有音频
python scripts/batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv

# 使用预计算模式训练（更快）
python scripts/train.py --scale medium --use_precomputed
```

#### 2. GPU加速
```bash
# 检查GPU可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 使用GPU训练
python scripts/train.py --scale medium --device cuda
```

#### 3. 多进程数据加载
```bash
# 编辑配置文件
num_workers: 4  # 根据CPU核心数调整
pin_memory: true
```

### 推理优化

#### 1. 模型量化
```python
import torch

# 加载模型
model = torch.load('checkpoints/best_model.pth')

# 量化模型
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'checkpoints/quantized_model.pth')
```

#### 2. 批处理推理
```python
from wavespectra2text.core.inference import BatchInference

# 批量推理
batch_inference = BatchInference(inference_core)
results = batch_inference.infer_audio_batch(
    audio_files, 
    method='beam',
    beam_size=3,
    show_progress=True
)
```

#### 3. 缓存机制
```python
# 启用特征缓存
from wavespectra2text.data.preprocessing import OfflinePreprocessor

preprocessor = PreprocessorFactory.create('spectrogram')
offline_processor = OfflinePreprocessor(preprocessor, cache_dir='cache/features')

# 处理文件（会自动缓存）
features = offline_processor.process_file('audio.wav')
```

## 🎓 最佳实践

### 1. 数据准备
- 使用高质量的音频文件
- 确保标签准确无误
- 保持数据平衡
- 定期验证数据完整性

### 2. 模型训练
- 从小规模开始测试
- 监控训练过程
- 使用验证集评估
- 保存最佳模型

### 3. 推理部署
- 使用频谱输入模式提升性能
- 批量处理提高效率
- 启用GPU加速
- 实现错误处理机制

### 4. 系统维护
- 定期更新依赖包
- 监控系统性能
- 备份重要数据
- 记录配置变更

## 📚 进阶使用

### 自定义预处理器
```python
from wavespectra2text.data.preprocessing import AudioPreprocessor
import librosa
import numpy as np

class CustomPreprocessor(AudioPreprocessor):
    def process(self, audio_path):
        # 自定义预处理逻辑
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 添加自定义特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma(y=audio, sr=sr)
        
        # 组合特征
        features = np.concatenate([mfcc.T, chroma.T], axis=1)
        
        return features
    
    def get_feature_shape(self):
        return (200, 25)  # (time, features)

# 注册自定义预处理器
from wavespectra2text.data.preprocessing import PreprocessorFactory
PreprocessorFactory.register('custom', CustomPreprocessor)
```

### 自定义训练器
```python
from wavespectra2text.training.trainer import BaseTrainer
import torch.optim as optim

class CustomTrainer(BaseTrainer):
    def _create_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.98)
        )
    
    def _create_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs']
        )
```

### 集成外部系统
```python
# 与外部音频处理系统集成
def external_audio_processing(audio_path):
    """外部系统的音频预处理"""
    from wavespectra2text import PreprocessorFactory
    
    processor = PreprocessorFactory.create('spectrogram')
    return processor.process(audio_path)

# 保存预处理结果
import numpy as np
spectrogram = external_audio_processing('audio.wav')
np.save('external_spectrogram.npy', spectrogram)

# 使用预处理结果进行推理
from wavespectra2text import DualInputSpeechRecognizer
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')
result = recognizer.recognize_from_spectrogram('external_spectrogram.npy')
```
