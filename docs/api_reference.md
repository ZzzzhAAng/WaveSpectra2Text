# API 参考文档

## 📚 核心模块 API

### 模型创建

#### `create_model()`
创建Transformer序列到序列模型。

```python
from wavespectra2text import create_model

model = create_model(
    vocab_size=14,           # 词汇表大小
    input_dim=513,           # 输入特征维度
    hidden_dim=256,          # 隐藏层维度
    encoder_layers=4,        # 编码器层数
    decoder_layers=4,        # 解码器层数
    dropout=0.1,             # Dropout比率
    device='cpu'             # 计算设备
)
```

**参数:**
- `vocab_size` (int): 词汇表大小，默认使用vocab.vocab_size
- `input_dim` (int): 输入频谱特征维度，默认513
- `hidden_dim` (int): Transformer隐藏层维度，默认256
- `encoder_layers` (int): 编码器层数，默认4
- `decoder_layers` (int): 解码器层数，默认4
- `dropout` (float): Dropout比率，默认0.1
- `device` (str): 计算设备，默认'cpu'

**返回:**
- `Seq2SeqModel`: 配置好的模型实例

### 词汇表管理

#### `vocab`
全局词汇表实例，包含中文数字1-10和特殊符号。

```python
from wavespectra2text import vocab

# 获取词汇表大小
size = vocab.vocab_size  # 14

# 编码文本为索引
indices = vocab.encode("一")  # [1, 2] (SOS + 一 + EOS)

# 解码索引为文本
text = vocab.decode([1, 2, 3])  # "一"

# 获取特殊符号索引
pad_idx = vocab.get_padding_idx()  # 0
sos_idx = vocab.get_sos_idx()      # 1
eos_idx = vocab.get_eos_idx()      # 2
unk_idx = vocab.get_unk_idx()      # 3
```

### 推理核心

#### `InferenceCore`
统一的推理核心类，提供模型加载和推理功能。

```python
from wavespectra2text import InferenceCore

# 创建推理核心
core = InferenceCore('checkpoints/best_model.pth', device='cpu')

# 从音频文件推理
result = core.infer_from_audio('audio.wav', method='auto')

# 从频谱特征推理
result = core.infer_from_spectrogram(spectrogram_array, method='beam')

# 获取模型信息
info = core.get_model_info()
```

**方法:**
- `infer_from_audio(audio_path, method='auto', beam_size=3)`: 从音频文件推理
- `infer_from_spectrogram(spectrogram, method='auto', beam_size=3)`: 从频谱特征推理
- `get_model_info()`: 获取模型信息

## 🎯 推理模块 API

### 双输入识别器

#### `DualInputSpeechRecognizer`
支持音频和频谱两种输入模式的语音识别器。

```python
from wavespectra2text import DualInputSpeechRecognizer

# 创建识别器
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth', device='cpu')

# 音频识别
result = recognizer.recognize_from_audio('audio.wav', show_details=True)

# 频谱识别
result = recognizer.recognize_from_spectrogram('spectrogram.npy', show_details=True)

# 内存数组识别
result = recognizer.recognize_from_spectrogram_array(spectrogram_array)

# 自动模式
result = recognizer.auto_recognize('input_file')
```

**方法:**
- `recognize_from_audio(audio_path, show_details=True)`: 从音频文件识别
- `recognize_from_spectrogram(spectrogram_path, show_details=True)`: 从频谱文件识别
- `recognize_from_spectrogram_array(spectrogram_array, show_details=True)`: 从内存数组识别
- `auto_recognize(input_path, show_details=True)`: 自动识别输入类型

**返回结果格式:**
```python
{
    'text': '识别结果文本',
    'success': True,
    'processing_time': {
        'preprocessing': 2.5,  # 预处理时间（秒）
        'inference': 0.3,      # 推理时间（秒）
        'total': 2.8           # 总时间（秒）
    },
    'input_type': 'audio_file',
    'spectrogram_shape': (200, 513),
    'method': 'beam_search',
    'mode': 'audio',
    'error': None
}
```

## 🏋️ 训练模块 API

### 训练器创建

#### `create_trainer()`
创建训练器实例。

```python
from wavespectra2text import create_trainer

trainer = create_trainer(
    trainer_type='improved',  # 'simple', 'improved', 'large'
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    config=config
)

# 开始训练
trainer.train(num_epochs=50, resume_path=None)
```

**训练器类型:**
- `'simple'`: 小数据集训练器，适合1-50样本
- `'improved'`: 中等数据集训练器，适合50-200样本
- `'large'`: 大数据集训练器，适合200+样本

### 配置管理

#### `get_default_config()`
获取不同规模的默认配置。

```python
from wavespectra2text.training.config import get_default_config

# 获取不同规模配置
small_config = get_default_config('small')
medium_config = get_default_config('medium')
large_config = get_default_config('large')
xlarge_config = get_default_config('xlarge')
```

**配置规模:**
- `'small'`: 小数据集配置 (batch_size=1, hidden_dim=64)
- `'medium'`: 中等数据集配置 (batch_size=2, hidden_dim=128)
- `'large'`: 大数据集配置 (batch_size=4, hidden_dim=256)
- `'xlarge'`: 超大数据集配置 (batch_size=8, hidden_dim=512)

## 📊 数据处理 API

### 数据集创建

#### `AudioDataset`
支持实时计算和预计算两种模式的音频数据集。

```python
from wavespectra2text import AudioDataset

# 实时计算模式
dataset = AudioDataset(
    labels_file='data/labels.csv',
    audio_dir='data/audio',
    mode='realtime'
)

# 预计算模式
dataset = AudioDataset(
    labels_file='data/labels.csv',
    precomputed_dir='data/features',
    mode='precomputed'
)
```

### 预处理器

#### `PreprocessorFactory`
预处理器工厂，支持多种预处理策略。

```python
from wavespectra2text import PreprocessorFactory

# 创建STFT频谱预处理器
preprocessor = PreprocessorFactory.create(
    'spectrogram',
    sample_rate=48000,
    n_fft=1024,
    hop_length=512,
    max_length=200
)

# 创建Mel频谱预处理器
mel_preprocessor = PreprocessorFactory.create(
    'mel_spectrogram',
    sample_rate=48000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    max_length=200
)

# 处理音频文件
features = preprocessor.process('audio.wav')

# 获取特征形状
shape = preprocessor.get_feature_shape()
```

**可用预处理器:**
- `'spectrogram'`: STFT频谱预处理器
- `'mel_spectrogram'`: Mel频谱预处理器

### 数据工具

#### `AudioProcessor`
音频处理工具类。

```python
from wavespectra2text import AudioProcessor

processor = AudioProcessor(
    sample_rate=48000,
    n_fft=1024,
    hop_length=512,
    max_length=200
)

# 提取频谱特征
spectrogram = processor.extract_spectrogram('audio.wav')
```

## 🔧 工具模块 API

### 文件工具

#### `FileUtils`
文件操作工具类。

```python
from wavespectra2text import FileUtils

# 检查文件存在性
exists = FileUtils.file_exists('path/to/file')

# 创建目录
FileUtils.create_dir('path/to/dir')

# 获取文件扩展名
ext = FileUtils.get_file_extension('file.wav')
```

### 标签管理

#### `LabelManager`
标签管理工具类。

```python
from wavespectra2text import LabelManager

manager = LabelManager('data/labels.csv')

# 获取所有标签
labels = manager.get_all_labels()

# 验证标签文件
is_valid = manager.validate_labels()

# 更新标签
manager.update_labels(new_labels)
```

## 📝 使用示例

### 完整训练流程

```python
from wavespectra2text import (
    create_model, vocab, create_trainer, AudioDataset,
    get_default_config
)
from torch.utils.data import DataLoader

# 1. 加载配置
config = get_default_config('medium')

# 2. 创建数据集
dataset = AudioDataset(
    labels_file='data/labels.csv',
    audio_dir='data/audio',
    mode='realtime'
)

# 3. 创建数据加载器
train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# 4. 创建模型
model = create_model(
    vocab_size=vocab.vocab_size,
    hidden_dim=config['hidden_dim'],
    encoder_layers=config['encoder_layers'],
    decoder_layers=config['decoder_layers'],
    dropout=config['dropout']
)

# 5. 创建训练器
trainer = create_trainer('improved', model, train_loader, val_loader, 'cpu', config)

# 6. 开始训练
trainer.train(config['num_epochs'])
```

### 完整推理流程

```python
from wavespectra2text import DualInputSpeechRecognizer

# 1. 创建识别器
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')

# 2. 音频识别
audio_result = recognizer.recognize_from_audio('data/audio/test.wav')
print(f"音频识别结果: {audio_result['text']}")

# 3. 频谱识别
spectrogram_result = recognizer.recognize_from_spectrogram('data/features/test.npy')
print(f"频谱识别结果: {spectrogram_result['text']}")

# 4. 自动模式
auto_result = recognizer.auto_recognize('data/audio/test.wav')
print(f"自动识别结果: {auto_result['text']}")
```

## 🚨 错误处理

### 常见异常

- `FileNotFoundError`: 文件不存在
- `ValueError`: 参数值错误
- `RuntimeError`: 运行时错误
- `ImportError`: 模块导入错误

### 错误处理示例

```python
try:
    recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')
    result = recognizer.recognize_from_audio('audio.wav')
    
    if result['success']:
        print(f"识别成功: {result['text']}")
    else:
        print(f"识别失败: {result['error']}")
        
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```
