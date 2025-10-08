# WaveSpectra2Text
# 双输入语音识别系统 - 音频与频谱特征

本项目实现了一个支持双输入模式的语音识别系统。系统支持两种输入方式：
1. **原始音频输入**：完整的音频处理流程（音频 → 频谱提取 → 模型推理 → 文本）
2. **频谱特征输入**：跳过预处理的快速推理（预处理频谱 → 模型推理 → 文本）

现在以中文数字1～10为例，展示了从频谱特征到文本的端到端识别能力。

## ✨ 主要特性

- 🎯 **双输入模式**: 支持原始音频和预处理频谱两种输入方式
- 🚀 **高性能推理**: 频谱输入模式可提升推理速度5-10倍
- 🧠 **Transformer架构**: 基于编码器-解码器的现代深度学习模型
- 📊 **统一训练系统**: 支持small/medium/large/xlarge四种规模训练
- 🔧 **模块化设计**: 低耦合、高扩展性的代码架构
- 📈 **智能推理**: 贪婪解码 + 束搜索 + 智能回退机制
- 🎵 **专业音频处理**: 基于librosa的高质量频谱提取

## 🏗️ 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                    WaveSpectra2Text 系统架构                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   音频输入    │    │  频谱输入    │    │   文本输出    │      │
│  │  (.wav)     │    │  (.npy)     │    │  (中文数字)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                   │                   ▲            │
│         ▼                   │                   │            │
│  ┌─────────────┐            │                   │            │
│  │ 音频预处理   │            │                   │            │
│  │ (librosa)  │            │                   │            │
│  └─────────────┘            │                   │            │
│         │                   │                   │            │
│         ▼                   ▼                   │            │
│  ┌─────────────────────────────────────────────┐            │
│  │            Transformer 模型                  │            │
│  │  ┌─────────────┐    ┌─────────────┐          │            │
│  │  │  编码器      │    │   解码器     │          │            │
│  │  │ (Encoder)   │    │ (Decoder)   │          │            │
│  │  └─────────────┘    └─────────────┘          │            │
│  └─────────────────────────────────────────────┘            │
│         │                   │                   │            │
│         ▼                   ▼                   │            │
│  ┌─────────────┐    ┌─────────────┐            │            │
│  │ 频谱特征     │    │ 注意力机制    │            │            │
│  │ (200×513)   │    │ (Multi-Head) │            │            │
│  └─────────────┘    └─────────────┘            │            │
│         │                   │                   │            │
│         └───────────────────┼───────────────────┘            │
│                             ▼                                │
│  ┌─────────────────────────────────────────────┐            │
│  │              推理策略                        │            │
│  │  ┌─────────────┐    ┌─────────────┐          │            │
│  │  │  贪婪解码    │    │   束搜索     │          │            │
│  │  │ (Greedy)    │    │ (Beam)     │          │            │
│  │  └─────────────┘    └─────────────┘          │            │
│  └─────────────────────────────────────────────┘            │
│                             │                                │
│                             ▼                                │
│  ┌─────────────────────────────────────────────┐            │
│  │              词汇表解码                      │            │
│  │         (中文数字1-10 + 特殊符号)              │            │
│  └─────────────────────────────────────────────┘            │
│                             │                                │
│                             ▼                                │
│  ┌─────────────────────────────────────────────┐            │
│  │              后处理                         │            │
│  │         (智能回退机制)                        │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
WaveSpectra2Text/
├── 📁 src/                           # 源代码目录
│   └── 📁 wavespectra2text/          # 主包
│       ├── 📄 __init__.py            # 包初始化
│       ├── 📁 core/                  # 核心模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 model.py          # Transformer模型
│       │   ├── 📄 vocab.py          # 词汇表管理
│       │   └── 📄 inference.py      # 推理核心
│       ├── 📁 data/                  # 数据处理模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 preprocessing.py  # 音频预处理
│       │   ├── 📄 dataset.py        # 数据集类
│       │   └── 📄 utils.py          # 数据工具
│       ├── 📁 training/              # 训练模块
│       │   ├── 📄 __init__.py
│       │   ├── 📄 trainer.py        # 训练基类
│       │   ├── 📄 config.py         # 配置管理
│       │   └── 📄 callbacks.py     # 训练回调
│       ├── 📁 inference/             # 推理模块
│       │   ├── 📄 __init__.py
│       │   └── 📄 recognizer.py     # 识别器
│       └── 📁 utils/                 # 工具模块
│           ├── 📄 __init__.py
│           └── (已简化，只保留核心工具)
│
├── 📁 scripts/                       # 脚本目录
│   ├── 📄 train.py                  # 统一训练脚本
│   ├── 📄 inference.py             # 统一推理脚本
│   ├── 📄 batch_preprocess.py      # 批量预处理脚本
│   ├── 📄 setup_data.py            # 数据设置脚本
│   ├── 📄 sync_data.py             # 数据同步脚本
│   └── 📄 auto_update_system.py    # 自动更新系统
│
├── 📁 configs/                      # 配置文件目录
│   ├── 📄 default.yaml             # 默认配置
│   ├── 📄 small_dataset.yaml       # 小数据集配置
│   ├── 📄 medium_dataset.yaml      # 中等数据集配置
│   ├── 📄 large_dataset.yaml       # 大数据集配置
│   └── 📄 xlarge_dataset.yaml      # 超大数据集配置
│
├── 📁 data/                         # 数据目录
│   ├── 📁 audio/                     # 原始音频文件
│   │   ├── Chinese_Number_01.wav     # 中文数字音频样本
│   │   └── ...
│   ├── 📁 features/                 # 特征文件
│   │   ├── Chinese_Number_01.npy     # 预处理频谱特征
│   │   ├── spectrum_index.csv       # 特征索引文件
│   │   └── ...
│   └── 📄 labels.csv                 # 标签文件
│
├── 📁 runs/                         # 训练运行记录
│   └── 📁 small_dataset/            # TensorBoard日志
│
├── 📁 checkpoints/                  # 模型检查点
│   ├── 📄 best_model.pth           # 最佳模型
│   └── 📄 checkpoint_epoch_*.pth   # 定期检查点
│
├── 📁 tests/                        # 测试目录
│   ├── 📄 test_core.py             # 核心模块测试
│   ├── 📄 test_data.py             # 数据模块测试
│   ├── 📄 test_training.py         # 训练模块测试
│   └── 📄 test_inference.py        # 推理模块测试
│
├── 📁 examples/                     # 示例目录
│   └── 📄 basic_usage.py           # 基本使用示例
│
├── 📁 docs/                         # 文档目录
│   └── 📄 training_scales.md       # 训练规模说明
│
├── 📄 pyproject.toml              # 项目配置
├── 📄 requirements.txt             # 依赖列表
└── 📄 README.md                    # 项目说明
```

## 🚀 快速开始

### 安装方式

#### 1. 开发安装（推荐）
```bash
# 克隆项目
git clone https://github.com/wavespectra2text/wavespectra2text.git
cd wavespectra2text

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

#### 2. 从源码安装
```bash
pip install .
```

#### 3. 仅安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. 准备数据
```bash
# 自动扫描音频文件并创建标签模板
python scripts/setup_data.py
```

#### 2. 训练模型
```bash
# 使用统一训练脚本 - 支持四种规模
python scripts/train.py --scale small    # 小数据集
python scripts/train.py --scale medium   # 中等数据集
python scripts/train.py --scale large    # 大数据集
python scripts/train.py --scale xlarge   # 超大数据集

# 使用配置文件
python scripts/train.py --config configs/small_dataset.yaml

# 从检查点恢复训练
python scripts/train.py --scale medium --resume checkpoints/checkpoint_epoch_50.pth
```

#### 3. 推理识别
```bash
# 使用统一推理脚本
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav

# 自动模式 - 根据文件扩展名自动选择输入模式
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav --mode auto

# 指定输入模式
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav --mode audio
python scripts/inference.py --model checkpoints/best_model.pth --input data/features/Chinese_Number_01.npy --mode spectrogram

# 查看性能对比和演示代码
python scripts/inference.py --compare
python scripts/inference.py --demo
```

#### 4. 编程接口
```python
from wavespectra2text import DualInputSpeechRecognizer, create_model, vocab

# 创建识别器
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')

# 音频识别
result = recognizer.recognize_from_audio('data/audio/Chinese_Number_01.wav')
print(f"识别结果: {result['text']}")

# 频谱识别（更快）
result = recognizer.recognize_from_spectrogram('data/features/Chinese_Number_01.npy')
print(f"识别结果: {result['text']}")

# 自动模式
result = recognizer.auto_recognize('data/audio/Chinese_Number_01.wav')  # 音频文件
result = recognizer.auto_recognize('data/features/Chinese_Number_01.npy')  # 频谱文件
```

## 数据准备

1. 将音频文件存放在 `data/audio/` 目录下
2. 创建 `data/labels.csv` 文件，格式如下：

```csv
filename,label
audio_1.wav,label_1
audio_2.wav,label_2
audio_3.wav,label_3
...
```

## 双输入系统架构

### 🎯 推理模式对比

| 特征 | 音频输入模式 | 频谱输入模式 |
|------|-------------|-------------|
| **输入类型** | 原始音频文件 (.wav, .mp3等) | 预处理频谱文件 (.npy) |
| **预处理时间** | 2-3秒 | 0秒（跳过） |
| **推理时间** | 0.3-0.5秒 | 0.3-0.5秒 |
| **总耗时** | 2.5-3.5秒 | 0.3-0.5秒 |
| **内存占用** | 中等 | 低 |
| **适用场景** | 一般使用、开发测试 | 高性能需求、批量处理、实时系统 |

### 🧠 模型架构

- **频谱编码器**: 基于Transformer的编码器，将STFT频谱特征（200×513）编码为隐藏表示
- **注意力解码器**: 基于Transformer的解码器，使用多头注意力机制解码为文本序列
- **词汇表**: 支持中文数字1～10 + 特殊符号（PAD, SOS, EOS, UNK），共14个token
- **推理策略**: 支持贪婪解码和束搜索，智能回退机制确保鲁棒性

## 🏋️ 训练

### 使用统一训练脚本（推荐）

```bash
# 四种规模训练
python scripts/train.py --scale small    # 小数据集 (1-50样本)
python scripts/train.py --scale medium   # 中等数据集 (50-200样本)
python scripts/train.py --scale large    # 大数据集 (200-1000样本)
python scripts/train.py --scale xlarge   # 超大数据集 (1000+样本)

# 从检查点恢复训练
python scripts/train.py --scale medium --resume checkpoints/checkpoint_epoch_50.pth
```

### 使用配置文件

```bash
# 使用YAML配置文件
python scripts/train.py --config configs/small_dataset.yaml
python scripts/train.py --config configs/medium_dataset.yaml
python scripts/train.py --config configs/large_dataset.yaml
python scripts/train.py --config configs/xlarge_dataset.yaml
```

### 编程接口训练

```python
from wavespectra2text import create_model, vocab, create_trainer
from wavespectra2text.training.config import get_default_config
from wavespectra2text.data.dataset import AudioDataset

# 加载配置
config = get_default_config('medium')

# 创建数据集
dataset = AudioDataset(
    labels_file='data/labels.csv',
    audio_dir='data/audio',
    mode='realtime'
)

# 创建模型
model = create_model(
    vocab_size=vocab.vocab_size,
    hidden_dim=config['hidden_dim'],
    encoder_layers=config['encoder_layers'],
    decoder_layers=config['decoder_layers'],
    dropout=config['dropout']
)

# 创建训练器
trainer = create_trainer('improved', model, train_loader, val_loader, device, config)

# 开始训练
trainer.train(config['num_epochs'])
```

## 🎯 推理

### 使用统一推理脚本（推荐）

```bash
# 自动模式 - 根据文件扩展名自动选择输入模式
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav

# 指定输入模式
python scripts/inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav --mode audio
python scripts/inference.py --model checkpoints/best_model.pth --input data/features/Chinese_Number_01.npy --mode spectrogram

# 查看性能对比和演示代码
python scripts/inference.py --compare
python scripts/inference.py --demo
```

### 编程接口推理

```python
from wavespectra2text import DualInputSpeechRecognizer

# 创建识别器
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')

# 音频识别
result = recognizer.recognize_from_audio('data/audio/Chinese_Number_01.wav')
print(f"识别结果: {result['text']}")

# 频谱识别（更快）
result = recognizer.recognize_from_spectrogram('data/features/Chinese_Number_01.npy')
print(f"识别结果: {result['text']}")

# 自动模式
result = recognizer.auto_recognize('data/audio/Chinese_Number_01.wav')  # 音频文件
result = recognizer.auto_recognize('data/features/Chinese_Number_01.npy')  # 频谱文件
```

## 🔧 数据预处理

### 快速开始 - 数据设置
```bash
# 自动扫描音频文件并创建标签模板
python scripts/setup_data.py
```

### 批量预处理（推荐用于大数据集）
```bash
# 批量预处理音频文件为频谱特征，大幅提升训练速度
python scripts/batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv --output_dir data/features

# 使用不同的预处理器
python scripts/batch_preprocess.py --preprocessor mel_spectrogram --n_mels 128

# 强制重新计算所有特征
python scripts/batch_preprocess.py --force_recompute
```

### 数据加载模式
```python
from wavespectra2text.data.dataset import AudioDataset, FlexibleDataLoader

# 实时模式：每次训练时计算特征
dataset = AudioDataset(
    labels_file='data/labels.csv',
    audio_dir='data/audio',
    mode='realtime'
)
dataloader = FlexibleDataLoader.create_dataloader(dataset, batch_size=4)

# 预计算模式：使用预处理好的特征（最快）
dataset = AudioDataset(
    labels_file='data/labels.csv',
    precomputed_dir='data/features',
    mode='precomputed'
)
dataloader = FlexibleDataLoader.create_dataloader(dataset, batch_size=4)
```

## ⚙️ 配置参数

### 训练规模配置

项目支持四种训练规模，每种规模都有对应的配置文件：

| 规模 | 样本数 | 配置文件 | Batch Size | Learning Rate | Hidden Dim | Layers |
|------|--------|----------|------------|---------------|------------|--------|
| **small** | 1-50 | `configs/small_dataset.yaml` | 1 | 1e-5 | 64 | 1+1 |
| **medium** | 50-200 | `configs/medium_dataset.yaml` | 2 | 5e-5 | 128 | 2+2 |
| **large** | 200-1000 | `configs/large_dataset.yaml` | 4 | 1e-4 | 256 | 4+4 |
| **xlarge** | 1000+ | `configs/xlarge_dataset.yaml` | 8 | 2e-4 | 512 | 6+6 |

### 音频预处理参数
- `sample_rate`: 采样率 (默认: 48000Hz)
- `n_fft`: FFT窗口大小 (默认: 1024)
- `hop_length`: 跳跃长度 (默认: 512)
- `max_length`: 最大序列长度 (默认: 200)

## 🚀 技术特点

### 双输入架构优势
1. **灵活性**: 支持原始音频和预处理频谱两种输入
2. **性能**: 频谱输入模式可提升推理速度5-10倍
3. **兼容性**: 可与外部预处理系统无缝集成
4. **可扩展**: 支持多种预处理策略（STFT、Mel频谱等）

### 核心技术栈
1. **深度学习框架**: PyTorch 1.9+ (支持GPU/CPU)
2. **音频处理**: librosa, soundfile (专业音频处理)
3. **频谱特征**: STFT提取音频频谱，转换为对数刻度 (200×513)
4. **模型架构**: Transformer编码器-解码器，多头注意力机制
5. **推理策略**: 贪婪解码 + 束搜索 + 智能回退机制
6. **数据处理**: pandas, numpy (高效数据处理)
7. **可视化**: TensorBoard (训练监控)
8. **工具链**: tqdm (进度条), argparse (命令行)

### 架构设计
1. **模块化设计**: 低耦合、高内聚的模块架构
2. **统一接口**: 统一的训练基类和推理核心
3. **缓存系统**: 支持特征缓存，避免重复计算
4. **批量处理**: 高效的批量预处理和推理
5. **配置管理**: YAML配置文件，支持四种规模训练
6. **错误处理**: 完善的异常处理和错误恢复机制
7. **自动更新**: 数据变化自动同步相关文件

## 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir runs
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. 数据相关
```bash
# 问题：音频文件不存在或路径错误
# 解决：运行数据验证工具
python scripts/setup_data.py

# 问题：标签文件格式错误
# 解决：检查CSV文件格式，确保包含filename和label列
```

#### 2. 依赖和环境
```bash
# 问题：ImportError或模块缺失
pip install -r requirements.txt

# 问题：librosa安装失败
pip install librosa soundfile

# 问题：CUDA错误（GPU不可用）
# 解决：系统会自动使用CPU，无需特殊处理
```

#### 3. 内存和性能
```bash
# 问题：内存不足 (OOM)
# 解决：减少batch_size或hidden_dim
# 编辑config.json: "batch_size": 1, "hidden_dim": 16

# 问题：训练速度慢
# 解决：使用预计算模式
python scripts/batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv
```

### 🚀 性能优化建议

#### 训练优化
1. **使用预计算特征**: 运行 `batch_preprocess.py` 预处理所有音频
2. **GPU加速**: 设置 `--device cuda` （如果有GPU）
3. **批大小调优**: 根据内存大小调整 `batch_size`
4. **模型大小**: 小数据集使用小模型（hidden_dim=32）

#### 推理优化
1. **使用频谱输入模式**: 速度提升5-10倍
2. **批量推理**: 使用 `dual_input_inference.py` 的批量功能
3. **缓存机制**: 启用特征缓存避免重复计算

## 🌟 扩展功能

### 已实现的扩展
- ✅ **双输入架构**: 支持音频和频谱两种输入
- ✅ **多种预处理器**: STFT频谱、Mel频谱
- ✅ **智能推理**: 贪婪解码 + 束搜索 + 回退机制
- ✅ **批量处理**: 高效的批量预处理和推理
- ✅ **缓存系统**: 特征缓存和配置管理
- ✅ **统一训练基类**: 减少代码冗余，支持四种规模训练
- ✅ **模块化设计**: 低耦合、高扩展性的架构
- ✅ **自动数据更新**: 数据变化自动同步
- ✅ **配置管理**: 支持YAML配置文件
- ✅ **训练规模**: 支持small/medium/large/xlarge四种规模

### 可扩展方向
- 🔄 **更多中文词汇**: 扩展词汇表支持更多汉字
- 🔄 **数据增强**: 添加噪声、速度变化等增强技术
- 🔄 **实时识别**: WebSocket实时音频流处理
- 🔄 **多语言支持**: 支持英文、日文等其他语言
- 🔄 **语言模型**: 集成N-gram或神经语言模型后处理
- 🔄 **模型压缩**: 量化、剪枝等模型压缩技术
- 🔄 **Web界面**: 基于Flask/FastAPI的Web服务
- 🔄 **移动端**: 支持iOS/Android的移动应用

## 其他说明

### 可选

- 可选音频格式参考：wav
- data/audio中的Chinese_Number_xx.wav等音频文件来自Logic Pro自带声音包，仅供学习使用
- 
