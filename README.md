# WaveSpectra2Text
# 双输入语音识别系统 - 音频与频谱特征

本项目实现了一个支持双输入模式的语音识别系统。系统支持两种输入方式：
1. **原始音频输入**：完整的音频处理流程（音频 → 频谱提取 → 模型推理 → 文本）
2. **频谱特征输入**：跳过预处理的快速推理（预处理频谱 → 模型推理 → 文本）

现在以中文数字1～10为例，展示了从频谱特征到文本的端到端识别能力。

## 项目结构

```
WaveSpectra2Text/
├── 📁 data/                          # 数据目录
│   ├── 📁 audio/                     # 原始音频文件
│   │   ├── Chinese_Number_01.wav     # 中文数字音频样本
│   │   └── ...
│   └── 📄 labels.csv                 # 标签文件
│
├── 🧠 核心模块
│   ├── 📄 model.py                   # Transformer编码器-解码器模型
│   ├── 📄 vocab.py                   # 词汇表管理（中文数字1-10）
│   ├── 📄 audio_preprocess.py        # 统一音频预处理模块
│   ├── 📄 audio_dataset.py           # 重构后的数据集（支持实时/预计算）
│   └── 📄 data_utils.py              # 数据处理工具（兼容层）
│
├── 🚀 推理系统
│   ├── 📄 inference_core.py          # 统一推理核心
│   └── 📄 dual_input_inference.py    # 双输入推理系统（统一推理接口）
│
├── 🏋️ 训练系统
│   ├── 📁 train_at_different_scales/ # 不同规模的训练脚本
│   │   ├── 📄 train_scale_1.py      # 小数据集训练
│   │   ├── 📄 train_scale_2.py      # 中等数据集训练
│   │   ├── 📄 train_scale_3.py      # 大数据集训练
│   │   └── 📄 train_scale_4.py      # 超大数据集训练
│   └── 📄 config.json               # 训练配置文件
│
├── 🔧 工具脚本
│   ├── 📄 common_utils.py           # 通用工具模块（新增）
│   ├── 📄 setup_data.py             # 数据设置工具
│   ├── 📄 batch_preprocess.py       # 批量预处理工具
│   ├── 📄 auto_update_system.py     # 自动更新系统（新增）
│   ├── 📄 simple_auto_update.py     # 简化自动更新（新增）
│   ├── 📄 sync_data.py              # 数据同步工具（新增）
│   └── 📄 watch_data_changes.py     # 实时监控工具（新增）
│
├── 🧪 测试
│   └── 📁 tests/
│       ├── 📄 test_refactor.py      # 重构测试
│       └── 📄 test_structure.py     # 结构测试
│
└── 📚 文档
    ├── 📄 README.md                 # 项目说明（本文件）
    ├── 📄 操作指南.md               # 完整操作手册（新增）
    ├── 📄 自动更新功能说明.md       # 自动更新说明（新增）
    ├── 📄 requirements.txt          # 依赖包列表
    └── 📄 parameters_tuning_guide.json # 参数调优指南
```

## 安装依赖

```bash
pip install -r requirements.txt
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

## 训练

根据训练集大小选择适用于不同规模的 train_scale_x.py，并相应更改 config.json

### 基本训练

```bash
python train_scale_1.py
```

### 使用自定义配置

```bash
python train_scale_1.py --config config.json
```

### 从检查点恢复训练

```bash
python train_scale_1.py --resume checkpoints/checkpoint_epoch_50.pth
```

## 推理

### 🚀 双输入推理系统（推荐）

#### 自动模式 - 根据文件扩展名自动选择输入模式
```bash
# 音频文件自动使用音频输入模式
python dual_input_inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav

# 频谱文件自动使用频谱输入模式  
python dual_input_inference.py --model checkpoints/best_model.pth --input data/features/Chinese_Number_01.npy
```

#### 指定输入模式
```bash
# 强制使用音频输入模式
python dual_input_inference.py --model checkpoints/best_model.pth --input data/audio/Chinese_Number_01.wav --mode audio

# 强制使用频谱输入模式
python dual_input_inference.py --model checkpoints/best_model.pth --input data/features/Chinese_Number_01.npy --mode spectrogram
```

#### 查看性能对比和演示代码
```bash
# 显示两种输入模式的性能对比
python dual_input_inference.py --compare

# 显示外部系统集成的演示代码
python dual_input_inference.py --demo
```

### 📊 高级使用

#### 批量处理
```bash
# 批量处理音频目录
python dual_input_inference.py --model checkpoints/best_model.pth --input data/audio/ --mode auto

# 批量处理频谱文件
python dual_input_inference.py --model checkpoints/best_model.pth --input data/features/ --mode spectrogram
```

#### 编程接口使用
```python
# 使用统一推理核心
from inference_core import InferenceCore
core = InferenceCore('checkpoints/best_model.pth')
result = core.infer_from_audio('audio.wav')
print(result['text'])

# 使用双输入识别器
from dual_input_inference import DualInputSpeechRecognizer
recognizer = DualInputSpeechRecognizer('checkpoints/best_model.pth')
result = recognizer.recognize_from_audio('audio.wav')
print(result['text'])
```

## 🔧 数据预处理

### 快速开始 - 数据设置
```bash
# 自动扫描音频文件并创建标签模板
python setup_data.py
```

### 批量预处理（推荐用于大数据集）
```bash
# 批量预处理音频文件为频谱特征，大幅提升训练速度
python batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv --output_dir data/features

# 使用不同的预处理器
python batch_preprocess.py --preprocessor mel_spectrogram --n_mels 128

# 强制重新计算所有特征
python batch_preprocess.py --force_recompute
```

### 数据加载模式
```python
from data_utils import get_dataloader

# 自动模式：优先使用预计算特征，否则实时计算
dataloader = get_dataloader(mode='auto', batch_size=4)

# 实时模式：每次训练时计算特征
dataloader = get_dataloader(mode='realtime', batch_size=4)

# 预计算模式：使用预处理好的特征（最快）
dataloader = get_dataloader(mode='precomputed', precomputed_dir='data/features')
```

## ⚙️ 配置参数

### 训练配置 (config.json)
```json
{
  "batch_size": 1,           // 批大小（小数据集推荐1）
  "learning_rate": 0.00005,  // 学习率
  "hidden_dim": 32,          // 隐藏层维度（小数据集推荐32）
  "encoder_layers": 2,       // 编码器层数
  "decoder_layers": 2,       // 解码器层数
  "dropout": 0.1,            // Dropout比率
  "num_epochs": 200          // 训练轮数
}
```

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

### 核心技术
1. **频谱特征**: STFT提取音频频谱，转换为对数刻度 (200×513)
2. **Transformer架构**: 编码器-解码器架构，支持注意力机制
3. **智能推理**: 贪婪解码 + 束搜索 + 智能回退机制
4. **缓存系统**: 支持特征缓存，避免重复计算
5. **批量处理**: 高效的批量预处理和推理

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
python setup_data.py

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
python batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv
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

### 可扩展方向
- 🔄 **更多中文词汇**: 扩展词汇表支持更多汉字
- 🔄 **数据增强**: 添加噪声、速度变化等增强技术
- 🔄 **实时识别**: WebSocket实时音频流处理
- 🔄 **多语言支持**: 支持英文、日文等其他语言
- 🔄 **语言模型**: 集成N-gram或神经语言模型后处理
- 🔄 **模型压缩**: 量化、剪枝等模型压缩技术

## 其他说明

### 可选

- 可选音频格式参考：wav
- data/audio中的Chinese_Number_xx.wav等音频文件来自Logic Pro自带声音包，仅供学习使用
- 
