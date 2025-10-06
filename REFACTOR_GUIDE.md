# 数据预处理重构指南

## 🎯 重构目标

本次重构解决了原有代码中的以下问题：
- **高耦合**: `data_utils.py` 和 `preprocess_spectrum.py` 存在重复的预处理逻辑
- **低复用**: 相同的音频处理代码在多个地方重复
- **内存浪费**: 数据集初始化时就加载所有数据到内存
- **扩展困难**: 难以支持新的预处理策略

## 🏗️ 新架构设计

### 核心模块

1. **`audio_preprocessing.py`** - 统一的预处理器框架
   - 抽象基类 `AudioPreprocessor`
   - 具体实现: `SpectrogramPreprocessor`, `MelSpectrogramPreprocessor`
   - 工厂模式: `PreprocessorFactory`
   - 离线处理: `OfflinePreprocessor`

2. **`audio_dataset.py`** - 灵活的数据集实现
   - 支持实时计算和预计算两种模式
   - 低耦合设计，可插拔的预处理器
   - 智能缓存机制

3. **`batch_preprocess.py`** - 批量预处理工具
   - 替代原来的 `preprocess_spectrum.py`
   - 支持多种预处理策略
   - 完整的验证和统计功能

4. **`data_utils.py`** - 重构后的数据工具
   - 保持向后兼容
   - 支持多种数据加载模式
   - 自动选择最优模式

## 🚀 使用方式

### 1. 快速开始 (兼容模式)

现有代码无需修改，直接使用：

```python
from data_utils import get_dataloader

# 自动选择最优模式
dataloader = get_dataloader(
    audio_dir='data/audio',
    labels_file='data/labels.csv',
    batch_size=4
)
```

### 2. 推荐用法 (新架构)

#### 步骤1: 批量预处理 (一次性)

```bash
# 基本用法
python batch_preprocess.py --audio_dir data/audio --labels_file data/labels.csv

# 自定义参数
python batch_preprocess.py \
    --audio_dir data/audio \
    --labels_file data/labels.csv \
    --output_dir data/features \
    --preprocessor spectrogram \
    --sample_rate 48000 \
    --max_length 200

# 使用Mel频谱
python batch_preprocess.py \
    --preprocessor mel_spectrogram \
    --n_mels 128
```

#### 步骤2: 使用预计算数据

```python
from data_utils import get_precomputed_dataloader

# 使用预计算特征 (最快)
dataloader = get_precomputed_dataloader(
    labels_file='data/labels.csv',
    precomputed_dir='data/features',
    batch_size=4
)
```

### 3. 高级用法

#### 自定义预处理器

```python
from audio_preprocessing import AudioPreprocessor, PreprocessorFactory

class CustomPreprocessor(AudioPreprocessor):
    def process(self, audio_path):
        # 你的自定义处理逻辑
        pass
    
    def get_feature_shape(self):
        return (200, 513)

# 注册自定义预处理器
PreprocessorFactory.register('custom', CustomPreprocessor)

# 使用自定义预处理器
from audio_dataset import create_realtime_dataset

dataset = create_realtime_dataset(
    labels_file='data/labels.csv',
    audio_dir='data/audio',
    preprocessor_type='custom'
)
```

#### 实时计算 + 缓存

```python
from data_utils import get_realtime_dataloader

# 实时计算，但使用缓存加速
dataloader = get_realtime_dataloader(
    audio_dir='data/audio',
    labels_file='data/labels.csv',
    cache_dir='cache/features',  # 缓存目录
    batch_size=4
)
```

## 📊 性能对比

| 模式 | 首次加载时间 | 后续加载时间 | 内存占用 | 磁盘占用 |
|------|-------------|-------------|----------|----------|
| 旧版本 | 很慢 | 很慢 | 很高 | 低 |
| 实时计算 | 慢 | 慢 | 低 | 低 |
| 实时+缓存 | 慢 | 快 | 低 | 中 |
| 预计算 | 快 | 很快 | 低 | 高 |

## 🔄 迁移步骤

### 从旧版本迁移

1. **保持现有代码不变** (兼容模式)
2. **运行批量预处理**:
   ```bash
   python batch_preprocess.py --migrate
   ```
3. **逐步切换到新接口**:
   ```python
   # 旧代码
   from data_utils import get_dataloader
   dataloader = get_dataloader()
   
   # 新代码
   from data_utils import get_precomputed_dataloader
   dataloader = get_precomputed_dataloader()
   ```

### 验证迁移结果

```bash
# 验证预处理结果
python batch_preprocess.py --validate --output_dir data/features

# 测试数据加载
python data_utils.py
```

## 🎨 扩展示例

### 添加新的预处理策略

```python
# 1. 实现预处理器
class MFCCPreprocessor(AudioPreprocessor):
    def __init__(self, n_mfcc=13, **kwargs):
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
    
    def process(self, audio_path):
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        return mfcc.T  # (time_steps, n_mfcc)
    
    def get_feature_shape(self):
        return (None, self.n_mfcc)  # 可变长度

# 2. 注册预处理器
PreprocessorFactory.register('mfcc', MFCCPreprocessor)

# 3. 使用新预处理器
python batch_preprocess.py --preprocessor mfcc --n_mfcc 13
```

## 🛠️ 故障排除

### 常见问题

1. **导入错误**: 确保新模块在Python路径中
2. **缓存问题**: 删除缓存目录重新生成
3. **内存不足**: 使用预计算模式
4. **文件不存在**: 检查路径配置

### 调试命令

```bash
# 检查预处理器
python -c "from audio_preprocessing import PreprocessorFactory; print(PreprocessorFactory.list_available())"

# 验证数据集
python audio_dataset.py

# 测试批量处理
python batch_preprocess.py --validate
```

## 📈 优势总结

### 🎯 低耦合
- 预处理逻辑完全独立
- 数据集与预处理器解耦
- 支持插件式扩展

### ⚡ 高性能
- 预计算模式避免重复计算
- 智能缓存机制
- 内存友好的懒加载

### 🔧 易扩展
- 工厂模式支持新预处理器
- 策略模式支持多种数据加载方式
- 配置驱动的参数管理

### 🔄 向后兼容
- 现有代码无需修改
- 渐进式迁移支持
- 完整的兼容层

---

**建议**: 对于新项目，直接使用新架构；对于现有项目，可以渐进式迁移。