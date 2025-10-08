# 训练规模统一配置说明

## 📊 支持的训练规模

本项目统一使用以下四个训练规模：

### 1. **small** - 小数据集
- **适用场景**: 快速测试、概念验证、小规模数据
- **配置特点**:
  - batch_size: 1
  - learning_rate: 1e-5
  - hidden_dim: 64
  - encoder_layers: 1
  - decoder_layers: 1
  - dropout: 0.5
  - num_epochs: 30
  - validation_split: 0.0 (不分割验证集)

### 2. **medium** - 中等数据集
- **适用场景**: 常规训练、中等规模数据
- **配置特点**:
  - batch_size: 2
  - learning_rate: 5e-5
  - hidden_dim: 128
  - encoder_layers: 2
  - decoder_layers: 2
  - dropout: 0.3
  - num_epochs: 50
  - validation_split: 0.2

### 3. **large** - 大数据集
- **适用场景**: 大规模数据训练、生产环境
- **配置特点**:
  - batch_size: 4
  - learning_rate: 1e-4
  - hidden_dim: 256
  - encoder_layers: 4
  - decoder_layers: 4
  - dropout: 0.2
  - num_epochs: 100
  - validation_split: 0.15

### 4. **xlarge** - 超大数据集
- **适用场景**: 超大规模数据、高性能训练
- **配置特点**:
  - batch_size: 8
  - learning_rate: 2e-4
  - hidden_dim: 512
  - encoder_layers: 6
  - decoder_layers: 6
  - dropout: 0.1
  - num_epochs: 200
  - validation_split: 0.1

## 🚀 使用方法

### 命令行训练
```bash
# 小数据集训练
python scripts/train.py --scale small

# 中等数据集训练
python scripts/train.py --scale medium

# 大数据集训练
python scripts/train.py --scale large

# 超大数据集训练
python scripts/train.py --scale xlarge
```

### 使用配置文件
```bash
# 使用预定义配置文件
python scripts/train.py --config configs/small_dataset.yaml
python scripts/train.py --config configs/medium_dataset.yaml
python scripts/train.py --config configs/large_dataset.yaml
python scripts/train.py --config configs/xlarge_dataset.yaml
```

### 编程接口
```python
from wavespectra2text.training.config import get_default_config
from wavespectra2text.training.trainer import create_trainer

# 获取配置
config = get_default_config('medium')

# 创建训练器
trainer = create_trainer('improved', model, train_loader, val_loader, device, config)
```

## 📁 配置文件位置

- `configs/small_dataset.yaml` - 小数据集配置
- `configs/medium_dataset.yaml` - 中等数据集配置
- `configs/large_dataset.yaml` - 大数据集配置
- `configs/xlarge_dataset.yaml` - 超大数据集配置
- `configs/default.yaml` - 默认配置

## 🔧 训练器映射

不同规模对应不同的训练器类型：

- **small** → `simple` (SimpleTrainer)
- **medium** → `improved` (ImprovedTrainer)
- **large** → `large` (LargeDatasetTrainer)
- **xlarge** → `large` (LargeDatasetTrainer)

## 💡 选择建议

- **开发测试**: 使用 `small` 规模快速验证
- **常规训练**: 使用 `medium` 规模平衡性能和速度
- **生产环境**: 使用 `large` 或 `xlarge` 规模获得最佳效果
- **资源受限**: 根据硬件条件选择合适的规模
