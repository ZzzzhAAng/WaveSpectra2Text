# WaveSpectra2Text
# 语音识别项目 - 从频谱到文本

本项目实现了一个从音频频谱直接识别对应文本额的语音识别系统。该系统的特点是在识别过程中不依赖原始音频，而是直接对音频的频谱特征进行分析，进而得到文本。

现在以中文数字1～10为例

## 项目结构

```

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

## 模型架构

- **编码器**: 基于Transformer的频谱编码器，将STFT频谱特征编码为隐藏表示
- **解码器**: 基于Transformer的注意力解码器，将编码特征解码为文本序列
- **词汇表**: 支持中文数字1～10（可扩展）以及特殊符号（PAD, SOS, EOS, UNK）

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

### 识别单个音频文件

```bash
python inference.py --model checkpoints/best_model.pth --audio data/audio/1.wav
```

### 批量识别

```bash
python inference.py --model checkpoints/best_model.pth --audio_dir data/audio --output results.csv
```

### 在数据集上评估

```bash
python inference.py --model checkpoints/best_model.pth --audio_dir data/audio --labels data/labels.csv --output evaluation.csv
```

### 使用束搜索

```bash
python inference.py --model checkpoints/best_model.pth --audio data/audio/audio_1.wav --beam_size 5
```

## 配置参数

主要配置参数说明：

- `batch_size`: 批大小
- `learning_rate`: 学习率
- `hidden_dim`: 隐藏层维度
- `encoder_layers`: 编码器层数
- `decoder_layers`: 解码器层数
- `dropout`: Dropout比率
- `num_epochs`: 训练轮数

## 技术特点

1. **频谱特征**: 使用STFT提取音频的频谱特征，转换为对数刻度
2. **Transformer架构**: 采用现代的Transformer编码器-解码器架构
3. **注意力机制**: 使用多头注意力机制捕捉频谱和文本之间的对应关系
4. **位置编码**: 为序列添加位置信息
5. **束搜索**: 支持束搜索和贪婪解码两种推理方式

## 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir runs
```

## 故障排除

### 常见问题

1. **ImportError**: 确保安装了所有依赖包
2. **CUDA错误**: 如果没有GPU，模型会自动使用CPU
3. **音频加载失败**: 确保音频文件格式正确且路径存在
4. **内存不足**: 减少batch_size或hidden_dim

### 性能优化

1. 使用GPU加速训练
2. 调整batch_size以充分利用内存
3. 使用混合精度训练（需要较新的PyTorch版本）
4. 调整模型大小以平衡性能和准确率

## 扩展功能

- 支持更多中文词汇
- 添加数据增强技术
- 实现实时语音识别
- 支持多语言识别
- 添加语言模型后处理

## 其他说明

### 可选

- 可选音频格式参考：wav
- data/audio中的Chinese_Number_xx.wav等音频文件来自Logic Pro自带声音包，仅供学习使用
- 
