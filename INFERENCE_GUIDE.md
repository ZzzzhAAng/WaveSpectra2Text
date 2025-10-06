
# 🎯 推理系统使用指南

## 基本用法

### 1. 单文件推理
```bash
python inference.py --model checkpoints/best_model.pth --audio data/audio/test.wav
```

### 2. 批量推理
```bash
python inference.py --model checkpoints/best_model.pth --audio_dir data/audio
```

### 3. 数据集评估
```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --audio_dir data/audio \
    --labels data/labels.csv \
    --output results.csv
```

## Python API 用法

```python
from inference import SpeechRecognizer

# 创建识别器
recognizer = SpeechRecognizer('checkpoints/best_model.pth')

# 单文件识别
result = recognizer.recognize_file('test.wav')
print(f"识别结果: {result['text']}")

# 批量识别
results = recognizer.recognize_batch(['file1.wav', 'file2.wav'])
for result in results:
    print(f"{result['file']}: {result['text']}")

# 数据集评估
results, accuracy = recognizer.evaluate_on_dataset('data/audio', 'data/labels.csv')
print(f"准确率: {accuracy:.2%}")
```

## 高级选项

- `--beam_size 5`: 束搜索大小
- `--no_beam_search`: 使用贪婪解码
- `--device cuda`: 使用GPU加速
- `--output results.csv`: 保存结果到文件
