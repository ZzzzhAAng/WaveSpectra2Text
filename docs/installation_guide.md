# 安装指南

## 📦 系统要求

### 硬件要求
- **CPU**: 支持AVX指令集的现代处理器（推荐Intel i5或AMD Ryzen 5以上）
- **内存**: 至少4GB RAM（推荐8GB以上）
- **存储**: 至少2GB可用空间
- **GPU**: 可选，支持CUDA的NVIDIA GPU可显著提升训练速度

### 软件要求
- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🚀 安装步骤

### 方法1: 从源码安装（推荐）

#### 1. 克隆项目
```bash
git clone https://github.com/wavespectra2text/wavespectra2text.git
cd wavespectra2text
```

#### 2. 创建虚拟环境（推荐）
```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 使用conda
conda create -n wavespectra2text python=3.9
conda activate wavespectra2text
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 开发模式安装
```bash
pip install -e .
```

### 方法2: 直接安装

#### 1. 安装依赖
```bash
pip install torch torchvision torchaudio
pip install librosa soundfile pandas numpy
pip install scikit-learn tqdm tensorboard
pip install pyyaml
```

#### 2. 下载项目文件
下载项目源码并解压到目标目录。

### 方法3: 使用pip安装（如果发布到PyPI）

```bash
pip install wavespectra2text
```

## 🔧 依赖包说明

### 核心依赖
- **torch**: PyTorch深度学习框架
- **librosa**: 音频处理库
- **soundfile**: 音频文件读写
- **pandas**: 数据处理
- **numpy**: 数值计算

### 可选依赖
- **tensorboard**: 训练可视化
- **scikit-learn**: 机器学习工具
- **tqdm**: 进度条显示
- **pyyaml**: YAML配置文件支持

### GPU支持（可选）
如果需要GPU加速，请安装CUDA版本的PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ✅ 验证安装

### 1. 检查Python版本
```bash
python --version
# 应该显示 Python 3.8 或更高版本
```

### 2. 检查依赖包
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import librosa; print(f'librosa版本: {librosa.__version__}')"
python -c "import pandas; print(f'pandas版本: {pandas.__version__}')"
```

### 3. 运行测试
```bash
# 运行基本测试
python -c "from wavespectra2text import create_model, vocab; print('安装成功!')"

# 运行完整测试套件
python tests/run_tests.py
```

### 4. 检查GPU支持（如果安装了CUDA版本）
```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA设备数: {torch.cuda.device_count()}')"
```

## 🐛 常见安装问题

### 问题1: librosa安装失败

**错误信息:**
```
ERROR: Failed building wheel for librosa
```

**解决方案:**
```bash
# 安装系统依赖
# Ubuntu/Debian
sudo apt-get install libsndfile1

# macOS
brew install libsndfile

# Windows
# 下载并安装 Microsoft Visual C++ Redistributable

# 然后重新安装
pip install librosa
```

### 问题2: PyTorch安装失败

**错误信息:**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解决方案:**
```bash
# 使用官方源安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或使用conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 问题3: 权限错误

**错误信息:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**解决方案:**
```bash
# 使用用户安装
pip install --user -r requirements.txt

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 问题4: 内存不足

**错误信息:**
```
ERROR: Could not install packages due to an MemoryError
```

**解决方案:**
```bash
# 增加pip缓存大小
pip install --no-cache-dir -r requirements.txt

# 或逐个安装
pip install torch
pip install librosa
pip install pandas
# ...
```

## 🔄 更新安装

### 更新项目
```bash
# 如果是从源码安装
cd wavespectra2text
git pull origin main
pip install -e .

# 如果是从PyPI安装
pip install --upgrade wavespectra2text
```

### 更新依赖
```bash
pip install --upgrade -r requirements.txt
```

## 🗑️ 卸载

### 完全卸载
```bash
# 卸载包
pip uninstall wavespectra2text

# 删除项目目录
rm -rf wavespectra2text

# 删除虚拟环境（如果使用）
rm -rf venv
```

### 保留数据卸载
```bash
# 只卸载包，保留数据和配置
pip uninstall wavespectra2text

# 数据文件会保留在以下目录：
# - data/
# - checkpoints/
# - runs/
# - configs/
```

## 📋 安装检查清单

安装完成后，请确认以下项目：

- [ ] Python版本 >= 3.8
- [ ] PyTorch安装成功
- [ ] librosa安装成功
- [ ] 项目模块可以正常导入
- [ ] 基本功能测试通过
- [ ] GPU支持（如果适用）
- [ ] 虚拟环境激活（如果使用）

## 🆘 获取帮助

如果遇到安装问题，请：

1. **查看错误日志**: 仔细阅读错误信息
2. **检查系统要求**: 确认满足最低要求
3. **尝试不同方法**: 使用conda或不同的pip源
4. **搜索已知问题**: 查看GitHub Issues
5. **提交问题报告**: 提供详细的错误信息和系统环境

### 有用的资源
- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [librosa安装指南](https://librosa.org/doc/latest/install.html)
- [Python虚拟环境指南](https://docs.python.org/3/tutorial/venv.html)
