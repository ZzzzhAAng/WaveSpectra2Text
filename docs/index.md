# 文档索引

## 📚 文档概览

欢迎使用WaveSpectra2Text文档！这里包含了项目的完整文档，帮助您快速上手和深入使用。

## 📖 主要文档

### 🚀 [README.md](../README.md)
项目的主要说明文档，包含：
- 项目介绍和特性
- 快速开始指南
- 项目架构说明
- 基本使用方法
- 技术特点介绍

### 📋 [API参考文档](api_reference.md)
完整的API参考，包含：
- 核心模块API
- 推理模块API
- 训练模块API
- 数据处理API
- 工具模块API
- 使用示例和错误处理

### 🎓 [用户指南](user_guide.md)
详细的用户指南，包含：
- 快速开始教程
- 训练指南和技巧
- 推理使用说明
- 数据管理方法
- 故障排除指南
- 性能优化建议
- 最佳实践

### 🔧 [安装指南](installation_guide.md)
完整的安装说明，包含：
- 系统要求
- 安装步骤
- 依赖包说明
- 验证安装
- 常见问题解决
- 更新和卸载

### 📊 [训练规模说明](training_scales.md)
训练规模配置说明，包含：
- 四种规模对比
- 配置参数说明
- 使用方法
- 选择建议

## 🎯 快速导航

### 新手用户
1. 📖 [README.md](../README.md) - 了解项目概况
2. 🔧 [安装指南](installation_guide.md) - 安装和配置
3. 🎓 [用户指南](user_guide.md) - 学习使用方法
4. 📊 [训练规模说明](training_scales.md) - 选择合适的训练配置

### 开发者用户
1. 📋 [API参考文档](api_reference.md) - 查看完整API
2. 🎓 [用户指南](user_guide.md) - 了解高级用法
3. 📖 [README.md](../README.md) - 查看技术架构

### 问题解决
1. 🎓 [用户指南 - 故障排除](user_guide.md#故障排除) - 常见问题解决
2. 🔧 [安装指南 - 常见问题](installation_guide.md#常见安装问题) - 安装问题解决
3. 📋 [API参考文档 - 错误处理](api_reference.md#错误处理) - API使用问题

## 📁 项目结构

```
WaveSpectra2Text/
├── 📄 README.md                    # 主要说明文档
├── 📁 docs/                        # 文档目录
│   ├── 📄 api_reference.md         # API参考文档
│   ├── 📄 user_guide.md           # 用户指南
│   ├── 📄 installation_guide.md   # 安装指南
│   ├── 📄 training_scales.md      # 训练规模说明
│   └── 📄 index.md                # 文档索引（本文件）
├── 📁 examples/                    # 示例代码
│   └── 📄 basic_usage.py          # 基本使用示例
├── 📁 configs/                     # 配置文件
│   ├── 📄 small_dataset.yaml      # 小数据集配置
│   ├── 📄 medium_dataset.yaml     # 中等数据集配置
│   ├── 📄 large_dataset.yaml      # 大数据集配置
│   └── 📄 xlarge_dataset.yaml     # 超大数据集配置
└── 📁 scripts/                     # 脚本文件
    ├── 📄 train.py                # 训练脚本
    ├── 📄 inference.py            # 推理脚本
    ├── 📄 batch_preprocess.py     # 批量预处理脚本
    └── 📄 setup_data.py           # 数据设置脚本
```

## 🔍 文档搜索

### 按功能搜索

#### 训练相关
- [训练指南](user_guide.md#训练指南)
- [训练规模说明](training_scales.md)
- [训练API](api_reference.md#训练模块-api)
- [训练脚本](../scripts/train.py)

#### 推理相关
- [推理指南](user_guide.md#推理指南)
- [推理API](api_reference.md#推理模块-api)
- [推理脚本](../scripts/inference.py)

#### 数据处理
- [数据管理](user_guide.md#数据管理)
- [预处理API](api_reference.md#预处理器)
- [批量预处理脚本](../scripts/batch_preprocess.py)

#### 配置管理
- [配置参数](README.md#配置参数)
- [训练规模配置](training_scales.md)
- [配置文件](../configs/)

### 按问题类型搜索

#### 安装问题
- [安装指南](installation_guide.md)
- [系统要求](installation_guide.md#系统要求)
- [常见安装问题](installation_guide.md#常见安装问题)

#### 使用问题
- [用户指南](user_guide.md)
- [快速开始](user_guide.md#快速开始)
- [故障排除](user_guide.md#故障排除)

#### 开发问题
- [API参考](api_reference.md)
- [示例代码](../examples/)
- [测试文件](../tests/)

## 📝 文档贡献

如果您发现文档中的问题或有改进建议，欢迎：

1. **提交Issue**: 在GitHub上提交问题报告
2. **提交PR**: 直接提交文档改进的Pull Request
3. **联系维护者**: 通过邮件联系项目维护者

### 文档编写规范
- 使用Markdown格式
- 保持结构清晰
- 提供完整的代码示例
- 包含错误处理说明
- 使用中文编写

## 🔄 文档更新

文档会随着项目的发展持续更新：

- **版本更新**: 每个主要版本都会更新相关文档
- **功能更新**: 新功能添加时会更新API文档
- **问题修复**: 根据用户反馈修复文档中的问题
- **内容完善**: 根据使用情况完善文档内容

## 📞 获取帮助

如果您在使用过程中遇到问题：

1. **查看文档**: 首先查看相关文档
2. **搜索问题**: 在GitHub Issues中搜索类似问题
3. **提交问题**: 如果找不到解决方案，提交新的Issue
4. **社区讨论**: 参与项目讨论区交流

---

**最后更新**: 2024年1月
**文档版本**: 1.0.0
**项目版本**: 1.0.0
