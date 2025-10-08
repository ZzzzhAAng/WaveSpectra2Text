"""
WaveSpectra2Text - 双输入语音识别系统

一个支持音频和频谱两种输入模式的语音识别系统，
基于Transformer架构，专门用于中文数字识别。

主要特性:
- 双输入模式：支持原始音频和预处理频谱
- Transformer架构：编码器-解码器设计
- 智能推理：贪婪解码 + 束搜索 + 回退机制
- 模块化设计：低耦合、高扩展性
- 统一接口：训练和推理的统一API

版本: 1.0.0
作者: WaveSpectra2Text Team
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "WaveSpectra2Text Team"
__email__ = "contact@wavespectra2text.com"
__license__ = "MIT"

# 导入主要类和函数
from .core.model import create_model, Seq2SeqModel
from .core.vocab import vocab, Vocabulary
from .core.inference import InferenceCore
from .data.preprocessing import AudioPreprocessor, PreprocessorFactory
from .data.dataset import AudioDataset
from .training.trainer import BaseTrainer, create_trainer
from .inference.recognizer import DualInputSpeechRecognizer
from .inference.strategies import DecoderStrategy, GreedyDecoder, BeamSearchDecoder, SamplingDecoder
from .utils.audio import AudioProcessor, LabelManager, FileUtils
from .utils.logging import setup_logging, get_logger
from .utils.metrics import calculate_accuracy, calculate_wer, calculate_bleu

# 定义公共API
__all__ = [
    # 核心模块
    "create_model",
    "Seq2SeqModel", 
    "vocab",
    "Vocabulary",
    "InferenceCore",
    
    # 数据处理
    "AudioPreprocessor",
    "PreprocessorFactory",
    "AudioDataset",
    
    # 训练
    "BaseTrainer",
    "create_trainer",
    
    # 推理
    "DualInputSpeechRecognizer",
    "DecoderStrategy",
    "GreedyDecoder",
    "BeamSearchDecoder", 
    "SamplingDecoder",
    
    # 工具
    "AudioProcessor",
    "LabelManager",
    "FileUtils",
    "setup_logging",
    "get_logger",
    "calculate_accuracy",
    "calculate_wer",
    "calculate_bleu",
    
    # 元数据
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
