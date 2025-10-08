"""
核心模块 - 包含模型、词汇表和推理核心
"""

from .model import create_model, Seq2SeqModel
from .vocab import vocab, Vocabulary
from .inference import InferenceCore

__all__ = [
    "create_model",
    "Seq2SeqModel",
    "vocab", 
    "Vocabulary",
    "InferenceCore",
]
