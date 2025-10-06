# -*- coding: utf-8 -*-
"""
词汇表管理模块
用于管理中文数字1-10的词汇映射
"""

class Vocabulary:
    def __init__(self):
        # 中文数字1-10的词汇表
        self.word_to_idx = {
            '<PAD>': 0,    # 填充符号
            '<SOS>': 1,    # 开始符号
            '<EOS>': 2,    # 结束符号
            '<UNK>': 3,    # 未知符号
            '一': 4,
            '二': 5,
            '三': 6,
            '四': 7,
            '五': 8,
            '六': 9,
            '七': 10,
            '八': 11,
            '九': 12,
            '十': 13,
            '你': 14,      # 新增
            '好': 15       # 新增
        }
        
        # 反向映射
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # 词汇表大小
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text):
        """将文本编码为索引序列"""
        if isinstance(text, str):
            # 单个字符串
            indices = [self.word_to_idx.get(char, self.word_to_idx['<UNK>']) for char in text]
            return [self.word_to_idx['<SOS>']] + indices + [self.word_to_idx['<EOS>']]
        elif isinstance(text, list):
            # 字符列表
            indices = [self.word_to_idx.get(char, self.word_to_idx['<UNK>']) for char in text]
            return [self.word_to_idx['<SOS>']] + indices + [self.word_to_idx['<EOS>']]
        else:
            raise ValueError("输入必须是字符串或字符列表")
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        words = []
        for idx in indices:
            if idx == self.word_to_idx['<EOS>']:
                break
            elif idx not in [self.word_to_idx['<PAD>'], self.word_to_idx['<SOS>']]:
                words.append(self.idx_to_word.get(idx, '<UNK>'))
        return ''.join(words)
    
    def get_padding_idx(self):
        """获取填充符号的索引"""
        return self.word_to_idx['<PAD>']
    
    def get_sos_idx(self):
        """获取开始符号的索引"""
        return self.word_to_idx['<SOS>']
    
    def get_eos_idx(self):
        """获取结束符号的索引"""
        return self.word_to_idx['<EOS>']

# 全局词汇表实例
vocab = Vocabulary()

if __name__ == "__main__":
    # 测试代码
    print("词汇表大小:", vocab.vocab_size)
    print("词汇映射:", vocab.word_to_idx)
    
    # 测试编码解码
    test_text = "一二三"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)
    
    print(f"原文本: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")