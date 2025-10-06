# -*- coding: utf-8 -*-
"""
语音识别模型 - 从频谱到文本
包含Encoder和Decoder模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpectrogramEncoder(nn.Module):
    """频谱编码器 - 将频谱特征编码为隐藏表示"""
    def __init__(self, input_dim=513, hidden_dim=256, num_layers=4, dropout=0.1):
        super(SpectrogramEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, spectrogram, src_mask=None):
        """
        Args:
            spectrogram: (batch_size, seq_len, input_dim) 频谱特征
            src_mask: 源序列掩码
        Returns:
            encoded: (batch_size, seq_len, hidden_dim) 编码后的特征
        """
        # 输入投影
        x = self.input_projection(spectrogram)  # (batch_size, seq_len, hidden_dim)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # 层归一化
        encoded = self.layer_norm(encoded)
        
        return encoded

class AttentionDecoder(nn.Module):
    """注意力解码器 - 将编码特征解码为文本序列"""
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt_tokens, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt_tokens: (batch_size, tgt_len) 目标token序列
            encoder_output: (batch_size, src_len, hidden_dim) 编码器输出
            tgt_mask: 目标序列掩码
            memory_mask: 记忆掩码
        Returns:
            output: (batch_size, tgt_len, vocab_size) 输出logits
        """
        # 词嵌入
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.hidden_dim)
        
        # 添加位置编码
        tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_len, batch_size, hidden_dim)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (batch_size, tgt_len, hidden_dim)
        
        # Dropout
        tgt_emb = self.dropout(tgt_emb)
        
        # Transformer解码
        decoded = self.transformer_decoder(
            tgt_emb, 
            encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask
        )
        
        # 输出投影
        output = self.output_projection(decoded)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class Seq2SeqModel(nn.Module):
    """完整的序列到序列模型"""
    def __init__(self, vocab_size, input_dim=513, hidden_dim=256, 
                 encoder_layers=4, decoder_layers=4, dropout=0.1):
        super(Seq2SeqModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 编码器和解码器
        self.encoder = SpectrogramEncoder(input_dim, hidden_dim, encoder_layers, dropout)
        self.decoder = AttentionDecoder(vocab_size, hidden_dim, decoder_layers, dropout)
        
    def forward(self, spectrogram, tgt_tokens, src_mask=None, tgt_mask=None):
        """
        Args:
            spectrogram: (batch_size, src_len, input_dim) 频谱特征
            tgt_tokens: (batch_size, tgt_len) 目标token序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            output: (batch_size, tgt_len, vocab_size) 输出logits
        """
        # 编码
        encoder_output = self.encoder(spectrogram, src_mask)
        
        # 解码
        if tgt_mask is None and tgt_tokens.size(1) > 1:
            tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_tokens.size(1))
            if tgt_tokens.is_cuda:
                tgt_mask = tgt_mask.cuda()
        
        output = self.decoder(tgt_tokens, encoder_output, tgt_mask, src_mask)
        
        return output
    
    def encode(self, spectrogram, src_mask=None):
        """仅编码"""
        return self.encoder(spectrogram, src_mask)
    
    def decode_step(self, tgt_tokens, encoder_output, memory_mask=None):
        """单步解码"""
        tgt_mask = None
        if tgt_tokens.size(1) > 1:
            tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_tokens.size(1))
            if tgt_tokens.is_cuda:
                tgt_mask = tgt_mask.cuda()
        
        return self.decoder(tgt_tokens, encoder_output, tgt_mask, memory_mask)

def create_model(vocab_size, input_dim=513, hidden_dim=256, 
                encoder_layers=4, decoder_layers=4, dropout=0.1):
    """创建模型实例"""
    model = Seq2SeqModel(
        vocab_size=vocab_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        dropout=dropout
    )
    
    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

if __name__ == "__main__":
    # 测试模型
    vocab_size = 14  # 词汇表大小
    batch_size = 2
    src_len = 100
    tgt_len = 5
    input_dim = 513
    
    # 创建模型
    model = create_model(vocab_size)
    
    # 创建测试数据
    spectrogram = torch.randn(batch_size, src_len, input_dim)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    output = model(spectrogram, tgt_tokens)
    
    print(f"输入频谱形状: {spectrogram.shape}")
    print(f"目标tokens形状: {tgt_tokens.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")