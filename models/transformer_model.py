"""Transformer模型组件

该模块包含构建Transformer模型的基础组件。
主要组件：
1. 位置编码器
2. 符号嵌入层
3. Transformer编码器
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from symbol_dictionary import SPECIAL_TOKENS

class PositionalEncoder(nn.Module):
    """位置编码器

    为输入序列添加位置信息，使用正弦和余弦函数生成位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        """初始化位置编码器

        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加batch维度

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 位置编码 [batch_size, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]


class SymbolEncoder(nn.Module):
    """符号嵌入层

    将符号ID转换为密集向量表示。
    """

    def __init__(self, num_tokens: int, d_model: int, init_range: float = 0.1):
        """初始化符号嵌入层

        Args:
            num_tokens: 词汇表大小
            d_model: 嵌入维度
            init_range: 初始化范围
        """
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(num_tokens, d_model,
                                  padding_idx=SPECIAL_TOKENS.PAD_ID)

        # 初始化嵌入权重
        self.embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            src: 输入符号ID序列 [batch_size, seq_len]

        Returns:
            torch.Tensor: 嵌入向量 [batch_size, seq_len, d_model]
        """
        # 嵌入并缩放
        return self.embed(src) * math.sqrt(self.d_model)


class TransformerEncoder(nn.Module):
    """Transformer编码器

    基于PyTorch的TransformerEncoder实现，支持自定义配置。
    """

    def __init__(self, d_model: int, num_heads: int, d_hidden: int,
                 num_layers: int, dropout: float = 0.1,
                 norm_first: bool = True, activation: str = 'gelu'):
        """初始化Transformer编码器

        Args:
            d_model: 模型维度
            num_heads: 多头注意力头数
            d_hidden: 前馈网络隐藏层维度
            num_layers: 编码器层数
            dropout: Dropout率
            norm_first: 是否先进行层归一化
            activation: 激活函数类型
        """
        super().__init__()

        # 创建编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_hidden,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        # 创建层归一化
        encoder_norm = nn.LayerNorm(d_model)

        # 创建Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=encoder_norm
        )

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            padding_mask: 填充掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 编码后的张量 [batch_size, seq_len, d_model]
        """
        return self.transformer_encoder(x, src_key_padding_mask=padding_mask)


class BaseTransformerModel(nn.Module):
    """基础Transformer模型

    包含符号嵌入、位置编码和Transformer编码器的完整模型。
    """

    def __init__(self, num_tokens: int, d_model: int, num_heads: int,
                 d_hidden: int, num_layers: int, dropout: float = 0.1,
                 norm_first: bool = True, activation: str = 'gelu',
                 init_range: float = 0.1):
        """初始化基础Transformer模型

        Args:
            num_tokens: 词汇表大小
            d_model: 模型维度
            num_heads: 多头注意力头数
            d_hidden: 前馈网络隐藏层维度
            num_layers: 编码器层数
            dropout: Dropout率
            norm_first: 是否先进行层归一化
            activation: 激活函数类型
            init_range: 嵌入层初始化范围
        """
        super().__init__()

        self.d_model = d_model
        self.num_tokens = num_tokens

        # 组件初始化
        self.symbol_encoder = SymbolEncoder(num_tokens, d_model, init_range)
        self.positional_encoder = PositionalEncoder(d_model)
        self.transformer_encoder = TransformerEncoder(
            d_model, num_heads, d_hidden, num_layers,
            dropout, norm_first, activation
        )
        # from transformers import AutoModel
        # model_name = "seyonec/ChemBERTa-zinc-base-v1"
        # self.bert = AutoModel.from_pretrained(model_name)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # return self.bert(src, padding_mask).last_hidden_state
        """前向传播

        Args:
            src: 输入符号ID序列 [batch_size, seq_len]
            padding_mask: 填充掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 编码后的表示 [batch_size, seq_len, d_model]
        """
        # 符号嵌入
        x = self.symbol_encoder(src)

        # 添加位置编码
        x = x + self.positional_encoder(x)

        # 应用dropout
        x = self.dropout(x)

        # Transformer编码
        x = self.transformer_encoder(x, padding_mask)

        return x

    def get_embedding_layer(self) -> nn.Embedding:
        """获取嵌入层"""
        return self.symbol_encoder.embed

    def get_encoder_layers(self) -> nn.ModuleList:
        """获取编码器层列表"""
        return self.transformer_encoder.transformer_encoder.layers

    def freeze_parameters(self, freeze_embedding: bool = True,
                          freeze_encoder: bool = True,
                          unfreeze_last_layers: int = 0):
        """冻结模型参数

        Args:
            freeze_embedding: 是否冻结嵌入层
            freeze_encoder: 是否冻结编码器
            unfreeze_last_layers: 解冻最后几层编码器
        """
        if freeze_embedding:
            for param in self.symbol_encoder.parameters():
                param.requires_grad = False

        if freeze_encoder:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

            # 解冻最后几层
            if unfreeze_last_layers > 0:
                encoder_layers = self.get_encoder_layers()
                for layer in encoder_layers[-unfreeze_last_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters()
                               if p.requires_grad)

        return {
            'num_tokens': self.num_tokens,
            'd_model': self.d_model,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }

    def save_pretrained(self, save_path: str):
        """保存预训练模型"""
        torch.save({
            'symbol_encoder': self.symbol_encoder.state_dict(),
            'transformer_encoder': self.transformer_encoder.state_dict(),
            'model_config': {
                'num_tokens': self.num_tokens,
                'd_model': self.d_model
            }
        }, save_path)

    def load_pretrained(self, load_path: str, strict: bool = True):
        """加载预训练模型

        Args:
            load_path: 模型路径
            strict: 是否严格匹配参数名
        """
        checkpoint = torch.load(load_path, map_location='cpu')

        if 'symbol_encoder' in checkpoint:
            self.symbol_encoder.load_state_dict(
                checkpoint['symbol_encoder'], strict=strict
            )

        if 'transformer_encoder' in checkpoint:
            self.transformer_encoder.load_state_dict(
                checkpoint['transformer_encoder'], strict=strict
            )
