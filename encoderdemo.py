#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: liushiyi
# @File Name: encoderdemo.py
# @Date Created: 2025-07-09 22:43:48
# @Description:


import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TonkenEmbedding(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, vocab_size, emb_size):
        """_summary_

        Args:
            vocab_size (_type_): _description_
            emb_size (_type_): _description_
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, emb_size, dropout, maxlen=200):
        super().__init__()
        den = torch.exp(
            -torch.arange(0, emb_size, 2).float() * math.log(100) / emb_size
        )
        pos = torch.arange(0, maxlen).float().reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        """_summary_

        Args:
            token_embedding (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class PoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_encoder_layer=4,
        emb_size=512,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.src_tok_emb = TonkenEmbedding(vocab_size=vocab_size, emb_size=emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size=emb_size, dropout=dropout
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size, nhead=8, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layer
        )
        self.generator = nn.Linear(emb_size, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, src_padding_mask):
        """_summary_

        Args:
            src (_type_): 输入序列
            src_mask (_type_): 输入序列的掩码
            src_padding_mask (_type_): 输入序列的padding部分掩码

        Returns:
            _type_: _description_
        """
        src_emb = self.src_tok_emb(src)
        src_emb = self.positional_encoding(src_emb)
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        logit = self.generator(memory)
        return memory, logit
