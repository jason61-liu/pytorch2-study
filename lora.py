#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: liushiyi
# @File Name: lora.py
# @Date Created: 2025-07-12 09:45:38
# @Description:

import torch as t
from torch import nn
import math


class LoRA(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwarges
    ):
        """_summary_

        Args:
            in_features (int): _description_
            out_features (int): _description_
            lora_rank (int): LoRA的秩，即低秩分解中使用的秩（r）
            lora_alpha (int, optional): LorA的缩放因子，调整低秩矩阵的影响力 Defaults to 1.
            lora_dropout (float, optional): 在应用LorA时使用的dropout的概率 Defaults to 0.0.
        """
        super().__init__(in_features, out_features, **kwarges)
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = float(lora_alpha) / float(lora_rank)

        self.lora_A = nn.Parameter(t.zeros((in_features, lora_rank)))
        self.lora_B = nn.Parameter(t.zeros((lora_rank, out_features)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.weight.requires_grad = False
        self.if_weights_mergerd = False

    def forward(self, x: t.Tensor):
        """_summary_

        Args:
            x (t.Tensor): _description_

        Returns:
            _type_: _description_
        """
        if not self.if_weights_mergerd:
            result = nn.Linear(x, self.weight, bia=self.bias)
            lora_output = self.lora_dropout(x) @ self.lora_A
            lora_output = lora_output @ self.lora_B
            result += lora_output * self.scaling
            return result
        else:
            return nn.Linear(x, self.weight, bias=self.bias)
