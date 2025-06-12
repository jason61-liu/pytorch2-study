#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/8  16:10
# @File:  one.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch


import torch


def create_self_mask(from_tensor, to_tensor=None):
    """
    创建自注意力 mask，忽略 pad (token_id=0)

    Args:
        from_tensor: [B, L], 输入 token IDs
        to_tensor: 可选，用于对齐 shape，如果提供则 shape 必须与 from_tensor 相同

    Returns:
        mask: [B, 1, L, L]
    """
    if to_tensor is None:
        to_tensor = from_tensor
    assert (
        from_tensor.shape == to_tensor.shape
    ), "from_tensor and to_tensor must have the same shape"

    batch_size, seq_len = from_tensor.shape

    # 创建 mask: 0 表示 padding token
    mask = torch.not_equal(from_tensor, 0)  # [B, L]
    mask = mask.unsqueeze(1)  # [B, 1, L]
    mask = mask.unsqueeze(2)  # [B, 1, L, 1]
    mask = mask.expand(-1, -1, -1, seq_len)  # [B, 1, L, L]

    return mask


# 示例输入：batch_size=2, seq_len=4
from_tensor = torch.tensor([[101, 2345, 6789, 102], [101, 4567, 0, 0]])

# 创建 mask
mask = create_self_mask(from_tensor)

print(mask.shape)  # 应输出: torch.Size([2, 1, 4, 4])
print(mask)
