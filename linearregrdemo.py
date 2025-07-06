#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/7/6  10:18
# @File:  linearregrdemo.py
# @Project:  pytorch2-study
# @Software:  PyCharm

import torch as t
from matplotlib import pyplot as plt
from IPython import display

# 根据你的硬件环境选择合适的设备
device = t.device('mps' if t.backends.mps.is_available() else 'cpu')  # 对于MPS可用的情况
# 如果你想使用CUDA，可以改为：
# device = t.device('cuda' if t.cuda.is_available() else 'cpu')

t.manual_seed(2021)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1, device=device)
    return x, y

x, y = get_fake_data(batch_size=16)

# 将数据移到CPU并转换为numpy数组
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()

plt.scatter(x_cpu.squeeze(), y_cpu.squeeze())
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Fake Data')
plt.show()