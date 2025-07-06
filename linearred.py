#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/7/6  10:25
# @File:  linearred.py
# @Project:  pytorch2-study
# @Software:  PyCharm
import numpy as np
import torch as t
from matplotlib import pyplot as plt
from IPython import display

device = t.device('mps' if t.backends.mps.is_available() else 'cpu')

t.manual_seed(2021)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1, device=device)
    return x, y

w = t.rand(1, 1, device=device,requires_grad=True)
b = t.zeros(1, 1, device=device,requires_grad=True)

lr = 0.02

losses=np.zeros(500)

for i in range(500):
    x, y = get_fake_data(batch_size=4)

    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[i]=loss.item()

    # 反向传播，自动计算梯度
    loss.backward()
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)
    w.grad.data.zero_()
    b.grad.data.zero_()

    # 手动计算梯度
    # dy_pred = y_pred - y
    # dw = x.t().mm(dy_pred)
    # db = dy_pred.sum()
    # w.sub_(lr * dw)
    # b.sub_(lr * db)

    if i % 50 == 0:
        display.clear_output(wait=True)
        x_plot = t.arange(0, 6).float().view(-1, 1).to(device)
        y_plot = x_plot.mm(w) + b.expand_as(x_plot)
        plt.plot(x_plot.cpu().detach().numpy(), y_plot.cpu().detach().numpy())

        x2, y2 = get_fake_data(batch_size=22)
        plt.scatter(x2.cpu().numpy(), y2.cpu().numpy())
        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print(f'w: {w.item():.3f}, b: {b.item():.3f}')
plt.plot(losses)
plt.ylim(5,50)
