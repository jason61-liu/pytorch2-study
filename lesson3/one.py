#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/29  23:29
# @File:  one.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import numpy as np
import torch


x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")

print(y_train_label[:10])

x = torch.tensor(y_train_label[:5], dtype=torch.int64)
y = torch.nn.functional.one_hot(x, 10)
print(y)


y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss)
