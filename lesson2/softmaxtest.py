#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/28  20:34
# @File:  softmaxtest.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import math
import numpy as np


def softmax(inMatrix):
    m, n = inMatrix.shape
    outMatrix = np.zeros((m, n))
    soft_num = 0
    for idx in range(0, n):
        outMatrix[0, idx] = math.exp(inMatrix[0, idx])
        soft_num += outMatrix[0, idx]

    for idx in range(0, n):
        outMatrix[0, idx] = outMatrix[0, idx] / soft_num
    return outMatrix


a = np.array([[1, 2, 1, 2, 1, 1, 3]])
print(softmax(a))
