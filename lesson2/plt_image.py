#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/28  22:53
# @File:  plt_image.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import numpy as np
import matplotlib.pyplot as plt
import torch

grid_size = 5
x_train = np.load("../dataset/mnist/x_train.npy")
image = torch.tensor(x_train[7]).to("mps")

image = image
print(image.shape)

image = image.cpu().numpy()
plt.imshow(image)
plt.savefig("./img/img.jpg")
plt.show()
