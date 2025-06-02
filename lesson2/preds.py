#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/28  23:03
# @File:  preds.py.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch
import numpy as np
import unet
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 320
epochs = 1024

device = torch.device("mps" if torch.mps.is_available() else "cpu")

model = unet.Unet()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")

x_train_batch = []
for i in range(len(y_train_label)):
    if y_train_label[i] <= 10:
        x_train_batch.append(x_train[i])

x_train = np.reshape(x_train_batch, [-1, 1, 28, 28])
x_train /= 512.0

image = x_train[np.random.randint(28)]
image = np.reshape(image, [28, 28])
plt.imshow(image)
plt.show()


state_dict = torch.load("./saver/unet.pth", map_location=torch.device("mps"))
model.load_state_dict(state_dict)
model.eval()

# 将图像转换为Tensor并移动到设备
image_tensor = torch.tensor(image).float().unsqueeze(0).unsqueeze(0).to(device)

# 通过模型传递图像
with torch.no_grad():
    img = model(image_tensor)

# 处理输出图像
img = torch.squeeze(img).detach().cpu().numpy()
plt.imshow(img)
plt.show()
