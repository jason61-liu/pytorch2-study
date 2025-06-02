#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/2  13:44
# @File:  test.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch
import resnet
import get_data

import numpy as np

# 获取CIFAR-10数据集
train_dataset, label_dataset, test_dataset, test_label_dataset = (
    get_data.get_CIFAR10_dataset(root="../dataset/cifar-10-batches-py/")
)

# 预处理数据
train_dataset = (
    np.reshape(train_dataset, [len(train_dataset), 3, 32, 32]).astype(np.float32)
    / 255.0
)
test_dataset = (
    np.reshape(test_dataset, [len(test_dataset), 3, 32, 32]).astype(np.float32) / 255.0
)
label_dataset = np.array(label_dataset)
test_label_dataset = np.array(test_label_dataset)

# 选择设备
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# 初始化模型、优化器和损失函数
model = resnet.resnet18()
model = model.to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

batch_size = 128
train_num = len(label_dataset) // batch_size

for epoch in range(63):
    train_loss = 0
    correct_train = 0
    total_train = 0

    # 训练阶段
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = torch.from_numpy(train_dataset[start:end]).to(device)
        y_batch = torch.from_numpy(label_dataset[start:end]).to(device)

        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = pred.max(1)
        total_train += y_batch.size(0)
        correct_train += predicted.eq(y_batch).sum().item()

    train_loss /= train_num
    train_accuracy = correct_train / total_train

    # 测试阶段
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for i in range(0, len(test_label_dataset), batch_size):
            start = i
            end = min(i + batch_size, len(test_label_dataset))
            x_test = torch.from_numpy(test_dataset[start:end]).to(device)
            y_test = torch.from_numpy(test_label_dataset[start:end]).to(device)

            pred = model(x_test)
            _, predicted = pred.max(1)
            total_test += y_test.size(0)
            correct_test += predicted.eq(y_test).sum().item()

    test_accuracy = correct_test / total_test

    print(
        f"epoch: {epoch}, train_loss: {round(train_loss, 2)}, train_accuracy: {round(train_accuracy, 2)}, test_accuracy: {round(test_accuracy, 2)}"
    )
