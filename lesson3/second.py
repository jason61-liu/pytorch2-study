#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/29  23:47
# @File:  get_data.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch
import numpy as np
from tqdm import tqdm

batch_size = 320
epochs = 1024

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 312),
            torch.nn.ReLU(),
            torch.nn.Linear(312, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
print(model)
torch.save(model, "./model.pth")
model = model.to(device)
model = torch.compile(model)
loss_fu = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")
train_num = len(x_train) // batch_size

for epoch in range(20):
    train_loss = 0
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        train_batch = torch.tensor(x_train[start:end]).to(device)
        label_batch = torch.tensor(y_train_label[start:end]).to(device)
        pred = model(train_batch)
        loss = loss_fu(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= train_num
    accuracy = (pred.argmax(1) == label_batch).type(
        torch.float32
    ).sum().item() / batch_size
    print("train_loss:", round(train_loss, 2), "accuracy:", round(accuracy, 2))
