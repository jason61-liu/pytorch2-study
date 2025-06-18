#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/1  09:31
# @File:  resnet.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import numpy as np
import torch


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


class ToTensor:
    def __call__(self, inputs, targets):
        inputs = np.reshape(inputs, [28 * 28])
        return torch.tensor(inputs), torch.tensor(targets)


class MNIST_DATASET(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.x_train = np.load("../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../dataset/mnist/y_train_label.npy")

        self.transform = transform

    def __getitem__(self, index):
        image = self.x_train[index]
        label = self.y_train_label[index]

        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.y_train_label)


import torch
import numpy as np

batch_size = 320
epochs = 42

mnist_dataset = MNIST_DATASET(transform=ToTensor())
from torch.utils.data import DataLoader

train_loader = DataLoader(mnist_dataset, batch_size=batch_size)


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
model = model.to(device)
torch.save(model, "./model.pth")
model = torch.compile(model)

loss_fu = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from tensorboardX import SummaryWriter

writer = SummaryWriter()

for epoch in range(epochs):
    train_loss = 0
    for image, label in train_loader:
        train_image = image.to(device)
        train_label = label.to(device)
        pred = model(train_image)
        loss = loss_fu(pred, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / batch_size
    print("epoch: ", epoch, "train_loss:", round(train_loss, 2))
    writer.add_scalars("evl", {"train_loss": train_loss}, epoch)

writer.close()
