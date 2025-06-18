#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/31  16:25
# @File:  resnet.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt
from sklearn.model_selection import train_test_split


class MnistNetwork(nn.Module):
    def __init__(self):
        super(MnistNetwork, self).__init__()
        self.convs_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=6, kernel_size=3),
        )
        self.logits_layer = nn.Linear(in_features=1536, out_features=10)

    def forward(self, inputs):
        x = self.convs_stack(inputs)
        x = elt.Rearrange("b c h w -> b (c h w)")(x)
        logits = self.logits_layer(x)
        return logits


def load_and_preprocess_data():
    x_train = np.load("../dataset/mnist/x_train.npy")
    y_train_label = np.load("../dataset/mnist/y_train_label.npy")

    # Normalize the data
    x_train = x_train.astype(np.float32) / 255.0

    # Add a channel dimension
    x_train = np.expand_dims(x_train, axis=1)

    # Split into training and validation sets
    x_train, x_val, y_train_label, y_val_label = train_test_split(
        x_train, y_train_label, test_size=0.1, random_state=42
    )

    return x_train, y_train_label, x_val, y_val_label


model = MnistNetwork()

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
batch_size = 64

x_train, y_train_label, x_val, y_val_label = load_and_preprocess_data()

for epoch in range(42):
    model.train()
    train_num = len(x_train) // batch_size
    train_loss = 0.0
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = torch.tensor(x_train[start:end]).to(device)
        y_batch = torch.tensor(y_train_label[start:end]).to(device)

        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= train_num

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_inputs = torch.tensor(x_val).to(device)
        val_labels = torch.tensor(y_val_label).to(device)
        val_outputs = model(val_inputs)
        val_loss = loss_fn(val_outputs, val_labels)
        _, predicted = torch.max(val_outputs.data, 1)
        total = val_labels.size(0)
        correct = (predicted == val_labels).sum().item()
        accuracy = correct / total

    print(
        f"Epoch [{epoch+1}/42], Train Loss: {train_loss:.4f}, Val Accuracy: {accuracy*100:.2f}%"
    )
