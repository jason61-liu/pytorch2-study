#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/30  22:27
# @File:  three.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
X_train = np.random.rand(100, 1) * 10
print(X_train)

print("-----------------")
y_train = 2 * X_train + 1 + np.random.randn(100, 1) * 0.5

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
print(X_train_tensor)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()
criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}")

print("weight:", model.linear.weight.item())
print("bias:", model.linear.bias.item())

import matplotlib.pyplot as plt

predicted = model(X_train_tensor).detach().numpy()

plt.scatter(X_train, y_train, label="Actual data")
plt.plot(X_train, predicted, color="red", label="Predicted line")
plt.title("Linear Regression with Mean Squared Error")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
