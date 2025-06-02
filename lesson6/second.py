#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/1  11:10
# @File:  second.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch


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

input_data = torch.rand(5, 784)

from tensorboardX import SummaryWriter

writer = SummaryWriter()

with writer:
    writer.add_graph(model, (input_data,))
