#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/2  16:57
# @File:  rnn.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import numpy as np

max_length = 80
labels = []
context = []
vocab = set()

with open("../dataset/cn/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(",")
        if int(line[0]) == 0:
            labels.append(0)
        else:
            labels.append(1)
        text = "".join(line[1:])
        context.append(text)
        for char in text:
            vocab.add(char)
vocab_list = list(sorted(vocab))

token_list = []
for text in context:
    token = [vocab_list.index(char) for char in text]
    token = token[:max_length] + [0] * (max_length - len(token))
    token_list.append(token)

seed = 17
np.random.seed(seed)
np.random.shuffle(token_list)
np.random.seed(seed)
np.random.shuffle(labels)

dev_list = np.array(token_list[:170])
dev_labels = np.array(labels[:170])
token_list = np.array(token_list[170:])
labels = np.array(labels[170:])

import torch


class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.embedding_table = torch.nn.Embedding(vocab_size, embedding_dim=312)
        self.gru = torch.nn.GRU(312, 256)
        self.batch_norm = torch.nn.LayerNorm(256, 256)

        self.gru2 = torch.nn.GRU(256, 128, bidirectional=True)

    def forward(self, token):
        token_inputs = token
        embedding = self.embedding_table(token_inputs)
        gru_out, _ = self.gru(embedding)
        embedding = self.batch_norm(gru_out)
        out, hidden = self.gru2(embedding)
        return out


def get_model(vocab_size=len(vocab_list), max_length=max_length):
    model = torch.nn.Sequential(
        RNNModel(vocab_size),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * max_length * 128, 2),
    )
    return model


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

model = get_model().to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

loss_func = torch.nn.CrossEntropyLoss()
batch_size = 128
train_length = len(labels)

for epoch in range(21):
    train_num = train_length // batch_size
    train_loss, train_correct = 0, 0

    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_input_ids = torch.tensor(token_list[start:end]).to(device)
        batch_labels = torch.tensor(labels[start:end]).to(device)

        pred = model(batch_input_ids)
        loss = loss_func(pred, batch_labels.type(torch.uint8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (torch.argmax(pred, dim=-1) == (batch_labels)).type(
            torch.float
        ).sum().item() / len(batch_labels)

    train_loss /= train_num
    train_correct /= train_num
    print("train_loss:", train_loss, "train_correct:", train_correct)

    test_pred = model(torch.tensor(dev_list).to(device))
    correct = (
        torch.argmax(test_pred, dim=-1) == (torch.tensor(dev_labels).to(device))
    ).type(torch.float).sum().item() / len(test_pred)
    print("test_acc:", correct)

    print("----------------------------")
