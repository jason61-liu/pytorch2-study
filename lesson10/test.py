#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/2  19:46
# @File:  test.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import torch as np
import torch
import attention_model
import get_data


max_length = 64
from tqdm import tqdm

char_vocab_size = 4462
pinyin_vocab_size = 1153


def get_model(embedding_dim=312):
    model = torch.nn.Sequential(
        attention_model.Encoder(pinyin_vocab_size, max_length=max_length),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(embedding_dim, char_vocab_size),
    )
    return model


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

model = get_model().to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_func = torch.nn.CrossEntropyLoss()

pinyin_vocab, hanzi_vocab, pinyin_token_ids, hanzi_token_ids = get_data.get_dataset()

batch_size = 32
train_length = len(pinyin_token_ids)

for epoch in range(21):
    train_num = train_length // batch_size
    train_loss, train_correct = [], []

    for i in tqdm(range((train_num))):
        model.zero_grad()
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_input_ids = torch.tensor(pinyin_token_ids[start:end]).int().to(device)
        batch_labels = torch.tensor(hanzi_token_ids[start:end]).to(device)
        pred = model(batch_input_ids)
        batch_labels = batch_labels.to(torch.uint8)
        active_loss = batch_labels.gt(0).view(-1) == 1

        loss = loss_func(
            pred.view(-1, char_vocab_size)[active_loss],
            batch_labels.view(-1)[active_loss],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        state = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, "./modelpara.pt")
