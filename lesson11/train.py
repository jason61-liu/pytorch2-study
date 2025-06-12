#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/8  12:24
# @File:  train.py
# @Project:  pytorchdemo
# @Software:  PyCharm


import torch
from torch.utils.data import Dataset, DataLoader
import model
from tqdm import tqdm
import os

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
model = model.ModelSimple().to(device)
model = torch.compile(model)  # 如果 PyTorch >= 2.0

# 优化器 & 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_func = torch.nn.CrossEntropyLoss()

# 加载数据
import get_data

token_list = get_data.token_list
labels = get_data.labels
dev_list = get_data.dev_list
dev_labels = get_data.dev_labels


# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取单条 input_ids 和 label，并转为 tensor
        input_ids = torch.tensor(self.encodings[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"input_ids": input_ids, "labels": label}


# 转换为 TensorDataset
train_dataset = TextDataset(token_list, labels)
val_dataset = TextDataset(dev_list, dev_labels)

# 创建 DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 存储最佳准确率
best_acc = 0.0
save_path = "./best_model.pth"

# 开始训练
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        batch_labels = batch["labels"].to(device)

        optimizer.zero_grad()
        pred = model(input_ids)
        loss = loss_func(pred, batch_labels.long())
        loss.backward()
        optimizer.step()

        # 累计损失和准确率
        train_loss += loss.item()
        preds = torch.argmax(pred, dim=-1)
        correct = (preds == batch_labels.to(device)).sum().item()
        train_correct += correct / batch_size

        # 更新进度条提示
        progress_bar.set_postfix(loss=loss.item(), acc=correct / batch_size)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / len(train_loader)

    # 验证阶段
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            pred = model(input_ids)
            preds = torch.argmax(pred, dim=-1)
            val_correct += (preds == batch_labels.to(device)).sum().item()

    val_acc = val_correct / len(val_dataset)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}\n{'-' * 30}")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with accuracy: {best_acc:.4f}")
