#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/8  11:40
# @File:  get_data.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import numpy as np
from transformers import BertTokenizer


local_model_path = "../model/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(local_model_path)

max_length = 80
labels = []
context = []
token_list = []


with open("../dataset/cn/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(",")
        if int(line[0]) == 0:
            labels.append(0)
        else:
            labels.append(1)
        text = "".join(line[1:])
        token = tokenizer.encode(
            text, max_length=max_length, padding="max_length", truncation=True
        )
        token_list.append(token)
        context.append(text)

print(len(labels))
print(len(token_list))

seed = 828
np.random.seed(seed)
np.random.shuffle(token_list)
np.random.seed(seed)
np.random.shuffle(labels)

# print(type(token_list))
# print(token_list[1])

# print(len(token_list))

# 验证集
dev_list = np.array(token_list[:170]).astype(int)
dev_labels = np.array(labels[:170]).astype(int)

# print(dev_list)

# 训练集
token_list = np.array(token_list[170:]).astype(int)
labels = np.array(labels[170:]).astype(int)
