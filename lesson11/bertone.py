#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/8  10:42
# @File:  bertone.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import torch
from transformers import BertTokenizer
from transformers import BertModel

# 直接下载
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# pretrain_model = BertModel.from_pretrained("bert-base-chinese")

# 通过modelscope download --model google-bert/bert-base-chinese --local_dir=./bert-base-chinese

local_model_path = "../model/bert-base-chinese/"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)


tokens = tokenizer.encode(
    "春眠不觉晓", max_length=12, padding="max_length", truncation=True
)
print(tokens)
print("================================")


print(tokenizer("春眠不觉晓", max_length=12, padding="max_length", truncation=True))
print("-------------------")
tokens = torch.tensor([tokens]).int()
print(model(tokens))
