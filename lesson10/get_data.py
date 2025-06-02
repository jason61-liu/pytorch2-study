# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/2  13:44
# @File:  test.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import os

max_length = 64
pinyin_vocab = set()
hanzi_vocab = set()
pinyin_list = []
hanzi_list = []

file_path = "../dataset/zh.tsv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

with open(file_path, encoding="UTF-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        parts = line.split("\t")
        if len(parts) < 3:
            print(f"Skipping malformed line: {line}")
            continue

        pinyin_sequence = parts[1].strip().split(" ")
        hanzi_sequence = parts[2].strip().split(" ")

        pinyin = ["GO"] + pinyin_sequence + ["END"]
        hanzi = ["GO"] + hanzi_sequence + ["END"]

        for _pinyin, _hanzi in zip(pinyin, hanzi):
            pinyin_vocab.add(_pinyin)
            hanzi_vocab.add(_hanzi)

        # 填充到 max_length
        pinyin += ["PAD"] * (max_length - len(pinyin))
        hanzi += ["PAD"] * (max_length - len(hanzi))

        pinyin_list.append(pinyin)
        hanzi_list.append(hanzi)

# # 打印一些信息以验证结果
# print(f"Number of unique pinyin tokens: {len(pinyin_vocab)}")
# print(f"Number of unique hanzi tokens: {len(hanzi_vocab)}")
# print(f"Number of sequences: {len(pinyin_list)}")
#
# # 示例打印前5个序列
# for i in range(min(5, len(pinyin_list))):
#     print(f"Pinyin sequence {i}: {pinyin_list[i]}")
#     print(f"Hanzi sequence {i}: {hanzi_list[i]}")

from tqdm import tqdm


def get_dataset():
    pinyin_token_ids = []  # 存储拼音序列对应的整数索引
    hanzi_token_ids = []  # 存储汉字序列对应的整数索引

    # 创建字典以便快速查找索引
    pinyin_to_index = {char: idx for idx, char in enumerate(pinyin_vocab)}
    hanzi_to_index = {char: idx for idx, char in enumerate(hanzi_vocab)}

    # 遍历拼音和汉字序列，并显示进度条
    for pinyin, hanzi in zip(tqdm(pinyin_list), hanzi_list):
        # 将拼音序列中的每个字符转换为其在 pinyin_vocab 中的索引
        pinyin_indices = [pinyin_to_index.get(char, -1) for char in pinyin]
        pinyin_token_ids.append(pinyin_indices)

        # 将汉字序列中的每个字符转换为其在 hanzi_vocab 中的索引
        hanzi_indices = [hanzi_to_index.get(char, -1) for char in hanzi]
        hanzi_token_ids.append(hanzi_indices)

    # 返回拼音词汇表、汉字词汇表、拼音整数索引序列和汉字整数索引序列
    return pinyin_vocab, hanzi_vocab, pinyin_token_ids, hanzi_token_ids
