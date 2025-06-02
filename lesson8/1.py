#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/2  14:34
# @File:  1.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import csv


def printall():
    agnews_train = csv.reader(open("../dataset/ag_news/dataset/train.csv", "r"))
    for line in agnews_train:
        print(line)


import re


def text_clear(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    text = text.split()
    return text


import numpy as np
from gensim.models import word2vec


def splitnews():
    agnews_label = []
    agnews_title = []
    agnews_text = []
    agnews_train = csv.reader(open("../dataset/ag_news/dataset/train.csv", "r"))
    for line in agnews_train:
        agnews_label.append(np.float32(line[0]))
        agnews_title.append(text_clear(line[1]))
        agnews_text.append(text_clear(line[2]))

    print("start train...")
    model = word2vec.Word2Vec(
        agnews_text, vector_size=64, min_count=0, window=5, epochs=128
    )
    model_name = "corpusWord2Vec.bin"
    model.save(model_name)

    model = word2vec.Word2Vec.load("./corpusWord2Vec.bin")
    model.train(agnews_title, epochs=model.epochs, total_examples=model.corpus_count)


if __name__ == "__main__":
    splitnews()
