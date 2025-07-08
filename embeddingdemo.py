#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: liushiyi
# @File Name: main.py
# @Date Created: 2025-07-08 22:48:43
# @Description:


import torch as t
from torch import nn

def testEmbedding():
    embedding=nn.Embedding(10,2)
    input=t.arange(0,6).view(3,2).long()
    output=embedding(input)
    print(output)
    print(embedding.weight.size())


def main():
    print("just main test")
    testEmbedding()

if __name__ == "__main__":
    main()