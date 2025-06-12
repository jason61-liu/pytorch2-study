#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/8  12:12
# @File:  model.py
# @Project:  pytorchdemo
# @Software:  PyCharm
import torch
import torch.utils.data as Data
from transformers import BertModel
from transformers import BertTokenizer
from transformers.models.auto.auto_factory import auto_class_update


class ModelSimple(torch.nn.Module):
    def __init__(self, pretrain_model_name="../model/bert-base-chinese/"):
        super().__init__()
        self.pretrain_model = BertModel.from_pretrained(pretrain_model_name)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids):
        with torch.no_grad():
            output = self.pretrain_model(input_ids=input_ids)
        output = self.fc(output[0][:, 0])
        output = output.softmax(dim=1)
        return output


class Model(torch.nn.Module):
    def __init__(self, pretrain_model_name="../model/bert-base-chinese/"):
        super().__init__()
        self.pretrain_model = BertModel.from_pretrained(pretrain_model_name)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            output = self.pretrain_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            output = self.fc(output[0][:, 0])
            ouput = output.softmax(dim=1)
            return ouput
