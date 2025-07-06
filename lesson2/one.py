#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/7/3  23:34
# @File:  one.py
# @Project:  pytorchstudy
# @Software:  PyCharm

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        #  卷积-->激活-->池化
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)

        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


if __name__=='__main__':
    net=Net()
    # print(net)
    # for name,parameters in net.named_parameters():
    #     print(name,':',parameters.size())

    input = t.randn(1,1,32,32)
    out=net(input)
    target=t.arange(0,10).view(1,10).float()
    criterion=nn.MSELoss()
    loss=criterion(out,target)
    # print(loss)

    # net.zero_grad()
    # print("反向传播之前 conv1.bias梯度：")
    # print(net.conv1.bias.grad)
    # loss.backward()
    # print('反向传播之后 conv1.bias的梯度')
    # print(net.conv1.bias.grad)
    optimizer=optim.SGD(net.parameters(),lr=0.01)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
