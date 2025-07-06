#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/7/5  23:42
# @File:  cifar-10demo.py
# @Project:  pytorch2-study
# @Software:  PyCharm

import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show=ToPILImage()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

trainset=tv.datasets.CIFAR10(
    root='./model/',
    train=True,
    download=False,
    transform=transform
)

trainloader=t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

testset=tv.datasets.CIFAR10(
    './model/',
    train=False,
    download=False,
    transform=transform
)

testloader=t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

classes=('plane','car','bird','cat','deer',
         'dog','frog','horse','ship','truck')

# (data,label)=trainset[100]
# print(classes[label])
# show((data+1)/2).resize((100,100))
#
# dataiter=iter(trainloader)
# images,labels=dataiter.next()
# print(''.join('%11s'%classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images+1)/2)).resize((400,100))