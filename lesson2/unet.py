#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/5/28  22:10
# @File:  unet.py.py
# @Project:  pytorchdemo
# @Software:  PyCharm

import torch


class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.first_block_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.second_block_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.latent_space_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.second_block_up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.first_block_up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.convUP_end = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )

    def forward(self, img_tensor):
        image = img_tensor
        image = self.first_block_down(image)
        print("first_block_down:", format(image.shape))
        image = self.second_block_down(image)
        print("second_block_down:", image.shape)
        image = self.latent_space_block(image)
        print("latent_space_block:", image.shape)

        image = self.second_block_up(image)
        print("second_block_up:", image.shape)
        image = self.first_block_up(image)
        print("first_block_up:", image.shape)
        image = self.convUP_end(image)
        print("convUP_end:", image.shape)

        return image


if __name__ == "__main__":
    image = torch.randn(size=(5, 1, 28, 18))
    unet_model = Unet()(image)
    torch.save(unet_model, "unet_model.pth")
