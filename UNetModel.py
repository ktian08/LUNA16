import numpy as np
import csv
import os
import re
import pandas as pd
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

"""
Begin training/evaluation of model.
"""


class UNetDBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetDBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_c)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.batch(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.batch(x)
        x = self.pool(x)
        return x


class UNetUBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetUBlock, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_c, in_c, 2, stride=2)
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.batch(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.batch(x)
        return x


class UNetBBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetBBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm3d(out_c)
        self.dropout = nn.Dropout3d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch(x)
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        initf = 32
        self.d1 = UNetDBlock(1, initf)
        self.d2 = UNetDBlock(initf, initf * 2)
        self.d3 = UNetDBlock(initf * 2, initf * 4)
        self.d4 = UNetDBlock(initf * 4, initf * 8)
        self.b = UNetBBlock(initf * 8, initf * 16)
        self.u1 = UNetUBlock(initf * 16, initf * 8)
        self.u2 = UNetUBlock(initf * 8, initf * 4)
        self.u3 = UNetUBlock(initf * 4, initf * 2)
        self.u4 = UNetUBlock(initf * 2, initf)
        self.final = nn.Conv3d(initf, 1, kernel_size=1)
        self.final_sig = nn.Sigmoid()

    def forward(self, x):  # D * H * W
        x = self.d1.forward(x)  # D * 512 * 512 -> D/2 * 256 * 256
        x = self.d2.forward(x)  # D/2 * 256 * 256 -> D/4 * 128 * 128
        x = self.d3.forward(x)  # D/4 * 128 * 128 -> D/8 * 64 * 64
        x = self.d4.forward(x)  # D/8 * 64 * 64 -> D/16 * 32 * 32

        x = self.b.forward(x)  # D/16 * 32 * 32 -> D/16 * 32 * 32

        x = self.u1.forward(x)  # D/16 * 32 * 32 -> D/8 * 64 * 64
        x = self.u2.forward(x)  # D/8 * 64 * 64 -> D/4 * 128 * 128
        x = self.u3.forward(x)  # D/4 * 128 * 128 -> D/2 * 256 * 256
        x = self.u4.forward(x)  # D/2 * 256 * 256 -> D * 512 * 512

        x = self.final(x)  # D * 512 * 512 -> D * 512 * 512 (1 channel)
        x = self.final_sig(x)  # D * 512 * 512 -> D * 512 * 512
        return x
