# coding=utf-8

from torch import nn


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)   # 输入和输出都是1维

    def forward(self, x):
        out = self.linear(x)
        return out



