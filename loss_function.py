# coding=utf-8


from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable


# 定义自己的损失函数, 由三项组成, 权重分别为w1, w2, w3
class myLoss(nn.Module):
    def __init__(self, w1, w2, w3):
        super(myLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, input, target, named_parameters):
        # 加入正则项, pytorch 自动求导要求从输入到输出全用Variable, 不能使用numpy或者Tensor
        res = Variable(torch.FloatTensor([0]))
        for name, param in named_parameters:
            if param.requires_grad:
                # print(name, param)
                # res += torch.squeeze(param)[0] ** 2
                res.add_(torch.squeeze(param)[0] ** 2)

        # 先按行求平均再按列取平均, 这个函数返回的是标量
        return self.w1 * torch.mean(torch.mean((input - target) ** 2, dim=1), dim=0) \
                + self.w2 * F.mse_loss(input, target) \
                + self.w3 * res


