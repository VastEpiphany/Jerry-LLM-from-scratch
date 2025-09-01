import torch
from torch import nn
import argparse

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps = eps # 坟墓中的小常数，防止除以0
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习参数权重，全初始化为1

    def _norm(self,x):
        # RMS计算公式
        # rsqrt为计算倒数平方根，mean(-1,keepdim=True)为计算均值 （最后一个维度）
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

    def forward(self,x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight # weight就是RMS公式中的可学习参数gamma
    
if __name__ == '__main__':
    '''
    测试：输入输出维度需一致，因为归一化并不会改变我们张量的形状
    '''
    norm = RMSNorm(dim=1024, eps=1e-6)
    x = torch.ones([3,256,1024])
    out = norm(x)
    print(out.shape) # torch.Size([3, 256, 1024])