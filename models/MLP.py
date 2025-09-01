import torch
import torch.nn as nn
import argparse

class MLP(nn.Module):
    '''
    LLaMA2的MLP模块实现
    Notice: LLaMA2的MLP采用了门控结构（SwiGLU），相较于传统只有一条线性变换的能让模型更灵活地表达特征（有点类似于residual的思想）

    '''
    def __init__(self,dim: int,hidden_dim: int,multiple_of: int,dropout: float):
        '''
        multiple_of: 指定隐藏层维度成为该参数的整数倍，使得处理张量更快
        '''
        super().__init__()
        # 没制定隐藏层维度，则将其默认设置为输入维度的4倍，然后将其减少至2/3,确保其是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2*hidden_dim/3) # LLaMA的设计，不知道为什么一定要这样做
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)
        self.w3 = nn.Linear(dim,hidden_dim,bias=False)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # 注意这里的w3相当于up_proj，然后w1相当于gate_proj,我们的activation function采用了silu函数
        return self.dropout(self.w2(self.silu(self.w1(x)) * self.w3(x)))
    
if __name__ == '__main__':
    # 简单本地测试：采用示例超参数
    dim = 768
    hidden_dim = 1024
    multiple_of = 4
    dropout = 0.0
    mlp = MLP(dim, hidden_dim, multiple_of, dropout)
    x = torch.ones((2,32,dim))
    print(mlp(x).shape)  # 期望: torch.Size([2, 32, 768])