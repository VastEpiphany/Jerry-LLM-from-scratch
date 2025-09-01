import torch
import torch.nn as nn

# 兼容：既支持包内相对导入，也支持单文件直接运行
try:
    from .ModelConfig import ModelConfig
    from .MLP import MLP
    from .RMSNorm import RMSNorm
    from .Attention import Attention, precompute_freqs_cis
except ImportError:
    from ModelConfig import ModelConfig
    from MLP import MLP
    from RMSNorm import RMSNorm
    from Attention import Attention, precompute_freqs_cis


class DecoderLayer(nn.Module):
    def __init__(self,layer_id: int,args:ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id # 定义层id
        # 因为RMSNorm有可学习的参数，所以要定义两个供使用
        self.attention_norm = RMSNorm(args.dim,eps=args.norm_eps) # 定义用于attention前的RMSNorm
        self.ffn_norm = RMSNorm(args.dim,eps=args.norm_eps) # 定义用于MLP前的RMSNorm

    def forward(self,x,freqs_cos,freqs_sin):
        h1 = self.attention(self.attention_norm(x),freqs_cos,freqs_sin) + x
        out = self.feed_forward(self.ffn_norm(h1)) + h1
        return out
    
if __name__ == '__main__':
    args = ModelConfig()
    dec_l = DecoderLayer(1,args)

    dim = args.dim
    seq_len = 50
    x = torch.randn(1,seq_len,dim)
    freqs_cos,freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
    out = dec_l(x,freqs_cos,freqs_sin)
    print(out.shape) # torch.Size([1, 50, 768]) 应该同输入一致