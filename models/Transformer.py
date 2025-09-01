import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
from typing import Optional

# 兼容：支持作为包导入或单文件直接运行
try:  # 相对导入（包内使用）
    from .ModelConfig import ModelConfig
    from .Decoder_Layer import DecoderLayer
    from .RMSNorm import RMSNorm
    from .Attention import precompute_freqs_cis
except ImportError:  # 直接脚本运行（python models/Transformer.py）
    from ModelConfig import ModelConfig
    from Decoder_Layer import DecoderLayer
    from RMSNorm import RMSNorm
    from Attention import precompute_freqs_cis

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel


class Transformer(PreTrainedModel):
    last_loss : Optional[torch.Tensor]
    config_class = ModelConfig

    def __init__(self,args: ModelConfig = None):
        super().__init__(args)
        if args is None:
            args = ModelConfig()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # 嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size,args.dim)
        self.dropout = nn.Dropout(args.dropout)
        # Decoder 堆叠
        self.layers = nn.ModuleList([DecoderLayer(i,args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim,eps=args.norm_eps)
        self.output = nn.Linear(args.dim,args.vocab_size,bias=False)
        # 权重共享 (输出层复用 embedding 权重)
        self.output.weight = self.tok_embeddings.weight
        # 预计算 RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # 初始化
        self.apply(self._init_weights)
        # 残差出口缩放初始化 (MLP w3, 注意力 Wo)
        for pn,p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2 * args.n_layers))
        self.last_loss = None
    def _init_weights(self,module):
        # 函数用于初始化对应的权重
        if isinstance(module,nn.Linear):
            # 对于一般的线性层我们使用正态分布初始化
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            # 如果线性层存在bias的话则将其初始化为0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self,tokens: torch.Tensor,targets:Optional[torch.Tensor]=None,**kwargs)->CausalLMOutputWithPast:
        '''
        - **kwargs:接受更多任意关键字的参数，并且将这些关键字和参数打包成dict放入kwargs供函数内部调用
        '''
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
    # 不把 attention_mask 当作 targets

        # 前向传播
        _bsz,seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        #相对位置嵌入的频率计算
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Decoder层
        for layer in self.layers:
            h = layer(h,freqs_cos,freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:,[-1],:])
            self.last_loss = None
        
        # 标准化输出：使用HF的CausalLMOutputWithPast
        # 若需要 last_loss，可通过 output.loss 或重新计算
        loss = None
        if self.last_loss is not None:
            # 这里返回平均 loss（需要逐token的外部可以改成返回 self.last_loss）
            loss = self.last_loss.mean()
        output = CausalLMOutputWithPast(logits=logits, loss=loss)
        # 兼容旧代码：暴露逐token未归约的 last_loss（形状: tokens数）
        setattr(output, 'last_loss', self.last_loss)
        return output
    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # 前向传播获取序列中最后一个位置的 logits （只看最后一个时间步的logits）
            # logits：最后一层线性层未经过softmax的向量，相当于原始分数，没有被归一化成概率
            logits = self(idx_cond).logits[:, -1, :]
            
            # 从下方的代码中也能看出temperature温度系数的作用：用于对logits缩放
            # 低温度 T < 1: 放大差异，使得分布更尖锐，最大值和其他值差距更大，概率集中在少数几个词上，所以输出更稳定保守，极端情况 T=0,greedy decoding
            # 高温度 T > 1: 缩小差异，使得分布更平缓，压缩logits，不同词间差距减小，概率分布更广，输出更多样化，但很可能胡言乱语，极端情况正无穷时近似均匀分布，输出随机
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token
    
if __name__ == '__main__':
    cfg = ModelConfig()
    model = Transformer(cfg)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)
    x = torch.randint(0,cfg.vocab_size,(1,50)) # 随机生成形状为(1,50)的
    print(f"Random generated x: {x}")
    out = model(x)
    print('logits shape:', out.logits.shape) #(bs,seq_len,vocab_size)
    model.eval()
    gen = model.generate(x[:,:10], max_new_tokens=5)
    print('generated tokens shape:', gen.shape)