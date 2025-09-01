from torch.utils.data import Dataset
import json
import numpy as np
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        '''
        支持obj[index] 操作
        '''
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 计算没满max_length的剩余部分,去进行padding操作
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len 
        loss_mask = [1] * text_len + [0] * padding_len # 0 表示不计算对应的损失

        input_id = np.array(input_id) # e.g. [BOS, T1, T2, T3, T4, T5, T6, T7, EOS]
        X = np.array(input_id[:-1]).astype(np.int64) # 输入x相当于input序列去掉最后一个tok e.g. [BOS, T1, T2, T3, T4, T5, T6, T7]
        Y = np.array(input_id[1:]).astype(np.int64)  # 目标Y相当于input序列去掉开头的第一个tok e.g.[T1, T2, T3, T4, T5, T6, T7, EOS]
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)
    

class SFTDataset(Dataset):
    '''
    SFT 任务监督微调数据集调整
    我们的SFTDataset是一个多轮对话数据集，目标是让模型学会如何进行多轮对话，
    SFT阶段输入是上一轮对话内容，输出是当前轮对话内容

    Modification: 之前的硬编码实在是不好，因为每个人的tokenizer很可能不一样，
    现在我们在 SFTDataset.__init__ 中用 tokenizer 对 <|im_start|>assistant\n 与 <|im_start|>assistant 做编码，收集得到的 token 序列作为候选前缀（去重）。
    若全部失败，兜底使用 bos_token_id。
    generate_mask 中按候选前缀（按长度降序）匹配，找到后向后查第一个 <|im_end|> (id=4)，区间内标 1
    '''
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        # 动态获取 assistant 前缀 token 序列，替换之前硬编码的 [3,1074,537,500,203]
        # 采用含换行和不含换行两个变体，提升鲁棒性
        variants = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
        self.assistant_prefix_candidates = []
        for v in variants:
            try:
                ids = self.tokenizer(v, add_special_tokens=False).input_ids
                if ids and ids not in self.assistant_prefix_candidates:
                    self.assistant_prefix_candidates.append(ids)
            except Exception:
                pass
        if not self.assistant_prefix_candidates:
            # 兜底：至少包含 <|im_start|> token id 后续匹配几乎不会触发，但防止空列表异常
            single_id = getattr(self.tokenizer, 'bos_token_id', None)
            if single_id is not None:
                self.assistant_prefix_candidates.append([single_id])

    def __len__(self):
        return len(self.data)
    
    def generate_mask(self,input_ids):
        '''
        找到所有对话中<|im_start|>assistant\n" … <|im_end|>的区间，然后将这些区间的位置标记为需要计算loss
        '''
        mask = [0] * len(input_ids)
        n = len(input_ids)
        i = 0
        # 依次尝试多个候选前缀（优先匹配更长的）
        candidates = sorted(self.assistant_prefix_candidates, key=len, reverse=True)
        while i < n:
            matched_len = 0
            for cand in candidates:
                L = len(cand)
                if i + L <= n and all(input_ids[i + k] == cand[k] for k in range(L)):
                    matched_len = L
                    break
            if matched_len > 0:
                # 找到assistant开头后，向后找第一个 <|im_end|> (id=4)
                j = None
                for idx in range(i + matched_len, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:  # <|im_end|>
                        j = idx
                        break
                if j is not None:
                    start = i + matched_len
                    end = j
                    for pos in range(start, end + 1):  # 包含 end（含<|im_end|>）
                        if pos < len(mask):
                            mask[pos] = 1
                i += matched_len
            else:
                i += 1
        return mask
    
    def __getitem__(self, index):
        sample = json.loads(self.data[index])
        # 同Pretrain不同：我们的text需要对sample采用chat模版进行
        text = self.tokenizer.apply_chat_template(sample,tokenize=False,add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 计算没满max_length的剩余部分,去进行padding操作
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len 
        loss_mask = self.generate_mask(input_id) # 同Pretrain不同：loss mask需要自行计算

        input_id = np.array(input_id) # e.g. [BOS, T1, T2, T3, T4, T5, T6, T7, EOS]
        X = np.array(input_id[:-1]).astype(np.int64) # 输入x相当于input序列去掉最后一个tok e.g. [BOS, T1, T2, T3, T4, T5, T6, T7]
        Y = np.array(input_id[1:]).astype(np.int64)  # 目标Y相当于input序列去掉开头的第一个tok e.g.[T1, T2, T3, T4, T5, T6, T7, EOS]
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)

