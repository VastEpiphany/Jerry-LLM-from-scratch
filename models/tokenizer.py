'''
This tokenizer.py aims to implement a BPE-based tokenizer by using "出门问问序列猴子" open source dataset.
'''
import random
import json
import os
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer
)
from tokenizers.normalizers import NFKC
from typing import Generator
import tarfile
from pathlib import Path

def read_texts_from_jsonl(file_path: str) -> Generator[str,None,None]:
    '''
    读取数据集中的jsonl文件并且提取文本数据，主要加了一些预防措施
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f,1): # enumerate(f,1)会遍历f,每次返回一个tuple (行号，行内容)
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecoderError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(f"Unexpected KeyError: {e}")
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False, # 是否在编码时自动添加开头标记（不需要，因为下面我们的模版已经有了）
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>", # decoder-only常用eos作为pad，但必须在训练时的labels中忽略padding，否则会让模型学了预测一堆eos
        "unk_token": "<unk>", # Out-Of-Vocabulary token specifier，doesn't exist if byte-level is used
        "model_max_length": 1000000000000000019884624838656, # 相当于一个超大的哨兵值，基本等于不要在tokenizer级别主动截断或者发送警告
        "clean_up_tokenization_spaces": False, # 解码时不做标点前空格等清理
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件 tokenizer_config.json
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def extract_archive(archive_path: str, extract_dir: str) -> str:
    """解压 .tar.bz2 数据集到指定目录。返回解压后的主目录路径。

    如果已经解压过则跳过。
    """
    archive_path = os.path.abspath(archive_path)
    extract_dir = os.path.abspath(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    flag_file = os.path.join(extract_dir, ".extracted.done")
    if os.path.exists(flag_file):
        return extract_dir
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"找不到数据压缩包: {archive_path}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(extract_dir)
    Path(flag_file).touch()
    return extract_dir

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # 初始化tokenizer
    tokenizer = Tokenizer(model=models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC() # 文本归一化 （）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 特殊token的设置
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size,
        min_frequency = 2,
        special_tokens = special_tokens,
        show_progress = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练tokenizer
    print(f"Training Tokenizer with data from {data_path}...")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer,length=os.path.getsize(data_path))

    # 验证特殊token的映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # 保存对应训练完的tokenizer到对应json
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # 配置文件的创建
    create_tokenizer_config(save_dir)
    print(f'Tokenizer config files created in {save_dir}')

    
def eval_tokenizer(tokenizer_path: str) -> None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # 加载预定义好的tokenizer
        except Exception as e:
            print("Error loading tokenizer:", e)
            raise

        print("\n=== Tokenizer基本信息 ===")
        print(f'Vocab Size: {len(tokenizer)}')
        print(f'Special Tokens: {tokenizer.all_special_tokens}')
        print(f'Special Tokens IDs: {tokenizer.all_special_ids}')

        # 测试聊天模板
        messages = [
            {"role": "system", "content": "你是一个AI助手。"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thank you. and you?"},
            {"role": "user", "content": "I'm good too."},
            {"role": "assistant", "content": "That's great to hear!"},
        ]
    
        print("\n=== 聊天模板测试 ===")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
        )
        print("Generated prompt:\n", prompt, sep="")

         # 测试编码解码
        print("\n=== 编码解码测试 ===")
        encoded = tokenizer(prompt,truncation=True,max_length=256)
        decoded = tokenizer.decode(encoded["input_ids"],skip_special_tokens=False)
        print("Decoded text matches original:", decoded == prompt)
        

if __name__ == '__main__':
    #  train_tokenizer(
    #      #data_path="./data/mobvoi_seq_monkey_general_open_corpus.jsonl",
    #      data_path="./data/subset_stream_5k/train.jsonl",
    #      save_dir="./tokenizer",
    #      vocab_size=8192
    #  )

    eval_tokenizer("./Tokenizer")
