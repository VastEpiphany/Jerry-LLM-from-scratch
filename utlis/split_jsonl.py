"""Stream split a large JSONL dataset into small subsets for tokenizer training.

特点:
1. 单次只读取一行, 不将整个文件载入内存.
2. 两种模式:
   - 直接流式划分 (默认): 按 train_ratio 概率把每行写入 train 或 val, 可限制最大样本数.
   - 水库抽样 (--total-samples): 先对全集做等概率采样 K 行 (仅占用 K 行内存), 再按比例切分.
3. 可生成分片 (shards) 的训练文件, 方便并行或增量训练.
4. 兼容 .jsonl 及 .jsonl.gz (自动判断 gzip).

用法示例:
# 简单按 99%:1% 随机划分 (最多采 200k 行作为学习子集)
python utils/split_jsonl.py \
  --input data/full_dataset.jsonl \
  --output-dir data/subsets \
  --train-ratio 0.99 \
  --max-train-samples 190000 \
  --max-val-samples 10000

# 使用水库抽样先随机采 50k 行再切分 (保证整体均匀随机)
python utils/split_jsonl.py \
  --input data/full_dataset.jsonl \
  --output-dir data/subsets_rs \
  --total-samples 50000 \
  --train-ratio 0.95

# 生成分片 train_00001.jsonl, train_00002.jsonl ... 每片 20000 行
python utils/split_jsonl.py \
  --input data/full_dataset.jsonl \
  --output-dir data/sharded \
  --train-ratio 0.98 \
  --shard-size 20000
"""
from __future__ import annotations
import os
import gzip
import json
import math
import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------- 基础 I/O ---------------------- #

def open_maybe_gzip(path: str, mode: str = 'rt', encoding: str = 'utf-8'):
    """根据扩展名自动选择 gzip.open 或内置 open."""
    if path.endswith('.gz'):
        return gzip.open(path, mode=mode, encoding=encoding)  # type: ignore[arg-type]
    return open(path, mode=mode, encoding=encoding)  # type: ignore[call-arg]

# ---------------------- 水库抽样 ---------------------- #

def reservoir_sample_jsonl(path: str, k: int, seed: int = 42) -> List[str]:
    """对巨大 JSONL 文件进行等概率采样 k 行 (保留原始行文本)。

    说明: 只存储 k 行, 内存占用 ~ O(k * 平均行长度)。
    """
    random.seed(seed)
    sample: List[str] = []
    with open_maybe_gzip(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < k:
                sample.append(line.rstrip('\n'))
            else:
                # 生成 [0, i] 区间整数, 若命中替换
                j = random.randint(0, i)
                if j < k:
                    sample[j] = line.rstrip('\n')
    return sample

# ---------------------- 流式随机划分 ---------------------- #

def stream_split_jsonl(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.98,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    shard_size: Optional[int] = None,
    seed: int = 42,
    val_filename: str = 'val.jsonl',
) -> Tuple[int, int]:
    """单次遍历文件, 按概率写入 train 或 val.

    返回: (train_lines_written, val_lines_written)
    """
    assert 0 < train_ratio < 1, 'train_ratio 必须在 (0,1)'
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 训练文件: 可能需要分片
    shard_idx = 0
    train_count = 0
    val_count = 0
    current_train_fp = None

    def open_new_shard():
        nonlocal shard_idx, current_train_fp
        shard_idx += 1
        if current_train_fp:
            current_train_fp.close()
        shard_name = f'train_{shard_idx:05d}.jsonl' if shard_size else 'train.jsonl'
        current_train_fp = open(os.path.join(output_dir, shard_name), 'w', encoding='utf-8')

    open_new_shard()
    val_fp = open(os.path.join(output_dir, val_filename), 'w', encoding='utf-8')

    try:
        with open_maybe_gzip(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                # 早停: 当两个集合都达到上限
                if max_train_samples is not None and train_count >= max_train_samples:
                    if max_val_samples is None or val_count >= max_val_samples:
                        break
                if max_val_samples is not None and val_count >= max_val_samples:
                    if max_train_samples is None or train_count >= max_train_samples:
                        break

                r = random.random()
                to_train = r < train_ratio
                if to_train:
                    if max_train_samples is not None and train_count >= max_train_samples:
                        to_train = False  # 转给 val
                else:
                    if max_val_samples is not None and val_count >= max_val_samples:
                        to_train = True  # 转给 train

                if to_train:
                    current_train_fp.write(line)
                    train_count += 1
                    if shard_size and train_count % shard_size == 0:
                        open_new_shard()
                else:
                    val_fp.write(line)
                    val_count += 1
    finally:
        if current_train_fp:
            current_train_fp.close()
        val_fp.close()
    return train_count, val_count

# ---------------------- 主逻辑 ---------------------- #

def main():
    parser = argparse.ArgumentParser(description='Split large JSONL into train/val subsets (streaming).')
    parser.add_argument('--input', type=str, required=True, help='原始巨大 JSONL (或 .jsonl.gz) 路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.98, help='训练集比例 (0-1)')
    parser.add_argument('--max-train-samples', type=int, default=None, help='训练集最大样本数 (行数)')
    parser.add_argument('--max-val-samples', type=int, default=None, help='验证集最大样本数 (行数)')
    parser.add_argument('--total-samples', type=int, default=None, help='水库抽样总样本 (启用则覆盖 max 参数 & 仅使用一次抽样)')
    parser.add_argument('--shard-size', type=int, default=None, help='训练集分片大小 (行数), 仅在流式模式下生效')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-filename', type=str, default='val.jsonl')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.total_samples:
        # 水库抽样模式
        reservoir = reservoir_sample_jsonl(args.input, args.total_samples, seed=args.seed)
        random.seed(args.seed)
        random.shuffle(reservoir)
        train_target = int(len(reservoir) * args.train_ratio)
        train_lines = reservoir[:train_target]
        val_lines = reservoir[train_target:]

        train_path = os.path.join(args.output_dir, 'train.jsonl')
        val_path = os.path.join(args.output_dir, args.val_filename)
        with open(train_path, 'w', encoding='utf-8') as tf:
            for l in train_lines:
                tf.write(l + '\n')
        with open(val_path, 'w', encoding='utf-8') as vf:
            for l in val_lines:
                vf.write(l + '\n')
        print(f'[Reservoir] train={len(train_lines)} val={len(val_lines)} -> {args.output_dir}')
    else:
        train_count, val_count = stream_split_jsonl(
            input_path=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            shard_size=args.shard_size,
            seed=args.seed,
            val_filename=args.val_filename,
        )
        print(f'[Stream] train={train_count} val={val_count} -> {args.output_dir}')

if __name__ == '__main__':
    main()
