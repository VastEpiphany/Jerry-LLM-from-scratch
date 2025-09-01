import argparse
import torch

def _default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jerry's LLM Pretraining")

    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="output_models",help="模型的输出目录")
    parser.add_argument("--epochs", type=int, default="1",help="训练轮数（大模型默认轮数为1）")
    parser.add_argument("--lr", type=float, default="2e-4",help="学习率")
    parser.add_argument("--batch_size", type=int, default=1,help="批次大小")
    parser.add_argument("--device", type=str, default=_default_device(),help="训练使用的设备")
    parser.add_argument("--dtype", type=str, default="auto", help="数据类型: auto|float16|bfloat16|float32")

    # 实验跟踪同数据加载
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载的工作进程数 (过大会占用很多RAM)")
    parser.add_argument("--data_path", type=str, default="./data/seq_monkey_split.jsonl", help="预训练训练数据路径")

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数 (增大它, 可在保持显存的同时扩大有效批次)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代数")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam优化器的epsilon参数")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")

    # 日志同保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 多GPU训练参数
    parser.add_argument("--gpus",type=str,default='0',help="多卡并行训练的GPU id")


    return parser

def parse_args(argv=None, overrides: dict | None = None):
    """
    argv: 传入一个列表来模拟命令行（None 则使用 sys.argv[1:]）
    overrides: 在解析前先改默认值（代码内改参数），命令行依然可以二次覆盖
    """
    parser = build_parser()
    if overrides:
        parser.set_defaults(**overrides)   # 代码内改默认
    args = parser.parse_args(argv)         # 命令行最终覆盖
    return args