import os
import torch
import math
import time
import logging
import swanlab
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from contextlib import nullcontext
from utlis.pretrain_args import parse_args
from utlis.dataset import PretrainDataset
from models.Transformer import Transformer
from models.ModelConfig import ModelConfig

# 提前初始化日志（避免在 init_model 中使用 Logger 时未定义）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
Logger = logging.getLogger("pretrain").info

def get_lr(cur_it,all_it):
    """
    学习率更新策略函数 （和之前ML Security所采用的差不多）
    1. 在warmup阶段，学习率从0线性增长到target lr
    2. 训练过程中采用余弦退火算法按照cos函数衰减到最小lr
    3. 超出训练步数后保持最小lr不变

    Args:
        cur_it (int): 当前迭代步数
        all_it (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters
    assert 0 <= warmup_iters <= all_it # 否则lr_decay_iters - warmup_iters容易为0
    lr_decay_iters = all_it
    min_lr = args.lr / 10 # 最小学习率，设置为初始学习率的1/10

    # 学习率预热步骤
    if cur_it < warmup_iters:
        return args.lr * cur_it / warmup_iters
    
    # 超出训练步骤，保持最小学习率
    if cur_it > lr_decay_iters:
        return min_lr
    
    # 中间的余弦退火阶段
    decay_ratio = (cur_it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1 # 
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.lr - min_lr)

def train_epoch(epoch):
    """
    训练一个epoch的函数
    
    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存
    
    Args:
        epoch (int): 当前epoch编号
    """
    start_time = time.time()

    # 遍历数据加载出每个batch
    for step, (X,Y,loss_mask) in enumerate(train_loader):
        # 将数据全部转移到对应设备上
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(cur_it=epoch * iter_per_epoch + step, all_it=args.epochs * iter_per_epoch)
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练上下文
        # ATTENTION： Python中的with...相当于自动的try，能在进入代码块前自动做某些准备、退出时清理
        with ctx:
            # 前向传播
            out = model(X,Y)
            # 反向计算损失
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask展平为1维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（注意要忽略padding值）
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        # scaler进行混合精度的反向传播
        scaler.scale(loss).backward()

        # 每到一个accumulation_step进行一次optimizer更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)

            # 执行优化器步骤
            scaler.step(optimizer)
            # 更新scaler缩放因子
            scaler.update()

            # 清空梯度 (set_to_none=True可以节省内存？)
            optimizer.zero_grad(set_to_none=True)

        # 每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # 如果启用SwanLab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            # 处理多卡保存：如果是DataParallel模型，需要访问.module属性
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式
        
        # 每20000步保存一个带步数标记的检查点
        if (step + 1) % 20000 == 0:
            model.eval()
            # 构建带步数的检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # 保存模型状态字典
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()

def init_model():

    def count_params(model):
        '''
        返回对应model中可训练带有反向梯度的参数数量
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./Tokenizer')
    # 加载定义好的LLaMA2模型
    model = Transformer(lm_config)
    # 多卡初始化
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUS with DataParallel...")
        # 将数据通过DP分布式训练
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)

    #计算打印LLM的参数总量
    Logger(f"LLM 参数总量： {count_params(model)/1e6:.3f} 百万")
    return model,tokenizer



if __name__ == '__main__':
    # ============================ 加载参数 =================================
    # 解析命令行参数（只执行一次）
    args = parse_args()

    # ============================ GPU环境设置 =================================
    if args.gpus is not None:
        # 设置可见 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 我们默认设置第一个GPU为主设备
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ============================ 训练过程追踪数据设置 =================================
    if args.use_swanlab:
        # swanlab.login(api_key='my key')
        run = swanlab.init(
            project="Jerry's-Own-LLM",  # 项目名称
            experiment_name="Pretrain-215M",  # 实验名称
            config=args,  # 保存所有超参数
        )
    
    # ============================ 模型的设置 (Transformer架构层面) =================================
    lm_config = ModelConfig(
        dim = 1024,
        n_layers = 18,
        # n_heads = args.model_heads,
        # n_kv_heads = args.model_kv_heads if args.model_kv_heads is not None else args.model_heads,
        # vocab_size = args.model_vocab_size,
        # max_seq_len = args.model_max_seq_len,
        # dropout = args.model_dropout,
        # multiple_of = args.model_multiple_of,
        # flash_attn = args.flash_attn,
    )

    # ============================ 训练环境设置 =================================
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)

    # 确保输出目录已被创建
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置随机树种确保结果的一致性
    torch.manual_seed(18)

    # 确定设备类型并用其选择对应合适的上下文管理器
    # 混合精度训练上下文管理器ctx：cpu时为nullcontext占位符（空的上下文管理器）
    '''
    为何要用混合精度amp.autocast()?
    在大模型预训练里，混合精度 = 该用低精度（FP16/BF16）时就用，提高吞吐、节省显存；但对数值敏感的算子仍保持 FP32，避免不稳定。
    autocast() 就是 自动帮你决定每个算子用什么精度 的上下文：

    对矩阵乘、卷积这类适合 Tensor Cores 的算子 → 低精度（更快、更省显存）；

    对归一化、求和、softmax 等数值更敏感的算子 → 保持 FP32（更稳）。
    这样既快又稳，而且我们就不用手动到处 .half()/.float() 了
    '''
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        try:
            ctx = torch.amp.autocast(device_type=device_type)
        except Exception as e:
            Logger(f"autocast init failed, fallback to nullcontext. Error: {e}")
            ctx = nullcontext()

    # ============================ 模型/数据加载同初始化 =================================
    model,tokenizer = init_model()

    # 训练数据集加载
    train_dataset = PretrainDataset(args.data_path,tokenizer,max_length=max_seq_len)

    # DataLoader的创建
    '''
    何时会使用drop_last:
    - 严格依赖BatchNorm的场景： BatchNorm 需要用“整批数据”统计均值方差。如果最后一批太小，统计量会失真，影响稳定性
    - 分布式训练（DDP）：要求多卡时拿到的数据必须数量一致，否则报错
    '''
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory = True,   # 是否将数据加载到固定内存中，进而加速GPU传输
        drop_last = False,   # 是否丢弃最后一个不完整的批次
        num_workers = args.num_workers
    )

    # ============================ 优化器同训练组件的初始化 =================================
    # 初始化我们的混合精度训练时同时需要用到的梯度缩放器，只有在使用float16/bfloat16时才使用
    use_fp16 = (args.dtype in ['float16','fp16'])
    use_bf16 = (args.dtype in ['bfloat16','bf16'])
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)  # bf16 不需要 scaler

    # 初始化Optimizer -- 采用Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)

    # 日志已在文件顶部初始化

    # ============================ 训练，启动！！！ =================================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)

    # 迭代循环
    if args.epochs > 0:
        for epoch in range(args.epochs):
            train_epoch(epoch)
    else:
        Logger("No training epochs (epochs=0), dry run finished.")