"""fsdp_training.py
FSDP + ZeRO 大模型训练
====================
• Fully Sharded Data Parallel (FSDP)
• ZeRO-3 内存优化
• 支持 Transformer 级别大模型
• 自动 offloading CPU/NVMe
"""
from __future__ import annotations

import os
from typing import Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.dist import init_distributed, is_rank_zero


class FSDAConfig:
    """FSDP 配置类"""
    
    def __init__(
        self,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        cpu_offload: bool = False,
        mixed_precision: bool = True,
        auto_wrap_min_params: int = 1e6,  # 100万参数以上自动分片
        backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE,
    ):
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision
        self.auto_wrap_min_params = auto_wrap_min_params
        self.backward_prefetch = backward_prefetch
    
    def get_fsdp_config(self) -> Dict[str, Any]:
        """获取 FSDP 配置字典"""
        config = {
            "sharding_strategy": self.sharding_strategy,
            "backward_prefetch": self.backward_prefetch,
            "auto_wrap_policy": size_based_auto_wrap_policy(
                min_num_params=self.auto_wrap_min_params
            ),
        }
        
        # Mixed Precision
        if self.mixed_precision:
            config["mixed_precision"] = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        
        # CPU Offload
        if self.cpu_offload:
            config["cpu_offload"] = CPUOffload(offload_params=True)
        
        return config


def setup_fsdp_model(
    model: nn.Module,
    config: FSDAConfig = None,
    device_id: int = None,
) -> FSDP:
    """设置 FSDP 模型"""
    if config is None:
        config = FSDAConfig()
    
    if device_id is not None:
        torch.cuda.set_device(device_id)
    
    fsdp_config = config.get_fsdp_config()
    
    # 包装为 FSDP
    fsdp_model = FSDP(
        model,
        device_id=device_id,
        **fsdp_config,
    )
    
    if is_rank_zero():
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[FSDP] 总参数: {total_params:,}")
        print(f"[FSDP] 可训练参数: {trainable_params:,}")
        print(f"[FSDP] 分片策略: {config.sharding_strategy}")
        print(f"[FSDP] CPU Offload: {config.cpu_offload}")
        print(f"[FSDP] Mixed Precision: {config.mixed_precision}")
    
    return fsdp_model


def train_fsdp_model(
    model: FSDP,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 100,
    lr: float = 1e-4,
    device_id: int = None,
    save_dir: str = "results/fsdp_checkpoints",
    save_interval: int = 10,
) -> FSDP:
    """FSDP 模型训练"""
    
    # 优化器 (FSDP 自动处理参数分片)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 梯度缩放器 (配合 mixed precision)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # 设置 epoch (DistributedSampler)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            if device_id is not None:
                data = data.to(f"cuda:{device_id}")
            
            optimizer.zero_grad()
            
            # 前向传播 (自动 mixed precision)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, data.y)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 内存清理 (FSDP 场景下重要)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # 验证
        val_loss = None
        if val_loader is not None:
            val_loss = _eval_fsdp_model(model, val_loader, criterion, device_id)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # 日志 (仅 rank-0)
        if is_rank_zero():
            lr_current = scheduler.get_last_lr()[0]
            if val_loss is not None:
                print(f"Epoch {epoch:03d} | train {avg_loss:.4f} | val {val_loss:.4f} | lr {lr_current:.2e}")
            else:
                print(f"Epoch {epoch:03d} | train {avg_loss:.4f} | lr {lr_current:.2e}")
        
        # 保存 checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            save_fsdp_checkpoint(model, optimizer, epoch, save_dir)
    
    return model


def _eval_fsdp_model(model: FSDP, loader: DataLoader, criterion, device_id: int = None) -> float:
    """评估 FSDP 模型"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data in loader:
            if device_id is not None:
                data = data.to(f"cuda:{device_id}")
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, data.y)
            
            total_loss += loss.item() * data.num_graphs
            num_samples += data.num_graphs
    
    return total_loss / num_samples


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer,
    epoch: int,
    save_dir: str,
):
    """保存 FSDP checkpoint"""
    if not is_rank_zero():
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # FSDP 状态字典
    with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
        model_state = model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[FSDP] Checkpoint saved: {checkpoint_path}")


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer,
    checkpoint_path: str,
) -> int:
    """加载 FSDP checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 加载模型状态
    with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint["epoch"]
    
    if is_rank_zero():
        print(f"[FSDP] Checkpoint loaded from epoch {epoch}")
    
    return epoch


# 使用示例
def run_fsdp_training_example():
    """FSDP 训练示例"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/gnn_data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # 初始化分布式
    init_distributed(args.local_rank)
    
    # 数据加载
    from .gnn_core import GraphDataset, TransformerModel
    dataset = GraphDataset(args.data_root)
    
    train_sampler = DistributedSampler(dataset, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    
    # 创建大模型 (示例: 大型 Transformer)
    model = TransformerModel(
        in_channels=dataset[0].x.size(1),
        hidden_channels=512,  # 大隐藏维度
        heads=16,
        dropout=0.1,
    )
    
    # FSDP 配置
    fsdp_config = FSDAConfig(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=args.cpu_offload,
        mixed_precision=args.mixed_precision,
        auto_wrap_min_params=1e5,  # 10万参数以上分片
    )
    
    # 设置 FSDP
    fsdp_model = setup_fsdp_model(model, fsdp_config, args.local_rank)
    
    # 训练
    train_fsdp_model(
        fsdp_model,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device_id=args.local_rank,
    )


if __name__ == "__main__":
    run_fsdp_training_example() 