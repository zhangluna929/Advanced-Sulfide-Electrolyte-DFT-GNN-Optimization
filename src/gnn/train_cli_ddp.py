#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分布式训练脚本"""
from __future__ import annotations

import argparse
from pathlib import Path
import builtins

import torch
import torch.distributed as dist
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.dist import init_distributed, is_rank_zero
from .gnn_core import (
    GraphDataset,
    GCNModel,
    MPNNModel,
    GATModel,
    train_model,
    predict_mc,
    save_preds_csv,
)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def _suppress_print_if_needed():
    """若非rank-0，静默打印以避免日志重复"""
    if not is_rank_zero():
        builtins.print = lambda *args, **kwargs: None
        try:
            from tqdm.auto import tqdm
            tqdm.write = lambda *args, **kwargs: None
        except Exception:
            pass


def build_model(name: str, in_dim: int, edge_dim: int, hidden: int = 64):
    name = name.lower()
    if name == "gcn":
        return GCNModel(in_channels=in_dim, hidden_channels=hidden)
    elif name == "mpnn":
        return MPNNModel(in_channels=in_dim, edge_dim=edge_dim, hidden_channels=hidden)
    elif name == "gat":
        return GATModel(in_channels=in_dim, hidden_channels=hidden)
    elif name == "transformer":
        from .gnn_core import TransformerModel
        return TransformerModel(in_channels=in_dim, hidden_channels=hidden)
    else:
        raise ValueError(f"Unknown model type: {name}")


# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN分布式训练CLI")
    parser.add_argument("--data_root", type=str, default="data/gnn_data", help="图数据目录")
    parser.add_argument("--model", type=str, choices=["gcn", "mpnn", "gat", "transformer"], default="gcn", help="模型类型")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏维度")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=32, help="批大小")
    parser.add_argument("--mc", type=int, default=30, help="MC Dropout采样次数")
    parser.add_argument("--csv", type=str, default="results/doping_predictions.csv", help="输出csv路径")
    parser.add_argument("--local_rank", type=int, default=0, help="本地GPU rank")
    parser.add_argument("--pretrained_path", type=str, default=None, help="预训练模型权重路径")
    parser.add_argument("--freeze_base", action="store_true", help="是否冻结除输出层外的参数")
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--save_interval", type=int, default=10, help="每N epoch保存一次checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default="results/checkpoints", help="checkpoint保存目录")
    parser.add_argument("--tasks", nargs="+", default=["conductivity", "stability"], help="预测任务列表")
    parser.add_argument("--use_dwa", action="store_true", help="启用Dynamic Weight Averaging")

    args = parser.parse_args()

    init_distributed(args.local_rank)
    _suppress_print_if_needed()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    # ------------------ 数据加载 ------------------
    from .gnn_core import GraphDataset
    dataset = GraphDataset(root=data_root, tasks=args.tasks)
    torch.manual_seed(42)
    perm = torch.randperm(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[perm[:n_train]]
    test_dataset = dataset[perm[n_train:]]

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)  # 推理阶段仅 rank-0 使用

    # ------------------ 构建模型 ------------------
    in_dim = dataset[0].x.size(1)
    edge_dim = dataset[0].edge_attr.size(1)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, in_dim=in_dim, edge_dim=edge_dim, hidden=args.hidden).to(device)

    # ------------------ 迁移学习: 加载并可选冻结 ------------------
    if args.pretrained_path is not None:
        state = torch.load(args.pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_rank_zero():
            print(f"[TL] 载入预训练权重: {args.pretrained_path}")
            if missing:
                print(f"   [33m缺失权重 [0m: {missing}")
            if unexpected:
                print(f"   [33m多余权重 [0m: {unexpected}")

    if args.freeze_base:
        for name, param in model.named_parameters():
            if "lin" not in name:  # 默认仅最后全连接层参与训练
                param.requires_grad = False
        if is_rank_zero():
            print("[TL] 已冻结基础层参数, 仅训练输出层")

    # DDP 封装
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # ------------------ 优化器 & AMP ------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # 多任务损失
    from .multitask_utils import MultiTaskLoss
    criterion = MultiTaskLoss(args.tasks)

    # ------------------ 训练循环 ------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad()

        for step, data in enumerate(train_loader, 1):
            data = data.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(data)
                loss_dict = criterion(out, data.y)
                loss = loss_dict["total"] / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * data.num_graphs * args.grad_accum  # 还原真实 loss

        if is_rank_zero():
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch:03d} | train {avg_loss:.4e}")
            
            # DWA 权重更新
            if args.use_dwa and hasattr(criterion, 'update_weights_dwa'):
                task_losses = torch.stack([loss_dict[task] for task in args.tasks])
                criterion.update_weights_dwa(task_losses)

            # checkpoint
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = Path(args.ckpt_dir) / f"model_epoch{epoch:03d}.pt"
                torch.save(model.module.state_dict(), ckpt_path)
                print(f"[Checkpoint] 已保存至 {ckpt_path}")

    # ------------------ 推理 (仅主进程) ------------------
    if is_rank_zero():
        mean, std, metas = predict_mc(model.module, test_loader, mc_samples=args.mc, device=device)
        csv_path = save_preds_csv(mean, std, metas, csv_path=args.csv)
        print(f"[rank-0] 预测结果 & 不确定性已保存至: {csv_path}")

    # 结束
    dist.destroy_process_group()


if __name__ == "__main__":
    main() 