#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行训练与推理脚本"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

import optuna
import yaml

from .gnn_core import (
    GraphDataset,
    GCNModel,
    MPNNModel,
    EvidentialGCN,
    AttJKModel,
    train_model,
    predict_mc,
    save_preds_csv,
    _eval_loss,
    VirtualNodeModel,
    EdgeNetworkModel,
    DiffPoolModel,
)


def build_model(
    name: str,
    in_dim: int,
    edge_dim: int,
    hidden: int = 64,
    dropout: float = 0.2,
):
    name = name.lower()
    if name == "gcn":
        return GCNModel(in_channels=in_dim, hidden_channels=hidden, dropout=dropout)
    elif name == "mpnn":
        return MPNNModel(
            in_channels=in_dim,
            edge_dim=edge_dim,
            hidden_channels=hidden,
            dropout=dropout,
        )
    elif name == "transformer":
        from .gnn_core import TransformerModel
        return TransformerModel(in_channels=in_dim, hidden_channels=hidden)
    elif name == "gat":
        return GATModel(
            in_channels=in_dim,
            hidden_channels=hidden,
            dropout=dropout,
        )
    elif name == "gat_edge":
        return EdgeGATModel(
            in_channels=in_dim,
            edge_dim=edge_dim,
            hidden_channels=hidden,
            dropout=dropout,
        )
    elif name == "attjk":
        return AttJKModel(in_channels=in_dim, hidden_channels=hidden)
    elif name == "evi":
        return EvidentialGCN(in_channels=in_dim, hidden_channels=hidden)
    elif name == "vn":
        return VirtualNodeModel(in_channels=in_dim, hidden_channels=hidden)
    elif name == "edge":
        return EdgeNetworkModel(in_channels=in_dim, edge_dim=edge_dim, hidden_channels=hidden)
    elif name == "diffpool":
        return DiffPoolModel(in_channels=in_dim, hidden_channels=hidden)
    else:
        raise ValueError(f"Unknown model type: {name}")


def main():
    parser = argparse.ArgumentParser(description="GNN 训练 + MC Dropout 推理 CLI")
    parser.add_argument("--data_root", type=str, default="data/gnn_data", help="图数据目录 (json)")
    parser.add_argument("--model", type=str, choices=["gcn", "mpnn", "attjk", "evi", "vn", "edge", "diffpool"], default="gcn", help="模型类型")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏维度")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=32, help="批大小")
    parser.add_argument("--mc", type=int, default=30, help="MC Dropout 采样次数")
    parser.add_argument("--optuna_trials", type=int, default=0, help="若 >0 则执行对应次数的 Optuna 超参数优化")
    parser.add_argument("--device", type=str, default="cpu", help="设备 cpu/cuda")
    parser.add_argument("--csv", type=str, default="results/doping_predictions.csv", help="输出 csv 路径")
    parser.add_argument("--use_best_config", action="store_true", help="使用 Optuna 得到的 best_config.yaml 进行推理")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    # 数据加载 (一次性, 若进行 Optuna 会在 closure 中复用)
    dataset = GraphDataset(root=data_root)
    torch.manual_seed(42)
    perm = torch.randperm(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[perm[:n_train]]
    test_dataset = dataset[perm[n_train:]]

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)

    # ---------- Optuna 超参数优化 ----------
    if args.optuna_trials > 0:

        def objective(trial):
            hidden = trial.suggest_categorical("hidden", [64, 128, 256])
            dropout_p = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

            in_dim = dataset[0].x.size(1)
            edge_dim = dataset[0].edge_attr.size(1)
            model = build_model(
                args.model,
                in_dim=in_dim,
                edge_dim=edge_dim,
                hidden=hidden,
                dropout=dropout_p,
            )

            model = train_model(
                model,
                train_loader,
                val_loader=test_loader,
                epochs=args.epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=args.device,
            )

            # 使用验证集 MSE 作为目标
            mse = _eval_loss(model, test_loader, torch.nn.MSELoss(), args.device)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.optuna_trials)

        print("Optuna 最优参数:", study.best_params)

        # 保存到 YAML
        Path("results").mkdir(exist_ok=True)
        cfg_path = Path("results/best_config.yaml")
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(study.best_params, f, allow_unicode=True)
        print(f"最优超参数已保存至 {cfg_path}")

        # 使用最佳参数重新训练完整模型, 然后继续预测
        best = study.best_params
        model = build_model(
            args.model,
            in_dim=dataset[0].x.size(1),
            edge_dim=dataset[0].edge_attr.size(1),
            hidden=best["hidden"],
            dropout=best["dropout"],
        )
        model = train_model(
            model,
            train_loader,
            val_loader=test_loader,
            epochs=args.epochs,
            lr=best["lr"],
            weight_decay=best["weight_decay"],
            device=args.device,
        )

    else:
        # -------- 原始固定超参数或加载 best_config 训练 --------
        if args.use_best_config:
            cfg_path = Path("results/best_config.yaml")
            if not cfg_path.exists():
                raise FileNotFoundError("best_config.yaml 未找到，请先运行 Optuna 优化生成该文件")
            with cfg_path.open("r", encoding="utf-8") as f:
                best = yaml.safe_load(f)

            model = build_model(
                args.model,
                in_dim=dataset[0].x.size(1),
                edge_dim=dataset[0].edge_attr.size(1),
                hidden=best.get("hidden", args.hidden),
                dropout=best.get("dropout", 0.2),
            )

            model = train_model(
                model,
                train_loader,
                val_loader=test_loader,
                epochs=args.epochs,
                lr=best.get("lr", args.lr),
                weight_decay=best.get("weight_decay", 1e-5),
                device=args.device,
            )
        else:
            model = build_model(args.model, in_dim=dataset[0].x.size(1), edge_dim=dataset[0].edge_attr.size(1), hidden=args.hidden)

            model = train_model(
                model,
                train_loader,
                val_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
            )

    # MC Dropout 预测 & 不确定性
    mean, std, metas = predict_mc(model, test_loader, mc_samples=args.mc, device=args.device)

    csv_path = save_preds_csv(mean, std, metas, csv_path=args.csv)
    print(f"预测结果 & 不确定性已保存至: {csv_path}")


if __name__ == "__main__":
    main() 