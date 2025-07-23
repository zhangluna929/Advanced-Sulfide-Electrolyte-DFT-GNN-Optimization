"""hyperopt_ddp.py
Optuna + DDP 超参数优化
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import optuna
from optuna.trial import TrialState

from src.utils.dist import init_distributed, is_rank_zero
from .gnn_core import GraphDataset, GCNModel, MPNNModel, TransformerModel
from .train_cli_ddp import build_model


def objective_ddp(
    trial: optuna.Trial,
    data_root: str,
    local_rank: int,
    epochs: int = 50,
    tasks: list[str] = None,
) -> float:
    
    model_type = trial.suggest_categorical("model", ["gcn", "mpnn", "transformer"])
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    
    if tasks is None:
        tasks = ["conductivity", "stability"]
    
    if not dist.is_initialized():
        init_distributed(local_rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    try:
        # 数据加载
        dataset = GraphDataset(root=data_root, tasks=tasks)
        torch.manual_seed(42)
        perm = torch.randperm(len(dataset))
        n_train = int(0.8 * len(dataset))
        n_val = int(0.1 * len(dataset))
        
        train_dataset = dataset[perm[:n_train]]
        val_dataset = dataset[perm[n_train:n_train + n_val]]
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 构建模型
        in_dim = dataset[0].x.size(1)
        edge_dim = dataset[0].edge_attr.size(1) if dataset[0].edge_attr is not None else 1
        
        if model_type == "gcn":
            model = GCNModel(in_channels=in_dim, hidden_channels=hidden_dim, dropout=dropout, tasks=len(tasks))
        elif model_type == "mpnn":
            model = MPNNModel(in_channels=in_dim, edge_dim=edge_dim, hidden_channels=hidden_dim, dropout=dropout, tasks=len(tasks))
        elif model_type == "transformer":
            model = TransformerModel(in_channels=in_dim, hidden_channels=hidden_dim, dropout=dropout)
        
        model = model.to(device)
        
        # DDP 封装
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
        
        # 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_sampler.set_epoch(epoch)
            
            # 训练一个 epoch
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
            
            # 验证
            if is_rank_zero():
                model.eval()
                val_loss = 0.0
                num_samples = 0
                
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        out = model(data)
                        loss = criterion(out, data.y)
                        val_loss += loss.item() * data.num_graphs
                        num_samples += data.num_graphs
                
                val_loss /= num_samples
                best_val_loss = min(best_val_loss, val_loss)
                
                # Pruning 检查
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return best_val_loss
        
    except Exception as e:
        if is_rank_zero():
            print(f"Trial failed: {e}")
        return float('inf')


def run_hyperopt_ddp(
    data_root: str,
    n_trials: int = 100,
    epochs: int = 50,
    local_rank: int = 0,
    study_name: str = "gnn_hyperopt",
    storage_url: str = None,
    tasks: list[str] = None,
) -> Dict[str, Any]:
    
    # 只有 rank-0 创建/管理 study
    if is_rank_zero():
        if storage_url is None:
            # 使用临时 SQLite 数据库
            temp_dir = tempfile.mkdtemp()
            storage_url = f"sqlite:///{temp_dir}/optuna.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True,
        )
        
        print(f"[Hyperopt] 开始超参优化: {n_trials} trials")
        print(f"[Hyperopt] Storage: {storage_url}")
        
        # 优化
        study.optimize(
            lambda trial: objective_ddp(trial, data_root, local_rank, epochs, tasks),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        
        # 结果统计
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        print(f"[Hyperopt] 最佳验证损失: {best_value:.6f}")
        print(f"[Hyperopt] 最佳参数: {best_params}")
        
        # 保存结果
        results_dir = Path("results/hyperopt")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "best_params.txt", "w") as f:
            f.write(f"Best validation loss: {best_value:.6f}\n")
            f.write(f"Best parameters:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
        }
    
    else:
        # 非主进程等待
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DDP 超参数优化")
    parser.add_argument("--data_root", type=str, default="data/gnn_data")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=["conductivity", "stability"])
    
    args = parser.parse_args()
    
    results = run_hyperopt_ddp(
        data_root=args.data_root,
        n_trials=args.n_trials,
        epochs=args.epochs,
        local_rank=args.local_rank,
        tasks=args.tasks,
    ) 