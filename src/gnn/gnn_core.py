"""
核心GNN模型模块
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    TransformerConv,
    global_mean_pool,
    MessagePassing,
    DenseGCNConv,
    dense_diff_pool,
)
from torch_geometric.utils import dropout_edge, to_dense_batch, to_dense_adj

from .feature_utils import get_atom_features, get_edge_features
from tqdm.auto import tqdm

LABEL_KEYS: Tuple[str, ...] = ("conductivity", "stability", "bandgap")
NUM_TASKS: int = len(LABEL_KEYS)

ATOM_TYPES: Tuple[str, ...] = (
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
)

def _encode_atom(atom_type: str) -> int:
    return ATOM_TYPES.index(atom_type) if atom_type in ATOM_TYPES else 0

class GraphDataset(InMemoryDataset):
    def __init__(self, root: str | Path, split: str | None = None):
        self.split = split
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.root, "raw", "graph_data.json")]

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.processed_dir, f"data_{self.split}.pt")]

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = json.load(f)

        data_list = []
        for item in raw_data:
            nodes = item['nodes']
            edges = item['edges']
            labels = item['labels']
            metadata = item.get('metadata', {})

            x = []
            for node in nodes:
                atom_type = node.get('atom_type', 'H')
                atom_idx = _encode_atom(atom_type)
                charge_density = node.get('charge_density', 0.0)
                x.append([atom_idx, charge_density])
            
            x = torch.tensor(x, dtype=torch.float)

            edge_index = []
            edge_attr = []
            for edge in edges:
                i, j = edge[0], edge[1]
                distance = edge[2] if len(edge) > 2 else 1.0
                edge_index.append([i, j])
                edge_attr.append([distance])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            y = torch.tensor([
                labels.get('conductivity', 0.0),
                labels.get('stability', 0.0),
                labels.get('bandgap', 0.0)
            ], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.metadata = metadata
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        tasks: int = 2,
        hetero: bool = False,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, tasks)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)


# ---------------------------------------------
# 新增: GAT 模型
# ---------------------------------------------


class GATModel(nn.Module):
    """多头图注意力网络 (GAT) 用于回归电导率 & 稳定性

    默认使用两层注意力卷积, 第一层多头 concat, 第二层平均。
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        tasks: int = 2,
        hetero: bool = False,
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )  # 输出维度 = hidden_channels * heads

        # 第二层设 concat=False 以便输出固定 hidden_channels 维
        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.dropout_p = dropout
        out_dim = tasks * (2 if hetero else 1)
        self.lin = nn.Linear(hidden_channels, out_dim)
        self.hetero = hetero

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        # 全局池化
        x = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        out = self.lin(x)
        return out


# ---------------------------------------------
# 2.4 Transformer GNN 模型
# ---------------------------------------------


class TransformerModel(nn.Module):
    """2-layer Graph Transformer (TransformerConv)"""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.dropout_p = dropout
        self.lin = nn.Linear(hidden_channels, 2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        out = self.lin(x)
        return out


# ---------------------------------------------
# 新增: Edge-aware GAT (TransformerConv)
# ---------------------------------------------


class EdgeGATModel(nn.Module):
    """基于 TransformerConv 的边特征注意力模型。

    TransformerConv 支持 edge_dim, 可直接结合多维边特征进行注意力权重计算。
    """

    def __init__(
        self,
        in_channels: int = 2,
        edge_dim: int = 18,
        hidden_channels: int = 64,
        heads: int = 4,
        out_dim: int = NUM_TASKS,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,
        )  # output: hidden_channels * heads (concat)

        self.conv2 = TransformerConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,
            concat=False,
        )

        self.dropout_p = dropout
        self.lin = nn.Linear(hidden_channels, out_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        out = self.lin(x)
        return out


# ---------------------------------------------
# Dynamic Edge Update Model
# ---------------------------------------------


class EdgeNetworkLayer(MessagePassing):
    """MPNN layer with dynamic edge feature update"""

    def __init__(self, in_channels: int, edge_dim: int, out_channels: int):
        super().__init__(aggr="add")
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        self.edge_dim = edge_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # 1. Update edge features first
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        new_edge_attr = self.edge_mlp(edge_input)
        
        # 2. Message passing with updated edges
        out = self.propagate(edge_index, x=x, edge_attr=new_edge_attr)
        return out, new_edge_attr

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        return aggr_out + x  # residual


class EdgeNetworkModel(nn.Module):
    """GNN with dynamic edge feature updates"""

    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 64, num_layers: int = 3, out_dim: int = NUM_TASKS, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout

        # Input projection
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        
        # Edge network layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EdgeNetworkLayer(hidden_channels, edge_dim, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_proj(x)
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        out = global_mean_pool(x, batch)
        return self.lin(out)


# ---------------------------------------------
# 支持异方差输出的 MPNNModel
# ---------------------------------------------


class MPNNModel(nn.Module):
    """2-layer Edge-conditioned MPNN (支持可变 edge_attr 维度)"""

    def __init__(
        self,
        in_channels: int = 2,
        edge_dim: int = 1,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        tasks: int = 2,
        hetero: bool = False,
    ):
        super().__init__()
        self.layer1 = EdgeMPNNLayer(in_channels, edge_dim, hidden_channels)
        self.layer2 = EdgeMPNNLayer(hidden_channels, edge_dim, hidden_channels)
        self.dropout_p = dropout
        out_dim = tasks * (2 if hetero else 1)
        self.lin = nn.Linear(hidden_channels, out_dim)
        self.hetero = hetero

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.layer1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.layer2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        out = self.lin(x)
        return out


# ---------------------------------------------
# Original EdgeMPNNLayer (keep for backward compatibility)
# ---------------------------------------------


class EdgeMPNNLayer(MessagePassing):
    """一个改进的 MPNN layer, 接收多维边特征"""

    def __init__(self, in_channels: int, edge_dim: int, out_channels: int):
        super().__init__(aggr="add")
        self.edge_dim = edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        # x_j: source node features
        m = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp(m)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        return aggr_out + x  # residual


# ---------------------------------------------
# Hierarchical Pooling (DiffPool)
# ---------------------------------------------


class DiffPoolModel(nn.Module):
    """Hierarchical GNN with DiffPool"""

    def __init__(self, in_channels: int, hidden_channels: int = 64, max_nodes: int = 100, out_dim: int = NUM_TASKS, dropout: float = 0.2):
        super().__init__()
        self.max_nodes = max_nodes
        self.dropout_p = dropout
        
        # First GCN block
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Pool assignment matrix computation
        self.pool1_conv = GCNConv(hidden_channels, max_nodes // 4)
        
        # Second GCN block after pooling
        self.conv3 = DenseGCNConv(hidden_channels, hidden_channels)
        self.conv4 = DenseGCNConv(hidden_channels, hidden_channels)
        
        self.lin = nn.Linear(hidden_channels, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # First GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Prepare for DiffPool - convert to dense format
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
        
        # Compute assignment matrix
        s = F.relu(self.conv1(data.x, edge_index))
        s = self.pool1_conv(s, edge_index)
        s_dense, _ = to_dense_batch(s, batch, max_num_nodes=self.max_nodes)
        s_dense = F.softmax(s_dense, dim=-1)
        
        # Apply DiffPool
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(x_dense, adj, s_dense, mask)
        
        # Second GCN block on pooled graph
        x_pool = F.relu(self.conv3(x_pool, adj_pool))
        x_pool = F.dropout(x_pool, p=self.dropout_p, training=self.training)
        x_pool = F.relu(self.conv4(x_pool, adj_pool))
        
        # Global pooling
        out = x_pool.mean(dim=1)  # mean over nodes
        
        return self.lin(out)


# ---------------------------------------------
# 3. 训练 / 评估 / 不确定性
# ---------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: str | torch.device = "cpu",
):
    """训练模型, 支持验证集早停"""
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 根据输出维度自动选择损失函数
    out_dim = next(model.parameters()).shape  # dummy to trigger iterator

    criterion_mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            if out.size(1) == data.y.size(1):
                # 普通 MSE
                loss = criterion_mse(out, data.y)
            else:
                # 异方差负对数似然
                mu, log_sigma = out[:, : data.y.size(1)], out[:, data.y.size(1) :]
                loss = ((data.y - mu) ** 2) * torch.exp(-log_sigma) + log_sigma
                loss = 0.5 * loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        train_loss = total_loss / len(train_loader.dataset)

        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, criterion_mse, device) # Changed criterion to criterion_mse
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            pbar_desc = f"Epoch {epoch:03d} | train {train_loss:.4e} | val {val_loss:.4e}"
        else:
            pbar_desc = f"Epoch {epoch:03d} | train {train_loss:.4e}"

        tqdm.write(pbar_desc)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_loss(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total += loss.item() * data.num_graphs
    return total / len(loader.dataset)


def predict_mc(
    model: nn.Module,
    loader: DataLoader,
    mc_samples: int = 30,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Monte-Carlo Dropout 推理. 返回 (mean, std, metadata)"""
    device = torch.device(device)
    model.to(device)
    model.train()  # 关键: 置于 train 模式以激活 dropout

    preds = []
    metas = []
    for _ in range(mc_samples):
        batch_preds = []
        batch_metas = []
        for data in loader:
            data = data.to(device)
            out = model(data).cpu().numpy()
            batch_preds.append(out)
            # 保存原始元数据 (仅保存一次即可)
            if len(metas) == 0:
                meta_batch = [
                    {
                        "material": d.material,
                        "dopant": d.dopant,
                        "concentration": d.concentration,
                        "position": d.position,
                    }
                    for d in data.to("cpu").to_data_list()
                ]
                batch_metas.extend(meta_batch)
        preds.append(np.concatenate(batch_preds, axis=0))
        if len(metas) == 0:
            metas = batch_metas

    preds = np.stack(preds, axis=0)  # [mc, N, 2]
    mean = preds.mean(axis=0)  # [N, 2]
    std = preds.std(axis=0)  # [N, 2]
    return mean, std, metas


# ---------------------------------------------
# 4. 节点级 MC 不确定性 (均值 & 方差)
# ---------------------------------------------


def _locate_last_node_layer(model: nn.Module) -> nn.Module:
    """根据模型类型定位最后一层节点级聚合前的层, 以便抓取节点表示。

    当前支持 GCNModel(conv3), GATModel(gat2), MPNNModel(layer2)。
    若未来添加更多模型, 请在此扩展。"""

    if hasattr(model, "conv3"):
        return model.conv3  # GCNModel
    if hasattr(model, "gat2"):
        return model.gat2  # GATModel
    if hasattr(model, "layer2"):
        return model.layer2  # MPNNModel
    raise ValueError("无法定位节点层, 请在 _locate_last_node_layer 中添加支持")


def compute_node_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    mc_samples: int = 30,
    device: str | torch.device = "cpu",
) -> List[Data]:
    """对 loader 中每个图计算 MC Dropout 下的节点级均值/方差。

    结果会写回 Data.x_var (方差) 与 Data.x_mean (均值)。

    注意: 建议将 DataLoader 的 batch_size 设为 1, 以避免跨图节点数不同导致的张量拼接问题。"""

    device = torch.device(device)
    model.to(device)

    node_layer = _locate_last_node_layer(model)

    results: List[Data] = []

    for data in loader:
        data = data.to(device)

        feat_samples: List[torch.Tensor] = []

        for _ in range(mc_samples):
            container: dict[str, torch.Tensor] = {}

            def _hook(_module, _inp, out):
                # out: Tensor [n_nodes, hidden]
                container["feat"] = out.detach()

            handle = node_layer.register_forward_hook(_hook)

            model.train()  # 激活 dropout
            _ = model(data)
            handle.remove()

            feat_samples.append(container["feat"].cpu())

        stack = torch.stack(feat_samples, dim=0)  # [mc, n_nodes, hidden]
        mean = stack.mean(dim=0)  # [n_nodes, hidden]
        var = stack.var(dim=0, unbiased=False)  # [n_nodes, hidden]

        # 将结果缓存回 Data, 方便后续传播
        data.x_mean = mean
        data.x_var = var

        results.append(data.cpu())

    return results


# ---------------------------------------------
# 5. 图级不确定性传播 (节点 -> 读出 -> 线性)
# ---------------------------------------------


def compute_graph_uncertainty(
    model: nn.Module,
    data_list: List[Data],
) -> Tuple[np.ndarray, np.ndarray]:
    """根据 data_list 节点级均值/方差, 解析模型权重, 计算图输出的均值/标准差。

    Args:
        model: 训练好的 GNN, 其 forward 包含 global_mean_pool + nn.Linear.
        data_list: 经过 `compute_node_uncertainty` 处理, 含 x_mean/x_var 属性。

    Returns:
        mean_arr: (n_graph, out_dim) ndarray
        std_arr : 同 shape 的标准差 ndarray
    """

    # 提取线性层权重与偏置, 假设为 model.lin
    if not hasattr(model, "lin"):
        raise AttributeError("模型缺少 lin 全连接层, 无法传播不确定性")

    W: torch.Tensor = model.lin.weight.detach().cpu()  # [out_dim, hidden]
    b: torch.Tensor = model.lin.bias.detach().cpu()  # [out_dim]

    W2 = W ** 2  # elementwise square for variance propagation

    mean_out_list: List[np.ndarray] = []
    std_out_list: List[np.ndarray] = []

    for data in data_list:
        if not hasattr(data, "x_mean") or not hasattr(data, "x_var"):
            raise AttributeError("Data 对象缺少 x_mean/x_var, 请先调用 compute_node_uncertainty")

        n_nodes = data.x_mean.shape[0]

        mean_pool = data.x_mean.mean(dim=0)  # [hidden]
        var_pool = data.x_var.sum(dim=0) / (n_nodes ** 2)  # [hidden]

        mean_out = torch.matmul(W, mean_pool) + b  # [out_dim]
        var_out = torch.matmul(W2, var_pool)  # [out_dim]

        mean_out_list.append(mean_out.numpy())
        std_out_list.append(torch.sqrt(var_out + 1e-12).numpy())

    mean_arr = np.vstack(mean_out_list)
    std_arr = np.vstack(std_out_list)

    return mean_arr, std_arr


# ---------------------------------------------
# 4. 辅助函数
# ---------------------------------------------

def save_preds_csv(
    mean: np.ndarray,
    std: np.ndarray,
    metas: List[Dict[str, Any]],
    csv_path: str | Path = "results/doping_predictions.csv",
):
    """保存预测结果到 CSV, 与 intelligent_decision.py 兼容"""
    records = []
    for m, s, meta in zip(mean, std, metas):
        records.append(
            {
                "Material": meta["material"],
                "Dopant": meta["dopant"],
                "Concentration": meta["concentration"],
                "Position": meta["position"],
                "Predicted_Conductivity": m[0],
                "Predicted_Stability": m[1],
                "Uncertainty_Conductivity": s[0],
                "Uncertainty_Stability": s[1],
            }
        )
    df = pd.DataFrame.from_records(records)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------
# 5. 示例用法
# ---------------------------------------------

def _example_run():
    """若直接运行模块, 执行一次简单的训练+预测流程."""

    data_root = Path("data/gnn_data")
    if not data_root.exists():
        raise FileNotFoundError(
            "示例运行依赖 data/gnn_data, 请先执行 data_processing_for_gnn.py 生成样本数据。"
        )

    full_dataset = GraphDataset(root=data_root)
    # 简单随机划分
    torch.manual_seed(42)
    perm = torch.randperm(len(full_dataset))
    n_train = int(0.8 * len(full_dataset))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train_ds = full_dataset[train_idx]
    test_ds = full_dataset[test_idx]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    in_dim = full_dataset[0].x.size(1)
    model = GCNModel(in_channels=in_dim)
    model = train_model(model, train_loader, val_loader=test_loader, epochs=50, device="cuda" if torch.cuda.is_available() else "cpu")

    mean, std, metas = predict_mc(model, test_loader, mc_samples=30, device="cuda" if torch.cuda.is_available() else "cpu")
    save_path = save_preds_csv(mean, std, metas)
    print(f"预测结果已保存至 {save_path}")


if __name__ == "__main__":
    _example_run() 