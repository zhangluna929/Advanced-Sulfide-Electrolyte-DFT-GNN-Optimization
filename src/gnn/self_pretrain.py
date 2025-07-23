from __future__ import annotations

"""self_pretrain.py
GraphCL 自监督预训练实现 (简版)
---------------------------------
• random_edge_dropout + attr_mask 两种图增强
• NT-Xent (InfoNCE) 对比学习损失
• 适配现有 GNN Backbone (GCN/MPNN/Transformer 等)

使用示例::
    from src.gnn.self_pretrain import graphcl_pretrain
    backbone = GCNModel(in_channels=4, hidden_channels=64)
    graphcl_pretrain(backbone, dataset, epochs=100, batch_size=128)
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader

# ------------- 图增强 -------------

def edge_dropout(data: Data, drop_ratio: float = 0.2) -> Data:
    """随机丢弃一定比例的边"""
    num_edges = data.edge_index.size(1)
    keep = int(num_edges * (1 - drop_ratio))
    perm = torch.randperm(num_edges)[:keep]
    data_aug = data.clone()
    data_aug.edge_index = data.edge_index[:, perm]
    if data.edge_attr is not None:
        data_aug.edge_attr = data.edge_attr[perm]
    return data_aug


def attr_mask(data: Data, mask_ratio: float = 0.2) -> Data:
    """随机掩盖节点属性 (高斯噪声)"""
    x = data.x.clone()
    num_nodes, feat_dim = x.size()
    mask_num = int(num_nodes * mask_ratio)
    idx = torch.randperm(num_nodes)[:mask_num]
    x[idx] = torch.randn_like(x[idx])  # 替换为噪声
    data_aug = data.clone()
    data_aug.x = x
    return data_aug


def random_aug(data: Data) -> Tuple[Data, Data]:
    """生成两份随机增强视图"""
    funcs = [edge_dropout, attr_mask]
    f1, f2 = random.sample(funcs, 2)
    return f1(data), f2(data)


# ------------- 投影头 -------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ------------- GraphCL 训练 -------------


def graphcl_pretrain(
    backbone: nn.Module,
    dataset,
    epochs: int = 100,
    batch_size: int = 128,
    proj_dim: int = 128,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
):
    """GraphCL 预训练, 返回训练后的 backbone"""
    device = torch.device(device)
    backbone.to(device)

    # 推断嵌入维度 (通过一次前向)
    dummy = dataset[0].clone()
    dummy.batch = torch.zeros(dummy.num_nodes, dtype=torch.long)
    with torch.no_grad():
        emb_dim = backbone(dummy.to(device)).size(-1)

    projector = ProjectionHead(emb_dim, proj_dim).to(device)

    parameters = list(backbone.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        backbone.train()
        projector.train()

        total_loss = 0.0
        for data in loader:
            # 构造增强视图
            views1: List[Data] = []
            views2: List[Data] = []
            for g in data.to_data_list():
                v1, v2 = random_aug(g)
                views1.append(v1)
                views2.append(v2)

            batch1 = DataLoader(views1, batch_size=len(views1))
            batch2 = DataLoader(views2, batch_size=len(views2))

            # 单批次 (两视图)
            g1 = next(iter(batch1)).to(device)
            g2 = next(iter(batch2)).to(device)

            z1 = projector(backbone(g1))
            z2 = projector(backbone(g2))

            # 标准化
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

            # NT-Xent
            batch_size_current = z1.size(0)
            representations = torch.cat([z1, z2], dim=0)
            similarity = torch.matmul(representations, representations.T)  # [2B,2B]
            # 排除对角
            mask = ~torch.eye(2 * batch_size_current, dtype=torch.bool, device=similarity.device)
            similarity = similarity / 0.5  # temperature

            positives = torch.cat([torch.diag(similarity, batch_size_current), torch.diag(similarity, -batch_size_current)])
            negatives = similarity[mask].view(2 * batch_size_current, -1)

            logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
            labels = torch.zeros(2 * batch_size_current, dtype=torch.long, device=device)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size_current

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[GraphCL] Epoch {epoch:03d} | loss {avg_loss:.4f}")

    return backbone 