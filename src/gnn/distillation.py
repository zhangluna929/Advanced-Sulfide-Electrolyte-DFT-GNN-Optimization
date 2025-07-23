"""distillation.py
知识蒸馏 + 量化
===============
• Teacher: 大型 GAT/Transformer
• Student: 轻量 GCN
• 特征匹配 + 输出蒸馏
• 8-bit 量化推理
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from .gnn_core import GCNModel, GATModel, TransformerModel


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        targets: torch.Tensor,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_logits: [batch, num_tasks]
            teacher_logits: [batch, num_tasks]  
            targets: [batch, num_tasks]
            student_features: [batch, hidden_dim] 可选特征匹配
            teacher_features: [batch, hidden_dim]
        """
        losses = {}
        
        # 1. 硬目标损失 (student vs ground truth)
        hard_loss = self.mse(student_logits, targets)
        losses["hard"] = hard_loss
        
        # 2. 软目标损失 (student vs teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        losses["soft"] = soft_loss
        
        # 3. 特征匹配损失 (可选)
        if student_features is not None and teacher_features is not None:
            feature_loss = self.mse(student_features, teacher_features)
            losses["feature"] = feature_loss
        else:
            losses["feature"] = torch.tensor(0.0)
        
        # 总损失
        total_loss = (
            (1 - self.alpha) * hard_loss + 
            self.alpha * soft_loss + 
            0.1 * losses["feature"]
        )
        losses["total"] = total_loss
        
        return losses


class FeatureMatchingGCN(GCNModel):
    """支持特征输出的 GCN Student"""
    
    def forward(self, data, return_features=False):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv3(x, edge_index)
        
        # 全局池化得到图级特征
        from torch_geometric.nn import global_mean_pool
        graph_features = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        
        out = self.lin(graph_features)
        
        if return_features:
            return out, graph_features
        return out


class FeatureMatchingGAT(GATModel):
    """支持特征输出的 GAT Teacher"""
    
    def forward(self, data, return_features=False):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 全局池化
        from torch_geometric.nn import global_mean_pool
        graph_features = global_mean_pool(x, batch=data.batch if hasattr(data, "batch") else None)
        
        out = self.lin(graph_features)
        
        if return_features:
            return out, graph_features
        return out


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 100,
    lr: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "cpu",
    save_path: str = None,
) -> nn.Module:
    """知识蒸馏训练"""
    device = torch.device(device)
    teacher.to(device)
    student.to(device)
    
    teacher.eval()  # Teacher 冻结
    for param in teacher.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        
        for data in train_loader:
            data = data.to(device)
            
            # Teacher 前向 (无梯度)
            with torch.no_grad():
                if hasattr(teacher, 'forward') and 'return_features' in teacher.forward.__code__.co_varnames:
                    teacher_logits, teacher_features = teacher(data, return_features=True)
                else:
                    teacher_logits = teacher(data)
                    teacher_features = None
            
            # Student 前向
            if hasattr(student, 'forward') and 'return_features' in student.forward.__code__.co_varnames:
                student_logits, student_features = student(data, return_features=True)
            else:
                student_logits = student(data)
                student_features = None
            
            # 蒸馏损失
            losses = criterion(
                student_logits, teacher_logits, data.y,
                student_features, teacher_features
            )
            
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            
            total_loss += losses["total"].item() * data.num_graphs
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # 验证
        if val_loader is not None:
            val_loss = _eval_distill_loss(student, teacher, val_loader, criterion, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = student.state_dict()
            
            print(f"Epoch {epoch:03d} | train {avg_loss:.4f} | val {val_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d} | train {avg_loss:.4f}")
    
    # 加载最佳权重
    if best_state is not None:
        student.load_state_dict(best_state)
    
    # 保存模型
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), save_path)
        print(f"Student model saved to {save_path}")
    
    return student


def _eval_distill_loss(student, teacher, loader, criterion, device):
    """评估蒸馏损失"""
    student.eval()
    teacher.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Teacher & Student 前向
            if hasattr(teacher, 'forward') and 'return_features' in teacher.forward.__code__.co_varnames:
                teacher_logits, teacher_features = teacher(data, return_features=True)
            else:
                teacher_logits = teacher(data)
                teacher_features = None
                
            if hasattr(student, 'forward') and 'return_features' in student.forward.__code__.co_varnames:
                student_logits, student_features = student(data, return_features=True)
            else:
                student_logits = student(data)
                student_features = None
            
            losses = criterion(
                student_logits, teacher_logits, data.y,
                student_features, teacher_features
            )
            
            total_loss += losses["total"].item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def quantize_model(model: nn.Module, calibration_loader: DataLoader = None) -> nn.Module:
    """8-bit 量化模型 (简化版)"""
    try:
        # 尝试使用 PyTorch 内置量化
        model.eval()
        
        # 量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        torch.quantization.prepare(model, inplace=True)
        
        # 校准 (如果提供校准数据)
        if calibration_loader is not None:
            with torch.no_grad():
                for data in calibration_loader:
                    model(data)
        
        # 转换为量化模型
        torch.quantization.convert(model, inplace=True)
        
        print("Model quantized to 8-bit")
        return model
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Returning original model")
        return model 