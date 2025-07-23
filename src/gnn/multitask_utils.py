"""multitask_utils.py
多任务学习工具模块
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class MultiTaskLoss(nn.Module):
    def __init__(self, task_names: List[str], loss_weights: Dict[str, float] | None = None):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        if loss_weights is None:
            loss_weights = {name: 1.0 for name in task_names}
        
        self.register_buffer(
            "weights", 
            torch.tensor([loss_weights.get(name, 1.0) for name in task_names])
        )
        
        self.criterions = nn.ModuleDict({
            name: nn.MSELoss() for name in task_names
        })
        
        self.prev_losses = None
        self.temp = 2.0
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        task_losses = []
        
        for i, task_name in enumerate(self.task_names):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            loss_i = self.criterions[task_name](pred_i, target_i)
            losses[task_name] = loss_i
            task_losses.append(loss_i)
        
        task_losses_tensor = torch.stack(task_losses)
        total_loss = torch.sum(self.weights * task_losses_tensor)
        losses["total"] = total_loss
        
        return losses
    
    def update_weights_dwa(self, current_losses: torch.Tensor):
        if self.prev_losses is None:
            self.prev_losses = current_losses.detach()
            return
        
        loss_ratios = current_losses / (self.prev_losses + 1e-8)
        
        weights = self.num_tasks * F.softmax(loss_ratios / self.temp, dim=0)
        self.weights.data = weights.detach()
        self.prev_losses = current_losses.detach()


def compute_task_metrics(predictions: torch.Tensor, targets: torch.Tensor, task_names: List[str]) -> Dict[str, float]:
    metrics = {}
    
    for i, task_name in enumerate(task_names):
        pred_i = predictions[:, i].detach().cpu()
        target_i = targets[:, i].detach().cpu()
        
        mae = torch.mean(torch.abs(pred_i - target_i)).item()
        metrics[f"{task_name}_mae"] = mae
        
        ss_res = torch.sum((target_i - pred_i) ** 2)
        ss_tot = torch.sum((target_i - torch.mean(target_i)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics[f"{task_name}_r2"] = r2.item()
    
    return metrics


class GradNormLoss(nn.Module):
    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.register_parameter("weights", nn.Parameter(torch.ones(num_tasks)))
    
    def forward(self, losses: torch.Tensor, shared_params) -> torch.Tensor:
        weighted_loss = torch.sum(self.weights * losses)
        
        grads = torch.autograd.grad(
            weighted_loss, shared_params, create_graph=True, retain_graph=True
        )
        grad_norms = torch.stack([torch.norm(g) for g in grads])
        
        loss_ratios = losses / torch.mean(losses)
        target_grad_norms = torch.mean(grad_norms) * (loss_ratios ** self.alpha)
        
        grad_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
        
        return grad_loss 