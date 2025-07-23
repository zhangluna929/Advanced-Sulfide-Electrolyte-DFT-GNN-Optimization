"""训练工具测试"""
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

from src.gnn.multitask_utils import MultiTaskLoss, compute_task_metrics


def test_multitask_loss():
    task_names = ["conductivity", "stability"]
    loss_fn = MultiTaskLoss(task_names)
    
    predictions = torch.randn(4, 2)
    targets = torch.randn(4, 2)
    
    loss_dict = loss_fn(predictions, targets)
    
    assert "total" in loss_dict
    assert "conductivity" in loss_dict
    assert "stability" in loss_dict
    assert loss_dict["total"] > 0


def test_task_metrics():
    predictions = torch.randn(4, 2)
    targets = torch.randn(4, 2)
    task_names = ["conductivity", "stability"]
    
    metrics = compute_task_metrics(predictions, targets, task_names)
    
    assert "conductivity_mae" in metrics
    assert "conductivity_r2" in metrics
    assert "stability_mae" in metrics
    assert "stability_r2" in metrics


def test_dwa_weight_update():
    task_names = ["conductivity", "stability"]
    loss_fn = MultiTaskLoss(task_names)
    
    current_losses = torch.tensor([0.5, 0.3])
    loss_fn.update_weights_dwa(current_losses)
    
    assert torch.all(loss_fn.weights > 0)
    assert torch.sum(loss_fn.weights) == pytest.approx(2.0, rel=1e-3) 