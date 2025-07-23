"""uncertainty.py
集成不确定性估计模块
==================
• Deep Ensemble: 训练多个独立模型
• MC-Dropout: 蒙特卡洛 Dropout 采样
• Ensemble + MC 混合策略
• 校准不确定性 (Temperature Scaling)
"""
from __future__ import annotations

import copy
from typing import List, Tuple, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import DataLoader

from .gnn_core import train_model


class DeepEnsemble:
    """Deep Ensemble 不确定性估计"""
    
    def __init__(self, model_factory, num_models: int = 5):
        """
        Args:
            model_factory: 返回新模型实例的函数
            num_models: 集成模型数量
        """
        self.model_factory = model_factory
        self.num_models = num_models
        self.models: List[nn.Module] = []
    
    def train_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cpu",
        save_dir: str | None = None,
    ):
        """训练集成模型"""
        self.models = []
        
        for i in range(self.num_models):
            print(f"[Ensemble] 训练模型 {i+1}/{self.num_models}")
            
            # 创建新模型实例
            model = self.model_factory()
            
            # 不同随机种子初始化
            torch.manual_seed(42 + i * 1000)
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            
            # 训练
            model = train_model(
                model, train_loader, val_loader, 
                epochs=epochs, lr=lr, device=device
            )
            
            self.models.append(model)
            
            # 可选保存
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(save_dir) / f"model_{i}.pt")
    
    def predict_ensemble(
        self, 
        loader: DataLoader, 
        device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """集成预测，返回 (mean, std, metadata)"""
        device = torch.device(device)
        
        all_preds = []
        metas = []
        
        for model in self.models:
            model.eval()
            model.to(device)
            
            batch_preds = []
            batch_metas = []
            
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data).cpu().numpy()
                    batch_preds.append(out)
                    
                    # 只保存一次 metadata
                    if len(metas) == 0:
                        for i in range(data.num_graphs):
                            meta = {
                                "material": getattr(data, "material", [""])[i] if hasattr(data, "material") else "",
                                "dopant": getattr(data, "dopant", [""])[i] if hasattr(data, "dopant") else "",
                                "concentration": getattr(data, "concentration", [0.0])[i] if hasattr(data, "concentration") else 0.0,
                                "position": getattr(data, "position", [""])[i] if hasattr(data, "position") else "",
                            }
                            batch_metas.append(meta)
            
            all_preds.append(np.concatenate(batch_preds, axis=0))
            if len(metas) == 0:
                metas = batch_metas
        
        # 统计: mean, std
        ensemble_preds = np.stack(all_preds, axis=0)  # [num_models, num_samples, num_tasks]
        mean = np.mean(ensemble_preds, axis=0)
        std = np.std(ensemble_preds, axis=0)
        
        return mean, std, metas


class MCDropoutUncertainty:
    """MC-Dropout 不确定性估计 (增强版)"""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def predict_mc(
        self,
        loader: DataLoader,
        mc_samples: int = 30,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """MC-Dropout 预测"""
        device = torch.device(device)
        self.model.to(device)
        self.model.train()  # 激活 dropout
        
        all_preds = []
        metas = []
        
        for _ in range(mc_samples):
            batch_preds = []
            batch_metas = []
            
            for data in loader:
                data = data.to(device)
                out = self.model(data).cpu().numpy()
                batch_preds.append(out)
                
                # 只保存一次 metadata
                if len(metas) == 0:
                    for i in range(data.num_graphs):
                        meta = {
                            "material": getattr(data, "material", [""])[i] if hasattr(data, "material") else "",
                            "dopant": getattr(data, "dopant", [""])[i] if hasattr(data, "dopant") else "",
                            "concentration": getattr(data, "concentration", [0.0])[i] if hasattr(data, "concentration") else 0.0,
                            "position": getattr(data, "position", [""])[i] if hasattr(data, "position") else "",
                        }
                        batch_metas.append(meta)
            
            all_preds.append(np.concatenate(batch_preds, axis=0))
            if len(metas) == 0:
                metas = batch_metas
        
        # 统计
        mc_preds = np.stack(all_preds, axis=0)  # [mc_samples, num_samples, num_tasks]
        mean = np.mean(mc_preds, axis=0)
        std = np.std(mc_preds, axis=0)
        
        return mean, std, metas


class HybridUncertainty:
    """混合不确定性: Ensemble + MC-Dropout"""
    
    def __init__(self, ensemble: DeepEnsemble, mc_samples: int = 10):
        self.ensemble = ensemble
        self.mc_samples = mc_samples
    
    def predict_hybrid(
        self,
        loader: DataLoader,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        混合预测
        Returns:
            mean, epistemic_std, aleatoric_std, metadata
        """
        device = torch.device(device)
        
        all_ensemble_means = []
        all_ensemble_stds = []
        metas = []
        
        # 对每个集成模型进行 MC-Dropout
        for model in self.ensemble.models:
            mc_uncertainty = MCDropoutUncertainty(model)
            mean, std, batch_metas = mc_uncertainty.predict_mc(
                loader, self.mc_samples, device
            )
            all_ensemble_means.append(mean)
            all_ensemble_stds.append(std)
            
            if len(metas) == 0:
                metas = batch_metas
        
        # 统计
        ensemble_means = np.stack(all_ensemble_means, axis=0)  # [num_models, num_samples, num_tasks]
        ensemble_stds = np.stack(all_ensemble_stds, axis=0)
        
        # 总均值
        total_mean = np.mean(ensemble_means, axis=0)
        
        # 认识不确定性 (模型间差异)
        epistemic_std = np.std(ensemble_means, axis=0)
        
        # 偶然不确定性 (模型内 MC 采样)
        aleatoric_std = np.mean(ensemble_stds, axis=0)
        
        return total_mean, epistemic_std, aleatoric_std, metas


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """温度缩放校准不确定性"""
    return logits / temperature 