"""GNN模型测试"""
import pytest
import torch
from torch_geometric.data import Data

from src.gnn.gnn_core import GCNModel, MPNNModel, GATModel


def test_gcn_model():
    model = GCNModel(in_channels=2, hidden_channels=32, out_channels=2)
    x = torch.randn(10, 2)
    edge_index = torch.randint(0, 10, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    
    output = model(data)
    assert output.shape == (1, 2)


def test_mpnn_model():
    model = MPNNModel(in_channels=2, edge_dim=1, hidden_channels=32, out_channels=2)
    x = torch.randn(10, 2)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    output = model(data)
    assert output.shape == (1, 2)


def test_gat_model():
    model = GATModel(in_channels=2, hidden_channels=32, out_channels=2)
    x = torch.randn(10, 2)
    edge_index = torch.randint(0, 10, (2, 20))
    data = Data(x=x, edge_index=edge_index)
    
    output = model(data)
    assert output.shape == (1, 2) 