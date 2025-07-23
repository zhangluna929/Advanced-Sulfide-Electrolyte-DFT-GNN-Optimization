"""subgraph_sampling.py
子图采样与大图处理
=================
• Cluster-GCN 子图采样
• METIS 图分割 (可选)
• FastGCN 节点采样
• GraphSAINT 边采样
"""
from __future__ import annotations

import random
from typing import List, Tuple, Dict
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import subgraph, to_networkx, from_networkx
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.transforms import ToUndirected

try:
    import networkx as nx
    import metis  # pip install metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    nx = None
    metis = None


class ClusterGCNSampler:
    """Cluster-GCN 子图采样器"""
    
    def __init__(
        self,
        num_parts: int = 1000,
        recursive: bool = False,
        save_dir: str = None,
    ):
        self.num_parts = num_parts
        self.recursive = recursive
        self.save_dir = save_dir
    
    def cluster_data(self, data: Data) -> ClusterData:
        """对图数据进行聚类分割"""
        
        # 确保图是无向的 (METIS 要求)
        transform = ToUndirected()
        data = transform(data)
        
        cluster_data = ClusterData(
            data,
            num_parts=self.num_parts,
            recursive=self.recursive,
            save_dir=self.save_dir,
        )
        
        return cluster_data
    
    def create_loader(
        self,
        cluster_data: ClusterData,
        batch_size: int = 1,
        shuffle: bool = True,
        **kwargs,
    ) -> ClusterLoader:
        """创建子图数据加载器"""
        
        loader = ClusterLoader(
            cluster_data,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        
        return loader


class METISPartitioner:
    """METIS 图分割器 (需要 metis 包)"""
    
    def __init__(self, num_parts: int = 10):
        if not METIS_AVAILABLE:
            raise ImportError("METIS not available. Install with: pip install metis")
        
        self.num_parts = num_parts
    
    def partition_graph(self, data: Data) -> List[torch.Tensor]:
        """使用 METIS 分割图"""
        
        # 转换为 NetworkX 图
        G = to_networkx(data, to_undirected=True)
        
        # 转换为 METIS 格式
        adjacency_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
        
        # METIS 分割
        cuts, parts = metis.part_graph(adjacency_list, nparts=self.num_parts)
        
        # 按分区组织节点
        partitions = [[] for _ in range(self.num_parts)]
        for node_id, part_id in enumerate(parts):
            partitions[part_id].append(node_id)
        
        # 转换为 tensor
        partition_tensors = [torch.tensor(part, dtype=torch.long) for part in partitions if part]
        
        return partition_tensors
    
    def create_subgraphs(self, data: Data, partitions: List[torch.Tensor]) -> List[Data]:
        """根据分区创建子图"""
        subgraphs = []
        
        for partition in partitions:
            if len(partition) == 0:
                continue
            
            # 提取子图
            edge_index, edge_attr = subgraph(
                partition, data.edge_index, data.edge_attr, 
                relabel_nodes=True, num_nodes=data.num_nodes
            )
            
            # 创建子图数据
            sub_data = Data(
                x=data.x[partition],
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=data.y,  # 图级标签保持不变
                num_nodes=len(partition),
            )
            
            # 复制其他属性
            for key, value in data:
                if key not in ['x', 'edge_index', 'edge_attr', 'y', 'num_nodes']:
                    setattr(sub_data, key, value)
            
            subgraphs.append(sub_data)
        
        return subgraphs


class FastGCNSampler:
    """FastGCN 节点采样器"""
    
    def __init__(self, sample_sizes: List[int] = [25, 10]):
        """
        Args:
            sample_sizes: 每层采样的节点数 [layer1, layer2, ...]
        """
        self.sample_sizes = sample_sizes
    
    def sample_nodes(self, data: Data, batch_nodes: torch.Tensor) -> Data:
        """对给定的批次节点进行邻居采样"""
        
        current_nodes = batch_nodes
        all_nodes = [current_nodes]
        
        # 逐层采样
        for sample_size in reversed(self.sample_sizes):
            # 获取当前节点的邻居
            neighbors = self._get_neighbors(data.edge_index, current_nodes)
            
            # 采样邻居
            if len(neighbors) > sample_size:
                sampled = torch.tensor(
                    random.sample(neighbors.tolist(), sample_size),
                    dtype=torch.long
                )
            else:
                sampled = neighbors
            
            current_nodes = torch.cat([current_nodes, sampled]).unique()
            all_nodes.append(sampled)
        
        # 合并所有节点
        sampled_nodes = torch.cat(all_nodes).unique()
        
        # 提取子图
        edge_index, edge_attr = subgraph(
            sampled_nodes, data.edge_index, data.edge_attr,
            relabel_nodes=True, num_nodes=data.num_nodes
        )
        
        sub_data = Data(
            x=data.x[sampled_nodes],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y,
            num_nodes=len(sampled_nodes),
        )
        
        return sub_data
    
    def _get_neighbors(self, edge_index: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """获取节点的邻居"""
        mask = torch.isin(edge_index[0], nodes)
        neighbors = edge_index[1][mask].unique()
        return neighbors


class GraphSAINTSampler:
    """GraphSAINT 边采样器"""
    
    def __init__(self, sample_coverage: float = 0.1):
        """
        Args:
            sample_coverage: 采样覆盖率 (0-1)
        """
        self.sample_coverage = sample_coverage
    
    def sample_edges(self, data: Data) -> Data:
        """随机采样边"""
        num_edges = data.edge_index.size(1)
        num_sample = int(num_edges * self.sample_coverage)
        
        # 随机选择边
        edge_indices = torch.randperm(num_edges)[:num_sample]
        sampled_edge_index = data.edge_index[:, edge_indices]
        sampled_edge_attr = data.edge_attr[edge_indices] if data.edge_attr is not None else None
        
        # 获取涉及的节点
        nodes = sampled_edge_index.unique()
        
        # 重新标号
        node_map = {old_id.item(): new_id for new_id, old_id in enumerate(nodes)}
        new_edge_index = torch.tensor([
            [node_map[src.item()], node_map[dst.item()]]
            for src, dst in sampled_edge_index.t()
        ], dtype=torch.long).t()
        
        sub_data = Data(
            x=data.x[nodes],
            edge_index=new_edge_index,
            edge_attr=sampled_edge_attr,
            y=data.y,
            num_nodes=len(nodes),
        )
        
        return sub_data


class SubgraphDataset:
    """子图数据集管理器"""
    
    def __init__(
        self,
        original_data: Data,
        sampling_method: str = "cluster",
        **sampler_kwargs,
    ):
        self.original_data = original_data
        self.sampling_method = sampling_method
        
        # 初始化采样器
        if sampling_method == "cluster":
            self.sampler = ClusterGCNSampler(**sampler_kwargs)
        elif sampling_method == "metis":
            self.sampler = METISPartitioner(**sampler_kwargs)
        elif sampling_method == "fastgcn":
            self.sampler = FastGCNSampler(**sampler_kwargs)
        elif sampling_method == "graphsaint":
            self.sampler = GraphSAINTSampler(**sampler_kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    def create_subgraphs(self) -> List[Data]:
        """创建子图列表"""
        if self.sampling_method == "cluster":
            cluster_data = self.sampler.cluster_data(self.original_data)
            return [cluster_data[i] for i in range(len(cluster_data))]
        
        elif self.sampling_method == "metis":
            partitions = self.sampler.partition_graph(self.original_data)
            return self.sampler.create_subgraphs(self.original_data, partitions)
        
        elif self.sampling_method in ["fastgcn", "graphsaint"]:
            # 这些方法需要在训练时动态采样
            return [self.original_data]  # 返回原图，在 DataLoader 中动态采样
        
        else:
            raise NotImplementedError
    
    def create_loader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        if self.sampling_method == "cluster":
            cluster_data = self.sampler.cluster_data(self.original_data)
            return self.sampler.create_loader(cluster_data, batch_size, shuffle)
        else:
            subgraphs = self.create_subgraphs()
            return DataLoader(subgraphs, batch_size=batch_size, shuffle=shuffle)


# 使用示例
def demo_subgraph_sampling():
    """子图采样演示"""
    
    # 创建示例图数据
    num_nodes = 1000
    x = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))
    y = torch.randn(2)  # 图级标签
    
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    
    print(f"原图: {num_nodes} 节点, {edge_index.size(1)} 边")
    
    # 1. Cluster-GCN
    print("\n=== Cluster-GCN ===")
    cluster_dataset = SubgraphDataset(data, "cluster", num_parts=10)
    cluster_loader = cluster_dataset.create_loader(batch_size=2)
    
    for i, batch in enumerate(cluster_loader):
        if i < 3:  # 只看前3个batch
            print(f"Batch {i}: {batch.num_nodes} 节点, {batch.edge_index.size(1)} 边")
    
    # 2. METIS (如果可用)
    if METIS_AVAILABLE:
        print("\n=== METIS ===")
        metis_dataset = SubgraphDataset(data, "metis", num_parts=5)
        subgraphs = metis_dataset.create_subgraphs()
        
        for i, subgraph in enumerate(subgraphs[:3]):
            print(f"子图 {i}: {subgraph.num_nodes} 节点, {subgraph.edge_index.size(1)} 边")
    
    # 3. GraphSAINT
    print("\n=== GraphSAINT ===")
    saint_sampler = GraphSAINTSampler(sample_coverage=0.2)
    saint_subgraph = saint_sampler.sample_edges(data)
    print(f"GraphSAINT 子图: {saint_subgraph.num_nodes} 节点, {saint_subgraph.edge_index.size(1)} 边")


if __name__ == "__main__":
    demo_subgraph_sampling() 