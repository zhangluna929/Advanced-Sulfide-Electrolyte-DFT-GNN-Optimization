"""
数据准备脚本

用于处理和准备掺杂预测所需的基础结构数据。
主要功能：
1. 从现有数据文件中提取基础结构信息
2. 处理和标准化特征
3. 保存处理后的数据
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import torch

def load_structure_data(file_path):
    """加载结构数据"""
    with open(file_path, 'r') as f:
        data = f.readlines()
    return [eval(line.strip()) for line in data]

def load_conductivity_data(file_path):
    """加载电导率数据"""
    return pd.read_csv(file_path, header=None).values

def process_base_structure(structures, conductivities, stability_data=None):
    """处理基础结构数据"""
    # 选择性能最好的结构作为基础
    if stability_data is not None:
        # 综合考虑电导率和稳定性
        normalized_conductivity = (conductivities - conductivities.min()) / (conductivities.max() - conductivities.min())
        normalized_stability = (stability_data - stability_data.min()) / (stability_data.max() - stability_data.min())
        performance_score = normalized_conductivity * 0.6 + normalized_stability * 0.4
        best_idx = performance_score.argmax()
    else:
        # 只考虑电导率
        best_idx = conductivities.argmax()
    
    base_structure = structures[best_idx]
    
    # 提取特征
    node_features = np.array(base_structure['node_features'])
    edge_index = np.array(base_structure['edge_index'])
    
    # 标准化特征
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(node_features)
    
    # 构建基础结构字典
    processed_structure = {
        'node_features': normalized_features.tolist(),
        'edge_index': edge_index.tolist(),
        'original_conductivity': float(conductivities[best_idx])
    }
    
    if stability_data is not None:
        processed_structure['original_stability'] = float(stability_data[best_idx])
    
    return processed_structure, scaler

def main():
    # 设置路径
    data_dir = Path('data')
    output_dir = Path('data/base_structures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    structures = load_structure_data(data_dir / 'graph_data.txt')
    conductivities = load_conductivity_data(data_dir / 'conductivity_results.txt')
    
    try:
        stability_data = load_conductivity_data(data_dir / 'conductivity_stability_results.txt')
    except:
        stability_data = None
        print("未找到稳定性数据，将只考虑电导率进行选择")
    
    # 处理数据
    base_structure, scaler = process_base_structure(structures, conductivities, stability_data)
    
    # 保存处理后的数据
    with open(output_dir / 'base_structure.json', 'w') as f:
        json.dump(base_structure, f, indent=2)
    
    with open(output_dir / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("数据处理完成！")
    print(f"基础结构保存至: {output_dir / 'base_structure.json'}")
    print(f"特征标准化器保存至: {output_dir / 'feature_scaler.pkl'}")

if __name__ == '__main__':
    main() 