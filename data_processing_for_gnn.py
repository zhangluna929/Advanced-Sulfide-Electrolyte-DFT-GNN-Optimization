# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 数据处理与转换脚本

import os
import numpy as np
import json

# 定义材料和掺杂元素，与数据收集脚本一致
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟从之前的结果中读取数据
def load_simulation_data():
    # 假设从之前的模拟结果中读取数据
    simulation_data = []
    for mat in materials:
        for dop in dopants:
            for conc in concentrations:
                for pos in positions:
                    simulation_data.append({
                        'material': mat,
                        'dopant': dop,
                        'concentration': conc,
                        'position': pos,
                        'bandgap': np.random.uniform(0.5, 4.0),
                        'conductivity': np.random.uniform(0.01, 20.0),
                        'stability': np.random.uniform(0.7, 1.0)
                    })
    return simulation_data

# 将结构数据转换为图数据格式
def convert_to_graph_data(entry):
    # 模拟将DFT结构数据（如原子电荷密度、原子间距、化学键、原子类型等）转化为图数据
    num_atoms = np.random.randint(10, 50)  # 模拟原子数量
    nodes = [{'atom_type': np.random.choice(['Li', 'P', 'S', 'Cl', 'Br', 'Ge', 'I']), 
              'charge_density': np.random.uniform(0.1, 1.0)} for _ in range(num_atoms)]
    edges = [(i, j, np.random.uniform(1.0, 3.0)) for i in range(num_atoms) for j in range(i+1, num_atoms) if np.random.rand() > 0.7]
    graph_data = {
        'nodes': nodes,
        'edges': edges,
        'labels': {
            'bandgap': entry['bandgap'],
            'conductivity': entry['conductivity'],
            'stability': entry['stability']
        },
        'metadata': {
            'material': entry['material'],
            'dopant': entry['dopant'],
            'concentration': entry['concentration'],
            'position': entry['position']
        }
    }
    return graph_data

# 加载模拟数据
simulation_data = load_simulation_data()

# 转换为图数据
graph_data_collection = [convert_to_graph_data(entry) for entry in simulation_data]

# 打印转换结果数量
print(f"总共转换了 {len(graph_data_collection)} 个数据点为图数据格式。")

# 保存图数据到文件
output_dir = '硫化物DFTGNN/data/gnn_data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'graph_data.json'), 'w') as f:
    json.dump(graph_data_collection, f, indent=2)

# 准备GNN监督学习任务的标签数据
labels_data = [{'bandgap': entry['labels']['bandgap'], 
                'conductivity': entry['labels']['conductivity'], 
                'stability': entry['labels']['stability'],
                'metadata': entry['metadata']} for entry in graph_data_collection]

# 保存标签数据到文件
with open(os.path.join(output_dir, 'labels_data.json'), 'w') as f:
    json.dump(labels_data, f, indent=2)

print("数据处理与转换完成，图数据和标签数据已保存到文件，准备好用于GNN监督学习任务。") 