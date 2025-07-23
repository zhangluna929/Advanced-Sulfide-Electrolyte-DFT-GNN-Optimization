# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 数据收集与处理脚本

import os
import numpy as np

# 定义材料和掺杂元素，扩展列表以确保数据量
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟DFT计算结果收集
def collect_dft_data(material, dopant, concentration, position):
    # 模拟结构优化、能带结构计算、电导率计算等
    bandgap = np.random.uniform(0.5, 4.0)  # 模拟带隙数据
    conductivity = np.random.uniform(0.01, 20.0)  # 模拟电导率数据
    structure_stability = np.random.uniform(0.8, 1.0)  # 模拟结构稳定性
    return {'bandgap': bandgap, 'conductivity': conductivity, 'stability': structure_stability}

# 数据存储
data_collection = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                data = collect_dft_data(mat, dop, conc, pos)
                data_collection.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'bandgap': data['bandgap'],
                    'conductivity': data['conductivity'],
                    'stability': data['stability']
                })

# 打印数据量
print(f"总共收集了 {len(data_collection)} 个数据点。")

# 打印前几个数据作为示例
for entry in data_collection[:5]:
    print(f"材料: {entry['material']}, 掺杂元素: {entry['dopant']}, 浓度: {entry['concentration']}, 位置: {entry['position']}, 带隙: {entry['bandgap']}, 电导率: {entry['conductivity']}, 稳定性: {entry['stability']}")

# 转换为GNN图数据格式
def convert_to_graph_data(data):
    # 模拟将结构数据转换为图数据
    graph_data = {
        'nodes': [],  # 原子作为节点
        'edges': [],  # 原子间连接作为边
        'labels': {'bandgap': data['bandgap'], 'conductivity': data['conductivity'], 'stability': data['stability']}
    }
    return graph_data

# 处理数据并保存
graph_data_collection = [convert_to_graph_data(entry) for entry in data_collection]

# 保存数据到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'graph_data.txt'), 'w') as f:
    for graph in graph_data_collection:
        f.write(str(graph) + '\n')

print("数据收集和转换完成，已保存到文件。") 