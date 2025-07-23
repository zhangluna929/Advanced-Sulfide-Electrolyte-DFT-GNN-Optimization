# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 结构优化脚本

import os
import numpy as np

# 定义材料和掺杂元素，与数据收集脚本一致
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟结构优化过程
def optimize_structure(material, dopant, concentration, position):
    # 模拟使用VASP或QuantumESPRESSO进行结构优化
    energy = np.random.uniform(-100, -50)  # 模拟优化后的能量
    stability_score = np.random.uniform(0.8, 1.0)  # 模拟稳定性得分，值越高越稳定
    return {'energy': energy, 'stability_score': stability_score}

# 数据存储
optimization_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = optimize_structure(mat, dop, conc, pos)
                optimization_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'energy': result['energy'],
                    'stability_score': result['stability_score']
                })

# 打印优化结果数量
print(f"总共进行了 {len(optimization_results)} 次结构优化。")

# 打印前几个优化结果作为示例
for result in optimization_results[:5]:
    print(f"材料: {result['material']}, 掺杂元素: {result['dopant']}, 浓度: {result['concentration']}, 位置: {result['position']}, 能量: {result['energy']}, 稳定性得分: {result['stability_score']}")

# 保存优化结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'structure_optimization_results.txt'), 'w') as f:
    for result in optimization_results:
        f.write(str(result) + '\n')

# 筛选稳定性得分高的结构
stable_structures = [result for result in optimization_results if result['stability_score'] >= 0.9]
print(f"筛选出 {len(stable_structures)} 个稳定性得分大于等于0.9的结构。")

# 保存稳定性高的结构到单独文件
with open(os.path.join(output_dir, 'stable_structures.txt'), 'w') as f:
    for result in stable_structures:
        f.write(str(result) + '\n')

print("结构优化完成，稳定性高的结构已筛选并保存到文件。") 