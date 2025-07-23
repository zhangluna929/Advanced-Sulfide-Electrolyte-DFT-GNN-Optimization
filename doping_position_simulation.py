# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 掺杂位置模拟脚本

import os
import numpy as np

# 定义材料和掺杂元素，与数据收集脚本一致
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟掺杂位置对结构和性能的影响
def simulate_doping_position(material, dopant, concentration, position):
    # 模拟使用VASP或其他DFT工具分析掺杂位置的影响
    energy_change = np.random.uniform(-5.0, 5.0)  # 模拟能量变化
    stability_score = np.random.uniform(0.7, 1.0)  # 模拟稳定性得分
    conductivity_impact = np.random.uniform(-0.5, 0.5)  # 模拟对电导率的影响
    return {
        'energy_change': energy_change,
        'stability_score': stability_score,
        'conductivity_impact': conductivity_impact
    }

# 数据存储
doping_position_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = simulate_doping_position(mat, dop, conc, pos)
                doping_position_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'energy_change': result['energy_change'],
                    'stability_score': result['stability_score'],
                    'conductivity_impact': result['conductivity_impact']
                })

# 打印掺杂位置模拟结果数量
print(f"总共进行了 {len(doping_position_results)} 次掺杂位置模拟。")

# 打印前几个模拟结果作为示例
for result in doping_position_results[:5]:
    print(f"材料: {result['material']}, 掺杂元素: {result['dopant']}, 浓度: {result['concentration']}, 位置: {result['position']}, 能量变化: {result['energy_change']}, 稳定性得分: {result['stability_score']}, 电导率影响: {result['conductivity_impact']}")

# 保存掺杂位置模拟结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'doping_position_results.txt'), 'w') as f:
    for result in doping_position_results:
        f.write(str(result) + '\n')

# 筛选对电导率有正面影响且稳定性高的结构
optimal_doping_positions = [result for result in doping_position_results if result['conductivity_impact'] > 0 and result['stability_score'] >= 0.9]
print(f"筛选出 {len(optimal_doping_positions)} 个对电导率有正面影响且稳定性高的掺杂位置。")

# 保存最佳掺杂位置到单独文件
with open(os.path.join(output_dir, 'optimal_doping_positions.txt'), 'w') as f:
    for result in optimal_doping_positions:
        f.write(str(result) + '\n')

print("掺杂位置模拟完成，最佳掺杂位置已筛选并保存到文件。") 