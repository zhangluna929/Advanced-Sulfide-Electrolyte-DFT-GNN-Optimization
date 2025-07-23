# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 能带结构计算脚本

import os
import numpy as np

# 定义材料和掺杂元素，与数据收集脚本一致
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟能带结构计算过程
def calculate_band_structure(material, dopant, concentration, position):
    # 模拟使用DFT工具计算能带结构
    bandgap = np.random.uniform(0.5, 4.0)  # 模拟带隙数据
    valence_band_max = np.random.uniform(-2.0, -0.5)  # 模拟价带最大值
    conduction_band_min = np.random.uniform(0.5, 2.0)  # 模拟导带最小值
    conductivity_impact = 'positive' if bandgap < 2.0 else 'negative'  # 简单判断掺杂对电导率的影响
    return {
        'bandgap': bandgap,
        'valence_band_max': valence_band_max,
        'conduction_band_min': conduction_band_min,
        'conductivity_impact': conductivity_impact
    }

# 数据存储
band_structure_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = calculate_band_structure(mat, dop, conc, pos)
                band_structure_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'bandgap': result['bandgap'],
                    'valence_band_max': result['valence_band_max'],
                    'conduction_band_min': result['conduction_band_min'],
                    'conductivity_impact': result['conductivity_impact']
                })

# 打印能带结构计算结果数量
print(f"总共进行了 {len(band_structure_results)} 次能带结构计算。")

# 打印前几个计算结果作为示例
for result in band_structure_results[:5]:
    print(f"材料: {result['material']}, 掺杂元素: {result['dopant']}, 浓度: {result['concentration']}, 位置: {result['position']}, 带隙: {result['bandgap']}, 价带最大值: {result['valence_band_max']}, 导带最小值: {result['conduction_band_min']}, 电导率影响: {result['conductivity_impact']}")

# 保存能带结构计算结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'band_structure_results.txt'), 'w') as f:
    for result in band_structure_results:
        f.write(str(result) + '\n')

# 筛选对电导率有正面影响的结构
positive_impact_structures = [result for result in band_structure_results if result['conductivity_impact'] == 'positive']
print(f"筛选出 {len(positive_impact_structures)} 个对电导率有正面影响的结构。")

# 保存对电导率有正面影响的结构到单独文件
with open(os.path.join(output_dir, 'positive_impact_structures.txt'), 'w') as f:
    for result in positive_impact_structures:
        f.write(str(result) + '\n')

print("能带结构计算完成，对电导率有正面影响的结构已筛选并保存到文件。") 