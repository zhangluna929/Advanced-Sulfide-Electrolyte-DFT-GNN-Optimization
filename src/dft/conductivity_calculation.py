# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 电导率计算脚本

import os
import numpy as np

# 定义材料和掺杂元素，与数据收集脚本一致
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li6PS5Br', 'Li10GeP2S12', 'Li3PS4', 'Li7P3S11', 'Li4PS4I', 'Li7PS6', 'Li9PS6']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟电导率计算过程
def calculate_conductivity(material, dopant, concentration, position):
    # 模拟从DFT计算中获取能带结构和载流子浓度等信息
    bandgap = np.random.uniform(0.5, 4.0)  # 模拟带隙数据
    carrier_concentration = np.random.uniform(1e18, 1e21)  # 模拟载流子浓度
    
    # 使用Nernst-Einstein关系估算离子导电率
    # 假设扩散系数与载流子浓度和温度相关，这里简化为一个随机值
    diffusion_coefficient = np.random.uniform(1e-6, 1e-4)  # 模拟扩散系数
    temperature = 298  # 假设室温，单位：K
    charge = 1.6e-19  # 电子电荷，单位：C
    k_b = 1.38e-23  # 玻尔兹曼常数，单位：J/K
    conductivity = (carrier_concentration * charge**2 * diffusion_coefficient) / (k_b * temperature)
    
    return {
        'bandgap': bandgap,
        'carrier_concentration': carrier_concentration,
        'conductivity': conductivity
    }

# 数据存储
conductivity_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = calculate_conductivity(mat, dop, conc, pos)
                conductivity_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'bandgap': result['bandgap'],
                    'carrier_concentration': result['carrier_concentration'],
                    'conductivity': result['conductivity']
                })

# 打印电导率计算结果数量
print(f"总共进行了 {len(conductivity_results)} 次电导率计算。")

# 打印前几个计算结果作为示例
for result in conductivity_results[:5]:
    print(f"材料: {result['material']}, 掺杂元素: {result['dopant']}, 浓度: {result['concentration']}, 位置: {result['position']}, 带隙: {result['bandgap']}, 载流子浓度: {result['carrier_concentration']}, 电导率: {result['conductivity']}")

# 保存电导率计算结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'conductivity_results.txt'), 'w') as f:
    for result in conductivity_results:
        f.write(str(result) + '\n')

# 筛选电导率较高的结构
high_conductivity_structures = [result for result in conductivity_results if result['conductivity'] > 1e-3]
print(f"筛选出 {len(high_conductivity_structures)} 个电导率较高的结构。")

# 保存电导率较高的结构到单独文件
with open(os.path.join(output_dir, 'high_conductivity_structures.txt'), 'w') as f:
    for result in high_conductivity_structures:
        f.write(str(result) + '\n')

print("电导率计算完成，电导率较高的结构已筛选并保存到文件。") 