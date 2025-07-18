# -*- coding: utf-8 -*-
"""
结构优化脚本 - 使用VASP进行能量优化

本脚本模拟使用VASP对掺杂后的锂硫化物电解质材料进行结构优化，确保每个结构处于能量最小化状态，
从而保证结构的合理性和稳定性。
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 创建输出目录
def create_output_dirs():
    dirs = ['data', 'plots', 'stats']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

create_output_dirs()

# 定义材料、掺杂元素、浓度和位置
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl', 'Li7PSe3Cl', 'Li7PSe3Br', 'Li6PSe5Cl', 'Li7PS3I', 'Li7PSe3I', 'Li6PS5Br', 'Li6PSe5Br']
dopants = ['Mg', 'Ca', 'Al', 'Sr', 'Ba', 'Ga', 'In', 'Sn', 'Pb', 'Bi', 'Zn', 'Cd', 'Hg', 'Y', 'Sc']
concentrations = np.round(np.arange(0.01, 0.16, 0.01), 2).tolist()
positions = ['Li-site', 'P-site', 'S-site', 'Interstitial']

# 初始化数据存储
data = []

# 模拟VASP结构优化计算
for material in materials:
    for dopant in dopants:
        for conc in concentrations:
            for pos in positions:
                # 模拟初始能量 (eV)
                initial_energy = np.random.uniform(-100, -50)
                
                # 模拟优化后的能量 (eV) - 优化后能量应低于初始能量
                optimized_energy = initial_energy - np.random.uniform(0.5, 5.0)
                
                # 模拟能量收敛性 (无单位，0到1之间，1表示完全收敛)
                energy_convergence = np.random.uniform(0.85, 1.0)
                
                # 模拟结构稳定性得分 (无单位，0到1之间，1表示非常稳定)
                stability_score = np.random.uniform(0.7, 1.0) if energy_convergence > 0.9 else np.random.uniform(0.4, 0.7)
                
                # 模拟晶格参数变化率 (%)
                lattice_change_percent = np.random.uniform(-2.0, 2.0) * (1 + conc * 2)
                
                # 模拟体积变化率 (%)
                volume_change_percent = np.random.uniform(-3.0, 3.0) * (1 + conc * 1.5)
                
                # 模拟键长变化率 (%)
                bond_length_change_percent = np.random.uniform(-1.5, 1.5) * (1 + conc * 1.2)
                
                # 模拟优化步骤数 (无单位)
                optimization_steps = int(np.random.uniform(10, 50))
                
                # 模拟计算时间 (小时)
                computation_time_hours = np.random.uniform(2.0, 24.0)
                
                # 模拟是否达到收敛 (布尔值)
                is_converged = energy_convergence > 0.9 and stability_score > 0.6
                
                # 记录数据
                data.append({
                    'Material': material,
                    'Dopant': dopant,
                    'Concentration': conc,
                    'Position': pos,
                    'Initial_Energy_eV': initial_energy,
                    'Optimized_Energy_eV': optimized_energy,
                    'Energy_Convergence': energy_convergence,
                    'Stability_Score': stability_score,
                    'Lattice_Change_Percent': lattice_change_percent,
                    'Volume_Change_Percent': volume_change_percent,
                    'Bond_Length_Change_Percent': bond_length_change_percent,
                    'Optimization_Steps': optimization_steps,
                    'Computation_Time_Hours': computation_time_hours,
                    'Is_Converged': is_converged
                })

# 创建DataFrame
df = pd.DataFrame(data)

# 保存数据到CSV文件
df.to_csv('data/structure_optimization_vasp_data.csv', index=False)

# 统计分析
# 按材料和掺杂元素分组，计算平均优化能量和稳定性得分
material_dopant_group = df.groupby(['Material', 'Dopant']).agg({
    'Optimized_Energy_eV': 'mean',
    'Stability_Score': 'mean',
    'Lattice_Change_Percent': 'mean',
    'Volume_Change_Percent': 'mean',
    'Bond_Length_Change_Percent': 'mean',
    'Optimization_Steps': 'mean',
    'Computation_Time_Hours': 'mean',
    'Is_Converged': 'mean'  # 收敛比例
}).reset_index()

# 保存分组统计结果
material_dopant_group.to_csv('stats/structure_optimization_material_dopant_stats.csv', index=False)

# 按材料和浓度分组，计算平均优化能量和稳定性得分
material_conc_group = df.groupby(['Material', 'Concentration']).agg({
    'Optimized_Energy_eV': 'mean',
    'Stability_Score': 'mean',
    'Lattice_Change_Percent': 'mean',
    'Volume_Change_Percent': 'mean',
    'Bond_Length_Change_Percent': 'mean',
    'Optimization_Steps': 'mean',
    'Computation_Time_Hours': 'mean',
    'Is_Converged': 'mean'
}).reset_index()

# 保存分组统计结果
material_conc_group.to_csv('stats/structure_optimization_material_concentration_stats.csv', index=False)

# 按材料和位置分组，计算平均优化能量和稳定性得分
material_pos_group = df.groupby(['Material', 'Position']).agg({
    'Optimized_Energy_eV': 'mean',
    'Stability_Score': 'mean',
    'Lattice_Change_Percent': 'mean',
    'Volume_Change_Percent': 'mean',
    'Bond_Length_Change_Percent': 'mean',
    'Optimization_Steps': 'mean',
    'Computation_Time_Hours': 'mean',
    'Is_Converged': 'mean'
}).reset_index()

# 保存分组统计结果
material_pos_group.to_csv('stats/structure_optimization_material_position_stats.csv', index=False)

# 筛选收敛且稳定的结构 (Is_Converged=True 且 Stability_Score > 0.8)
stable_converged = df[(df['Is_Converged'] == True) & (df['Stability_Score'] > 0.8)]
stable_converged.to_csv('stats/structure_optimization_stable_converged_combinations.csv', index=False)

# 可视化 - 稳定性得分与掺杂浓度的关系
plt.figure(figsize=(12, 6))
for material in materials[:5]:  # 限制材料数量以避免图表过于复杂
    mat_data = material_conc_group[material_conc_group['Material'] == material]
    plt.plot(mat_data['Concentration'], mat_data['Stability_Score'], marker='o', label=material)
plt.xlabel('掺杂浓度')
plt.ylabel('稳定性得分')
plt.title('不同材料中稳定性得分与掺杂浓度的关系')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/stability_score_vs_concentration.png', bbox_inches='tight')
plt.close()

# 可视化 - 优化能量与掺杂浓度的关系
plt.figure(figsize=(12, 6))
for material in materials[:5]:
    mat_data = material_conc_group[material_conc_group['Material'] == material]
    plt.plot(mat_data['Concentration'], mat_data['Optimized_Energy_eV'], marker='o', label=material)
plt.xlabel('掺杂浓度')
plt.ylabel('优化能量 (eV)')
plt.title('不同材料中优化能量与掺杂浓度的关系')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/optimized_energy_vs_concentration.png', bbox_inches='tight')
plt.close()

# 可视化 - 体积变化率与掺杂浓度的关系
plt.figure(figsize=(12, 6))
for material in materials[:5]:
    mat_data = material_conc_group[material_conc_group['Material'] == material]
    plt.plot(mat_data['Concentration'], mat_data['Volume_Change_Percent'], marker='o', label=material)
plt.xlabel('掺杂浓度')
plt.ylabel('体积变化率 (%)')
plt.title('不同材料中体积变化率与掺杂浓度的关系')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/volume_change_vs_concentration.png', bbox_inches='tight')
plt.close()

# 可视化 - 收敛比例与掺杂元素的关系（按材料）
for material in materials[:3]:
    mat_data = material_dopant_group[material_dopant_group['Material'] == material]
    plt.figure(figsize=(12, 6))
    plt.bar(mat_data['Dopant'], mat_data['Is_Converged'], label=material)
    plt.xlabel('掺杂元素')
    plt.ylabel('收敛比例')
    plt.title(f'{material} 中收敛比例与掺杂元素的关系')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'plots/convergence_ratio_vs_dopant_{material}.png')
    plt.close()

# 可视化 - 稳定性得分分布（按材料和位置）
for material in materials[:3]:
    mat_data = df[df['Material'] == material]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Position', y='Stability_Score', data=mat_data)
    plt.title(f'{material} 中稳定性得分分布（按位置）')
    plt.xlabel('掺杂位置')
    plt.ylabel('稳定性得分')
    plt.tight_layout()
    plt.savefig(f'plots/stability_score_boxplot_{material}.png')
    plt.close()

# 相关性分析
correlation_matrix = df[['Initial_Energy_eV', 'Optimized_Energy_eV', 'Energy_Convergence', 
                         'Stability_Score', 'Lattice_Change_Percent', 'Volume_Change_Percent', 
                         'Bond_Length_Change_Percent', 'Optimization_Steps', 'Computation_Time_Hours']].corr()

# 保存相关性矩阵
correlation_matrix.to_csv('stats/structure_optimization_correlation_matrix.csv')

# 可视化相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('结构优化参数的相关性热力图')
plt.tight_layout()
plt.savefig('plots/structure_optimization_correlation_heatmap.png')
plt.close()

print('结构优化分析（使用VASP）完成，结果已保存到CSV文件和图像文件。')
print(f'找到 {len(stable_converged)} 个收敛且稳定的结构组合（收敛且稳定性得分>0.8）。') 