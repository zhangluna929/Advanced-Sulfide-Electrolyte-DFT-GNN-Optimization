# -*- coding: utf-8 -*-
"""
DFT数据处理脚本

本脚本处理DFT计算得到的结构信息，提取材料特征，并进行统计分析。
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

# 定义原子类型和可能的化学键类型
atom_types = ['Li', 'P', 'S', 'Se', 'Cl', 'Br', 'I'] + dopants
bond_types = ['Li-S', 'Li-Se', 'Li-Cl', 'Li-Br', 'Li-I', 'P-S', 'P-Se', 'S-S', 'Se-Se']
for dopant in dopants:
    bond_types.extend([f'{dopant}-S', f'{dopant}-Se', f'{dopant}-Li'])

# 初始化数据存储
material_data = []
atom_features_data = []
bond_features_data = []

# 模拟DFT结构数据的处理
for material in materials:
    for dopant in dopants:
        for conc in concentrations:
            for pos in positions:
                # 模拟晶体结构中的原子数量 (根据材料和掺杂浓度调整)
                num_atoms = int(np.random.uniform(20, 50) * (1 + conc * 0.5))
                
                # 模拟材料ID
                material_id = f'{material}_{dopant}_{conc}_{pos}'
                
                # 模拟原子类型分布
                atoms = np.random.choice(atom_types, size=num_atoms, replace=True)
                
                # 模拟原子位置 (x, y, z坐标，单位：Angstrom)
                positions_array = np.random.uniform(0, 10, size=(num_atoms, 3))
                
                # 模拟化学键 (边)
                num_bonds = int(num_atoms * np.random.uniform(1.5, 3.0))
                bonds = []
                for _ in range(num_bonds):
                    atom1_idx = np.random.randint(0, num_atoms)
                    atom2_idx = np.random.randint(0, num_atoms)
                    while atom2_idx == atom1_idx:
                        atom2_idx = np.random.randint(0, num_atoms)
                    bond_type = np.random.choice(bond_types)
                    bond_length = np.random.uniform(1.5, 3.5)
                    bonds.append((atom1_idx, atom2_idx, bond_type, bond_length))
                
                # 记录材料数据
                material_data.append({
                    'Material_ID': material_id,
                    'Material': material,
                    'Dopant': dopant,
                    'Concentration': conc,
                    'Position': pos,
                    'Num_Atoms': num_atoms,
                    'Num_Bonds': num_bonds
                })
                
                # 模拟原子特征 (每个原子的局部环境特征)
                for atom_idx in range(num_atoms):
                    atom_type = atoms[atom_idx]
                    atom_pos = positions_array[atom_idx]
                    # 模拟原子电荷 (e)
                    atom_charge = np.random.uniform(-1.0, 1.0)
                    # 模拟局部协调数 (无单位)
                    coordination_number = np.random.randint(2, 8)
                    # 模拟平均键长 (Angstrom)
                    avg_bond_length = np.random.uniform(1.8, 3.2)
                    # 模拟平均键角 (度)
                    avg_bond_angle = np.random.uniform(90, 120)
                    # 模拟局部密度 (无单位)
                    local_density = np.random.uniform(0.5, 1.5)
                    # 模拟局部电势 (eV)
                    local_potential = np.random.uniform(-5.0, 5.0)
                    
                    atom_features_data.append({
                        'Material_ID': material_id,
                        'Atom_Index': atom_idx,
                        'Atom_Type': atom_type,
                        'Pos_X': atom_pos[0],
                        'Pos_Y': atom_pos[1],
                        'Pos_Z': atom_pos[2],
                        'Atom_Charge_e': atom_charge,
                        'Coordination_Number': coordination_number,
                        'Avg_Bond_Length_Angstrom': avg_bond_length,
                        'Avg_Bond_Angle_Degrees': avg_bond_angle,
                        'Local_Density': local_density,
                        'Local_Potential_eV': local_potential
                    })
                
                # 模拟化学键特征
                for bond_idx, (atom1_idx, atom2_idx, bond_type, bond_length) in enumerate(bonds):
                    bond_features_data.append({
                        'Material_ID': material_id,
                        'Bond_Index': bond_idx,
                        'Atom1_Index': atom1_idx,
                        'Atom2_Index': atom2_idx,
                        'Bond_Type': bond_type,
                        'Bond_Length_Angstrom': bond_length
                    })

# 创建DataFrame
material_df = pd.DataFrame(material_data)
atom_features_df = pd.DataFrame(atom_features_data)
bond_features_df = pd.DataFrame(bond_features_data)

# 保存数据到CSV文件
material_df.to_csv('data/graph_data.csv', index=False)
atom_features_df.to_csv('data/node_features_data.csv', index=False)
bond_features_df.to_csv('data/edge_features_data.csv', index=False)

# 统计分析
# 按材料和掺杂元素分组，计算平均原子数和化学键数
material_dopant_group = material_df.groupby(['Material', 'Dopant']).agg({
    'Num_Atoms': 'mean',
    'Num_Bonds': 'mean'
}).reset_index()

# 保存分组统计结果
material_dopant_group.to_csv('stats/graph_material_dopant_stats.csv', index=False)

# 按材料和浓度分组，计算平均原子数和化学键数
material_conc_group = material_df.groupby(['Material', 'Concentration']).agg({
    'Num_Atoms': 'mean',
    'Num_Bonds': 'mean'
}).reset_index()

# 保存分组统计结果
material_conc_group.to_csv('stats/graph_material_concentration_stats.csv', index=False)

# 原子特征统计 - 按材料分组，计算平均原子电荷、协调数等
atom_material_group = atom_features_df.merge(material_df[['Material_ID', 'Material']], on='Material_ID')
atom_material_stats = atom_material_group.groupby('Material').agg({
    'Atom_Charge_e': 'mean',
    'Coordination_Number': 'mean',
    'Avg_Bond_Length_Angstrom': 'mean',
    'Avg_Bond_Angle_Degrees': 'mean',
    'Local_Density': 'mean',
    'Local_Potential_eV': 'mean'
}).reset_index()

# 保存原子特征统计结果
atom_material_stats.to_csv('stats/node_features_material_stats.csv', index=False)

# 化学键特征统计 - 按材料分组，计算平均键长
bond_material_group = bond_features_df.merge(material_df[['Material_ID', 'Material']], on='Material_ID')
bond_material_stats = bond_material_group.groupby('Material').agg({
    'Bond_Length_Angstrom': 'mean'
}).reset_index()

# 保存化学键特征统计结果
bond_material_stats.to_csv('stats/edge_features_material_stats.csv', index=False)

# 可视化 - 原子数与掺杂浓度的关系
plt.figure(figsize=(12, 6))
for material in materials[:5]:  # 限制材料数量以避免图表过于复杂
    mat_data = material_conc_group[material_conc_group['Material'] == material]
    plt.plot(mat_data['Concentration'], mat_data['Num_Atoms'], marker='o', label=material)
plt.xlabel('掺杂浓度')
plt.ylabel('平均原子数')
plt.title('不同材料中平均原子数与掺杂浓度的关系')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/num_atoms_vs_concentration.png', bbox_inches='tight')
plt.close()

# 可视化 - 化学键数与掺杂浓度的关系
plt.figure(figsize=(12, 6))
for material in materials[:5]:
    mat_data = material_conc_group[material_conc_group['Material'] == material]
    plt.plot(mat_data['Concentration'], mat_data['Num_Bonds'], marker='o', label=material)
plt.xlabel('掺杂浓度')
plt.ylabel('平均化学键数')
plt.title('不同材料中平均化学键数与掺杂浓度的关系')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/num_bonds_vs_concentration.png', bbox_inches='tight')
plt.close()

# 可视化 - 原子电荷分布（按材料）
for material in materials[:3]:
    mat_data = atom_material_group[atom_material_group['Material'] == material]
    plt.figure(figsize=(10, 6))
    plt.hist(mat_data['Atom_Charge_e'], bins=30, density=True, alpha=0.7)
    plt.xlabel('原子电荷 (e)')
    plt.ylabel('频率密度')
    plt.title(f'{material} 中原子电荷分布')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/atom_charge_distribution_{material}.png')
    plt.close()

# 可视化 - 平均键长分布（按材料）
for material in materials[:3]:
    mat_data = bond_material_group[bond_material_group['Material'] == material]
    plt.figure(figsize=(10, 6))
    plt.hist(mat_data['Bond_Length_Angstrom'], bins=30, density=True, alpha=0.7)
    plt.xlabel('键长 (Angstrom)')
    plt.ylabel('频率密度')
    plt.title(f'{material} 中键长分布')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/bond_length_distribution_{material}.png')
    plt.close()

# 可视化 - 协调数分布（按材料）
for material in materials[:3]:
    mat_data = atom_material_group[atom_material_group['Material'] == material]
    plt.figure(figsize=(10, 6))
    plt.hist(mat_data['Coordination_Number'], bins=range(1, 10), density=True, alpha=0.7)
    plt.xlabel('协调数')
    plt.ylabel('频率密度')
    plt.title(f'{material} 中协调数分布')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/coordination_number_distribution_{material}.png')
    plt.close()

print('DFT数据处理和统计分析完成，结果已保存到CSV文件和图像文件。') 