# -*- coding: utf-8 -*-
"""
材料数据分析脚本

本脚本用于分析硫化物电解质材料的DFT计算结果，处理结构数据、能带结构和电导率数据。
包含数据加载、统计分析和可视化功能。
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 创建输出目录
def create_output_dirs():
    dirs = ['data', 'results', 'plots', 'stats']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

create_output_dirs()

# 加载DFT计算数据
def load_dft_data(file_path='data/graph_data.txt'):
    """
    加载DFT计算结果数据
    Args:
        file_path: 数据文件路径
    Returns:
        DataFrame: 包含材料、掺杂元素、浓度、位置和性能数据的表格
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, sep='\s+', header=None,
                           names=['Material', 'Dopant', 'Concentration', 'Position', 
                                  'Bandgap', 'Conductivity', 'Stability'])
        print(f'从 {file_path} 加载了 {len(data)} 条数据')
        return data
    else:
        print(f'数据文件 {file_path} 不存在，使用模拟数据')
        # 模拟数据
        materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl']
        dopants = ['Mg', 'Ca', 'Ba', 'Al', 'Sr']
        concentrations = [0.01, 0.02, 0.03, 0.04, 0.05]
        positions = ['site1', 'site2', 'site3']
        data = []
        for mat in materials:
            for dop in dopants:
                for conc in concentrations:
                    for pos in positions:
                        data.append({
                            'Material': mat,
                            'Dopant': dop,
                            'Concentration': conc,
                            'Position': pos,
                            'Bandgap': np.random.uniform(0.5, 4.0),
                            'Conductivity': np.random.uniform(0.01, 20.0),
                            'Stability': np.random.uniform(0.7, 1.0)
                        })
        data = pd.DataFrame(data)
        data.to_csv(file_path, sep=' ', index=False)
        print(f'生成了 {len(data)} 条模拟数据并保存到 {file_path}')
        return data

# 统计分析
def analyze_data(data):
    """
    对数据进行统计分析
    Args:
        data: DataFrame，包含材料性能数据
    Returns:
        dict: 统计结果
    """
    stats = {
        'total_samples': len(data),
        'materials': data['Material'].unique().tolist(),
        'dopants': data['Dopant'].unique().tolist(),
        'conductivity_mean': data['Conductivity'].mean(),
        'conductivity_std': data['Conductivity'].std(),
        'bandgap_mean': data['Bandgap'].mean(),
        'bandgap_std': data['Bandgap'].std(),
        'stability_mean': data['Stability'].mean(),
        'stability_std': data['Stability'].std(),
    }
    print('统计分析结果：')
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f'{key}: {value:.2f}')
        else:
            print(f'{key}: {value}')
    return stats

# 可视化数据
def visualize_data(data):
    """
    可视化材料性能数据
    Args:
        data: DataFrame，包含材料性能数据
    """
    # 按材料分组的电导率分布
    plt.figure(figsize=(10, 6))
    for material in data['Material'].unique():
        mat_data = data[data['Material'] == material]
        plt.hist(mat_data['Conductivity'], bins=20, alpha=0.5, label=material)
    plt.title('不同材料的电导率分布')
    plt.xlabel('电导率 (S/cm)')
    plt.ylabel('频次')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/conductivity_distribution_by_material.png')
    plt.close()

    # 按掺杂元素分组的电导率
    plt.figure(figsize=(10, 6))
    for dopant in data['Dopant'].unique():
        dop_data = data[data['Dopant'] == dopant]
        plt.scatter(dop_data['Concentration'], dop_data['Conductivity'], alpha=0.5, label=dopant)
    plt.title('不同掺杂元素和浓度的电导率')
    plt.xlabel('掺杂浓度 (%)')
    plt.ylabel('电导率 (S/cm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/conductivity_by_dopant_concentration.png')
    plt.close()

    print('数据可视化已完成，图表保存到 plots/ 目录')

# 筛选高性能结构
def screen_high_performance(data, conductivity_threshold=10.0, stability_threshold=0.9):
    """
    筛选高电导率和高稳定性的结构
    Args:
        data: DataFrame，包含材料性能数据
        conductivity_threshold: 电导率阈值
        stability_threshold: 稳定性阈值
    Returns:
        DataFrame: 筛选后的高性能结构数据
    """
    high_performance = data[(data['Conductivity'] > conductivity_threshold) & 
                            (data['Stability'] > stability_threshold)]
    print(f'筛选出 {len(high_performance)} 个高性能结构（电导率 > {conductivity_threshold}, 稳定性 > {stability_threshold}）')
    high_performance.to_csv('results/high_performance_structures.csv', index=False)
    return high_performance

# 主函数
def main():
    # 加载数据
    data = load_dft_data()
    
    # 统计分析
    stats = analyze_data(data)
    with open('stats/material_stats.txt', 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value}\n')
    
    # 可视化
    visualize_data(data)
    
    # 筛选高性能结构
    high_performance = screen_high_performance(data)
    print('材料数据分析完成，结果已保存。')

if __name__ == '__main__':
    main() 