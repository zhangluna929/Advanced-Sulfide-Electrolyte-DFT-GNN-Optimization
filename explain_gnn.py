# -*- coding: utf-8 -*-
"""
data_visualization.py

DFT数据可视化分析脚本，支持对DFT计算结果的解释和可视化。
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置参数
data_path = 'data/graph_data.csv'
save_dir = 'results/visualization/'

# 确保输出目录存在
os.makedirs(save_dir, exist_ok=True)

def visualize_dft_data(data_path, save_dir):
    """
    可视化DFT计算数据
    Args:
        data_path: DFT数据文件路径
        save_dir: 图像保存目录
    """
    # 加载数据
    print(f'从 {data_path} 加载DFT数据...')
    data = pd.read_csv(data_path)
    print(f'成功加载数据，包含 {len(data)} 条记录')
    
    # 设置绘图风格
    plt.style.use('seaborn')
    
    # 1. 电导率分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Conductivity', kde=True)
    plt.title('电导率分布图')
    plt.xlabel('电导率 (S/cm)')
    plt.ylabel('频次')
    plt.savefig(os.path.join(save_dir, 'conductivity_distribution.png'))
    plt.close()
    
    # 2. 稳定性分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Stability', kde=True, color='green')
    plt.title('稳定性分布图')
    plt.xlabel('稳定性得分')
    plt.ylabel('频次')
    plt.savefig(os.path.join(save_dir, 'stability_distribution.png'))
    plt.close()
    
    # 3. 掺杂元素对电导率的影响
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Dopant', y='Conductivity', data=data)
    plt.title('不同掺杂元素对电导率的影响')
    plt.xlabel('掺杂元素')
    plt.ylabel('电导率 (S/cm)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_dir, 'dopant_conductivity_boxplot.png'))
    plt.close()
    
    # 4. 掺杂浓度对电导率的影响
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Concentration', y='Conductivity', hue='Dopant', data=data)
    plt.title('掺杂浓度对电导率的影响')
    plt.xlabel('掺杂浓度')
    plt.ylabel('电导率 (S/cm)')
    plt.savefig(os.path.join(save_dir, 'concentration_conductivity_scatter.png'))
    plt.close()
    
    print(f'可视化分析完成，结果已保存到: {save_dir}')

def main():
    visualize_dft_data(data_path, save_dir)

if __name__ == '__main__':
    main() 