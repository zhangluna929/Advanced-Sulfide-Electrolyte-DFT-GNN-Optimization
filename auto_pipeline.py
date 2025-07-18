# -*- coding: utf-8 -*-
"""
batch_analysis.py

批量分析脚本：
1. 读取多个DFT计算结果文件
2. 进行统计分析和筛选
3. 输出高性能材料的推荐列表
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置参数
data_dir = 'data'  # DFT计算结果目录
output_path = 'results/batch_analysis_results.csv'
N_TOP = 20  # 输出top-N推荐

# 1. 批量读取DFT计算结果
def load_dft_results(data_dir):
    """
    批量读取DFT计算结果文件（假设为CSV格式，每个文件包含多个结构的结果）
    Args:
        data_dir: 数据目录路径
    Returns:
        list: 包含多个DataFrame的列表，每个DataFrame代表一个文件的结果
    """
    results = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(file_path)
                results.append(df)
            except Exception as e:
                print(f"读取文件 {fname} 时出错: {str(e)}")
    return results

# 2. 分析和筛选高性能材料
def analyze_and_filter(results, n_top=N_TOP):
    """
    分析DFT计算结果并筛选高性能材料
    Args:
        results: 包含多个DataFrame的列表
        n_top: 筛选出的顶级材料数量
    Returns:
        pd.DataFrame: 筛选出的顶级材料结果
    """
    # 合并所有结果
    all_data = pd.concat(results, ignore_index=True)
    print(f"共加载 {len(all_data)} 条DFT计算结果")
    
    # 按电导率排序，筛选top-N
    top_data = all_data.sort_values('Conductivity', ascending=False).head(n_top)
    return top_data

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载批量DFT计算结果
    print('加载批量DFT计算结果...')
    results = load_dft_results(data_dir)
    print(f'共加载 {len(results)} 个文件')
    
    # 分析和筛选
    top_results = analyze_and_filter(results, n_top=N_TOP)
    
    # 保存结果
    top_results.to_csv(output_path, index=False)
    print(f'批量分析完成，top-{N_TOP} 结果已保存到: {output_path}')

if __name__ == '__main__':
    main() 