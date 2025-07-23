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
import argparse
from pathlib import Path

try:
    from src.analysis.intelligent_decision import rank_candidates
except ImportError:
    rank_candidates = None  # module may not yet be in path

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
    parser = argparse.ArgumentParser(description="Batch analyse DFT results & optional intelligent ranking")
    parser.add_argument("--top", type=int, default=N_TOP, help="Top-N by conductivity in first stage")
    parser.add_argument("--decision", action="store_true", help="Enable intelligent decision ranking using GNN uncertainty")
    parser.add_argument("--pred_csv", type=str, default="results/doping_predictions.csv", help="Path to GNN prediction csv")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载批量 DFT 计算结果
    print("加载批量DFT计算结果...")
    results = load_dft_results(data_dir)
    print(f"共加载 {len(results)} 个文件")

    # 分析和筛选 by conductivity
    top_results = analyze_and_filter(results, n_top=args.top)
    top_results.to_csv(output_path, index=False)
    print(f"批量分析完成，top-{args.top} 结果已保存到: {output_path}")

    # 若开启智能决策
    if args.decision and rank_candidates is not None:
        decision_csv = Path(args.pred_csv)
        if decision_csv.exists():
            ranked = rank_candidates(decision_csv, top_n=args.top, save_path="results/intelligent_ranking.csv")
            print(f"智能决策排序已保存至 results/intelligent_ranking.csv")
        else:
            print(f"未找到预测文件 {decision_csv}，跳过智能决策模块")

if __name__ == '__main__':
    main() 