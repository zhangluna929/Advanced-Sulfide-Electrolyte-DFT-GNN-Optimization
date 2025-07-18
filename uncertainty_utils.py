# -*- coding: utf-8 -*-
"""
statistical_utils.py

实现DFT数据统计分析工具，包括数据分布分析和置信区间计算。
"""
import numpy as np
import pandas as pd
from scipy import stats

def calculate_confidence_interval(data, confidence=0.95):
    """
    计算数据的置信区间
    Args:
        data: 输入数据数组
        confidence: 置信水平，默认为0.95
    Returns:
        mean: 数据均值
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
    """
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    if n < 2:
        return mean, mean, mean
    stderr = stats.sem(data)
    interval = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - interval, mean + interval

def analyze_data_distribution(data, property_name='Property'):
    """
    分析数据分布特性
    Args:
        data: 输入数据数组
        property_name: 属性名称，用于报告
    Returns:
        dict: 包含分布特性的字典
    """
    data = np.array(data)
    result = {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'min': np.min(data),
        'max': np.max(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
    print(f"{property_name} 数据分布分析:")
    print(f"均值: {result['mean']:.4f}")
    print(f"标准差: {result['std']:.4f}")
    print(f"中位数: {result['median']:.4f}")
    print(f"最小值: {result['min']:.4f}")
    print(f"最大值: {result['max']:.4f}")
    print(f"偏度: {result['skewness']:.4f}")
    print(f"峰度: {result['kurtosis']:.4f}")
    return result

def bootstrap_analysis(data, n_bootstrap=1000, func=np.mean):
    """
    使用Bootstrap方法进行统计分析
    Args:
        data: 输入数据数组
        n_bootstrap: Bootstrap采样次数
        func: 统计函数，默认为均值
    Returns:
        dict: 包含Bootstrap分析结果的字典
    """
    data = np.array(data)
    bootstrap_results = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_results.append(func(sample))
    bootstrap_results = np.array(bootstrap_results)
    return {
        'estimate': func(data),
        'bootstrap_mean': np.mean(bootstrap_results),
        'bootstrap_std': np.std(bootstrap_results),
        'ci_lower': np.percentile(bootstrap_results, 2.5),
        'ci_upper': np.percentile(bootstrap_results, 97.5)
    } 