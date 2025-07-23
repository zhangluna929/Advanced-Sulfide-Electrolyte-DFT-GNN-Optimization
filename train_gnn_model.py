# -*- coding: utf-8 -*-
"""
材料数据分析脚本

用于分析硫化物电解质材料的DFT计算结果，评估掺杂对电导率和稳定性的影响。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 创建输出目录
def create_output_dirs():
    dirs = ['data', 'results', 'plots']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

create_output_dirs()

# 加载DFT计算数据
def load_dft_data(data_dir):
    """
    加载DFT计算数据
    Args:
        data_dir: 数据目录路径
    Returns:
        data: 包含材料属性和计算结果的DataFrame
    """
    print("正在加载DFT计算数据...")
    try:
        # 假设数据存储在CSV文件中
        data = pd.read_csv(data_dir / 'graph_data.csv')
        print(f"成功加载数据，包含 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"数据加载出错: {str(e)}")
        raise

# 分析掺杂对电导率的影响
def analyze_conductivity(data):
    """
    分析掺杂元素和浓度对电导率的影响
    Args:
        data: 包含材料属性和计算结果的DataFrame
    Returns:
        results: 分析结果字典
    """
    print("正在分析掺杂对电导率的影响...")
    results = {}
    
    # 按掺杂元素和浓度分组，计算平均电导率
    grouped = data.groupby(['Dopant', 'Concentration'])['Conductivity'].mean().reset_index()
    for dopant in grouped['Dopant'].unique():
        dopant_data = grouped[grouped['Dopant'] == dopant]
        results[dopant] = {
            'concentration': dopant_data['Concentration'].values,
            'conductivity': dopant_data['Conductivity'].values
        }
    
    # 找出最佳掺杂组合
    best_combination = grouped.loc[grouped['Conductivity'].idxmax()]
    results['best_combination'] = {
        'dopant': best_combination['Dopant'],
        'concentration': best_combination['Concentration'],
        'conductivity': best_combination['Conductivity']
    }
    
    print(f"最佳掺杂组合: {results['best_combination']['dopant']} at {results['best_combination']['concentration']*100:.1f}% with conductivity {results['best_combination']['conductivity']:.2e} S/cm")
    return results

# 可视化分析结果
def plot_analysis_results(results, save_path):
    """
    可视化分析结果
    Args:
        results: 分析结果字典
        save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6))
    for dopant, data in results.items():
        if dopant != 'best_combination':
            plt.plot(data['concentration'], data['conductivity'], marker='o', label=dopant)
    plt.xlabel('掺杂浓度')
    plt.ylabel('电导率 (S/cm)')
    plt.title('不同掺杂元素和浓度对电导率的影响')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 使用线性回归模型评估材料属性与性能的关系
def evaluate_material_properties(data):
    """
    使用线性回归模型评估材料属性与性能的关系
    Args:
        data: 包含材料属性和计算结果的DataFrame
    Returns:
        model_results: 模型评估结果
    """
    print("正在评估材料属性与性能的关系...")
    # 选择特征和目标变量
    features = ['Concentration', 'AtomicRadius', 'Electronegativity']
    target = 'Conductivity'
    
    # 确保数据完整性
    data = data.dropna(subset=features + [target])
    X = data[features]
    y = data[target]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results = {
        'coefficients': dict(zip(features, model.coef_)),
        'intercept': model.intercept_,
        'mse': mse,
        'r2': r2
    }
    
    print(f"线性回归模型 - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    print(f"模型系数: {model_results['coefficients']}")
    return model_results

# 主函数
def main():
    try:
        # 设置路径
        data_dir = Path('data')
        results_dir = Path('results')
        plots_dir = Path('plots')
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        data = load_dft_data(data_dir)
        
        # 分析掺杂对电导率的影响
        conductivity_results = analyze_conductivity(data)
        plot_analysis_results(conductivity_results, plots_dir / 'conductivity_analysis.png')
        
        # 评估材料属性与性能的关系
        model_results = evaluate_material_properties(data)
        
        # 保存分析结果
        with open(results_dir / 'conductivity_analysis_results.txt', 'w') as f:
            for dopant, result in conductivity_results.items():
                if dopant != 'best_combination':
                    f.write(f"{dopant} 掺杂结果:\n")
                    f.write(f"浓度: {result['concentration'].tolist()}\n")
                    f.write(f"电导率: {result['conductivity'].tolist()}\n\n")
            f.write(f"最佳掺杂组合: {conductivity_results['best_combination']['dopant']} at {conductivity_results['best_combination']['concentration']*100:.1f}% with conductivity {conductivity_results['best_combination']['conductivity']:.2e} S/cm\n")
        
        with open(results_dir / 'model_evaluation_results.txt', 'w') as f:
            f.write(f"线性回归模型评估结果:\n")
            f.write(f"MSE: {model_results['mse']:.4f}\n")
            f.write(f"R2 Score: {model_results['r2']:.4f}\n")
            f.write(f"模型系数: {model_results['coefficients']}\n")
            f.write(f"截距: {model_results['intercept']:.4f}\n")
        
        print("分析完成，结果已保存到文件。")
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 