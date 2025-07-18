# -*- coding: utf-8 -*-
"""
掺杂效果分析模块

本模块分析不同掺杂元素和浓度组合对硫化物电解质材料电导率和稳定性的影响，使用DFT计算数据。
主要功能：
1. 加载DFT计算结果
2. 分析掺杂元素（Mg、Ca、Sr等）与性能的关系
3. 可视化分析结果
4. 生成分析报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple

# 创建输出目录
def create_output_dirs():
    dirs = ['data', 'results', 'plots']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

create_output_dirs()

class DopingAnalyzer:
    def __init__(self, data_path: str):
        """
        初始化掺杂分析器
        Args:
            data_path: DFT计算数据路径
        """
        self.data_path = data_path
        self.data = self.load_data()
        
        # 定义掺杂元素的基本属性
        self.dopant_properties = {
            'Mg': {'atomic_number': 12, 'ionic_radius': 0.72, 'electronegativity': 1.31},
            'Ca': {'atomic_number': 20, 'ionic_radius': 1.00, 'electronegativity': 1.00},
            'Sr': {'atomic_number': 38, 'ionic_radius': 1.18, 'electronegativity': 0.95}
            # 可以添加更多掺杂元素
        }

    def load_data(self) -> pd.DataFrame:
        """
        加载DFT计算数据
        Returns:
            pd.DataFrame: 包含DFT计算结果的数据框
        """
        print(f"从 {self.data_path} 加载DFT计算数据...")
        try:
            data = pd.read_csv(self.data_path)
            print(f"成功加载数据，包含 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"数据加载出错: {str(e)}")
            raise

    def analyze_doping_effects(self, dopants: List[str], concentrations: List[float]) -> pd.DataFrame:
        """
        分析不同掺杂配置的效果
        Args:
            dopants: 掺杂元素列表
            concentrations: 浓度列表
        Returns:
            pd.DataFrame: 包含分析结果的数据框
        """
        results = []
        
        for dopant in dopants:
            for conc in concentrations:
                # 筛选符合当前掺杂元素和浓度条件的数据
                subset = self.data[(self.data['Dopant'] == dopant) & 
                                 (self.data['Concentration'] == conc)]
                
                if not subset.empty:
                    # 计算平均电导率和稳定性
                    conductivity_mean = subset['Conductivity'].mean()
                    conductivity_std = subset['Conductivity'].std()
                    stability_mean = subset['Stability'].mean()
                    stability_std = subset['Stability'].std()
                    
                    # 记录结果
                    results.append({
                        'dopant': dopant,
                        'concentration': conc,
                        'conductivity_mean': conductivity_mean,
                        'conductivity_std': conductivity_std if not np.isnan(conductivity_std) else 0,
                        'stability_mean': stability_mean,
                        'stability_std': stability_std if not np.isnan(stability_std) else 0,
                        'overall_score': (conductivity_mean / subset['Conductivity'].max()) * 0.6 + 
                                       (stability_mean / subset['Stability'].max()) * 0.4
                    })
        
        return pd.DataFrame(results)

    def plot_doping_effects(self, results: pd.DataFrame, save_path: str = None):
        """
        可视化掺杂效果
        Args:
            results: 分析结果数据框
            save_path: 图像保存路径（可选）
        """
        # 设置绘图风格
        plt.style.use('seaborn')
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制电导率热图
        pivot_cond = results.pivot(
            index='dopant',
            columns='concentration',
            values='conductivity_mean'
        )
        sns.heatmap(pivot_cond, ax=ax1, cmap='viridis', annot=True,
                    fmt='.2e', cbar_kws={'label': '电导率 (S/cm)'})
        ax1.set_title('电导率与掺杂配置的关系')
        ax1.set_xlabel('掺杂浓度')
        ax1.set_ylabel('掺杂元素')
        
        # 绘制稳定性热图
        pivot_stab = results.pivot(
            index='dopant',
            columns='concentration',
            values='stability_mean'
        )
        sns.heatmap(pivot_stab, ax=ax2, cmap='RdYlGn', annot=True,
                    fmt='.2f', cbar_kws={'label': '稳定性得分'})
        ax2.set_title('稳定性与掺杂配置的关系')
        ax2.set_xlabel('掺杂浓度')
        ax2.set_ylabel('掺杂元素')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def generate_report(self, results: pd.DataFrame, save_path: str = None):
        """
        生成掺杂分析报告
        Args:
            results: 分析结果数据框
            save_path: 报告保存路径（可选）
        Returns:
            str: 报告文本
        """
        report = []
        
        # 最佳电导率配置
        best_cond_idx = results['conductivity_mean'].idxmax()
        best_cond = results.iloc[best_cond_idx]
        
        # 最佳稳定性配置
        best_stab_idx = results['stability_mean'].idxmax()
        best_stab = results.iloc[best_stab_idx]
        
        # 综合性能最佳配置（归一化后的得分）
        best_overall_idx = results['overall_score'].idxmax()
        best_overall = results.iloc[best_overall_idx]
        
        report.append("掺杂效果分析报告")
        report.append("====================")
        report.append("1. 最佳电导率配置：")
        report.append(f"   - 掺杂元素: {best_cond['dopant']}")
        report.append(f"   - 掺杂浓度: {best_cond['concentration']:.2%}")
        report.append(f"   - 电导率: {best_cond['conductivity_mean']:.2e} S/cm")
        report.append(f"   - 稳定性得分: {best_cond['stability_mean']:.2f}")
        
        report.append("\n2. 最佳稳定性配置：")
        report.append(f"   - 掺杂元素: {best_stab['dopant']}")
        report.append(f"   - 掺杂浓度: {best_stab['concentration']:.2%}")
        report.append(f"   - 电导率: {best_stab['conductivity_mean']:.2e} S/cm")
        report.append(f"   - 稳定性得分: {best_stab['stability_mean']:.2f}")
        
        report.append("\n3. 综合性能最佳配置：")
        report.append(f"   - 掺杂元素: {best_overall['dopant']}")
        report.append(f"   - 掺杂浓度: {best_overall['concentration']:.2%}")
        report.append(f"   - 电导率: {best_overall['conductivity_mean']:.2e} S/cm")
        report.append(f"   - 稳定性得分: {best_overall['stability_mean']:.2f}")
        report.append(f"   - 综合得分: {best_overall['overall_score']:.2f}")
        
        report.append("\n4. 掺杂元素趋势分析：")
        for dopant in results['dopant'].unique():
            dopant_data = results[results['dopant'] == dopant]
            report.append(f"\n   {dopant}:")
            report.append(f"   - 电导率范围: {dopant_data['conductivity_mean'].min():.2e} - {dopant_data['conductivity_mean'].max():.2e} S/cm")
            report.append(f"   - 最佳掺杂浓度: {dopant_data.iloc[dopant_data['conductivity_mean'].idxmax()]['concentration']:.2%}")
            report.append(f"   - 稳定性趋势: {'随浓度增加而提高' if dopant_data['stability_mean'].corr(dopant_data['concentration']) > 0 else '随浓度增加而降低'}")
        
        report_text = '\n'.join(report)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text

def main():
    """
    主函数：演示如何使用DopingAnalyzer类
    """
    # 加载数据和基础结构
    analyzer = DopingAnalyzer(
        data_path='data/graph_data.csv'
    )
    
    # 分析掺杂效果
    dopants = ['Mg', 'Ca', 'Sr']
    concentrations = [0.01, 0.02, 0.03, 0.04, 0.05]
    results = analyzer.analyze_doping_effects(dopants, concentrations)
    
    # 可视化结果
    analyzer.plot_doping_effects(results, save_path='plots/doping_effects_heatmap.png')
    
    # 生成报告
    report = analyzer.generate_report(results, save_path='results/doping_analysis_report.txt')
    print(report)

if __name__ == "__main__":
    main() 