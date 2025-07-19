# -*- coding: utf-8 -*-
"""
离子迁移率分析脚本

本脚本使用DFT计算结果模拟材料的离子迁移率，分析掺杂对离子迁移的影响，并评估其对电导率的作用。
包含详细的物理化学参数、统计分析和可视化。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

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

temperature_range = np.arange(300, 601, 50)  # 温度范围：300K到600K

# 初始化数据存储
data = []

# 模拟CI-NEB方法计算Li扩散路径
def calculate_li_diffusion_path_ci_neb(material, dopant, concentration, position, temperature):
    # 模拟使用CI-NEB方法计算Li离子在硫化物固态电解质中的扩散路径
    num_images = 9  # 假设使用9个图像点来描述路径，增加点数以提高精度
    
    # 初始状态和最终状态之间的距离（Angstrom）
    base_path_length = np.random.uniform(2.5, 5.5) * (1 + 0.15 * (concentration / 0.15))
    
    # 模拟扩散能垒 (eV)，基于材料和掺杂特性
    base_barrier = np.random.uniform(0.15, 0.9) * (1 - 0.35 * (concentration / 0.15))
    temp_effect = np.exp(-0.012 * (temperature - 300) / 300)
    dopant_effect = 1.0 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 1.2
    position_effect = 0.9 if position == 'Li-site' else 1.1
    diffusion_barrier = base_barrier * temp_effect * dopant_effect * position_effect
    
    # 模拟路径能量曲线，表示从初始到过渡态再到最终状态的能量变化
    energy_profile = [0.0]
    transition_state_index = num_images // 2
    for i in range(1, num_images - 1):
        if i < transition_state_index:
            energy = diffusion_barrier * (1 - ((i - transition_state_index) / transition_state_index)**2)
        elif i == transition_state_index:
            energy = diffusion_barrier
        else:
            energy = diffusion_barrier * (1 - ((i - transition_state_index) / (num_images - transition_state_index - 1))**2)
        energy += np.random.uniform(-0.02, 0.02) * diffusion_barrier
        energy_profile.append(energy)
    energy_profile.append(0.0)
    
    # 模拟CI-NEB计算中的“爬坡”过程，调整过渡态能量
    climbing_energy_adjustment = np.random.uniform(-0.05, 0.05) * diffusion_barrier
    energy_profile[transition_state_index] += climbing_energy_adjustment
    diffusion_barrier = max(energy_profile)
    
    # 模拟扩散速率 (Hz)，基于Arrhenius关系
    k_b = 1.38e-23  # 玻尔兹曼常数，单位：J/K
    h = 6.626e-34  # 普朗克常数，单位：J·s
    attempt_frequency = 1e13 * (1 + 0.2 * (temperature - 300) / 300)
    diffusion_rate = attempt_frequency * np.exp(-diffusion_barrier * 1.602e-19 / (k_b * temperature))
    
    # 模拟路径的几何特性
    path_length = base_path_length * (1 + 0.1 * (diffusion_barrier / 0.5))
    
    # 模拟CI-NEB计算中的力收敛性
    max_force_per_image = np.random.uniform(0.01, 0.05) * (1 + 0.3 * (concentration / 0.15))
    convergence_criterion = 0.03  # 收敛标准为0.03 eV/Angstrom
    is_path_converged = max_force_per_image <= convergence_criterion
    
    # 模拟路径对称性
    path_symmetry_index = np.random.uniform(0.7, 1.0) * (1 - 0.2 * (concentration / 0.15))
    
    # 模拟局部环境对扩散路径的影响
    local_coordination_change = np.random.uniform(-1.0, 1.0) * (1 + 0.3 * (concentration / 0.15))
    
    # 模拟声子耦合对扩散的影响
    phonon_ion_coupling = np.random.uniform(0.1, 0.8) * (1 - 0.2 * (diffusion_barrier / 0.5))
    
    return {
        'diffusion_barrier_ci_neb': diffusion_barrier,
        'path_length_ci_neb': path_length,
        'energy_profile_ci_neb': energy_profile,
        'diffusion_rate_ci_neb': diffusion_rate,
        'max_force_per_image_ci_neb': max_force_per_image,
        'is_path_converged_ci_neb': is_path_converged,
        'path_symmetry_index_ci_neb': path_symmetry_index,
        'local_coordination_change_ci_neb': local_coordination_change,
        'phonon_ion_coupling_ci_neb': phonon_ion_coupling
    }

# 模拟DFT计算离子迁移率
for material in materials:
    for dopant in dopants:
        for conc in concentrations:
            for pos in positions:
                for temp in temperature_range:
                    # 模拟离子迁移率 (cm^2/Vs)
                    # 考虑温度影响（Arrhenius关系）
                    base_mobility = np.random.uniform(1e-5, 1e-3) * np.exp(-0.02 * (temp - 300) / 300)
                    dopant_effect = np.random.uniform(-0.3, 0.3) * base_mobility
                    conc_effect = np.random.uniform(-0.15, 0.15) * base_mobility * conc
                    pos_effect = np.random.uniform(-0.1, 0.1) * base_mobility
                    ion_mobility = base_mobility + dopant_effect + conc_effect + pos_effect
                    
                    # 确保迁移率非负
                    ion_mobility = max(ion_mobility, 1e-6)
                    
                    # 模拟电导率影响因子 (无单位，相对于未掺杂材料的倍数)
                    conductivity_factor = ion_mobility / base_mobility
                    
                    # 模拟离子扩散能垒 (eV)
                    diffusion_barrier = np.random.uniform(0.05, 0.6) * (1 - 0.4 * conductivity_factor) * (1 + 0.01 * (temp - 300) / 300)
                    
                    # 模拟离子跳跃频率 (Hz)
                    jump_frequency = np.random.uniform(1e9, 1e13) * conductivity_factor * np.exp(-0.015 * (temp - 300) / 300)
                    
                    # 模拟离子迁移路径长度 (Angstrom)
                    path_length = np.random.uniform(1.5, 6.0) * (1 + 0.25 * conductivity_factor)
                    
                    # 模拟离子-声子耦合强度 (无单位)
                    ion_phonon_coupling = np.random.uniform(0.1, 1.0) * (1 - 0.3 * conductivity_factor)
                    
                    # 模拟局部结构畸变指数 (无单位)
                    local_distortion_index = np.random.uniform(0.05, 0.5) * (1 - 0.2 * conductivity_factor) * (1 + conc * 2)
                    
                    # 模拟离子通道体积变化率 (%)
                    channel_volume_change = np.random.uniform(-5.0, 10.0) * conductivity_factor * conc
                    
                    # 模拟缺陷-离子相互作用能量 (eV)
                    defect_ion_interaction = np.random.uniform(-0.5, 0.5) * (1 + conc * 1.5)
                    
                    # 模拟离子迁移的熵变 (J/mol·K)
                    entropy_change = np.random.uniform(-10.0, 10.0) * conductivity_factor
                    
                    # 模拟离子迁移的焓变 (kJ/mol)
                    enthalpy_change = np.random.uniform(-50.0, 50.0) * conductivity_factor
                    
                    # 模拟吉布斯自由能变 (kJ/mol)
                    gibbs_free_energy = enthalpy_change - temp * entropy_change / 1000
                    
                    # 模拟声子散射率 (无单位)
                    phonon_scattering_rate = np.random.uniform(0.1, 2.0) * (1 - 0.3 * conductivity_factor) * (1 + 0.02 * (temp - 300) / 300)
                    
                    # 使用CI-NEB方法计算Li扩散路径
                    ci_neb_results = calculate_li_diffusion_path_ci_neb(material, dopant, conc, pos, temp)
                    
                    data.append({
                        'Material': material,
                        'Dopant': dopant,
                        'Concentration': conc,
                        'Position': pos,
                        'Temperature': temp,
                        'Ion_Mobility': ion_mobility,
                        'Conductivity_Factor': conductivity_factor,
                        'Diffusion_Barrier': diffusion_barrier,
                        'Jump_Frequency': jump_frequency,
                        'Path_Length': path_length,
                        'Ion_Phonon_Coupling': ion_phonon_coupling,
                        'Local_Distortion_Index': local_distortion_index,
                        'Channel_Volume_Change': channel_volume_change,
                        'Defect_Ion_Interaction': defect_ion_interaction,
                        'Entropy_Change': entropy_change,
                        'Enthalpy_Change': enthalpy_change,
                        'Gibbs_Free_Energy': gibbs_free_energy,
                        'Phonon_Scattering_Rate': phonon_scattering_rate,
                        'Diffusion_Barrier_CI_NEB': ci_neb_results['diffusion_barrier_ci_neb'],
                        'Path_Length_CI_NEB': ci_neb_results['path_length_ci_neb'],
                        'Energy_Profile_CI_NEB': ci_neb_results['energy_profile_ci_neb'],
                        'Diffusion_Rate_CI_NEB': ci_neb_results['diffusion_rate_ci_neb'],
                        'Max_Force_Per_Image_CI_NEB': ci_neb_results['max_force_per_image_ci_neb'],
                        'Is_Path_Converged_CI_NEB': ci_neb_results['is_path_converged_ci_neb'],
                        'Path_Symmetry_Index_CI_NEB': ci_neb_results['path_symmetry_index_ci_neb'],
                        'Local_Coordination_Change_CI_NEB': ci_neb_results['local_coordination_change_ci_neb'],
                        'Phonon_Ion_Coupling_CI_NEB': ci_neb_results['phonon_ion_coupling_ci_neb']
                    })

# 创建DataFrame
df = pd.DataFrame(data)

# 保存数据到CSV文件
df.to_csv('data/ion_mobility_analysis_data.csv', index=False)

# 统计分析
# 按材料、掺杂元素、温度分组，计算平均离子迁移率和电导率影响因子
material_dopant_temp_group = df.groupby(['Material', 'Dopant', 'Temperature_K']).agg({
    'Ion_Mobility_cm2_Vs': 'mean',
    'Conductivity_Factor': 'mean',
    'Diffusion_Barrier_eV': 'mean',
    'Jump_Frequency_Hz': 'mean',
    'Path_Length_Angstrom': 'mean',
    'Ion_Phonon_Coupling': 'mean',
    'Local_Distortion_Index': 'mean',
    'Channel_Volume_Change_Percent': 'mean',
    'Defect_Ion_Interaction_eV': 'mean',
    'Entropy_Change_J_molK': 'mean',
    'Enthalpy_Change_kJ_mol': 'mean',
    'Gibbs_Free_Energy_kJ_mol': 'mean',
    'Phonon_Scattering_Rate': 'mean'
}).reset_index()

# 保存分组统计结果
material_dopant_temp_group.to_csv('stats/ion_mobility_material_dopant_temp_stats.csv', index=False)

# 按材料、浓度、温度分组，计算平均离子迁移率和电导率影响因子
material_conc_temp_group = df.groupby(['Material', 'Concentration', 'Temperature_K']).agg({
    'Ion_Mobility_cm2_Vs': 'mean',
    'Conductivity_Factor': 'mean',
    'Diffusion_Barrier_eV': 'mean',
    'Jump_Frequency_Hz': 'mean',
    'Path_Length_Angstrom': 'mean',
    'Ion_Phonon_Coupling': 'mean',
    'Local_Distortion_Index': 'mean',
    'Channel_Volume_Change_Percent': 'mean',
    'Defect_Ion_Interaction_eV': 'mean',
    'Entropy_Change_J_molK': 'mean',
    'Enthalpy_Change_kJ_mol': 'mean',
    'Gibbs_Free_Energy_kJ_mol': 'mean',
    'Phonon_Scattering_Rate': 'mean'
}).reset_index()

# 保存分组统计结果
material_conc_temp_group.to_csv('stats/ion_mobility_material_concentration_temp_stats.csv', index=False)

# 按材料、位置、温度分组，计算平均离子迁移率和电导率影响因子
material_pos_temp_group = df.groupby(['Material', 'Position', 'Temperature_K']).agg({
    'Ion_Mobility_cm2_Vs': 'mean',
    'Conductivity_Factor': 'mean',
    'Diffusion_Barrier_eV': 'mean',
    'Jump_Frequency_Hz': 'mean',
    'Path_Length_Angstrom': 'mean',
    'Ion_Phonon_Coupling': 'mean',
    'Local_Distortion_Index': 'mean',
    'Channel_Volume_Change_Percent': 'mean',
    'Defect_Ion_Interaction_eV': 'mean',
    'Entropy_Change_J_molK': 'mean',
    'Enthalpy_Change_kJ_mol': 'mean',
    'Gibbs_Free_Energy_kJ_mol': 'mean',
    'Phonon_Scattering_Rate': 'mean'
}).reset_index()

# 保存分组统计结果
material_pos_temp_group.to_csv('stats/ion_mobility_material_position_temp_stats.csv', index=False)

# 筛选对电导率有正面影响的掺杂组合 (Conductivity_Factor > 1.2) 且温度为300K
positive_effect_300K = df[(df['Conductivity_Factor'] > 1.2) & (df['Temperature_K'] == 300)]
positive_effect_300K.to_csv('stats/ion_mobility_positive_effect_combinations_300K.csv', index=False)

# 筛选对电导率有正面影响的掺杂组合 (Conductivity_Factor > 1.2) 且温度为500K
positive_effect_500K = df[(df['Conductivity_Factor'] > 1.2) & (df['Temperature_K'] == 500)]
positive_effect_500K.to_csv('stats/ion_mobility_positive_effect_combinations_500K.csv', index=False)

# 可视化 - 离子迁移率与掺杂浓度的关系（按温度和材料）
for temp in [300, 400, 500]:
    plt.figure(figsize=(12, 6))
    temp_data = material_conc_temp_group[material_conc_temp_group['Temperature_K'] == temp]
    for material in materials[:5]:  # 限制材料数量以避免图表过于复杂
        mat_data = temp_data[temp_data['Material'] == material]
        plt.plot(mat_data['Concentration'], mat_data['Ion_Mobility_cm2_Vs'], marker='o', label=material)
    plt.xlabel('掺杂浓度')
    plt.ylabel('离子迁移率 (cm^2/Vs)')
    plt.title(f'温度 {temp}K 下不同材料中离子迁移率与掺杂浓度的关系')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/ion_mobility_vs_concentration_temp_{temp}K.png', bbox_inches='tight')
    plt.close()

# 可视化 - 电导率影响因子与掺杂浓度的关系（按温度和材料）
for temp in [300, 400, 500]:
    plt.figure(figsize=(12, 6))
    temp_data = material_conc_temp_group[material_conc_temp_group['Temperature_K'] == temp]
    for material in materials[:5]:
        mat_data = temp_data[temp_data['Material'] == material]
        plt.plot(mat_data['Concentration'], mat_data['Conductivity_Factor'], marker='o', label=material)
    plt.xlabel('掺杂浓度')
    plt.ylabel('电导率影响因子')
    plt.title(f'温度 {temp}K 下不同材料中电导率影响因子与掺杂浓度的关系')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/conductivity_factor_vs_concentration_temp_{temp}K.png', bbox_inches='tight')
    plt.close()

# 可视化 - 离子迁移率与温度的关系（按材料和掺杂元素）
for material in materials[:3]:
    plt.figure(figsize=(12, 6))
    mat_data = material_dopant_temp_group[material_dopant_temp_group['Material'] == material]
    for dopant in dopants[:5]:  # 限制掺杂元素数量
        dop_data = mat_data[mat_data['Dopant'] == dopant]
        plt.plot(dop_data['Temperature_K'], dop_data['Ion_Mobility_cm2_Vs'], marker='o', label=dopant)
    plt.xlabel('温度 (K)')
    plt.ylabel('离子迁移率 (cm^2/Vs)')
    plt.title(f'{material} 中离子迁移率与温度的关系（按掺杂元素）')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/ion_mobility_vs_temperature_{material}.png', bbox_inches='tight')
    plt.close()

# 可视化 - 扩散能垒与温度的关系（按材料）
for material in materials[:3]:
    plt.figure(figsize=(12, 6))
    mat_data = material_conc_temp_group[material_conc_temp_group['Material'] == material]
    for conc in concentrations[:5:2]:  # 每隔两个浓度取一个
        conc_data = mat_data[mat_data['Concentration'] == conc]
        plt.plot(conc_data['Temperature_K'], conc_data['Diffusion_Barrier_eV'], marker='o', label=f'浓度 {conc}')
    plt.xlabel('温度 (K)')
    plt.ylabel('扩散能垒 (eV)')
    plt.title(f'{material} 中扩散能垒与温度的关系（按浓度）')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/diffusion_barrier_vs_temperature_{material}.png', bbox_inches='tight')
    plt.close()

# 可视化 - 热力图：离子迁移率与掺杂元素和浓度的关系（按材料，温度300K）
for material in materials[:3]:
    mat_data = df[(df['Material'] == material) & (df['Temperature_K'] == 300)]
    pivot_table = mat_data.pivot_table(values='Ion_Mobility_cm2_Vs', index='Dopant', columns='Concentration', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2e', cmap='YlGnBu')
    plt.title(f'{material} 中离子迁移率热力图 (温度 300K)')
    plt.xlabel('掺杂浓度')
    plt.ylabel('掺杂元素')
    plt.tight_layout()
    plt.savefig(f'plots/ion_mobility_heatmap_{material}_300K.png')
    plt.close()

# 可视化 - 箱线图：离子迁移率分布（按材料和位置，温度300K）
for material in materials[:3]:
    mat_data = df[(df['Material'] == material) & (df['Temperature_K'] == 300)]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Position', y='Ion_Mobility_cm2_Vs', data=mat_data)
    plt.title(f'{material} 中离子迁移率分布（按位置，温度 300K）')
    plt.xlabel('掺杂位置')
    plt.ylabel('离子迁移率 (cm^2/Vs)')
    plt.tight_layout()
    plt.savefig(f'plots/ion_mobility_boxplot_{material}_300K.png')
    plt.close()

# 机器学习分析 - 预测离子迁移率
# 将分类变量转换为数值编码
df_encoded = df.copy()
df_encoded['Material_Code'] = pd.Categorical(df['Material']).codes
df_encoded['Dopant_Code'] = pd.Categorical(df['Dopant']).codes
df_encoded['Position_Code'] = pd.Categorical(df['Position']).codes

# 自变量：材料、浓度、掺杂元素编码、位置编码、温度
X = df_encoded[['Material_Code', 'Concentration', 'Dopant_Code', 'Position_Code', 'Temperature_K']]
# 因变量：离子迁移率
y = df_encoded['Ion_Mobility_cm2_Vs']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print('线性回归模型 - MSE:', lr_mse)
print('线性回归模型 - R2 Score:', lr_r2)
print('线性回归系数:', lr_model.coef_)
print('线性回归截距:', lr_model.intercept_)

# 随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print('随机森林回归模型 - MSE:', rf_mse)
print('随机森林回归模型 - R2 Score:', rf_r2)
print('随机森林特征重要性:', rf_model.feature_importances_)

# 保存预测结果
df_test = X_test.copy()
df_test['Actual_Ion_Mobility'] = y_test
df_test['Predicted_Ion_Mobility_LR'] = lr_predictions
df_test['Predicted_Ion_Mobility_RF'] = rf_predictions
df_test.to_csv('stats/ion_mobility_ml_predictions.csv', index=False)

# 相关性分析
correlation_matrix = df[['Ion_Mobility_cm2_Vs', 'Conductivity_Factor', 'Diffusion_Barrier_eV', 
                         'Jump_Frequency_Hz', 'Path_Length_Angstrom', 'Ion_Phonon_Coupling', 
                         'Local_Distortion_Index', 'Channel_Volume_Change_Percent', 
                         'Defect_Ion_Interaction_eV', 'Entropy_Change_J_molK', 
                         'Enthalpy_Change_kJ_mol', 'Gibbs_Free_Energy_kJ_mol', 
                         'Phonon_Scattering_Rate', 'Temperature_K']].corr()

# 保存相关性矩阵
correlation_matrix.to_csv('stats/ion_mobility_correlation_matrix.csv')

# 可视化相关性热力图
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('离子迁移率及相关参数的相关性热力图')
plt.tight_layout()
plt.savefig('plots/ion_mobility_correlation_heatmap.png')
plt.close()

# 筛选最佳掺杂组合：综合考虑离子迁移率、扩散能垒和电导率影响因子
# 条件：Conductivity_Factor > 1.2, Diffusion_Barrier_eV < 0.3, Ion_Mobility_cm2_Vs > 5e-4
optimal_combinations = df[(df['Conductivity_Factor'] > 1.2) & 
                          (df['Diffusion_Barrier_eV'] < 0.3) & 
                          (df['Ion_Mobility_cm2_Vs'] > 5e-4)]
optimal_combinations.to_csv('stats/ion_mobility_optimal_combinations.csv', index=False)

print('离子迁移率详细分析完成，结果已保存到CSV文件和图像文件。')
print(f'找到 {len(optimal_combinations)} 个最佳掺杂组合（电导率因子>1.2，扩散能垒<0.3 eV，离子迁移率>5e-4 cm^2/Vs）。') 