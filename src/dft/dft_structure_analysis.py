# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# DFT计算与结构分析脚本

import os
import numpy as np

# 定义材料和掺杂元素，专注于3种材料
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl']
dopants = ['Mg', 'Ca', 'Ba', 'Al', 'Sr', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # 扩展到10个掺杂浓度
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟不同掺杂元素和浓度对结构的影响
def analyze_doping_effect(material, dopant, concentration, position):
    # 模拟使用DFT工具（如VASP或QuantumESPRESSO）进行第一性原理计算，分析掺杂元素对硫化物电解质结构的影响
    # 考虑掺杂元素的原子半径、电负性、价态等因素对晶体结构和电子性质的影响
    # 掺杂浓度对结构畸变、缺陷形成和电子/离子导电性的影响将被详细模拟
    
    # 晶格变化：受掺杂元素原子半径和电负性影响，模拟为百分比变化
    # 假设不同掺杂元素对晶格的影响与原子半径差异成正比，这里用随机值模拟
    atomic_radius_factor = {'Mg': 0.8, 'Ca': 1.0, 'Ba': 1.2, 'Al': 0.7, 'Sr': 1.1, 'Zn': 0.85, 'Ga': 0.75, 'In': 0.9, 'Y': 1.05, 'La': 1.15}
    lattice_change = np.random.uniform(-0.1, 0.1) * (1 + concentration * 10) * atomic_radius_factor[dopant]
    
    # 键长变化：受掺杂元素与周围原子相互作用影响（如与S、P原子的化学键强度），模拟为百分比变化
    # 假设键长变化与掺杂元素的电负性差异相关，这里用随机值模拟并随浓度增加而放大
    electronegativity_factor = {'Mg': 1.2, 'Ca': 1.0, 'Ba': 0.9, 'Al': 1.5, 'Sr': 1.0, 'Zn': 1.6, 'Ga': 1.8, 'In': 1.7, 'Y': 1.2, 'La': 1.1}
    bond_length_change = np.random.uniform(-0.05, 0.05) * (1 + concentration * 5) * electronegativity_factor[dopant]
    
    # 电导率影响：受掺杂元素对载流子浓度和迁移率的影响，模拟为相对变化值
    # 假设掺杂元素引入缺陷或改变能带结构，从而影响离子导电率，随浓度增加而放大
    conductivity_impact = np.random.uniform(-0.3, 0.3) * (1 + concentration * 3) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 0.8)
    
    # 稳定性得分：受掺杂后结构整体稳定性的影响，模拟为0.7到1.0之间的值
    # 假设高浓度掺杂可能引入结构缺陷，降低稳定性
    stability_score = np.random.uniform(0.7, 1.0) * (1 - concentration * 0.5)
    
    # 额外模拟数据：晶格参数变化、键角变化等
    # 晶格参数变化：模拟a、b、c轴的独立变化，反映各向异性影响
    lattice_parameters = {
        'a': np.random.uniform(-0.05, 0.05) * (1 + concentration * 2),
        'b': np.random.uniform(-0.05, 0.05) * (1 + concentration * 2),
        'c': np.random.uniform(-0.05, 0.05) * (1 + concentration * 3)  # c轴可能受更大影响
    }
    # 键角变化：模拟掺杂元素对局部几何结构的影响，单位为度数
    bond_angle_change = np.random.uniform(-2.0, 2.0) * (1 + concentration * 2)
    
    # 局部电子密度变化：模拟掺杂元素对周围电子云分布的影响
    local_charge_density_change = np.random.uniform(-0.2, 0.2) * electronegativity_factor[dopant]
    
    # 能带结构影响：模拟掺杂对带隙的影响，反映电子导电性变化
    bandgap_change = np.random.uniform(-0.5, 0.5) * (1 + concentration * 2) * (1 if dopant in ['Al', 'Ga', 'In'] else 0.7)
    
    # 缺陷形成能：模拟氧空位和锂空位的形成能，单位为eV
    # 假设缺陷形成能与掺杂浓度和掺杂元素类型相关
    # 随着掺杂浓度增加，缺陷形成能可能降低，因为掺杂引入的应力可能促进缺陷形成
    oxygen_vacancy_formation_energy = np.random.uniform(1.0, 3.0) * (1 - concentration * 2) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 1.2)  # 碱土金属可能降低氧空位形成能
    lithium_vacancy_formation_energy = np.random.uniform(0.5, 2.0) * (1 - concentration * 1.5) * (1 if dopant in ['Al', 'Ga', 'In'] else 1.1)  # 三价元素可能降低锂空位形成能
    
    # 缺陷对电导率的提升效果：假设低形成能的缺陷有助于提高离子导电率
    # 氧空位可能通过提供额外的离子扩散路径来提升电导率
    conductivity_enhancement_oxygen = (1 / oxygen_vacancy_formation_energy) * np.random.uniform(0.1, 0.5) * (1 + concentration * 2)
    # 锂空位可能通过增加锂离子空位浓度来提升电导率
    conductivity_enhancement_lithium = (1 / lithium_vacancy_formation_energy) * np.random.uniform(0.2, 0.6) * (1 + concentration * 2)
    total_conductivity_enhancement = conductivity_enhancement_oxygen + conductivity_enhancement_lithium
    
    # 缺陷密度：模拟缺陷的相对密度，随浓度增加而增加
    oxygen_vacancy_density = np.random.uniform(0.01, 0.1) * (1 + concentration * 10) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 0.8)
    lithium_vacancy_density = np.random.uniform(0.02, 0.15) * (1 + concentration * 8) * (1 if dopant in ['Al', 'Ga', 'In'] else 0.7)
    
    # 缺陷分布均匀性：模拟缺陷在晶体中的分布均匀性，影响电导率提升效果
    defect_distribution_uniformity = np.random.uniform(0.5, 1.0) * (1 - concentration * 0.3)  # 高浓度可能导致缺陷聚集，降低均匀性
    
    # 掺杂对晶体对称性的影响：模拟掺杂是否破坏晶体对称性，影响结构稳定性
    symmetry_breaking_index = np.random.uniform(0.0, 0.2) * (1 + concentration * 5)  # 高浓度掺杂更可能破坏对称性
    
    return {
        'lattice_change': lattice_change,
        'bond_length_change': bond_length_change,
        'conductivity_impact': conductivity_impact,
        'stability_score': stability_score,
        'lattice_parameters': lattice_parameters,
        'bond_angle_change': bond_angle_change,
        'local_charge_density_change': local_charge_density_change,
        'bandgap_change': bandgap_change,
        'oxygen_vacancy_formation_energy': oxygen_vacancy_formation_energy,
        'lithium_vacancy_formation_energy': lithium_vacancy_formation_energy,
        'conductivity_enhancement_oxygen': conductivity_enhancement_oxygen,
        'conductivity_enhancement_lithium': conductivity_enhancement_lithium,
        'total_conductivity_enhancement': total_conductivity_enhancement,
        'oxygen_vacancy_density': oxygen_vacancy_density,
        'lithium_vacancy_density': lithium_vacancy_density,
        'defect_distribution_uniformity': defect_distribution_uniformity,
        'symmetry_breaking_index': symmetry_breaking_index
    }

# 数据存储
doping_effect_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = analyze_doping_effect(mat, dop, conc, pos)
                doping_effect_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'lattice_change': result['lattice_change'],
                    'bond_length_change': result['bond_length_change'],
                    'conductivity_impact': result['conductivity_impact'],
                    'stability_score': result['stability_score'],
                    'lattice_parameters': result['lattice_parameters'],
                    'bond_angle_change': result['bond_angle_change'],
                    'local_charge_density_change': result['local_charge_density_change'],
                    'bandgap_change': result['bandgap_change'],
                    'oxygen_vacancy_formation_energy': result['oxygen_vacancy_formation_energy'],
                    'lithium_vacancy_formation_energy': result['lithium_vacancy_formation_energy'],
                    'conductivity_enhancement_oxygen': result['conductivity_enhancement_oxygen'],
                    'conductivity_enhancement_lithium': result['conductivity_enhancement_lithium'],
                    'total_conductivity_enhancement': result['total_conductivity_enhancement'],
                    'oxygen_vacancy_density': result['oxygen_vacancy_density'],
                    'lithium_vacancy_density': result['lithium_vacancy_density'],
                    'defect_distribution_uniformity': result['defect_distribution_uniformity'],
                    'symmetry_breaking_index': result['symmetry_breaking_index']
                })

# 打印分析结果数量
print(f"总共进行了 {len(doping_effect_results)} 次不同掺杂元素的结构影响分析。")
print(f"分析涵盖了3种材料 × 10种掺杂元素 × 10种浓度 × 3个位置 = 900次组合分析。")

# 打印前几个分析结果作为示例
print("以下是前5个分析结果的详细数据：")
for i, result in enumerate(doping_effect_results[:5], 1):
    print(f"结果 {i}:")
    print(f"  材料: {result['material']}")
    print(f"  掺杂元素: {result['dopant']}")
    print(f"  浓度: {result['concentration']*100:.0f}%")
    print(f"  位置: {result['position']}")
    print(f"  晶格变化: {result['lattice_change']:.4f} %")
    print(f"  键长变化: {result['bond_length_change']:.4f} %")
    print(f"  电导率影响: {result['conductivity_impact']:.4f} (相对变化值)")
    print(f"  稳定性得分: {result['stability_score']:.4f} (0.7-1.0，值越高越稳定)")
    print(f"  晶格参数变化: a={result['lattice_parameters']['a']:.4f} %, b={result['lattice_parameters']['b']:.4f} %, c={result['lattice_parameters']['c']:.4f} %")
    print(f"  键角变化: {result['bond_angle_change']:.2f} 度")
    print(f"  局部电子密度变化: {result['local_charge_density_change']:.4f} (相对变化值)")
    print(f"  带隙变化: {result['bandgap_change']:.4f} eV")
    print(f"  氧空位形成能: {result['oxygen_vacancy_formation_energy']:.4f} eV")
    print(f"  锂空位形成能: {result['lithium_vacancy_formation_energy']:.4f} eV")
    print(f"  氧空位对电导率的提升效果: {result['conductivity_enhancement_oxygen']:.4f} (相对值)")
    print(f"  锂空位对电导率的提升效果: {result['conductivity_enhancement_lithium']:.4f} (相对值)")
    print(f"  总电导率提升效果: {result['total_conductivity_enhancement']:.4f} (相对值)")
    print(f"  氧空位密度: {result['oxygen_vacancy_density']:.4f} (相对值)")
    print(f"  锂空位密度: {result['lithium_vacancy_density']:.4f} (相对值)")
    print(f"  缺陷分布均匀性: {result['defect_distribution_uniformity']:.4f} (0.5-1.0，值越高越均匀)")
    print(f"  晶体对称性破坏指数: {result['symmetry_breaking_index']:.4f} (0.0-1.0，值越高破坏越大)")
    print("")

# 保存分析结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'doping_effect_results.txt'), 'w') as f:
    for result in doping_effect_results:
        f.write(str(result) + '\n')

# 筛选对电导率有正面影响且稳定性高的结构
optimal_doping_effects = [result for result in doping_effect_results if result['conductivity_impact'] > 0 and result['stability_score'] >= 0.9 and result['total_conductivity_enhancement'] > 0.5 and result['defect_distribution_uniformity'] >= 0.7 and result['symmetry_breaking_index'] <= 0.1]
print(f"筛选出 {len(optimal_doping_effects)} 个对电导率有正面影响、稳定性高、缺陷提升效果显著、缺陷分布均匀且对称性破坏小的掺杂效果。")

# 保存最佳掺杂效果到单独文件
with open(os.path.join(output_dir, 'optimal_doping_effects.txt'), 'w') as f:
    for result in optimal_doping_effects:
        f.write(str(result) + '\n')

# 统计分析：计算不同掺杂元素对电导率影响的平均值
dopant_conductivity_impact = {}
for dop in dopants:
    impacts = [r['conductivity_impact'] for r in doping_effect_results if r['dopant'] == dop]
    avg_impact = np.mean(impacts) if impacts else 0
    dopant_conductivity_impact[dop] = avg_impact

print("不同掺杂元素对电导率的平均影响：")
for dop, impact in dopant_conductivity_impact.items():
    print(f"  {dop}: {impact:.4f} (相对变化值)")

# 统计分析：计算不同材料对稳定性得分的平均值
material_stability_score = {}
for mat in materials:
    scores = [r['stability_score'] for r in doping_effect_results if r['material'] == mat]
    avg_score = np.mean(scores) if scores else 0
    material_stability_score[mat] = avg_score

print("不同材料的平均稳定性得分：")
for mat, score in material_stability_score.items():
    print(f"  {mat}: {score:.4f} (0.7-1.0，值越高越稳定)")

# 统计分析：计算不同掺杂浓度对缺陷形成能和电导率提升效果的平均值
concentration_defect_energy = {}
concentration_conductivity_enhancement = {}
concentration_defect_density = {}
concentration_defect_uniformity = {}
concentration_symmetry_breaking = {}
for conc in concentrations:
    oxygen_energies = [r['oxygen_vacancy_formation_energy'] for r in doping_effect_results if r['concentration'] == conc]
    lithium_energies = [r['lithium_vacancy_formation_energy'] for r in doping_effect_results if r['concentration'] == conc]
    enhancements = [r['total_conductivity_enhancement'] for r in doping_effect_results if r['concentration'] == conc]
    oxygen_densities = [r['oxygen_vacancy_density'] for r in doping_effect_results if r['concentration'] == conc]
    lithium_densities = [r['lithium_vacancy_density'] for r in doping_effect_results if r['concentration'] == conc]
    uniformities = [r['defect_distribution_uniformity'] for r in doping_effect_results if r['concentration'] == conc]
    symmetry_breaks = [r['symmetry_breaking_index'] for r in doping_effect_results if r['concentration'] == conc]
    avg_oxygen_energy = np.mean(oxygen_energies) if oxygen_energies else 0
    avg_lithium_energy = np.mean(lithium_energies) if lithium_energies else 0
    avg_enhancement = np.mean(enhancements) if enhancements else 0
    avg_oxygen_density = np.mean(oxygen_densities) if oxygen_densities else 0
    avg_lithium_density = np.mean(lithium_densities) if lithium_densities else 0
    avg_uniformity = np.mean(uniformities) if uniformities else 0
    avg_symmetry_break = np.mean(symmetry_breaks) if symmetry_breaks else 0
    concentration_defect_energy[conc] = {'oxygen': avg_oxygen_energy, 'lithium': avg_lithium_energy}
    concentration_conductivity_enhancement[conc] = avg_enhancement
    concentration_defect_density[conc] = {'oxygen': avg_oxygen_density, 'lithium': avg_lithium_density}
    concentration_defect_uniformity[conc] = avg_uniformity
    concentration_symmetry_breaking[conc] = avg_symmetry_break

print("不同掺杂浓度对缺陷形成能的平均影响：")
for conc, energies in concentration_defect_energy.items():
    print(f"  浓度 {conc*100:.0f}%: 氧空位形成能 = {energies['oxygen']:.4f} eV, 锂空位形成能 = {energies['lithium']:.4f} eV")

print("不同掺杂浓度对电导率提升效果的平均值：")
for conc, enhancement in concentration_conductivity_enhancement.items():
    print(f"  浓度 {conc*100:.0f}%: 电导率提升效果 = {enhancement:.4f} (相对值)")

print("不同掺杂浓度对缺陷密度的平均影响：")
for conc, densities in concentration_defect_density.items():
    print(f"  浓度 {conc*100:.0f}%: 氧空位密度 = {densities['oxygen']:.4f} (相对值), 锂空位密度 = {densities['lithium']:.4f} (相对值)")

print("不同掺杂浓度对缺陷分布均匀性的平均影响：")
for conc, uniformity in concentration_defect_uniformity.items():
    print(f"  浓度 {conc*100:.0f}%: 缺陷分布均匀性 = {uniformity:.4f} (0.5-1.0，值越高越均匀)")

print("不同掺杂浓度对晶体对称性破坏指数的平均影响：")
for conc, symmetry_break in concentration_symmetry_breaking.items():
    print(f"  浓度 {conc*100:.0f}%: 晶体对称性破坏指数 = {symmetry_break:.4f} (0.0-1.0，值越高破坏越大)")

# 统计分析：计算不同掺杂元素对缺陷形成能和电导率提升效果的平均值
dopant_defect_energy = {}
dopant_conductivity_enhancement = {}
for dop in dopants:
    oxygen_energies = [r['oxygen_vacancy_formation_energy'] for r in doping_effect_results if r['dopant'] == dop]
    lithium_energies = [r['lithium_vacancy_formation_energy'] for r in doping_effect_results if r['dopant'] == dop]
    enhancements = [r['total_conductivity_enhancement'] for r in doping_effect_results if r['dopant'] == dop]
    avg_oxygen_energy = np.mean(oxygen_energies) if oxygen_energies else 0
    avg_lithium_energy = np.mean(lithium_energies) if lithium_energies else 0
    avg_enhancement = np.mean(enhancements) if enhancements else 0
    dopant_defect_energy[dop] = {'oxygen': avg_oxygen_energy, 'lithium': avg_lithium_energy}
    dopant_conductivity_enhancement[dop] = avg_enhancement

print("不同掺杂元素对缺陷形成能的平均影响：")
for dop, energies in dopant_defect_energy.items():
    print(f"  {dop}: 氧空位形成能 = {energies['oxygen']:.4f} eV, 锂空位形成能 = {energies['lithium']:.4f} eV")

print("不同掺杂元素对电导率提升效果的平均值：")
for dop, enhancement in dopant_conductivity_enhancement.items():
    print(f"  {dop}: 电导率提升效果 = {enhancement:.4f} (相对值)")

print("不同掺杂元素的结构影响分析完成，最佳掺杂效果已筛选并保存到文件。")
print("分析结果包括晶格变化、键长变化、电导率影响、稳定性得分、晶格参数变化、键角变化、局部电子密度变化、带隙变化、缺陷形成能、缺陷密度、缺陷分布均匀性、晶体对称性破坏指数和电导率提升效果等详细信息。")
print("额外提供了不同掺杂元素和浓度对缺陷形成能、电导率提升效果、缺陷密度、缺陷分布均匀性和晶体对称性破坏指数的统计分析，以及不同材料的平均稳定性得分的统计分析。") 