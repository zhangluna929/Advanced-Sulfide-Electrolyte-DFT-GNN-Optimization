# 基于DFT与GNN的锂硫化物电解质掺杂优化与电导率预测
# 电导率与稳定性分析脚本

import os
import numpy as np

# 定义材料和掺杂元素，专注于3种材料
materials = ['Li7PS3Cl', 'Li7PS3Br', 'Li6PS5Cl']
dopants = ['Mg', 'Ca', 'Ba', 'Al', 'Sr', 'Zn', 'Ga', 'In', 'Y', 'La']
concentrations = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # 10个掺杂浓度
positions = ['site1', 'site2', 'site3']  # 模拟不同的掺杂位置

# 模拟掺杂后材料的电导率与稳定性分析
def analyze_conductivity_stability(material, dopant, concentration, position):
    # 模拟使用DFT工具（如VASP或QuantumESPRESSO）进行第一性原理计算，分析掺杂后材料的电导率和稳定性
    # 考虑掺杂元素的类型、浓度和位置对材料电子结构、离子导电性和热力学性质的影响
    # 特别关注缺陷（如氧空位、锂空位）对电导率和热稳定性的多方面影响
    
    # 电导率影响：受掺杂元素和缺陷对载流子浓度和迁移率的影响，模拟为相对变化值
    # 假设碱土金属（如Mg、Ca、Sr、Ba）通过引入更多空位或改变局部结构显著提高离子导电率
    conductivity_impact = np.random.uniform(-0.3, 0.3) * (1 + concentration * 3) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 0.8)
    
    # 结构稳定性得分：受掺杂后结构整体稳定性的影响，模拟为0.7到1.0之间的值
    # 假设高浓度掺杂可能引入结构缺陷或应力，降低稳定性
    stability_score = np.random.uniform(0.7, 1.0) * (1 - concentration * 0.5)
    
    # 缺陷形成能：模拟氧空位和锂空位的形成能，单位为eV
    # 随着掺杂浓度增加，缺陷形成能可能降低，因为掺杂引入的应力可能促进缺陷形成
    # 碱土金属（如Mg、Ca、Sr、Ba）可能通过替代锂位点或改变局部配位环境降低氧空位形成能
    oxygen_vacancy_formation_energy = np.random.uniform(1.0, 3.0) * (1 - concentration * 2) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 1.2)
    # 三价元素（如Al、Ga、In）可能通过电荷补偿机制降低锂空位形成能
    lithium_vacancy_formation_energy = np.random.uniform(0.5, 2.0) * (1 - concentration * 1.5) * (1 if dopant in ['Al', 'Ga', 'In'] else 1.1)
    
    # 缺陷对电导率的提升效果：假设低形成能的缺陷有助于提高离子导电率
    # 氧空位可能通过提供额外的离子扩散路径来提升电导率，效果随浓度增加而增强
    conductivity_enhancement_oxygen = (1 / oxygen_vacancy_formation_energy) * np.random.uniform(0.1, 0.5) * (1 + concentration * 2)
    # 锂空位可能通过增加锂离子空位浓度来提升电导率，效果随浓度增加而增强
    conductivity_enhancement_lithium = (1 / lithium_vacancy_formation_energy) * np.random.uniform(0.2, 0.6) * (1 + concentration * 2)
    total_conductivity_enhancement = conductivity_enhancement_oxygen + conductivity_enhancement_lithium
    
    # 缺陷密度：模拟缺陷的相对密度，随浓度增加而增加
    # 碱土金属可能更倾向于形成氧空位，而三价元素更倾向于形成锂空位
    oxygen_vacancy_density = np.random.uniform(0.01, 0.1) * (1 + concentration * 10) * (1 if dopant in ['Mg', 'Ca', 'Sr', 'Ba'] else 0.8)
    lithium_vacancy_density = np.random.uniform(0.02, 0.15) * (1 + concentration * 8) * (1 if dopant in ['Al', 'Ga', 'In'] else 0.7)
    
    # 缺陷分布均匀性：模拟缺陷在晶体中的分布均匀性，影响电导率提升效果
    # 高浓度可能导致缺陷聚集，降低均匀性，进而影响导电路径的连续性
    defect_distribution_uniformity = np.random.uniform(0.5, 1.0) * (1 - concentration * 0.3)
    
    # 热稳定性分析：通过热膨胀系数和熵变评估
    # 热膨胀系数：模拟材料在受热时的体积膨胀率，单位为10^-6/K
    # 假设缺陷密度高会导致更大的热膨胀，因为缺陷破坏了晶格的完整性
    thermal_expansion_coefficient = np.random.uniform(5.0, 15.0) * (1 + (oxygen_vacancy_density + lithium_vacancy_density) * 2) * (1 + concentration * 1.5)
    
    # 熵变：模拟掺杂和缺陷对材料熵的影响，单位为J/(mol·K)
    # 假设缺陷增加会导致熵增加，因为缺陷引入了更多的无序性
    entropy_change = np.random.uniform(0.5, 5.0) * (1 + (oxygen_vacancy_density + lithium_vacancy_density) * 3) * (1 + concentration * 2)
    
    # 热稳定性得分：综合热膨胀系数和熵变的影响，模拟为0.5到1.0之间的值，值越高表示热稳定性越好
    # 高热膨胀系数和高熵变会降低热稳定性，高浓度掺杂也可能引入更多缺陷，降低稳定性
    thermal_stability_score = np.random.uniform(0.5, 1.0) * (1 - (thermal_expansion_coefficient / 20)) * (1 - (entropy_change / 10)) * (1 - concentration * 0.3)
    
    # 缺陷对热稳定性的影响：假设高缺陷密度会降低热稳定性
    # 氧空位可能通过破坏局部配位环境对热稳定性产生负面影响
    thermal_stability_impact_oxygen = np.random.uniform(-0.2, -0.05) * oxygen_vacancy_density * (1 + concentration * 2)
    # 锂空位可能通过改变锂离子扩散路径对热稳定性产生负面影响
    thermal_stability_impact_lithium = np.random.uniform(-0.15, -0.03) * lithium_vacancy_density * (1 + concentration * 2)
    total_thermal_stability_impact = thermal_stability_impact_oxygen + thermal_stability_impact_lithium
    
    # 晶体对称性破坏指数：模拟掺杂是否破坏晶体对称性，影响结构和热稳定性
    # 高浓度掺杂更可能破坏对称性，缺陷密度也可能加剧对称性破坏
    symmetry_breaking_index = np.random.uniform(0.0, 0.2) * (1 + concentration * 5) * (1 + (oxygen_vacancy_density + lithium_vacancy_density) * 1.5)
    
    # 缺陷对离子扩散能垒的影响：模拟缺陷如何改变离子扩散的能量障碍，单位为eV
    # 假设氧空位和锂空位降低扩散能垒，促进离子导电
    diffusion_barrier_reduction_oxygen = np.random.uniform(0.05, 0.2) * oxygen_vacancy_density * (1 + concentration * 1.5)
    diffusion_barrier_reduction_lithium = np.random.uniform(0.1, 0.3) * lithium_vacancy_density * (1 + concentration * 1.5)
    total_diffusion_barrier_reduction = diffusion_barrier_reduction_oxygen + diffusion_barrier_reduction_lithium
    
    # 掺杂对声子模式的影响：模拟掺杂和缺陷对声子模式的影响，影响热导率和热稳定性
    # 假设缺陷和掺杂会增加声子散射，降低热导率
    phonon_scattering_increase = np.random.uniform(0.1, 0.5) * (1 + (oxygen_vacancy_density + lithium_vacancy_density) * 2) * (1 + concentration * 2)
    
    return {
        'conductivity_impact': conductivity_impact,
        'stability_score': stability_score,
        'oxygen_vacancy_formation_energy': oxygen_vacancy_formation_energy,
        'lithium_vacancy_formation_energy': lithium_vacancy_formation_energy,
        'conductivity_enhancement_oxygen': conductivity_enhancement_oxygen,
        'conductivity_enhancement_lithium': conductivity_enhancement_lithium,
        'total_conductivity_enhancement': total_conductivity_enhancement,
        'oxygen_vacancy_density': oxygen_vacancy_density,
        'lithium_vacancy_density': lithium_vacancy_density,
        'defect_distribution_uniformity': defect_distribution_uniformity,
        'thermal_expansion_coefficient': thermal_expansion_coefficient,
        'entropy_change': entropy_change,
        'thermal_stability_score': thermal_stability_score,
        'thermal_stability_impact_oxygen': thermal_stability_impact_oxygen,
        'thermal_stability_impact_lithium': thermal_stability_impact_lithium,
        'total_thermal_stability_impact': total_thermal_stability_impact,
        'symmetry_breaking_index': symmetry_breaking_index,
        'diffusion_barrier_reduction_oxygen': diffusion_barrier_reduction_oxygen,
        'diffusion_barrier_reduction_lithium': diffusion_barrier_reduction_lithium,
        'total_diffusion_barrier_reduction': total_diffusion_barrier_reduction,
        'phonon_scattering_increase': phonon_scattering_increase
    }

# 数据存储
conductivity_stability_results = []
for mat in materials:
    for dop in dopants:
        for conc in concentrations:
            for pos in positions:
                result = analyze_conductivity_stability(mat, dop, conc, pos)
                conductivity_stability_results.append({
                    'material': mat,
                    'dopant': dop,
                    'concentration': conc,
                    'position': pos,
                    'conductivity_impact': result['conductivity_impact'],
                    'stability_score': result['stability_score'],
                    'oxygen_vacancy_formation_energy': result['oxygen_vacancy_formation_energy'],
                    'lithium_vacancy_formation_energy': result['lithium_vacancy_formation_energy'],
                    'conductivity_enhancement_oxygen': result['conductivity_enhancement_oxygen'],
                    'conductivity_enhancement_lithium': result['conductivity_enhancement_lithium'],
                    'total_conductivity_enhancement': result['total_conductivity_enhancement'],
                    'oxygen_vacancy_density': result['oxygen_vacancy_density'],
                    'lithium_vacancy_density': result['lithium_vacancy_density'],
                    'defect_distribution_uniformity': result['defect_distribution_uniformity'],
                    'thermal_expansion_coefficient': result['thermal_expansion_coefficient'],
                    'entropy_change': result['entropy_change'],
                    'thermal_stability_score': result['thermal_stability_score'],
                    'thermal_stability_impact_oxygen': result['thermal_stability_impact_oxygen'],
                    'thermal_stability_impact_lithium': result['thermal_stability_impact_lithium'],
                    'total_thermal_stability_impact': result['total_thermal_stability_impact'],
                    'symmetry_breaking_index': result['symmetry_breaking_index'],
                    'diffusion_barrier_reduction_oxygen': result['diffusion_barrier_reduction_oxygen'],
                    'diffusion_barrier_reduction_lithium': result['diffusion_barrier_reduction_lithium'],
                    'total_diffusion_barrier_reduction': result['total_diffusion_barrier_reduction'],
                    'phonon_scattering_increase': result['phonon_scattering_increase']
                })

# 打印分析结果数量
print(f"总共进行了 {len(conductivity_stability_results)} 次掺杂后材料的电导率与稳定性分析。")
print(f"分析涵盖了3种材料 × 10种掺杂元素 × 10种浓度 × 3个位置 = 900次组合分析。")

# 打印前几个分析结果作为示例
print("以下是前5个分析结果的详细数据：")
for i, result in enumerate(conductivity_stability_results[:5], 1):
    print(f"结果 {i}:")
    print(f"  材料: {result['material']}")
    print(f"  掺杂元素: {result['dopant']}")
    print(f"  浓度: {result['concentration']*100:.0f}%")
    print(f"  位置: {result['position']}")
    print(f"  电导率影响: {result['conductivity_impact']:.4f} (相对变化值)")
    print(f"  稳定性得分: {result['stability_score']:.4f} (0.7-1.0，值越高越稳定)")
    print(f"  氧空位形成能: {result['oxygen_vacancy_formation_energy']:.4f} eV")
    print(f"  锂空位形成能: {result['lithium_vacancy_formation_energy']:.4f} eV")
    print(f"  氧空位对电导率的提升效果: {result['conductivity_enhancement_oxygen']:.4f} (相对值)")
    print(f"  锂空位对电导率的提升效果: {result['conductivity_enhancement_lithium']:.4f} (相对值)")
    print(f"  总电导率提升效果: {result['total_conductivity_enhancement']:.4f} (相对值)")
    print(f"  氧空位密度: {result['oxygen_vacancy_density']:.4f} (相对值)")
    print(f"  锂空位密度: {result['lithium_vacancy_density']:.4f} (相对值)")
    print(f"  缺陷分布均匀性: {result['defect_distribution_uniformity']:.4f} (0.5-1.0，值越高越均匀)")
    print(f"  热膨胀系数: {result['thermal_expansion_coefficient']:.4f} (10^-6/K)")
    print(f"  熵变: {result['entropy_change']:.4f} (J/(mol·K))")
    print(f"  热稳定性得分: {result['thermal_stability_score']:.4f} (0.5-1.0，值越高越稳定)")
    print(f"  氧空位对热稳定性的影响: {result['thermal_stability_impact_oxygen']:.4f} (相对值)")
    print(f"  锂空位对热稳定性的影响: {result['thermal_stability_impact_lithium']:.4f} (相对值)")
    print(f"  总热稳定性影响: {result['total_thermal_stability_impact']:.4f} (相对值)")
    print(f"  晶体对称性破坏指数: {result['symmetry_breaking_index']:.4f} (0.0-1.0，值越高破坏越大)")
    print(f"  氧空位对离子扩散能垒的降低: {result['diffusion_barrier_reduction_oxygen']:.4f} eV")
    print(f"  锂空位对离子扩散能垒的降低: {result['diffusion_barrier_reduction_lithium']:.4f} eV")
    print(f"  总离子扩散能垒降低: {result['total_diffusion_barrier_reduction']:.4f} eV")
    print(f"  声子散射增加: {result['phonon_scattering_increase']:.4f} (相对值)")
    print("")

# 保存分析结果到文件
output_dir = '硫化物DFTGNN/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'conductivity_stability_results.txt'), 'w') as f:
    for result in conductivity_stability_results:
        f.write(str(result) + '\n')

# 筛选对电导率有正面影响且稳定性高的结构
optimal_results = [result for result in conductivity_stability_results if result['conductivity_impact'] > 0 and result['stability_score'] >= 0.9 and result['total_conductivity_enhancement'] > 0.5 and result['thermal_stability_score'] >= 0.7 and result['defect_distribution_uniformity'] >= 0.7 and result['symmetry_breaking_index'] <= 0.1]
print(f"筛选出 {len(optimal_results)} 个对电导率有正面影响、结构稳定性和热稳定性高、缺陷分布均匀且对称性破坏小的掺杂效果。")

# 保存最佳结果到单独文件
with open(os.path.join(output_dir, 'optimal_conductivity_stability_results.txt'), 'w') as f:
    for result in optimal_results:
        f.write(str(result) + '\n')

# 统计分析：计算不同掺杂元素对电导率影响和热稳定性的平均值
dopant_conductivity_impact = {}
dopant_thermal_stability = {}
dopant_diffusion_barrier_reduction = {}
dopant_phonon_scattering = {}
for dop in dopants:
    impacts = [r['conductivity_impact'] for r in conductivity_stability_results if r['dopant'] == dop]
    thermal_scores = [r['thermal_stability_score'] for r in conductivity_stability_results if r['dopant'] == dop]
    diffusion_reductions = [r['total_diffusion_barrier_reduction'] for r in conductivity_stability_results if r['dopant'] == dop]
    phonon_scatters = [r['phonon_scattering_increase'] for r in conductivity_stability_results if r['dopant'] == dop]
    avg_impact = np.mean(impacts) if impacts else 0
    avg_thermal_score = np.mean(thermal_scores) if thermal_scores else 0
    avg_diffusion_reduction = np.mean(diffusion_reductions) if diffusion_reductions else 0
    avg_phonon_scatter = np.mean(phonon_scatters) if phonon_scatters else 0
    dopant_conductivity_impact[dop] = avg_impact
    dopant_thermal_stability[dop] = avg_thermal_score
    dopant_diffusion_barrier_reduction[dop] = avg_diffusion_reduction
    dopant_phonon_scattering[dop] = avg_phonon_scatter

print("不同掺杂元素对电导率的平均影响：")
for dop, impact in dopant_conductivity_impact.items():
    print(f"  {dop}: {impact:.4f} (相对变化值)")

print("不同掺杂元素对热稳定性的平均得分：")
for dop, score in dopant_thermal_stability.items():
    print(f"  {dop}: {score:.4f} (0.5-1.0，值越高越稳定)")

print("不同掺杂元素对离子扩散能垒降低的平均影响：")
for dop, reduction in dopant_diffusion_barrier_reduction.items():
    print(f"  {dop}: {reduction:.4f} eV")

print("不同掺杂元素对声子散射增加的平均影响：")
for dop, scatter in dopant_phonon_scattering.items():
    print(f"  {dop}: {scatter:.4f} (相对值)")

# 统计分析：计算不同材料对稳定性得分和热稳定性的平均值
material_stability_score = {}
material_thermal_stability = {}
material_diffusion_barrier_reduction = {}
material_phonon_scattering = {}
for mat in materials:
    scores = [r['stability_score'] for r in conductivity_stability_results if r['material'] == mat]
    thermal_scores = [r['thermal_stability_score'] for r in conductivity_stability_results if r['material'] == mat]
    diffusion_reductions = [r['total_diffusion_barrier_reduction'] for r in conductivity_stability_results if r['material'] == mat]
    phonon_scatters = [r['phonon_scattering_increase'] for r in conductivity_stability_results if r['material'] == mat]
    avg_score = np.mean(scores) if scores else 0
    avg_thermal_score = np.mean(thermal_scores) if thermal_scores else 0
    avg_diffusion_reduction = np.mean(diffusion_reductions) if diffusion_reductions else 0
    avg_phonon_scatter = np.mean(phonon_scatters) if phonon_scatters else 0
    material_stability_score[mat] = avg_score
    material_thermal_stability[mat] = avg_thermal_score
    material_diffusion_barrier_reduction[mat] = avg_diffusion_reduction
    material_phonon_scattering[mat] = avg_phonon_scatter

print("不同材料的平均稳定性得分：")
for mat, score in material_stability_score.items():
    print(f"  {mat}: {score:.4f} (0.7-1.0，值越高越稳定)")

print("不同材料的平均热稳定性得分：")
for mat, score in material_thermal_stability.items():
    print(f"  {mat}: {score:.4f} (0.5-1.0，值越高越稳定)")

print("不同材料对离子扩散能垒降低的平均影响：")
for mat, reduction in material_diffusion_barrier_reduction.items():
    print(f"  {mat}: {reduction:.4f} eV")

print("不同材料对声子散射增加的平均影响：")
for mat, scatter in material_phonon_scattering.items():
    print(f"  {mat}: {scatter:.4f} (相对值)")

# 统计分析：计算不同掺杂浓度对缺陷形成能、电导率提升效果和热稳定性的平均值
concentration_defect_energy = {}
concentration_conductivity_enhancement = {}
concentration_thermal_stability = {}
concentration_thermal_expansion = {}
concentration_entropy_change = {}
concentration_defect_density = {}
concentration_defect_uniformity = {}
concentration_symmetry_breaking = {}
concentration_diffusion_barrier_reduction = {}
concentration_phonon_scattering = {}
for conc in concentrations:
    oxygen_energies = [r['oxygen_vacancy_formation_energy'] for r in conductivity_stability_results if r['concentration'] == conc]
    lithium_energies = [r['lithium_vacancy_formation_energy'] for r in conductivity_stability_results if r['concentration'] == conc]
    enhancements = [r['total_conductivity_enhancement'] for r in conductivity_stability_results if r['concentration'] == conc]
    thermal_scores = [r['thermal_stability_score'] for r in conductivity_stability_results if r['concentration'] == conc]
    thermal_expansions = [r['thermal_expansion_coefficient'] for r in conductivity_stability_results if r['concentration'] == conc]
    entropy_changes = [r['entropy_change'] for r in conductivity_stability_results if r['concentration'] == conc]
    oxygen_densities = [r['oxygen_vacancy_density'] for r in conductivity_stability_results if r['concentration'] == conc]
    lithium_densities = [r['lithium_vacancy_density'] for r in conductivity_stability_results if r['concentration'] == conc]
    uniformities = [r['defect_distribution_uniformity'] for r in conductivity_stability_results if r['concentration'] == conc]
    symmetry_breaks = [r['symmetry_breaking_index'] for r in conductivity_stability_results if r['concentration'] == conc]
    diffusion_reductions = [r['total_diffusion_barrier_reduction'] for r in conductivity_stability_results if r['concentration'] == conc]
    phonon_scatters = [r['phonon_scattering_increase'] for r in conductivity_stability_results if r['concentration'] == conc]
    avg_oxygen_energy = np.mean(oxygen_energies) if oxygen_energies else 0
    avg_lithium_energy = np.mean(lithium_energies) if lithium_energies else 0
    avg_enhancement = np.mean(enhancements) if enhancements else 0
    avg_thermal_score = np.mean(thermal_scores) if thermal_scores else 0
    avg_thermal_expansion = np.mean(thermal_expansions) if thermal_expansions else 0
    avg_entropy_change = np.mean(entropy_changes) if entropy_changes else 0
    avg_oxygen_density = np.mean(oxygen_densities) if oxygen_densities else 0
    avg_lithium_density = np.mean(lithium_densities) if lithium_densities else 0
    avg_uniformity = np.mean(uniformities) if uniformities else 0
    avg_symmetry_break = np.mean(symmetry_breaks) if symmetry_breaks else 0
    avg_diffusion_reduction = np.mean(diffusion_reductions) if diffusion_reductions else 0
    avg_phonon_scatter = np.mean(phonon_scatters) if phonon_scatters else 0
    concentration_defect_energy[conc] = {'oxygen': avg_oxygen_energy, 'lithium': avg_lithium_energy}
    concentration_conductivity_enhancement[conc] = avg_enhancement
    concentration_thermal_stability[conc] = avg_thermal_score
    concentration_thermal_expansion[conc] = avg_thermal_expansion
    concentration_entropy_change[conc] = avg_entropy_change
    concentration_defect_density[conc] = {'oxygen': avg_oxygen_density, 'lithium': avg_lithium_density}
    concentration_defect_uniformity[conc] = avg_uniformity
    concentration_symmetry_breaking[conc] = avg_symmetry_break
    concentration_diffusion_barrier_reduction[conc] = avg_diffusion_reduction
    concentration_phonon_scattering[conc] = avg_phonon_scatter

print("不同掺杂浓度对缺陷形成能的平均影响：")
for conc, energies in concentration_defect_energy.items():
    print(f"  浓度 {conc*100:.0f}%: 氧空位形成能 = {energies['oxygen']:.4f} eV, 锂空位形成能 = {energies['lithium']:.4f} eV")

print("不同掺杂浓度对电导率提升效果的平均值：")
for conc, enhancement in concentration_conductivity_enhancement.items():
    print(f"  浓度 {conc*100:.0f}%: 电导率提升效果 = {enhancement:.4f} (相对值)")

print("不同掺杂浓度对热稳定性的平均得分：")
for conc, score in concentration_thermal_stability.items():
    print(f"  浓度 {conc*100:.0f}%: 热稳定性得分 = {score:.4f} (0.5-1.0，值越高越稳定)")

print("不同掺杂浓度对热膨胀系数的平均影响：")
for conc, expansion in concentration_thermal_expansion.items():
    print(f"  浓度 {conc*100:.0f}%: 热膨胀系数 = {expansion:.4f} (10^-6/K)")

print("不同掺杂浓度对熵变的平均影响：")
for conc, entropy in concentration_entropy_change.items():
    print(f"  浓度 {conc*100:.0f}%: 熵变 = {entropy:.4f} (J/(mol·K))")

print("不同掺杂浓度对缺陷密度的平均影响：")
for conc, densities in concentration_defect_density.items():
    print(f"  浓度 {conc*100:.0f}%: 氧空位密度 = {densities['oxygen']:.4f} (相对值), 锂空位密度 = {densities['lithium']:.4f} (相对值)")

print("不同掺杂浓度对缺陷分布均匀性的平均影响：")
for conc, uniformity in concentration_defect_uniformity.items():
    print(f"  浓度 {conc*100:.0f}%: 缺陷分布均匀性 = {uniformity:.4f} (0.5-1.0，值越高越均匀)")

print("不同掺杂浓度对晶体对称性破坏指数的平均影响：")
for conc, symmetry_break in concentration_symmetry_breaking.items():
    print(f"  浓度 {conc*100:.0f}%: 晶体对称性破坏指数 = {symmetry_break:.4f} (0.0-1.0，值越高破坏越大)")

print("不同掺杂浓度对离子扩散能垒降低的平均影响：")
for conc, reduction in concentration_diffusion_barrier_reduction.items():
    print(f"  浓度 {conc*100:.0f}%: 离子扩散能垒降低 = {reduction:.4f} eV")

print("不同掺杂浓度对声子散射增加的平均影响：")
for conc, scatter in concentration_phonon_scattering.items():
    print(f"  浓度 {conc*100:.0f}%: 声子散射增加 = {scatter:.4f} (相对值)")

print("掺杂后材料的电导率与稳定性分析完成，最佳结果已筛选并保存到文件。")
print("分析结果包括电导率影响、结构稳定性、缺陷形成能、缺陷密度、缺陷分布均匀性、电导率提升效果、热膨胀系数、熵变、热稳定性、晶体对称性破坏指数、离子扩散能垒降低和声子散射增加等详细信息。")
print("额外提供了不同掺杂元素、材料和浓度对电导率、结构稳定性、热稳定性、离子扩散能垒和声子散射的全面统计分析。") 