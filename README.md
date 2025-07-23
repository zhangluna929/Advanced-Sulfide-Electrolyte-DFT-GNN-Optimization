# Advanced Sulfide Electrolyte Design via Integrated DFT-GNN Workflow

## 1. 项目标题 (Project Title)

硫化物电解质 **DFT-GNN 智能筛选、设计与决策平台**  
DFT-GNN Intelligent Screening, Design & Decision Platform for Sulfide Electrolytes

## 2. 项目简介 (Project Description)

本平台面向下一代固态锂电池核心瓶颈——硫化物固态电解质的离子输运与热力学稳定性——构建了一个跨尺度、多保真、多策略的自主材料发现体系：

1. 以 *ab initio* 量子计算为“高保真机理源”，在原子尺度解析掺杂-缺陷-晶格三耦合对能垒与电荷分布的微观起因；
2. 以图神经网络为“经验萃取器”，通过物理先验嵌入与不确定性量化，将稀疏高成本数据映射为连续性能景观；
3. 以主动学习-贝叶斯实验设计为“智能决策器”，实时评估信息增益与性能前景，指挥 HPC 集群分配 DFT 配额；
4. 通过高速数据链路闭合“计算-预测”回路，实现循环增益式的 **机理-学习协同**。

该系统在 ~10^4 级掺杂空间内实现了三个数量级的搜索加速，理论与数据双重收敛下输出可立即用于实验检验的候选集合，为面向功能陶瓷的“智能设计”范式提供了可复制蓝本。

**English Version**  
Targeting the critical bottlenecks of next-generation solid-state Li batteries—ionic transport and thermo-structural stability of sulfide electrolytes—this platform establishes an autonomous materials-discovery framework that integrates multi-fidelity physics and data-driven intelligence:

1. *Ab initio* quantum calculations serve as a high-fidelity mechanistic source, resolving the microscopic origins of dopant–defect–lattice coupling on diffusion barriers and charge distribution.
2. Graph neural networks act as empirical extractors: with physics-informed priors and uncertainty quantification, sparse, high-cost DFT data are projected onto a continuous performance landscape.
3. An active-learning Bayesian experimental-design module functions as an intelligent decision maker, evaluating information gain and performance prospects in real time to steer HPC resources for the next DFT batch.
4. A high-speed data pipeline closes the compute–predict loop, enabling a self-reinforcing synergy between mechanism and learning.

Across a compositional space of ~10^4 doped configurations the system delivers a three-order-of-magnitude acceleration, achieving theory-and-data convergence while outputting experimentally actionable candidates—offering a replicable blueprint for intelligent design of functional ceramics.

---

## 3. 方法论 (Methodology)

| 步骤 | 说明 / Description |
|---------|------------------|
| ① 结构生成 | 基于对称性算法构造 10 种母相 + 15 元素 × 15 浓度 × 4 位点 ≈ 9 000 结构；Symmetry-aware enumeration yields ~9 000 derived structures from 10 parent phases, 15 dopants, 15 concentrations and 4 crystallographic sites. |
| ② DFT 计算 | VASP-PBE-D3，EDIFF 1e-6 eV，EDIFFG −0.02 eV/Å，K-mesh 4×4×4；CI-NEB 解析扩散路径；VASP-PBE-D3 for relaxation/band/NEB with EDIFF 1e-6 eV, EDIFFG −0.02 eV/Å, 4×4×4 k-mesh. |
| ③ 特征构造 | 提取节点（Z, electronegativity, radius）与边（距离、键序）33 维特征，标准化后写入 `data/graph_data.json`; 33-dim node/edge features standardised and serialised. |
| ④ GNN 训练 | Advanced-GCN & Advanced-MPNN；5-fold CV；Optuna 120 trials，hidden_dim 256, depth 5; Advanced-GCN & Advanced-MPNN with 5-fold CV, Optuna selects hidden_dim 256, depth 5. |
| ⑤ 不确定性 | MC-Dropout (N = 30) + Ensemble (N = 5) 评估置信区间；MC-Dropout (30) and 5-model ensemble quantify uncertainty. |
| ⑥ 解释性 | GNN-Explainer (mask ≈ 8 %) 揭示关键 S–P–Li 三角配位；GNN-Explainer (mask ≈ 8 %) reveals critical S–P–Li coordination. |
| ⑦ 高通量筛选 | `auto_pipeline.py` 按 σ > 1 e-3 S cm⁻¹ ∧ ΔE < 0.3 eV ∧ Stab > 0.9 选 Top-N；`auto_pipeline.py` selects Top-N by σ > 1 e-3 S cm⁻¹, ΔE < 0.3 eV, stability > 0.9. |

---

## 4. 功能与亮点 (Features and Highlights)

* **全面掺杂空间扫描**：10 种母体材料 × 15 元素 × 15 浓度 × 4 晶格位点 ≈ *9,000* 结构。
* **自动化 DFT 工作流**：VASP-PBE + DFT-D3，力/能量双收敛判据，85 % 收敛率，结果实时落库。
* **双 GNN 模型**：Advanced-GCN 与 Advanced-MPNN，Optuna 超参优化；σ-MAE 0.09 S cm⁻¹，R²≈0.99。
* **可解释性与不确定性**：GNN-Explainer 解析关键配位，MC-Dropout 提供置信区间覆盖率 92 %。
* **一键高通量筛选**：`auto_pipeline.py` 自动输出 Top-N 候选结构列表，可直连实验数据库。
* **DFT-GNN 协同工作流**：高保真计算与预测闭环。
* **智能决策模块**：基于不确定性主动学习，动态分配 DFT 资源。

## 5. 智能决策流程 (Intelligent Decision Workflow)
1. `train_gnn_model.py` 训练模型并输出带不确定性的预测 (`results/doping_predictions.csv`)。
2. `auto_pipeline.py --decision --top 50` 调用 `intelligent_decision` 计算综合得分，输出 `results/intelligent_ranking.csv`。
3. 根据排名自动生成下一批 VASP 输入，提交至 HPC。
4. 新的 DFT 结果回流数据集，再训练 GNN——完成闭环优化。

## 6. 使用方法 (Usage)

| 智能决策 | `python auto_pipeline.py --decision --top 50` | 选出下一批候选结构 |

## 7. 技术栈 (Technologies Used)

| 范畴 | 具体工具 | 说明 |
|------|----------|------|
| 电子结构计算 | VASP 6.x (PBE, DFT-D3) | 结构优化、能带及 NEB 路径 |
| 数据处理 | Python 3.10, NumPy, Pandas | DFT 结果解析与特征工程 |
| 机器学习 | PyTorch 2.x, PyTorch-Geometric 2.x | GCN / MPNN 构建与训练 |
| 可视化 | Matplotlib, Seaborn | 统计图与热力图 |
| 自动化 | tqdm, argparse | 批处理进度与 CLI 接口 |

## 6. 安装与配置 (Installation and Setup)

```bash
# 1. 克隆仓库
$ git clone <repo-url>
$ cd 硫化物DFTGNN

# 2. 创建虚拟环境
$ python -m venv venv
$ source venv/bin/activate  # Windows 下为 venv\Scripts\activate

# 3. 安装依赖
$ pip install -r requirements.txt  # 如未提供，请参考 /docs/requirements_example.txt

# 3.1 安装 PyTorch / PyTorch-Geometric (按需选择 CPU 或 CUDA 版本，以下示例为 CPU)
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# 4. (可选) 配置 VASP
export VASP_CMD="/path/to/vasp_std"
```

> **注意**：VASP 需合法授权；若仅复现 GNN 部分，可跳过第 4 步。

## 7. 数据与实验 (Data and Experiments)

| 指标 / Metric | 结果 / Result |
|----------------|--------------|
| 结构优化收敛率 | 85 % (7 650 / 9 000) |
| 稳定性筛选 | 2 100 structures with stability ≥ 0.9 |
| 电导率峰值 | 2.1 × 10⁻¹ S cm⁻¹ (Mg-3 %@Li₆PS₅Cl) |
| 平均 σ 提升 | 5.4× over pristine baseline |
| 最低扩散能垒 ΔE | 0.22 eV (Ca-4 %@Li₆PS₅Cl) |
| μₗᵢ > 5e-4 比例 | 400 / 63 000 路径 (0.63 %) |
| GCN σ-MAE | 0.091 S cm⁻¹ |
| MPNN Stab-MAE | 0.087 |

> 图表与进一步统计：详见 `plots/` 与 `stats/` 目录；`results/advanced_model_evaluation_results.csv` 提供完整交叉验证曲线。

---

## 8. 许可证 (License)

本项目采用 **MIT License**，详见 `LICENSE` 文件。

## 9. 联系方式与作者信息 (Contact and Author Information)

作者：**Luna Zhang**  

## 10. 未来的工作 (Future Work)

* **实验验证**：与合作实验室联合合成并测试 Top-5 预测结构。
* **HSE-DFT 精修**：提升电子结构精度，校正带隙偏差。
* **自监督预训练**：在 Materials Project 400 k+ 图数据库上进行编码器预训练。
* **界面工程拓展**：将框架迁移到固-固界面与外延薄膜应力调控场景。

## 11. 项目结构 (Project Structure)

```
DFT-GNN Intelligent Screening, Design & Decision Platform for Sulfide Electrolytes/
├── auto_pipeline.py               # 高通量筛选入口
├── band_structure_calculation.py  # 能带计算
├── conductivity_calculation.py    # 电导率估算
├── conductivity_stability_analysis.py
├── data/                          # DFT & GNN 数据
│   ├── graph_data.json
│   └── ...
├── gnn_model_architecture.py      # GCN / MPNN 定义
├── train_gnn_model.py             # 训练与评估脚本
├── ion_mobility_analysis.py       # CI-NEB 离子迁移率
├── structure_optimization_vasp.py # VASP 结构优化
├── explain_gnn.py                 # GNN-Explainer
└── results/|plots/|stats/         # 输出数据与可视化
```
