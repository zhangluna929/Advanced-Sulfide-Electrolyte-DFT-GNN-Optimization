# Advanced Sulfide Electrolyte DFT-GNN Optimization

## 项目简介 / Project Introduction
本项目基于图神经网络（GNN）和第一性原理计算（DFT），实现了硫化物电解质材料的高通量性能预测、掺杂优化和可解释性分析。项目针对多种锂硫化物电解质材料进行了详细的计算和预测，旨在提高其离子导电率和稳定性。

This project leverages Graph Neural Networks (GNN) and Density Functional Theory (DFT) calculations to achieve high-throughput performance prediction, doping optimization, and interpretability analysis of sulfide electrolyte materials. It focuses on various lithium sulfide electrolytes with detailed computations and predictions aimed at enhancing ionic conductivity and stability.

## 研究对象与方法 / Research Subjects and Methods
### 目标材料 / Target Materials
本研究聚焦于以下锂硫化物电解质材料：
- **核心材料 / Core Materials**：`Li7PS3Cl`, `Li7PS3Br`, `Li6PS5Cl`
- **扩展材料 / Extended Materials**：`Li6PS5Br`, `Li10GeP2S12`, `Li3PS4`, `Li7P3S11`, `Li4PS4I`, `Li7PS6`, `Li9PS6`, 以及部分硒替代材料如 `Li7PSe3Cl`, `Li7PSe3Br`, `Li6PSe5Cl`

This study focuses on the following lithium sulfide electrolyte materials:
- **Core Materials**: `Li7PS3Cl`, `Li7PS3Br`, `Li6PS5Cl`
- **Extended Materials**: `Li6PS5Br`, `Li10GeP2S12`, `Li3PS4`, `Li7P3S11`, `Li4PS4I`, `Li7PS6`, `Li9PS6`, and some selenium-substituted materials such as `Li7PSe3Cl`, `Li7PSe3Br`, `Li6PSe5Cl`

### 掺杂策略 / Doping Strategies
对上述材料进行了多种掺杂元素和浓度的模拟，掺杂元素包括：
- **碱土金属 / Alkaline Earth Metals**：`Mg`, `Ca`, `Sr`, `Ba`
- **过渡金属和类金属 / Transition and Post-Transition Metals**：`Al`, `Zn`, `Ga`, `In`, `Y`, `La`, `Sn`, `Pb`, `Bi`, `Cd`, `Hg`, `Sc`
- **掺杂浓度 / Doping Concentrations**：从1%到15%，步长为1%
- **掺杂位置 / Doping Positions**：包括`Li-site`, `P-site`, `S-site`, `Interstitial`等多种位置

Simulations with various dopants and concentrations were conducted on the above materials, including:
- **Alkaline Earth Metals**: `Mg`, `Ca`, `Sr`, `Ba`
- **Transition and Post-Transition Metals**: `Al`, `Zn`, `Ga`, `In`, `Y`, `La`, `Sn`, `Pb`, `Bi`, `Cd`, `Hg`, `Sc`
- **Doping Concentrations**: From 1% to 15%, with a step of 1%
- **Doping Positions**: Including `Li-site`, `P-site`, `S-site`, `Interstitial`, and other positions

### 计算与模拟 / Calculations and Simulations
利用DFT（密度泛函理论）进行了以下计算：
- **结构优化 / Structural Optimization**：使用VASP模拟了掺杂对晶格参数、体积、键长和能量收敛性的影响，筛选出稳定性得分≥0.9的结构。
- **能带结构分析 / Band Structure Analysis**：计算了掺杂对带隙、价带和导带的影响，筛选出对电导率有正面影响的结构。
- **电导率计算 / Conductivity Calculation**：基于Nernst-Einstein关系估算离子导电率，筛选出高电导率结构。
- **离子迁移率分析 / Ion Mobility Analysis**：模拟了温度（300K-600K）、掺杂元素、浓度和位置对离子迁移率的影响，分析了扩散能垒和跳跃频率。
- **缺陷分析 / Defect Analysis**：研究了氧空位和锂空位形成能及其对电导率的提升效果，评估了缺陷密度和分布均匀性。

The following calculations were performed using DFT (Density Functional Theory):
- **Structural Optimization**: Used VASP to simulate the effects of doping on lattice parameters, volume, bond length, and energy convergence, screening structures with stability scores ≥0.9.
- **Band Structure Analysis**: Calculated the impact of doping on bandgap, valence band, and conduction band, identifying structures with positive conductivity effects.
- **Conductivity Calculation**: Estimated ionic conductivity based on the Nernst-Einstein relation, selecting high-conductivity structures.
- **Ion Mobility Analysis**: Simulated the effects of temperature (300K-600K), dopants, concentrations, and positions on ion mobility, analyzing diffusion barriers and jump frequencies.
- **Defect Analysis**: Investigated the formation energy of oxygen and lithium vacancies and their enhancement of conductivity, evaluating defect density and distribution uniformity.

### 预测模型 / Predictive Models
构建并训练了两种高级图神经网络（GNN）模型用于性能预测：
- **Advanced GCN Model**（图卷积网络 / Graph Convolutional Network）：包含多层GCN卷积、残差连接、批归一化和注意力机制的全局池化，优化了电导率和稳定性的预测。
- **Advanced MPNN Model**（消息传递神经网络 / Message Passing Neural Network）：实现了自定义消息传递和更新函数，支持边特征，结合残差连接和注意力池化，提升了预测精度。
- **训练策略 / Training Strategies**：采用K折交叉验证、Optuna超参数优化、早停机制和学习率调度，数据集划分为训练集（70%）、验证集（15%）和测试集（15%）。
- **不确定性量化 / Uncertainty Quantification**：集成了MC Dropout和Ensemble方法，输出预测的置信区间。

Two advanced Graph Neural Network (GNN) models were developed and trained for performance prediction:
- **Advanced GCN Model** (Graph Convolutional Network): Incorporates multiple GCN layers, residual connections, batch normalization, and attention-based global pooling, optimizing predictions of conductivity and stability.
- **Advanced MPNN Model** (Message Passing Neural Network): Features custom message passing and update functions, supports edge features, and combines residual connections with attention pooling for improved prediction accuracy.
- **Training Strategies**: Employed K-fold cross-validation, Optuna hyperparameter optimization, early stopping, and learning rate scheduling, with the dataset split into training (70%), validation (15%), and test (15%) sets.
- **Uncertainty Quantification**: Integrated MC Dropout and Ensemble methods to provide confidence intervals for predictions.

### 模型性能 / Model Performance
两种模型在测试集上均展示出良好的预测能力，具体指标如下：
- **Advanced GCN**：测试集损失较低，MAE和R2分数显示出良好的预测能力，特别是在电导率预测方面。
- **Advanced MPNN**：测试集损失略优于GCN，MAE和R2分数表明其在复杂结构预测中表现更佳，尤其是在稳定性预测上。
- 详细评估结果和损失曲线已保存至 `results/advanced_model_evaluation_results.csv` 和 `plots/` 目录。

Both models performed excellently on the test set, with specific metrics as follows (based on simulated data):
- **Advanced GCN**: Low test set loss, with MAE and R2 scores indicating strong predictive capability, especially for conductivity.
- **Advanced MPNN**: Slightly better test set loss than GCN, with MAE and R2 scores showing superior performance in predicting complex structures, particularly for stability.
- Detailed evaluation results and loss curves are saved in `results/advanced_model_evaluation_results.csv` and the `plots/` directory.

### 主要结果 / Key Results
- **高性能结构筛选 / High-Performance Structure Screening**：通过DFT和GNN预测，筛选出多种高电导率且稳定性优异的掺杂配置，例如特定浓度下`Mg`和`Ca`掺杂的`Li6PS5Cl`结构。
- **趋势分析 / Trend Analysis**：发现碱土金属（如`Mg`, `Ca`, `Sr`, `Ba`）通过引入空位显著提高离子导电率，而三价元素（如`Al`, `Ga`, `In`）通过电荷补偿机制优化锂空位形成能。
- **最佳掺杂组合 / Optimal Doping Combinations**：综合离子迁移率（>5e-4 cm²/Vs）、扩散能垒（<0.3 eV）和电导率影响因子（>1.2），筛选出最佳掺杂组合，保存至 `stats/ion_mobility_optimal_combinations.csv`。
- **可视化与解释性 / Visualization and Interpretability**：通过GNNExplainer分析模型决策依据，生成结构解释图，揭示关键原子和连接对性能的影响。

- **High-Performance Structure Screening**: Through DFT and GNN predictions, multiple doping configurations with high conductivity and excellent stability were identified, such as `Mg` and `Ca` doped `Li6PS5Cl` at specific concentrations.
- **Trend Analysis**: Found that alkaline earth metals (e.g., `Mg`, `Ca`, `Sr`, `Ba`) significantly enhance ionic conductivity by introducing vacancies, while trivalent elements (e.g., `Al`, `Ga`, `In`) optimize lithium vacancy formation energy via charge compensation mechanisms.
- **Optimal Doping Combinations**: Based on ion mobility (>5e-4 cm²/Vs), diffusion barrier (<0.3 eV), and conductivity factor (>1.2), optimal doping combinations were screened and saved to `stats/ion_mobility_optimal_combinations.csv`.
- **Visualization and Interpretability**: Used GNNExplainer to analyze the basis of model decisions, generating structural explanation graphs to reveal the impact of key atoms and connections on performance.

## 主要功能 / Main Features
1. **不确定性量化 / Uncertainty Quantification**：支持集成方法和MC Dropout，输出预测置信区间。
2. **训练与评估改进 / Training and Evaluation Enhancements**：集成K折交叉验证、Optuna超参数优化、迁移学习接口。
3. **自动化与高通量筛选 / Automation and High-Throughput Screening**：一键批量预测和筛选top-N候选材料。
4. **可视化与解释性 / Visualization and Interpretability**：集成GNNExplainer，分析模型决策依据。

1. **Uncertainty Quantification**: Supports ensemble methods and MC Dropout, outputting prediction confidence intervals.
2. **Training and Evaluation Enhancements**: Integrates K-fold cross-validation, Optuna hyperparameter optimization, and transfer learning interfaces.
3. **Automation and High-Throughput Screening**: One-click batch prediction and screening of top-N candidate materials.
4. **Visualization and Interpretability**: Integrates GNNExplainer to analyze the basis of model decisions.

## 目录结构 / Directory Structure
- `gnn_model_architecture.py`：GNN模型定义 / GNN Model Definition
- `train_gnn_model.py`：模型训练、K折、超参优化 / Model Training, K-fold, Hyperparameter Optimization
- `doping_prediction.py`：掺杂性能预测与不确定性量化 / Doping Performance Prediction and Uncertainty Quantification
- `uncertainty_utils.py`：不确定性量化工具 / Uncertainty Quantification Tools
- `auto_pipeline.py`：自动化高通量筛选 / Automated High-Throughput Screening
- `explain_gnn.py`：模型可解释性分析 / Model Interpretability Analysis
- `data_preparation.py`：基础结构与特征处理 / Basic Structure and Feature Processing
- `data/`：数据文件夹 / Data Folder
- `models/`：模型文件夹 / Models Folder
- `results/`：结果输出 / Results Output
- `plots/`：可视化图表 / Visualization Charts
- `stats/`：统计分析结果 / Statistical Analysis Results

## 快速开始 / Quick Start
### 1. 数据准备 / Data Preparation
- 运行`data_preparation.py`生成标准化结构和特征
- Run `data_preparation.py` to generate standardized structures and features

### 2. 模型训练与评估 / Model Training and Evaluation
- 运行`train_gnn_model.py`，支持K折交叉验证和Optuna超参数优化
- Run `train_gnn_model.py`, supporting K-fold cross-validation and Optuna hyperparameter optimization

### 3. 掺杂性能预测与不确定性量化 / Doping Performance Prediction and Uncertainty Quantification
- 运行`doping_prediction.py`，可选参数`uncertainty_method='ensemble'`或`'mc_dropout'`
- Run `doping_prediction.py` with optional parameters `uncertainty_method='ensemble'` or `'mc_dropout'`

### 4. 高通量自动筛选 / High-Throughput Automated Screening
- 将待筛选结构（json格式）放入`data/high_throughput_structures/`
- 运行`auto_pipeline.py`