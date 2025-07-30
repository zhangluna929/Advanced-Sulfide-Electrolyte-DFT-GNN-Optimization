# 硫化物DFTGNN: 硫化物晶体电化学性能预测的高通量图神经网络模型
# Sulfide-DFTGNN: A High-Throughput Graph Neural Network Framework for Large-Scale Electromechanical Property Prediction of Sulfide Crystals

## 1. 项目介绍 / Project Overview

### 1.1 高通量自动化工作流 / High-Throughput Workflow
深度集成 Snakemake 与 VASP，覆盖“计算—解析—训练—推理—决策”全链路，可同时处理百到万级硫化物体系。
A Snakemake-driven pipeline tightly couples with VASP to cover the entire chain from calculation to decision, scaling from hundreds to tens-of-thousands of sulfide compounds.

### 1.2 多任务图神经网络 / Multi-Task GNN
端到端预测离子电导率、热力学稳定性及电子带隙，通过共享编码器 + 任务特定头实现信息迁移与协同学习。
An end-to-end GNN predicts ionic conductivity, thermodynamic stability and band gap via a shared encoder plus task-specific heads, enabling knowledge transfer and multi-task synergy.

### 1.3 大规模分布式训练 / Large-Scale Distributed Training
Optuna + DDP 用于超参搜索，FSDP (ZeRO-3) 支持百万参数 Transformer-style GNN，具备 CPU/NVMe offloading 提升显存利用率。
Optuna with DDP explores hyperparameters, while ZeRO-3 FSDP trains million-parameter Transformer-style GNNs with CPU/NVMe offloading for memory efficiency.

### 1.4 不确定性量化与智能排序 / Uncertainty Quantification & Intelligent Ranking
Monte Carlo Dropout 输出置信区间，决策模块以风险调整后的性能指标排序候选材料。
Monte Carlo Dropout provides calibrated uncertainty; the decision module ranks materials by performance-risk trade-off.

### 1.5 跨领域迁移与自监督预训练 / Domain Adaptation & Self-Supervised Pre-Training
GraphCL 自监督学习结合冻结微调，将 DFT 域知识迁移到分子动力学任务，提高样本效率。
GraphCL self-supervised pre-training plus frozen-layer fine-tuning transfers knowledge from DFT to MD tasks, boosting data efficiency.

## 2. 软件栈与硬件加速 / Software Stack & HPC Support
代码基于 Python ≥3.9、PyTorch ≥2.0、PyTorch-Geometric ≥2.4，并使用 Optuna、Snakemake、tqdm 等常用生态。分布式部分支持 NCCL 与 Gloo 后端，FSDP 结合 ZeRO-3 可在多 GPU 节点上训练百万-级参数 Transformer-style GNN；CPU/NVMe offloading 允许超出显存限制的模型并行。
The implementation relies on Python ≥3.9, PyTorch ≥2.0 and PyTorch-Geometric ≥2.4, together with Optuna, Snakemake and tqdm. Distributed execution supports both NCCL and Gloo back-ends. FSDP with ZeRO-3 shards parameters across multiple GPUs and optionally offloads tensors to CPU/NVMe, enabling training of million-parameter Transformer-like GNNs beyond single-GPU memory.

## 3. 数据流水线 / Data Pipeline
1. 解析 VASP 输出 (vasprun.xml, OUTCAR) 至 JSON 图描述 (脚本 `parse_dft.py`)。
2. `graph_data_construction.py` 转换为 PyTorch-Geometric `Data` 对象，包含原子特征、键特征及多任务标签。
3. `GraphDataset` 负责切分数据并持久化为 `processed/data_*.pt`，供训练加载。
1. VASP outputs are parsed to graph JSON via `parse_dft.py`.
2. `graph_data_construction.py` converts JSON into PyG `Data` objects with atom, bond features and multi-task labels.
3. `GraphDataset` splits and serializes datasets as `processed/data_*.pt` for fast loading.

## 4. GNN 架构 / Graph Neural Network Architecture
- **GCN / GAT / TransformerConv**: 基础卷积、注意力与 Transformer 变体；
- **Edge-Aware MPNN & EdgeNetwork**: 联合节点及边更新，显式编码键长、电负性等；
- **DiffPoolModel**: 层次化可微池化以建模超大晶胞；
- 所有模型共享全局平均池化、任务特定线性头，支持多标签回归 (`conductivity`, `stability`, `bandgap`)。
- GCN, GAT and TransformerConv as baseline convolutions; edge-aware MPNN/EdgeNetwork jointly update node and edge states; DiffPool introduces hierarchical pooling for large super-cells. All variants share global mean pooling followed by task-specific linear heads to regress conductivity, stability and bandgap.

## 5. 训练与超参数优化 / Training & Hyperparameter Optimization
`hyperopt_ddp.py` 采用 Optuna + PyTorch DDP 并行搜索隐藏维度、学习率、批大小与 dropout；`fsdp_training.py` 提供 ZeRO-3 FSDP 封装与检查点保存；`train_cli.py` 和 `auto_train.py` 支持单机或多机启动，一键生成最优模型到 `models/best_model.pth`。
`hyperopt_ddp.py` leverages Optuna and PyTorch DDP to explore hidden dimension, learning rate, batch size and dropout. `fsdp_training.py` wraps ZeRO-3 FSDP with checkpointing, while `train_cli.py` and `auto_train.py` offer single- or multi-node entry points that export the best model to `models/best_model.pth`.

## 6. 不确定性与智能决策 / Uncertainty & Intelligent Ranking
`infer_mc.py` 对训练好的模型执行 Monte Carlo Dropout，输出均值与标准差；`uncertainty_utils.py` 计算置信区间，`auto_pipeline.py` 根据期望性能-风险比排序候选材料并生成 `intelligent_ranking.csv`。
`infer_mc.py` performs Monte Carlo Dropout inference, emitting mean and standard deviation. `uncertainty_utils.py` derives confidence intervals, and `auto_pipeline.py` ranks candidate materials by expected performance-risk trade-off, exporting `intelligent_ranking.csv`.

## 7. 领域迁移与自监督 / Domain Adaptation & Self-Supervised Learning
`domain_adaptation.py` 演示 GraphCL 预训练 + 层冻结微调，将 DFT 域知识迁移至分子动力学数据；支持自定义源/目标数据目录与 GPU 选择。
`domain_adaptation.py` shows GraphCL pre-training followed by frozen-layer fine-tuning, transferring knowledge from DFT to molecular dynamics data. Custom source/target datasets and GPU selection are supported.

## 8. Snakemake 工作流 / Snakemake Workflow
`Snakefile` 将 **解析 → 数据集构建 → 超参搜索/训练 → 推理 → 决策** 串成单一规则图；通过 `snakemake -j 32 --config input_dir=vasp_runs epochs=150` 在 32 线程上完成全流程。
The `Snakefile` composes parsing, dataset construction, hyperparameter search/training, inference and decision into a unified DAG. Executing `snakemake -j 32 --config input_dir=vasp_runs epochs=150` completes the entire pipeline on 32 cores.

## 9. 目录结构 / Repository Structure
```
硫化物DFTGNN/
├── src/                  # 核心实现 (gnn, dft, utils, analysis)
├── data/                 # 输入数据与处理后张量
├── models/               # 训练权重与检查点
├── results/              # 推理输出与可视化
├── plots/                # 绘图脚本与图片
├── Snakefile             # Snakemake 工作流
└── requirements.txt      # 依赖列表
```

## 10. 快速开始 / Quick Start
```bash
# 安装依赖
conda create -n dftgnn python=3.9 -y
conda activate dftgnn
pip install -r 硫化物DFTGNN/requirements.txt
# 构建数据并训练
cd 硫化物DFTGNN
snakemake -j 16 --config input_dir=vasp_runs epochs=100
# Monte Carlo 推理
python infer_mc.py --model_path models/best_model.pth --data_root data/gnn_data --output results/pred.csv
```
